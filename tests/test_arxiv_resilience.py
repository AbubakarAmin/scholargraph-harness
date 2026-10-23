"""Regression tests for arXiv 429-storm resilience.

Covers the Oct-02 run where one query burned ~50s in the arxiv lib's internal
retries and then the gateway retried the whole search on top — up to ~3 min
per query before degrading to []. The client no longer retries internally
(single retry owner = gateway), pages match actual demand, throttle warnings
stay one-liners, and structured HTTP status drives detection.
"""

import logging
import time as _time

import arxiv
import pytest

from core.api_gateway import APIGateway
from tests.test_topic_hunter_features_8_15 import _make_agent


class _FakeAuthor:
    def __init__(self, name):
        self.name = name


class _FakePublished:
    year = 2026


class _FakeResult:
    title = "A"
    summary = "Abstract"
    entry_id = "https://arxiv.org/abs/2601.00001"
    categories = ["cs.CV"]
    published = _FakePublished()
    authors = []


class _FakeArxivClient:
    """Injectable stand-in for arxiv.Client.results()."""

    def __init__(self, failures_before_success=0):
        self.calls = 0
        self.failures_before_success = failures_before_success

    def results(self, search):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise arxiv.HTTPError("https://export.arxiv.org/api/query?x", 0, 429)
        return iter([_FakeResult()])


def _fast_gateway(monkeypatch, sleeps):
    """Fresh gateway with a generous bucket (no token waits) and recorded sleeps."""
    gw = APIGateway(rates={"arxiv": (10.0, 10)})
    monkeypatch.setattr("core.api_gateway.get_gateway", lambda **_k: gw)
    monkeypatch.setattr(
        _time, "sleep", lambda s: sleeps.append(s)
    )
    return gw


def _agent_with_fake_client(failures_before_success=0):
    agent = _make_agent()
    agent._arxiv_consecutive_failures = 0
    agent._arxiv_client = _FakeArxivClient(failures_before_success)
    return agent


def test_shared_client_has_single_retry_owner():
    """The shared client must not retry internally (gateway owns retries)."""
    from agents import topic_hunter

    monkeypatch_client = topic_hunter._ARXIV_CLIENT
    topic_hunter._ARXIV_CLIENT = None
    try:
        client = topic_hunter._get_arxiv_client()
        assert client.num_retries == 0
        assert client.delay_seconds == 3.0  # ToS pacing stays
        assert client.page_size <= 50  # callers ask for <=50; no 100-row pages
    finally:
        topic_hunter._ARXIV_CLIENT = monkeypatch_client


def test_persistent_429_is_bounded_and_degrades_to_empty(monkeypatch):
    """A stormed query costs ~3 paced attempts (not nested retry loops), then []."""
    sleeps = []
    _fast_gateway(monkeypatch, sleeps)
    agent = _agent_with_fake_client(failures_before_success=999)

    out = agent.search_arxiv("cat:cs.CV AND test", 30)

    assert out == []
    assert agent._arxiv_client.calls == 3, "1 initial + 2 gateway retries"
    assert len(sleeps) == 2, f"no sleep after the final attempt, got {sleeps}"
    assert sum(sleeps) < 60, f"total backoff must stay bounded, got {sleeps}"


def test_transient_429_recovers_on_retry(monkeypatch):
    """One 429 then success returns papers with exactly 2 attempts."""
    sleeps = []
    _fast_gateway(monkeypatch, sleeps)
    agent = _agent_with_fake_client(failures_before_success=1)

    out = agent.search_arxiv("cat:cs.CV AND test", 30)

    assert len(out) == 1 and out[0]["title"] == "A"
    assert agent._arxiv_client.calls == 2


def test_throttle_signal_prefers_structured_status():
    """An exc carrying status=429 is throttling even if the message hides it."""
    from core.api_gateway import APIGateway

    class StatusOnly(Exception):
        status = 429

    assert APIGateway._parse_throttle_signal(StatusOnly("some proxy grumble")) == 5.0

    class OtherStatus(Exception):
        status = 500

    assert APIGateway._parse_throttle_signal(OtherStatus("boom 500")) is None
    assert APIGateway._parse_throttle_signal(ValueError("plain failure")) is None


def test_throttle_warning_is_one_line_without_traceback(monkeypatch, caplog):
    """Throttled retries must not dump 60-line chained tracebacks into logs."""
    sleeps = []
    gw = _fast_gateway(monkeypatch, sleeps)

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise arxiv.HTTPError("https://export.arxiv.org/api/query?x", 0, 429)
        return "ok"

    with caplog.at_level(logging.WARNING, logger="core.api_gateway"):
        assert gw.request("arxiv", flaky, retries=1, backoff_base=1.0) == "ok"

    throttled = [r for r in caplog.records if "throttled" in r.getMessage()]
    assert len(throttled) == 1
    assert throttled[0].exc_info is None
    assert attempts["n"] == 2
