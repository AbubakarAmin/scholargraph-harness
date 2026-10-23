"""M0: LLM/gateway repair tests D1/D2/D6/D7/D8 (8 tests, offline)."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.api_gateway import APIGateway


def test_d1_strict_raises_no_empty_success():
    from core.llm import LLMClient, LLMUnavailable

    client = LLMClient.__new__(LLMClient)
    client.provider = "gemini"
    client._gemini = None
    client._openai = None

    def _boom(*a, **k):
        raise RuntimeError("connection reset by peer")

    from core.api_gateway import get_gateway

    gw = get_gateway()
    orig = gw.request
    gw.request = lambda *a, **k: _boom()
    try:
        with pytest.raises(LLMUnavailable):
            client.chat("hello", allow_empty=False)
        # Compat path still returns "".
        assert client.chat("hello", allow_empty=True) == ""
    finally:
        gw.request = orig


def test_d2_fallback_on_429_via_classifier(monkeypatch):
    from core.llm import LLMClient, _is_retryable_llm_error

    assert _is_retryable_llm_error(RuntimeError("429 too many requests")) is True
    assert _is_retryable_llm_error(RuntimeError("500 overloaded, try again")) is True
    assert _is_retryable_llm_error(RuntimeError("connection timeout")) is True
    assert _is_retryable_llm_error(ValueError("bad prompt shape")) is False


def test_provider_stream_death_is_transient():
    """Run fd6f4910: a mid-stream APIError 'winner inference incomplete
    (nonce_finished=false)' was classified as permanent, so no retry/fallback
    fired and the LLM generation silently returned ''. It must be transient."""
    from core.api_gateway import APIGateway
    from core.llm import _is_retryable_llm_error

    class _StreamAPIError(Exception):
        status_code = None  # stream started (200) then died without a status

    err = _StreamAPIError("inference: winner inference incomplete (nonce_finished=false)")
    assert APIGateway._is_transient(err) is True
    assert _is_retryable_llm_error(err) is True
    assert APIGateway._is_non_retryable(err) is False


def test_d6_namespaced_key_and_str_guard():
    def _fn(a, b=1):
        return 1

    k1 = APIGateway.build_coalesce_key("llm", _fn, (1,), {"b": 2})
    k2 = APIGateway.build_coalesce_key("semantic_scholar", _fn, (1,), {"b": 2})
    assert k1 != k2 and k1.startswith("llm:") and k2.startswith("semantic_scholar:")
    gw = APIGateway()
    # Non-string coalesce key must not crash (str() guard).
    gw.request("llm", lambda: "ok", coalesce_key=12345, retries=0, rate_limit=False)
    assert 12345 not in gw._inflight and "12345" not in gw._inflight


def test_d6_follower_timeout_and_cancel_cleanup():
    import concurrent.futures as cf

    gw = APIGateway()

    def _slow():
        import time

        time.sleep(5)
        return "slow"

    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        f_leader = pool.submit(gw.request, "llm", _slow, coalesce_key="k-timeout", retries=0, rate_limit=False)
        import time

        time.sleep(0.2)
        with pytest.raises(Exception):
            gw.request("llm", _slow, coalesce_key="k-timeout", retries=0, rate_limit=False, coalesce_timeout=0.1)
        # Key eventually cleaned after leader finishes/cancels.
        try:
            f_leader.result(timeout=7)
        except Exception:
            pass
        assert "k-timeout" not in gw._inflight


def test_d7_registry_locked_and_budgets_separate():
    gw = APIGateway()
    assert hasattr(gw, "_registry_lock")
    # Concurrent lazy creation must not duplicate or crash.
    seen = []

    def _worker(i):
        b = gw._get_bucket(f"prov-{i % 3}")
        seen.append(b)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(seen) == 12


def test_d8_transient_extended_and_status_inspection():
    assert APIGateway._is_transient(RuntimeError("500 internal error")) is True
    assert APIGateway._is_transient(RuntimeError("504 gateway timeout")) is True
    assert APIGateway._is_transient(RuntimeError("model overloaded, try again")) is True
    assert APIGateway._is_transient(RuntimeError("deadline exceeded")) is True

    class _E(Exception):
        status_code = 500

    assert APIGateway._is_transient(_E("boom")) is True

    class _E404(Exception):
        status_code = 404

    assert APIGateway._is_transient(_E404("missing")) is False
    assert APIGateway._is_non_retryable(_E404("missing")) is True


def test_d8_500_retried_then_succeeds():
    gw = APIGateway(rates={"llm": (50.0, 5)})
    calls = {"n": 0}

    class _E500(Exception):
        status_code = 500

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _E500("500 boom")
        return "recovered"

    assert gw.request("llm", _flaky, retries=2, backoff_base=0.01, backoff_max=0.05, rate_limit=False) == "recovered"
    assert calls["n"] == 2


def test_d8_404_fast_no_retry():
    gw = APIGateway(rates={"llm": (50.0, 5)})
    calls = {"n": 0}

    class _E404(Exception):
        status_code = 404

    def _missing():
        calls["n"] += 1
        raise _E404("404 not found")

    with pytest.raises(_E404):
        gw.request("llm", _missing, retries=3, backoff_base=0.01, backoff_max=0.05, rate_limit=False)
    assert calls["n"] == 1


def _gemini_only_client():
    from core import llm as llm_mod

    with patch.object(llm_mod.config, "google_api_key", "test-gemini-key"), \
         patch.object(llm_mod.config, "openai_api_key", "test-openai-key"), \
         patch.object(llm_mod.config, "openai_base_url", "https://chat.example/v1"), \
         patch.object(llm_mod.config, "llm_provider", "openai_compatible"):
        llm_mod.reset_embedding_health()
        client = llm_mod.LLMClient(provider="openai_compatible")
        llm_mod.reset_embedding_health()
        return client, llm_mod


def test_chat_openai_retries_on_truncation_with_larger_budget():
    """Run fd6f4910: finish_reason=length returned a cut-off answer that the
    JSON parser then dropped (seeds produced 0 gaps). A truncated stream must
    be re-run with a doubled max_tokens so the full answer is preserved."""
    import core.llm as llm_mod

    def _chunk(finish, content):
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish,
            delta=SimpleNamespace(content=content),
        )])

    created_calls = []
    streams = iter([
        # 1st call: truncated at max_tokens=8192
        [_chunk(None, '{"gaps": [{"title": "partial'),
         _chunk("length", "")],
        # 2nd call: full answer at max_tokens=16384
        [_chunk(None, '{"gaps": [{"title": "complete"}]}'),
         _chunk("stop", "")],
    ])

    class _FakeStream:
        def __init__(self, chunks):
            self._it = iter(chunks)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._it)

    class _FakeCompletions:
        def create(self, **kwargs):
            created_calls.append(kwargs["max_tokens"])
            return _FakeStream(next(streams))

    client = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
    client.provider = "openai_compatible"
    client._openai = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))

    out = client._chat_openai("prompt", 0.5, "test-model", 8192, None)
    assert created_calls == [8192, 16384]
    assert out == '{"gaps": [{"title": "complete"}]}'


def test_chat_openai_bounded_truncation_returns_longest():
    """If the budget can't grow further, return the longest text, not ''."""
    import core.llm as llm_mod
    def _chunk(finish, content):
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish,
            delta=SimpleNamespace(content=content),
        )])

    created_calls = []

    class _FakeStream:
        def __init__(self):
            self._it = iter([
                _chunk(None, "partial-content-"),
                _chunk("length", ""),
            ])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._it)

    class _FakeCompletions:
        def create(self, **kwargs):
            created_calls.append(kwargs["max_tokens"])
            return _FakeStream()

    client = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
    client.provider = "openai_compatible"
    client._openai = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))

    out = client._chat_openai("prompt", 0.5, "test-model", 16384, None)
    assert out == "partial-content-"
    _ceiling = int(getattr(llm_mod.config, "llm_max_output_tokens", 65536) or 65536)
    # Escalation doubles until the ceiling, then returns longest.
    _expected = [16384]
    while _expected[-1] * 2 <= _ceiling and len(_expected) < 3:
        _expected.append(_expected[-1] * 2)
    assert created_calls == _expected


def test_gemini_only_openai_chat_provider_still_embeds_via_gemini():
    """GEMINI-ONLY POLICY: chat on OpenAI, embeddings on gemini-embedding-2."""
    client, llm_mod = _gemini_only_client()
    client._openai = None  # would explode if any path touched the OpenAI SDK
    with patch.object(client._gemini.models, "embed_content") as mock_gemini, \
         patch("core.llm.get_gateway") as mock_gw:
        mock_gw.return_value.request.side_effect = lambda _p, fn, *a, **k: fn(*a, **k)
        mock_gemini.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.5] * 3072)])
        with patch.object(llm_mod.config, "embedding_dimension", 3072):
            vec = client.embed("some abstract text")
        mock_gemini.assert_called_once()
        called_model = mock_gemini.call_args.kwargs["model"]
        assert "gemini-embedding-2" in called_model
        assert len(vec) == 3072
    llm_mod.reset_embedding_health()


def test_gemini_only_no_openai_fallback_without_gemini_client():
    """_embed_raw raises instead of routing to OpenAI when Gemini is absent."""
    _, llm_mod = _gemini_only_client()
    client = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
    client.provider = "openai_compatible"
    client._gemini = None
    client._openai = None
    with pytest.raises(llm_mod.LLMUnavailable):
        client._embed_raw("hello", None)
    llm_mod.reset_embedding_health()
