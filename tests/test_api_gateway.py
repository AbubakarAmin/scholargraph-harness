"""Tests for core.api_gateway: TokenBucket, CircuitBreaker, and APIGateway."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from core.api_gateway import (
    APIGateway,
    CircuitBreaker,
    CircuitOpen,
    CircuitState,
    RateLimitError,
    TokenBucket,
    SourceHealthTracker,
)


# ────────────────────────────────────────────
# TokenBucket
# ────────────────────────────────────────────

class TestTokenBucket:
    def test_acquire_returns_immediately_when_tokens_available(self):
        bucket = TokenBucket(rate=10.0, burst=5)
        assert bucket.acquire(timeout=1.0) is True

    def test_acquire_blocks_until_token_refills(self):
        bucket = TokenBucket(rate=10.0, burst=1)
        assert bucket.acquire(timeout=1.0) is True
        # Second acquire should block briefly
        start = time.monotonic()
        assert bucket.acquire(timeout=2.0) is True
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05, f"Should have waited at least 50ms, waited {elapsed:.3f}s"

    def test_acquire_timeout_returns_false(self):
        bucket = TokenBucket(rate=0.001, burst=0)
        assert bucket.acquire(timeout=0.05) is False

    def test_burst_allows_multiple_immediate(self):
        bucket = TokenBucket(rate=1.0, burst=3)
        assert bucket.acquire(timeout=0.1) is True
        assert bucket.acquire(timeout=0.1) is True
        assert bucket.acquire(timeout=0.1) is True
        # 4th should block
        assert bucket.acquire(timeout=0.05) is False

    def test_concurrent_access(self):
        bucket = TokenBucket(rate=100.0, burst=10)
        acquired = []
        lock = threading.Lock()

        def worker():
            if bucket.acquire(timeout=2.0):
                with lock:
                    acquired.append(True)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)
        # All 20 should eventually acquire (burst=10, rate=100/s, 2s timeout)
        assert len(acquired) == 20


# ────────────────────────────────────────────
# CircuitBreaker
# ────────────────────────────────────────────

class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(threshold=3, cooldown=60.0)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_acquire_raises_when_open(self):
        cb = CircuitBreaker(threshold=3, cooldown=60.0)
        for _ in range(3):
            cb.record_failure()
        with pytest.raises(CircuitOpen):
            cb.acquire()

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(threshold=3, cooldown=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_closes_from_half_open(self):
        cb = CircuitBreaker(threshold=3, cooldown=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_reopens_from_half_open(self):
        cb = CircuitBreaker(threshold=3, cooldown=0.1)
        for _ in range(3):
            cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_counter(self):
        cb = CircuitBreaker(threshold=3, cooldown=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Counter reset, so 2 more failures won't trip
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED


# ────────────────────────────────────────────
# SourceHealthTracker
# ────────────────────────────────────────────

class TestSourceHealthTracker:
    def test_empty_returns_1_0(self):
        h = SourceHealthTracker()
        assert h.success_rate("arxiv") == 1.0

    def test_all_success(self):
        h = SourceHealthTracker()
        for _ in range(5):
            h.record("arxiv", True)
        assert h.success_rate("arxiv") == 1.0

    def test_all_failure(self):
        h = SourceHealthTracker()
        for _ in range(5):
            h.record("arxiv", False)
        assert h.success_rate("arxiv") == 0.0

    def test_mixed(self):
        h = SourceHealthTracker(window=10)
        for _ in range(3):
            h.record("arxiv", True)
        for _ in range(7):
            h.record("arxiv", False)
        assert abs(h.success_rate("arxiv") - 0.3) < 0.01


# ────────────────────────────────────────────
# APIGateway
# ────────────────────────────────────────────

class TestAPIGateway:
    def test_request_success(self):
        gw = APIGateway()
        result = gw.request("llm", lambda: "hello")
        assert result == "hello"

    def test_request_passes_args(self):
        gw = APIGateway()
        result = gw.request("llm", lambda a, b: a + b, 1, 2)
        assert result == 3

    def test_request_passes_kwargs(self):
        gw = APIGateway()
        result = gw.request("llm", lambda x=0: x * 2, x=5)
        assert result == 10

    def test_retry_on_transient_error(self):
        gw = APIGateway()
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("timeout")
            return "ok"

        result = gw.request("llm", flaky, retries=3, backoff_base=0.01)
        assert result == "ok"
        assert call_count == 3

    def test_retry_exhausted_raises(self):
        gw = APIGateway()

        def always_fail():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            gw.request("llm", always_fail, retries=2, backoff_base=0.01)

    def test_circuit_breaker_trips_on_real_failures(self):
        gw = APIGateway(circuit_threshold=2, circuit_cooldown=10.0)

        def fail_transient():
            raise ConnectionError("connection reset")

        # First two calls fail and trip the breaker
        for _ in range(2):
            with pytest.raises(ConnectionError):
                gw.request("arxiv", fail_transient, retries=0)

        # Third call should be blocked by circuit breaker
        with pytest.raises(CircuitOpen):
            gw.request("arxiv", fail_transient, retries=0)

    def test_circuit_breaker_not_tripped_by_429s(self):
        """429 is a flow-control signal, not a provider failure.
        The breaker should NOT open on sustained 429s — the AIMD
        penalty already handles rate reduction."""
        gw = APIGateway(circuit_threshold=3, circuit_cooldown=60.0)

        def always_429():
            raise RateLimitError("semantic_scholar", 1.0)

        # 429s should NOT trip the breaker
        for _ in range(5):
            with pytest.raises(RateLimitError):
                gw.request("semantic_scholar", always_429, retries=0)

        # Breaker should still be closed — 429s are not failures
        assert gw.breaker_state("semantic_scholar") == CircuitState.CLOSED

    def test_health_tracking(self):
        gw = APIGateway()
        gw.request("llm", lambda: "ok")
        assert gw.health.success_rate("llm") == 1.0

        with pytest.raises(Exception):
            gw.request("llm", lambda: (_ for _ in ()).throw(ValueError("bad")), retries=0)
        assert gw.health.success_rate("llm") == 0.5

    def test_rate_limit_error_retries_then_raises(self):
        """429 should be retried up to retries times, then raise."""
        gw = APIGateway()
        call_count = 0

        def rate_limited():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("openalex", 0.01)

        with pytest.raises(RateLimitError) as exc_info:
            gw.request("openalex", rate_limited, retries=3, backoff_base=0.01)
        assert "openalex" in str(exc_info.value)
        assert call_count == 4  # 1 initial + 3 retries

    def test_circuit_breaker_carries_provider_name(self):
        cb = CircuitBreaker(provider="arxiv", threshold=2, cooldown=60.0)
        cb.record_failure()
        cb.record_failure()
        with pytest.raises(CircuitOpen) as exc_info:
            cb.acquire()
        assert exc_info.value.provider == "arxiv"
        assert "arxiv" in str(exc_info.value)


# ────────────────────────────────────────────
# Sustained-throttle gate + provider 429 floor
# ────────────────────────────────────────────

class TestSustainedThrottleGate:
    def test_consecutive_429s_pause_via_is_available_not_breaker(self):
        """Run fd6f4910: S2 429'd even at ~0.05 req/s. Sustained throttling
        must pause the provider for is_available() callers WITHOUT opening the
        circuit breaker (429 is flow control — pinned test above forbids it)."""
        gw = APIGateway()
        gw._THROTTLE_OPEN_THRESHOLD = 3  # shrink for the test

        def always_429():
            raise RateLimitError("semantic_scholar", 1.0)

        # Feed 3 throttle signals (retries=0 -> raise after 1 attempt each).
        for _ in range(3):
            with pytest.raises(RateLimitError):
                gw.request("semantic_scholar", always_429, retries=0)

        # Provider is now throttled out for callers that check is_available().
        assert gw.is_available("semantic_scholar") is False
        # But the circuit breaker itself is untouched (429 != failure).
        assert gw.breaker_state("semantic_scholar") == CircuitState.CLOSED

    def test_success_clears_throttle_gate(self):
        gw = APIGateway()
        gw._THROTTLE_OPEN_THRESHOLD = 3
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] <= 3:
                raise RateLimitError("semantic_scholar_bulk", 1.0)
            return "ok"

        for _ in range(3):
            with pytest.raises(RateLimitError):
                gw.request("semantic_scholar_bulk", flaky, retries=0)
        assert gw.is_available("semantic_scholar_bulk") is False
        # A request that skips the gate check still runs; success clears it.
        with __import__("unittest.mock").mock.patch("core.api_gateway.time.sleep"):
            assert gw.request("semantic_scholar_bulk", flaky, retries=0) == "ok"
        assert gw.is_available("semantic_scholar_bulk") is True

    def test_provider_429_floor_applies_to_s2(self):
        """S2 retries must wait at least _THROTTLE_FLOOR seconds even when the
        caller passes a tiny backoff_base (previously ~3s, kept getting 429)."""
        gw = APIGateway(rates={"semantic_scholar": (50.0, 5)})
        sleeps = []

        def always_429():
            raise RateLimitError("semantic_scholar", 1.0)

        from unittest.mock import patch
        import core.api_gateway as ag_mod

        def fake_sleep(s):
            sleeps.append(s)

        with patch.object(ag_mod.time, "sleep", side_effect=fake_sleep):
            with pytest.raises(RateLimitError):
                gw.request("semantic_scholar", always_429, retries=1, backoff_base=0.01)
        # Both the initial (attempt 0) and retry sleeps respect the floor.
        assert len(sleeps) == 1
        assert sleeps[0] >= gw._THROTTLE_FLOOR["semantic_scholar"] * 0.8  # jitter can only shrink 20%

    def test_non_s2_floor_not_applied(self):
        """Other providers keep the caller's backoff (no 10s floor)."""
        gw = APIGateway(rates={"openalex": (50.0, 5)})
        sleeps = []

        def always_429():
            raise RateLimitError("openalex", 0.01)

        from unittest.mock import patch
        import core.api_gateway as ag_mod

        def fake_sleep(s):
            sleeps.append(s)

        with patch.object(ag_mod.time, "sleep", side_effect=fake_sleep):
            with pytest.raises(RateLimitError):
                gw.request("openalex", always_429, retries=1, backoff_base=0.01)
        assert len(sleeps) == 1
        assert sleeps[0] < gw._THROTTLE_FLOOR["semantic_scholar"]
