"""Centralized API gateway: adaptive rate limiting, circuit breaking, and retry for all external APIs.

Design (v2):
  - TokenBucket: classic fixed-rate bucket (kept for direct-test compatibility).
  - AdaptiveTokenBucket: AIMD controller on top — the effective rate is cut
    multiplicatively whenever the provider signals throttling (HTTP 429 / 503
    with Retry-After) and recovers additively after sustained successes. This
    replaces static "1 req / 3s forever" pacing with self-tuning throughput.
  - CircuitBreaker: unchanged CLOSED -> OPEN -> HALF_OPEN machine.
  - APIGateway.request: adds +/-20% jitter to all backoff waits, optional
    single-flight coalescing for identical read-only calls, and
    is_available()/breaker_state() probes so callers can skip providers whose
    breaker is open instead of paying retry sleeps.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(Exception):
    """Raised when the circuit breaker is open and requests are blocked."""
    def __init__(self, provider: str, cooldown_remaining: float):
        self.provider = provider
        self.cooldown_remaining = cooldown_remaining
        super().__init__(f"Circuit breaker open for {provider}, {cooldown_remaining:.0f}s remaining")


class RateLimitError(Exception):
    """Raised when a request is rate-limited by the provider (HTTP 429)."""
    def __init__(self, provider: str, retry_after: float = 5.0):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limited by {provider}, retry after {retry_after:.1f}s")


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Tokens are generated at a fixed rate. Threads block on acquire() until
    a token is available. No sleep-outside-lock bugs.
    """

    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: Tokens generated per second (sustained throughput).
            burst: Maximum tokens in the bucket (allows short bursts).
        """
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 60.0) -> bool:
        """Block until a token is available. Returns True if acquired, False on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                # Calculate wait time for next token
                wait = (1.0 - self._tokens) / self.rate
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(wait, remaining))

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(float(self.burst), self._tokens + elapsed * self.rate)
        self._last_refill = now


class AdaptiveTokenBucket(TokenBucket):
    """Token bucket with AIMD-style rate adaptation.

    Starts at the provider's documented sustainable rate. When the provider
    signals throttling (429/503 with Retry-After), the effective rate is cut
    multiplicatively (floor 0.05 req/s) and held for a cooldown window.
    After 5 consecutive successes the rate recovers additively toward base.
    """

    MIN_RATE = 0.05

    def __init__(self, rate: float, burst: int = 1):
        super().__init__(rate=rate, burst=burst)
        self.base_rate = float(rate)
        self._penalized_until = 0.0
        self._success_streak = 0
        self._recover_step = max(rate * 0.1, 0.05)

    def current_rate(self) -> float:
        with self._lock:
            return self.rate

    def penalize(self, factor: float = 0.5, cooldown: float = 30.0):
        """Multiplicative decrease (the 'MD' in AIMD)."""
        with self._lock:
            factor = max(0.05, min(1.0, float(factor)))
            self.rate = max(self.MIN_RATE, self.rate * factor)
            self._penalized_until = time.monotonic() + max(0.0, cooldown)
            self._success_streak = 0

    def reward(self):
        """Additive increase toward the base rate after sustained success."""
        with self._lock:
            if self.rate >= self.base_rate:
                return
            if time.monotonic() < self._penalized_until:
                return
            self._success_streak += 1
            if self._success_streak >= 5:
                self.rate = min(self.base_rate, self.rate + self._recover_step)
                self._success_streak = 0


class CircuitBreaker:
    """Thread-safe circuit breaker with CLOSED → OPEN → HALF_OPEN state machine.

    After `threshold` consecutive failures, opens for `cooldown` seconds.
    On cooldown expiry, allows one probe request. If it succeeds, closes;
    if it fails, reopens.

    acquire() never sleeps — it raises CircuitOpen immediately so the caller
    can handle the wait externally.  This avoids blocking threads while the
    breaker is open.
    """

    def __init__(self, provider: str = "unknown", threshold: int = 5, cooldown: float = 60.0):
        self.provider = provider
        self.threshold = threshold
        self.cooldown = cooldown
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() >= self._open_until:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def acquire(self):
        """Raise CircuitOpen immediately if breaker is blocking. Never sleeps."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() >= self._open_until:
                    self._state = CircuitState.HALF_OPEN
                else:
                    remaining = self._open_until - time.monotonic()
                    raise CircuitOpen(self.provider, remaining)

    def record_success(self):
        with self._lock:
            self._consecutive_failures = 0
            if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker closed for %s (recovered)", self.provider)

    def record_failure(self):
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.threshold:
                self._state = CircuitState.OPEN
                self._open_until = time.monotonic() + self.cooldown
                logger.warning(
                    "Circuit breaker OPENED for %s — %d consecutive failures, pausing for %.0fs",
                    self.provider, self._consecutive_failures, self.cooldown,
                )

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN


class SourceHealthTracker:
    """Tracks per-provider success/failure rates over a sliding window."""

    def __init__(self, window: int = 10):
        self._window = window
        self._results: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, provider: str, success: bool):
        self._results[provider].append(success)

    def success_rate(self, provider: str) -> float:
        results = self._results[provider]
        if not results:
            return 1.0
        return sum(results) / len(results)


class APIGateway:
    """Centralized rate limiting, circuit breaking, and retry for all external APIs.

    Usage:
        gateway = APIGateway()
        result = gateway.request("arxiv", search_fn, query, limit=100)
    """

    # Default rate limits based on official API documentation:
    #   arXiv:     1 req / 3 seconds (https://info.arxiv.org/help/api/user-manual.html)
    #   OpenAlex:  10 req/s polite pool, 100k/day (https://help.openalex.org/api/authentication)
    #   Semantic Scholar: 1 RPS with API key (free tier), heavily
    #     shared-quota throttled even when authenticated (run 21a145ad)
    #   LLM:       Provider-dependent; conservative for shared endpoints
    #
    # These are STARTING points: AdaptiveTokenBucket cuts the effective rate on
    # 429/503 signals and recovers after sustained success, so a provider that
    # quietly enforces a stricter limit than documented converges to it.
    DEFAULT_RATES = {
        "arxiv":     (0.33, 1),   # 1 req / 3s, no burst
        "openalex":  (5.0,  3),   # 5 req/s sustained, burst 3
        "semantic_scholar":        (1.0,  2),   # 1 req/s, burst 2
        "llm":       (0.5,  1),   # 0.5 req/s, burst 1
        "embedding": (1.0,  1),   # 1 req/s, burst 1
        "semantic_scholar_bulk":   (1.0,  2),   # 1 req/s, burst 2 (Semantic Scholar bulk)
        "crossref":  (1.0,  1),   # polite pool w/ mailto
        "openreview": (0.5, 1),
        "huggingface": (2.0, 2),  # 2 req/s, burst 2 (HF API polite pool)
        # M2 review sources (§5.1, all free, all via gateway + cache + breaker).
        "unpaywall": (1.0,  1),   # 1 req/s, burst 1 (api.unpaywall.org)
        "pmc":       (2.0,  2),   # 2 req/s, burst 2 (OA subset only)
        "doaj":      (2.0,  2),   # 2 req/s, burst 2 (weekly journal cache)
        "oa_text":   (1.0,  1),   # 1 req/s, burst 1 (publisher OA full-text; separate
                                  # from the unpaywall API bucket so a slow domain
                                  # never penalizes api.unpaywall.org traffic)
    }

    # Providers sharing one upstream host share one AIMD bucket + breaker.
    # semantic_scholar (Graph paper API) and semantic_scholar_bulk (bulk
    # search) both hit api.semanticscholar.org: separate buckets let each
    # converge to ~1 req/s while S2 sees the SUM and 429s both (run
    # 21a145ad: every paper fetch 429'd at 1 call/min with the key
    # configured). Shared state paces the combined traffic S2 actually
    # enforces against. Legacy "s2"/"s2_bulk" names alias to the same
    # canonical provider so older callers/persisted events stay on the
    # shared bucket.
    _BUCKET_ALIAS = {
        "semantic_scholar_bulk": "semantic_scholar",
        "s2_bulk": "semantic_scholar",
        "s2": "semantic_scholar",
    }

    # Providers whose 429/503 retry needs a LONGER minimum wait than the
    # caller-supplied backoff_base. Observed in run fd6f4910: the S2 Graph
    # API 429'd us persistently even after AIMD cut the effective rate to
    # ~0.05 req/s (one call / 20s) — a 2-3s retry floor just burns every
    # retry on another 429. These providers get a hard floor so a retry is
    # actually likely to succeed. (semantic_scholar_bulk shares the floor
    # via the bucket alias above.)
    _THROTTLE_FLOOR = {"semantic_scholar": 10.0}

    # Sustained-throttle gate: after this many consecutive throttle signals
    # (429/503) a provider's requests are paused via a separate throttle gate
    # (NOT the circuit breaker) so is_available() short-circuits instead of
    # paying repeated ~3s sleeps that keep hitting 429. Distinct from the
    # breaker by design — 429 is flow control, not a hard failure, and the
    # pinned test test_circuit_breaker_not_tripped_by_429s forbids opening
    # the breaker on 429s. The gate expires and one probe request re-tests.
    _THROTTLE_OPEN_THRESHOLD = 6
    _THROTTLE_GATE_COOLDOWN = 90.0
    # Windowed gate: ALSO trip when throttles dominate recent traffic even
    # with successes interleaved (run 21a145ad: 429, sleep, retry-success,
    # next-paper 429 — the consecutive streak reset every cycle so the gate
    # above never engaged across an 8-minute 429 storm). The window persists
    # as memory; the trailing `and window[-1]` requires the LATEST signal to
    # be a throttle so one successful probe re-opens traffic immediately.
    _THROTTLE_WINDOW = 10
    _THROTTLE_WINDOW_THRESHOLD = 5

    @classmethod
    def _canonical_provider(cls, provider: str) -> str:
        """Shared-host alias: aliased providers pace/share one bucket+breaker."""
        try:
            return cls._BUCKET_ALIAS.get(provider, provider)
        except Exception:
            return provider

    def __init__(
        self,
        rates: Optional[dict[str, tuple[float, int]]] = None,
        circuit_threshold: int = 5,
        circuit_cooldown: float = 60.0,
    ):
        rates = rates or self.DEFAULT_RATES
        self._buckets: dict[str, AdaptiveTokenBucket] = {
            name: AdaptiveTokenBucket(rate=r, burst=b)
            for name, (r, b) in rates.items()
        }
        self._breakers = {
            name: CircuitBreaker(provider=name, threshold=circuit_threshold, cooldown=circuit_cooldown)
            for name in rates
        }
        self.health = SourceHealthTracker()
        # Single-flight coalescing: identical read-only calls issued while one
        # is already in flight share its Future instead of duplicating the HTTP
        # round trip (and the rate-limit slot).
        self._inflight: dict[str, Future] = {}
        self._inflight_lock = threading.Lock()
        # D7: guard bucket/breaker lazy creation (threads created them racily).
        self._registry_lock = threading.Lock()
        # Sustained-throttle gate state (see _THROTTLE_OPEN_THRESHOLD above).
        # Kept separate from the circuit breaker so 429 flow control never
        # trips the breaker, while is_available() can still fast-skip a
        # provider that is rejecting every request.
        self._throttle_streak: dict[str, int] = defaultdict(int)
        self._throttle_gate_until: dict[str, float] = {}
        # Windowed memory: last-N throttle flags per canonical provider, so
        # interleaved success/429 storms still trip the gate (streak alone
        # cannot — one success resets it). deque(maxlen) bounds memory.
        self._throttle_window: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

    @staticmethod
    def build_coalesce_key(provider: str, fn: Callable, args: tuple, kwargs: dict) -> str:
        """Namespaced single-flight key (D6): (provider, qualname, args, kwargs)."""
        try:
            qual = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or str(type(fn))
        except Exception:
            qual = "fn"
        try:
            payload = repr((args, sorted(((str(k), str(v)) for k, v in (kwargs or {}).items()))))
        except Exception:
            payload = "args"
        return f"{provider}:{qual}:{hash(payload) & 0xFFFFFFFF:08x}"

    def request(
        self,
        provider: str,
        fn: Callable,
        *args,
        retries: int = 3,
        backoff_base: float = 3.0,
        backoff_max: float = 180.0,
        rate_limit: bool = True,
        coalesce_key: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Execute fn(*args, **kwargs) with rate limiting, circuit breaking, and retry.

        Retry strategy:
          - 429 (RateLimitError): NO fast retry. The adaptive bucket is
            penalized (effective rate cut) and we sleep max(Retry-After,
            exponential backoff) before the next attempt. A 429 means
            "stop hammering me" — the bucket learns from it.
          - Transient errors (524, 502, 503, timeout, connection reset):
            Retry with exponential backoff + jitter.
          - Other errors: Raise immediately.

        Args:
            provider: API provider name (arxiv, openalex,
                semantic_scholar, llm, embedding).
            fn: Callable to execute.
            retries: Max retry attempts on transient errors.
            backoff_base: Base seconds for exponential backoff.
            backoff_max: Maximum backoff seconds.
            rate_limit: If False, skip token-bucket rate limiting (use when the
                        library or caller already enforces delays).
            coalesce_key: Optional key for single-flight dedup — identical
                          concurrent read-only calls share one execution.

        Returns:
            Result of fn(*args, **kwargs).

        Raises:
            CircuitOpen: If circuit breaker is open.
            RateLimitError: If rate limited (429).
            Exception: Original exception for non-retryable errors.
        """
        if coalesce_key is None:
            return self._request_inner(
                provider, fn, *args,
                retries=retries, backoff_base=backoff_base,
                backoff_max=backoff_max, rate_limit=rate_limit, **kwargs,
            )

        # Single-flight: only the leader executes; followers share its Future.
        # D6: str() guard + timeout + remove-on-cancel.
        coalesce_key = str(coalesce_key) if coalesce_key is not None else None
        coalesce_timeout = float(kwargs.pop("coalesce_timeout", 180.0)) if isinstance(kwargs, dict) else 180.0
        with self._inflight_lock:
            fut = self._inflight.get(coalesce_key)
            if fut is None:
                fut = Future()
                self._inflight[coalesce_key] = fut
                leader = True
            else:
                leader = False
        if not leader:
            logger.debug("Coalesced duplicate request for %s (%s)", provider, coalesce_key[:80])
            try:
                return fut.result(timeout=coalesce_timeout)
            except Exception:
                with self._inflight_lock:
                    self._inflight.pop(coalesce_key, None)
                raise
        try:
            result = self._request_inner(
                provider, fn, *args,
                retries=retries, backoff_base=backoff_base,
                backoff_max=backoff_max, rate_limit=rate_limit, **kwargs,
            )
            fut.set_result(result)
            return result
        except BaseException as e:
            fut.set_exception(e)
            raise
        finally:
            with self._inflight_lock:
                self._inflight.pop(coalesce_key, None)

    def _request_inner(
        self,
        provider: str,
        fn: Callable,
        *args,
        retries: int = 3,
        backoff_base: float = 3.0,
        backoff_max: float = 180.0,
        rate_limit: bool = True,
        **kwargs,
    ) -> Any:
        # Canonicalize early: throttle floors, health records, and
        # log lines all key on the canonical provider name, so a
        # legacy "s2" caller shares the semantic_scholar bucket,
        # breaker, gate, and 10s retry floor exactly.
        provider = self._canonical_provider(provider)
        # NOTE (e0ec9e91, reverted): fail-fast on the throttle gate lived here
        # briefly, but request() is the probe path — tests pin that a gated
        # provider still executes so a success can clear the gate
        # (test_success_clears_throttle_gate, windowed-gate + probe-reopen
        # tests). The 12-post burn is fixed one layer up instead:
        # LLMClient.chat pre-checks is_available("llm") and skips
        # model/provider fallbacks while the gate holds, so logical LLM calls
        # degrade without touching the gateway probe contract.
        bucket = self._get_bucket(provider) if rate_limit else None
        breaker = self._get_breaker(provider)
        last_error = None

        for attempt in range(retries + 1):
            # Circuit breaker check — acquire() never sleeps, raises immediately
            try:
                breaker.acquire()
            except CircuitOpen as e:
                if attempt < retries:
                    wait = self._jitter(min(backoff_max, backoff_base * (2 ** attempt)))
                    logger.warning(
                        "Circuit breaker open for %s — waiting %.1fs (attempt %d/%d)",
                        provider, wait, attempt + 1, retries + 1,
                    )
                    time.sleep(wait)
                    continue
                self.health.record(provider, False)
                raise

            # Rate limit: wait for token. D7: wait budget is separate from
            # the retry backoff budget (capped at 60s per attempt).
            wait_timeout = min(float(backoff_max), 60.0)
            if bucket and not bucket.acquire(timeout=wait_timeout):
                logger.warning("Rate limit timeout for %s after %.0fs", provider, wait_timeout)
                self.health.record(provider, False)
                continue

            try:
                result = fn(*args, **kwargs)
                breaker.record_success()
                self.health.record(provider, True)
                if bucket:
                    bucket.reward()
                self._clear_throttle(provider)
                return result
            except RateLimitError as e:
                last_error = e
                # 429 is a flow-control signal, NOT a provider failure.
                # Do NOT call breaker.record_failure() — the AIMD penalty
                # already slows us down. Counting 429s toward the breaker
                # threshold causes premature circuit-opening during brief
                # congestion spikes, blocking ALL traffic for cooldown.
                self.health.record(provider, False)
                self._note_throttle(provider)
                if bucket:
                    # AIMD: the provider told us we're too fast — slow down.
                    bucket.penalize(factor=0.5, cooldown=max(15.0, min(e.retry_after, 120.0)))
                if attempt >= retries:
                    raise
                sleep_time = max(e.retry_after, backoff_base * (2 ** attempt))
                sleep_time = max(sleep_time, self._THROTTLE_FLOOR.get(provider, 0.0))
                sleep_time = min(backoff_max, self._jitter(sleep_time))
                # Concise one-liner, deliberately WITHOUT exc_info (same
                # rationale as the throttle path below): a rate-limited retry
                # is routine flow control and the chained traceback turns 429
                # storms into apparent crashes in the logs.
                logger.warning(
                    "%s rate limited (429) — effective rate now %.2f req/s, sleeping %.1fs before retry %d/%d",
                    provider, bucket.current_rate() if bucket else -1, sleep_time, attempt + 1, retries + 1,
                )
                time.sleep(sleep_time)
                continue
            except Exception as e:
                last_error = e
                # Permanent client errors (4xx except 429) will NEVER succeed
                # on retry — e.g. an embeddings endpoint that answers 405
                # Method Not Allowed. Retrying those with exponential backoff
                # burns ~20s per call and trips the circuit breaker for no
                # benefit. Fail fast: one breaker count, one concise log line,
                # no sleeps, no exc_info.
                if self._is_non_retryable(e):
                    self.health.record(provider, False)
                    breaker.record_failure()
                    logger.warning(
                        "%s permanent client error, not retrying: %s",
                        provider, str(e)[:160],
                    )
                    raise
                # Generic throttling signals from third-party libs (e.g. the
                # official `arxiv` client raises plain HTTPError("Page request
                # resulted in HTTP 429 (...)"), NOT RateLimitError). Without
                # this, arxiv-429s skip the AIMD penalty above, never slow the
                # bucket, and instead count toward the circuit breaker.
                retry_after_hint = self._parse_throttle_signal(e)
                if retry_after_hint is not None:
                    self.health.record(provider, False)
                    self._note_throttle(provider)
                    if bucket:
                        bucket.penalize(
                            factor=0.5,
                            cooldown=max(15.0, min(retry_after_hint, 120.0)),
                        )
                    if attempt >= retries:
                        raise
                    sleep_time = max(retry_after_hint, backoff_base * (2 ** attempt))
                    sleep_time = max(sleep_time, self._THROTTLE_FLOOR.get(provider, 0.0))
                    sleep_time = min(backoff_max, self._jitter(sleep_time))
                    # Concise one-liner, deliberately WITHOUT exc_info: a
                    # throttled retry is routine flow control, not a bug, and
                    # the chained traceback (~60 lines x N retries) is what
                    # makes 429 storms look like crashes in the logs.
                    logger.warning(
                        "%s throttled (%s) — effective rate now %.2f req/s, sleeping %.1fs before retry %d/%d",
                        provider, str(e)[:120],
                        bucket.current_rate() if bucket else -1,
                        sleep_time, attempt + 1, retries + 1,
                    )
                    time.sleep(sleep_time)
                    continue
                if self._is_transient(e):
                    if attempt < retries:
                        # Respect provider's retry_after hint (e.g., Cloudflare 524)
                        provider_hint = self._parse_retry_after(e)
                        wait = min(backoff_max, self._jitter(provider_hint or (backoff_base * (2 ** attempt))))
                        # Concise one-liner, deliberately WITHOUT exc_info: a
                        # transient retry (network blip, DNS hiccup) is normal
                        # operation and the final failure below still logs the
                        # full traceback. During the run b099dcaa DNS outage
                        # this path emitted ~15 chained 80-line tracebacks.
                        logger.warning(
                            "%s transient error (attempt %d/%d): %s — retrying in %.1fs",
                            provider, attempt + 1, retries + 1, e, wait,
                        )
                        time.sleep(wait)
                        continue
                # Non-transient errors and exhausted transient retries
                # count as real failures for the circuit breaker.
                breaker.record_failure()
                self.health.record(provider, False)
                raise

        # All retries exhausted
        self.health.record(provider, False)
        if last_error:
            raise last_error
        return None

    @staticmethod
    def _jitter(wait: Optional[float]) -> float:
        """Add +/-20% jitter to a backoff wait to avoid synchronized retries."""
        if not wait or wait <= 0:
            return wait or 0.0
        return max(0.0, wait * random.uniform(0.8, 1.2))

    def is_available(self, provider: str) -> bool:
        """True if the circuit breaker for `provider` is not open.

        Also False while the provider is inside a sustained-throttle gate
        (consecutive 429s, OR throttles dominating the recent window with the
        latest signal a throttle). Callers (e.g. TopicHunter) probe this
        before building queries so an open breaker OR a throttled-out
        provider short-circuits to a skip instead of paying retry sleeps.
        Aliased providers (semantic_scholar_bulk → semantic_scholar,
        legacy s2/s2_bulk → semantic_scholar) share gate state.
        """
        canon = self._canonical_provider(provider)
        with self._registry_lock:
            gate_until = self._throttle_gate_until.get(canon, 0.0)
            if gate_until and gate_until <= time.monotonic():
                # Cooldown expired: fresh probe slate. The window is kept as
                # memory so continued failure re-gates fast, but the stale
                # gate flag itself must not block the probe.
                self._throttle_gate_until.pop(canon, None)
                self._throttle_streak[canon] = 0
            elif gate_until > time.monotonic():
                return False
        return not self._get_breaker(provider).is_open

    def _note_throttle(self, provider: str) -> None:
        """Record a 429/503 flow-control signal.

        Sustained throttling pauses the provider via the throttle gate so
        is_available() fast-skips it. The breaker is deliberately untouched
        (429 is flow control, not a failure). Gate trips on EITHER a
        consecutive streak (fast storm) OR a throttle-dominated recent window
        whose latest signal is a throttle (interleaved storm — a lone success
        must not grant amnesty while the provider still rejects most traffic,
        but a successful *probe* re-opens traffic since the window's tail
        then shows success).
        """
        canon = self._canonical_provider(provider)
        with self._registry_lock:
            streak = self._throttle_streak.get(canon, 0) + 1
            self._throttle_streak[canon] = streak
            window = self._throttle_window[canon]
            window.append(True)
            windowed = sum(1 for v in window if v)
            if streak >= self._THROTTLE_OPEN_THRESHOLD or (
                windowed >= self._THROTTLE_WINDOW_THRESHOLD and window[-1]
            ):
                self._throttle_gate_until[canon] = time.monotonic() + self._THROTTLE_GATE_COOLDOWN
                self._throttle_streak[canon] = 0
                logger.warning(
                    "%s: %d throttle signals in recent window (%d consecutive) — pausing %s traffic for %.0fs "
                    "(is_available() will short-circuit)",
                    provider, windowed, streak, provider, self._THROTTLE_GATE_COOLDOWN,
                )

    def _clear_throttle(self, provider: str) -> None:
        """Decay throttle memory after a successful request."""
        canon = self._canonical_provider(provider)
        with self._registry_lock:
            self._throttle_streak[canon] = 0
            self._throttle_window[canon].append(False)
            self._throttle_gate_until.pop(canon, None)

    def breaker_state(self, provider: str) -> CircuitState:
        return self._get_breaker(provider).state

    def effective_rate(self, provider: str) -> float:
        """Current adaptive rate for a provider (requests/second)."""
        return self._get_bucket(provider).current_rate()

    def _get_bucket(self, provider: str) -> AdaptiveTokenBucket:
        # D7: locked lazy creation. Aliased providers share one bucket.
        provider = self._canonical_provider(provider)
        with self._registry_lock:
            if provider not in self._buckets:
                self._buckets[provider] = AdaptiveTokenBucket(rate=0.5, burst=1)
                self._breakers[provider] = CircuitBreaker(provider=provider)
            return self._buckets[provider]

    def _get_breaker(self, provider: str) -> CircuitBreaker:
        provider = self._canonical_provider(provider)
        with self._registry_lock:
            if provider not in self._breakers:
                self._breakers[provider] = CircuitBreaker(provider=provider)
            return self._breakers[provider]

    @staticmethod
    def _is_transient(e: Exception) -> bool:
        """Check if exception is transient (worth retrying). D8 extended."""
        # Structured status/code inspection first (D8).
        for attr in ("status_code", "status", "code"):
            try:
                raw = getattr(e, attr, None)
                code = int(raw) if raw is not None and str(raw).isdigit() else 0
            except (TypeError, ValueError):
                code = 0
            if code in (429, 500, 502, 503, 504):
                return True
        msg = str(e).lower()
        return any(kw in msg for kw in (
            "524", "502", "503", "500", "504", "timeout", "timed out",
            "connection", "reset", "eof", "broken pipe", "connectionreset",
            "overloaded", "unavailable", "deadline", "try again",
            # Provider-side mid-stream failures: the stream starts (200) and
            # then dies without a status code. Observed from the dahl.global
            # inference endpoint: "winner inference incomplete (nonce_finished
            # =false)". These are transient infra hiccups and MUST retry /
            # fall back — treating them as permanent silently zeroed an LLM
            # generation in run fd6f4910 (seed produced 0 gaps).
            "nonce_finished", "winner inference incomplete", "stream ended",
            "incomplete", "truncated stream",
        ))

    # HTTP status codes that are permanent client errors: retrying cannot help.
    # 429 (rate limit) is deliberately excluded — it is flow control and is
    # handled by the AIMD path above. 503 is excluded — it is transient.
    _NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 405, 409, 410, 422})

    # Plain-language markers of permanent client errors, for exceptions that
    # do not carry a structured status code.
    _NON_RETRYABLE_PHRASES = (
        "method not allowed",
        "not found",
        "unauthorized",
        "forbidden",
        "authentication",
        "invalid model",
        "does not exist",
    )

    @staticmethod
    def _is_non_retryable(e: Exception) -> bool:
        """True when `e` is a permanent client error that must not be retried.

        Checks structured status codes first (`status_code` as used by the
        OpenAI SDK, `status` as used by the `arxiv` client), then falls back
        to message sniffing. 429/503 are never non-retryable here — they are
        owned by the throttle/transient paths.
        """
        for attr in ("status_code", "status"):
            try:
                status = int(getattr(e, attr, 0) or 0)
            except (TypeError, ValueError):
                status = 0
            if status in APIGateway._NON_RETRYABLE_STATUS:
                return True
            if status in (429, 503):
                return False
        import re as _re
        msg = str(e) or ""
        m = _re.search(r"\berror\s*code\s*:\s*(\d{3})\b", msg, _re.IGNORECASE)
        if not m:
            m = _re.search(r"\bstatus\s*code\s*(\d{3})\b", msg, _re.IGNORECASE)
        if not m:
            m = _re.search(r"\bhttp\s*(\d{3})\b", msg, _re.IGNORECASE)
        if m:
            try:
                code = int(m.group(1))
            except ValueError:
                code = 0
            if code in APIGateway._NON_RETRYABLE_STATUS:
                return True
            if code in (429, 503):
                return False
        low = msg.lower()
        return any(phrase in low for phrase in APIGateway._NON_RETRYABLE_PHRASES)

    @staticmethod
    def _parse_retry_after(e: Exception) -> Optional[float]:
        """Extract retry_after hint from error response (e.g., Cloudflare 524)."""
        import json as _json
        import re as _re
        msg = str(e)
        # Look for 'retry_after': N in the error dict
        for token in ("'retry_after': ", '"retry_after": '):
            idx = msg.find(token)
            if idx != -1:
                rest = msg[idx + len(token):]
                num_str = ""
                for ch in rest:
                    if ch.isdigit() or ch == '.':
                        num_str += ch
                    else:
                        break
                if num_str:
                    try:
                        return float(num_str)
                    except ValueError:
                        pass
        # HTTP-style hints: "Retry-After: 30", "retry after 30s", "retry_after=30"
        m = _re.search(r"retry[\s_-]*after\s*[:=]?\s*(\d+(?:\.\d+)?)", msg, _re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    @staticmethod
    def _parse_throttle_signal(e: Exception) -> Optional[float]:
        """Detect generic throttling errors from third-party libs.

        Returns the retry-after hint (seconds) when `e` looks like a
        rate-limit / 429 / 503 signal, else None. Covers e.g. the official
        `arxiv` client error: "Page request resulted in HTTP 429 (...)".
        Callers use the hint as the floor for exponential backoff; when no
        explicit hint is present the caller's backoff_base applies.
        """
        import re as _re
        # Structured signal first: e.g. arxiv.HTTPError.status == 429. This
        # survives message rewording, unlike the string sniffing below.
        try:
            status = int(getattr(e, "status", 0) or 0)
        except (TypeError, ValueError):
            status = 0
        if status in (429, 503):
            hint = APIGateway._parse_retry_after(e)
            return hint if hint is not None else 5.0
        msg = str(e) or ""
        low = msg.lower()
        throttled = (
            "429" in msg
            or "503" in msg
            or "rate limit" in low
            or "rate_limit" in low
            or "rate-limited" in low
            or "too many requests" in low
            or "retry-after" in low
            or "retry_after" in low
        )
        if not throttled:
            return None
        # Avoid false positives like "error 4290" — require word boundary.
        if not (
            _re.search(r"\b429\b", msg)
            or _re.search(r"\b503\b", msg)
            or "rate limit" in low
            or "rate_limit" in low
            or "rate-limited" in low
            or "too many requests" in low
            or "retry-after" in low
            or "retry_after" in low
        ):
            return None
        hint = APIGateway._parse_retry_after(e)
        # No explicit hint from arXiv-style errors — fall back to a
        # conservative floor so the next attempt is not immediate.
        return hint if hint is not None else 5.0


# Module-level singleton
_gateway: Optional[APIGateway] = None
_gateway_lock = threading.Lock()


def get_gateway(**kwargs) -> APIGateway:
    global _gateway
    if _gateway is None:
        with _gateway_lock:
            if _gateway is None:
                _gateway = APIGateway(**kwargs)
    return _gateway


def reset_gateway():
    global _gateway
    with _gateway_lock:
        _gateway = None
