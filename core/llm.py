"""
Multi-provider LLM client: Gemini, OpenAI, and any OpenAI-compatible endpoint.
Agents should use call_llm / generate_embedding / get_llm_client — not provider SDKs directly.

Rate limiting and circuit breaking are handled by core.api_gateway.

GEMINI-ONLY EMBEDDING POLICY (owner-locked, 2026-10-03):
  Chat may use any provider (gemini | openai | openai_compatible), but
  embeddings ALWAYS go through `gemini-embedding-2` at native 3072 dims,
  no matter which chat provider is configured. Rationale: compatible chat
  endpoints frequently expose no embeddings route at all (observed 100%
  HTTP 405 from a MiniMax endpoint), which silently zeroed every novelty
  vector; and the on-disk FAISS index (memory/vector_db) plus all stored
  vectors are 3072-wide Gemini vectors — mixing providers would corrupt
  similarity search. There is intentionally NO OpenAI embeddings call
  anywhere in this codebase (no `embeddings.create`); do not add one.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Optional

import numpy as np

from .api_gateway import get_gateway
from .config import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "2.0"))
_THINKING_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


class LLMUnavailable(Exception):
    """Raised on strict LLM paths (D1: allow_empty=False) instead of ""-success."""


def is_llm_outage() -> bool:
    """Global outage detector (M0 §6.2.1): llm_failures == llm_calls (calls > 0).

    When tripped, callers must label infra_llm_unavailable and must NOT
    record science lessons from the empty outputs.
    """
    try:
        from .run_log import get_tracker

        tracker = get_tracker()
        if not tracker:
            return False
        stats = tracker.stats or {}
        calls = int(stats.get("llm_calls") or 0)
        fails = int(stats.get("llm_failures") or 0)
        return calls > 0 and fails >= calls
    except Exception:
        return False


def _is_retryable_llm_error(e: Exception) -> bool:
    """D2: fallback via gateway transient/throttle classifiers (not substrings)."""
    try:
        from .api_gateway import APIGateway

        if APIGateway._is_transient(e):
            return True
        if APIGateway._parse_throttle_signal(e) is not None:
            return True
        msg = str(e).lower()
        if any(k in msg for k in ("429", "500", "502", "503", "504", "overloaded", "unavailable", "timed out", "deadline", "try again", "connection", "timeout", "empty", "no response", "truncated", "blocked",
                                  "nonce_finished", "winner inference incomplete", "incomplete", "truncated stream")):
            return True
    except Exception:
        pass
    return False

# --- Embedding backend health (Gemini-only route) ---
# A permanent failure (missing key, 4xx such as 404 unknown model / 405 no
# route) poisons every novelty vector with zeros. The flag below latches on
# the first such failure so later calls short-circuit without paying gateway
# retries or flooding the log; callers switch to the TF-IDF fallback.
_EMBEDDING_DEAD = False
_EMBEDDING_FAILURE_LOGGED = False
_EMBEDDING_LOCK = None  # lazy threading.Lock (avoid import cost at module load)


def _embedding_lock():
    global _EMBEDDING_LOCK
    if _EMBEDDING_LOCK is None:
        import threading as _th

        _EMBEDDING_LOCK = _th.Lock()
    return _EMBEDDING_LOCK


def _embedding_permanently_failed() -> bool:
    with _embedding_lock():
        return _EMBEDDING_DEAD


def is_embedding_available() -> bool:
    """True when the Gemini embedding route is usable (key present, no latch)."""
    if _EMBEDDING_DEAD:
        return False
    try:
        gateway_available = get_gateway().is_available("embedding")
    except Exception:
        gateway_available = True
    return bool(config.google_api_key) and gateway_available


def reset_embedding_health() -> None:
    """Clear the embedding-failure latch (tests / key rotation)."""
    global _EMBEDDING_DEAD, _EMBEDDING_FAILURE_LOGGED
    with _embedding_lock():
        _EMBEDDING_DEAD = False
        _EMBEDDING_FAILURE_LOGGED = False


def _classify_embedding_failure(e: Exception) -> bool:
    """True when `e` is permanent (latch the backend dead)."""
    try:
        from .api_gateway import APIGateway

        if APIGateway._is_non_retryable(e):
            return True
    except Exception:
        pass
    msg = str(e).lower()
    return any(kw in msg for kw in (
        "method not allowed", "not found", "unauthorized", "forbidden",
        "api key", "apikey", "permission denied", "invalid model",
    ))


def _note_embedding_unavailable(reason: str) -> None:
    global _EMBEDDING_DEAD, _EMBEDDING_FAILURE_LOGGED
    with _embedding_lock():
        _EMBEDDING_DEAD = True
        first = not _EMBEDDING_FAILURE_LOGGED
        _EMBEDDING_FAILURE_LOGGED = True
    # Full traceback once for diagnosis; afterwards callers already know the
    # backend is down and log their own one-line fallback notices.
    if first:
        logger.warning("Embedding backend unavailable (%s); using TF-IDF fallback", reason[:200], exc_info=True)
    else:
        logger.debug("Embedding backend still unavailable (%s)", reason[:120])


def _note_embedding_transient(e: Exception) -> None:
    # Transient (rate limit / 5xx / socket): worth exactly one concise line —
    # the gateway already logged the retry detail. Never exc_info here: a
    # novelty precompute fans out to ~20 of these per seed and the chained
    # tracebacks were 90% of a 12k-line run log.
    logger.debug("Transient embedding failure, zeros returned: %s", str(e)[:160])


class LLMClient:
    """Thin wrapper around Gemini or OpenAI-compatible chat + embeddings."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or config.llm_provider or "gemini").lower()
        self._gemini = None
        self._openai = None
        self._init_clients()

    def _init_clients(self):
        # Chat follows the configured provider. Embeddings ALWAYS use Gemini
        # (see embed()): an OpenAI-compatible chat endpoint (MiniMax, Ollama,
        # local vLLM, …) frequently has no /v1/embeddings route at all, and
        # routing embedding traffic there produced 100% 405 failures.
        # Both clients are therefore initialised opportunistically whenever
        # their key is present; only the *chat* provider's key is mandatory.
        self._gemini = None
        self._openai = None
        from google import genai
        if config.google_api_key:
            self._gemini = genai.Client(api_key=config.google_api_key)

        if self.provider in ("openai", "openai_compatible"):
            from openai import OpenAI

            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI-compatible provider")
            kwargs = {"api_key": config.openai_api_key, "max_retries": 0, "timeout": float(getattr(config, "llm_timeout_sec", 1800.0) or 1800.0)}
            if config.openai_base_url:
                kwargs["base_url"] = config.openai_base_url
            self._openai = OpenAI(**kwargs)
        elif self.provider == "gemini":
            if not self._gemini:
                raise ValueError("GOOGLE_API_KEY required for Gemini provider")
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        # Opportunistic OpenAI client for provider-level chat fallback even
        # when Gemini is primary (mirrors _get_fallback_provider).
        if self._openai is None and config.openai_api_key:
            try:
                from openai import OpenAI as _OpenAI

                _kwargs: dict = {"api_key": config.openai_api_key, "max_retries": 0,
                                 "timeout": float(getattr(config, "llm_timeout_sec", 1800.0) or 1800.0)}
                if config.openai_base_url:
                    _kwargs["base_url"] = config.openai_base_url
                self._openai = _OpenAI(**_kwargs)
            except Exception:
                self._openai = None

    def chat(
        self,
        prompt: str,
        temperature: float = 0.7,
        model: Optional[str] = None,
        max_tokens: int = 8192,
        system: Optional[str] = None,
        allow_empty: bool = True,
        retry_on_truncation: bool = True,
        media_parts: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        model_id = model or config.resolve_model("default")
        gateway = get_gateway()
        logger.debug("LLM chat request: provider=%s model=%s temp=%.2f prompt_len=%d",
                      self.provider, model_id, temperature, len(prompt))
        if media_parts and self.provider != "gemini":
            raise LLMUnavailable("File content requires a Gemini model; the configured provider does not accept this media input.")
        # e0ec9e91: never pay 12 posts when the gate is holding. Pre-check
        # so logical calls fail fast and let debate/planner degrade.
        # Compat: allow_empty callers expect "" on infra, not a raise.
        try:
            if not gateway.is_available("llm"):
                if allow_empty:
                    logger.warning("LLM gate holding; returning empty to allow_empty caller (no retry posts)")
                    return ""
                raise LLMUnavailable("LLM throttle gate holding for llm; failing fast (no retry posts)")
        except LLMUnavailable:
            raise
        except Exception:
            pass
        try:
            result = gateway.request(
                "llm",
                self._chat_raw,
                prompt, temperature, model_id, max_tokens, system,
                retry_on_truncation=retry_on_truncation,
                media_parts=media_parts,
            )
            if result:
                result = _THINKING_TAG_RE.sub("", result).strip()
            # 0dfc1c58: HTTP-200 provider quota prose ("Sorry, to prevent
            # abuse of free resources ... topup") is NOT model output. It
            # passed every caller as a valid argument/code (188 chars ->
            # SyntaxError line 1 x48, 20 plan revisions). Fail fast here so
            # it counts as llm_failure/infra instead of science.
            try:
                from .utils import is_provider_quota_text as _is_quota

                if result and _is_quota(result):
                    raise LLMUnavailable(
                        f"LLM provider quota notice ({self.provider}/{model_id}): "
                        + str(result)[:200]
                    )
            except LLMUnavailable:
                raise
            except Exception:
                pass
            if not result and not allow_empty:
                raise LLMUnavailable(f"LLM returned empty ({self.provider}/{model_id})")
            return result
        except LLMUnavailable:
            # P0-1: infra accounting must happen even on the re-raise path.
            # Previously llm_failures stayed 0 on truncation/empty (c22341c4),
            # blinding is_llm_outage(). Bump + emit before propagating so the
            # debate layer can classify infra vs science.
            try:
                from .run_log import emit_event as _emit, get_tracker as _get_tracker

                _tracker = _get_tracker()
                if _tracker:
                    _tracker.bump("llm_failures")
                    _tracker.message(f"LLM infra failure ({self.provider}/{model_id})", level="error")
                    try:
                        _emit("llm_error", {"model": model_id, "cause_class": "infra"}, agent="LLMClient")
                    except Exception:
                        pass
            except Exception:
                pass
            raise
        except Exception as e:
            # D2: fallback via gateway transient/throttle classifiers.
            # e0ec9e91: 3x4=12 posts per logical call during a shared-host
            # outage. When the gate/breaker is holding, every fallback shares
            # the same bucket — skip them and fail fast.
            if _is_retryable_llm_error(e):
                if media_parts:
                    raise LLMUnavailable(f"Gemini media request failed: {e}") from e
                try:
                    if not gateway.is_available("llm"):
                        logger.warning("LLM gate holding; skipping model/provider fallbacks (fail fast)")
                        raise LLMUnavailable("LLM gate holding; fallbacks skipped")
                except LLMUnavailable:
                    raise
                except Exception:
                    pass
                try:
                    fallback_model = config.resolve_fallback_model(model_id)
                except Exception:
                    fallback_model = getattr(config, "llm_model_fallback", None)
                if fallback_model and fallback_model != model_id:
                    logger.warning("LLM primary model %s failed (%s), trying fallback %s",
                                   model_id, str(e)[:100], fallback_model)
                    try:
                        result = gateway.request(
                            "llm",
                            self._chat_raw,
                            prompt, temperature, fallback_model, max_tokens, system,
                            retry_on_truncation=retry_on_truncation,
                        )
                        if result:
                            result = _THINKING_TAG_RE.sub("", result).strip()
                        try:
                            from .utils import is_provider_quota_text as _is_quota2

                            if result and _is_quota2(result):
                                raise LLMUnavailable(
                                    f"LLM provider quota notice ({fallback_model}): "
                                    + str(result)[:200]
                                )
                        except LLMUnavailable:
                            raise
                        except Exception:
                            pass
                        if not result and not allow_empty:
                            raise LLMUnavailable(f"LLM fallback returned empty ({fallback_model})")
                        return result
                    except LLMUnavailable:
                        raise
                    except Exception as fallback_err:
                        logger.warning("LLM fallback also failed (%s/%s): %s",
                                       self.provider, fallback_model, fallback_err)
                other_provider, other_model = self._get_fallback_provider(model_id)
                if other_provider:
                    logger.warning("LLM provider %s failed (%s), trying provider fallback %s/%s",
                                   self.provider, str(e)[:80], other_provider, other_model)
                    try:
                        client = LLMClient(provider=other_provider)
                        result = gateway.request(
                            "llm",
                            client._chat_raw,
                            prompt, temperature, other_model, max_tokens, system,
                            retry_on_truncation=retry_on_truncation,
                        )
                        if result:
                            result = _THINKING_TAG_RE.sub("", result).strip()
                        try:
                            from .utils import is_provider_quota_text as _is_quota3

                            if result and _is_quota3(result):
                                raise LLMUnavailable(
                                    f"LLM provider quota notice ({other_provider}): "
                                    + str(result)[:200]
                                )
                        except LLMUnavailable:
                            raise
                        except Exception:
                            pass
                        if not result and not allow_empty:
                            raise LLMUnavailable(f"LLM provider fallback returned empty ({other_provider})")
                        return result
                    except LLMUnavailable:
                        raise
                    except Exception as prov_err:
                        logger.warning("LLM provider fallback %s also failed: %s", other_provider, prov_err)

            # P0-3: concise one-liner, never exc_info. A transport/DNS blip is
            # routine on a slow endpoint; the chained 80-line traceback per
            # retry turned the b099dcaa outage into ~800 lines of noise.
            # Full tracebacks are logged only by the run-level failure dossier.
            logger.warning("LLM chat failed (%s/%s): %s", self.provider, model_id, str(e)[:200])
            try:
                from .run_log import emit_event, get_tracker

                tracker = get_tracker()
                if tracker:
                    tracker.bump("llm_failures")
                    tracker.message(f"LLM call failed ({self.provider}/{model_id}): {e}", level="error")
                    try:
                        emit_event("llm_error", {"model": model_id, "error": str(e)[:300], "cause_class": "infra"}, agent="LLMClient")
                    except Exception:
                        pass
                    if is_llm_outage():
                        tracker.message("LLM outage detected (failures == calls): infra_llm_unavailable, no science lessons", level="error")
            except Exception as tracker_err:
                logger.debug("Failed to record LLM failure: %s", tracker_err)
            if not allow_empty:
                raise LLMUnavailable(str(e)) from e
            return ""

    def _get_fallback_provider(self, current_model: str) -> tuple:
        """Return (fallback_provider, fallback_model) or (None, None) if no fallback available."""
        # Use dedicated provider-level fallback config if set
        configured_provider = getattr(config, "llm_fallback_provider", "")
        configured_model = getattr(config, "llm_fallback_model", "")
        if configured_provider and configured_model:
            return (configured_provider, configured_model)

        # Auto-detect: fall back to the other provider
        if self.provider in ("openai", "openai_compatible"):
            # Reuse the configured Gemini chat model (env GEMINI_MODEL) — a
            # hardcoded default goes stale as providers retire models
            # (gemini-2.0-flash started 404ing in run b099dcaa).
            gemini_model = configured_model or getattr(config, "gemini_model", "") or "gemini-2.5-flash"
            return ("gemini", gemini_model)
        elif self.provider == "gemini":
            if getattr(config, "openai_api_key", None):
                openai_model = configured_model or "gpt-4o-mini"
                return ("openai_compatible", openai_model)
        return (None, None)

    def _chat_raw(
        self,
        prompt: str,
        temperature: float,
        model_id: str,
        max_tokens: int,
        system: Optional[str],
        retry_on_truncation: bool = True,
        media_parts: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        if self.provider == "gemini":
            return self._chat_gemini(prompt, temperature, model_id, max_tokens, system,
                                     retry_on_truncation=retry_on_truncation, media_parts=media_parts)
        if media_parts:
            raise LLMUnavailable("File content requires a Gemini model; this provider does not accept media input.")
        return self._chat_openai(prompt, temperature, model_id, max_tokens, system,
                                 retry_on_truncation=retry_on_truncation)

    def _chat_gemini(
        self,
        prompt: str,
        temperature: float,
        model_id: str,
        max_tokens: int,
        system: Optional[str],
        retry_on_truncation: bool = True,
        media_parts: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        contents: Any = prompt
        if media_parts:
            from google.genai import types

            contents = [prompt]
            for media in media_parts:
                contents.append(
                    types.Part.from_bytes(
                        data=media["data"],
                        mime_type=media["mime_type"],
                    )
                )
        gen_config: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            gen_config["system_instruction"] = system
        # Gemma 4's hosted Gemini API supports a binary thinking control.
        # Apply it in this single shared chat path so every agent, prompt
        # profile, JSON repair, and engineering call benefits when Gemma is
        # selected.  Other Gemini-family models and all other providers keep
        # their existing request shape unchanged.
        try:
            if config.is_gemma_4_model(model_id):
                gen_config["thinking_config"] = {
                    "thinking_level": str(config.gemma_thinking_level).strip().lower(),
                }
        except Exception:
            # A mocked/older Config must never prevent an ordinary request.
            pass
        model_name = model_id if model_id.startswith("models/") else f"models/{model_id}"

        def _extract(response) -> tuple:
            try:
                cand = (getattr(response, "candidates", None) or [None])[0]
                finish = str(getattr(cand, "finish_reason", "") or getattr(response, "finish_reason", "") or "").upper()
            except Exception:
                finish = ""
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception as e:
                logger.debug("Failed to parse Gemini response: %s", e)
                text = getattr(response, "text", "") or ""
            return finish, text

        # D5: finish_reason=length must NOT silently return a cut-off answer —
        # same rationale as _chat_openai: truncated JSON wipes whole seeds.
        eff_tokens = int(max_tokens or 8192)
        longest = ""
        for _attempt in range(3):
            try:
                response = self._gemini.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config={**gen_config, "max_output_tokens": eff_tokens},
                )
            except Exception as retry_err:
                if _attempt > 0:
                    logger.warning("Gemini retry with max_output_tokens=%d failed: %s — returning best-effort text",
                                   eff_tokens, retry_err)
                    break
                raise
            finish, text = _extract(response)
            blocked = any(k in finish for k in ("SAFETY", "BLOCK", "RECITATION", "PROHIBITED", "SPII"))
            if blocked:
                raise LLMUnavailable(f"LLM blocked (finish_reason={finish})")
            if ("LENGTH" in finish or "MAX_TOKENS" in finish) and text:
                if len(text) > len(longest):
                    longest = text
                if not retry_on_truncation:
                    logger.warning("Gemini truncated at %d tokens (json_mode: returning partial %d chars for repair)",
                                   eff_tokens, len(text))
                    return text
                next_tokens = eff_tokens * 2
                _ceiling = int(getattr(config, "llm_max_output_tokens", 65536) or 65536)
                if next_tokens > _ceiling or next_tokens <= eff_tokens:
                    logger.warning("Gemini response truncated (finish_reason=%s) at max budget %d — returning best effort (%d chars)",
                                   finish, eff_tokens, len(text))
                    return text
                logger.warning("Gemini response truncated (finish_reason=%s); retrying with max_output_tokens=%d", finish, next_tokens)
                eff_tokens = next_tokens
                continue
            return text
        return longest or ""

    def _chat_openai(
        self,
        prompt: str,
        temperature: float,
        model_id: str,
        max_tokens: int,
        system: Optional[str],
        retry_on_truncation: bool = True,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # finish_reason=length must NOT silently return a cut-off payload:
        # a truncated JSON answer gets parsed as "no gaps" and the whole seed
        # is wasted (run fd6f4910: 3 truncated answers -> 3 seeds with 0 gaps).
        # So retry with a doubled token budget (bounded) — the model re-runs
        # fresh, thinking tokens excluded by the caller, and the full answer
        # is preserved. If the enlarged budget is rejected by the provider,
        # fall back to the longest text we did collect.
        eff_tokens = int(max_tokens or 8192)
        longest = ""
        for _attempt in range(3):
            try:
                stream = self._openai.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=eff_tokens,
                    stream=True,
                )
            except Exception as retry_err:
                # Enlarged budget rejected (e.g. provider caps max_tokens) or
                # transport error — return best effort rather than crashing.
                if _attempt > 0:
                    logger.warning("LLM retry with max_tokens=%d failed: %s — returning best-effort text",
                                   eff_tokens, retry_err)
                    break
                raise
            collected = []
            finish_reasons: list[str] = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                try:
                    fr = str(getattr(choice, "finish_reason", "") or "")
                    if fr and fr != "None":
                        finish_reasons.append(fr)
                except Exception:
                    pass
                delta = choice.delta
                if delta and delta.content is not None:
                    collected.append(delta.content)
            # D5: surface truncation/blocked instead of silent short output.
            try:
                joined_fr = " ".join(finish_reasons).lower()
                if any(k in joined_fr for k in ("content_filter", "blocked", "safety")):
                    raise LLMUnavailable(f"LLM blocked (finish_reason={finish_reasons})")
                text = "".join(collected)
                if "length" in joined_fr and not text:
                    raise LLMUnavailable("LLM truncated with no content (finish_reason=length)")
                if "length" in joined_fr and text:
                    if len(text) > len(longest):
                        longest = text
                    if not retry_on_truncation:
                        # P0-3 json_mode: strict-JSON callers (formalize delta
                        # contract) prefer the partial payload now and let
                        # call_llm_json repair it, instead of paying a second
                        # ~300s full re-run on a slow endpoint.
                        logger.warning("LLM truncated at %d tokens (json_mode: returning partial %d chars for repair)",
                                       eff_tokens, len(text))
                        return text
                    next_tokens = eff_tokens * 2
                    _ceiling = int(getattr(config, "llm_max_output_tokens", 65536) or 65536)
                    if next_tokens > _ceiling or next_tokens <= eff_tokens:
                        logger.warning("LLM response truncated (finish_reason=length) at max budget %d — returning best effort (%d chars)",
                                       eff_tokens, len(text))
                        return text
                    logger.warning("LLM response truncated (finish_reason=length); retrying with max_tokens=%d", next_tokens)
                    eff_tokens = next_tokens
                    continue
                return text
            except LLMUnavailable:
                raise
            except Exception:
                pass
        return longest or "".join(collected)

    def embed(self, text: str, model: Optional[str] = None) -> np.ndarray:
        """Generate an embedding, ALWAYS via gemini-embedding-2, through the gateway.

        GEMINI-ONLY POLICY: the chat provider (Gemini vs OpenAI-compatible)
        selects the chat route only and NEVER the embedding route. Embedding
        traffic is pinned to Gemini because compatible chat endpoints
        frequently expose no embeddings route (observed: 100% HTTP 405 from
        a MiniMax endpoint), which silently zeroed every novelty vector.
        When Gemini is unavailable the failure is recorded once
        (module-level permanent flag + concise logging) and every later
        call short-circuits to zeros without paying gateway retries — callers
        must consult is_embedding_available() and use the TF-IDF fallback in
        core.utils instead of trusting the zeros.
        """
        logger.debug("Embedding request: text_len=%d", len(text))
        if self._gemini is None:
            _note_embedding_unavailable("no GOOGLE_API_KEY: Gemini embeddings unavailable")
            return np.zeros(config.embedding_dimension)
        if _embedding_permanently_failed():
            logger.debug("Skipping embedding request (backend marked permanently failed)")
            return np.zeros(config.embedding_dimension)
        try:
            # Embeddings used to bypass rate limiting and circuit breaking,
            # making novelty precomputation an unbounded source of stalls.
            # Route them through the dedicated embedding bucket. The raw fn
            # below is Gemini-only — the chat provider is never honored here.
            return get_gateway().request("embedding", self._embed_gemini_raw, text, model)
        except Exception as e:
            if _classify_embedding_failure(e):
                _note_embedding_unavailable(f"Gemini embeddings permanently failed: {e}")
            else:
                _note_embedding_transient(e)
            # D4: zeros only with a visible llm_error (§10.4 forbids silent zeros).
            try:
                from .run_log import emit_event, get_tracker

                tracker = get_tracker()
                if tracker:
                    tracker.bump("llm_failures")
                    tracker.message(f"Embedding failed, zeros returned: {e}", level="error")
                    try:
                        emit_event("llm_error", {"model": "embedding", "error": str(e)[:300], "cause_class": "infra"}, agent="LLMClient")
                    except Exception:
                        pass
            except Exception:
                pass
            return np.zeros(config.embedding_dimension)

    def _embed_gemini_raw(self, text: str, model: Optional[str]) -> np.ndarray:
        return self._embed_gemini(text, model)

    def _embed_raw(self, text: str, model: Optional[str]) -> np.ndarray:
        # Backward-compat entry point for the gateway/tests. GEMINI-ONLY
        # POLICY: embeddings are gemini-embedding-2 regardless of the chat
        # provider. When no Gemini client exists this raises instead of
        # falling back to another provider — mixing embedding spaces would
        # silently corrupt FAISS similarity (3072-wide Gemini vectors).
        if self._gemini is not None:
            return self._embed_gemini(text, model)
        raise LLMUnavailable("no Gemini client: embeddings unavailable (never OpenAI)")

    def _embed_gemini(self, text: str, model: Optional[str]) -> np.ndarray:
        # GEMINI-ONLY: `model` is an override for the Gemini embedding model
        # id only — never pass an OpenAI model id here.
        model_id = model or config.gemini_embedding_model
        model_name = model_id if model_id.startswith("models/") else f"models/{model_id}"
        response = self._gemini.models.embed_content(
            model=model_name,
            contents=text,
            config={"output_dimensionality": config.embedding_dimension},
        )
        if hasattr(response, "embeddings") and response.embeddings:
            vec = np.array(response.embeddings[0].values, dtype=float)
        elif hasattr(response, "embedding") and response.embedding:
            vals = response.embedding.values
            vec = np.array(vals[0].values if hasattr(vals[0], "values") else vals, dtype=float)
        else:
            vec = np.array(response.values[0].values, dtype=float)
        actual_dim = vec.shape[0]
        if actual_dim != config.embedding_dimension:
            logger.warning(
                "Gemini embedding returned %d dims, expected %d. Adjusting.",
                actual_dim, config.embedding_dimension,
            )
            if actual_dim < config.embedding_dimension:
                vec = np.pad(vec, (0, config.embedding_dimension - actual_dim))
            else:
                vec = vec[:config.embedding_dimension]
        return vec


_client: Optional[LLMClient] = None


def get_llm_client(force_new: bool = False) -> LLMClient:
    global _client
    if _client is None or force_new:
        _client = LLMClient()
    return _client


def reset_llm_client():
    """Call after runtime key changes from the web UI."""
    global _client
    _client = None
    # D4: clear the embedding latch too (stale dead-flag survived key rotation).
    try:
        reset_embedding_health()
    except Exception:
        pass


def call_llm(
    prompt: str,
    client: Any = None,
    temperature: float = 0.7,
    model: Optional[str] = None,
    tier: str = "default",
    system: Optional[str] = None,
    max_tokens: int = 8192,
    allow_empty: bool = True,
    retry_on_truncation: bool = True,
    media_parts: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Primary LLM entry point used by all agents (D1 strict via allow_empty=False).

    retry_on_truncation=False is json_mode: strict-JSON callers get the partial
    payload immediately for self-repair instead of a second ~300s full re-run.
    """
    from .pricing import load_price_table, tokens_of, usd_for

    llm = client if isinstance(client, LLMClient) else get_llm_client()
    model_id = model or config.resolve_model(tier)
    try:
        load_price_table(getattr(config, "llm_price_table_json", "") or "")
    except Exception:
        pass
    # D3: real token counting (provider tokenizer when available).
    input_tokens_est, _input_est = tokens_of((system or "") + prompt)
    try:
        result = llm.chat(
            prompt,
            temperature=temperature,
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            allow_empty=allow_empty,
            retry_on_truncation=retry_on_truncation,
            media_parts=media_parts,
        )
    except Exception:
        # P0-1: count the attempt even when chat raises (truncation, transport,
        # model-404). Previously llm_calls stayed flat while llm_failures grew,
        # breaking is_llm_outage() and hiding outage scale in run stats.
        try:
            from .run_log import get_tracker as _get_tracker2

            _t = _get_tracker2()
            if _t:
                _t.bump("llm_calls")
                _t.bump("llm_tokens_in", amount=input_tokens_est)
        except Exception:
            pass
        raise
    output_tokens_est, _out_est = tokens_of(result or "")
    usd, _usd_est = usd_for(model_id, input_tokens_est, output_tokens_est)
    try:
        from .run_log import get_tracker

        tracker = get_tracker()
        if tracker:
            tracker.bump("llm_calls")
            tracker.bump("llm_tokens_in", amount=input_tokens_est)
            tracker.bump("llm_tokens_out", amount=output_tokens_est)
            # D3: write cost_estimate_usd on every bump (float via stats dict).
            try:
                tracker.stats["cost_estimate_usd"] = float(tracker.stats.get("cost_estimate_usd") or 0.0) + float(usd)
            except Exception:
                pass
    except Exception as tracker_err:
        logger.debug("Failed to record LLM stats: %s", tracker_err)
    return result


def generate_embedding(text: str, model: Any = None) -> np.ndarray:
    """Generate an embedding via gemini-embedding-2 (GEMINI-ONLY POLICY).

    Every agent in this project (topic_hunter, engineer, planner, writer,
    editor, utils.similarity) funnels through here, so the chat provider
    in `core.config` can never change the embedding space. `model` is a
    legacy alias for a prebuilt LLMClient, not a model id.
    """
    llm = model if isinstance(model, LLMClient) else get_llm_client()
    return llm.embed(text)


# --- Backward-compatible aliases (old agents import these) ---

def setup_gemini():
    """Legacy: returns LLMClient (works with call_gemini / generate_embedding)."""
    return get_llm_client()


def call_gemini(prompt: str, model: Any = None, temperature: float = 0.7) -> str:
    """Legacy alias for call_llm."""
    return call_llm(prompt, client=model, temperature=temperature)
