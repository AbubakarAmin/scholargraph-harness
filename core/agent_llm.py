"""Spec-driven LLM entry point for agents (agentic-harness upgrade).

Every agent call flows through here instead of bare ``call_llm(prompt)``:

- ``system``/``user`` split from the versioned registry (``core.prompts``).
- Tier / temperature / max_tokens pinned per prompt spec.
- Cheap-drafter -> strong-verifier cascade (``LLM_CASCADE_ENABLED``).
- JSON schema validation (``validate_schema``) with fail-closed fallback.
- Doom-loop detection (identical outputs ``repeat`` times -> stop).
- ``max_steps`` agentic cap per prompt spec.
- ``prompt_version`` stamped into the run state for reproducibility.

Fail-closed: transport/parse failures return ``None`` (JSON) or ``""``
(text) — callers keep their existing fallback ladders.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

CASCADE_UPGRADE = {"cheap": "strong", "strong": "judge"}


def resolve_tier(base_tier: str, attempt: int, cascade_at: int = 3) -> str:
    """Escalate tier after ``cascade_at`` attempts (FrugalGPT-style cascade)."""
    try:
        from core.config import config

        if not bool(getattr(config, "llm_cascade_enabled", True)):
            return base_tier
        threshold = int(getattr(config, "engineer_cascade_attempt", cascade_at)
                        or cascade_at)
    except Exception:
        threshold = cascade_at
    tier = str(base_tier or "default")
    if attempt >= threshold and tier in CASCADE_UPGRADE:
        return CASCADE_UPGRADE[tier]
    return tier


def agent_text(
    prompt_name: str,
    call_fn: Callable[..., str],
    state: Optional[Dict[str, Any]] = None,
    attempt: int = 0,
    _hashes: Optional[List[str]] = None,
    **slots: Any,
) -> str:
    """Render + call a raw-text prompt through the registry spec."""
    from core import prompts as _registry

    spec = _registry.get_prompt(prompt_name)
    system = _registry.render_system(prompt_name, **slots)
    user = _registry.render(prompt_name, **slots)
    tier = resolve_tier(str(spec.get("tier") or "default"), attempt)
    temperature = float(spec.get("temperature", 0.3))
    max_tokens = int(spec.get("max_tokens", 8192) or 8192)
    _stamp_version(state, prompt_name, spec)
    try:
        return call_fn(user, temperature=temperature, tier=tier,
                       system=system, max_tokens=max_tokens) or ""
    except TypeError:
        # Injectable stubs without system/max_tokens support.
        try:
            return call_fn(user, temperature=temperature, tier=tier) or ""
        except TypeError:
            return call_fn(user) or ""
    except Exception as exc:
        logger.debug("agent_text %s failed: %s", prompt_name, exc)
        return ""


def agent_json(
    prompt_name: str,
    call_fn: Optional[Callable[..., Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    attempts: Optional[int] = None,
    _hashes: Optional[List[str]] = None,
    **slots: Any,
) -> Optional[Any]:
    """Render + call a JSON prompt with self-correction, schema check,
    doom-loop detection, and max_steps cap."""
    from core import prompts as _registry
    from core.utils import call_llm_json

    spec = _registry.get_prompt(prompt_name)
    system = _registry.render_system(prompt_name, **slots)
    user = _registry.render(prompt_name, **slots)
    max_steps = int(spec.get("max_steps", 2) or 2)
    budget = max(1, min(int(attempts or 3), max_steps + 1))
    temperature = float(spec.get("temperature", 0.3))
    max_tokens = int(spec.get("max_tokens", 8192) or 8192)
    _stamp_version(state, prompt_name, spec)
    hashes: List[str] = list(_hashes or [])
    attempt_no = [0]

    def _call(**kwargs: Any) -> Any:
        tier = resolve_tier(str(spec.get("tier") or "default"),
                            attempt_no[0])
        attempt_no[0] += 1
        fn = call_fn or _default_call_fn()
        try:
            return fn(kwargs.get("prompt", user),
                      temperature=kwargs.get("temperature", temperature),
                      tier=tier, system=system, max_tokens=max_tokens)
        except TypeError:
            try:
                return fn(kwargs.get("prompt", user),
                          temperature=kwargs.get("temperature", temperature),
                          tier=tier)
            except TypeError:
                return fn(kwargs.get("prompt", user))

    last: Optional[Any] = None
    for _ in range(budget):
        try:
            last = call_llm_json(user, attempts=1, temperature=temperature,
                                 tier=str(spec.get("tier") or "default"),
                                 system=system, max_tokens=max_tokens,
                                 call_fn=_call)
        except TypeError:
            # Older call_llm_json without system support.
            last = call_llm_json(user, attempts=1, temperature=temperature,
                                 tier=str(spec.get("tier") or "default"),
                                 max_tokens=max_tokens, call_fn=call_fn)
        if last is None:
            continue
        try:
            missing = _registry.validate_schema(prompt_name, last)
        except Exception:
            missing = []
        if missing:
            logger.debug("agent_json %s schema miss: %s", prompt_name, missing)
            user = (user + "\nSchema repair: response is missing keys "
                    f"{missing}. Return ONLY the valid JSON object now.")
            continue
        try:
            h = _registry.hash_output(last)
        except Exception:
            h = ""
        if h:
            hashes.append(h)
            if _registry.check_doom_loop(hashes):
                logger.warning("agent_json %s doom-loop (identical x3), stopping",
                               prompt_name)
                return last
        return last
    return last


def _default_call_fn() -> Callable[..., Any]:
    from core.llm import call_llm

    return call_llm


def _stamp_version(state: Optional[Dict[str, Any]],
                   name: str, spec: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    try:
        versions = dict(state.get("prompt_version") or {})
        versions[name] = str(spec.get("version", "v1"))
        from core.prompts import skill_fingerprint, skills_for_prompt

        for skill_name in skills_for_prompt(name):
            fingerprint = skill_fingerprint(skill_name)
            if fingerprint:
                versions[f"skill:{skill_name}"] = fingerprint
        state["prompt_version"] = versions
    except Exception:
        pass
