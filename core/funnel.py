"""Run-funnel counters (P0): where runs die before engineering.

Pure helpers over ResearchState dicts — no behavior change, only measurement
so deferred-routing / wall-removal gains are visible.
"""

from __future__ import annotations

from typing import Any, Dict

FUNNEL_STAGES: tuple[str, ...] = (
    "topic_discovery",
    "hypothesis_debate",
    "planning",
    "data_validation",
    "engineering",
    "writing_results",
)


def funnel_stage(state: Dict[str, Any]) -> str:
    """Deepest stage reached: current_phase unless terminal (then last ok phase)."""
    try:
        phase = str(state.get("current_phase") or "")
        if phase and phase != "complete":
            return phase
        # Terminal: infer from artifacts present.
        if state.get("plan"):
            if state.get("engineer_outputs"):
                return "engineering"
            return "planning"
        if state.get("selected_topic"):
            return "hypothesis_debate"
        if state.get("topics"):
            return "topic_discovery"
        return "intake"
    except Exception:
        return "unknown"


def funnel_summary(states: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Count states per deepest stage + terminal reasons."""
    counts: Dict[str, int] = {}
    terminals: Dict[str, int] = {}
    for s in states or []:
        try:
            st = funnel_stage(s)
            counts[st] = counts.get(st, 0) + 1
            term = str((s or {}).get("terminal_error") or "")
            if term:
                key = term.split(":")[0][:60]
                terminals[key] = terminals.get(key, 0) + 1
        except Exception:
            continue
    reached_eng = sum(v for k, v in counts.items() if k in ("engineering", "writing_results", "supervision", "editing"))
    return {
        "counts": counts,
        "terminals": terminals,
        "total": len(states or []),
        "reached_engineering": reached_eng,
    }
