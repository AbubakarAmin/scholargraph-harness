"""Phase registry: paper types, pipelines, pure router (proposal §3.2–§3.3).

13 kept phases = §2.1 graph node names.
6 new phases: intake, systematic_review, causal_audit, survey_analysis,
case_protocol, thesis_argument.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

PaperType = Literal[
    "experimental",
    "comparative",
    "empirical",
    "review",
    "analytical",
    "descriptive",
    "argumentative",
    "cause_effect",
    "case_study",
    "survey",
]

ALL_PAPER_TYPES: tuple[str, ...] = (
    "experimental",
    "comparative",
    "empirical",
    "review",
    "analytical",
    "descriptive",
    "argumentative",
    "cause_effect",
    "case_study",
    "survey",
)

# Refusal taxonomy (exact literals, §1.5).
REFUSALS: tuple[str, ...] = (
    "needs_primary_data",
    "needs_wetlab",
    "needs_closed_data",
    "needs_gpu_scale",
    "needs_response_table",
    "needs_case_pack",
    "needs_pack_invalid",
    "budget_exhausted",
    "infra_llm_unavailable",
    "infra_sources_unavailable",
    "infra_sandbox_unavailable",
)

# 13 kept + 6 new = 19 phases. `node` is the LangGraph node name;
# new non-graph phases map to their agent node when wired (M1–M4).
PHASES: dict[str, dict[str, Any]] = {
    # --- kept (§2.1) ---
    "topic_discovery": {"name": "topic_discovery", "node": "topic_discovery", "requires": [], "produces": ["topics", "selected_topic"], "optional": False, "cost": "cheap"},
    "hypothesis_debate": {"name": "hypothesis_debate", "node": "hypothesis_debate", "requires": ["selected_topic"], "produces": ["debate_results"], "optional": False, "cost": "strong"},
    "planning": {"name": "planning", "node": "planning", "requires": ["debate_results"], "produces": ["plan"], "optional": False, "cost": "strong"},
    "terminal_planning_failure": {"name": "terminal_planning_failure", "node": "terminal_planning_failure", "requires": [], "produces": ["terminal_error"], "optional": True, "cost": "cheap"},
    "data_validation": {"name": "data_validation", "node": "data_validation", "requires": ["plan"], "produces": ["data_validation"], "optional": False, "cost": "cheap"},
    "writing_narrative": {"name": "writing_narrative", "node": "writing_narrative", "requires": ["plan"], "produces": ["draft_sections"], "optional": False, "cost": "strong"},
    "engineering": {"name": "engineering", "node": "engineering", "requires": ["plan"], "produces": ["engineer_outputs"], "optional": False, "cost": "compute"},
    "independent_validation": {"name": "independent_validation", "node": "independent_validation", "requires": ["engineer_outputs"], "produces": ["analysis_reports"], "optional": False, "cost": "compute"},
    "writing_results": {"name": "writing_results", "node": "writing_results", "requires": ["engineer_outputs"], "produces": ["draft_sections"], "optional": False, "cost": "strong"},
    "supervision": {"name": "supervision", "node": "supervision", "requires": ["draft_sections"], "produces": ["supervisor_scores"], "optional": False, "cost": "judge"},
    "meta_evaluation": {"name": "meta_evaluation", "node": "meta_evaluation", "requires": ["supervisor_scores"], "produces": ["meta_feedback"], "optional": True, "cost": "judge"},
    "editing": {"name": "editing", "node": "editing", "requires": ["draft_sections"], "produces": ["final_paper", "latex_output"], "optional": False, "cost": "strong"},
    "reset": {"name": "reset", "node": "reset", "requires": [], "produces": [], "optional": True, "cost": "cheap"},
    # --- new (§3.2) ---
    "intake": {"name": "intake", "node": "intake", "requires": [], "produces": ["intake_report", "pipeline_recipe"], "optional": False, "cost": "cheap"},
    "systematic_review": {"name": "systematic_review", "node": "systematic_review", "requires": ["intake_report"], "produces": ["corpus"], "optional": False, "cost": "strong"},
    "causal_audit": {"name": "causal_audit", "node": "causal_audit", "requires": ["engineer_outputs"], "produces": ["causal_claim"], "optional": False, "cost": "strong"},
    "survey_analysis": {"name": "survey_analysis", "node": "survey_analysis", "requires": ["intake_report"], "produces": ["analysis_reports"], "optional": False, "cost": "cheap"},
    "case_protocol": {"name": "case_protocol", "node": "case_protocol", "requires": ["intake_report"], "produces": ["analysis_reports"], "optional": False, "cost": "cheap"},
    "thesis_argument": {"name": "thesis_argument", "node": "thesis_argument", "requires": ["intake_report"], "produces": ["thesis"], "optional": False, "cost": "strong"},
}

# Exact ordered recipes (§3.3).
PIPELINES: dict[str, list[str]] = {
    "experimental": ["intake", "topic_discovery", "hypothesis_debate", "planning", "data_validation", "writing_narrative", "engineering", "independent_validation", "writing_results", "supervision", "editing"],
    "comparative": ["intake", "topic_discovery", "hypothesis_debate", "planning", "data_validation", "writing_narrative", "engineering", "independent_validation", "writing_results", "supervision", "editing"],
    "empirical": ["intake", "topic_discovery", "planning", "data_validation", "engineering", "writing_results", "supervision", "editing"],
    "review": ["intake", "topic_discovery", "systematic_review", "thesis_argument", "writing_narrative", "writing_results", "supervision", "editing"],
    "analytical": ["intake", "topic_discovery", "thesis_argument", "writing_narrative", "writing_results", "supervision", "editing"],
    "descriptive": ["intake", "topic_discovery", "data_validation", "survey_analysis", "writing_narrative", "writing_results", "supervision", "editing"],
    "argumentative": ["intake", "topic_discovery", "thesis_argument", "hypothesis_debate", "writing_narrative", "writing_results", "supervision", "editing"],
    "cause_effect": ["intake", "topic_discovery", "hypothesis_debate", "planning", "engineering", "causal_audit", "independent_validation", "writing_results", "supervision", "editing"],
    "case_study": ["intake", "topic_discovery", "case_protocol", "writing_narrative", "writing_results", "supervision", "editing"],
    "survey": ["intake", "topic_discovery", "data_validation", "survey_analysis", "writing_results", "supervision", "editing"],
}

# Per-type evidence requirements (§3.1 evidence_requirements keys).
EVIDENCE_REQUIREMENTS: dict[str, dict[str, bool]] = {
    "experimental": {"needs_experiment": True, "needs_prisma": False, "needs_pack": False, "needs_audit": False, "needs_thesis": False},
    "comparative": {"needs_experiment": True, "needs_prisma": False, "needs_pack": False, "needs_audit": False, "needs_thesis": False},
    "empirical": {"needs_experiment": False, "needs_prisma": False, "needs_pack": True, "needs_audit": False, "needs_thesis": False},
    "review": {"needs_experiment": False, "needs_prisma": True, "needs_pack": False, "needs_audit": False, "needs_thesis": True},
    "analytical": {"needs_experiment": False, "needs_prisma": False, "needs_pack": False, "needs_audit": False, "needs_thesis": True},
    "descriptive": {"needs_experiment": False, "needs_prisma": False, "needs_pack": False, "needs_audit": False, "needs_thesis": True},
    "argumentative": {"needs_experiment": False, "needs_prisma": False, "needs_pack": False, "needs_audit": False, "needs_thesis": True},
    "cause_effect": {"needs_experiment": True, "needs_prisma": False, "needs_pack": False, "needs_audit": True, "needs_thesis": False},
    "case_study": {"needs_experiment": False, "needs_prisma": False, "needs_pack": True, "needs_audit": False, "needs_thesis": False},
    "survey": {"needs_experiment": False, "needs_prisma": False, "needs_pack": True, "needs_audit": False, "needs_thesis": False},
}


def make_refusal(
    reason: str,
    provided: str = "",
    alternative: str = "",
    how_to_proceed: str = "",
) -> dict[str, str]:
    """Build a refusal record with taxonomy literal (§1.5)."""
    if reason not in REFUSALS:
        raise ValueError(f"Unknown refusal reason: {reason!r}. Must be one of {REFUSALS}")
    return {
        "reason": reason,
        "provided": provided,
        "alternative": alternative,
        "how_to_proceed": how_to_proceed,
    }


def validate_pack(paper_type: str, pack: dict[str, Any] | None) -> tuple[bool, dict[str, str] | None]:
    """Validate required packs per type. Returns (ok, refusal_or_None).

    - case_study requires a case pack manifest.
    - survey requires a response-table manifest.
    Bad packs map to named reasons (never tracebacks).
    """
    pack = pack or {}
    if paper_type == "case_study":
        manifest = pack.get("case_pack") or pack.get("pack_manifest")
        if not manifest:
            return (False, make_refusal(
                "needs_case_pack",
                provided="no case pack supplied",
                alternative="pick empirical or analytical",
                how_to_proceed="upload case_pack.zip (>=2 source types) or pick another type",
            ))
        entries = manifest.get("entries", manifest) if isinstance(manifest, dict) else manifest
        types = set()
        if isinstance(entries, list):
            for e in entries:
                if isinstance(e, dict) and e.get("type"):
                    types.add(e["type"])
        elif isinstance(manifest, dict) and manifest.get("files"):
            for f in manifest["files"]:
                if isinstance(f, dict) and f.get("type"):
                    types.add(f["type"])
        if len(types) < 2:
            return (False, make_refusal(
                "needs_pack_invalid",
                provided=f"case pack has {len(types)} source type(s), need >=2",
                alternative="add a second source type",
                how_to_proceed="include >=2 of doc/transcript/table/audio_note",
            ))
    if paper_type == "survey":
        manifest = pack.get("response_table") or pack.get("pack_manifest")
        if not manifest:
            return (False, make_refusal(
                "needs_response_table",
                provided="no response table supplied",
                alternative="pick empirical-secondary with public data",
                how_to_proceed="upload response_table.csv/json + codebook.csv or pick another type",
            ))
    return (True, None)


# Hunt modes for topic discovery (P2): gap = legacy world-gap hunt,
# try_and_report = simple try-approach-and-report, replication = re-test prior work.
HUNT_MODES: tuple[str, ...] = ("gap", "try_and_report", "replication")

# Deferred binding marker: intake leaves paper_type unresolved until evidence arrives.
DEFERRED_TYPE: str = "auto-deferred"

# Single dedup/threshold config (P4: was scattered across topic_hunter).
DEDUP_CONFIG: dict[str, float] = {
    "excluded_title_overlap": 0.55,
    "topic_dedup_overlap": 0.70,
    "seed_fuzzy_overlap": 0.60,
    "theme_family_overlap": 0.80,
    "theme_family_count": 5.0,
    "near_duplicate_sim": 0.95,
}


def infer_candidates(pack: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Rank feasible paper types from pack evidence only (no literature/data yet).

    Returns a ranked candidate list, never a single collapsed ID. The final
    binding happens in post-evidence routing (core/workflow_nodes.py), not here.
    Pack-strong signals rank first; experimental is always a fallback candidate,
    never the forced answer.
    """
    pack = pack or {}
    candidates: list[dict[str, Any]] = []
    if pack.get("case_pack"):
        candidates.append({"type": "case_study", "confidence": 0.9, "why": "case_pack present"})
    if pack.get("response_table"):
        candidates.append({"type": "survey", "confidence": 0.9, "why": "response_table present"})
    if pack.get("corpus_override") or pack.get("protocol"):
        candidates.append({"type": "review", "confidence": 0.8, "why": "corpus/protocol present"})
    if pack.get("causal_claim") or pack.get("dag"):
        candidates.append({"type": "cause_effect", "confidence": 0.8, "why": "causal claim/dag present"})
    # Generic evidence-free hunt can support multiple outcomes; list them all
    # with honest low confidence instead of collapsing to experimental.
    for t, why in (
        ("experimental", "default executable fallback"),
        ("empirical", "public-data reanalysis possible"),
        ("analytical", "thesis synthesis possible"),
        ("descriptive", "characterization possible"),
    ):
        if t not in {c["type"] for c in candidates}:
            candidates.append({"type": t, "confidence": 0.4, "why": why})
    return candidates


def _auto_route(pack: dict[str, Any] | None) -> str:
    """Deterministic auto tiebreak (pure, no LLM). Logged by caller.

    Kept for backward compat; new code should use infer_candidates + deferred
    routing. Still returns a single ID for legacy callers.
    """
    pack = pack or {}
    if pack.get("case_pack"):
        return "case_study"
    if pack.get("response_table"):
        return "survey"
    if pack.get("corpus_override") or pack.get("protocol"):
        return "review"
    if pack.get("causal_claim") or pack.get("dag"):
        return "cause_effect"
    return "experimental"


def resolve_recipe(
    paper_type: str,
    pack: dict[str, Any] | None = None,
    llm_tiebreak_fn: Any = None,
) -> tuple[list[str], list[dict[str, str]], dict[str, bool], dict[str, Any] | None]:
    """Resolve (recipe, skipped_phases, evidence_requirements, pack_manifest).

    Pure function except one cheap Auto-tiebreak call when paper_type ==
    "auto" and llm_tiebreak_fn is supplied (logged by caller per §3.2).
    Raises ValueError on unknown type; returns refusal info via skipped
    entries only for non-run phases (pack failures surface via
    validate_pack, not exceptions).
    """
    requested = paper_type
    if requested == "auto":
        routed: str | None = None
        if llm_tiebreak_fn is not None:
            try:
                routed = llm_tiebreak_fn(pack or {})
            except Exception as e:
                logger.warning("Auto tiebreak LLM call failed, deterministic fallback: %s", e, exc_info=True)
                routed = None
        if not routed or routed not in PIPELINES:
            routed = _auto_route(pack)
            logger.info("type_router auto-tiebreak: requested=auto routed=%s (deterministic)", routed)
        else:
            logger.info("type_router auto-tiebreak: requested=auto routed=%s (llm)", routed)
        paper_type = routed
    if paper_type not in PIPELINES:
        raise ValueError(f"Unknown paper_type: {paper_type!r}. Must be one of {sorted(PIPELINES)}")
    recipe = list(PIPELINES[paper_type])
    recipe_set = set(recipe)
    skipped = [
        {"phase": name, "reason": f"not in {paper_type} recipe"}
        for name in PHASES
        if name not in recipe_set and name not in ("reset", "terminal_planning_failure")
    ]
    evidence = dict(EVIDENCE_REQUIREMENTS[paper_type])
    manifest = (pack or {}).get("pack_manifest")
    return (recipe, skipped, evidence, manifest)


def is_phase_required(paper_type: str, phase: str) -> bool:
    """True when phase is in the paper type's recipe."""
    if paper_type == "auto":
        raise ValueError("paper_type must never be 'auto' after intake (§1.3)")
    if paper_type not in PIPELINES:
        raise ValueError(f"Unknown paper_type: {paper_type!r}")
    return phase in PIPELINES[paper_type]


# --- M6 hardening (§11-M6 steps 1–2) -------------------------------------
# Parallel fan-out is OPT-IN ONLY behind PARALLEL_PHASES_ENABLED (default
# false, §9). The helper below returns the *plan* (which phase pairs may fan
# out); execution stays sequential — no ThreadPoolExecutor for external API
# calls per AGENTS inviolable rules. Soak = plan observed over runs while
# the graph still executes sequentially.

# Methods (written by writing_narrative) ∥ engineering (experiment code).
# Both read the plan and produce disjoint artifacts (draft_sections vs
# engineer_outputs), so they are the only safe fan-out candidate.
PARALLEL_FANOUT_PAIRS: tuple[tuple[str, str], ...] = (
    ("writing_narrative", "engineering"),
)


def parallel_plan_for_recipe(
    recipe: list[str] | None,
    enabled: bool | None = None,
) -> list[tuple[str, str]]:
    """Return fan-out pairs for a recipe (M6 step 1).

    Returns [] unless parallelism is explicitly enabled — via the `enabled`
    override or `config.parallel_phases_enabled` (default False, §9), so
    flag-off behavior is sequential and unchanged. A pair is returned only
    when BOTH phases are present in the recipe.
    """
    if enabled is None:
        try:
            from .config import config as _cfg

            enabled = bool(getattr(_cfg, "parallel_phases_enabled", False))
        except Exception:
            enabled = False
    if not enabled:
        return []
    names = set(recipe or [])
    return [pair for pair in PARALLEL_FANOUT_PAIRS if pair[0] in names and pair[1] in names]


def record_phase_step(state: dict[str, Any], phase: str) -> dict[str, int]:
    """Increment the per-phase step counter (M6 step 2, §8.5).

    Stores counts under state["phase_steps"]. Mutates `state` in place and
    returns the counter dict. Never raises on odd input (fail-safe: returns
    a best-effort dict).
    """
    try:
        steps = dict(state.get("phase_steps") or {})
    except Exception:
        steps = {}
    key = str(phase)
    try:
        steps[key] = int(steps.get(key, 0) or 0) + 1
    except Exception:
        steps[key] = 1
    try:
        state["phase_steps"] = steps
    except Exception:
        pass
    return steps


def total_phase_steps(state: dict[str, Any]) -> int:
    """Total executed phases recorded in state["phase_steps"]."""
    try:
        steps = state.get("phase_steps") or {}
        return sum(int(v or 0) for v in steps.values())
    except Exception:
        return 0


def check_step_budget(
    state: dict[str, Any],
    *,
    recursion_limit: int = 1000,
) -> tuple[bool, dict[str, str] | None]:
    """Check executed-phase count against the recursion budget (M6 step 2).

    Trips when total executed phases >= `recursion_limit` (LangGraph stream
    cap, §2.2/§6.2.5; MAX_ITERATIONS stays the meta-loop guard it already
    is — both numbers are cited in the refusal). Returns
    (tripped, refusal_or_None); the refusal uses the locked `budget_exhausted`
    taxonomy literal (§1.5) with reason/provided/alternative/how_to_proceed.
    Pure: never mutates state, never emits.
    """
    try:
        limit = int(recursion_limit)
    except Exception:
        limit = 1000
    total = total_phase_steps(state)
    if total < limit:
        return (False, None)
    try:
        from .config import config as _cfg

        max_iterations = int(getattr(_cfg, "max_iterations", 10))
    except Exception:
        max_iterations = 10
    return (True, make_refusal(
        "budget_exhausted",
        provided=f"{total} executed phases vs recursion budget {limit} (MAX_ITERATIONS={max_iterations})",
        alternative="narrower recipe or higher recursion budget",
        how_to_proceed=f"resume with a higher recursion_limit, or reduce recipe/reset loops; partial bundle retained",
    ))


def apply_step_budget(
    state: dict[str, Any],
    *,
    recursion_limit: int = 1000,
    emit: bool = True,
) -> bool:
    """Stamp a budget_exhausted terminal when the step budget is exceeded.

    On trip (and no pre-existing terminal_error): sets
    terminal_error/current_phase/should_continue and emits a `budget_check`
    event through the existing run_log transport (§10.4; no new transport).
    Returns True when tripped. Never raises — accounting must not crash runs.
    """
    try:
        tripped, refusal = check_step_budget(state, recursion_limit=recursion_limit)
    except Exception:
        return False
    if not tripped or refusal is None:
        return False
    try:
        if state.get("terminal_error"):
            return True  # already terminal; still report tripped
        total = total_phase_steps(state)
        state["terminal_error"] = f"budget_exhausted: {refusal['provided']}"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        try:
            state.setdefault("evidence_gate", {})
            if isinstance(state.get("evidence_gate"), dict):
                state["evidence_gate"]["reason_code"] = "budget_exhausted"
        except Exception:
            pass
    except Exception:
        return True
    if emit:
        try:
            from .run_log import emit_event as _emit

            _emit(
                "budget_check",
                {"phase": "run", "agent": "StepGuard", "status": "blocked",
                 "message_human": f"step budget exceeded: {refusal['provided']}",
                 "threshold": recursion_limit, "actual": total,
                 "refusal": refusal, "cause_class": "budget"},
                run_id=state.get("run_id"),
                agent="StepGuard",
            )
        except Exception:
            pass
    try:
        import logging as _logging

        _logging.getLogger(__name__).error(
            "Step budget exceeded (%s); terminal budget_exhausted", refusal["provided"]
        )
    except Exception:
        pass
    return True
