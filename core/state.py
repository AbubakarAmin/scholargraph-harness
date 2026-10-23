"""Shared state contract for the ScholarGraph workflow."""

from typing import Any, Dict, List, Optional, TypedDict

from .contracts import DatasetArtifact, ExecutionArtifact, ExperimentContract, ExperimentOutput, LiteratureContext, Plan, StatisticalReport, Topic, VerificationReport


class ResearchState(TypedDict):
    """Mutable state passed between LangGraph workflow nodes."""

    iteration: int
    current_phase: str
    should_reset: bool
    should_continue: bool
    error_count: int
    topics: List[Topic]
    selected_topic: Optional[Topic]
    debate_results: List[Dict[str, Any]]
    hypothesis_passed: bool
    plan: Optional[Plan]
    data_artifacts: Dict[str, DatasetArtifact]
    data_validation: VerificationReport
    execution_artifacts: Dict[str, ExecutionArtifact]
    analysis_reports: Dict[str, StatisticalReport]
    verification_findings: List[Dict[str, Any]]
    draft_sections: Dict[str, str]
    current_section: Optional[str]
    engineer_outputs: Dict[str, ExperimentOutput]
    supervisor_scores: Dict[str, float]
    supervisor_feedback: Dict[str, str]
    meta_feedback: List[str]
    final_paper: Optional[Dict[str, Any]]
    latex_output: Optional[str]
    plan_revision_requests: List[Dict[str, Any]]
    run_id: Optional[str]
    results_redraft_count: int
    narrative_revision_count: int
    editor_repair_count: int
    editor_repair_findings: Optional[str]
    results_verification: Dict[str, VerificationReport]
    reproducibility: VerificationReport
    terminal_error: Optional[str]
    experiment_contracts: Dict[str, ExperimentContract]
    experiment_outcomes: Dict[str, str]
    technical_failures: Dict[str, Dict[str, Any]]
    evidence_gate: Dict[str, Any]
    human_approved: bool
    outcome_calibration: Dict[str, Any]
    literature_context: Optional[LiteratureContext]
    user_query: Optional[str]
    # --- State v2 additions (proposal §3.1, all additive) ---
    paper_type: str
    pipeline_recipe: List[str]
    evidence_requirements: Dict[str, bool]
    corpus: Optional[Dict[str, Any]]
    instrument: Optional[Dict[str, Any]]
    causal_claim: Optional[Dict[str, Any]]
    thesis: Optional[Dict[str, Any]]
    skipped_phases: List[Dict[str, str]]
    intake_report: Optional[Dict[str, Any]]
    readiness_level: str
    checkpoint_schema_version: int
    # --- Deferred routing additions (P1): intake no longer locks paper_type.
    # paper_type may be "auto-deferred" until post-evidence routing resolves it.
    # paper_candidates ranks feasible types; selected_artifacts supports
    # multi-paper emission from a single hunt.
    paper_candidates: List[Dict[str, Any]]
    selected_artifacts: List[Dict[str, Any]]
    hunt_mode: str
    routing_provenance: Optional[Dict[str, Any]]
    # --- M6 step accounting (proposal §11-M6 step 2 + §8.5) ---
    # Per-phase execution counts, incremented once per executed phase node.
    # Checked against the recursion budget; additive-only (default {}).
    phase_steps: Dict[str, int]
    # --- Lab society bus (agentic upgrade, additive-only) ---
    # messages: inter-agent dialogue (help_request, handoff, standup...).
    # claims: falsifiable claims ledger with evidence ids + status.
    # tool_ledger_ids: obs_ids from broker tool calls for citation grounding.
    messages: List[Dict[str, Any]]
    claims: List[Dict[str, Any]]
    tool_ledger_ids: List[str]
    # --- Agentic harness v2 (additive-only) ---
    # work_plans: per-phase 2-3 step work plans (opencode todowrite equivalent).
    # prompt_version: pinned prompt registry version for reproducibility.
    # tool_calls: audited broker tool-call records for this run.
    # provenance: evidence honesty latch (clean/exploratory/synthetic/fallback).
    work_plans: List[Dict[str, Any]]
    prompt_version: Dict[str, str]
    tool_calls: List[Dict[str, Any]]
    provenance: str


STATE_V2_DEFAULTS: Dict[str, Any] = {
    "paper_type": "experimental",
    "pipeline_recipe": [],
    "evidence_requirements": {},
    "corpus": None,
    "instrument": None,
    "causal_claim": None,
    "thesis": None,
    "skipped_phases": [],
    "intake_report": None,
    "readiness_level": "L1",
    "checkpoint_schema_version": 2,
    "phase_steps": {},
    "paper_candidates": [],
    "selected_artifacts": [],
    "hunt_mode": "try_and_report",
    "routing_provenance": None,
    "messages": [],
    "claims": [],
    "tool_ledger_ids": [],
    "work_plans": [],
    "prompt_version": {},
    "tool_calls": [],
    "provenance": "clean",
}

# Evidence honesty latch: any non-clean provenance forces publishable=False
# and a scoping banner, and can never pass the release referee.
PROVENANCE_ORDER = ("clean", "exploratory", "synthetic", "fallback")


def downgrade_provenance(current: str, candidate: str) -> str:
    """Worst-wins provenance merge (clean < exploratory < synthetic < fallback)."""
    order = list(PROVENANCE_ORDER)
    try:
        cur_i = order.index(str(current or "clean"))
    except ValueError:
        cur_i = 0
    try:
        cand_i = order.index(str(candidate or "clean"))
    except ValueError:
        return str(current or "clean")
    return order[max(cur_i, cand_i)]

# Ordered experimental spine recipe used to backfill v1 checkpoints (§3.3).
EXPERIMENTAL_SPINE_RECIPE: List[str] = [
    "intake",
    "topic_discovery",
    "hypothesis_debate",
    "planning",
    "data_validation",
    "writing_narrative",
    "engineering",
    "independent_validation",
    "writing_results",
    "supervision",
    "editing",
]


def migrate_state_v1_to_v2(state: Dict[str, Any]) -> Dict[str, Any]:
    """Backfill v1 checkpoints with every v2 field + experimental spine (M0).

    Additive-only: never drops keys, never rewrites existing v2 values.
    P1: v1 states with no explicit type stay deferred (not forced experimental)
    so resumed runs get post-evidence routing instead of the legacy default.
    """
    migrated = dict(state)
    _had_type = state.get("paper_type") if isinstance(state, dict) else None
    for key, default in STATE_V2_DEFAULTS.items():
        if key not in migrated:
            migrated[key] = default() if callable(default) else (dict(default) if isinstance(default, dict) else (list(default) if isinstance(default, list) else default))
    if not migrated.get("pipeline_recipe"):
        # Deferred when the type was never explicitly chosen; explicit legacy
        # experimental runs keep their spine.
        if _had_type in (None, "", "auto", "auto-deferred"):
            migrated["paper_type"] = "auto-deferred"
            migrated["pipeline_recipe"] = []
        else:
            migrated["pipeline_recipe"] = list(EXPERIMENTAL_SPINE_RECIPE)
    if not migrated.get("paper_type"):
        migrated["paper_type"] = "auto-deferred"
    # Lab bus backfill (additive-only): old checkpoints get empty bus.
    for _k, _d in (("messages", []), ("claims", []), ("tool_ledger_ids", [])):
        if _k not in migrated:
            migrated[_k] = list(_d)
    migrated["checkpoint_schema_version"] = 2
    # Agentic v2 backfill (additive-only, stays schema v2): work_plans/prompt/tool/provenance.
    for _k, _d in (("work_plans", []), ("prompt_version", {}),
                   ("tool_calls", [])):
        if _k not in migrated:
            migrated[_k] = dict(_d) if isinstance(_d, dict) else list(_d)
    if not migrated.get("provenance"):
        migrated["provenance"] = "clean"
    return migrated


def initialize_state() -> ResearchState:
    """Return a fresh state for a new research run."""
    return ResearchState(
        iteration=0,
        current_phase="topic_discovery",
        should_reset=False,
        should_continue=True,
        error_count=0,
        topics=[],
        selected_topic=None,
        debate_results=[],
        hypothesis_passed=False,
        plan=None,
        data_artifacts={},
        data_validation={},
        execution_artifacts={},
        analysis_reports={},
        verification_findings=[],
        draft_sections={},
        current_section=None,
        engineer_outputs={},
        supervisor_scores={},
        supervisor_feedback={},
        meta_feedback=[],
        final_paper=None,
        latex_output=None,
        plan_revision_requests=[],
        run_id=None,
        results_redraft_count=0,
        narrative_revision_count=0,
        editor_repair_count=0,
        editor_repair_findings=None,
        results_verification={},
        reproducibility={},
        terminal_error=None,
        experiment_contracts={},
        experiment_outcomes={},
        technical_failures={},
        evidence_gate={},
        human_approved=False,
        outcome_calibration={},
        literature_context=None,
        user_query=None,
        paper_type="auto-deferred",
        pipeline_recipe=[],
        evidence_requirements={},
        corpus=None,
        instrument=None,
        causal_claim=None,
        thesis=None,
        skipped_phases=[],
        intake_report=None,
        readiness_level="L1",
        checkpoint_schema_version=2,
        phase_steps={},
        paper_candidates=[],
        selected_artifacts=[],
        hunt_mode="try_and_report",
        routing_provenance=None,
        messages=[],
        claims=[],
        tool_ledger_ids=[],
        work_plans=[],
        prompt_version={},
        tool_calls=[],
        provenance="clean",
    )