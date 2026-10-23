"""Typed handoff contracts shared by agents and workflow state."""

from typing import Any, Dict, List, TypedDict


class ArtifactProvenance(TypedDict, total=False):
    """Traceability metadata shared by every research artifact."""

    artifact_id: str
    run_id: str
    producer: str
    artifact_type: str
    content_hash: str
    created_at: str
    source: str
    parent_artifacts: List[str]
    environment: Dict[str, Any]
    limitations: List[str]
    status: str


class ResearchQuestion(TypedDict, total=False):
    question_id: str
    question: str
    hypothesis: str
    falsifiable_prediction: str
    null_hypothesis: str
    primary_metric: str
    status: str


class DatasetSpec(TypedDict, total=False):
    name: str
    source: str
    version: str
    license: str
    task: str
    features: List[str]
    target: str
    split_policy: str
    leakage_controls: List[str]
    access_policy: str


class SourceArtifact(TypedDict, total=False):
    provenance: ArtifactProvenance
    source: str
    url: str
    retrieved_at: str
    status: str
    response_hash: str
    content: Dict[str, Any]
    warnings: List[str]


class DatasetArtifact(TypedDict, total=False):
    provenance: ArtifactProvenance
    spec: DatasetSpec
    location: str
    content_hash: str
    schema: Dict[str, Any]
    row_count: int
    validation: "VerificationReport"


class CodeArtifact(TypedDict, total=False):
    provenance: ArtifactProvenance
    experiment_name: str
    source_code: str
    execution_backend: str
    language: str
    dependencies: List[str]
    entrypoint: str
    validation: "VerificationReport"


class ExecutionRequest(TypedDict, total=False):
    experiment_name: str
    code_artifact_id: str
    dataset_artifact_ids: List[str]
    seeds: List[int]
    timeout_seconds: int
    resource_limits: Dict[str, Any]


class ExecutionArtifact(TypedDict, total=False):
    provenance: ArtifactProvenance
    request: ExecutionRequest
    raw_results_path: str
    environment: Dict[str, Any]
    seed_results: Dict[str, Any]
    status: str
    error: str
    content_hash: str


class AnalysisPlan(TypedDict, total=False):
    primary_metric: str
    secondary_metrics: List[str]
    statistical_test: str
    effect_size: str
    confidence_level: float
    multiple_comparison_policy: str
    stopping_rule: str


class StatisticalReport(TypedDict, total=False):
    provenance: ArtifactProvenance
    analysis_plan: AnalysisPlan
    metrics: Dict[str, Any]
    comparisons: List[Dict[str, Any]]
    warnings: List[str]
    passed: bool


class Claim(TypedDict, total=False):
    claim_id: str
    text: str
    claim_type: str
    evidence_artifact_ids: List[str]
    status: str
    limitations: List[str]


class EvidenceBundle(TypedDict, total=False):
    claim: Claim
    dataset_artifacts: List[DatasetArtifact]
    code_artifacts: List[CodeArtifact]
    execution_artifacts: List[ExecutionArtifact]
    statistical_reports: List[StatisticalReport]
    verification_findings: List[Dict[str, Any]]


class VerificationFinding(TypedDict, total=False):
    finding_id: str
    severity: str
    check: str
    message: str
    artifact_ids: List[str]
    blocking: bool
    status: str


class MemoryEntry(TypedDict, total=False):
    """Persisted memory metadata with an explicit prompt-retrieval policy."""

    id: int
    timestamp: str
    namespace: str
    content_class: str
    retrieval_eligible: bool
    run_id: str
    agent: str
    outcome_status: str
    signal: Dict[str, Any]
    content: Any


class AgentCapabilityManifest(TypedDict, total=False):
    agent: str
    role: str
    allowed_capabilities: List[str]
    forbidden_capabilities: List[str]
    input_artifact_types: List[str]
    output_artifact_types: List[str]
    can_mutate: List[str]


class Topic(TypedDict, total=False):
    title: str
    description: str
    rationale: str
    impact: str
    feasibility: float
    score: float
    rank: int
    keywords: List[str]
    anchor_paper: str
    dataset_plan: str


class PlanSection(TypedDict, total=False):
    name: str
    content_requirements: str
    key_points: List[str]
    expected_length: str
    dependencies: List[str]


class Contribution(TypedDict, total=False):
    claim: str
    falsifiable_prediction: str
    statistical_test: str
    components: List[str]


Dependency = TypedDict(
    "Dependency",
    {"from": str, "to": str, "type": str, "description": str},
    total=False,
)


class RevisionRequest(TypedDict, total=False):
    reason: str
    detail: str
    experiment: str


class Timeline(TypedDict, total=False):
    phases: List[Dict[str, Any]]
    total_duration: str


class ExperimentSpec(TypedDict, total=False):
    name: str
    type: str
    description: str
    methodology: str
    baselines: List[str]
    variants: List[Dict[str, Any]]
    alternatives: List[Dict[str, Any]]
    falsifiable_prediction: str
    statistical_test: str
    claimed_components: List[str]
    evaluation_metrics: List[str]
    dataset: Dict[str, Any]
    split_policy: str
    seeds: List[int]
    stopping_rule: str
    contract_hash: str


class Plan(TypedDict, total=False):
    title: str
    topic: str
    domain: str
    sections: List[PlanSection]
    research_questions: List[str]
    methodology: str
    contributions: List[Contribution]
    expected_contributions: List[str]
    experiments: List[ExperimentSpec]
    dependencies: List[Dependency]
    timeline: Timeline
    revision_history: List[Dict[str, Any]]


class ExperimentOutput(TypedDict, total=False):
    experiment_name: str
    success: bool
    code: str
    aggregate_metrics: Dict[str, Any]
    results: Dict[str, Any]
    validation: Dict[str, Any]
    ablation: Dict[str, Any]
    raw_results_path: str
    error: str
    timestamp: str
    contract_hash: str
    outcome: str
    failure_kind: str
    attempts: List[Dict[str, Any]]


class ExperimentContract(TypedDict, total=False):
    experiment_name: str
    hypothesis: str
    dataset: Dict[str, Any]
    requirements: Dict[str, Any]
    baselines: List[str]
    evaluation_metrics: List[str]
    split_policy: str
    seeds: List[int]
    stopping_rule: str
    analysis_protocol: Dict[str, Any]
    contract_hash: str


class EvidenceGateDecision(TypedDict, total=False):
    allowed: bool
    terminal: bool
    reason_code: str
    message: str
    experiment_names: List[str]
    contract_hashes: Dict[str, str]


class VerificationReport(TypedDict, total=False):
    passed: bool
    score: float
    note: str
    mismatches: List[str]
    checks: Dict[str, Any]


class LiteratureContext(TypedDict, total=False):
    """Raw literature retrieval output shared across research phases."""
    papers: List[Dict[str, Any]]
    graph_signals: List[Dict[str, Any]]
    evidence_map: Dict[str, Any]
    query: str
    domain: str


class FeasibilityReport(TypedDict, total=False):
    ok: bool
    reasons: List[str]


class CodeClaimReport(TypedDict, total=False):
    consistent: bool
    score: float
    notes: List[str]


class MetaDashboard(TypedDict, total=False):
    iteration: int
    phase: str
    rejected_topics: int
    debate_rounds: int
    sections_bounced: int
    pivots: int
    refines: int
    plan_revisions: int
    hard_check_fails: int
    avg_supervisor: float
    lessons_available: int


class MetaChatResult(TypedDict, total=False):
    reply: str
    control: str
    control_reason: str
    summary: Dict[str, Any]


class Paper(TypedDict, total=False):
    topic: Topic
    sections: Dict[str, str]
    plan: Plan
    engineer_outputs: Dict[str, ExperimentOutput]
    bibliography: str
    bib_map: Dict[str, str]
    companion_repo: Dict[str, str]
    debate_results: List[Any]
    timestamp: str


# ---------------------------------------------------------------------------
# Research Quality Pipeline Contracts (Discovery -> Debate -> Planning)
# ---------------------------------------------------------------------------

class ResearchGapReport(TypedDict, total=False):
    """Structured report produced by the ResearchScreener."""
    gap_type: str  # evaluation_gap, dataset_gap, method_gap, generalization_gap, robustness_gap, theoretical_gap, reproducibility_gap, resource_constraint_gap
    gap_claim: str
    supporting_papers: List[Dict[str, Any]]
    contradicting_papers: List[Dict[str, Any]]
    closest_prior_work: List[Dict[str, Any]]
    why_existing_work_is_insufficient: str
    proposed_contribution: str
    evidence_strength: float
    citation_gap_signal: float
    status: str  # PASS | FAIL | INSUFFICIENT_EVIDENCE
    reason: str


class NoveltyComparison(TypedDict, total=False):
    """Detailed contribution-level comparison against prior work."""
    paper: str
    problem: str
    dataset: str
    method: str
    variables: str
    evaluation: str
    claim: str
    contribution: str
    overlap: str
    difference: str
    remaining_gap: str
    novelty_relevance: str
    is_novel: bool
    verdict: str  # NOVEL | LIKELY_DUPLICATE | INSUFFICIENT_DIFFERENCE


class MinimumViableExperiment(TypedDict, total=False):
    """Smallest discriminative experiment required before debate pass."""
    dataset: str
    models: List[str]
    conditions: List[str]
    metrics: List[str]
    seeds: int
    baseline: str
    expected_result: str
    falsification_test: str
    competing_explanations: List[str]
    is_executable: bool
    is_discriminative: bool
    validation_notes: List[str]


class StructuredHypothesis(TypedDict, total=False):
    """Machine-checkable scientific hypothesis contract."""
    research_question: str
    hypothesis: str
    independent_variables: List[Dict[str, Any]]  # e.g. [{"name": "...", "values": [...]}]
    dependent_variables: List[str]  # e.g. ["ECE", "accuracy"]
    expected_relationship: str
    falsification_condition: str
    novelty_claim: str
    closest_prior_work: List[Dict[str, Any]]
    baselines: List[str]
    metrics: List[str]
    confounders: List[str]
    competing_explanations: List[str]
    minimum_viable_experiment: MinimumViableExperiment
    required_resources: Dict[str, Any]  # {"cpu": bool, "gpu": bool, "max_memory_mb": int, "max_runtime_seconds": int}
    gap_report: ResearchGapReport
    novelty_report: Dict[str, Any]


class DebateObjection(TypedDict, total=False):
    """Structured critique tracked across all debate rounds."""
    criterion: str  # soundness, significance, reproducibility, ethics, novelty, feasibility, confounder, baseline, evaluation, statistical
    objection: str
    severity: int  # 1-5
    status: str  # unresolved | resolved
    required_resolution: str
    resolution: str
    source: str  # challenger_audit | capability_manifest | minimum_experiment_check


class DebateDecision(TypedDict, total=False):
    """Moderator outcome combining hard deterministic gates and soft scores."""
    hard_gates: Dict[str, bool]
    scores: Dict[str, float]
    judge_agreement: float
    disagreement: float
    unresolved_objections: List[DebateObjection]
    decision: str  # PASS | FAIL
    passed: bool
    reasoning: str
    valid_judge_responses: int
