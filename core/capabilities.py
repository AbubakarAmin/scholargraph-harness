"""Scoped capability policy for agent tool access.

This module deliberately contains policy only. Concrete source, execution, and
artifact adapters can be added behind these capability names later.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Set

from .contracts import AgentCapabilityManifest


CAPABILITIES = {
    "literature.search",
    "literature.fetch",
    "literature.citation_graph",
    "literature.contradiction_scan",
    "literature.evidence_synthesize",
    "memory.recall",
    "memory.record",
    "dataset.catalog",
    "dataset.download",
    # Scoped development-only tokens (post-M6 review decision): an agent that
    # can iterate on code must never be able to self-validate, and an agent
    # that fetches data through the sanctioned broker must never gain raw
    # outbound access. The coarse tokens stay exclusive (ExecutionAgent owns
    # independent replay via code.execute; DataAgent owns raw acquisition via
    # dataset.download); the scoped tokens below are disjoint from them.
    "dataset.download_brokered",
    "artifact.read",
    "artifact.write",
    "code.generate",
    "code.execute",
    "code.execute_sandbox",
    "analysis.statistics",
    "verification.replay",
    "verification.claims",
}

# Disjointness contract (machine-checked in tests/test_agent_registry.py):
# no manifest may hold both a development token and its validation-grade
# counterpart (code.execute_sandbox vs code.execute; dataset.download_brokered
# vs dataset.download). The pairings below encode which tokens conflict.
CONFLICTING_CAPABILITY_PAIRS = (
    ("code.execute_sandbox", "code.execute"),
    ("dataset.download_brokered", "dataset.download"),
)


@dataclass(frozen=True)
class CapabilityDecision:
    """Auditable result of a capability request."""

    allowed: bool
    agent: str
    capability: str
    reason: str


@dataclass(frozen=True)
class SandboxCapabilityManifest:
    """Shared execution limits used by planning, debate, and engineering."""

    # NOTE: the wall clock intentionally mirrors SANDBOX_TIMEOUT_SEC (the
    # value the sandbox dispatcher actually enforces). It is read at import
    # time; processes that change the env at runtime (web UI key sync) should
    # call refresh_manifest_wall_clock(). The previous hardcoded 120 silently
    # disagreed with a 320s sandbox and let oversized designs through.
    max_wall_clock_seconds: int = int(os.getenv("SANDBOX_TIMEOUT_SEC", "600"))
    available_libraries: tuple[str, ...] = ("numpy", "scipy", "pandas", "sklearn", "statsmodels")
    gpu_available: bool = False
    outbound_network: bool = False
    dataset_downloads: bool = False
    max_dataset_rows: int = 100_000
    max_training_epochs: int = 50
    max_samples: int = 100_000

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
            "available_libraries": list(self.available_libraries),
            "gpu_available": self.gpu_available,
            "outbound_network": self.outbound_network,
            "dataset_downloads": self.dataset_downloads,
            "max_dataset_rows": self.max_dataset_rows,
            "max_training_epochs": self.max_training_epochs,
            "max_samples": self.max_samples,
        }


SANDBOX_CAPABILITY_MANIFEST = SandboxCapabilityManifest()


# Tokens that indicate the experimental METHOD needs an accelerator.
# Deliberately method-level, NOT subject-level: a study *about* GPU systems
# (speculative decoding, serving, kernel scheduling) can run on CPU via
# simulation, small models, or statistical methodology — only a method that
# executes on the accelerator is unexecutable here. Subject-matter phrases
# ("llm serving", "kernel scheduling", "memory allocation") therefore do NOT
# appear here; the hunter steers those to CPU proxies via prompt guidance.
_GPU_METHOD_TOKENS = (
    "gpu", "cuda", "tpu", "nccl", "tensorrt", "vllm",
    "multi-gpu", "multi gpu", "distributed training",
    "large language model fine-tune", "deep neural network",
    "cuda kernel", "gpu profiling", "gpu cluster",
)

# Sentence-level cues that a GPU-token mention is a disclaimer or a
# CPU-proxy framing, not a requirement: "runs CPU-only and does not require
# a GPU", "use CPUs instead of GPUs", "we simulate GPU kernel behavior".
# The planner already rewrites bare GPU mentions to CPU-compatible text, so
# the checker must agree: only an un-disclaimed requirement is an error.
_GPU_DISCLAIMER_CUES = (
    "cpu-only", "cpu only", "does not require", "doesn't require",
    "do not require", "don't require", "no gpu", "without gpu",
    "without a gpu", "instead of", "rather than", "simulat",
    "proxy", "proxies", "analytical model", "emulat",
    # e0ec9e91: manifest/JSON echoes like '"gpu": false' or
    # 'gpu_available: False' are declarations of ABSENCE, not requirements.
    '"gpu": false', "'gpu': false", "gpu_available",
    "gpu: false", "false",
)


def _gpu_token_hit(sentence_lower: str, token: str) -> bool:
    """Shared word-boundary matcher for GPU-method tokens.

    Single source of truth for checker AND sanitizer: bare-substring checks
    false-positive on ordinary prose ("tpu" inside "output", "gpu" inside
    '"gpu": false'). Multi-word tokens match as phrases; single short tokens
    require word boundaries. Run e0ec9e91 root cause.
    """
    token = token.lower()
    if " " in token:
        return token in sentence_lower
    # Short accelerator abbreviations must not fire inside longer words.
    if len(token) <= 4:
        return re.search(r"\b" + re.escape(token) + r"\b", sentence_lower) is not None
    return re.search(r"\b" + re.escape(token) + r"\b", sentence_lower) is not None


def _gpu_required_by_text(plan_text: str) -> bool:
    """True when some sentence requires accelerator execution.

    A sentence containing a GPU-method token counts as a requirement unless
    it also carries a disclaimer cue (CPU-only framing, negation, or an
    explicit simulation/proxy design — the sanctioned CPU way to study
    accelerator systems). Uses word-boundary matching shared with the
    planner sanitizer so checker and sanitizer agree.
    """
    for sentence in re.split(r"[.\n;]+", plan_text):
        lowered = sentence.lower()
        if any(_gpu_token_hit(lowered, token) for token in _GPU_METHOD_TOKENS) and not any(
            cue in lowered for cue in _GPU_DISCLAIMER_CUES
        ):
            return True
    return False


def refresh_manifest_wall_clock(manifest: SandboxCapabilityManifest = SANDBOX_CAPABILITY_MANIFEST) -> int:
    """Re-read SANDBOX_TIMEOUT_SEC into the shared manifest (frozen: rebuild).

    Returns the effective wall clock. Callers holding the old object keep
    working — only the module singleton is replaced.
    """
    global SANDBOX_CAPABILITY_MANIFEST
    try:
        wall_clock = max(1, int(os.getenv("SANDBOX_TIMEOUT_SEC", "") or manifest.max_wall_clock_seconds))
    except (TypeError, ValueError):
        wall_clock = manifest.max_wall_clock_seconds
    if wall_clock != manifest.max_wall_clock_seconds:
        SANDBOX_CAPABILITY_MANIFEST = SandboxCapabilityManifest(
            max_wall_clock_seconds=wall_clock,
            available_libraries=manifest.available_libraries,
            gpu_available=manifest.gpu_available,
            outbound_network=manifest.outbound_network,
            dataset_downloads=manifest.dataset_downloads,
            max_dataset_rows=manifest.max_dataset_rows,
            max_training_epochs=manifest.max_training_epochs,
            max_samples=manifest.max_samples,
        )
    return SANDBOX_CAPABILITY_MANIFEST.max_wall_clock_seconds


def effective_wall_clock_seconds(manifest: SandboxCapabilityManifest = SANDBOX_CAPABILITY_MANIFEST) -> int:
    """Current enforceable wall clock: live env wins, manifest is the floor."""
    try:
        live = int(os.getenv("SANDBOX_TIMEOUT_SEC", "") or 0)
    except (TypeError, ValueError):
        live = 0
    return max(1, live or manifest.max_wall_clock_seconds)


def check_plan_feasibility(plan: Mapping[str, Any], manifest: SandboxCapabilityManifest = SANDBOX_CAPABILITY_MANIFEST) -> List[str]:
    """Return blocking reasons before an experiment contract is committed.

    Tiered dataset policy (v5):
      Tier-0 local (bundled_synthetic, sklearn_*, openreview_calibration,
        local-only) — always allowed.
      Tier-1 HF-brokered (huggingface / hf-cached / memory/dataset_cache) —
        allowed via the sanctioned broker in core/datasets.py which enforces
        HUGGINGFACE_MAX_DOWNLOAD_MB + row caps and caches locally before the
        sandbox runs. Generic words like "download" in that context do NOT
        count as raw outbound access.
      Tier-2 raw outbound (yahoo finance, wikipedia traffic, uci, bare
        internet/outbound download) — forbidden while dataset_downloads=False.
    """
    errors: List[str] = []
    # Exclude audit/metadata keys — their own labels (e.g. "outbound_network",
    # "capability_rescope.reason: Outbound downloads/GPU...") would
    # false-positive on substring matching. Only executable content is scanned.
    _AUDIT_KEYS = {
        "capability_manifest",
        "capability_rescope",
        "feasibility_errors",
        "dataset_catalog",
        "dataset_resolutions",
        "revision_history",
        "fallback_reason",
        "plan_mode",
    }
    plan_for_text = {k: v for k, v in plan.items() if k not in _AUDIT_KEYS}
    plan_text = str(plan_for_text).lower()
    SANCTIONED_SOURCE_TOKENS = (
        "huggingface",
        "hf-cached",
        "hf_cached",
        "memory/dataset_cache",
        "memory\\dataset_cache",
        "dataset_cache",
        "bundled_synthetic",
        "bundled synthetic",
        "bundled-synthetic",
        "bundled benchmark",
        "bounded local",
        "local-only",
        "local only",
        "sklearn_iris",
        "sklearn_digits",
        "sklearn",
        "openreview_calibration",
        "scikit-learn bundled",
    )
    RAW_DATASET_TOKENS = ("yahoo finance", "wikipedia traffic", "uci")
    GENERIC_NETWORK_TOKENS = ("internet", "outbound", "download")
    # Word-boundary matching: bare-substring checks false-positive on ordinary
    # prose ("uci" ⊂ "reproducibility", "download" ⊂ "downloads"). Multi-word
    # tokens still match as phrases; single short tokens require boundaries.
    # This mirrors the \b regexes already used by the planner's auto-fix
    # sanitizer, so checker and sanitizer now agree on what a violation is.
    def _token_hit(token: str) -> bool:
        if " " in token:
            return token in plan_text
        return re.search(r"\b" + re.escape(token) + r"\b", plan_text) is not None
    has_sanctioned = any(token in plan_text for token in SANCTIONED_SOURCE_TOKENS)
    if not manifest.dataset_downloads:
        if any(_token_hit(token) for token in RAW_DATASET_TOKENS):
            errors.append("plan requires outbound dataset access, which the sandbox forbids")
        elif any(_token_hit(token) for token in GENERIC_NETWORK_TOKENS) and not has_sanctioned:
            errors.append("plan requires outbound dataset access, which the sandbox forbids")
    if not manifest.gpu_available and _gpu_required_by_text(plan_text):
        errors.append("plan requires GPU-scale execution, but no GPU is available")
    for index, experiment in enumerate(plan.get("experiments") or []):
        if not isinstance(experiment, Mapping):
            continue
        dataset = experiment.get("dataset") or {}
        rows = dataset.get("rows") or dataset.get("row_count") or dataset.get("size")
        if isinstance(rows, (int, float)) and rows > manifest.max_dataset_rows:
            errors.append(f"experiment[{index}] dataset exceeds max rows ({manifest.max_dataset_rows})")
        epochs = experiment.get("epochs") or experiment.get("training_epochs")
        if isinstance(epochs, (int, float)) and epochs > manifest.max_training_epochs:
            errors.append(f"experiment[{index}] exceeds max training epochs ({manifest.max_training_epochs})")
        samples = experiment.get("samples") or experiment.get("n_samples")
        if isinstance(samples, (int, float)) and samples > manifest.max_samples:
            errors.append(f"experiment[{index}] exceeds max samples ({manifest.max_samples})")
    return errors


def manifest_for(
    agent: str,
    role: str,
    allowed: Iterable[str],
    forbidden: Iterable[str] = (),
    inputs: Iterable[str] = (),
    outputs: Iterable[str] = (),
    can_mutate: Iterable[str] = (),
) -> AgentCapabilityManifest:
    """Build a normalized manifest and reject unknown capability names."""
    allowed_set = set(allowed)
    forbidden_set = set(forbidden)
    unknown = (allowed_set | forbidden_set) - CAPABILITIES
    if unknown:
        raise ValueError(f"Unknown capabilities: {sorted(unknown)}")
    overlap = allowed_set & forbidden_set
    if overlap:
        raise ValueError(f"Capabilities cannot be both allowed and forbidden: {sorted(overlap)}")
    return {
        "agent": agent,
        "role": role,
        "allowed_capabilities": sorted(allowed_set),
        "forbidden_capabilities": sorted(forbidden_set),
        "input_artifact_types": sorted(set(inputs)),
        "output_artifact_types": sorted(set(outputs)),
        "can_mutate": sorted(set(can_mutate)),
    }


def authorize(
    manifest: Mapping[str, object],
    capability: str,
) -> CapabilityDecision:
    """Authorize one operation using only the agent's declared manifest."""
    agent = str(manifest.get("agent") or "unknown")
    if capability not in CAPABILITIES:
        return CapabilityDecision(False, agent, capability, "unknown capability")
    allowed = set(manifest.get("allowed_capabilities") or [])
    forbidden = set(manifest.get("forbidden_capabilities") or [])
    if capability in forbidden:
        return CapabilityDecision(False, agent, capability, "explicitly forbidden")
    if capability not in allowed:
        return CapabilityDecision(False, agent, capability, "not declared by manifest")
    return CapabilityDecision(True, agent, capability, "declared by manifest")


DEFAULT_MANIFESTS = {
    "TopicHunterAgent": manifest_for(
        "TopicHunterAgent",
        "research discovery",
        {"literature.search", "literature.fetch", "literature.citation_graph",
         "literature.contradiction_scan", "literature.evidence_synthesize",
         "artifact.write", "artifact.read", "memory.recall", "memory.record"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"research_domain", "prior_lessons"},
        {"topic_candidates"},
        {"topic_candidates"},
    ),
    "HypothesisDebateSystem": manifest_for(
        "HypothesisDebateSystem",
        "hypothesis review",
        {"literature.search", "literature.fetch", "artifact.read", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics"},
        {"topic_candidate", "prior_lessons"},
        {"debate_result"},
        {"debate_result"},
    ),
    # Debate sub-roles (§4.1/§4.4): proposer argues point-by-point, challenger
    # scores the severity checklist, moderator runs the ensemble. None may
    # download data, generate/execute code, run statistics, or raise claims.
    "ProposerAgent": manifest_for(
        "ProposerAgent",
        "debate proposition",
        {"literature.search", "literature.fetch", "literature.citation_graph",
         "literature.evidence_synthesize", "artifact.read", "artifact.write", "memory.recall"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"topic_candidate", "literature_evidence"},
        {"argument"},
        {"argument"},
    ),
    "ChallengerAgent": manifest_for(
        "ChallengerAgent",
        "debate challenge",
        {"artifact.read", "artifact.write", "literature.search", "literature.fetch",
         "literature.citation_graph", "literature.contradiction_scan", "memory.recall"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"topic_candidate", "argument"},
        {"objections"},
        {"objections"},
    ),
    "ModeratorAgent": manifest_for(
        "ModeratorAgent",
        "debate moderation",
        {"artifact.read", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims", "literature.search", "literature.fetch"},
        {"arguments", "objections"},
        {"debate_result"},
        {"debate_result"},
    ),
    "PlannerAgent": manifest_for(
        "PlannerAgent",
        "experiment planning",
        {"artifact.read", "artifact.write", "memory.recall", "memory.record"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"topic", "debate_result"},
        {"plan"},
        {"plan"},
    ),
    "WriterAgent": manifest_for(
        "WriterAgent",
        "scientific writing",
        {"artifact.read", "artifact.write", "literature.search", "literature.fetch"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics"},
        {"topic", "plan", "evidence"},
        {"draft_section"},
        {"draft_section"},
    ),
    "SupervisorAgent": manifest_for(
        "SupervisorAgent",
        "quality supervision",
        {"artifact.read", "verification.claims", "literature.search", "literature.fetch"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics"},
        {"draft_section", "evidence"},
        {"supervisor_feedback"},
        {"supervisor_feedback"},
    ),
    "MetaAgent": manifest_for(
        "MetaAgent",
        "workflow evaluation",
        {"artifact.read", "artifact.write", "verification.claims"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics"},
        {"workflow_state", "feedback"},
        {"meta_feedback"},
        {"meta_feedback"},
    ),
    "EditorAgent": manifest_for(
        "EditorAgent",
        "manuscript assembly",
        {"artifact.read", "artifact.write", "verification.claims", "literature.fetch"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics"},
        {"draft_sections", "verification_findings", "citations"},
        {"final_paper", "latex_output"},
        {"final_paper", "latex_output"},
    ),
    # EngineerAgent (preserved §4.1 behavior: generate → validate →
    # run_multi_seed → refine): holds the sandbox-scoped development tokens
    # ONLY — code.execute_sandbox (own-candidate pre-runs inside the AST /
    # container sandbox, never independent validation) and
    # dataset.download_brokered (local catalog → Tier-1 HF broker with caps +
    # cache, never raw outbound). Explicitly forbidden the validation-grade
    # counterparts, so the engineer can never self-validate its own code.
    "EngineerAgent": manifest_for(
        "EngineerAgent",
        "implementation",
        {"code.generate", "code.execute_sandbox", "dataset.download_brokered",
         "artifact.read", "artifact.write", "memory.recall", "memory.record"},
        {"dataset.download", "code.execute", "analysis.statistics", "verification.claims"},
        {"plan", "dataset_spec"},
        {"code"},
        {"code"},
    ),
    "DataAgent": manifest_for(
        "DataAgent",
        "data stewardship",
        {"literature.search", "literature.fetch", "dataset.catalog", "dataset.download", "artifact.write"},
        {"code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"research_question", "dataset_spec"},
        {"dataset"},
        {"dataset"},
    ),
    "ExecutionAgent": manifest_for(
        "ExecutionAgent",
        "experiment execution",
        {"artifact.read", "code.execute", "artifact.write"},
        {"code.generate", "dataset.download", "analysis.statistics", "verification.claims"},
        {"code", "dataset", "execution_request"},
        {"execution"},
        {"execution"},
    ),
    "AnalysisAgent": manifest_for(
        "AnalysisAgent",
        "statistical analysis",
        {"artifact.read", "analysis.statistics", "artifact.write"},
        {"code.generate", "code.execute", "dataset.download", "verification.claims"},
        {"execution", "analysis_plan"},
        {"statistical_report"},
        {"statistical_report"},
    ),
    "VerificationAgent": manifest_for(
        "VerificationAgent",
        "independent verification",
        {"artifact.read", "verification.replay", "verification.claims", "artifact.write"},
        {"code.generate", "dataset.download", "analysis.statistics"},
        {"dataset", "code", "execution", "statistical_report", "claim"},
        {"verification_finding"},
        {"verification_finding"},
    ),
    # M1 (§4.3): thesis construction for no-experiment types. OA full-text
    # fetch allowed for ReviewAgent only (M2); ThesisAgent has no execute.
    "ThesisAgent": manifest_for(
        "ThesisAgent",
        "thesis construction",
        {"literature.search", "literature.fetch", "artifact.read", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"topic", "literature_evidence"},
        {"thesis"},
        {"thesis"},
    ),
    # M2 (§4.3): systematic review. OA full-text fetch allowed license-checked only.
    "ReviewAgent": manifest_for(
        "ReviewAgent",
        "systematic review",
        {"literature.search", "literature.fetch", "artifact.read", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"topic", "corpus_candidate", "intake_report"},
        {"corpus"},
        {"corpus"},
    ),
    # M3 (§4.3): survey analysis + case protocol. Both read packs via the
    # DataAgent handoff only (no download/generate/execute/claims).
    "SurveyAnalysisAgent": manifest_for(
        "SurveyAnalysisAgent",
        "survey analysis",
        {"artifact.read", "analysis.statistics", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "verification.claims"},
        {"response_table", "codebook", "intake_report"},
        {"survey_analysis"},
        {"survey_analysis"},
    ),
    "SurveyAgent": manifest_for(
        "SurveyAgent",
        "survey analysis",
        {"artifact.read", "analysis.statistics", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "verification.claims"},
        {"response_table", "codebook", "intake_report"},
        {"survey_analysis"},
        {"survey_analysis"},
    ),
    "CaseProtocolAgent": manifest_for(
        "CaseProtocolAgent",
        "case protocol",
        {"literature.search", "literature.fetch", "artifact.read", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"topic", "case_pack", "intake_report"},
        {"case_protocol"},
        {"case_protocol"},
    ),
    "CaseAgent": manifest_for(
        "CaseAgent",
        "case protocol",
        {"literature.search", "literature.fetch", "artifact.read", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "analysis.statistics", "verification.claims"},
        {"topic", "case_pack", "intake_report"},
        {"case_protocol"},
        {"case_protocol"},
    ),
    # M4 (§4.3): causal identification audit. Reads plans/outputs via the
    # workflow handoff only (no generate/execute/download/claims).
    "CausalAuditAgent": manifest_for(
        "CausalAuditAgent",
        "causal audit",
        {"artifact.read", "analysis.statistics", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "verification.claims"},
        {"plan", "engineer_outputs"},
        {"causal_claim"},
        {"causal_claim"},
    ),
    "CausalAgent": manifest_for(
        "CausalAgent",
        "causal audit",
        {"artifact.read", "analysis.statistics", "artifact.write"},
        {"dataset.download", "code.generate", "code.execute", "verification.claims"},
        {"plan", "engineer_outputs"},
        {"causal_claim"},
        {"causal_claim"},
    ),
}
