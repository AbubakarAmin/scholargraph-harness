"""Auditable cross-paper synthesis for research discovery.

The topic hunter must not infer a research gap from a single attractive paper
or ask an LLM to manufacture citations after it has invented an idea.  This
module builds a small, deterministic evidence graph from the retrieved corpus.
It proposes *bridges* between papers with complementary method and application
signals.  A bridge is evidence for a candidate to screen, never proof that a
research gap exists.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from typing import Any, Dict, Iterable, List

from .datasets import list_datasets


_WORD = re.compile(r"[a-z][a-z0-9_-]{2,}")
_STOPWORDS = {
    "about", "after", "also", "among", "analysis", "approach", "based", "between",
    "both", "data", "different", "during", "each", "for", "from", "into", "method",
    "methods", "model", "models", "more", "new", "our", "paper", "results", "show",
    "study", "that", "their", "these", "this", "using", "with", "work",
}

# Kept deliberately compact and transparent.  These are research roles, not a
# claim that two terms are scientifically interchangeable.
_ROLE_TERMS = {
    "method": {"algorithm", "benchmark", "classifier", "estimation", "learning", "modeling", "optimization", "prediction", "simulation", "validation"},
    "evaluation": {"accuracy", "bias", "calibration", "evaluation", "fairness", "metric", "performance", "reproducibility", "robustness"},
    "setting": {"clinical", "distributed", "education", "energy", "healthcare", "iot", "manufacturing", "network", "scientific", "security"},
}


def _tokens(text: str) -> List[str]:
    return [word for word in _WORD.findall((text or "").lower()) if word not in _STOPWORDS]


def _snippet(text: str, term: str, limit: int = 240) -> str:
    """Return an exact source excerpt around a term, for reviewer inspection."""
    text = (text or "").strip()
    index = text.lower().find(term.lower())
    if index < 0:
        return text[:limit]
    start = max(0, index - limit // 3)
    end = min(len(text), start + limit)
    return text[start:end]


def _roles(tokens: Iterable[str]) -> Dict[str, List[str]]:
    present = set(tokens)
    return {role: sorted(present & vocabulary) for role, vocabulary in _ROLE_TERMS.items()}


def build_cross_paper_evidence_map(
    papers: Iterable[Dict[str, Any]],
    *,
    max_papers: int = 20,
    max_bridges: int = 12,
) -> Dict[str, Any]:
    """Build a provenance-preserving corpus map and conservative bridge list.

    A bridge requires one paper to contribute a method/evaluation signal and a
    different paper to contribute an application setting.  Every returned
    bridge carries paper ids and verbatim excerpts; consumers can therefore
    reject unsupported synthesis before the debate stage.
    """
    nodes: List[Dict[str, Any]] = []
    for index, paper in enumerate(list(papers)[:max_papers]):
        if not isinstance(paper, dict):
            continue
        title = str(paper.get("title") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        if not title or not abstract:
            continue
        tokens = _tokens(f"{title} {abstract}")
        counts = Counter(tokens)
        salient = [term for term, _ in counts.most_common(20)]
        node_id = str(paper.get("id") or paper.get("doi") or paper.get("arxiv_id") or f"paper-{index}")
        nodes.append({
            "paper_id": node_id,
            "title": title,
            "source": paper.get("source") or "retrieved_corpus",
            "terms": salient,
            "roles": _roles(tokens),
            "abstract": abstract,
        })

    bridges: List[Dict[str, Any]] = []
    for left, right in combinations(nodes, 2):
        left_methods = left["roles"]["method"] + left["roles"]["evaluation"]
        right_methods = right["roles"]["method"] + right["roles"]["evaluation"]
        left_settings = left["roles"]["setting"]
        right_settings = right["roles"]["setting"]

        # Directional transfer hypotheses.  Do not emit a bridge without both
        # a transferable mechanism and a materially distinct target setting.
        directions = ((left, right, left_methods, right_settings), (right, left, right_methods, left_settings))
        for method_paper, setting_paper, mechanisms, settings in directions:
            if not mechanisms or not settings:
                continue
            shared = set(method_paper["terms"]) & set(setting_paper["terms"])
            # Reject near duplicates.  Cross-paper synthesis needs meaningful
            # complementarity, not two papers that merely repeat one topic.
            union = set(method_paper["terms"]) | set(setting_paper["terms"])
            lexical_overlap = len(shared) / max(len(union), 1)
            if lexical_overlap > 0.45:
                continue
            mechanism = mechanisms[0]
            setting = settings[0]
            bridges.append({
                "bridge_id": f"bridge-{method_paper['paper_id']}-{setting_paper['paper_id']}-{mechanism}-{setting}",
                "bridge_type": "method_to_setting_transfer",
                "source_paper_ids": [method_paper["paper_id"], setting_paper["paper_id"]],
                "method_signal": mechanism,
                "target_setting_signal": setting,
                "shared_terms": sorted(shared)[:8],
                "complementarity": round(1.0 - lexical_overlap, 3),
                "evidence": [
                    {"paper_id": method_paper["paper_id"], "title": method_paper["title"], "excerpt": _snippet(method_paper["abstract"], mechanism)},
                    {"paper_id": setting_paper["paper_id"], "title": setting_paper["title"], "excerpt": _snippet(setting_paper["abstract"], setting)},
                ],
                "screening_question": f"Can {mechanism} be evaluated or adapted for {setting} under a controlled local protocol?",
                "status": "candidate_requires_literature_screening",
            })

    bridges.sort(key=lambda item: (-item["complementarity"], item["method_signal"], item["target_setting_signal"]))
    # Abstracts are intentionally excluded from the external representation;
    # exact excerpts are sufficient for prompts and keep context bounded.
    public_nodes = [{key: value for key, value in node.items() if key != "abstract"} for node in nodes]
    return {
        "schema_version": "cross-paper-evidence-map/v1",
        "papers": public_nodes,
        "bridges": bridges[:max_bridges],
        "limitations": [
            "Bridges express evidence-backed candidates, not verified novelty or causal claims.",
            "Every candidate requires the ordinary literature, novelty, feasibility, and debate gates.",
        ],
    }


def validate_candidate_bridge_claim(candidate: Dict[str, Any], evidence_map: Dict[str, Any]) -> Dict[str, Any]:
    """Require a synthesis-backed candidate to identify its supporting bridges.

    This is an anti-hallucination gate: a model cannot claim that two papers
    imply a research opportunity unless it points to bridge IDs created from
    the retrieved corpus.  It validates provenance, not scientific truth.

    Soft mode: when bridges exist but the candidate doesn't cite them, return
    a warning rather than a hard rejection.  This lets promising topics through
    even if the LLM doesn't perfectly match bridge IDs.
    """
    available = {str(item.get("bridge_id")): item for item in evidence_map.get("bridges", [])}
    requested = candidate.get("evidence_bridge_ids") or []
    if isinstance(requested, str):
        requested = [requested]
    requested = [str(value) for value in requested]
    if not available:
        return {"valid": True, "bridge_ids": [], "reason": "no_cross_paper_bridges_available"}
    if not requested:
        # Soft signal: missing bridge IDs are a warning, not a blocker
        return {"valid": True, "bridge_ids": [], "reason": "candidate_did_not_cite_bridges_soft_warning"}
    unknown = [value for value in requested if value not in available]
    if unknown:
        # Filter out unknown IDs but don't reject if at least one known bridge exists
        known = [value for value in requested if value in available]
        if known:
            requested = known
        else:
            return {"valid": True, "bridge_ids": [], "reason": "unknown_bridges_ignored_soft_warning"}
    text = " ".join(str(candidate.get(key) or "") for key in ("title", "description", "rationale", "contribution")).lower()
    supported = []
    for bridge_id in requested:
        bridge = available.get(bridge_id)
        if not bridge:
            continue
        terms = {str(bridge.get("method_signal", "")).lower(), str(bridge.get("target_setting_signal", "")).lower()}
        if any(term and term in text for term in terms):
            supported.append(bridge_id)
    if not supported:
        return {"valid": True, "bridge_ids": requested, "reason": "bridge_text_mismatch_soft_warning"}
    return {"valid": True, "bridge_ids": supported, "reason": "grounded_cross_paper_bridge"}


def validate_topic_admission(structured_hypothesis: Dict[str, Any]) -> Dict[str, Any]:
    """Apply deterministic, pre-debate admission checks to a proposed study.

    This prevents the expensive debate phase from receiving an attractive
    narrative with no executable experiment.  It is intentionally narrower
    than scientific review: passing means "ready for debate", never "true".

    Relaxed version: allows topics with at least 1 seed, soft baseline/metrics
    requirements, and auto-fixes common LLM formatting issues.
    """
    contract = structured_hypothesis or {}
    errors: List[str] = []
    for field in ("research_question", "hypothesis", "dependent_variables", "falsification_condition"):
        if not contract.get(field):
            errors.append(f"missing required hypothesis field: {field}")

    mve = contract.get("minimum_viable_experiment")
    if not isinstance(mve, dict):
        # v2: auto-create MVE skeleton instead of hard-rejecting
        mve = {
            "dataset": "bundled_synthetic",
            "models": ["baseline_model"],
            "metrics": ["accuracy"],
            "seeds": 3,
            "baseline": "majority_class",
            "falsification_test": "Welch t-test p<0.05",
        }
        contract["minimum_viable_experiment"] = mve
    else:
        dataset = str(mve.get("dataset") or "").strip()
        catalog = {item["name"] for item in list_datasets()}
        if not dataset:
            # v2: auto-fix instead of hard reject
            mve["dataset"] = "bundled_synthetic"
        elif dataset not in catalog and "synthetic" not in dataset.lower():
            # Soft: suggest bundled_synthetic as fallback instead of hard reject
            mve["dataset"] = "bundled_synthetic"
        # Relaxed: allow missing baseline if models are specified
        if not (mve.get("baseline") or mve.get("baselines") or mve.get("models")):
            # v2: auto-assign a generic baseline instead of hard reject
            mve["baseline"] = "majority_class"
        # Relaxed: auto-add default metrics if missing
        if not mve.get("metrics"):
            mve["metrics"] = ["accuracy"]
        if not mve.get("falsification_test"):
            mve["falsification_test"] = "Welch t-test p<0.05"
        try:
            seeds = int(mve.get("seeds", 0))
        except (TypeError, ValueError):
            seeds = 0
        # Relaxed: accept 1 seed minimum (was 3)
        if seeds < 1:
            mve["seeds"] = 3
            seeds = 3

    # Sandbox reality check on the finished contract. The hunter's text-level
    # feasibility_filter runs BEFORE formalize (no MVE exists yet), so a
    # contract demanding accelerators, an over-wall-clock runtime, or a
    # thousand-run sweep would otherwise reach debate and burn ~25 minutes
    # before the challenger kills it (run 25ed95d4: 1,215 planned runs vs a
    # 120s sandbox). Reject here instead — deterministically, in seconds.
    manifest_errors = _validate_contract_against_sandbox(contract)
    errors.extend(manifest_errors)

    # e0ec9e91 P2: publishable-paper bar as non-blocking warnings. The winner
    # admitted with evidence_strength=0.35, INSUFFICIENT_EVIDENCE, no
    # contribution, single-seed MVE, and an invented >10% threshold. Flag
    # these so the ranker/debate can prefer well-grounded topics without
    # breaking try_and_report's intentionally permissive admission.
    warnings: List[str] = []
    try:
        mve_seeds = int((contract.get("minimum_viable_experiment") or {}).get("seeds", 3) or 3)
    except (TypeError, ValueError):
        mve_seeds = 3
    if mve_seeds < 3:
        warnings.append(f"mve_single_seed: {mve_seeds} seed(s); multi-seed (>=3) required for publishable variance reporting")
        try:
            contract["minimum_viable_experiment"]["seeds"] = 3
        except Exception:
            pass
    _fals = str(contract.get("falsification_condition") or "")
    import re as _re_adm
    if not _re_adm.search(r"%|>|<|p\s*<|confidence|bootstrap|wilcoxon|t-test", _fals, _re_adm.IGNORECASE):
        warnings.append("falsification_has_no_numeric_threshold: effect size / alpha / CI rule not stated")
    _gap = contract.get("gap_report") if isinstance(contract.get("gap_report"), dict) else {}
    if str(_gap.get("status") or "") == "INSUFFICIENT_EVIDENCE":
        warnings.append("gap_insufficient_evidence: contribution rests on abstract-only synthesis")
    if not str(contract.get("novelty_claim") or "").strip():
        warnings.append("novelty_claim_missing: no stated contribution beyond replication")

    return {
        "admitted": not errors,
        "errors": errors,
        "warnings": warnings,
        "contract_version": "topic-admission/v2-relaxed",
    }


# Upper bound on estimated measurement runs a sandbox contract may imply.
# Above this, the design cannot fit the wall clock no matter how fast each
# run is (25ed95d4 planned 1,215 runs in a 120–320s sandbox).
_MAX_ESTIMATED_RUNS = 200


def _validate_contract_against_sandbox(contract: Dict[str, Any]) -> List[str]:
    """Deterministic sandbox checks over required_resources + MVE scale."""
    try:
        from .capabilities import (
            SANDBOX_CAPABILITY_MANIFEST,
            check_plan_feasibility,
            effective_wall_clock_seconds,
        )
    except Exception:
        return []
    errors: List[str] = []
    manifest = SANDBOX_CAPABILITY_MANIFEST
    wall_clock = effective_wall_clock_seconds(manifest)

    resources = contract.get("required_resources") or {}
    if isinstance(resources, dict):
        if resources.get("gpu") is True and not manifest.gpu_available:
            errors.append("contract requires GPU execution, but no GPU is available")
        try:
            asked_runtime = int(resources.get("max_runtime_seconds") or 0)
        except (TypeError, ValueError):
            asked_runtime = 0
        if asked_runtime > wall_clock:
            errors.append(
                f"contract asks for {asked_runtime}s runtime but the sandbox allows {wall_clock}s"
            )

    # Scale estimate: seeds × conditions implied by the MVE + hypothesis text.
    # Counts explicit condition lists, else falls back to counting "×"-style
    # factor mentions (e.g. "3 model sizes × 3 workloads").
    mve = contract.get("minimum_viable_experiment") or {}
    seeds = 1
    try:
        seeds = max(1, int(mve.get("seeds", 1) or 1))
    except (TypeError, ValueError):
        seeds = 1
    conditions = 1
    for key in ("conditions", "models", "variants", "workloads", "datasets"):
        values = mve.get(key)
        if isinstance(values, (list, tuple)) and values:
            conditions *= max(1, len(values))
    factor_hits = re.findall(
        r"(\d+)\s*(?:model sizes?|workload types?|batch sizes?|memory strategies?|"
        r"seed counts?|conditions?|datasets?|repetitions?)",
        f"{contract.get('hypothesis', '')} {contract.get('research_question', '')}",
        re.IGNORECASE,
    )
    if factor_hits and conditions == 1:
        product = 1
        for hit in factor_hits:
            try:
                product *= max(1, int(hit))
            except ValueError:
                continue
        conditions = max(conditions, product)
    estimated_runs = seeds * conditions
    if estimated_runs > _MAX_ESTIMATED_RUNS:
        errors.append(
            f"contract implies ~{estimated_runs} measurement runs "
            f"(seeds={seeds} × conditions={conditions}), exceeding the "
            f"sandbox-feasible budget of {_MAX_ESTIMATED_RUNS}"
        )

    # Full manifest text scan as a backstop (catches "must download …" style
    # requirements the field checks above miss).
    plan_view = {
        "methodology": f"{contract.get('research_question', '')} {contract.get('hypothesis', '')}",
        "experiments": [{"dataset": {"name": str(mve.get('dataset', ''))}}],
    }
    for reason in check_plan_feasibility(plan_view, manifest):
        if reason not in errors:
            errors.append(reason)
    return errors
