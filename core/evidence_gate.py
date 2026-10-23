"""Deterministic gates for immutable experiments and evidence handoffs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .contracts import EvidenceGateDecision, ExperimentContract, ExperimentSpec
from .verification import FIELD_WAIVER_TYPES, NON_EMPIRICAL_TYPES

# R2/R3 (§6.1): plan shapes with no experiment artifacts. Non-empirical
# recipes (analytical/descriptive/argumentative/review, §3.3) and field
# recipes whose artifact is a coverage table / triangulation matrix rather
# than experiment outcomes (survey/case_study, M3) may carry
# ``experiments: []`` with a ``non_empirical`` plan_mode. Empirical /
# experimental / comparative / cause_effect ALWAYS require experiments.
EXPERIMENT_WAIVER_TYPES = frozenset(NON_EMPIRICAL_TYPES | FIELD_WAIVER_TYPES)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _baselines_of(experiment: Mapping[str, Any]) -> List[Any]:
    """Declared baselines for one experiment (both legacy key spellings)."""
    baselines = experiment.get("baselines") or experiment.get("baseline_comparison") or []
    if isinstance(baselines, str):
        baselines = [baselines]
    return [item for item in (baselines or []) if str(item).strip()]


# M4 (§4.5): comparative plans compare a proposed method against a baseline
# matrix — every experiment must name at least two baselines.
COMPARATIVE_MIN_BASELINES = 2


def validate_comparative_plan(experiments: Any) -> List[str]:
    """Comparative per-type schema (§4.5): min_baselines=2, ablation=on."""
    errors: List[str] = []
    if not isinstance(experiments, list) or not experiments:
        return ["comparative plan requires a non-empty experiment list"]
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            continue
        name = experiment.get("name") or "unnamed"
        baselines = _baselines_of(experiment)
        if len(baselines) < COMPARATIVE_MIN_BASELINES:
            errors.append(
                f"{name} declares {len(baselines)} baseline(s); "
                f"comparative requires >= {COMPARATIVE_MIN_BASELINES} "
                "(refusal: needs stronger comparison)"
            )
    return errors


def validate_causal_plan(plan: Mapping[str, Any] | None) -> List[str]:
    """Cause-effect per-type schema (§4.5): plan requires causal_spec{dag, id_strategy}."""
    errors: List[str] = []
    plan = plan or {}
    spec = plan.get("causal_spec")
    if not isinstance(spec, Mapping):
        return ["cause_effect plan requires causal_spec{dag, id_strategy}"]
    dag = spec.get("dag")
    if not isinstance(dag, Mapping) or not dag:
        errors.append("cause_effect plan requires causal_spec.dag (non-empty mapping)")
    if not str(spec.get("id_strategy") or "").strip():
        errors.append("cause_effect plan requires causal_spec.id_strategy")
    return errors


def validate_experiments(experiments: Any, paper_type: str | None = None, plan: Mapping[str, Any] | None = None) -> List[str]:
    """Return deterministic validation errors for a plan's experiment list.

    ``paper_type`` selects the §4.5 per-type schema (additive: ``None``
    preserves the legacy generic check exactly). Experimental types forbid
    ``experiments: []``; comparative additionally requires ≥2 baselines per
    experiment; cause_effect additionally requires ``causal_spec{dag,
    id_strategy}`` on the plan.

    R3 (§6.1): an empty experiment list is accepted when the plan is
    non-experimental — ``paper_type`` (or the plan's own ``paper_type``) in
    ``NON_EMPIRICAL_TYPES | FIELD_WAIVER_TYPES``, or the plan carrying
    ``plan_mode == "non_empirical"``. Experimental / comparative /
    cause_effect plans keep the legacy strict rejection.
    """
    errors: List[str] = []
    if not isinstance(experiments, list) or not experiments:
        effective_type = paper_type
        plan_mode: Any = None
        if isinstance(plan, Mapping):
            if effective_type is None:
                effective_type = plan.get("paper_type")
            plan_mode = plan.get("plan_mode")
        if effective_type in EXPERIMENT_WAIVER_TYPES or plan_mode == "non_empirical":
            return []
        return ["plan.experiments must be a non-empty list"]
    names = set()
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            errors.append(f"experiment[{index}] must be an object")
            continue
        name = experiment.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"experiment[{index}] must have a non-empty name")
            continue
        if name in names:
            errors.append(f"duplicate experiment name: {name}")
        names.add(name)
        if not experiment.get("evaluation_metrics"):
            errors.append(f"{name} must declare evaluation_metrics")
        variants = experiment.get("variants") or experiment.get("alternatives") or []
        for variant_index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                errors.append(f"{name} variant[{variant_index}] must be an object")
    if paper_type == "comparative":
        errors.extend(validate_comparative_plan(experiments))
    if paper_type == "cause_effect":
        errors.extend(validate_causal_plan(plan if isinstance(plan, Mapping) else {}))
    return errors


def build_contract(experiment: ExperimentSpec, dataset: Mapping[str, Any] | None = None) -> ExperimentContract:
    """Create the immutable snapshot used by all repair and validation attempts."""
    contract: ExperimentContract = {
        "experiment_name": str(experiment.get("name") or ""),
        "hypothesis": str(experiment.get("falsifiable_prediction") or experiment.get("description") or ""),
        "dataset": dict(dataset or experiment.get("dataset") or {}),
        "requirements": {
            "type": experiment.get("type"),
            "methodology": experiment.get("methodology"),
            "claimed_components": experiment.get("claimed_components") or [],
        },
        "baselines": list(experiment.get("baselines") or experiment.get("baseline_comparison") or []),
        "evaluation_metrics": list(experiment.get("evaluation_metrics") or []),
        "split_policy": str(experiment.get("split_policy") or ""),
        "seeds": list(experiment.get("seeds") or []),
        "stopping_rule": str(experiment.get("stopping_rule") or ""),
        "analysis_protocol": {
            "statistical_test": experiment.get("statistical_test"),
            "analysis_plan": experiment.get("analysis_plan") or {},
        },
    }
    contract["contract_hash"] = _stable_hash(contract)
    return contract


def contract_matches(contract: Mapping[str, Any], experiment: Mapping[str, Any]) -> bool:
    # L9: no dataset override — recompute from the experiment alone so
    # dataset drift changes the hash and is detected.
    expected = build_contract(experiment)
    return expected.get("contract_hash") == contract.get("contract_hash")


def validate_dataset_identity(
    experiments: Iterable[Mapping[str, Any]],
    data_artifacts: Mapping[str, Mapping[str, Any]] | None,
) -> List[str]:
    """Ensure declared dataset names and hashes resolve to validated artifacts."""
    artifacts = data_artifacts or {}
    errors: List[str] = []
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            continue
        name = experiment.get("name") or "unnamed"
        declared = experiment.get("dataset") or {}
        if not declared:
            continue
        dataset_name = declared.get("name") or declared.get("id")
        if not dataset_name:
            errors.append(f"{name} declares a dataset without a name or id")
            continue
        artifact = artifacts.get(dataset_name)
        if not artifact:
            # bundled_synthetic is generated locally by the execution layer;
            # it has no user-supplied file artifact to validate in advance.
            # Keep external dataset identity strict, but allow this explicit
            # built-in dataset through to engineering.
            if str(dataset_name).strip().lower() in {"bundled_synthetic", "synthetic"}:
                continue
            errors.append(f"{name} dataset artifact not found: {dataset_name}")
            continue
        if artifact.get("validation", {}).get("passed") is False:
            errors.append(f"{name} dataset failed validation: {dataset_name}")
        declared_hash = declared.get("content_hash") or declared.get("hash")
        actual_hash = artifact.get("content_hash") or artifact.get("provenance", {}).get("content_hash")
        if declared_hash and actual_hash and declared_hash != actual_hash:
            errors.append(f"{name} dataset hash changed: {dataset_name}")
    return errors


def gate_engineering_outputs(
    plan: Mapping[str, Any] | None,
    outputs: Mapping[str, Mapping[str, Any]] | None,
    contracts: Mapping[str, Mapping[str, Any]] | None,
    paper_type: str | None = None,
) -> EvidenceGateDecision:
    """Decide whether engineering may hand off to independent validation.

    ``paper_type`` is additive (``None`` preserves the legacy strict check):
    when absent it falls back to the plan's own ``paper_type``. R2/R3 (§6.1):
    a non-experimental plan (``experiments: []`` + waiver type /
    ``non_empirical`` mode) carries no raw-results or metrics demands — there
    are no per-experiment artifacts to gate on — so the empty handoff is
    allowed instead of terminal.
    """
    experiments = (plan or {}).get("experiments") or []
    effective_type = paper_type
    if effective_type is None and isinstance(plan, Mapping):
        effective_type = plan.get("paper_type")
    errors = validate_experiments(experiments, paper_type=effective_type, plan=plan if isinstance(plan, Mapping) else None)
    if errors:
        return {
            "allowed": False,
            "terminal": True,
            "reason_code": "invalid_experiment_plan",
            "message": "; ".join(errors),
        }
    outputs = outputs or {}
    contracts = contracts or {}
    names = [experiment["name"] for experiment in experiments if isinstance(experiment, dict)]
    missing = [name for name in names if name not in outputs]
    if missing:
        return {
            "allowed": False,
            "terminal": True,
            "reason_code": "missing_experiment_output",
            "message": f"No engineering output for: {', '.join(missing)}",
            "experiment_names": names,
        }
    for name in names:
        # L16 grouped success + L17 canonical unwrap + L10 safe access.
        output = outputs[name] or {}
        # v5 (f537f638): fallback_synthetic is exploratory plumbing — it must
        # never hand off to validation/writing as if it were evidence.
        if str(output.get("outcome") or "") == "fallback_synthetic":
            return {
                "allowed": False,
                "terminal": True,
                "reason_code": "exploratory_only",
                "message": f"Experiment {name} is fallback_synthetic exploratory plumbing, not publishable evidence",
                "experiment_names": names,
            }
        if output.get("success") is not True:
            return {
                "allowed": False,
                "terminal": True,
                "reason_code": "technical_execution_failure",
                "message": f"Experiment {name} did not complete: {output.get('error') or 'unknown error'}",
                "experiment_names": names,
            }
        # success-True but empty metrics falls through to the retryable
        # missing_metrics check below (L11).
        contract = contracts.get(name) or {}
        if not contract.get("contract_hash") or output.get("contract_hash") != contract.get("contract_hash"):
            return {
                "allowed": False,
                "terminal": True,
                "reason_code": "contract_provenance_mismatch",
                "message": f"Experiment {name} is not tied to its committed contract",
                "experiment_names": names,
            }
        raw_path = output.get("raw_results_path")
        if not raw_path or not Path(raw_path).is_file():
            return {
                "allowed": False,
                # L11: retryable, not structural-terminal.
                "terminal": False,
                "reason_code": "missing_raw_results",
                "message": f"Experiment {name} has no raw results artifact",
                "experiment_names": names,
            }
        if not output.get("aggregate_metrics") and not (output.get("results") or {}).get("metrics"):
            return {
                "allowed": False,
                # L11: retryable, not structural-terminal.
                "terminal": False,
                "reason_code": "missing_metrics",
                "message": f"Experiment {name} produced no metrics",
                "experiment_names": names,
            }
        # v5: degenerate variance (identical values / std=0 on tiny n) is
        # retryable — the run may still recover via plan revision — but it
        # must never pass the gate as evidence.
        for metric, agg in (output.get("aggregate_metrics") or {}).items():
            if not isinstance(agg, dict):
                continue
            values = agg.get("values") or []
            n = int(agg.get("n") or len(values) or 0)
            if values and len({float(v) for v in values if isinstance(v, (int, float))}) <= 1:
                return {
                    "allowed": False,
                    "terminal": False,
                    "reason_code": "degenerate_metrics",
                    "message": f"Experiment {name}.{metric} has identical values across seeds (no seed variation)",
                    "experiment_names": names,
                }
    return {
        "allowed": True,
        "terminal": False,
        "reason_code": "engineering_complete",
        "message": "All committed experiments completed with raw metrics and provenance",
        "experiment_names": names,
        "contract_hashes": {name: contracts[name]["contract_hash"] for name in names},
    }
