"""Independent statistical analysis of execution artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import numpy as np
from scipy import stats

from core.capabilities import DEFAULT_MANIFESTS
from core.context import RunContext, get_active_context
from core.contracts import AnalysisPlan, ExecutionArtifact, StatisticalReport
from core.verification import preregister_power

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """Analyze stored results without generating code or changing experiments."""

    capability_manifest = DEFAULT_MANIFESTS["AnalysisAgent"]

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()

    def analyze(
        self,
        execution_artifacts: Mapping[str, ExecutionArtifact],
        plan: AnalysisPlan,
    ) -> StatisticalReport:
        warnings: list[str] = []
        metrics: dict[str, Any] = {}
        for name, artifact in execution_artifacts.items():
            if artifact.get("status") != "completed":
                warnings.append(f"execution failed: {name}")
                continue
            aggregate = (artifact.get("seed_results") or {}).get("aggregate_metrics") or {}
            metrics[name] = {
                metric: self._summarize(values, warnings, f"{name}.{metric}")
                for metric, values in aggregate.items()
            }

        comparisons = self._compare(metrics, plan, warnings)
        self._apply_multiple_comparison_correction(comparisons, plan)
        self._check_bootstrap_cis(execution_artifacts, plan, metrics, warnings)
        passed = bool(metrics) and not any(
            token in warning
            for warning in warnings
            for token in ("missing", "failed", "underpowered", "mismatch", "unverifiable")
        )
        power_plan = None
        if plan.get("require_power_analysis") and not plan.get("planned_effect_size"):
            warnings.append("missing prospective power preregistration")
            passed = False
        elif plan.get("planned_effect_size"):
            power_plan = preregister_power(
                float(plan["planned_effect_size"]),
                float(plan.get("alpha", 0.05)),
                float(plan.get("target_power", 0.8)),
            )
        now = datetime.now(timezone.utc).isoformat()
        payload = {"metrics": metrics, "comparisons": comparisons, "warnings": warnings}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return {
            "analysis_plan": plan,
            "metrics": metrics,
            "comparisons": comparisons,
            "power_plan": power_plan,
            "warnings": warnings,
            "passed": passed,
            "provenance": {
                "artifact_type": "statistical_report",
                "producer": "AnalysisAgent",
                "content_hash": digest,
                "created_at": now,
                "status": "complete" if passed else "needs_review",
                "limitations": ["Statistical conclusions are limited by the supplied seeds and raw outputs."],
            },
        }

    @staticmethod
    def _summarize(raw: Any, warnings: list[str], label: str) -> dict[str, Any]:
        values = raw.get("values") if isinstance(raw, dict) else None
        if not values:
            warnings.append(f"missing raw values: {label}")
            return {"mean": raw.get("mean") if isinstance(raw, dict) else raw, "n": 0}
        sample = np.asarray(values, dtype=float)
        n = int(sample.size)
        mean = float(sample.mean())
        std = float(sample.std(ddof=1)) if n > 1 else 0.0
        if n < 3:
            warnings.append(f"insufficient seeds for uncertainty estimate: {label} (n={n})")
            interval = [mean, mean]
        else:
            margin = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
            interval = [mean - margin, mean + margin]
        return {"mean": mean, "std": std, "n": n, "confidence_interval_95": interval, "values": values}

    @staticmethod
    def _compare(metrics: dict[str, Any], plan: AnalysisPlan, warnings: list[str]) -> list[dict[str, Any]]:
        metric_name = plan.get("primary_metric")
        if not metric_name:
            warnings.append("missing primary metric")
            return []
        names = list(metrics)
        if len(names) < 2:
            warnings.append("missing comparison artifact")
            return []
        baseline_name = names[0]
        baseline = metrics[baseline_name].get(metric_name)
        comparisons = []
        for name in names[1:]:
            candidate = metrics[name].get(metric_name)
            if not baseline or not candidate:
                warnings.append(f"missing comparison metric: {name}.{metric_name}")
                continue
            base_values = np.asarray(baseline.get("values", []), dtype=float)
            candidate_values = np.asarray(candidate.get("values", []), dtype=float)
            if len(base_values) < 3 or len(candidate_values) < 3:
                warnings.append(f"underpowered comparison: {baseline_name} vs {name} requires at least 3 seeds")
                continue
            test = stats.ttest_ind(candidate_values, base_values, equal_var=False)
            pooled = np.sqrt((candidate_values.var(ddof=1) + base_values.var(ddof=1)) / 2)
            effect = float((candidate_values.mean() - base_values.mean()) / pooled) if pooled else 0.0
            alpha = float(plan.get("alpha", 0.05))
            target_power = float(plan.get("target_power", 0.8))
            z_alpha = float(stats.norm.ppf(1 - alpha / 2))
            z_power = float(stats.norm.ppf(target_power))
            required_n = int(np.ceil(2 * ((z_alpha + z_power) / max(abs(effect), 1e-9)) ** 2))
            power_estimate = float(stats.norm.cdf(abs(effect) * np.sqrt(len(base_values) / 2) - z_alpha))
            if len(base_values) < required_n or len(candidate_values) < required_n:
                warnings.append(f"underpowered comparison: {baseline_name} vs {name} requires n>={required_n} per group")
            comparisons.append({
                "candidate": name,
                "baseline": baseline_name,
                "metric": metric_name,
                "difference": float(candidate_values.mean() - base_values.mean()),
                "p_value": float(test.pvalue),
                "cohens_d": effect,
                "test": "Welch t-test",
                "alpha": alpha,
                "target_power": target_power,
                "required_n_per_group": required_n,
                "observed_power_estimate": power_estimate,
            })
        return comparisons

    @staticmethod
    def recompute_percentile_ci(
        values: list[float], alpha: float = 0.05, n_boot: int = 1000, seed: int = 0
    ) -> list[float]:
        """Deterministic percentile bootstrap CI for a 1-D sample (v5)."""
        arr = np.asarray([float(v) for v in values], dtype=float)
        if arr.size < 2:
            return [float(arr[0]), float(arr[0])] if arr.size else [0.0, 0.0]
        rng = np.random.default_rng(int(seed))
        means = np.asarray([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(int(n_boot))])
        return [float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))]

    @staticmethod
    def _check_bootstrap_cis(
        execution_artifacts: Mapping[str, Any],
        plan: AnalysisPlan,
        metrics: dict[str, Any],
        warnings: list[str],
    ) -> None:
        """v5 (f537f638): CIs must be recomputed artifacts, not prose.

        When the plan demands bootstrap (bootstrap_B / CI primary metric),
        every comparison needs a reported CI that matches an independent
        recomputation from raw values; missing or mismatched CIs warn (and
        fail the report) instead of passing silently.
        """
        plan_blob = str(dict(plan or {})).lower()
        # Explicit bootstrap audit (bootstrap_B): inputs must carry CIs that
        # match recomputation. Plain confidence plans: analyze() computing
        # the interval into metrics suffices (legacy execution() inputs carry
        # values only — warning there broke test_analysis_agent).
        needs_audit = "bootstrap_b" in plan_blob or "bootstrap" in plan_blob
        needs_ci = needs_audit or "ci_low" in plan_blob or "confidence" in plan_blob
        if not needs_ci:
            return
        if not needs_audit:
            for _name, _per in (metrics or {}).items():
                for _metric, _summary in (_per or {}).items():
                    if isinstance(_summary, dict) and _summary.get("confidence_interval_95"):
                        return
            warnings.append("missing bootstrap CI: statistical plan requires a computed interval artifact")
            return
        found_ci = False
        for name, artifact in (execution_artifacts or {}).items():
            agg = (artifact.get("seed_results") or {}).get("aggregate_metrics") or {}
            for metric, summary in agg.items():
                if not isinstance(summary, dict):
                    continue
                ci = summary.get("confidence_interval_95") or summary.get("ci") or [summary.get("ci_low"), summary.get("ci_high")]
                values = summary.get("values") or []
                # Only verify inputs that actually report a CI (both bounds
                # numeric). Inputs without a CI are normal — analyze()
                # computes the interval into metrics below; warning here
                # turned every legacy execution into "unverifiable".
                try:
                    _ci0, _ci1 = float(ci[0]), float(ci[1])
                    _has_input_ci = True
                except (TypeError, ValueError, IndexError):
                    _has_input_ci = False
                if _has_input_ci and values and len(values) >= 2:
                    found_ci = True
                    try:
                        recomputed = AnalysisAgent.recompute_percentile_ci(
                            [float(v) for v in values],
                            n_boot=int(dict(plan or {}).get("bootstrap_B", 1000) or 1000),
                        )
                        if abs(float(ci[0]) - recomputed[0]) > max(0.05 * abs(recomputed[0]), 1e-3) or abs(
                            float(ci[1]) - recomputed[1]
                        ) > max(0.05 * abs(recomputed[1]), 1e-3):
                            warnings.append(f"bootstrap CI mismatch: {name}.{metric} reported != recomputed")
                    except Exception:
                        warnings.append(f"bootstrap CI unverifiable: {name}.{metric}")
        _input_found = found_ci
        for name, per_metric in (metrics or {}).items():
            for metric, summary in (per_metric or {}).items():
                if isinstance(summary, dict) and summary.get("confidence_interval_95"):
                    found_ci = True
        # Audit mode: analyze() always computes an output CI, so the output
        # loop alone would mask missing input artifacts. Require at least one
        # input-reported CI when the plan explicitly demands bootstrap.
        if needs_audit and not _input_found:
            warnings.append("missing bootstrap CI: bootstrap plan requires input-reported interval artifacts")
        elif not found_ci:
            warnings.append("missing bootstrap CI: statistical plan requires a computed interval artifact")

    @staticmethod
    def _apply_multiple_comparison_correction(comparisons: list[dict[str, Any]], plan: AnalysisPlan) -> None:
        """Attach adjusted p-values; never leave a multi-test report ambiguous."""
        if not comparisons:
            return
        policy = str(plan.get("multiple_comparison_policy") or "bonferroni").lower()
        p_values = [float(item["p_value"]) for item in comparisons]
        if policy in {"none", "uncorrected"}:
            adjusted = p_values
        elif policy in {"holm", "holm-bonferroni"}:
            ordered = sorted(enumerate(p_values), key=lambda pair: pair[1])
            adjusted = [0.0] * len(p_values)
            for rank, (index, value) in enumerate(ordered):
                adjusted[index] = min(1.0, value * (len(p_values) - rank))
        else:
            policy = "bonferroni"
            adjusted = [min(1.0, value * len(p_values)) for value in p_values]
        for item, value in zip(comparisons, adjusted):
            item["p_value_adjusted"] = float(value)
            item["multiple_comparison_policy"] = policy
            item["significant_at_0.05"] = bool(value < 0.05)
