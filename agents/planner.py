"""
PlannerAgent — falsifiable, revisable experiment designer.
Bidirectional Engineer → Planner revision path; baselines required.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action, parse_json_from_llm
from core.llm import call_llm, generate_embedding
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.contracts import Contribution, Dependency, ExperimentSpec, Plan, RevisionRequest, Timeline, Topic
from core.evidence_gate import validate_experiments
from core.memory import memory
from core.run_log import get_tracker, CrossRunMemory
from core.capabilities import (
    SANDBOX_CAPABILITY_MANIFEST,
    SandboxCapabilityManifest,
    check_plan_feasibility,
)
from core.verification import preregister_power
from core.datasets import list_datasets, get_dataset_info, download_hf_dataset, load_local_dataset

logger = logging.getLogger(__name__)


def _planner_system(prompt_name: str, **slots: Any) -> Dict[str, Any]:
    """System kwarg from the v2 registry (stub-safe: {} when unavailable)."""
    try:
        from core.prompts import render_system

        system = render_system(prompt_name, **slots)
        return {"system": system} if system else {}
    except Exception:
        return {}


class PlannerAgent:
    """Creates falsifiable research plans with baselines and experiment branches."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    @property
    def vector_memory(self):
        return self.context.memory if self.context else memory

    def create_plan(self, topic: Topic, paper_type: Optional[str] = None) -> Plan:
        logger.info("Starting plan creation")
        logger.info("Planning for topic: %s", topic.get("title"))
        log_agent_action("PlannerAgent", "start_planning", {"topic": topic.get("title")})
        lessons = json.dumps(CrossRunMemory().get_prompt_context(), sort_keys=True)
        # M4 (§4.5): planner per-type schemas. Legacy callers pass only the
        # topic; the topic may itself carry a paper_type hint (typed runs).
        resolved_type = str(
            paper_type or (topic.get("paper_type") if isinstance(topic, dict) else "") or "experimental"
        )
        if resolved_type == "auto":
            resolved_type = "experimental"

        plan = self._generate_plan_structure(topic, lessons)
        logger.info("Plan structure generated: %d sections", len(plan.get("sections", [])))
        plan["dataset_catalog"] = list_datasets()
        # Carried on the plan (not a call kwarg) so legacy two-arg helpers
        # and test doubles keep their exact call shape.
        plan["paper_type"] = resolved_type

        # Retain upstream lineage to StructuredHypothesis
        structured_hyp = topic.get("structured_hypothesis")
        if isinstance(structured_hyp, dict):
            plan["structured_hypothesis"] = structured_hyp
            if structured_hyp.get("research_question"):
                plan["research_questions"] = [structured_hyp["research_question"]]
            if structured_hyp.get("falsification_condition"):
                plan["falsification_condition"] = structured_hyp["falsification_condition"]
            if structured_hyp.get("closest_prior_work"):
                plan["closest_prior_work"] = structured_hyp["closest_prior_work"]
            # Power analysis integration if required
            if plan.get("require_power_analysis"):
                effect = structured_hyp.get("planned_effect_size")
                alpha = structured_hyp.get("alpha", 0.05)
                target_power = structured_hyp.get("target_power", 0.8)
                if effect is not None:
                    plan["power_analysis"] = preregister_power(effect, alpha=alpha, target_power=target_power)

        plan["contributions"] = self._ensure_falsifiable_contributions(plan, topic)
        plan["experiments"] = self._generate_experiments(topic, plan)
        plan["experiments"] = self._normalize_experiments(plan["experiments"])
        logger.info("Experiments planned: %d experiments", len(plan.get("experiments", [])))

        unfalsifiable = self._flag_unfalsifiable(plan)
        missing_baselines = self._flag_missing_baselines(plan)
        # P2: try_and_report topics may proceed with a single MVE and advisory
        # baselines — missing baselines do not force a repair loop here; the
        # fallback MVE still carries a trivial baseline for grounding.
        _is_try = str((topic or {}).get("hunt_mode") or "") in ("try_and_report", "replication")
        if _is_try:
            missing_baselines = []
        if unfalsifiable or missing_baselines:
            logger.info("Plan repair needed: %d unfalsifiable, %d missing baselines",
                         len(unfalsifiable), len(missing_baselines))
            plan = self._repair_plan(plan, topic, unfalsifiable, missing_baselines)
            plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])

        plan["experiments"] = self._attach_variants(plan.get("experiments") or [])
        plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])
        plan["experiments"] = self._ensure_statistical_protocol(plan.get("experiments") or [], topic)
        # v5: exploratory handoffs stay exploratory — the plan carries the
        # scope limit so writer + release gates forbid inferential adjectives.
        if isinstance(topic, dict) and (topic.get("exploratory_no_release") or topic.get("fallback_provenance")):
            plan["exploratory_no_release"] = True
            plan["claim_scope"] = "exploratory: execution plumbing only unless real CIs are computed"
            plan["comparative_language_forbidden"] = True

        # M4 (§4.5): per-type experiment schemas. Comparative requires a
        # baseline matrix (≥2 baselines per experiment, ablation on);
        # cause_effect requires causal_spec{dag, id_strategy} on the plan.
        # `experiments: []` stays forbidden for all four experimental types
        # (validate_experiments enforces the non-empty list).
        if resolved_type == "comparative":
            plan["experiments"] = self._ensure_comparative_experiments(plan.get("experiments") or [])
        if resolved_type == "cause_effect":
            plan["causal_spec"] = self._build_causal_spec(topic, plan)
        plan["paper_type"] = resolved_type

        plan["dependencies"] = self._generate_dependencies(plan)
        plan["timeline"] = self._generate_timeline(plan)
        plan["unfalsifiable_flags"] = self._flag_unfalsifiable(plan)
        plan["missing_baseline_flags"] = self._flag_missing_baselines(plan)
        plan["revision_history"] = []

        schema_errors = validate_experiments(
            plan.get("experiments"), paper_type=resolved_type, plan=plan
        )
        if schema_errors:
            for _ in range(2):
                plan = self._repair_plan(
                    plan,
                    topic,
                    unfalsifiable=[],
                    missing_baselines=[],
                    schema_errors=schema_errors,
                )
                plan["experiments"] = self._attach_variants(plan.get("experiments") or [])
                plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])
                if resolved_type == "comparative":
                    plan["experiments"] = self._ensure_comparative_experiments(plan.get("experiments") or [])
                if resolved_type == "cause_effect" and not isinstance(plan.get("causal_spec"), dict):
                    plan["causal_spec"] = self._build_causal_spec(topic, plan)
                schema_errors = validate_experiments(
                    plan.get("experiments"), paper_type=resolved_type, plan=plan
                )
                if not schema_errors:
                    break
            if schema_errors:
                raise ValueError("Plan experiments failed contract validation: " + "; ".join(schema_errors))

        feasibility_errors = check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST)
        if feasibility_errors:
            logger.info("Capability rescoping required: %d feasibility errors", len(feasibility_errors))
            self._apply_capability_rescope(plan, topic, feasibility_errors)
            plan["methodology"] = str(plan.get("methodology", "")).replace("GPU", "CPU-compatible").replace("download", "use bundled")
            plan["compute_budget"] = "CPU-only, bounded synthetic or bundled small dataset"
            for experiment in plan.get("experiments") or []:
                if isinstance(experiment, dict):
                    dataset = experiment.setdefault("dataset", {})
                    # Word-boundary match (not bare substring): "uci" must not
                    # fire inside ordinary prose the way it did in
                    # check_plan_feasibility before the \b fix.
                    if re.search(r"\b(download|yahoo|uci|internet|wikipedia)\b", str(dataset).lower()):
                        experiment["dataset"] = {"name": "bundled_synthetic", "access_policy": "local-only"}
                    experiment.pop("gpu", None)
            # Deterministic safety net: scrub every text field (title, abstract,
            # contributions, sections, experiment methodology/descriptions) so
            # residual banned tokens cannot survive to the final gate. The same
            # helper already guards revise_plan(); create_plan must use it too.
            self._auto_fix_manifest_violations(plan, SANDBOX_CAPABILITY_MANIFEST)
            feasibility_errors = check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST)
            if feasibility_errors:
                logger.warning("Rescope + auto-fix left %d feasibility errors; will fail closed below", len(feasibility_errors))
        plan["capability_manifest"] = SANDBOX_CAPABILITY_MANIFEST.as_dict()
        # Final gate scans the full plan (title/abstract/contributions/sections
        # included) — the earlier executable_plan subset hid violations that the
        # tiered checker would still flag downstream.
        plan["feasibility_errors"] = check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST)
        if plan["feasibility_errors"]:
            # Deterministic last resort (no LLM): coerce the whole plan to the
            # Tier-0 local envelope instead of killing the run. A bounded
            # executable plan beats a terminal planning failure — the
            # rescope provenance below keeps the downgrade auditable.
            logger.warning("Final gate failed (%d errors); forcing Tier-0 local plan",
                           len(plan["feasibility_errors"]))
            self._force_tier0_plan(plan, topic, plan["feasibility_errors"])
            self._auto_fix_manifest_violations(plan, SANDBOX_CAPABILITY_MANIFEST)
            plan["feasibility_errors"] = check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST)
        if plan["feasibility_errors"]:
            # Forensics (d5b5a1db hole): the rejected plan JSON was discarded
            # here, leaving only "plan requires GPU-scale..." with no offending
            # sentence. Persist it before raising so operators can see the exact
            # token that failed the gate.
            try:
                import os as _os
                _outdir = getattr(self.runtime_config, "output_dir", "./output")
                _os.makedirs(_outdir, exist_ok=True)
                _rej_path = _os.path.join(
                    _outdir, f"plan_rejected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(_rej_path, "w", encoding="utf-8") as _f:
                    json.dump({"plan": plan, "topic": topic,
                               "feasibility_errors": plan["feasibility_errors"]},
                              _f, indent=2, default=str)
                log_agent_action("PlannerAgent", "plan_rejected_persisted", {
                    "path": _rej_path,
                    "feasibility_errors": plan["feasibility_errors"][:5],
                })
            except Exception as _e:
                logger.warning("Failed to persist rejected plan: %s", _e)
            raise ValueError("Plan rejected by sandbox capability manifest: " + "; ".join(plan["feasibility_errors"]))

        self._store_plan(plan, topic)

        # Resolve datasets for each experiment: get info, download if small
        plan["dataset_resolutions"] = self._resolve_experiment_datasets(plan)
        logger.info("Dataset requirements resolved: %d experiments mapped", len(plan.get("dataset_resolutions", [])))

        log_agent_action("PlannerAgent", "plan_created", {
            "sections": len(plan.get("sections", [])),
            "experiments": len(plan.get("experiments", [])),
            "unfalsifiable": len(plan["unfalsifiable_flags"]),
        })
        logger.info("Plan creation complete: %d sections, %d experiments, %d unfalsifiable flags",
                     len(plan.get("sections", [])),
                     len(plan.get("experiments", [])),
                     len(plan.get("unfalsifiable_flags", [])))
        return plan

    def _resolve_experiment_datasets(self, plan: Plan) -> List[Dict[str, Any]]:
        """Resolve datasets for each experiment: get metadata, download if small.

        Called during plan creation so datasets are ready before Engineer phase.
        Tier-0 local catalog loads directly; Tier-1 HuggingFace ids are
        pre-cached to memory/dataset_cache (bounded by manifest caps) and the
        experiment spec is rewritten to the sanctioned hf-cached form so the
        tiered feasibility checker and the offline sandbox both accept it.
        Returns a list of resolution records for each experiment.
        """
        resolutions = []
        for exp in plan.get("experiments") or []:
            if not isinstance(exp, dict):
                continue
            dataset = exp.get("dataset") or {}
            dataset_name = dataset.get("name", "") if isinstance(dataset, dict) else str(dataset)
            if not dataset_name:
                resolutions.append({"experiment": exp.get("name", ""), "dataset": None, "status": "no_dataset"})
                continue
            if dataset_name in ("bundled_synthetic", "synthetic"):
                resolutions.append({
                    "experiment": exp.get("name", ""),
                    "dataset": dataset_name,
                    "status": "synthetic_local",
                    "rows": 0,
                })
                if isinstance(dataset, dict):
                    exp["dataset"] = {"name": "bundled_synthetic", "access_policy": "local-only"}
                else:
                    exp["dataset"] = {"name": "bundled_synthetic", "access_policy": "local-only"}
                continue

            # Try local catalog first
            try:
                local = load_local_dataset(dataset_name)
                if local and local.get("row_count", 0) > 0:
                    resolutions.append({
                        "experiment": exp.get("name", ""),
                        "dataset": dataset_name,
                        "status": "local_loaded",
                        "rows": local["row_count"],
                    })
                    log_agent_action("PlannerAgent", "dataset_resolved_local", {
                        "experiment": exp.get("name", ""), "dataset": dataset_name, "rows": local["row_count"],
                    })
                    if isinstance(exp.get("dataset"), dict):
                        exp["dataset"].setdefault("access_policy", "local-only")
                    continue
            except Exception:
                pass

            # Get info (metadata only — no download yet)
            try:
                info = get_dataset_info(dataset_name)
            except Exception:
                info = None
            if info and info.get("source") == "huggingface":
                max_samples = min(1000, SANDBOX_CAPABILITY_MANIFEST.max_samples, SANDBOX_CAPABILITY_MANIFEST.max_dataset_rows)
                try:
                    dl = download_hf_dataset(dataset_name, split="train", max_samples=max_samples)
                except Exception:
                    dl = None
                if dl and dl.get("row_count"):
                    exp["dataset"] = {
                        "name": dataset_name,
                        "source": "huggingface",
                        "access_policy": "hf-cached",
                        "split": dl.get("split", "train"),
                        "rows": dl.get("row_count"),
                    }
                    resolutions.append({
                        "experiment": exp.get("name", ""),
                        "dataset": dataset_name,
                        "status": "hf_cached",
                        "source": "huggingface",
                        "rows": dl.get("row_count"),
                        "license": info.get("license"),
                    })
                    log_agent_action("PlannerAgent", "dataset_resolved_hf_cached", {
                        "experiment": exp.get("name", ""), "dataset": dataset_name, "rows": dl.get("row_count"),
                    })
                    continue
                resolutions.append({
                    "experiment": exp.get("name", ""),
                    "dataset": dataset_name,
                    "status": "info_resolved",
                    "source": info.get("source"),
                    "size_info": info.get("size_info"),
                    "license": info.get("license"),
                })
                log_agent_action("PlannerAgent", "dataset_resolved_info", {
                    "experiment": exp.get("name", ""), "dataset": dataset_name, "source": info.get("source"),
                })
            elif info:
                resolutions.append({
                    "experiment": exp.get("name", ""),
                    "dataset": dataset_name,
                    "status": "info_resolved",
                    "source": info.get("source"),
                    "size_info": info.get("size_info"),
                    "license": info.get("license"),
                })
                log_agent_action("PlannerAgent", "dataset_resolved_info", {
                    "experiment": exp.get("name", ""), "dataset": dataset_name, "source": info.get("source"),
                })
            else:
                resolutions.append({
                    "experiment": exp.get("name", ""),
                    "dataset": dataset_name,
                    "status": "not_found",
                })
                log_agent_action("PlannerAgent", "dataset_not_found", {
                    "experiment": exp.get("name", ""), "dataset": dataset_name,
                })

        return resolutions

    @staticmethod
    def _apply_capability_rescope(plan: Plan, topic: Topic, reasons: List[str]) -> None:
        """Rewrite publication framing when execution scope changes with full dataset rescoping traceability."""
        original_title = str(plan.get("title") or topic.get("title") or "Research study")
        original_contributions = plan.get("contributions") or plan.get("expected_contributions") or []
        replacement = "bounded local synthetic or bundled benchmark"
        title = original_title
        for token in ("Yahoo Finance", "UCI", "internet", "production", "real-world"):
            title = title.replace(token, replacement)
        if title == original_title:
            title = f"{original_title} on a {replacement}"

        rewritten = []
        for contribution in original_contributions:
            if isinstance(contribution, dict):
                item = dict(contribution)
                if item.get("claim"):
                    item["claim"] = f"{item['claim']} within a {replacement}"
                rewritten.append(item)
            else:
                rewritten.append(f"{contribution} within a {replacement}")

        plan["capability_rescope"] = {
            "from": reasons,
            "original_title": original_title,
            "to": replacement,
            "original_dataset_requirement": str((topic or {}).get("dataset_plan") or "external_dataset"),
            "replacement_dataset": "bundled_synthetic",
            "reason": "; ".join(reasons) or "External data and GPU-scale compute unavailable in sandbox; rescoped to catalogued synthetic benchmark",
            "replacement_dataset_policy": "catalogued or generated locally; no external data",
        }
        plan["title"] = title
        plan["abstract"] = (
            f"This study evaluates the proposed approach on a {replacement}. "
            "Claims are limited to the recorded local protocol and do not establish performance on the originally requested external setting."
        )
        plan["research_questions"] = [
            f"Under the bounded local protocol, does the proposed method satisfy its falsifiable prediction?"
        ]
        plan["expected_contributions"] = rewritten
        plan["contributions"] = rewritten
        if isinstance(topic, dict):
            topic["title"] = title
            topic["description"] = (
                f"Evaluate the proposed method using a {replacement}; do not claim external-domain or production validity."
            )

    @staticmethod
    def _auto_fix_manifest_violations(plan: Plan, manifest: SandboxCapabilityManifest) -> None:
        """Programmatically strip/replace known-shape manifest violations.

        This is a deterministic safety net that runs after the LLM re-prompt.
        It cannot introduce new violations — it only removes or downgrades
        non-compliant fields so the plan passes ``check_plan_feasibility``.

        HF-sanctioned phrasing (huggingface / hf-cached / memory/dataset_cache /
        bundled_synthetic / local-only) is preserved: generic "download" in that
        context is allowed by the tiered checker, so only raw tokens
        (yahoo finance / wikipedia / uci) are rewritten there.
        """
        # d5b5a1db: bare gpu|cuda scrub let "deep neural network / vllm /
        # distributed training / tpu ..." survive rescope -> Tier-0 -> final
        # gate still failed. Cover the full _GPU_METHOD_TOKENS list from
        # core/capabilities.py; replacement carries a disclaimer cue
        # (CPU-only + proxy + simulation) so _gpu_required_by_text agrees.
        GPU_TOKENS = re.compile(
            r"multi[\s\-]?gpu|distributed\s+training|large\s+language\s+model\s+fine[\s\-]?tune"
            r"|deep\s+neural\s+network|cuda\s+kernel|gpu\s+profiling|gpu\s+cluster"
            r"|\bgpu\b|\bcuda\b|\btpu\b|\bnccl\b|\btensorrt\b|\bvllm\b",
            re.IGNORECASE,
        )
        GPU_REPLACEMENT = "CPU-only proxy simulation (small sklearn baseline)"
        RAW_TOKENS = re.compile(r"\b(yahoo\s*finance|wikipedia|uci)\b", re.IGNORECASE)
        DOWNLOAD_TOKENS = re.compile(r"\b(download|internet|outbound|yahoo\s*finance|wikipedia|uci)\b", re.IGNORECASE)
        SANCTIONED_HINT = re.compile(r"huggingface|hf-cached|hf_cached|dataset_cache|bundled_synthetic|local-only|sklearn", re.IGNORECASE)

        def _sanitize(text: str) -> str:
            if not isinstance(text, str) or not text:
                return text
            out = text
            if not manifest.gpu_available:
                out = GPU_TOKENS.sub(GPU_REPLACEMENT, out)
            if not manifest.dataset_downloads:
                if SANCTIONED_HINT.search(out):
                    out = RAW_TOKENS.sub("bundled synthetic", out)
                else:
                    out = DOWNLOAD_TOKENS.sub("bundled synthetic", out)
            return out

        # --- Fix methodology-level violations ---
        methodology = str(plan.get("methodology") or "")
        if not manifest.gpu_available:
            plan["methodology"] = GPU_TOKENS.sub(GPU_REPLACEMENT, methodology)
        if not manifest.dataset_downloads:
            plan["methodology"] = _sanitize(plan.get("methodology", ""))
        for key in ("abstract", "compute_budget", "title"):
            if isinstance(plan.get(key), str):
                plan[key] = _sanitize(plan[key])
        for key in ("research_questions",):
            vals = plan.get(key)
            if isinstance(vals, list):
                plan[key] = [_sanitize(v) if isinstance(v, str) else v for v in vals]
        for key in ("contributions", "expected_contributions"):
            items = plan.get(key)
            if isinstance(items, list):
                fixed = []
                for item in items:
                    if isinstance(item, dict):
                        item = dict(item)
                        for fk in ("claim", "falsifiable_prediction", "statistical_test", "description"):
                            if isinstance(item.get(fk), str):
                                item[fk] = _sanitize(item[fk])
                        fixed.append(item)
                    elif isinstance(item, str):
                        fixed.append(_sanitize(item))
                    else:
                        fixed.append(item)
                plan[key] = fixed
        for section in plan.get("sections") or []:
            if not isinstance(section, dict):
                continue
            for fk in ("content_requirements", "description", "name"):
                if isinstance(section.get(fk), str):
                    section[fk] = _sanitize(section[fk])
            for fk in ("key_points",):
                if isinstance(section.get(fk), list):
                    section[fk] = [_sanitize(v) if isinstance(v, str) else v for v in section[fk]]

        # e0ec9e91: structured_hypothesis (gap_report/novelty_report/
        # literature abstracts like "GPU-oriented transformer") survives Tier-0
        # and fails the final gate. Recursively sanitize it too.
        def _sanitize_obj(obj: object) -> object:
            if isinstance(obj, str):
                return _sanitize(obj)
            if isinstance(obj, list):
                return [_sanitize_obj(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _sanitize_obj(v) for k, v in obj.items()}
            return obj

        for _hkey in ("structured_hypothesis", "falsification_condition",
                      "research_questions", "closest_prior_work"):
            if _hkey in plan and plan[_hkey] is not None:
                plan[_hkey] = _sanitize_obj(plan[_hkey])

        # --- Fix per-experiment violations ---
        for experiment in plan.get("experiments") or []:
            if not isinstance(experiment, dict):
                continue
            # GPU references
            if not manifest.gpu_available:
                for key in ("gpu", "cuda", "accelerator"):
                    experiment.pop(key, None)
                for fk in ("methodology", "description"):
                    if isinstance(experiment.get(fk), str):
                        experiment[fk] = _sanitize(experiment[fk])
            elif not manifest.dataset_downloads:
                for fk in ("methodology", "description"):
                    if isinstance(experiment.get(fk), str):
                        experiment[fk] = _sanitize(experiment[fk])
            # Dataset access — preserve sanctioned HF-brokered specs
            if not manifest.dataset_downloads:
                dataset = experiment.get("dataset") or {}
                if isinstance(dataset, dict):
                    ds_text = str(dataset).lower()
                    access = str(dataset.get("access_policy", "")).lower()
                    name = str(dataset.get("name", "")).lower()
                    sanctioned = (
                        access in ("local-only", "hf-cached")
                        or "huggingface" in ds_text
                        or "hf-cached" in ds_text
                        or "dataset_cache" in ds_text
                        or name in ("bundled_synthetic", "synthetic", "sklearn_iris", "sklearn_digits", "openreview_calibration")
                    )
                    if not sanctioned and any(tok in ds_text for tok in ("download", "internet", "outbound", "yahoo", "uci", "wikipedia")):
                        experiment["dataset"] = {"name": "bundled_synthetic", "access_policy": "local-only"}
            # Cap epochs
            for key in ("epochs", "training_epochs"):
                val = experiment.get(key)
                if isinstance(val, (int, float)) and val > manifest.max_training_epochs:
                    experiment[key] = manifest.max_training_epochs
            # Cap samples
            for key in ("samples", "n_samples"):
                val = experiment.get(key)
                if isinstance(val, (int, float)) and val > manifest.max_samples:
                    experiment[key] = manifest.max_samples

            # Fix variants too
            for variant in experiment.get("variants") or []:
                if not isinstance(variant, dict):
                    continue
                if not manifest.gpu_available:
                    for key in ("gpu", "cuda", "accelerator"):
                        variant.pop(key, None)
                for fk in ("methodology", "description"):
                    if isinstance(variant.get(fk), str):
                        variant[fk] = _sanitize(variant[fk])
                if not manifest.dataset_downloads:
                    vds = variant.get("dataset") or {}
                    if isinstance(vds, dict):
                        vds_text = str(vds).lower()
                        v_access = str(vds.get("access_policy", "")).lower()
                        v_sanctioned = (
                            v_access in ("local-only", "hf-cached")
                            or "huggingface" in vds_text
                            or "hf-cached" in vds_text
                            or "dataset_cache" in vds_text
                        )
                        if not v_sanctioned and re.search(r"\b(download|internet|outbound|yahoo|uci|wikipedia)\b", vds_text):
                            variant["dataset"] = {"name": "bundled_synthetic", "access_policy": "local-only"}

    def _force_tier0_plan(self, plan: Plan, topic: Topic, reasons: List[str]) -> None:
        """Deterministically coerce a plan into the Tier-0 local envelope.

        No LLM calls: rewrites methodology/compute budget to CPU-only local
        execution and pins every experiment (and variant) dataset to
        bundled_synthetic/local-only. Records provenance so supervision can
        see the plan was downgraded, not originally scoped this way.
        """
        plan["methodology"] = (
            "Deterministic bundled-synthetic baseline comparison with controlled seeds."
        )
        plan["compute_budget"] = "CPU-only, bounded synthetic local dataset"
        plan["dataset_catalog_choice"] = "bundled_synthetic"
        # d5b5a1db: Tier-0 rewrote methodology/compute/dataset only, leaving
        # "deep neural network / vllm / ..." in contributions/sections/
        # experiment methodology to fail the final gate. Sanitize those too.
        _tier0_sanitize = re.compile(
            r"multi[\s\-]?gpu|distributed\s+training|large\s+language\s+model\s+fine[\s\-]?tune"
            r"|deep\s+neural\s+network|cuda\s+kernel|gpu\s+profiling|gpu\s+cluster"
            r"|\bgpu\b|\bcuda\b|\btpu\b|\bnccl\b|\btensorrt\b|\bvllm\b",
            re.IGNORECASE,
        )
        _tier0_repl = "CPU-only proxy simulation (small sklearn baseline)"

        def _tier0_sub(obj: object) -> object:
            if isinstance(obj, str):
                return _tier0_sanitize.sub(_tier0_repl, obj)
            if isinstance(obj, list):
                return [_tier0_sub(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _tier0_sub(v) for k, v in obj.items()}
            return obj

        for _key in ("abstract", "title", "structured_hypothesis",
                     "falsification_condition", "research_questions",
                     "closest_prior_work"):
            if _key in plan and plan[_key] is not None:
                plan[_key] = _tier0_sub(plan[_key])
        for _key in ("abstract", "title"):
            if isinstance(plan.get(_key), str):
                plan[_key] = _tier0_sanitize.sub(_tier0_repl, plan[_key])
        for _ckey in ("contributions", "expected_contributions"):
            for _item in plan.get(_ckey) or []:
                if isinstance(_item, dict):
                    for _fk in ("claim", "falsifiable_prediction", "statistical_test", "description"):
                        if isinstance(_item.get(_fk), str):
                            _item[_fk] = _tier0_sanitize.sub(_tier0_repl, _item[_fk])
                elif isinstance(_item, str):
                    pass
        for _sec in plan.get("sections") or []:
            if isinstance(_sec, dict):
                for _fk in ("content_requirements", "description"):
                    if isinstance(_sec.get(_fk), str):
                        _sec[_fk] = _tier0_sanitize.sub(_tier0_repl, _sec[_fk])
        experiments = plan.get("experiments") or []
        if not experiments:
            experiments = [{"name": "minimum_viable_experiment"}]
            plan["experiments"] = experiments
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            experiment["dataset"] = {"name": "bundled_synthetic", "access_policy": "local-only"}
            for key in ("gpu", "cuda", "accelerator"):
                experiment.pop(key, None)
            for fk in ("methodology", "description"):
                if isinstance(experiment.get(fk), str):
                    experiment[fk] = _tier0_sanitize.sub(_tier0_repl, experiment[fk])
            # Tier-0 must stay schema-valid: guarantee name/metrics/baselines
            # so validate_experiments doesn't reject the coerced plan.
            experiment.setdefault("name", "minimum_viable_experiment")
            experiment.setdefault("evaluation_metrics", ["accuracy", "seed_variance"])
            experiment.setdefault("baselines", ["majority_class"])
            experiment.setdefault("seeds", [0, 1, 2])
            for variant in experiment.get("variants") or []:
                if isinstance(variant, dict):
                    variant["dataset"] = {"name": "bundled_synthetic", "access_policy": "local-only"}
                    for key in ("gpu", "cuda", "accelerator"):
                        variant.pop(key, None)
        history = plan.setdefault("revision_history", [])
        if isinstance(history, list):
            history.append({
                "reason": "tier0_forced_local_coercion",
                "from_errors": list(reasons),
                "at": datetime.now().isoformat(),
            })
        plan["tier0_coerced"] = True

    def revise_plan(
        self,
        plan: Plan,
        revision_request: RevisionRequest,
        topic: Optional[Topic] = None,
    ) -> Plan:
        """Bidirectional edge: Engineer requested a plan revision.

        The revision LLM prompt is seeded with the active capability manifest
        so the model cannot propose GPU / outbound / undeclared-library work.
        A post-generation feasibility gate mirrors create_plan enforcement; on
        first violation the LLM is re-prompted with the specific constraints.
        A second violation raises ``planner_manifest_violation`` (distinct from
        the generic sandbox-rejection message) so operators can immediately
        distinguish "Planner ignored constraints" from "task infeasible".
        """
        log_agent_action("PlannerAgent", "revise_plan", revision_request)
        logger.info("Revision requested: reason=%s, experiment=%s",
                     revision_request.get("reason", "N/A"),
                     revision_request.get("experiment", "N/A"))
        tracker = get_tracker()
        if tracker:
            tracker.bump("plan_revisions")
            tracker.scratch("PlannerAgent", "revision", revision_request)

        manifest = plan.get("capability_manifest") or SANDBOX_CAPABILITY_MANIFEST.as_dict()
        manifest_summary = (
            f"Capability manifest: gpu_available={manifest.get('gpu_available', False)}, "
            f"outbound_network={manifest.get('outbound_network', False)}, "
            f"dataset_downloads={manifest.get('dataset_downloads', False)}, "
            f"available_libraries={manifest.get('available_libraries', [])}"
        )

        prompt = f"""
Revise this research plan based on an Engineer failure or validation request.

Reason: {revision_request.get('reason')}
Detail: {revision_request.get('detail')}
Failed experiment: {revision_request.get('experiment')}

Current plan JSON (compact — only the failing experiment plus plan head):
{json.dumps({"title": plan.get("title"), "methodology": str(plan.get("methodology") or "")[:800], "experiments": [e for e in (plan.get("experiments") or []) if isinstance(e, dict) and e.get("name") == revision_request.get("experiment")] or (plan.get("experiments") or [])[:1]}, indent=2)[:2500]}

{manifest_summary}

Requirements:
- Fix unsupported assumptions / APIs / data requirements
- You MUST respect the capability manifest above. Do NOT propose GPU, CUDA, outbound network access, dataset downloads, or libraries not in available_libraries.
- You MUST explicitly import every sklearn/scipy symbol used in any methodology snippet (e.g. from sklearn.datasets import make_classification).
- Every contribution needs falsifiable_prediction + statistical_test
- Every experiment MUST be an object with "name", "baselines", "evaluation_metrics", "falsifiable_prediction", "statistical_test"
- Every item in "variants" MUST be a JSON object with at least "name" and "methodology" (e.g. [{{"name": "Variant Name", "methodology": "..."}}]), NOT strings.
- Keep the returned fragment compact (<4000 chars).
Return full updated plan fragment as JSON with keys: methodology, contributions, experiments, revision_notes
"""
        # e0ec9e91: full-plan [:6000] dump at default 8192 tokens truncated
        # every revision under load. Compact prompt + 16384 + json_mode.
        response = call_llm(prompt, temperature=0.4, tier="strong",
                            max_tokens=16384, retry_on_truncation=False,
                            **_planner_system("planner.repair"))
        logger.info("Revision LLM response received, parsing and validating")
        schema_errors: List[str] = []
        for attempt in range(3):
            parsed = parse_json_from_llm(response) or {}
            if isinstance(parsed, dict):
                for key in ("methodology", "contributions", "experiments"):
                    if key in parsed:
                        plan[key] = parsed[key]
                if "experiments" in parsed:
                    plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])
                    plan["experiments"] = self._attach_variants(plan.get("experiments") or [])
                    plan["experiments"] = self._normalize_experiments(plan["experiments"])
                    # M4 (§4.5): revised plans keep their per-type schema.
                    _rev_type = str(plan.get("paper_type") or "")
                    if _rev_type == "comparative":
                        plan["experiments"] = self._ensure_comparative_experiments(plan["experiments"])
                    if _rev_type == "cause_effect" and not isinstance(plan.get("causal_spec"), dict):
                        plan["causal_spec"] = self._build_causal_spec(topic, plan)
                schema_errors = validate_experiments(
                    plan.get("experiments"),
                    paper_type=str(plan.get("paper_type") or ""),
                    plan=plan,
                )
                if not schema_errors:
                    break
                retry_prompt = f"""
Previous revision had schema errors: {'; '.join(schema_errors)}
Please fix the schema errors.
Every experiment MUST be an object with "name", "baselines", "evaluation_metrics", "falsifiable_prediction", "statistical_test".
Every variant in "variants" MUST be an object with "name" and "methodology" (e.g. [{{"name": "Variant A", "methodology": "..."}}]).

Original revision request: {revision_request.get('detail')}
Return JSON with keys: methodology, contributions, experiments, revision_notes
"""
                response = call_llm(retry_prompt, temperature=0.3, tier="strong",
                                    **_planner_system("planner.repair"))
        if schema_errors:
            raise ValueError("Revised plan failed contract validation: " + "; ".join(schema_errors))

        feasibility_errors = check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST)
        if feasibility_errors:
            # Re-prompt once with explicit constraint violations before failing closed.
            logger.info("Revised plan has %d feasibility violations, re-prompting LLM", len(feasibility_errors))
            constraint_detail = "; ".join(feasibility_errors)
            fix_prompt = f"""
Your revised plan violated the capability manifest:
{constraint_detail}

{manifest_summary}

Rewrite ONLY the violating fields to comply. Do NOT use GPU, CUDA, outbound network, dataset downloads, or unlisted libraries.
Keep the same experiment structure and schema. Return JSON with keys: methodology, contributions, experiments, revision_notes
"""
            fix_response = call_llm(fix_prompt, temperature=0.3, tier="strong",
                                    **_planner_system("planner.repair"))
            fix_parsed = parse_json_from_llm(fix_response) or {}
            if isinstance(fix_parsed, dict):
                for key in ("methodology", "contributions", "experiments"):
                    if key in fix_parsed:
                        plan[key] = fix_parsed[key]
                if "experiments" in fix_parsed:
                    plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])
                    plan["experiments"] = self._attach_variants(plan.get("experiments") or [])
                    plan["experiments"] = self._normalize_experiments(plan["experiments"])

            # Second feasibility check — fail closed with distinct error code
            feasibility_errors = check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST)
            if feasibility_errors:
                # Deterministic auto-repair: programmatically fix known violation shapes
                # instead of relying solely on the LLM to self-correct.
                self._auto_fix_manifest_violations(plan, SANDBOX_CAPABILITY_MANIFEST)
                feasibility_errors = check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST)
                if feasibility_errors:
                    raise ValueError(
                        "planner_manifest_violation: Revised plan still violates capability manifest after re-prompt and auto-repair: "
                        + "; ".join(feasibility_errors)
                    )

        plan.setdefault("revision_history", []).append({
            "request": revision_request,
            "notes": (parse_json_from_llm(response) or {}).get("revision_notes", ""),
            "at": datetime.now().isoformat(),
        })
        CrossRunMemory().record_plan_revision(
            revision_request.get("reason", "revise"),
            meta={"experiment": revision_request.get("experiment")},
        )
        logger.info("Plan revision complete: %d experiments, %d revision history entries",
                     len(plan.get("experiments", [])),
                     len(plan.get("revision_history", [])))
        return plan

    def _flag_unfalsifiable(self, plan: Plan) -> List[str]:
        flags = []
        contribs = plan.get("contributions") or []
        if not contribs and plan.get("expected_contributions"):
            for c in plan["expected_contributions"]:
                flags.append(f"Contribution lacks falsifiable prediction: {c}")
            return flags
        for c in contribs:
            if isinstance(c, str):
                flags.append(f"Contribution lacks falsifiable prediction: {c}")
                continue
            if not c.get("falsifiable_prediction"):
                flags.append(f"Missing falsifiable_prediction: {c.get('claim', c)}")
            if not c.get("statistical_test"):
                flags.append(f"Missing statistical_test: {c.get('claim', c)}")
        return flags

    def _flag_missing_baselines(self, plan: Plan) -> List[str]:
        flags = []
        for exp in plan.get("experiments") or []:
            baselines = exp.get("baselines") or []
            legacy = exp.get("baseline_comparison") or ""
            if not baselines and not str(legacy).strip():
                flags.append(f"Experiment '{exp.get('name')}' has no baselines")
            elif not baselines and legacy:
                # normalize later
                pass
        return flags

    def _ensure_falsifiable_contributions(self, plan: Plan, topic: Topic) -> List[Contribution]:
        existing = plan.get("contributions")
        if existing and isinstance(existing, list) and existing and isinstance(existing[0], dict):
            return existing
        claims = plan.get("expected_contributions") or [f"Advance {topic.get('title')}"]
        prompt = f"""
For each contribution claim, produce a falsifiable prediction and statistical test.

Claims: {json.dumps(claims)}
Topic: {topic.get('title')}

Return JSON array:
[{{"claim": "...", "falsifiable_prediction": "...", "statistical_test": "e.g. Welch t-test p<0.05 on accuracy vs baseline", "components": ["component_a"]}}]
"""
        parsed = parse_json_from_llm(call_llm(
            prompt, temperature=0.3, tier="strong",
            **_planner_system("planner.falsifiability")))
        if isinstance(parsed, list) and parsed:
            return parsed
        return [
            {
                "claim": c,
                "falsifiable_prediction": f"Method outperforms named baseline on primary metric",
                "statistical_test": "Welch t-test on metric across >=3 seeds, p<0.05",
                "components": ["main_method"],
            }
            for c in claims
        ]

    def _generate_plan_structure(self, topic: Topic, lessons: str) -> Plan:
        manifest = SANDBOX_CAPABILITY_MANIFEST.as_dict()
        prompt = f"""
Create a detailed research plan.

Topic: {topic.get('title')}
Description: {topic.get('description')}
Rationale: {topic.get('rationale', 'N/A')}
Feasibility: {topic.get('feasibility', 5)}/10

Capability manifest (HARD CONSTRAINTS — violating plans are rejected):
gpu_available={manifest.get('gpu_available')}, outbound_network={manifest.get('outbound_network')}, dataset_downloads={manifest.get('dataset_downloads')},
available_libraries={manifest.get('available_libraries')}, max_dataset_rows={manifest.get('max_dataset_rows')}, max_samples={manifest.get('max_samples')}.
Allowed data sources ONLY: local catalog (bundled_synthetic, sklearn_iris, sklearn_digits, openreview_calibration with access_policy local-only)
or HuggingFace-brokered (source huggingface, access_policy hf-cached, cached to memory/dataset_cache by the system — write e.g. {{"name": "squad", "source": "huggingface", "access_policy": "hf-cached"}}).
FORBIDDEN: Yahoo Finance, Wikipedia traffic, UCI downloads, bare "download from internet", "outbound", GPU/CUDA, unlisted libraries.
Do NOT emit the words download/internet/outbound except inside an hf-cached dataset spec.

Prior-run lessons:
{lessons}

Return JSON:
{{
  "title": "...",
  "sections": [{{"name": "Abstract", "content_requirements": "...", "key_points": [], "expected_length": "...", "dependencies": []}}],
  "research_questions": [],
  "methodology": "...",
  "expected_contributions": [],
  "compute_budget": "CPU-only sklearn-scale synthetic or small public datasets",
    "dataset_availability": "synthetic or clearly named public dataset",
    "dataset_catalog_choice": "bundled_synthetic|sklearn_iris|sklearn_digits"
}}
Include standard sections: Abstract, Introduction, Related Work, Methods, Experiments, Results, Discussion, Limitations, Conclusion.
"""
        try:
            parsed = parse_json_from_llm(call_llm(
                prompt, temperature=0.5, tier="strong",
                **_planner_system("planner.generate")))
            if isinstance(parsed, dict) and "sections" in parsed:
                parsed["created_at"] = datetime.now().isoformat()
                parsed["topic"] = topic.get("title")
                parsed["domain"] = self.runtime_config.research_domain
                return parsed
        except Exception as e:
            log_agent_action("PlannerAgent", "plan_generation_error", {"error": str(e)})
        return self._create_fallback_plan(topic)

    def _generate_experiments(self, topic: Topic, plan: Plan) -> List[ExperimentSpec]:
        contribs = plan.get("contributions") or []
        manifest = SANDBOX_CAPABILITY_MANIFEST.as_dict()
        # M4 (§4.5): type-aware experiment briefs read from plan["paper_type"]
        # (carried on the plan so the two-arg call shape is preserved).
        paper_type = str(plan.get("paper_type") or "experimental")
        # M4 (§4.5): type-aware experiment briefs. Comparative plans need a
        # baseline matrix (≥2 named baselines per experiment); cause-effect
        # plans need treatment/outcome/confounder framing for the DAG.
        type_brief = ""
        if paper_type == "comparative":
            type_brief = (
                "COMPARATIVE type: every experiment MUST name >=2 baselines "
                "(the comparison matrix, e.g. a strong model baseline plus a "
                "trivial baseline) and declare claimed_components for the "
                "compulsory ablation.\n"
            )
        elif paper_type == "cause_effect":
            type_brief = (
                "CAUSE-EFFECT type: every experiment MUST state the treatment, "
                "the outcome, and the identification strategy (randomized "
                "assignment, quasi-experimental design, or synthetic-DAG "
                "benchmark) in its methodology. Naive observational comparison "
                "without an identification strategy will be downgraded.\n"
            )
        prompt = f"""
Design experiments for:
Topic: {topic.get('title')}
Methodology: {plan.get('methodology')}
Contributions: {json.dumps(contribs)[:3000]}
Compute budget: {plan.get('compute_budget')}

Capability manifest (HARD CONSTRAINTS): gpu_available={manifest.get('gpu_available')}, outbound_network={manifest.get('outbound_network')}, dataset_downloads={manifest.get('dataset_downloads')}, available_libraries={manifest.get('available_libraries')}.
Allowed datasets ONLY: {{"name": "bundled_synthetic|sklearn_iris|sklearn_digits|openreview_calibration", "access_policy": "local-only"}} or {{"name": "<hf-dataset-id>", "source": "huggingface", "access_policy": "hf-cached"}} (system caches to memory/dataset_cache; max_samples<={manifest.get('max_samples')}).
FORBIDDEN dataset text: Yahoo Finance, Wikipedia traffic, UCI, "download from internet", "outbound". Do NOT write those words outside an hf-cached spec. CPU-only, no GPU/CUDA.
{type_brief}
Each experiment MUST include:
- name: short non-empty experiment name (REQUIRED key is "name", never "experiment_name")
- baselines: list of real comparison methods (not empty)
- falsifiable_prediction
- statistical_test
- variants: list of 2-3 alternative design objects, e.g. [{{"name": "Variant Name", "methodology": "..."}}] (MUST be objects with "name", NOT plain strings)
- claimed_components: for ablations
- evaluation_metrics

Return JSON array of experiments. Example object keys:
{{"name": "Baseline Comparison", "baselines": ["logistic_regression"], "evaluation_metrics": ["accuracy"],
  "falsifiable_prediction": "...", "statistical_test": "...", "variants": [{{"name": "Simpler Model", "methodology": "Linear baseline"}}], "claimed_components": []}}
"""
        try:
            parsed = parse_json_from_llm(call_llm(
                prompt, temperature=0.5, tier="strong",
                **_planner_system("planner.experiments")))
            if isinstance(parsed, list) and parsed:
                for exp in parsed:
                    if not exp.get("baselines") and exp.get("baseline_comparison"):
                        exp["baselines"] = [exp["baseline_comparison"]]
                return self._normalize_experiments(parsed)
        except Exception as e:
            log_agent_action("PlannerAgent", "experiment_generation_error", {"error": str(e)})
        return self._create_fallback_experiments(topic)

    @staticmethod
    def _normalize_experiment_name(experiment: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        """Align LLM aliases onto the ExperimentSpec contract field `name`."""
        exp = dict(experiment)
        name = exp.get("name")
        if not (isinstance(name, str) and name.strip()):
            for alias in ("experiment_name", "title", "experiment", "id", "variant_name", "variant"):
                alt = exp.get(alias)
                if isinstance(alt, str) and alt.strip():
                    exp["name"] = alt.strip()
                    break
        if isinstance(exp.get("name"), str):
            exp["name"] = exp["name"].strip()
        if not exp.get("name"):
            exp["name"] = f"experiment_{index + 1}"
        # Drop alias so downstream validators and contracts see one canonical key.
        exp.pop("experiment_name", None)
        exp.pop("variant_name", None)
        return exp

    @staticmethod
    def _normalize_experiments(experiments: List[ExperimentSpec]) -> List[ExperimentSpec]:
        normalized: List[ExperimentSpec] = []
        for index, experiment in enumerate(experiments or []):
            if not isinstance(experiment, dict):
                continue
            exp = PlannerAgent._normalize_experiment_name(experiment, index)
            for key in ("variants", "alternatives"):
                variants = exp.get(key)
                if not isinstance(variants, list):
                    continue
                fixed = []
                for variant_index, variant in enumerate(variants):
                    if isinstance(variant, dict):
                        norm_var = PlannerAgent._normalize_experiment_name(variant, variant_index)
                        if not norm_var.get("name"):
                            norm_var["name"] = f"{exp.get('name')}_variant_{variant_index + 1}"
                        if not norm_var.get("methodology") and not norm_var.get("description"):
                            norm_var["methodology"] = f"{exp.get('methodology', '')} (variant {variant_index + 1})".strip()
                        fixed.append(norm_var)
                    elif isinstance(variant, str) and variant.strip():
                        var_text = variant.strip()
                        fixed.append({
                            "name": f"{exp.get('name')}_variant_{variant_index + 1}",
                            "description": var_text,
                            "methodology": var_text,
                        })
                    elif variant is not None:
                        var_text = str(variant).strip()
                        fixed.append({
                            "name": f"{exp.get('name')}_variant_{variant_index + 1}",
                            "description": var_text,
                            "methodology": var_text,
                        })
                exp[key] = fixed
            normalized.append(exp)
        return normalized

    def _attach_variants(self, experiments: List[ExperimentSpec]) -> List[ExperimentSpec]:
        experiments = self._normalize_experiments(experiments)
        for exp in experiments:
            if not exp.get("variants"):
                # e0ec9e91: invented variant_a/b filler ("same methodology +
                # (simpler model)") was probed by the engineer as real
                # candidates, burning 2 LLM codegen calls per experiment on
                # identical code. No filler: the engineer branches on real
                # variants only; single-design experiments run directly.
                exp["variants"] = []
            else:
                # Ensure existing variants in the list are all valid dict objects
                exp["variants"] = [
                    v if isinstance(v, dict) else {"name": f"{exp.get('name')}_variant_{i+1}", "methodology": str(v), "description": str(v)}
                    for i, v in enumerate(exp["variants"])
                ]
            if not exp.get("baselines"):
                exp["baselines"] = ["logistic_regression", "random_guess"]
            if not exp.get("evaluation_metrics"):
                exp["evaluation_metrics"] = ["accuracy"]
            if not exp.get("falsifiable_prediction"):
                exp["falsifiable_prediction"] = "Proposed method mean metric > best baseline mean across seeds"
            if not exp.get("statistical_test"):
                exp["statistical_test"] = "Welch t-test, p<0.05, n_seeds>=3"
        return self._normalize_experiments(experiments)

    def _ensure_comparative_experiments(
        self, experiments: List[ExperimentSpec]
    ) -> List[ExperimentSpec]:
        """Enforce the comparative per-type schema (§4.5, M4).

        Every experiment names ≥2 baselines (the comparison matrix) and
        carries ``ablation_required=True`` (ablation=on, compulsory in the
        Engineer + release gate). Missing second baselines are padded from
        the deterministic fallback pool — the planner's job is a complete
        matrix, and padding is visible in the plan (never silent at the
        gate: ``validate_comparative_plan`` still refuses matrices <2).
        """
        from core.evidence_gate import COMPARATIVE_MIN_BASELINES

        fallback_pool = ["random_guess", "majority_class", "logistic_regression", "random_forest"]
        for exp in experiments:
            if not isinstance(exp, dict):
                continue
            baselines = exp.get("baselines") or []
            if isinstance(baselines, str):
                baselines = [baselines]
            baselines = [str(item) for item in baselines if str(item).strip()]
            legacy = str(exp.get("baseline_comparison") or "").strip()
            if legacy and legacy not in baselines:
                baselines.append(legacy)
            for candidate in fallback_pool:
                if len(baselines) >= COMPARATIVE_MIN_BASELINES:
                    break
                if candidate not in baselines:
                    baselines.append(candidate)
            exp["baselines"] = baselines
            exp["ablation_required"] = True
        return self._normalize_experiments(experiments)

    @staticmethod
    def _is_statistical_topic(topic: Topic) -> bool:
        import re as _re

        text = " ".join(
            str((topic or {}).get(k) or "") for k in ("title", "description", "rationale")
        )
        return bool(_re.search(
            r"bootstrap|confidence interval|\bCI\b|hypothesis test|t-test|t test|p-value|accuracy diff",
            text, _re.IGNORECASE))

    def _ensure_statistical_protocol(
        self, experiments: List[ExperimentSpec], topic: Topic
    ) -> List[ExperimentSpec]:
        """Deterministic bootstrap-CI protocol (v5: f537f638 had none).

        Statistical topics must carry difference CIs computed in code —
        never prose adjectives. Pads (visibly, in-plan): difference metrics,
        bootstrap statistical_test, analysis_plan{B>=1000, unit=client,
        n_clients>=20, n_seeds>=10, coverage_sim}, >=2 baselines, and a
        CI-framed falsifiable prediction. No LLM: padding is auditable.
        """
        if not self._is_statistical_topic(topic):
            return experiments
        pool = ["majority_class", "random_guess", "logistic_regression"]
        for exp in experiments:
            if not isinstance(exp, dict):
                continue
            metrics = list(exp.get("evaluation_metrics") or [])
            for required in ("accuracy_difference", "ci_low", "ci_high"):
                if required not in metrics:
                    metrics.append(required)
            exp["evaluation_metrics"] = metrics
            test = str(exp.get("statistical_test") or "")
            if "bootstrap" not in test.lower():
                exp["statistical_test"] = (
                    (test + "; " if test else "")
                    + "bootstrap percentile + BCa over clients (B>=1000) vs Welch t-test baseline"
                )
            plan_block = dict(exp.get("analysis_plan") or {})
            plan_block.setdefault("bootstrap_B", 1000)
            plan_block.setdefault("resampling_unit", "client")
            plan_block.setdefault("n_clients", 20)
            plan_block.setdefault("n_seeds", 10)
            plan_block.setdefault("coverage_simulation", True)
            plan_block.setdefault("report", ["mean", "std", "ci_low", "ci_high", "includes_zero_rate", "width_ratio_vs_ttest"])
            exp["analysis_plan"] = plan_block
            baselines = exp.get("baselines") or []
            if isinstance(baselines, str):
                baselines = [baselines]
            baselines = [str(b) for b in baselines if str(b).strip()]
            for candidate in pool:
                if len(baselines) >= 2:
                    break
                if candidate not in baselines:
                    baselines.append(candidate)
            exp["baselines"] = baselines
            pred = str(exp.get("falsifiable_prediction") or "")
            if "ci" not in pred.lower() and "zero" not in pred.lower():
                exp["falsifiable_prediction"] = (
                    (pred + " " if pred else "")
                    + "Falsifiable: the 95% bootstrap CI for the accuracy difference includes zero "
                    "more often than reported point gaps imply."
                )
        return self._normalize_experiments(experiments)

    def _build_causal_spec(self, topic: Topic, plan: Plan) -> Dict[str, Any]:
        """Mint the cause-effect per-type schema (§4.5, M4).

        ``causal_spec = {dag{treatment, outcome, confounders[]},
        id_strategy}``. The DAG skeleton is deterministic (treatment/outcome
        from the research questions, confounders empty unless the topic
        carries them); the strategy is classified from the plan methodology.
        A stub the methodology cannot support classifies ``naive`` — the
        causal audit then honestly downgrades to analytical (§4.7) instead
        of passing silently.
        """
        from agents.causal_agent import classify_id as _classify_id

        topic = topic if isinstance(topic, dict) else {}
        questions = plan.get("research_questions") or []
        first_question = str(questions[0]) if questions else str(topic.get("title") or "")
        carried = topic.get("dag") if isinstance(topic.get("dag"), dict) else {}
        carried_spec = topic.get("causal_spec") if isinstance(topic.get("causal_spec"), dict) else {}
        dag = dict(carried.get("dag") if isinstance(carried.get("dag"), dict) else carried_spec.get("dag") or {})
        if not dag:
            dag = {
                "treatment": str(topic.get("treatment") or "intervention"),
                "outcome": str(topic.get("outcome") or first_question or "outcome"),
                "confounders": list(topic.get("confounders") or []),
            }
        methodology = str(plan.get("methodology") or "")
        experiments_text = " ".join(
            str(exp.get("methodology") or "") for exp in (plan.get("experiments") or []) if isinstance(exp, dict)
        )
        id_strategy = str(
            carried_spec.get("id_strategy") or _classify_id(
                {"dag": dag, "methodology": methodology, "experiments_text": experiments_text}
            ).get("strategy") or "naive"
        )
        return {
            "dag": dag,
            "id_strategy": id_strategy,
            "research_question": first_question,
        }

    def _repair_plan(self, plan, topic, unfalsifiable=None, missing_baselines=None, schema_errors=None):
        unfalsifiable = unfalsifiable or []
        missing_baselines = missing_baselines or []
        schema_errors = schema_errors or []
        prompt = f"""
Repair this plan to satisfy all validation requirements.
Schema errors: {schema_errors}
Unfalsifiable flags: {unfalsifiable}
Missing baselines: {missing_baselines}

Requirements:
- Every experiment MUST be an object with non-empty "name", "baselines", "evaluation_metrics", "falsifiable_prediction", "statistical_test"
- Every item in "variants" MUST be a JSON object with at least "name" and "methodology" (e.g. [{{"name": "Variant Name", "methodology": "..."}}]), NOT strings or flat lists.

Plan fragment: {json.dumps({'contributions': plan.get('contributions'), 'experiments': plan.get('experiments')}, default=str)[:5000]}
Return JSON {{"contributions": [...], "experiments": [...]}}
"""
        parsed = parse_json_from_llm(call_llm(
            prompt, temperature=0.3, tier="strong",
            **_planner_system("planner.repair")))
        if isinstance(parsed, dict):
            plan.update({k: parsed[k] for k in ("contributions", "experiments") if k in parsed})
            plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])
            plan["experiments"] = self._attach_variants(plan.get("experiments") or [])
            plan["experiments"] = self._normalize_experiments(plan.get("experiments") or [])
        return plan

    def _generate_dependencies(self, plan: Plan) -> List[Dependency]:
        deps = []
        for section in plan.get("sections") or []:
            if section.get("name") in ("Methods", "Experiments", "Results"):
                deps.append({
                    "from": "Introduction",
                    "to": section["name"],
                    "type": "content_dependency",
                    "description": f'{section["name"]} builds on Introduction',
                })
        return deps

    def _generate_timeline(self, plan: Plan) -> Timeline:
        return {
            "phases": [
                {"name": "Planning", "duration": "1 week", "tasks": ["Literature", "Falsifiable claims"]},
                {"name": "Branch search", "duration": "1 week", "tasks": ["Cheap probes", "Promote winner"]},
                {"name": "Full experiments", "duration": "1-2 weeks", "tasks": ["Multi-seed", "Ablations"]},
                {"name": "Writing + verification", "duration": "1-2 weeks", "tasks": ["Draft", "Hard checks"]},
            ],
            "total_duration": "4-6 weeks",
        }

    def _store_plan(self, plan: Plan, topic: Topic):
        try:
            plan_path = f"{self.runtime_config.output_dir}/plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(plan_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, default=str)
            self.vector_memory.add_embedding(
                generate_embedding(json.dumps({"title": plan.get("title"), "topic": topic.get("title")})),
                {
                    "type": "research_plan",
                    "namespace": "planner",
                    "content_class": "generated_narrative",
                    "retrieval_eligible": False,
                    "agent": "PlannerAgent",
                    "outcome_status": "unknown",
                    "topic": topic.get("title"),
                    "plan_file": plan_path,
                },
            )
        except Exception as e:
            log_agent_action("PlannerAgent", "plan_storage_error", {"error": str(e)})

    def _create_fallback_plan(self, topic: Topic) -> Plan:
        return {
            "title": f"Research on {topic.get('title')}",
            "sections": [
                {"name": n, "content_requirements": n, "key_points": [], "expected_length": "1-3 pages", "dependencies": []}
                for n in [
                    "Abstract", "Introduction", "Related Work", "Methods",
                    "Experiments", "Results", "Discussion", "Limitations", "Conclusion",
                ]
            ],
            "research_questions": ["Does the proposed method outperform baselines?"],
            "methodology": "Controlled comparison on synthetic/public data with multi-seed stats",
            "expected_contributions": ["A method that beats named baselines on primary metric"],
            "compute_budget": "CPU sklearn-scale",
            "dataset_availability": "synthetic",
            "created_at": datetime.now().isoformat(),
            "topic": topic.get("title"),
            "domain": self.runtime_config.research_domain,
        }

    def _create_fallback_experiments(self, topic: Topic) -> List[ExperimentSpec]:
        return [
            {
                "name": "Baseline Comparison",
                "purpose": "Compare proposed method vs baselines",
                "methodology": "Train proposed model and baselines on same splits",
                "baselines": ["logistic_regression", "random_forest"],
                "baseline_comparison": "logistic_regression, random_forest",
                "evaluation_metrics": ["accuracy", "f1"],
                "falsifiable_prediction": "Proposed mean accuracy > best baseline mean accuracy",
                "statistical_test": "Welch t-test p<0.05 across 3 seeds",
                "claimed_components": ["proposed_model"],
                "code_requirements": "Python, sklearn, numpy",
                "data_requirements": "synthetic classification data",
                "variants": [],
            },
            {
                "name": "Ablation Study",
                "purpose": "Measure contribution of claimed components",
                "methodology": "Remove each component and measure degradation",
                "baselines": ["full_model"],
                "evaluation_metrics": ["accuracy_delta"],
                "falsifiable_prediction": "Removing claimed component reduces accuracy by >1pp",
                "statistical_test": "paired comparison across seeds",
                "claimed_components": ["proposed_model"],
                "code_requirements": "Python",
                "data_requirements": "same as main",
                "variants": [],
            },
        ]
