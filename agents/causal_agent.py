"""CausalAgent — cause-effect identification audit (proposal §4.7, M4).

Exact algorithm:
- ``classify_id`` ∈ {rct, quasi, synthetic_benchmark, naive} from DAG + plan.
- Adjustment-set lint: named confounders must appear in code covariates or banner.
- Placebo hint: permutation or dummy-outcome where sandbox-feasible;
  else ``not_feasible`` + limitation entry.
- ``verb_level``: ``causal`` only when strategy ≠ naive + sensitivity note
  non-empty + indep pass; else ``associational`` (governor rewrites
  ``causes``/``ATE`` → ``associated``/``predicts``).
- Causal failure downgrades to analytical (locked §0 decision 3), never
  terminal: the audit returns a ``downgrade`` record + banner and the
  workflow node reroutes the run.

Mirrors the SurveyAgent/CaseAgent module shape so the offline harness can
exercise every step without an LLM: all classifiers are pure/deterministic.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action
from core.context import RunContext, get_active_context
from core.memory import memory

logger = logging.getLogger(__name__)

# Identification strategies (exact literals, §4.7).
ID_STRATEGIES: tuple[str, ...] = ("rct", "quasi", "synthetic_benchmark", "naive")

# Verb levels (exact literals, §4.7).
VERB_LEVELS: tuple[str, ...] = ("causal", "associational")

# Strong causal claims the governor rewrites when the level is associational.
# (pattern, replacement) pairs applied with word boundaries, case-insensitive.
_GOVERNOR_RULES: tuple[tuple[str, str], ...] = (
    (r"\baverage treatment effects?\b", "adjusted differences"),
    (r"\bATEs?\b", "adjusted difference"),
    (r"\btreatment effects?\b", "observed associations"),
    (r"\bcausal effects?\b", "observed associations"),
    (r"\bcauses\b", "is associated with"),
    (r"\bcause\b", "association"),
    (r"\bcaused\b", "associated with"),
    (r"\bcausing\b", "associated with"),
    (r"\bcausal\b", "associational"),
)

# Strong causal verbs that must not appear in Results/Abstract prose when the
# verb level is associational (manuscript gate, §7.1).
_ASSOCIATIONAL_BANNED: tuple[str, ...] = (
    "causes",
    "caused",
    "causing",
    "causal effect",
    "treatment effect",
    "average treatment effect",
    r"ATE",
)

_RCT_TOKENS = (
    "random assignment",
    "randomly assigned",
    "randomized controlled",
    "randomised controlled",
    "randomized trial",
    "randomised trial",
)

_QUASI_TOKENS = (
    "instrumental variable",
    "diff-in-diff",
    "difference-in-differences",
    "regression discontinuity",
    "propensity score",
    "propensity-score",
    "matching estimator",
    "doubly robust",
    "quasi-experiment",
    "quasi experiment",
    "synthetic control",
    "fixed effects",
)

_SYNTHETIC_TOKENS = (
    "synthetic",
    "simulated dag",
    "simulated-dag",
    "dag benchmark",
    "structural causal model",
)


def _evidence_text(evidence: Any) -> str:
    """Flatten DAG + plan text into one lowercase search blob."""
    parts: List[str] = []
    if isinstance(evidence, dict):
        for key in ("methodology", "experiments_text", "research_question"):
            value = evidence.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        dag = evidence.get("dag")
        if isinstance(dag, dict):
            parts.append(str(dag))
        spec = evidence.get("causal_spec")
        if isinstance(spec, dict):
            parts.append(str(spec))
    elif isinstance(evidence, str):
        parts.append(evidence)
    return "\n".join(parts).lower()


def classify_id(evidence: Any) -> Dict[str, Any]:
    """Classify the identification strategy from DAG + plan (pure).

    Returns ``{strategy, reasons}`` with strategy ∈ {rct, quasi,
    synthetic_benchmark, naive}. A declared ``id_strategy`` is honoured
    only when the plan text corroborates it — the classifier never trusts
    a label the design cannot support.
    """
    text = _evidence_text(evidence)
    declared = ""
    if isinstance(evidence, dict):
        spec = evidence.get("causal_spec")
        if isinstance(spec, dict):
            declared = str(spec.get("id_strategy") or "").strip().lower()
        else:
            declared = str(evidence.get("id_strategy") or "").strip().lower()

    def _supported(strategy: str) -> bool:
        return not declared or declared == strategy

    if any(token in text for token in _RCT_TOKENS) and _supported("rct"):
        return {"strategy": "rct", "reasons": ["randomized assignment described in plan"]}
    if any(token in text for token in _SYNTHETIC_TOKENS) and ("dag" in text or "benchmark" in text) and _supported("synthetic_benchmark"):
        return {"strategy": "synthetic_benchmark", "reasons": ["synthetic-DAG benchmark described in plan"]}
    if any(token in text for token in _QUASI_TOKENS) and _supported("quasi"):
        return {"strategy": "quasi", "reasons": ["quasi-experimental design described in plan"]}
    reasons = ["no randomized, quasi-experimental, or synthetic-DAG identification described"]
    if declared and declared not in ID_STRATEGIES:
        reasons.append(f"declared id_strategy {declared!r} is not a known strategy")
    return {"strategy": "naive", "reasons": reasons}


def lint_adjustment_set(
    dag: Any,
    code_covariates: Any = None,
    code_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Adjustment-set lint (pure): named confounders must appear in code covariates or banner.

    ``code_covariates`` is an explicit covariate list when the caller has
    one; otherwise names are searched in ``code_text``. Missing names
    produce a banner (advisory, never blocking — §4.7).
    """
    confounders: List[str] = []
    if isinstance(dag, dict):
        raw = dag.get("confounders") or dag.get("adjustment_set") or []
        if isinstance(raw, (list, tuple)):
            confounders = [str(item).strip() for item in raw if str(item).strip()]
    haystack = ""
    if isinstance(code_covariates, (list, tuple)):
        haystack = " ".join(str(item) for item in code_covariates).lower()
    elif isinstance(code_text, str):
        haystack = code_text.lower()
    covered = [name for name in confounders if name.lower() in haystack]
    missing = [name for name in confounders if name.lower() not in haystack]
    banner = ""
    if missing:
        banner = (
            "Adjustment-set lint: confounder(s) "
            + ", ".join(missing)
            + " named in the DAG do not appear in code covariates"
        )
    return {"confounders": confounders, "covered": covered, "missing": missing, "banner": banner}


def placebo_hint(
    plan: Any = None,
    code_text: Optional[str] = None,
    sandbox_feasible: Optional[bool] = None,
) -> Dict[str, Any]:
    """Placebo/refutation hint (pure, §4.7).

    Returns ``{feasible, hint}`` when a permutation or dummy-outcome
    placebo is sandbox-feasible, else ``{feasible: False, limitation}``
    with a limitation entry (never a block).
    """
    text = (code_text or "").lower()
    feasible = sandbox_feasible
    if feasible is None:
        feasible = any(token in text for token in ("permut", "placebo", "shuffle", "dummy outcome", "randomization test"))
    if feasible:
        return {
            "feasible": True,
            "hint": (
                "Run a permutation placebo: shuffle the treatment labels (or "
                "use a dummy outcome known to be unaffected) and re-estimate. "
                "A null placebo distribution supports the identification claim."
            ),
            "limitation": "",
        }
    return {
        "feasible": False,
        "hint": "",
        "limitation": (
            "Placebo/refutation check not feasible in the sandbox for this "
            "design (not_feasible): no permutation or dummy-outcome placebo "
            "was run; causal interpretation is correspondingly weaker."
        ),
    }


def find_sensitivity_note(plan: Any) -> str:
    """Extract the sensitivity-analysis note from a plan (pure)."""
    candidates: List[str] = []
    if isinstance(plan, dict):
        spec = plan.get("causal_spec")
        if isinstance(spec, dict) and str(spec.get("sensitivity") or "").strip():
            candidates.append(str(spec["sensitivity"]))
        for exp in plan.get("experiments") or []:
            if isinstance(exp, dict):
                for key in ("sensitivity", "sensitivity_analysis"):
                    if str(exp.get(key) or "").strip():
                        candidates.append(str(exp[key]))
        for key in ("sensitivity", "sensitivity_analysis"):
            if str(plan.get(key) or "").strip():
                candidates.append(str(plan[key]))
        blob = " ".join(
            str(value)
            for key in ("methodology", "analysis_plan")
            for value in ([plan.get(key)] if plan.get(key) else [])
        )
        for sentence in re.split(r"(?<=[.!?])\s+", blob):
            if "sensitiv" in sentence.lower() and sentence.strip():
                candidates.append(sentence.strip())
    return candidates[0] if candidates else ""


def verb_level(
    strategy: str,
    sensitivity_note: str = "",
    indep_pass: bool = False,
) -> str:
    """Causal verb level (pure, §4.7).

    ``causal`` only when the strategy is identified (≠ naive) AND the
    sensitivity note is non-empty AND independent validation passed;
    otherwise ``associational``.
    """
    if str(strategy) != "naive" and str(sensitivity_note or "").strip() and bool(indep_pass):
        return "causal"
    return "associational"


def govern_text(text: str, level: str) -> str:
    """Rewrite strong causal claims down when the level is associational (pure).

    ``causal`` level returns the text unchanged. ``associational`` level
    applies the §4.7 governor (causes/ATE → associated/predicts family).
    """
    if level == "causal" or not text:
        return text
    governed = str(text)
    for pattern, replacement in _GOVERNOR_RULES:
        governed = re.sub(pattern, replacement, governed, flags=re.IGNORECASE)
    return governed


class CausalAgent:
    """Audit cause-effect identification and govern causal language."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    @property
    def vector_memory(self):
        return self.context.memory if self.context else memory

    def run_audit(
        self,
        plan: Optional[Dict[str, Any]] = None,
        engineer_outputs: Optional[Dict[str, Any]] = None,
        code_text: Optional[str] = None,
        indep_pass: bool = False,
    ) -> Dict[str, Any]:
        """Run the §4.7 audit over one cause-effect plan (deterministic).

        Returns the ``causal_claim`` bundle
        ``{dag, id_strategy, sensitivity, verb_level}`` plus classification,
        adjustment lint, placebo, limitations, and — for naive designs — a
        downgrade record to analytical (never terminal).
        """
        plan = dict(plan or {})
        outputs = engineer_outputs or {}
        spec = plan.get("causal_spec") if isinstance(plan.get("causal_spec"), dict) else {}
        dag = spec.get("dag") if isinstance(spec.get("dag"), dict) else {}
        evidence = {
            "dag": dag,
            "causal_spec": spec,
            "methodology": str(plan.get("methodology") or ""),
            "experiments_text": " ".join(
                str(exp.get("methodology") or "")
                for exp in (plan.get("experiments") or [])
                if isinstance(exp, dict)
            ),
        }
        classification = classify_id(evidence)
        strategy = classification["strategy"]
        adjustment = lint_adjustment_set(dag, code_text=code_text)
        placebo = placebo_hint(plan, code_text=code_text)
        sensitivity = find_sensitivity_note(plan)
        level = verb_level(strategy, sensitivity, indep_pass)

        limitations: List[str] = []
        if adjustment.get("banner"):
            limitations.append(adjustment["banner"])
        if not placebo.get("feasible") and placebo.get("limitation"):
            limitations.append(placebo["limitation"])
        if not sensitivity:
            limitations.append(
                "No sensitivity analysis reported: causal interpretation "
                "requires a robustness note (Rosenbaum-style bounds or "
                "coefficient-stability check)."
            )

        engine_ok = bool(outputs) and all(
            isinstance(item, dict) and item.get("success") is True
            for item in outputs.values()
        )
        downgrade: Optional[Dict[str, Any]] = None
        passed = True
        findings: List[Dict[str, Any]] = []
        if strategy == "naive":
            passed = False
            downgrade = {
                "to": "analytical",
                "reason": "naive observational design without identification",
                "banner": (
                    "Causal identification missing (naive observational "
                    "design): downgraded cause_effect → analytical. Causal "
                    "verbs are blocked; the manuscript proceeds as an "
                    "associational synthesis."
                ),
            }
            findings.append({"check": "identification", "blocking": True,
                             "details": "; ".join(classification["reasons"])})
            limitations.append(downgrade["banner"])
        log_agent_action("CausalAgent", "audit_complete", {
            "strategy": strategy, "verb_level": level,
            "downgraded": bool(downgrade), "engine_ok": engine_ok,
        })
        return {
            "dag": dag,
            "id_strategy": strategy,
            "sensitivity": sensitivity,
            "verb_level": level,
            "classification": classification,
            "adjustment": adjustment,
            "placebo": placebo,
            "limitations": limitations,
            "engine_ok": engine_ok,
            "indep_pass": bool(indep_pass),
            "passed": passed,
            "downgrade": downgrade,
            "findings": findings,
        }


# §4.3 role alias (mirrors SurveyAgent/SurveyAnalysisAgent precedent).
CausalAuditAgent = CausalAgent
