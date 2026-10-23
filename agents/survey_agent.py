"""SurveyAgent — survey response-table analysis (proposal §4.8, M3).

Exact algorithm:
- ``validate_response_table``: target present, target not in features,
  non-empty, encoding fallback.
- Descriptives + 95% CI.
- ``cronbach_alpha`` for multi-item scales (report value; warn when < 0.7).
- ``nonresponse``: item rates + unit rate, both reported.
- Fishing guard: non-preregistered subgroups → advisory finding, unreported
  by default.
- ``detect_synthetic`` (duplicate-row rate + entropy floor) → blocking when
  tripped (``ALLOW_SYNTHETIC_RESPONDENTS=false`` is locked).

Reads packs via the DataAgent handoff only (no direct download/execute):
the workflow node validates the file with ``DataAgent`` and hands this
agent a DataFrame (or the agent reads a local path with the same D17
resilience). Mirrors the ReviewAgent/ThesisAgent module shape so the
offline harness can exercise every step without an LLM.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action
from core.context import RunContext, get_active_context
from core.memory import memory

logger = logging.getLogger(__name__)

# Alpha warning bar (§4.8): report value; warn when < 0.7.
CRONBACH_ALPHA_WARN = 0.7

# Synthetic screen bars (§4.8): duplicate-row rate + entropy floor.
SYNTHETIC_DUPLICATE_BAR = 0.2
SYNTHETIC_UNIQUE_FLOOR = 0.05


def validate_response_table(
    frame: Any,
    target: Optional[str] = None,
    features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate a response DataFrame (§4.8 step 1, pure, deterministic).

    Returns ``{passed, reasons}`` with named ``needs_*`` reasons (never a
    traceback). Encoding fallback already happened at read time (DataAgent /
    core.packs D17) — here the checks are target present, target not in
    features, and non-empty.
    """
    reasons: List[str] = []
    columns: List[str] = []
    n_rows = 0
    try:
        columns = list(frame.columns)
        n_rows = int(len(frame))
    except Exception:
        return {"passed": False, "reasons": ["needs_pack_invalid: response_table unreadable"],
                "rows": 0, "columns": []}
    if n_rows == 0 or not columns:
        reasons.append("needs_pack_invalid: response_table has no rows")
        return {"passed": False, "reasons": reasons, "rows": n_rows, "columns": columns}
    if target and target not in columns:
        reasons.append(f"needs_pack_invalid: target {target!r} not in response_table columns")
    overlap = sorted(set(features or []) & ({target} if target else set()))
    if overlap:
        reasons.append(f"needs_pack_invalid: target appears in features: {overlap[0]}")
    return {"passed": not reasons, "reasons": reasons, "rows": n_rows, "columns": columns}


def descriptives(frame: Any, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Descriptives + 95% CI per numeric column (pure, deterministic).

    CI uses the normal approximation (mean ± 1.96·std/√n). Non-numeric
    columns report counts only. Returns ``{columns: {name: stats}}``.
    """
    import pandas as pd  # noqa: F401  (kept local: offline-safe import)

    stats: Dict[str, Any] = {}
    try:
        names = list(columns) if columns else list(frame.columns)
    except Exception:
        return {"columns": {}}
    for name in names:
        try:
            series = frame[name].dropna()
        except Exception:
            continue
        numeric = []
        for value in series.tolist():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric.append(float(value))
        if not numeric:
            stats[str(name)] = {"n": int(len(series)), "numeric": False,
                                "counts": {str(k): int(v) for k, v in series.value_counts().head(10).items()}}
            continue
        n = len(numeric)
        mean = sum(numeric) / n
        var = sum((v - mean) ** 2 for v in numeric) / max(1, n - 1) if n > 1 else 0.0
        std = math.sqrt(var)
        half = 1.96 * std / math.sqrt(n) if n else 0.0
        stats[str(name)] = {"n": n, "numeric": True,
                            "mean": round(mean, 4), "std": round(std, 4),
                            "min": round(min(numeric), 4), "max": round(max(numeric), 4),
                            "ci95": {"lo": round(mean - half, 4), "hi": round(mean + half, 4)}}
    return {"columns": stats}


def cronbach_alpha(frame: Any, items: Optional[List[str]] = None) -> Dict[str, Any]:
    """Cronbach's α for multi-item scales (pure, deterministic).

    Returns ``{alpha, n_items, n_rows, warn}``; ``warn`` is True when
    α < 0.7 (§4.8). Fewer than 2 usable items → ``alpha=None`` + note
    (not a failure — the scale gate simply does not apply).
    """
    try:
        names = list(items) if items else [c for c in list(frame.columns)]
    except Exception:
        return {"alpha": None, "n_items": 0, "n_rows": 0, "warn": False,
                "note": "unreadable frame"}
    vectors: List[List[float]] = []
    for name in names:
        try:
            series = frame[name].dropna()
        except Exception:
            continue
        vals = [float(v) for v in series.tolist() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if vals:
            vectors.append(vals)
    if len(vectors) < 2:
        return {"alpha": None, "n_items": len(vectors),
                "n_rows": int(len(frame)) if hasattr(frame, "__len__") else 0,
                "warn": False, "note": "fewer than 2 numeric items: alpha not applicable"}
    n_rows = min(len(v) for v in vectors)
    cols = [v[:n_rows] for v in vectors]
    k = len(cols)
    item_vars = []
    for col in cols:
        mean = sum(col) / n_rows
        item_vars.append(sum((v - mean) ** 2 for v in col) / max(1, n_rows - 1) if n_rows > 1 else 0.0)
    totals = [sum(col[i] for col in cols) for i in range(n_rows)]
    tmean = sum(totals) / n_rows
    total_var = sum((t - tmean) ** 2 for t in totals) / max(1, n_rows - 1) if n_rows > 1 else 0.0
    if total_var <= 0:
        return {"alpha": 0.0, "n_items": k, "n_rows": n_rows, "warn": True,
                "note": "zero total variance"}
    alpha = (k / (k - 1)) * (1 - sum(item_vars) / total_var) if k > 1 else 0.0
    alpha = max(0.0, min(1.0, alpha))
    return {"alpha": round(alpha, 3), "n_items": k, "n_rows": n_rows,
            "warn": bool(alpha < CRONBACH_ALPHA_WARN),
            "note": "warn: alpha < 0.7" if alpha < CRONBACH_ALPHA_WARN else "alpha ok"}


def nonresponse_report(frame: Any) -> Dict[str, Any]:
    """Nonresponse: item rates + unit rate, both reported (pure, deterministic)."""
    try:
        n_rows = int(len(frame))
        columns = list(frame.columns)
    except Exception:
        return {"item_rates": {}, "unit_rate": 0.0, "n_rows": 0, "reported": False}
    item_rates: Dict[str, float] = {}
    rows_with_gap = 0
    for name in columns:
        try:
            missing = int(frame[name].isna().sum())
        except Exception:
            missing = 0
        item_rates[str(name)] = round(missing / n_rows, 4) if n_rows else 0.0
    try:
        rows_with_gap = int(frame.isna().any(axis=1).sum())
    except Exception:
        rows_with_gap = 0
    unit_rate = round(rows_with_gap / n_rows, 4) if n_rows else 0.0
    return {"item_rates": item_rates, "unit_rate": unit_rate,
            "n_rows": n_rows, "reported": True}


def fishing_guard(
    requested_subgroups: Optional[List[str]] = None,
    preregistered: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fishing guard (pure, deterministic).

    Non-preregistered subgroups → advisory finding, unreported by default:
    ``reported`` carries only preregistered names; ``advisory`` names the
    withheld ones so the manuscript never silently publishes a fished cut.
    """
    requested = [str(s or "").strip() for s in (requested_subgroups or []) if str(s or "").strip()]
    prereg = {str(s or "").strip().lower() for s in (preregistered or []) if str(s or "").strip()}
    reported = [s for s in requested if s.lower() in prereg]
    advisory = [s for s in requested if s.lower() not in prereg]
    finding = None
    if advisory:
        finding = {"check": "fishing_guard", "blocking": False,
                   "details": f"non-preregistered subgroups withheld from report: {advisory}"}
    return {"reported": reported, "withheld": advisory,
            "advisory": finding,
            "note": "non-preregistered subgroups are advisory-only, unreported by default"}


def detect_synthetic(frame: Any) -> Dict[str, Any]:
    """Synthetic-respondent screen: duplicate-row rate + entropy floor (pure).

    Trips (``is_synthetic=True``, ``blocking=True``) when the duplicate-row
    rate exceeds 0.2 or the mean unique-value ratio falls below 0.05 —
    both classic fingerprints of fabricated response tables. The
    ``ALLOW_SYNTHETIC_RESPONDENTS=false`` latch stays locked: a tripped
    screen blocks release (the caller records a blocking finding).
    """
    try:
        n_rows = int(len(frame))
        columns = list(frame.columns)
    except Exception:
        return {"is_synthetic": False, "blocking": False, "duplicate_rate": 0.0,
                "mean_unique_ratio": 1.0, "reasons": ["unreadable frame"]}
    if n_rows == 0:
        return {"is_synthetic": False, "blocking": False, "duplicate_rate": 0.0,
                "mean_unique_ratio": 0.0, "reasons": ["empty frame"]}
    try:
        n_unique_rows = int(frame.drop_duplicates().shape[0])
    except Exception:
        n_unique_rows = n_rows
    duplicate_rate = round(1 - n_unique_rows / n_rows, 4) if n_rows else 0.0
    ratios: List[float] = []
    for name in columns:
        try:
            nunique = int(frame[name].nunique(dropna=True))
        except Exception:
            continue
        ratios.append(nunique / n_rows if n_rows else 0.0)
    mean_unique = round(sum(ratios) / len(ratios), 4) if ratios else 1.0
    reasons: List[str] = []
    if duplicate_rate > SYNTHETIC_DUPLICATE_BAR:
        reasons.append(f"duplicate-row rate {duplicate_rate} > {SYNTHETIC_DUPLICATE_BAR}")
    if mean_unique < SYNTHETIC_UNIQUE_FLOOR:
        reasons.append(f"mean unique-value ratio {mean_unique} < {SYNTHETIC_UNIQUE_FLOOR} (entropy floor)")
    tripped = bool(reasons)
    latch_open = bool(getattr(config, "allow_synthetic_respondents", False))
    return {"is_synthetic": tripped,
            "blocking": bool(tripped and not latch_open),
            "duplicate_rate": duplicate_rate,
            "mean_unique_ratio": mean_unique,
            "reasons": reasons,
            "latch": "open" if latch_open else "locked-false"}


def _hash_frame(frame: Any) -> str:
    try:
        payload = frame.to_csv(index=False).encode("utf-8", errors="replace")
    except Exception:
        payload = repr(frame).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


class SurveyAgent:
    """Descriptives → α → nonresponse → fishing guard → synthetic screen (§4.8)."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    @property
    def vector_memory(self):
        return self.context.memory if self.context else memory

    def run_analysis(
        self,
        frame: Any,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        scale_items: Optional[List[str]] = None,
        requested_subgroups: Optional[List[str]] = None,
        preregistered_subgroups: Optional[List[str]] = None,
        codebook: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full §4.8 pass over an already-validated response frame.

        ``preregistered_subgroups`` defaults to the codebook's
        ``subgroup_preregistered`` flags when omitted. Returns the analysis
        bundle plus the ``instrument`` record
        (``{questionnaire, codebook_hash, response_hash, nonresponse}``).
        """
        validation = validate_response_table(frame, target, features)
        if codebook and not preregistered_subgroups:
            preregistered_subgroups = list((codebook or {}).get("preregistered_subgroups") or [])
        desc = descriptives(frame)
        alpha = cronbach_alpha(frame, scale_items)
        nonresponse = nonresponse_report(frame)
        fishing = fishing_guard(requested_subgroups, preregistered_subgroups)
        synthetic = detect_synthetic(frame)
        codebook_blob = "|".join(sorted(str(i) for i in ((codebook or {}).get("items") or [])))
        instrument = {
            "questionnaire": (codebook or {}).get("manifest", {}).get("name") if isinstance(codebook, dict) else None,
            "codebook_hash": hashlib.sha256(codebook_blob.encode("utf-8")).hexdigest() if codebook_blob else None,
            "response_hash": _hash_frame(frame),
            "nonresponse": {"item_rates": nonresponse["item_rates"], "unit_rate": nonresponse["unit_rate"]},
        }
        findings: List[Dict[str, Any]] = []
        if not validation["passed"]:
            findings.append({"check": "response_table", "blocking": True,
                             "details": "; ".join(validation["reasons"])})
        if alpha.get("warn"):
            findings.append({"check": "cronbach_alpha", "blocking": False,
                             "details": f"alpha {alpha.get('alpha')} < 0.7"})
        if fishing.get("advisory"):
            findings.append(fishing["advisory"])
        if synthetic.get("blocking"):
            findings.append({"check": "synthetic_screen", "blocking": True,
                             "details": "synthetic respondents suspected: " + "; ".join(synthetic["reasons"])})
        passed = (validation["passed"] and not synthetic.get("blocking"))
        log_agent_action("SurveyAgent", "analysis_complete", {
            "rows": validation.get("rows"), "alpha": alpha.get("alpha"),
            "unit_nonresponse": nonresponse.get("unit_rate"),
            "synthetic": synthetic.get("is_synthetic"), "passed": passed})
        return {"validation": validation, "descriptives": desc, "alpha": alpha,
                "nonresponse": nonresponse, "fishing": fishing,
                "synthetic": synthetic, "instrument": instrument,
                "passed": passed, "findings": findings}


#: Proposal §4.3 manifest name (reads packs via DataAgent handoff only).
SurveyAnalysisAgent = SurveyAgent
