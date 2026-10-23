"""Per-type release readiness rules (proposal §7.2, M5 backend).

Pure function — no I/O, no LLM, no gateway. Maps a stored ``paper_type``
plus deterministic gate evidence to ``{"ready": bool, "missing": [...]}``.

Formulas (§7.2 exact):
- base (all types): ``not terminal AND no blocking findings``.
- exp/comp/emp/cause_effect: ``+ reproducibility.passed``.
- review: ``+ prisma_reconciled AND extraction >= 0.9 AND recall >= 4/5``.
- analytical/argumentative: ``+ thesis triad + bridges == 100%``
  (no reproducibility required).
- descriptive/survey: ``+ tables recomputed (+ survey: nonresponse reported)``.
- case: ``+ triangulation ok``.

Downgrade/refusal inputs in ``extra`` pass through as banner fields on the
result — never silent.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

#: Types that additionally require reproducibility.passed (§7.2).
REPRO_TYPES = frozenset({"experimental", "comparative", "empirical", "cause_effect"})

#: Types gated on the thesis triad + 100% bridge coverage (no repro required).
THESIS_TYPES = frozenset({"analytical", "argumentative"})

#: Banner keys that pass through verbatim when present in ``extra``.
BANNER_KEYS = (
    "banner",
    "banners",
    "downgrade",
    "downgraded_from",
    "downgraded_to",
    "refusal",
    "refusals",
    "refusal_reason",
    "demote_to_scoping",
    "mode",
)


def _content_tokens(text: Any) -> List[str]:
    """Tokenize like ThesisAgent._content_tokens (mirror, no import)."""
    return [w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 3]


def _lookup(extra: Dict[str, Any], *names: str) -> Any:
    """Find the first present (non-None) value for names, top-level then nested.

    Nested carriers checked one level deep: corpus, review, thesis, survey,
    analysis, analysis_reports, case, protocol, case_protocol, causal, instrument.
    """
    for name in names:
        if name in extra and extra[name] is not None:
            return extra[name]
    for carrier in (
        "corpus", "review", "thesis", "survey", "analysis", "analysis_reports",
        "case", "protocol", "case_protocol", "causal", "instrument",
    ):
        nested = extra.get(carrier)
        if isinstance(nested, dict):
            for name in names:
                if name in nested and nested[name] is not None:
                    return nested[name]
    return None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("passed", "ok", "reconciled", "value"):
            if isinstance(value.get(key), bool):
                return bool(value[key])
        return None
    return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("recall", "completeness", "coverage", "value", "rate"):
            inner = value.get(key)
            if isinstance(inner, (int, float)) and not isinstance(inner, bool):
                return float(inner)
    return None


def _thesis_triad_ok(extra: Dict[str, Any], thesis: Any) -> Optional[bool]:
    """Triad status: explicit flag wins, else derive from the thesis dict."""
    flag = _lookup(extra, "thesis_triad", "triad_ok", "thesis_ok", "thesis_passed")
    as_bool = _as_bool(flag)
    if as_bool is not None:
        return as_bool
    if isinstance(flag, dict):
        try:
            return all(bool(v) for v in flag.values())
        except Exception:
            return None
    if not isinstance(thesis, dict) or not thesis:
        return None
    claim_ok = bool(str(thesis.get("claim") or "").strip())
    bridges = thesis.get("bridges") or []
    bridges_ok = bool(bridges) and all(
        isinstance(b, dict) and bool(str(b.get("statement") or "").strip())
        for b in bridges
    )
    steel_conc_ok = bool(str(thesis.get("steelman") or "").strip()) and bool(
        str(thesis.get("concession") or "").strip()
    )
    return bool(claim_ok and bridges_ok and steel_conc_ok)


def _bridges_full_ok(extra: Dict[str, Any], thesis: Any) -> Optional[bool]:
    """100% bridge coverage: explicit flag wins, else compute deterministically."""
    flag = _lookup(extra, "bridge_coverage", "bridges_100", "bridges_ok", "bridges_full")
    as_bool = _as_bool(flag)
    if as_bool is not None:
        return as_bool
    cov = _as_float(flag)
    if cov is not None:
        return bool(cov >= 0.999)
    if not isinstance(thesis, dict):
        return None
    bridges = thesis.get("bridges") or []
    if not bridges:
        return None
    sections = _lookup(extra, "sections", "draft_sections", "manuscript")
    manuscript = ""
    if isinstance(sections, dict):
        manuscript = "\n".join(v for v in sections.values() if isinstance(v, str))
    manuscript_tokens = set(_content_tokens(manuscript))
    for bridge in bridges:
        if not isinstance(bridge, dict):
            return False
        tokens = set(_content_tokens(bridge.get("statement") or ""))
        if not tokens:
            return False
        recall = sum(1 for t in tokens if t in manuscript_tokens) / len(tokens)
        if recall < 0.5:
            return False
    return True


def _tables_evidence_ok(tables: Any) -> bool:
    """True only with recomputation proof (fail-closed).

    Accepts explicit ``True`` markers (``tables_recomputed``/``passed``/
    ``ok``/``recomputed``/``verified``), a non-empty descriptives/columns
    mapping as produced by ``SurveyAgent.run_analysis`` (deterministically
    computed over the response table), or a passing ``validation`` block.
    Any other truthy bundle — including ``{"columns": {}}`` — is ``False``:
    presence of a bundle dict is not proof its tables were recomputed.
    """
    if tables is None:
        return False
    if isinstance(tables, bool):
        return tables
    if isinstance(tables, dict):
        for key in (
            "tables_recomputed", "tables_ok", "descriptives_ok",
            "passed", "ok", "recomputed", "verified",
        ):
            if tables.get(key) is True:
                return True
            if tables.get(key) is False and key in ("passed", "ok", "tables_recomputed"):
                return False
        desc = tables.get("descriptives")
        if isinstance(desc, dict):
            cols = desc.get("columns")
            if isinstance(cols, dict) and len(cols):
                return True
        cols = tables.get("columns")
        if isinstance(cols, dict) and len(cols):
            return True
        validation = tables.get("validation")
        if isinstance(validation, dict) and validation.get("passed") is True:
            return True
        return False
    return False


def evaluate_release(
    paper_type: Any,
    *,
    terminal: Any = False,
    blocking_findings: Any = None,
    reproducibility_passed: Any = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate per-type release readiness (§7.2).

    Args:
        paper_type: one of the 10 type IDs (falsy/unknown → base gate only,
            except falsy which defaults to ``experimental``).
        terminal: terminal signal — falsy/None/False/"" means no terminal;
            a message string, True, or dict with message/details means terminal.
        blocking_findings: findings with ``blocking`` set — list of dicts
            (``check``/``category``/``message``/``details``) or strings.
        reproducibility_passed: truthy when reproducibility passed.
        extra: per-type evidence (corpus/thesis/instrument/analysis/causal
            bundles, explicit flags, downgrade/refusal banners).

    Returns:
        ``{"ready": bool, "missing": [...]}`` plus any banner fields found
        in ``extra`` (downgrade/refusal pass through, never silent).
    """
    ptype = str(paper_type or "experimental").strip().lower() or "experimental"
    if ptype == "case":
        ptype = "case_study"
    extra = dict(extra or {})
    missing: List[str] = []

    # -- base gate (all types): not terminal AND no blocking findings --
    terminal_msg: Optional[str] = None
    if terminal:
        if isinstance(terminal, str) and terminal.strip():
            terminal_msg = terminal.strip()
        elif isinstance(terminal, dict):
            terminal_msg = str(
                terminal.get("message") or terminal.get("terminal_error")
                or terminal.get("reason") or terminal.get("details") or "terminal"
            )
        else:
            terminal_msg = "terminal_error"
        missing.append(f"terminal:{terminal_msg}")

    blocking = blocking_findings or []
    if isinstance(blocking, dict):
        blocking = [blocking]
    for finding in list(blocking):
        if isinstance(finding, dict):
            if finding.get("blocking") is False:
                continue
            check = finding.get("check") or finding.get("category") or finding.get("message") or "finding"
            missing.append(f"blocking:{check}")
        elif isinstance(finding, str) and finding.strip():
            missing.append(f"blocking:{finding.strip()}")

    # -- per-type additions (§7.2) --
    if ptype in REPRO_TYPES:
        if not reproducibility_passed:
            missing.append("reproducibility")
    elif ptype == "review":
        prisma = _lookup(extra, "prisma_reconciled", "prisma_passed", "reconciled", "prisma")
        prisma_ok = _as_bool(prisma)
        if prisma_ok is None and isinstance(prisma, dict):
            prisma_ok = bool(prisma.get("passed", prisma.get("reconciled", False)))
        if prisma_ok is None:
            prisma_ok = False
        if not prisma_ok:
            missing.append("prisma_reconciled")
        extraction = _lookup(extra, "extraction_completeness", "extraction", "completeness")
        ext_val = _as_float(extraction)
        ext_ok = _as_bool(extraction)
        if ext_ok is None:
            ext_ok = ext_val is not None and ext_val >= 0.9
        if not ext_ok:
            missing.append("extraction>=0.9")
        recall = _lookup(extra, "recall", "seed_recall", "recall_passed", "recall_ok")
        recall_ok = _as_bool(recall)
        if recall_ok is None:
            recall_val = _as_float(recall)
            if recall_val is not None:
                recall_ok = recall_val >= 0.8 if recall_val <= 1.0 else recall_val >= 4.0
            else:
                recall_ok = False
        if not recall_ok:
            missing.append("recall>=4/5")
    elif ptype in THESIS_TYPES:
        thesis = _lookup(extra, "thesis")
        triad_ok = _thesis_triad_ok(extra, thesis)
        if not triad_ok:
            missing.append("thesis_triad")
        bridges_ok = _bridges_full_ok(extra, thesis)
        if not bridges_ok:
            missing.append("bridges==100%")
    elif ptype in ("descriptive", "survey"):
        tables = _lookup(
            extra, "tables_recomputed", "tables_ok", "descriptives_ok",
            "descriptives", "analysis", "survey_analysis",
        )
        tables_ok = _as_bool(tables)
        if tables_ok is None:
            tables_ok = _tables_evidence_ok(tables)
        if not tables_ok:
            missing.append("tables_recomputed")
        if ptype == "survey":
            nonresponse = _lookup(
                extra, "nonresponse_reported", "nonresponse_ok", "nonresponse",
            )
            nr_ok = _as_bool(nonresponse)
            if nr_ok is None:
                if isinstance(nonresponse, dict):
                    item_rates = nonresponse.get("item_rates")
                    unit_rate = nonresponse.get("unit_rate")
                    nr_ok = (
                        isinstance(item_rates, dict) and bool(item_rates)
                        and isinstance(unit_rate, (int, float))
                        and not isinstance(unit_rate, bool)
                    )
                else:
                    nr_ok = False
            if not nr_ok:
                missing.append("nonresponse_reported")
    elif ptype == "case_study":
        triang = _lookup(
            extra, "triangulation_ok", "triangulation", "case_protocol", "protocol",
        )
        triang_ok = _as_bool(triang)
        if triang_ok is None:
            if isinstance(triang, dict):
                flags = triang.get("single_source_flags") or []
                limitations = triang.get("limitations") or _lookup(extra, "limitations") or []
                rival = triang.get("rival")
                # Re-audit (M6): fail-closed like verify_case — a missing rival
                # dict means no rival evidence, not an implicit pass. Explicit
                # bool flags (triangulation_ok=True) still short-circuit above.
                rival_ok = isinstance(rival, dict) and bool(rival.get("passed", False))
                triang_ok = bool((not flags or limitations) and rival_ok)
            else:
                triang_ok = False
        if not triang_ok:
            missing.append("triangulation_ok")
    # Unknown paper types: base gate only (no extra requirements imposed).

    result: Dict[str, Any] = {"ready": not missing, "missing": missing}
    for key in BANNER_KEYS:
        if key in extra and extra[key] is not None:
            result[key] = extra[key]
    return result
