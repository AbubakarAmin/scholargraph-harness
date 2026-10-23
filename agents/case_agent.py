"""CaseAgent — case-pack validation + triangulation + rival check (proposal §4.9, M3).

Exact algorithm:
- ``validate_case_pack``: manifest present + ≥2 distinct source types.
- ``triangulation_matrix``: every material claim → source list. Claims with
  <2 independent sources → ``single_source_flag`` + limitation entry
  (flagged, run continues — never silent, never terminal).
- ``rival_check``: ≥1 rival explanation addressed or explicit gap recorded.

Mirrors the ReviewAgent/SurveyAgent module shape: pure deterministic
helpers plus a thin orchestrating class. No LLM, no network, no execution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.utils import log_agent_action
from core.context import RunContext, get_active_context
from core.memory import memory

logger = logging.getLogger(__name__)


def validate_case_pack(manifest: Any) -> Dict[str, Any]:
    """Validate a case-pack manifest (pure, deterministic).

    Accepts the ``core.packs`` shape (``{manifest, entries, distinct_types}``),
    a bare entries list, or a ``{files: [...]}`` carrier. Returns
    ``{passed, distinct_types, n_entries, reasons}`` with named
    ``needs_case_pack`` / ``needs_pack_invalid`` reasons (never a traceback).
    """
    entries: List[Dict[str, Any]] = []
    if isinstance(manifest, dict):
        raw = manifest.get("entries", manifest.get("files", manifest.get("manifest")))
        if isinstance(raw, list):
            entries = [e for e in raw if isinstance(e, dict)]
        elif isinstance(manifest.get("entries"), list):
            entries = [e for e in manifest["entries"] if isinstance(e, dict)]
        elif isinstance(manifest, dict) and manifest.get("type"):
            entries = [manifest]
    elif isinstance(manifest, list):
        entries = [e for e in manifest if isinstance(e, dict)]
    if not entries:
        return {"passed": False, "distinct_types": [], "n_entries": 0,
                "reasons": ["needs_case_pack: no case pack supplied (upload case_pack.zip with >=2 source types)"]}
    distinct = sorted({str(e.get("type") or "").strip() for e in entries if str(e.get("type") or "").strip()})
    if len(distinct) < 2:
        return {"passed": False, "distinct_types": distinct, "n_entries": len(entries),
                "reasons": [f"needs_pack_invalid: case pack has {len(distinct)} source type(s), need >=2 (found: {distinct or ['none']})"]}
    return {"passed": True, "distinct_types": distinct, "n_entries": len(entries), "reasons": []}


def triangulation_matrix(
    claims: Optional[List[Any]] = None,
    evidence_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the claim → sources matrix (pure, deterministic).

    ``evidence_map`` maps each material claim to its source list (each
    source a ``{type, ref}`` dict or a plain string). Claims with <2
    independent sources (distinct ``type`` values) get a
    ``single_source_flag`` + limitation entry; the run continues with the
    flag visible (warn, not terminal).
    """
    evidence_map = dict(evidence_map or {})
    matrix: Dict[str, List[Dict[str, str]]] = {}
    single_source_flags: List[Dict[str, Any]] = []
    limitations: List[str] = []
    for claim in (claims or []):
        text = str(claim or "").strip()
        if not text:
            continue
        raw_sources = evidence_map.get(text, evidence_map.get(str(claim), []))
        sources: List[Dict[str, str]] = []
        if isinstance(raw_sources, list):
            for source in raw_sources:
                if isinstance(source, dict):
                    sources.append({"type": str(source.get("type") or "doc"),
                                    "ref": str(source.get("ref") or source.get("path") or "")})
                elif isinstance(source, str) and source.strip():
                    sources.append({"type": "doc", "ref": source.strip()})
        matrix[text] = sources
        independent = {s["type"] for s in sources if s.get("type")}
        if len(independent) < 2:
            flag = {"claim": text, "sources": sources,
                    "independent_types": sorted(independent),
                    "flag": "single_source_flag"}
            single_source_flags.append(flag)
            limitations.append(
                f"single-source claim ({len(independent)} independent source(s)): {text[:160]}")
    return {"matrix": matrix, "single_source_flags": single_source_flags,
            "limitations": limitations,
            "n_claims": len(matrix), "n_flagged": len(single_source_flags)}


def rival_check(
    rivals_addressed: Optional[List[Any]] = None,
    gap_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Rival-explanation check (pure, deterministic).

    Passes when ≥1 rival explanation is addressed, or when an explicit gap
    is recorded (honest downgrade, not silence). Fails (blocking) only when
    neither exists.
    """
    rivals = [str(r or "").strip() for r in (rivals_addressed or []) if str(r or "").strip()]
    gap = str(gap_note or "").strip()
    if rivals:
        return {"passed": True, "rivals": rivals, "gap_recorded": False, "gap": None,
                "findings": [], "note": f"{len(rivals)} rival explanation(s) addressed"}
    if gap:
        return {"passed": True, "rivals": [], "gap_recorded": True, "gap": gap,
                "findings": [{"check": "rival_check", "blocking": False,
                              "details": f"no rival addressed; explicit gap recorded: {gap[:200]}"}],
                "note": "explicit rival gap recorded"}
    return {"passed": False, "rivals": [], "gap_recorded": False, "gap": None,
            "findings": [{"check": "rival_check", "blocking": True,
                          "details": "no rival explanation addressed and no explicit gap recorded"}],
            "note": "rival check failed"}


class CaseAgent:
    """Pack validation → triangulation matrix → rival check (§4.9)."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()

    @property
    def vector_memory(self):
        return self.context.memory if self.context else memory

    def run_protocol(
        self,
        manifest: Any,
        claims: Optional[List[Any]] = None,
        evidence_map: Optional[Dict[str, Any]] = None,
        rivals_addressed: Optional[List[Any]] = None,
        gap_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full §4.9 pass: validate → triangulate → rival-check.

        Single-source flags are warnings with limitation entries (run
        continues); pack-invalid and rival-missing are blocking. Returns
        ``{pack, triangulation, rival, passed, findings, limitations}``.
        """
        pack = validate_case_pack(manifest)
        triang = triangulation_matrix(claims, evidence_map)
        rival = rival_check(rivals_addressed, gap_note)
        findings: List[Dict[str, Any]] = []
        if not pack["passed"]:
            findings.append({"check": "case_pack", "blocking": True,
                             "details": "; ".join(pack["reasons"])})
        for flag in triang["single_source_flags"]:
            findings.append({"check": "single_source_flag", "blocking": False,
                             "details": flag["claim"][:300],
                             "sources": flag["sources"]})
        if not rival["passed"]:
            findings.extend(rival["findings"])
        else:
            findings.extend([f for f in rival["findings"] if not f.get("blocking")])
        limitations = list(triang["limitations"])
        if rival.get("gap_recorded"):
            limitations.append(f"rival gap: {rival.get('gap', '')[:200]}")
        passed = pack["passed"] and rival["passed"]
        log_agent_action("CaseAgent", "protocol_complete", {
            "n_claims": triang["n_claims"], "n_flagged": triang["n_flagged"],
            "rivals": len(rival.get("rivals") or []),
            "gap_recorded": rival.get("gap_recorded"), "passed": passed})
        return {"pack": pack, "triangulation": triang, "rival": rival,
                "passed": passed, "findings": findings,
                "limitations": limitations}


#: Proposal §4.3 manifest name (search/fetch/read/write; no generate/execute).
CaseProtocolAgent = CaseAgent
