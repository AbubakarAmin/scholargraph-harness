"""ThesisAgent — thesis construction + verification for no-experiment paper types.

Proposal §4 (M1): analytical / descriptive / argumentative (and review synthesis
in M2) carry a structured thesis instead of experiment contracts:

    thesis = {"claim": str, "bridges": [...], "steelman": str, "concession": str}

- ``build_thesis`` mints the thesis from the topic + retrieved literature
  evidence (one self-correcting LLM call, deterministic fallback).
- ``verify_thesis`` is pure/deterministic: thesis triad (claim + bridges +
  steelman/concession) present AND 100% bridge coverage in the manuscript
  (§7.1 release rule for analytical/argumentative).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action, call_llm_json
from core.llm import call_llm, get_llm_client
from core.context import RunContext, get_active_context
from core.memory import memory

logger = logging.getLogger(__name__)


def _thesis_system(**slots) -> dict:
    """System kwarg from the v2 registry (stub-safe: {} when unavailable)."""
    try:
        from core.prompts import render_system

        system = render_system("thesis.build", **slots)
        return {"system": system} if system else {}
    except Exception:
        return {}

# Paper types whose release rule requires a verified thesis (§7.1–§7.2).
THESIS_TYPES: tuple[str, ...] = ("analytical", "descriptive", "argumentative", "review")

# Budget cap: at most 12 evidence bridges per thesis (§5.3).
MAX_THESIS_BRIDGES = 12

# Minimum token recall for one bridge statement to count as covered by the
# manuscript text (deterministic, offline-testable).
BRIDGE_COVERAGE_RECALL = 0.5


def _content_tokens(text: str) -> List[str]:
    return [word for word in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(word) > 3]


def _fallback_thesis(
    topic: Dict[str, Any],
    literature_evidence: List[Dict[str, Any]],
    paper_type: str,
) -> Dict[str, Any]:
    """Deterministic thesis when the LLM call fails (fail-closed, never empty)."""
    structured = topic.get("structured_hypothesis") if isinstance(topic, dict) else {}
    claim = ""
    if isinstance(structured, dict):
        claim = str(structured.get("hypothesis") or structured.get("research_question") or "")
    if not claim:
        claim = str((topic or {}).get("description") or (topic or {}).get("title") or "Unstated thesis")
    bridges = []
    for paper in (literature_evidence or [])[:MAX_THESIS_BRIDGES]:
        if not isinstance(paper, dict):
            continue
        title = str(paper.get("title") or "").strip()
        if not title:
            continue
        bridges.append({
            "statement": f"Prior work ({title}) establishes relevant background for this thesis.",
            "source": paper.get("doi") or paper.get("arxiv_id") or title,
        })
    return {
        "claim": claim,
        "bridges": bridges,
        "steelman": "The strongest counter-reading is that the available evidence is insufficient to support the claim as stated.",
        "concession": "This thesis concedes its scope limits where the retrieved evidence is thin.",
        "paper_type": paper_type,
        "fallback_used": True,
    }


class ThesisAgent:
    """Build and verify structured theses for no-experiment paper types."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    @property
    def vector_memory(self):
        return self.context.memory if self.context else memory

    def build_thesis(
        self,
        topic: Dict[str, Any],
        literature_evidence: Optional[List[Dict[str, Any]]] = None,
        paper_type: str = "analytical",
    ) -> Dict[str, Any]:
        """Mint {claim, bridges[], steelman, concession} for one topic."""
        topic = topic or {}
        evidence = literature_evidence if literature_evidence is not None else topic.get("literature_evidence") or []
        slim = []
        for paper in evidence[:MAX_THESIS_BRIDGES]:
            if not isinstance(paper, dict):
                continue
            slim.append({
                "title": paper.get("title", ""),
                "abstract": (paper.get("abstract") or "")[:400],
                "doi": paper.get("doi"),
                "arxiv_id": paper.get("arxiv_id"),
            })
        log_agent_action("ThesisAgent", "build_thesis_start", {
            "topic": topic.get("title", "<untitled>"),
            "paper_type": paper_type,
            "n_evidence": len(slim),
        })
        prompt = f"""You are building the structured thesis for a {paper_type} research paper.
A thesis has four parts: a central claim, evidence bridges to prior work, the
strongest counter-reading (steelman), and an explicit concession of scope limits.

Topic: {topic.get('title', '<untitled>')}
Description: {topic.get('description', '')}

Retrieved literature evidence (bridges must cite ONLY these sources):
{json.dumps(slim, default=str)[:8000]}

Return JSON only:
{{"claim": "one falsifiable-or-arguable central sentence",
  "bridges": [{{"statement": "how this source supports or bounds the claim",
                "source": "DOI or arXiv ID from the evidence above"}}],
  "steelman": "the strongest legitimate counter-reading of the claim",
  "concession": "what the thesis explicitly concedes"}}
"""
        try:
            parsed = call_llm_json(prompt, temperature=0.4, tier="strong", attempts=2, call_fn=call_llm,
                                   **_thesis_system(paper_type=paper_type))
        except Exception as exc:
            logger.error("Thesis build failed: %s", exc, exc_info=True)
            parsed = None
        thesis: Dict[str, Any] = {}
        if isinstance(parsed, dict) and str(parsed.get("claim") or "").strip():
            bridges = [b for b in (parsed.get("bridges") or []) if isinstance(b, dict) and str(b.get("statement") or "").strip()]
            thesis = {
                "claim": str(parsed["claim"]).strip(),
                "bridges": bridges[:MAX_THESIS_BRIDGES],
                "steelman": str(parsed.get("steelman") or "").strip(),
                "concession": str(parsed.get("concession") or "").strip(),
                "paper_type": paper_type,
                "fallback_used": False,
            }
        else:
            thesis = _fallback_thesis(topic, evidence, paper_type)
        # Triad completeness is enforced downstream by verify_thesis; log gaps now.
        triad = self._triad_status(thesis)
        log_agent_action("ThesisAgent", "build_thesis_complete", {
            "paper_type": paper_type,
            "triad": triad,
            "n_bridges": len(thesis.get("bridges") or []),
            "fallback_used": bool(thesis.get("fallback_used")),
        })
        return thesis

    @staticmethod
    def _triad_status(thesis: Dict[str, Any]) -> Dict[str, bool]:
        bridges = thesis.get("bridges") or []
        return {
            "claim": bool(str(thesis.get("claim") or "").strip()),
            "bridges": bool(bridges) and all(
                isinstance(b, dict) and bool(str(b.get("statement") or "").strip()) for b in bridges
            ),
            "steelman_concession": bool(str(thesis.get("steelman") or "").strip())
            and bool(str(thesis.get("concession") or "").strip()),
        }


def verify_thesis(
    thesis: Optional[Dict[str, Any]],
    sections: Dict[str, str],
    literature_evidence: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Deterministic thesis gate (§7.1): triad present AND 100% bridge coverage.

    Pure function (no LLM): every bridge statement must be covered by the
    manuscript text (token recall >= BRIDGE_COVERAGE_RECALL). Returns
    ``passed`` plus per-bridge coverage for audit.
    """
    thesis = thesis or {}
    sections = sections or {}
    manuscript = "\n".join(v for v in sections.values() if isinstance(v, str))
    manuscript_tokens = set(_content_tokens(manuscript))
    triad = ThesisAgent._triad_status(thesis)
    covered: List[Dict[str, Any]] = []
    uncovered: List[Dict[str, Any]] = []
    for bridge in thesis.get("bridges") or []:
        if not isinstance(bridge, dict):
            continue
        statement = str(bridge.get("statement") or "")
        tokens = _content_tokens(statement)
        if not tokens:
            uncovered.append({"bridge": bridge, "recall": 0.0, "reason": "empty bridge statement"})
            continue
        hits = sum(1 for t in set(tokens) if t in manuscript_tokens)
        recall = hits / len(set(tokens))
        record = {"bridge": bridge, "recall": round(recall, 3)}
        (covered if recall >= BRIDGE_COVERAGE_RECALL else uncovered).append(record)
    bridges_ok = bool(thesis.get("bridges")) and not uncovered
    passed = all(triad.values()) and bridges_ok
    findings = []
    for part, ok in triad.items():
        if not ok:
            findings.append({"check": "thesis_triad", "part": part, "details": f"thesis part '{part}' missing or empty"})
    for item in uncovered:
        findings.append({"check": "bridge_coverage", "details": item})
    return {
        "passed": passed,
        "triad": triad,
        "bridge_coverage": 1.0 if not (covered or uncovered) else round(len(covered) / max(1, len(covered) + len(uncovered)), 3),
        "bridges_covered": len(covered),
        "bridges_total": len(covered) + len(uncovered),
        "uncovered": uncovered,
        "findings": findings,
    }
