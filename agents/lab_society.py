"""Lab society helpers: teamwork workflows on top of existing agents.

Non-invasive: existing agents keep their signatures; these helpers add
fan-out, enrichment, handoffs, and standups via ResearchState messages/claims.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _persist_bus(state: Dict[str, Any], event_type: str, payload: Dict[str, Any], agent: str) -> None:
    """Persist lab-bus messages/claims to the research ledger (best-effort).

    The bus previously lived only in LangGraph state (lost on crash, never
    in research_db). Ledger writes are fire-and-forget: failures never break
    the run.
    """
    try:
        from core.research_db import research_db

        run_id = None
        try:
            from core.run_log import get_tracker

            tracker = get_tracker()
            run_id = tracker.run_id if tracker else None
        except Exception:
            pass
        research_db.record_event(event_type, payload, run_id=run_id, agent=agent)
    except Exception as exc:
        logger.debug("lab bus persist failed: %s", exc)


def hunter_fan_out(topics: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """Select top-k diverse topics for parallel debate (not just topics[0])."""
    ranked = [t for t in (topics or []) if isinstance(t, dict)]
    # Prefer higher novelty/score when present, else keep order.
    def _score(t: Dict[str, Any]) -> float:
        for key in ("novelty_score", "score", "rank_score"):
            try:
                return float(t.get(key) or 0.0)
            except Exception:
                continue
        return 0.0
    ranked = sorted(ranked, key=_score, reverse=True)
    # Diversity: drop near-duplicate titles.
    seen: List[str] = []
    out: List[Dict[str, Any]] = []
    for t in ranked:
        title = str(t.get("title") or "").lower()[:80]
        if any(title and title in s for s in seen):
            continue
        seen.append(title)
        out.append(t)
        if len(out) >= max(1, int(k or 3)):
            break
    return out or ranked[: max(1, int(k or 3))]


def enrich_topics_with_lab_graph(
    topics: List[Dict[str, Any]],
    state: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Attach citation-graph + contradiction + evidence bridges to each topic.

    Best-effort, never raises: failures leave the topic unenriched with a note.
    All dispatch goes through the broker-only entry point (audited, manifest
    authorized) — never direct registry access.
    """
    from core.tool_broker import broker_call

    enriched: List[Dict[str, Any]] = []
    for t in topics or []:
        if not isinstance(t, dict):
            continue
        t = dict(t)
        try:
            papers = t.get("literature_evidence") or []
            if papers:
                res = broker_call("TopicHunterAgent", "evidence_synthesize",
                                  state, papers=papers)
                t["evidence_bridges"] = (res or {}).get("evidence_map", {})
                res = broker_call("TopicHunterAgent", "contradiction_scan",
                                  state, papers=papers)
                t["contradictions"] = (res or {}).get("contradictions", [])
            # Citation neighbors for the top paper (cheap, 1 query).
            if papers and isinstance(papers[0], dict):
                top = papers[0]
                res = broker_call(
                    "TopicHunterAgent", "citation_graph", state,
                    doi=str(top.get("doi") or ""),
                    arxiv_id=str(top.get("arxiv_id") or ""))
                t["citation_neighbors"] = (res or {}).get("papers", [])[:5]
        except Exception as exc:
            t["lab_enrich_note"] = str(exc)[:160]
        enriched.append(t)
    return enriched


def post_handoff(state: Dict[str, Any], frm: str, to: str, payload: Dict[str, Any]) -> None:
    msgs = list(state.get("messages") or [])
    msg = {"from": frm, "to": to, "type": "handoff", "payload": dict(payload), "obs_ids": []}
    msgs.append(msg)
    state["messages"] = msgs
    _persist_bus(state, "lab_handoff", {"from": frm, "to": to, **dict(payload)}, frm)


def post_repair_request(state: Dict[str, Any], frm: str, to: str, reason: str, details: Dict[str, Any] | None = None) -> None:
    msgs = list(state.get("messages") or [])
    payload = {"reason": reason, **(details or {})}
    msg = {"from": frm, "to": to, "type": "repair_request", "payload": payload, "obs_ids": []}
    msgs.append(msg)
    # Planner<->Engineer compat: also append to plan_revision_requests.
    if to.lower().startswith("plan"):
        reqs = list(state.get("plan_revision_requests") or [])
        reqs.append(payload)
        state["plan_revision_requests"] = reqs
    state["messages"] = msgs
    _persist_bus(state, "lab_repair_request", {"from": frm, "to": to, **payload}, frm)


def post_claim(state: Dict[str, Any], claim_id: str, text: str, owner: str, evidence_ids: List[str] | None = None) -> None:
    claims = list(state.get("claims") or [])
    claim = {
        "claim_id": claim_id, "text": text, "owner": owner,
        "evidence_ids": list(evidence_ids or []), "status": "proposed",
    }
    claims.append(claim)
    state["claims"] = claims
    _persist_bus(state, "lab_claim", dict(claim), owner)
