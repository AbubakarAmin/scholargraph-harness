"""Structural gap mining via bibliographic coupling analysis."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def find_coupling_gaps(
    papers: List[Dict[str, Any]],
    s2_headers: Dict[str, str],
    base_url: str = "https://api.semanticscholar.org/graph/v1",
    min_shared_refs: int = 3,
    max_pairs: int = 8,
) -> List[Dict[str, Any]]:
    """Find bibliographic coupling gaps: pairs of papers that share many
    references but do not cite each other directly.

    All S2 calls route through the API gateway (pacing + breaker + adaptive
    rate). Fail-open: any API error or <2 resolvable papers → return [].
    """
    if not papers or len(papers) < 2:
        return []

    try:
        from core.config import config as _cfg

        if not getattr(_cfg, "semantic_scholar_enabled", True):
            logger.debug("find_coupling_gaps: SEMANTIC_SCHOLAR_ENABLED=false — skipping")
            return []
    except Exception:
        pass

    from .api_gateway import get_gateway, RateLimitError

    def _s2_get(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
        """Single gateway-managed attempt; gateway owns retries and pacing."""

        def _do_fetch():
            r = requests.get(url, headers=s2_headers, params=params or {}, timeout=15)
            if r.status_code == 429:
                raise RateLimitError("semantic_scholar", float(r.headers.get("Retry-After", "3")))
            return r

        try:
            if not get_gateway().is_available("semantic_scholar"):
                return None
            return get_gateway().request("semantic_scholar", _do_fetch, retries=1, backoff_base=2.0)
        except Exception as e:
            logger.debug("find_coupling_gaps: S2 fetch error for %s: %s", url, e)
            return None

    candidate_papers = papers[:20]
    paper_refs: Dict[str, List[str]] = {}
    paper_titles: Dict[str, str] = {}

    # Skip the whole loop when the semantic_scholar breaker is open — nothing will resolve.
    try:
        if not get_gateway().is_available("semantic_scholar"):
            return []
    except Exception:
        pass

    consecutive_misses = 0
    for p in candidate_papers:
        # Fail-fast: during an S2 429 storm every fetch burns ~10s of sleep
        # for nothing (run 21a145ad: 8 min of back-to-back 429s across one
        # seed's loop). After 3 straight misses the provider is clearly
        # throttling us — stop spending this seed's budget and return the
        # partial signal. The windowed throttle gate makes is_available()
        # False on the next seed, skipping the loop entirely.
        if consecutive_misses >= 3:
            logger.debug("find_coupling_gaps: %d consecutive S2 misses, skipping remaining %d papers",
                         consecutive_misses, len(candidate_papers))
            break
        raw_doi = (p.get("doi") or "").replace("https://doi.org/", "").strip()
        raw_arxiv = (p.get("arxiv_id") or "").strip()
        raw_s2 = (p.get("s2_paper_id") or "").strip()
        if raw_s2:
            paper_id = raw_s2
        elif raw_doi:
            paper_id = f"DOI:{raw_doi}"
        elif raw_arxiv:
            arxiv_num = raw_arxiv.split("/")[-1] if "/" in raw_arxiv else raw_arxiv
            paper_id = f"ARXIV:{arxiv_num}"
        else:
            continue
        title = p.get("title", "unknown")
        paper_titles[paper_id] = title
        url = f"{base_url}/paper/{paper_id}"
        r = _s2_get(url, params={"fields": "references.paperId,citations.paperId"})
        if r is None or r.status_code != 200:
            paper_refs[paper_id] = []
            consecutive_misses += 1
            continue
        consecutive_misses = 0
        try:
            data = r.json()
            ref_ids = set()
            for ref in data.get("references") or []:
                rid = ref.get("paperId")
                if rid:
                    ref_ids.add(rid)
            paper_refs[paper_id] = list(ref_ids)
            paper_titles[paper_id] = data.get("title") or title
        except Exception as e:
            logger.debug("find_coupling_gaps: S2 parse error for %s: %s", paper_id, e)
            paper_refs[paper_id] = []

    resolved_ids = [pid for pid, refs in paper_refs.items() if refs]
    if len(resolved_ids) < 2:
        return []

    # Precompute citation sets for direct-citation check
    citation_sets: Dict[str, set] = {}
    for pid in resolved_ids:
        citation_sets[pid] = set(paper_refs.get(pid, []))

    pairs: List[Dict[str, Any]] = []
    for i in range(len(resolved_ids)):
        for j in range(i + 1, len(resolved_ids)):
            a_id = resolved_ids[i]
            b_id = resolved_ids[j]
            refs_a = citation_sets.get(a_id, set())
            refs_b = citation_sets.get(b_id, set())
            shared = refs_a & refs_b
            if len(shared) < min_shared_refs:
                continue
            # No direct citation check
            if b_id in refs_a or a_id in refs_b:
                continue
            pairs.append({
                "paper_a": paper_titles.get(a_id, a_id),
                "paper_b": paper_titles.get(b_id, b_id),
                "paper_a_id": a_id,
                "paper_b_id": b_id,
                "shared_reference_count": len(shared),
                "gap_type": "bibliographic_coupling_no_direct_citation",
            })

    pairs.sort(key=lambda x: x["shared_reference_count"], reverse=True)
    return pairs[:max_pairs]


def find_coupling_gaps_openalex(
    papers: List[Dict[str, Any]],
    min_shared_refs: int = 3,
    max_pairs: int = 8,
) -> List[Dict[str, Any]]:
    """Bibliographic coupling from OpenAlex `referenced_works` (no S2 needed).

    Pure in-memory join over the rows OpenAlex search already returned — works
    when Semantic Scholar is disabled. A pair couples when both papers cite
    >= min_shared_refs of the same OpenAlex work IDs and neither lists the
    other as a reference (approximated via DOI/title cross-check).
    Fail-open: missing `referenced_works` fields or <2 usable rows → [].
    """
    try:
        if not papers or len(papers) < 2:
            return []
        refs: Dict[str, set] = {}
        titles: Dict[str, str] = {}
        dois: Dict[str, str] = {}
        order: List[str] = []
        for i, p in enumerate(papers):
            if not isinstance(p, dict):
                continue
            pid = (
                (p.get("doi") or "").strip().lower()
                or (p.get("arxiv_id") or "").strip().lower()
                or f"row-{i}"
            )
            works = p.get("referenced_works") or []
            ref_ids = set()
            for w in works:
                wid = str(w or "").strip()
                if wid:
                    ref_ids.add(wid.rstrip("/").split("/")[-1].upper())
            if not ref_ids:
                continue
            refs[pid] = ref_ids
            titles[pid] = (p.get("title") or pid)[:150]
            dois[pid] = (p.get("doi") or "").strip().lower().replace("https://doi.org/", "")
            order.append(pid)
        if len(order) < 2:
            return []
        pairs: List[Dict[str, Any]] = []
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a_id, b_id = order[i], order[j]
                shared = refs[a_id] & refs[b_id]
                if len(shared) < min_shared_refs:
                    continue
                # No-direct-citation approximation: neither paper's DOI appears
                # in the other's reference id set, and titles differ.
                if dois[a_id] and any(dois[a_id] in r.lower() for r in refs[b_id]):
                    continue
                if dois[b_id] and any(dois[b_id] in r.lower() for r in refs[a_id]):
                    continue
                pairs.append({
                    "paper_a": titles[a_id],
                    "paper_b": titles[b_id],
                    "paper_a_id": a_id,
                    "paper_b_id": b_id,
                    "shared_reference_count": len(shared),
                    "gap_type": "bibliographic_coupling_no_direct_citation",
                    "source": "openalex_native",
                })
        pairs.sort(key=lambda x: x["shared_reference_count"], reverse=True)
        return pairs[:max_pairs]
    except Exception as e:
        logger.debug("find_coupling_gaps_openalex failed: %s", e)
        return []
