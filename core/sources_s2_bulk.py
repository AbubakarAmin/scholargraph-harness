"""
Semantic Scholar bulk search — free, no API key required for moderate volume,
higher effective throughput than arXiv's export API. Use as a primary text
search source alongside OpenAlex, with arXiv as a third/fallback leg rather
than the sole source, so an arXiv outage/429 storm doesn't stall discovery.
"""


from __future__ import annotations


import logging
from typing import Any, Dict, List, Optional


import requests


logger = logging.getLogger(__name__)


_S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


def search_s2_bulk(
    query: str,
    headers: Dict[str, str],
    limit: int = 50,
    year_from: Optional[int] = None,
    fields: str = "title,abstract,year,externalIds,citationCount",
) -> List[Dict[str, Any]]:
    try:
        from core.config import config as _cfg

        if not getattr(_cfg, "semantic_scholar_enabled", True):
            logger.debug("S2 bulk disabled via SEMANTIC_SCHOLAR_ENABLED — skipping query %r", query)
            return []
    except Exception:
        pass
    from core.api_gateway import get_gateway, RateLimitError

    def _do_fetch():
        params = {"query": query, "fields": fields, "limit": min(limit, 100)}
        if year_from:
            params["year"] = f"{year_from}-"
        r = requests.get(_S2_BULK_URL, headers=headers, params=params, timeout=30)
        if r.status_code == 429:
            raise RateLimitError("semantic_scholar_bulk", float(r.headers.get("Retry-After", "3")))
        r.raise_for_status()
        return r.json().get("data", [])

    try:
        from core.api_gateway import get_gateway as _get_gw

        _gw = _get_gw()
        if not _gw.is_available("semantic_scholar_bulk"):
            logger.debug("S2 bulk breaker open — skipping query %r", query)
            return []
        rows = _gw.request("semantic_scholar_bulk", _do_fetch, retries=3, backoff_base=3.0)
    except Exception as e:
        logger.debug("S2 bulk search failed for %r: %s", query, e)
        return []

    results = []
    for row in rows or []:
        ext = row.get("externalIds") or {}
        results.append({
            "title": row.get("title", ""),
            "abstract": row.get("abstract", "") or "",
            "year": row.get("year", 0) or 0,
            "doi": ext.get("DOI"),
            "arxiv_id": ext.get("ArXiv"),
            "cited_by_count": row.get("citationCount", 0) or 0,
            "source": "semantic_scholar_bulk",
        })
    return results
