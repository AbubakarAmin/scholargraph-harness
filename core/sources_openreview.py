"""
OpenReview client: fetches submissions + their Official_Review replies for a
venue, and extracts reviewer-stated weaknesses as pre-formalized research gap
signals — literal expert-identified limitations, not LLM-inferred ones.


Auth is required (OpenReview API v2 needs a session even for public reads at
reasonable rate). Uses api2.openreview.net (v2) — v1 (api.openreview.net) is
legacy and many current venues (ICLR 2025+, NeurIPS 2024+) are v2-only.
"""


from __future__ import annotations


import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# Content field names to try, in priority order, per review-form convention.
# Different venues/years use different schemas; this list covers ICLR/NeurIPS
# (summary/strengths/weaknesses/questions) and TMLR (strengths_and_weaknesses).
_WEAKNESS_FIELD_CANDIDATES = [
    "weaknesses",
    "strengths_and_weaknesses",   # TMLR — contains both; extracted whole
    "limitations",
    "concerns",
]
_SUMMARY_FIELD_CANDIDATES = ["summary", "summary_of_contributions", "review"]


# Known venue invitation templates. Extend this dict as you add venues/years —
# invitation strings are brittle and NOT discoverable via a generic query, per
# OpenReview's own docs. Wrong string silently returns zero results (looks
# like "no papers" — it's actually "wrong invitation").
_VENUE_TEMPLATES = {
    "iclr2025": "ICLR.cc/2025/Conference",
    "iclr2026": "ICLR.cc/2026/Conference",
    "neurips2024": "NeurIPS.cc/2024/Conference",
    "neurips2025": "NeurIPS.cc/2025/Conference",
    "tmlr": "TMLR",
}

# Domain → venue keys (subset of _VENUE_TEMPLATES ONLY — never invent keys).
# Software engineering has no native OpenReview venue (ICSE/FSE/ASE use
# HotCRP), so SE runs use the ML venues (LLM-for-SE work appears there) plus
# TMLR's empirical track, and the hunter relevance-ranks the signals per seed.
_VENUES_BY_DOMAIN = {
    # e0ec9e91: TMLR cost ~5 min per run for a guaranteed 0 rows (wrong
    # invitation shape for the fetch path) — dropped from defaults. Restore
    # explicitly via OPENREVIEW_VENUES=...,tmlr if the template is ever fixed.
    "software_engineering": ["iclr2025", "iclr2026", "neurips2025"],
    "machine_learning": ["iclr2025", "iclr2026", "neurips2025"],
    "natural_language_processing": ["iclr2025", "iclr2026", "neurips2025"],
    "computer_vision": ["iclr2025", "iclr2026", "neurips2025"],
    "default": ["iclr2025", "iclr2026", "neurips2025"],
}


def venues_for_domain(domain: str, subcategory: str = "") -> List[str]:
    """Venue keys for a research domain (v5).

    Precedence: explicit OPENREVIEW_VENUES env override (comma-separated) >
    domain/subcategory map > default. Unknown keys are dropped (never sent —
    a wrong invitation silently returns zero rows).
    """
    try:
        import os as _os

        override = _os.getenv("OPENREVIEW_VENUES", "").strip()
        if override:
            keys = [k.strip().lower() for k in override.split(",") if k.strip()]
            valid = [k for k in keys if k in _VENUE_TEMPLATES]
            if valid:
                return valid
        key = (subcategory or "").strip().lower() or (domain or "").strip().lower()
        key = key.replace(" ", "_")
        venues = _VENUES_BY_DOMAIN.get(key, _VENUES_BY_DOMAIN["default"])
        try:
            from core.config import config as _cfg

            configured = list(getattr(_cfg, "openreview_venues", []) or [])
            if configured and set(configured) != set(_VENUES_BY_DOMAIN["default"]):
                venues = [v for v in configured if v in _VENUE_TEMPLATES] or venues
        except Exception:
            pass
        return [v for v in venues if v in _VENUE_TEMPLATES] or _VENUES_BY_DOMAIN["default"]
    except Exception:
        return _VENUES_BY_DOMAIN["default"]




class OpenReviewUnavailable(RuntimeError):
    pass




class OpenReviewClient:
    def __init__(self, username: str, password: str, enabled: bool = True):
        self.enabled = enabled and bool(username) and bool(password)
        self._client = None
        if not self.enabled:
            return
        try:
            # editdistance is a transitive dependency of openreview-py that
            # requires a C++ compiler. On Python 3.14+ there are no pre-built
            # wheels and compilation fails. We only use the OpenReview API
            # client (not editdistance), so stub it if genuinely unavailable.
            # This does NOT swallow errors from openreview itself.
            import editdistance  # noqa: F401
        except ImportError:
            import types as _types
            import sys as _sys
            if "editdistance" not in _sys.modules:
                _stub = _types.ModuleType("editdistance")
                _stub.eval = lambda a, b: 0  # type: ignore[attr-defined]
                _sys.modules["editdistance"] = _stub
        try:
            import openreview
            self._client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net",
                username=username,
                password=password,
            )
        except Exception as e:
            logger.warning("OpenReview auth failed, disabling source: %s", e, exc_info=True)
            self.enabled = False
            self._client = None


    def _extract_field(self, content: Dict[str, Any], candidates: List[str]) -> str:
        for key in candidates:
            entry = content.get(key)
            if isinstance(entry, dict) and entry.get("value"):
                return str(entry["value"])
            if isinstance(entry, str) and entry:
                return entry
        return ""


    def fetch_venue_gap_signals(
        self, venue_key: str, max_submissions: int = 60
    ) -> List[Dict[str, Any]]:
        """Return a list of {title, abstract, weaknesses, venue, year, arxiv_id?}
        dicts, one per submission that has at least one Official_Review with a
        non-empty weaknesses field. Shape is compatible with the existing
        `recent_papers` list consumed by TopicHunterAgent — plus an extra
        `weaknesses` key that callers can inject directly into gap-mining
        prompts as pre-formalized, expert-stated limitations.
        """
        if not self.enabled or self._client is None:
            return []
        venue_id = _VENUE_TEMPLATES.get(venue_key.lower())
        if not venue_id:
            logger.warning("Unknown OpenReview venue key: %s", venue_key)
            return []


        try:
            # Paginate with get_notes (not get_all_notes) to avoid fetching all
            # 11k+ submissions when max_submissions is small. Process as we go
            # and stop once we have enough results.
            PAGE_SIZE = min(max_submissions * 2, 200)  # over-fetch slightly since some lack reviews
            offset = 0
            results = []
            while len(results) < max_submissions:
                try:
                    from .api_gateway import get_gateway

                    _gw = get_gateway()
                    if not _gw.is_available("openreview"):
                        logger.warning("OpenReview breaker open — skipping paged read (offset=%s)", offset)
                        break

                    def _do_page(_inv: str = f"{venue_id}/-/Submission", _off: int = offset, _lim: int = PAGE_SIZE):
                        return self._client.get_notes(
                            invitation=_inv,
                            details="replies",
                            offset=_off,
                            limit=_lim,
                        )

                    page = _gw.request("openreview", _do_page, retries=2, backoff_base=3.0)
                except Exception:
                    page = self._client.get_notes(
                        invitation=f"{venue_id}/-/Submission",
                        details="replies",
                        offset=offset,
                        limit=PAGE_SIZE,
                    )
                if not page:
                    break
                for sub in page:
                    content = sub.content or {}
                    title = self._extract_field(content, ["title"])
                    abstract = self._extract_field(content, ["abstract"])
                    if not title:
                        continue

                    replies = (sub.details or {}).get("replies", [])
                    weaknesses_all = []
                    for reply in replies:
                        invitations = reply.get("invitations", []) or [reply.get("invitation", "")]
                        if not any(str(inv).endswith("/-/Official_Review") for inv in invitations):
                            continue
                        r_content = reply.get("content", {}) or {}
                        w = self._extract_field(r_content, _WEAKNESS_FIELD_CANDIDATES)
                        if w:
                            weaknesses_all.append(w)

                    if not weaknesses_all:
                        continue  # no value-add without reviewer weaknesses

                    results.append({
                        "title": title,
                        "abstract": abstract,
                        "weaknesses": weaknesses_all[:4],
                        "venue": venue_id,
                        "year": int(venue_id.split("/")[1]) if "/" in venue_id and venue_id.split("/")[1].isdigit() else 0,
                        "source": "openreview",
                        "doi": None,
                        "arxiv_id": None,
                    })
                    if len(results) >= max_submissions:
                        break
                offset += len(page)
        except Exception as e:
            logger.warning("OpenReview fetch failed for %s: %s", venue_id, e, exc_info=True)
            return []
        logger.info("OpenReview: %s -> %d papers with reviewer weaknesses", venue_id, len(results))
        return results




_or_client: Optional[OpenReviewClient] = None




def get_openreview_client() -> OpenReviewClient:
    global _or_client
    if _or_client is None:
        from core.config import config
        _or_client = OpenReviewClient(
            username=config.openreview_username,
            password=config.openreview_password,
            enabled=config.openreview_enabled,
        )
    return _or_client
