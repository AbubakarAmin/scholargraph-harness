"""
One-time ingestion script: pull ICLR review data from OpenReview (API v1 + v2).

Produces a local, hash-pinned dataset of papers paired with their real review
outcomes (scores, accept/reject decision, and meta-review text) for
calibrating Supervisor.evaluate_section() and Editor.final_manuscript_referee()
in Phase 4b.

SCOPE: ICLR 2021-2026 (six consecutive years). Real submission counts you
should expect (for sizing/runtime planning, not hardcoded anywhere):
roughly 2021 ~3.0k, 2022 ~2.8k, 2023 ~5.0k, 2024 ~7.4k, 2025 ~11.7k,
2026 ~19.5k valid submissions (~13.8k with a final decision) -- around
49,000 submissions total across all six years. This is a genuinely large,
multi-hour pull; the resumability described below is not optional polish.

TWO API VERSIONS - THIS IS THE PART THAT CHANGED:
OpenReview migrated hosts between ICLR 2023 and ICLR 2024. Venues before
2024 live on the legacy API v1 host (api.openreview.net, invitation-based
queries, flat JSON content). ICLR 2024 onward lives on API v2
(api2.openreview.net, group-based /notes/search, {"value": ...}-wrapped
JSON content). Querying a pre-2024 venue through the v2 host silently
returns zero results -- it does not error, it just finds nothing, which is
exactly what happened on the first run of the earlier version of this
script. This version authenticates against BOTH hosts (same OpenReview
username/password works on both, but each host issues its own bearer
token) and routes each venue to the correct one based on its
`api_version` field below.

SCALE DRIFT WARNING: ICLR 2026's overall "rating" field uses a different,
discrete scale ({0, 2, 4, 6, 8, 10}) than prior years. This is exactly why
every score field here stores both the `raw` value and a best-effort
`numeric` parse, plus the paper's `year` -- Phase 4b must normalize
per-year before comparing across years, not average raw numbers blindly.

RESUMABILITY (this is the main point of this rewrite, and matters even
more now that the real dataset is ~49k papers, not ~21k):
  - Every paper is written to its own JSON file the moment it's fetched.
    On any interruption (network drop, rate limit exhaustion, Ctrl-C,
    crash), simply re-run the same command. Papers whose output file
    already exists are skipped by default -- nothing is re-downloaded.
  - Use --force to re-fetch papers even if a local file already exists
    (e.g. you fixed an extraction bug, or you ran once without
    credentials and want to backfill review data).
  - The submission list itself is cached per venue in
    `_submissions_cache.json` so a restart doesn't need to re-run
    pagination from scratch. Use --refresh-submissions to bypass it.
  - Ctrl-C (SIGINT) triggers a clean shutdown: in-flight work finishes or
    is abandoned safely (nothing partially written), and the manifest is
    regenerated from whatever is actually on disk before exiting.

CONCURRENCY:
  - Reply-fetching runs across a small thread pool (default 4 workers,
    --workers to change). Each worker gets its own `requests.Session`
    (not shared -- Session objects aren't documented thread-safe),
    carrying the correct bearer token for whichever API version that
    venue uses. A single shared, lock-protected adaptive delay throttles
    the aggregate request rate per API host, so raising --workers doesn't
    bypass rate limiting.

Usage:
    python ingest_openreview.py                     # pull ICLR 2021-2026
    python ingest_openreview.py --venue ICLR_2021 ICLR_2022
    python ingest_openreview.py --dry-run           # list-only, no downloads
    python ingest_openreview.py --max-papers 25     # small smoke test
    python ingest_openreview.py --force             # re-fetch existing files
    python ingest_openreview.py --refresh-submissions
    python ingest_openreview.py --workers 8

Output directory: data/review_calibration/{venue_label}/{paper_id}.json
Manifest:         data/review_calibration/_manifest.json

AUTHENTICATION: both API hosts return a 403 ChallengeRequiredError (v2) or
simply omit reviewer-only content (v1) for unauthenticated ("guest")
requests to the endpoint that returns reviews/decisions. ICLR's review
process is public by design -- this looks like anti-bot / permission
handling reacting to non-browser traffic, not a genuine "reviews are
private" policy. Logging in with a real OpenReview account (free) clears
it on both hosts. If you hit persistent 403s while authenticated on a
GIVEN host, that is worth escalating -- it would mean this assumption is
wrong for that host specifically.

    OPENREVIEW_USERNAME=you@example.com
    OPENREVIEW_PASSWORD=your_password

in your .env. Without credentials the script still saves paper metadata
(title/authors/abstract) but reviews/decisions/meta-reviews will be empty
for every venue, on both API versions.

TODO before treating this dataset as done: register it in core.datasets'
catalog using the same pattern as the PMLB/OpenML-CC18 entries (this
script only produces the hash-pinned files + manifest; it does not call
into core.datasets itself, since that module wasn't in scope to modify
here without reading it first per the phase preamble). See the updated
Phase 4b prompt -- this registration is now step 1 there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load .env from project root (same convention as core/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_V1 = "https://api.openreview.net"       # legacy - pre-2024 venues
API_BASE_V2 = "https://api2.openreview.net"      # current - 2024+ venues

SEARCH_BATCH_SIZE = 1000
SEARCH_MAX_RESULTS = 25000     # raised - 2026 alone has ~19.5k submissions

RATE_LIMIT_BASE = float(os.environ.get("OR_RATE_LIMIT_BASE", "0.5"))
RATE_LIMIT_MAX = float(os.environ.get("OR_RATE_LIMIT_MAX", "30.0"))
RATE_LIMIT_GROWTH = 1.5
RATE_LIMIT_JITTER = 0.25

DEFAULT_WORKERS = int(os.environ.get("OR_WORKERS", "4"))

OUTPUT_ROOT = Path(__file__).resolve().parent / "data" / "review_calibration"

# api_version drives which host + query style is used. submission_invitations
# is only used for v1 venues, tried in order until one returns results (older
# venues use "Blind_Submission"; some later-migrated ones use "Submission").
VENUES = [
    {"venue_id": "ICLR.cc/2021/Conference", "year": 2021, "label": "ICLR_2021",
     "api_version": "v1", "submission_invitations": ["Blind_Submission", "Submission"]},
    {"venue_id": "ICLR.cc/2022/Conference", "year": 2022, "label": "ICLR_2022",
     "api_version": "v1", "submission_invitations": ["Blind_Submission", "Submission"]},
    {"venue_id": "ICLR.cc/2023/Conference", "year": 2023, "label": "ICLR_2023",
     "api_version": "v1", "submission_invitations": ["Blind_Submission", "Submission"]},
    {"venue_id": "ICLR.cc/2024/Conference", "year": 2024, "label": "ICLR_2024",
     "api_version": "v2"},
    {"venue_id": "ICLR.cc/2025/Conference", "year": 2025, "label": "ICLR_2025",
     "api_version": "v2"},
    {"venue_id": "ICLR.cc/2026/Conference", "year": 2026, "label": "ICLR_2026",
     "api_version": "v2"},
]
VENUE_LABELS = [v["label"] for v in VENUES]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_openreview")

_shutdown_requested = threading.Event()


def _handle_sigint(signum, frame):  # noqa: ARG001
    if _shutdown_requested.is_set():
        log.warning("Second interrupt received, exiting immediately.")
        sys.exit(1)
    log.warning("Interrupt received - finishing in-flight requests, then "
                "writing a manifest for what's on disk so far. Re-run the "
                "same command to resume; Ctrl-C again to force-quit.")
    _shutdown_requested.set()


signal.signal(signal.SIGINT, _handle_sigint)


# ---------------------------------------------------------------------------
# Shared, thread-safe adaptive rate limiter (one per API host, since v1 and
# v2 are different services with independent, unrelated rate limits)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, base: float, ceiling: float, growth: float):
        self._delay = base
        self._base = base
        self._ceiling = ceiling
        self._growth = growth
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._delay
        jitter = delay * RATE_LIMIT_JITTER
        time.sleep(max(0.0, delay + random.uniform(-jitter, jitter)))

    def note_response(self, resp: Optional[requests.Response]) -> None:
        if resp is None:
            return
        with self._lock:
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_s = min(float(retry_after), self._ceiling)
                    except ValueError:
                        wait_s = min(self._delay * self._growth, self._ceiling)
                else:
                    wait_s = min(self._delay * self._growth, self._ceiling)
                self._delay = wait_s
                log.warning("Rate limited (429). New shared delay: %.1fs", wait_s)
                return
            if resp.status_code == 200:
                remaining = resp.headers.get("X-RateLimit-Remaining")
                if remaining:
                    try:
                        remaining_n = int(remaining)
                        if remaining_n < 10:
                            self._delay = min(self._delay * 1.2, self._ceiling)
                        elif remaining_n > 50 and self._delay > self._base:
                            self._delay = max(self._delay * 0.9, self._base)
                    except (ValueError, TypeError):
                        pass

    def block_for_429(self, resp: requests.Response) -> None:
        retry_after = resp.headers.get("Retry-After")
        try:
            wait_s = min(float(retry_after), self._ceiling) if retry_after else self._delay
        except ValueError:
            wait_s = self._delay
        time.sleep(wait_s)


_rate_limiters: Dict[str, RateLimiter] = {
    "v1": RateLimiter(RATE_LIMIT_BASE, RATE_LIMIT_MAX, RATE_LIMIT_GROWTH),
    "v2": RateLimiter(RATE_LIMIT_BASE, RATE_LIMIT_MAX, RATE_LIMIT_GROWTH),
}


def _api_version_for_url(url: str) -> str:
    return "v1" if url.startswith(API_BASE_V1) else "v2"


# ---------------------------------------------------------------------------
# Authentication + per-thread, per-API-version sessions
# ---------------------------------------------------------------------------

def _build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=1.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=DEFAULT_WORKERS + 2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "ScholarGraph-Ingest/3.0 (+local research calibration dataset)",
        "Accept": "application/json",
    })
    return session


def authenticate(base_url: str) -> Optional[str]:
    """Log in against a specific API host and return its bearer token, or
    None if credentials are absent or login failed on that host."""
    username = os.environ.get("OPENREVIEW_USERNAME", "")
    password = os.environ.get("OPENREVIEW_PASSWORD", "")
    if not (username and password):
        return None

    session = _build_http_session()
    try:
        resp = session.post(f"{base_url}/login",
                             json={"id": username, "password": password}, timeout=30)
        if resp.status_code == 200:
            token = resp.json().get("token", "")
            if token:
                log.info("Authenticated with OpenReview (%s) as %s", base_url, username)
                return token
        log.warning("OpenReview login failed on %s (status %d): %s",
                     base_url, resp.status_code, resp.text[:300])
    except requests.exceptions.RequestException as e:
        log.warning("OpenReview login request failed on %s: %s", base_url, e)
    return None


_thread_local = threading.local()
_auth_tokens: Dict[str, Optional[str]] = {"v1": None, "v2": None}
_auth_lock = threading.Lock()


def get_thread_session(api_version: str) -> requests.Session:
    """One Session per thread per API version, each carrying that
    version's own bearer token."""
    attr = f"session_{api_version}"
    session = getattr(_thread_local, attr, None)
    if session is None:
        session = _build_http_session()
        token = _auth_tokens.get(api_version)
        if token:
            session.headers["Authorization"] = f"Bearer {token}"
        setattr(_thread_local, attr, session)
    return session


def reauthenticate_if_needed(resp: requests.Response, api_version: str, base_url: str) -> None:
    if resp.status_code != 401:
        return
    with _auth_lock:
        log.warning("Got 401 on %s API - token may have expired. Re-authenticating...", api_version)
        new_token = authenticate(base_url)
        if new_token:
            _auth_tokens[api_version] = new_token
            attr = f"session_{api_version}"
            if hasattr(_thread_local, attr):
                delattr(_thread_local, attr)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _request_with_retries(
    api_version: str,
    base_url: str,
    method: str,
    url: str,
    max_attempts: int = 5,
    **kwargs,
) -> Optional[requests.Response]:
    rate_limiter = _rate_limiters[api_version]
    for attempt in range(max_attempts):
        if _shutdown_requested.is_set():
            return None
        session = get_thread_session(api_version)
        try:
            resp = session.request(method, url, timeout=60, **kwargs)
        except requests.exceptions.RequestException as e:
            wait = min(2 ** (attempt + 1), 30)
            log.warning("Request error (%s), retrying in %ds: %s", url, wait, e)
            time.sleep(wait)
            continue

        rate_limiter.note_response(resp)

        if resp.status_code == 429:
            rate_limiter.block_for_429(resp)
            continue
        if resp.status_code == 401:
            reauthenticate_if_needed(resp, api_version, base_url)
            continue

        return resp
    return None


def _search_notes_v2(venue_id: str, max_results: int) -> List[Dict[str, Any]]:
    """API v2: /notes/search by group, offset/limit pagination."""
    params = {"query": "*", "group": venue_id, "source": "forum",
              "limit": min(SEARCH_BATCH_SIZE, max_results), "offset": 0}
    all_notes: List[Dict[str, Any]] = []

    while len(all_notes) < max_results and not _shutdown_requested.is_set():
        resp = _request_with_retries("v2", API_BASE_V2, "GET",
                                       f"{API_BASE_V2}/notes/search", params=params)
        if resp is None or resp.status_code != 200:
            log.error("v2 search failed at offset %s (status %s), stopping.",
                       params["offset"], getattr(resp, "status_code", "n/a"))
            break
        data = resp.json()
        notes = data.get("notes", [])
        if not notes:
            break
        all_notes.extend(notes)
        log.info("  Search (v2): fetched %d notes (running total: %d)", len(notes), len(all_notes))
        if len(notes) < params["limit"]:
            break
        params["offset"] += params["limit"]

    return all_notes[:max_results]


def _search_notes_v1(venue_id: str, invitation_suffixes: List[str], max_results: int) -> List[Dict[str, Any]]:
    """API v1: /notes filtered by invitation, offset/limit pagination.
    Tries each invitation suffix in order (e.g. Blind_Submission, then
    Submission) and uses the first one that returns any results."""
    for suffix in invitation_suffixes:
        invitation = f"{venue_id}/-/{suffix}"
        params = {"invitation": invitation,
                  "limit": min(SEARCH_BATCH_SIZE, max_results), "offset": 0}
        all_notes: List[Dict[str, Any]] = []

        while len(all_notes) < max_results and not _shutdown_requested.is_set():
            resp = _request_with_retries("v1", API_BASE_V1, "GET",
                                           f"{API_BASE_V1}/notes", params=params)
            if resp is None or resp.status_code != 200:
                log.warning("  v1 search failed for invitation=%s (status %s).",
                             invitation, getattr(resp, "status_code", "n/a"))
                break
            data = resp.json()
            notes = data.get("notes", [])
            if not notes:
                break
            all_notes.extend(notes)
            log.info("  Search (v1, %s): fetched %d notes (running total: %d)",
                       suffix, len(notes), len(all_notes))
            if len(notes) < params["limit"]:
                break
            params["offset"] += params["limit"]

        if all_notes:
            log.info("  v1 invitation '%s' matched - using this suffix for %s.", suffix, venue_id)
            return all_notes[:max_results]
        log.info("  v1 invitation suffix '%s' returned 0 results for %s, trying next.",
                   suffix, venue_id)

    log.error("  No v1 invitation suffix returned results for %s. Submissions "
               "for this venue may use a different invitation naming scheme - "
               "check https://openreview.net/group?id=%s manually.", venue_id, venue_id)
    return []


def _get_note_replies(api_version: str, forum_id: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch all replies to a forum note. Works the same way on both API
    versions (GET /notes?forum=...), just against different hosts.
    Returns None (distinct from an empty list) on a genuine
    challenge/permission wall so callers can tell 'no replies yet' apart
    from 'couldn't read replies'."""
    base_url = API_BASE_V1 if api_version == "v1" else API_BASE_V2
    resp = _request_with_retries(api_version, base_url, "GET", f"{base_url}/notes",
                                   params={"forum": forum_id, "trash": "false"})
    if resp is None:
        return None
    if resp.status_code == 403:
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if data.get("name") == "ChallengeRequiredError":
            return None
        return None
    if resp.status_code != 200:
        return None
    return resp.json().get("notes", [])


# ---------------------------------------------------------------------------
# Field extraction (handles both v1-flat and v2-{value:...} content shapes)
# ---------------------------------------------------------------------------

def _extract_text(field_val: Any) -> str:
    if isinstance(field_val, dict):
        field_val = field_val.get("value", "")
    return field_val if isinstance(field_val, str) else ""


def _extract_list(field_val: Any) -> List[Any]:
    if isinstance(field_val, dict):
        field_val = field_val.get("value", [])
    return field_val if isinstance(field_val, list) else []


# Historical ICLR review forms have used different literal key names for
# what is conceptually the same field. "rating" is the current name; older
# years (or years where OpenReview back-ported/varied the form) have used
# other names for the overall score specifically. soundness/presentation/
# contribution are genuinely new (ACL-style rubric, ~2023+) and legitimately
# absent in older years - that's expected, not a bug. A missing "rating" on
# every review of a paper, across many papers in a venue, is NOT expected
# and means the real key name for that venue-year needs to be found via the
# field audit this function now also produces.
SCORE_FIELD_ALIASES: Dict[str, List[str]] = {
    "rating": ["rating", "recommendation", "overall_rating", "review_rating", "score"],
    "confidence": ["confidence", "reviewer_confidence"],
    "soundness": ["soundness", "technical_quality", "correctness"],
    "presentation": ["presentation", "clarity"],
    "contribution": ["contribution", "originality", "novelty"],
}


def _extract_scores(review_content: Dict[str, Any]) -> Dict[str, Any]:
    """Extract score-like fields, keeping the raw value alongside any
    numeric parse - scales differ across years (e.g. ICLR 2026's rating
    field uses a discrete {0,2,4,6,8,10} scale), so Phase 4b normalizes
    using year + raw value, never a value assumed comparable as-is.

    A field is only recorded if its key is genuinely present in this
    review's content (checked via `in`, not `.get(field, {})` - the two
    are NOT equivalent: a present-but-empty value and an absent key both
    used to collapse to the same {"raw": {}} output, which made it
    impossible to tell "reviewer left it blank" apart from "wrong key
    name" apart from "field didn't exist yet that year". This version
    keeps them distinct."""
    scores: Dict[str, Any] = {}
    for canonical, aliases in SCORE_FIELD_ALIASES.items():
        found_key = None
        raw_container = None
        for alias in aliases:
            if alias in review_content:
                found_key = alias
                raw_container = review_content[alias]
                break
        if found_key is None:
            continue  # key genuinely absent under any known alias - skip, don't fabricate a placeholder

        val = raw_container.get("value", raw_container) if isinstance(raw_container, dict) else raw_container
        if val in (None, "", {}):
            continue  # key present but genuinely empty (e.g. reviewer skipped it) - also skip

        entry: Dict[str, Any] = {"raw": val, "source_key": found_key}
        numeric = val
        if isinstance(numeric, str) and ":" in numeric:
            numeric = numeric.split(":")[0].strip()
        try:
            entry["numeric"] = float(numeric)
        except (ValueError, TypeError):
            pass
        scores[canonical] = entry
    return scores


def _audit_content_keys(review_content: Dict[str, Any]) -> List[str]:
    """Return the literal top-level keys present in a review's content
    dict, so per-venue-year field naming can be verified from real data
    instead of assumed. Cheap to store per-review; invaluable for
    catching a silently-wrong field-name assumption like the one that
    surfaced on ICLR 2022."""
    return sorted(review_content.keys())


def _extract_review_text(review_content: Dict[str, Any]) -> str:
    parts = []
    for field in ("summary", "strengths", "weaknesses", "questions", "main_review"):
        val = _extract_text(review_content.get(field, {}))
        if val:
            parts.append(f"[{field.replace('_', ' ').title()}]\n{val}")
    return "\n\n".join(parts)


def _extract_decision(decision_content: Dict[str, Any]) -> str:
    for field in ("decision", "recommendation", "final_recommendation"):
        val = _extract_text(decision_content.get(field, {}))
        if val:
            return val
    return ""


# ---------------------------------------------------------------------------
# Per-paper processing (runs inside worker threads)
# ---------------------------------------------------------------------------

def _process_submission(sub: Dict[str, Any], venue: Dict[str, Any], out_dir: Path, force: bool) -> Dict[str, Any]:
    paper_id = sub.get("id", "unknown")
    paper_path = out_dir / f"{paper_id}.json"

    if paper_path.exists() and not force:
        return {"paper_id": paper_id, "status": "skipped_existing"}

    api_version = venue["api_version"]
    forum_id = sub.get("forum", paper_id)
    sub_content = sub.get("content", {})

    paper_data: Dict[str, Any] = {
        "paper_id": paper_id,
        "forum_id": forum_id,
        "number": sub.get("number"),
        "venue_id": venue["venue_id"],
        "year": venue["year"],
        "api_version": api_version,
        "title": _extract_text(sub_content.get("title", {})),
        "authors": _extract_list(sub_content.get("authors", {})),
        "abstract": _extract_text(sub_content.get("abstract", {})),
        "keywords": _extract_list(sub_content.get("keywords", {})),
        "venueid": _extract_text(sub_content.get("venueid", {})),
        "reviews": [],
        "meta_reviews": [],
        "decision": "",
        "decision_note_id": None,
        "replies_fetch_status": "ok",
    }

    try:
        replies = _get_note_replies(api_version, forum_id)
    except Exception as e:  # noqa: BLE001 - a bad paper must not kill the run
        paper_data["replies_fetch_status"] = f"error: {e}"
        replies = None

    if replies is None:
        paper_data["replies_fetch_status"] = "challenge_or_permission_denied"
        replies = []

    for reply in replies:
        if reply.get("id") == paper_id:
            continue
        inv = " ".join(reply.get("invitations", [])).lower()
        content = reply.get("content", {})

        if "official_review" in inv or inv.endswith("review"):
            paper_data["reviews"].append({
                "review_id": reply.get("id", ""),
                "signatures": reply.get("signatures", []),
                "scores": _extract_scores(content),
                "review_text": _extract_review_text(content),
                "content_keys": _audit_content_keys(content),
                "timestamp": reply.get("tcdate", 0),
            })
        elif "meta_review" in inv or "metareview" in inv:
            meta_text = _extract_text(content.get("metareview", {})) or _extract_text(content.get("review", {}))
            paper_data["meta_reviews"].append({
                "meta_review_id": reply.get("id", ""),
                "recommendation": _extract_decision(content),
                "text": meta_text,
                "timestamp": reply.get("tcdate", 0),
            })
        elif "decision" in inv:
            decision = _extract_decision(content)
            if decision:
                paper_data["decision"] = decision
                paper_data["decision_note_id"] = reply.get("id", "")

    paper_path.write_text(json.dumps(paper_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "paper_id": paper_id,
        "status": "processed",
        "has_scores": bool(paper_data["reviews"]),
        "has_decision": bool(paper_data["decision"]),
        "has_metareview": bool(paper_data["meta_reviews"]),
    }


# ---------------------------------------------------------------------------
# Venue-level orchestration
# ---------------------------------------------------------------------------

def _submissions_cache_path(out_dir: Path) -> Path:
    return out_dir / "_submissions_cache.json"


def _write_field_audit(label: str, out_dir: Path) -> None:
    """Scan every paper file just written for this venue and report which
    literal content keys actually showed up on reviews, and how often each
    canonical score field (rating/confidence/etc.) was found vs. missing.
    This turns 'rating looks empty for 2022' from a one-off manual
    observation into something checkable for every venue automatically."""
    key_counts: Dict[str, int] = {}
    canonical_found: Dict[str, int] = {c: 0 for c in SCORE_FIELD_ALIASES}
    canonical_missing: Dict[str, int] = {c: 0 for c in SCORE_FIELD_ALIASES}
    reviews_seen = 0

    for f in out_dir.glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        for review in data.get("reviews", []):
            reviews_seen += 1
            for key in review.get("content_keys", []):
                key_counts[key] = key_counts.get(key, 0) + 1
            scores = review.get("scores", {})
            for canonical in SCORE_FIELD_ALIASES:
                if canonical in scores:
                    canonical_found[canonical] += 1
                else:
                    canonical_missing[canonical] += 1

    audit = {
        "venue": label,
        "reviews_scanned": reviews_seen,
        "content_key_frequency": dict(sorted(key_counts.items(), key=lambda kv: -kv[1])),
        "canonical_field_coverage": {
            c: {"found": canonical_found[c], "missing": canonical_missing[c]}
            for c in SCORE_FIELD_ALIASES
        },
    }
    (out_dir / "_field_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if reviews_seen and canonical_missing.get("rating", 0) == reviews_seen:
        log.warning("  FIELD AUDIT: 'rating' was not found under ANY known alias "
                     "in %d/%d reviews for %s. Check %s/_field_audit.json's "
                     "content_key_frequency to find the actual key name this "
                     "venue-year uses, then add it to SCORE_FIELD_ALIASES['rating'] "
                     "and re-run with --force to backfill.",
                     reviews_seen, reviews_seen, label, out_dir)


def _load_or_fetch_submissions(venue: Dict[str, Any], out_dir: Path, max_results: int, refresh: bool) -> List[Dict[str, Any]]:
    cache_path = _submissions_cache_path(out_dir)
    if cache_path.exists() and not refresh:
        log.info("  Using cached submission list (%s). Pass --refresh-submissions to re-list.",
                   cache_path.name)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if venue["api_version"] == "v2":
        submissions = _search_notes_v2(venue["venue_id"], max_results)
        real_submissions = []
        for sub in submissions:
            vc = sub.get("content", {})
            venueid = _extract_text(vc.get("venueid", {}))
            invitation = " ".join(sub.get("invitations", []))
            if venue["venue_id"] in invitation or venue["venue_id"] in venueid or "Submission" in invitation:
                real_submissions.append(sub)
    else:
        real_submissions = _search_notes_v1(
            venue["venue_id"], venue["submission_invitations"], max_results,
        )

    cache_path.write_text(json.dumps(real_submissions, ensure_ascii=False), encoding="utf-8")
    return real_submissions


def ingest_venue(venue: Dict[str, Any], dry_run: bool, max_papers: int, force: bool,
                  refresh_submissions: bool, workers: int) -> Dict[str, Any]:
    label = venue["label"]
    out_dir = OUTPUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== %s (%s, API %s) ===", label, venue["venue_id"], venue["api_version"])

    real_submissions = _load_or_fetch_submissions(
        venue, out_dir, max_results=SEARCH_MAX_RESULTS, refresh=refresh_submissions,
    )
    log.info("  Submissions found: %d", len(real_submissions))

    if not real_submissions:
        log.warning("  Zero submissions for %s - if this is a v1 venue, check the "
                     "invitation suffix list; if v2, check the venue_id is correct.", label)

    if dry_run:
        already = sum(1 for s in real_submissions if (out_dir / f"{s.get('id')}.json").exists())
        log.info("  [DRY RUN] Would process %d papers (%d already on disk).",
                   len(real_submissions), already)
        return {"venue": label, "venue_id": venue["venue_id"], "year": venue["year"],
                "submissions": len(real_submissions), "dry_run": True}

    if max_papers > 0:
        real_submissions = real_submissions[:max_papers]

    processed = skipped = errors = with_scores = with_decision = with_metareview = 0

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{label}-w") as pool:
        futures = {pool.submit(_process_submission, sub, venue, out_dir, force): sub
                   for sub in real_submissions}
        done_count = 0
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                log.error("  Worker error: %s", e)
                continue

            done_count += 1
            if result["status"] == "skipped_existing":
                skipped += 1
            elif result["status"] == "processed":
                processed += 1
                with_scores += int(result.get("has_scores", False))
                with_decision += int(result.get("has_decision", False))
                with_metareview += int(result.get("has_metareview", False))

            if done_count % 100 == 0 or done_count == len(real_submissions):
                log.info("  Progress: %d/%d (processed=%d skipped=%d errors=%d)",
                           done_count, len(real_submissions), processed, skipped, errors)

            if _shutdown_requested.is_set():
                break

    log.info("=== %s complete: processed=%d skipped=%d errors=%d "
              "with_scores=%d with_decision=%d with_metareview=%d ===",
              label, processed, skipped, errors, with_scores, with_decision, with_metareview)

    _write_field_audit(label, out_dir)

    return {
        "venue": label, "venue_id": venue["venue_id"], "year": venue["year"],
        "submissions": len(real_submissions),
        "papers_processed_this_run": processed,
        "papers_skipped_existing": skipped,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Manifest (always recomputed from what's actually on disk)
# ---------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(run_results: List[Dict[str, Any]]) -> None:
    manifest: Dict[str, Any] = {
        "description": "ICLR 2021-2026 review calibration dataset for "
                        "Supervisor/Editor scoring calibration (Phase 4a/4b).",
        "generated_by": "ingest_openreview.py",
        "api_bases": {"v1": API_BASE_V1, "v2": API_BASE_V2},
        "note": "Score fields carry both 'raw' and (when parseable) "
                "'numeric' values. Rating/confidence scales differ across "
                "ICLR years (e.g. 2026 uses a discrete 0/2/4/6/8/10 rating "
                "scale) - normalize per-year in Phase 4b.",
        "venues": [],
    }

    totals = {"papers": 0, "with_scores": 0, "with_decision": 0, "with_metareview": 0,
              "fetch_blocked": 0, "fetch_error": 0}

    for venue in VENUES:
        label = venue["label"]
        out_dir = OUTPUT_ROOT / label
        if not out_dir.exists():
            continue
        # Exclude our own bookkeeping files (_submissions_cache.json etc.) -
        # only files that are per-paper records (a JSON object) belong here.
        files = sorted(p for p in out_dir.glob("*.json") if not p.name.startswith("_"))
        if not files:
            continue

        file_hashes, v_scores, v_decision, v_meta, v_blocked, v_error = {}, 0, 0, 0, 0, 0
        for f in files:
            file_hashes[f.name] = _file_sha256(f)
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, dict):
                # Defensive: skip anything that isn't a per-paper record,
                # in case a future bookkeeping file also lacks a leading "_".
                continue
            if data.get("reviews"):
                v_scores += 1
            if data.get("decision"):
                v_decision += 1
            if data.get("meta_reviews"):
                v_meta += 1
            status = data.get("replies_fetch_status", "ok")
            if status == "challenge_or_permission_denied":
                v_blocked += 1
            elif status not in ("ok",):
                v_error += 1

        manifest["venues"].append({
            "label": label, "venue_id": venue["venue_id"], "year": venue["year"],
            "api_version": venue["api_version"],
            "paper_count": len(files),
            "papers_with_scores": v_scores,
            "papers_with_decision": v_decision,
            "papers_with_metareview": v_meta,
            "papers_fetch_blocked": v_blocked,
            "papers_fetch_error": v_error,
            "file_hashes": file_hashes,
            "directory": str(out_dir.relative_to(OUTPUT_ROOT.parent)),
        })

        totals["papers"] += len(files)
        totals["with_scores"] += v_scores
        totals["with_decision"] += v_decision
        totals["with_metareview"] += v_meta
        totals["fetch_blocked"] += v_blocked
        totals["fetch_error"] += v_error

    manifest["total_papers"] = totals["papers"]
    manifest["total_with_scores"] = totals["with_scores"]
    manifest["total_with_decision"] = totals["with_decision"]
    manifest["total_with_metareview"] = totals["with_metareview"]
    manifest["total_fetch_blocked"] = totals["fetch_blocked"]
    manifest["total_fetch_error"] = totals["fetch_error"]
    manifest["run_results"] = run_results

    manifest_path = OUTPUT_ROOT / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Manifest written to %s", manifest_path)

    log.info("%s", "=" * 70)
    log.info("PER-YEAR BREAKDOWN (check for drift/gaps before Phase 4b)")
    log.info("%s", "=" * 70)
    for v in manifest["venues"]:
        log.info("  %-10s [%s]  papers=%-7d scores=%-7d decisions=%-7d metareviews=%-7d",
                   v["label"], v["api_version"], v["paper_count"], v["papers_with_scores"],
                   v["papers_with_decision"], v["papers_with_metareview"])
    log.info("  %-15s papers=%-7d scores=%-7d decisions=%-7d metareviews=%-7d",
               "TOTAL", totals["papers"], totals["with_scores"],
               totals["with_decision"], totals["with_metareview"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ICLR 2021-2026 review data from OpenReview")
    parser.add_argument("--venue", nargs="*", choices=VENUE_LABELS,
                         help="Specific venue(s) to ingest (default: all six years)")
    parser.add_argument("--dry-run", action="store_true",
                         help="List submissions only, download nothing")
    parser.add_argument("--max-papers", type=int, default=0,
                         help="Cap papers processed per venue (0 = no cap). Useful for smoke tests.")
    parser.add_argument("--force", action="store_true",
                         help="Re-fetch papers even if a local JSON file already exists")
    parser.add_argument("--refresh-submissions", action="store_true",
                         help="Bypass the cached submission list and re-query the search endpoint")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help=f"Concurrent reply-fetch workers (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    venues_to_pull = [v for v in VENUES if v["label"] in args.venue] if args.venue else VENUES
    if not venues_to_pull:
        log.error("No venues to pull.")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    needed_versions = {v["api_version"] for v in venues_to_pull}
    if "v1" in needed_versions:
        _auth_tokens["v1"] = authenticate(API_BASE_V1)
    if "v2" in needed_versions:
        _auth_tokens["v2"] = authenticate(API_BASE_V2)

    for ver in needed_versions:
        if not _auth_tokens[ver]:
            host = API_BASE_V1 if ver == "v1" else API_BASE_V2
            log.warning("No working credentials for API %s (%s) - reviews/decisions/"
                        "meta-reviews will be empty for venues on this host. Set "
                        "OPENREVIEW_USERNAME/PASSWORD in .env; the same account works "
                        "on both hosts, so if one version authenticated and the other "
                        "didn't, that's worth a second look, not just a retry.",
                        ver, host)

    log.info("Pulling %d venue(s): %s", len(venues_to_pull), [v["label"] for v in venues_to_pull])
    log.info("Workers: %d | Output: %s", args.workers, OUTPUT_ROOT)
    if args.dry_run:
        log.info("DRY RUN - listing only")

    run_results: List[Dict[str, Any]] = []
    for venue in venues_to_pull:
        if _shutdown_requested.is_set():
            log.warning("Shutdown requested - stopping before starting %s.", venue["label"])
            break
        try:
            result = ingest_venue(
                venue, dry_run=args.dry_run, max_papers=args.max_papers,
                force=args.force, refresh_submissions=args.refresh_submissions,
                workers=args.workers,
            )
            run_results.append(result)
        except Exception as e:  # noqa: BLE001 - one bad venue must not kill the rest
            log.error("Failed to ingest %s: %s", venue["label"], e)
            run_results.append({"venue": venue["label"], "error": str(e)})

    if not args.dry_run:
        write_manifest(run_results)

    log.info("%s", "=" * 70)
    log.info("RUN SUMMARY")
    log.info("%s", "=" * 70)
    for r in run_results:
        if "error" in r:
            log.info("  %s: FAILED - %s", r["venue"], r["error"])
        elif r.get("dry_run"):
            log.info("  %s: dry run, %d submissions listed", r["venue"], r["submissions"])
        else:
            log.info("  %s: +%d new, %d already had files, %d errors this run",
                       r["venue"], r.get("papers_processed_this_run", 0),
                       r.get("papers_skipped_existing", 0), r.get("errors", 0))

    if _shutdown_requested.is_set():
        log.warning("Exited early due to interrupt. Re-run the same command to resume.")


if __name__ == "__main__":
    main()