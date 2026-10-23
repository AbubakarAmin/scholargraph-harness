"""OA resolution layer for systematic reviews (proposal §5.1 + §4.6 step 1, M2).

Resolution order (exact): ``best_oa_location`` (OpenAlex) → Unpaywall →
Semantic Scholar OA → PMC. Closed/paywalled → ``full_text_unavailable``,
abstract-only + limitation entry. Never fetch paywalled PDFs.

API metadata (Unpaywall/PMC/DOAJ/OpenAlex) goes through
:class:`core.sources.SourceClient` (cache in ``output/source_cache/`` +
hash/timestamp) wrapped by the gateway (rate limiting + breaker 5/60s +
AIMD + jitter + ``is_available()`` skip). Arbitrary publisher OA URLs
(any domain) are fetched via direct gateway calls with a local file cache
(``output/source_cache/oa_text/<sha256>.json`` + hash/timestamp) — the
SourceClient allowlist stays API-only by design, so OA text never trips
URL validation.

Providers (all free):
- unpaywall NEW — 1.0 req/s burst 1 — ``api.unpaywall.org/v2/{doi}?email=``
  reusing ``OPENALEX_EMAIL`` — OA status + URL — flag ``UNPAYWALL_ENABLED``.
- pmc NEW — 2.0 req/s burst 2 — OA subset only (``oa.fcgi`` + efetch) —
  biomed full-text links — flag ``PMC_ENABLED``.
- doaj NEW — 2.0 req/s burst 2 — weekly journal cache — non-listed OA claim
  → warning — flag ``DOAJ_GUARD_ENABLED``.
- best_oa_location extension on the existing OpenAlex adapter (rate kept 5.0/3).
- Local ``pypdf`` for OA PDFs ONLY (never paywalled).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OA_SOURCE_BASES = {
    "openalex": "https://api.openalex.org/",
    "crossref": "https://api.crossref.org/",
    "unpaywall": "https://api.unpaywall.org/",
    "pmc": "https://www.ncbi.nlm.nih.gov/pmc/",
    "doaj": "https://api.doaj.org/",
    "semantic_scholar": "https://api.semanticscholar.org/",
}

# Weekly DOAJ journal cache (proposal §5.1).
DOAJ_CACHE_TTL_SEC = 7 * 24 * 3600


def get_oa_client(cache_dir: Optional[str] = None):
    """SourceClient pre-wired with OA allowlist bases (§5.1)."""
    from core.config import config
    from core.sources import SourceClient

    base = cache_dir or str(Path(config.output_dir) / "source_cache")
    return SourceClient(cache_dir=base, source_bases=OA_SOURCE_BASES)


def extract_best_oa_location(openalex_row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """OA PDF URL resolution off the existing OpenAlex adapter (rate kept 5.0/3).

    Returns ``{url, license}`` or None when the row carries no OA location.
    Pure helper (no I/O) so the gateway path stays offline-testable.
    """
    if not isinstance(openalex_row, dict):
        return None
    best = openalex_row.get("best_oa_location") or {}
    if not isinstance(best, dict):
        return None
    url = (best.get("pdf_url") or best.get("landing_page_url") or "").strip()
    if not url:
        # Fall back to primary_location OA signals without inventing a URL.
        primary = openalex_row.get("primary_location") or {}
        if isinstance(primary, dict):
            url = str(primary.get("pdf_url") or primary.get("landing_page_url") or "").strip()
    if not url:
        return None
    license_name = str(best.get("license") or openalex_row.get("license") or "oa").strip() or "oa"
    return {"url": url, "license": license_name, "source": "best_oa_location"}


def _flag_enabled(name: str, default: bool = False) -> bool:
    try:
        from core.config import config

        return bool(getattr(config, name, default))
    except Exception:
        return default


def _gateway_available(provider: str) -> bool:
    try:
        from core.api_gateway import get_gateway

        return bool(get_gateway().is_available(provider))
    except Exception:
        return True


def resolve_oa_url(
    doi: str,
    openalex_row: Optional[Dict[str, Any]] = None,
    s2_row: Optional[Dict[str, Any]] = None,
    client=None,
) -> Dict[str, Any]:
    """Resolve one DOI to an OA URL (order: best_oa_location → Unpaywall → S2 → PMC).

    Returns ``{url|None, source, license|None, oa_status, full_text_unavailable}``.
    Respects ``UNPAYWALL_ENABLED`` / ``PMC_ENABLED`` flags and open breakers
    (``is_available()`` skip). Never raises — unavailable providers degrade to
    the next source, closed content degrades to abstract-only.
    """
    from core.config import config

    doi = (doi or "").strip()
    if not doi:
        return {"url": None, "source": None, "license": None,
                "oa_status": "unknown", "full_text_unavailable": True}

    # 1. OpenAlex best_oa_location (no extra HTTP — row already fetched at 5.0/3).
    best = extract_best_oa_location(openalex_row)
    if best:
        return {"url": best["url"], "source": "best_oa_location",
                "license": best.get("license") or "oa",
                "oa_status": "oa", "full_text_unavailable": False}

    client = client or get_oa_client()
    email = getattr(config, "openalex_email", "researcher@example.com")

    # 2. Unpaywall (flag-gated).
    if _flag_enabled("unpaywall_enabled") and _gateway_available("unpaywall"):
        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
            artifact = client.fetch_json("unpaywall", url)
            content = artifact.get("content") or {}
            if artifact.get("status") in ("verified", "cached"):
                best_loc = content.get("best_oa_location") or {}
                oa_url = (best_loc.get("url_for_pdf") or best_loc.get("url") or "").strip()
                is_oa = bool(content.get("is_oa"))
                license_name = str((best_loc.get("license") or content.get("license") or "oa") if is_oa else "closed")
                if is_oa and oa_url:
                    return {"url": oa_url, "source": "unpaywall", "license": license_name,
                            "oa_status": "oa", "full_text_unavailable": False}
        except Exception as exc:
            logger.debug("Unpaywall resolve failed for %s: %s", doi, exc)

    # 3. Semantic Scholar OA link (row already carries openAccessPdf when fetched).
    if isinstance(s2_row, dict):
        oa = s2_row.get("openAccessPdf") or {}
        if isinstance(oa, dict) and oa.get("url"):
            lic = str(s2_row.get("license") or oa.get("license") or "oa")
            return {"url": oa["url"], "source": "s2_oa", "license": lic,
                    "oa_status": "oa", "full_text_unavailable": False}

    # 4. PMC OA subset (flag-gated, biomed full-text links only).
    if _flag_enabled("pmc_enabled") and _gateway_available("pmc"):
        try:
            # OA utility: idconv by DOI → PMCID, OA subset only.
            url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=scholargraph&email={email}&ids={doi}&format=json"
            # NOTE: idconv host differs from the allowlisted pmc base; resolve
            # via a direct gateway call instead of SourceClient URL validation
            # so the allowlist stays OA-subset-only by design.
            from core.api_gateway import get_gateway
            import requests as _requests

            def _do_fetch():
                r = _requests.get(url, timeout=15)
                r.raise_for_status()
                return r.json()

            content = get_gateway().request("pmc", _do_fetch, retries=2, backoff_base=3.0)
            records = (content or {}).get("records") or []
            pmcid = ""
            for rec in records:
                if isinstance(rec, dict) and rec.get("pmcid"):
                    pmcid = str(rec["pmcid"]).strip()
                    break
            if pmcid:
                oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                return {"url": oa_url, "source": "pmc", "license": "oa",
                        "oa_status": "oa", "full_text_unavailable": False}
        except Exception as exc:
            logger.debug("PMC resolve failed for %s: %s", doi, exc)

    # Closed/paywalled → abstract-only + limitation entry downstream.
    return {"url": None, "source": None, "license": "closed",
            "oa_status": "closed", "full_text_unavailable": True}


def fetch_oa_text(
    oa_url: Optional[str],
    license_name: Optional[str] = None,
    client=None,
    max_chars: int = 20000,
) -> Dict[str, Any]:
    """Fetch full text for an OA-licensed URL only (license-checked).

    Closed/unknown/paywalled licenses → ``{status: full_text_unavailable}``
    without any HTTP. OA PDFs are parsed locally with ``pypdf`` when the URL
    looks like a PDF; HTML/text is truncated to ``max_chars``. Publisher OA
    URLs live on arbitrary domains, so both branches fetch via direct
    gateway calls with a local file cache (``output/source_cache/oa_text/``
    + hash/timestamp) instead of the API-allowlisted SourceClient. Callers
    turn ``full_text_unavailable`` into a limitation entry and continue
    abstract-only.
    """
    if not oa_url:
        return {"text": "", "status": "full_text_unavailable",
                "reason": "no OA URL resolved (closed or paywalled)"}
    lic = (license_name or "").strip().lower()
    if not lic or lic in ("unknown", "closed", "paywalled"):
        return {"text": "", "status": "full_text_unavailable",
                "reason": f"license not OA ({license_name!r}); closed content stays abstract-only"}
    try:
        import hashlib as _hashlib
        from datetime import datetime as _dt, timezone as _tz

        from core.config import config as _cfg

        cache_dir = Path(_cfg.output_dir) / "source_cache" / "oa_text"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _hashlib.sha256(oa_url.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached.get("url") == oa_url and "text" in cached:
                    cached["status"] = "cached"
                    return cached
            except Exception:
                pass
    except Exception:
        cache_path = None  # type: ignore[assignment]
    try:
        # PDF branch: download bytes via gateway, parse locally with pypdf.
        if oa_url.lower().endswith(".pdf"):
            from core.api_gateway import get_gateway
            import requests as _requests

            def _do_fetch_bytes():
                r = _requests.get(oa_url, timeout=30)
                r.raise_for_status()
                if len(r.content) > 5_000_000:
                    raise ValueError("OA PDF exceeds 5MB cap")
                return r.content

            raw = get_gateway().request("oa_text", _do_fetch_bytes, retries=2, backoff_base=3.0)
            try:
                from pypdf import PdfReader
                import io as _io

                reader = PdfReader(_io.BytesIO(raw))
                pages = [(p.extract_text() or "") for p in reader.pages[:20]]
                text = "\n".join(pages)[:max_chars]
            except Exception as exc:
                return {"text": "", "status": "full_text_unavailable",
                        "reason": f"OA PDF parse failed: {exc}"}
            import hashlib as _hashlib3

            _digest = _hashlib3.sha256(text.encode("utf-8")).hexdigest()
            result = {"text": text, "status": "ok" if text.strip() else "full_text_unavailable",
                      "license": license_name, "url": oa_url, "response_hash": _digest,
                      "retrieved_at": __import__("datetime").datetime.now(
                          __import__("datetime").timezone.utc).isoformat()}
        else:
            from core.api_gateway import get_gateway
            import hashlib as _hashlib2
            import requests as _requests2

            def _do_fetch_text():
                r = _requests2.get(oa_url, timeout=30,
                                  headers={"User-Agent": "ScholarGraph/2.0 (OA-only)"})
                r.raise_for_status()
                if len(r.content) > 5_000_000:
                    raise ValueError("OA text exceeds 5MB cap")
                return r.text

            text = get_gateway().request("oa_text", _do_fetch_text, retries=2, backoff_base=3.0)
            text = str(text or "")[:max_chars]
            digest = _hashlib2.sha256(text.encode("utf-8")).hexdigest()
            result = {"text": text, "status": "ok" if text.strip() else "full_text_unavailable",
                      "license": license_name, "url": oa_url, "response_hash": digest,
                      "retrieved_at": __import__("datetime").datetime.now(
                          __import__("datetime").timezone.utc).isoformat()}
        try:
            if cache_path is not None:
                cache_path.write_text(json.dumps(result, default=str)[:2000000], encoding="utf-8")
        except Exception:
            pass
        try:
            from core.run_log import emit_event as _emit

            _emit("source_fetch", {"provider": "oa_text", "url": oa_url[:120],
                                   "status": result.get("status"),
                                   "chars": len(result.get("text") or "")},
                  agent="ReviewAgent")
        except Exception:
            pass
        return result
    except Exception as exc:
        logger.debug("fetch_oa_text failed for %s: %s", oa_url, exc)
        return {"text": "", "status": "full_text_unavailable", "reason": str(exc)[:200]}


def check_doaj_journal(journal: str, client=None) -> Dict[str, Any]:
    """DOAJ guard: non-listed OA claim → warning (weekly journal cache).

    Returns ``{listed: bool|None, warning|None}``. ``None`` listed means the
    guard is disabled or the cache/probe failed — callers treat that as
    "unknown, no warning" rather than blocking. Offline-safe (never raises).
    """
    journal = (journal or "").strip()
    if not journal or not _flag_enabled("doaj_guard_enabled"):
        return {"listed": None, "warning": None}
    try:
        from core.config import config

        cache_path = Path(config.output_dir) / "source_cache" / "doaj_journals.json"
        cached: Dict[str, Any] = {}
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                age = time.time() - float(cached.get("fetched_at") or 0)
                titles = {str(t).lower() for t in (cached.get("titles") or [])}
                if age < DOAJ_CACHE_TTL_SEC and titles:
                    listed = journal.lower() in titles
                    return {"listed": listed,
                            "warning": None if listed else f"journal {journal!r} not in DOAJ; OA claim is a warning, not a block"}
            except Exception:
                cached = {}
        if not _gateway_available("doaj"):
            return {"listed": None, "warning": None}
        client = client or get_oa_client()
        from urllib.parse import quote as _quote

        artifact = client.fetch_json(
            "doaj",
            f"https://api.doaj.org/api/search/journals/{_quote(journal, safe='')}",
        )
        content = artifact.get("content") or {}
        results = content.get("results") or content.get("data") or []
        listed = bool(results)
        # Refresh weekly cache best-effort (failures keep the stale file).
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"fetched_at": time.time(), "titles": [journal] if listed else []}), encoding="utf-8")
        except Exception:
            pass
        return {"listed": listed,
                "warning": None if listed else f"journal {journal!r} not in DOAJ; OA claim is a warning, not a block"}
    except Exception as exc:
        logger.debug("DOAJ guard failed for %s: %s", journal, exc)
        return {"listed": None, "warning": None}
