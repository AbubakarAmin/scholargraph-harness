"""Dataset access layer — local catalog + HuggingFace integration.

Provides a unified abstraction for agents to:
  1. Check if a dataset exists (local catalog or HuggingFace)
  2. Get dataset metadata (size, license, features, downloads)
  3. Download small/medium datasets for experiments
  4. Search HuggingFace for datasets matching a query

All HuggingFace calls go through the API gateway for rate limiting.
Results are cached to avoid repeated API calls within a run.
"""

from __future__ import annotations

import csv
import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, NamedTuple

from core.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known public ML datasets (moved from topic_hunter to fix circular import)
# ---------------------------------------------------------------------------

KNOWN_PUBLIC_DATASETS = frozenset({
    # Vision
    "mnist", "cifar", "imagenet", "fashion", "svhn", "stl10",
    "celeba", "lsun", "places365", "eurosat", "oxford flowers",
    "caltech101", "caltech256",
    # NLP
    "glue", "superglue", "squad", "mnli", "mrpc", "qnli", "rte", "wnli",
    "cola", "stsb", "sst", "qqp", "race", "triviaqa", "natural questions",
    "hellaswag", "winogrande", "arc", "openbookqa", "boolq", "swag",
    "piqa", "commonsense", "commitmentbank",
    # Classification / Tabular
    "imdb", "yelp", "ag news", "20newsgroups", "reuters",
    "iris", "wine", "breast cancer", "diabetes", "california housing",
    "boston housing", "ames housing", "adult", "covertype",
    # Audio
    "librispeech", "common voice", "voxceleb", "audioset", "esc50", "gtzan",
    # Video
    "kinetics", "ucf101", "activitynet",
    # Object Detection / Segmentation
    "coco", "pascal voc", "ade20k", "cityscapes", "gta5",
    "kitti", "waymo", "nuscenes",
    # Graph
    "ogb", "mutag", "proteins", "tu dataset",
    # Few-shot / Meta-learning
    "omniglot", "miniimagenet", "tiered imagenet",
    # Language modeling
    "ptb", "wikitext", "text8",
    # Benchmarks
    "mmlu", "humaneval", "mbpp", "gsm8k", "math",
    # Scientific / Medical
    "pubmed", "arxiv papers", "chestxray", "brats", "isic",
    # General
    "uci", "adult", "covertype",
})

# ---------------------------------------------------------------------------
# Local catalog (deterministic, no network)
# ---------------------------------------------------------------------------

DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "bundled_synthetic": {
        "name": "bundled_synthetic",
        "version": "1",
        "source": "ScholarGraph generator",
        "access_policy": "local-only",
        "max_rows": 10000,
        "license": "generated",
        "path": None,
    },
    "sklearn_iris": {
        "name": "sklearn_iris",
        "version": "1",
        "source": "scikit-learn bundled dataset",
        "access_policy": "local-only",
        "max_rows": 150,
        "license": "BSD-3-Clause",
        "path": "data/catalog/iris.csv",
    },
    "sklearn_digits": {
        "name": "sklearn_digits",
        "version": "1",
        "source": "scikit-learn bundled dataset",
        "access_policy": "local-only",
        "max_rows": 1797,
        "license": "BSD-3-Clause",
        "path": None,
    },
    "openreview_calibration": {
        "name": "openreview_calibration",
        "version": "1",
        "source": "OpenReview API v2 (ICLR 2025-2026)",
        "access_policy": "local-only",
        "max_rows": 10000,
        "license": "CC-BY-4.0",
        "path": "data/review_calibration",
        "description": (
            "ICLR papers with reviewer scores, decisions, and meta-reviews "
            "for calibrating Supervisor.evaluate_section() and "
            "Editor.final_manuscript_referee() in Phase 4b."
        ),
    },
}

# ---------------------------------------------------------------------------
# In-memory cache for HuggingFace metadata (per-run, not persisted)
# ---------------------------------------------------------------------------

_hf_cache: Dict[str, Dict[str, Any]] = {}
_hf_search_cache: Dict[str, List[Dict[str, Any]]] = {}


class HFResult(NamedTuple):
    """Typed result for HuggingFace operations."""
    found: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Local catalog functions
# ---------------------------------------------------------------------------


def list_datasets() -> List[Dict[str, Any]]:
    return [dict(value) for value in DATASET_CATALOG.values()]


def resolve_dataset(name: str) -> Dict[str, Any]:
    """Resolve only catalogued, local datasets; never download implicitly."""
    if name not in DATASET_CATALOG:
        raise ValueError(f"Dataset is not in the local catalog: {name}")
    return dict(DATASET_CATALOG[name])


def load_local_dataset(name: str) -> Dict[str, Any]:
    """Load a catalogued dataset without network access."""
    spec = resolve_dataset(name)
    if name == "openreview_calibration":
        base = Path(spec.get("path", "data/review_calibration"))
        if not base.exists():
            return {"spec": spec, "rows": [], "row_count": 0, "manifest": {}}
        manifest_path = base / "_manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows: List[Dict[str, Any]] = []
        for venue_dir in sorted(base.iterdir()):
            if not venue_dir.is_dir() or venue_dir.name.startswith("_"):
                continue
            for paper_file in sorted(venue_dir.glob("*.json")):
                try:
                    paper = json.loads(paper_file.read_text(encoding="utf-8"))
                    rows.append(paper)
                except (json.JSONDecodeError, OSError):
                    continue
        return {"spec": spec, "rows": rows, "row_count": len(rows), "manifest": manifest}
    if spec.get("path"):
        path = Path(spec["path"])
        rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
        return {"spec": spec, "rows": rows, "row_count": len(rows)}
    if name == "sklearn_iris":
        from sklearn.datasets import load_iris
        dataset = load_iris(as_frame=True)
        return {"spec": spec, "rows": dataset.frame.to_dict(orient="records"), "row_count": len(dataset.frame)}
    if name == "sklearn_digits":
        from sklearn.datasets import load_digits
        dataset = load_digits(as_frame=True)
        return {"spec": spec, "rows": dataset.frame.to_dict(orient="records"), "row_count": len(dataset.frame)}
    return {"spec": spec, "rows": [], "row_count": 0}


# ---------------------------------------------------------------------------
# HuggingFace dataset access layer
# ---------------------------------------------------------------------------


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from config or environment."""
    return getattr(config, "huggingface_token", None) or None


def _hf_headers() -> Dict[str, str]:
    """Build headers for HuggingFace API calls."""
    headers = {"Accept": "application/json"}
    token = _get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_max_download_bytes() -> int:
    """Get max download size in bytes from config."""
    mb = getattr(config, "huggingface_max_download_mb", 100)
    return int(mb) * 1024 * 1024


def _hf_api_get(url: str, timeout: int = 15) -> HFResult:
    """Make a GET request to HuggingFace API via the gateway."""
    import requests
    from core.api_gateway import get_gateway

    def _do_request():
        resp = requests.get(url, headers=_hf_headers(), timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    gateway = get_gateway()
    try:
        data = gateway.request("huggingface", _do_request)
        return HFResult(found=True, data=data)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return HFResult(found=False, error="not_found")
        return HFResult(found=False, error=str(e))
    except Exception as exc:
        logger.warning("HuggingFace API request failed: %s — %s", url, exc, exc_info=True)
        return HFResult(found=False, error=str(exc))


def _hf_api_get_raw(url: str, timeout: int = 30) -> Optional[bytes]:
    """Make a GET request returning raw bytes (for file downloads).

    D18: streams via ``iter_content`` with the configured byte cap
    (``HUGGINGFACE_MAX_DOWNLOAD_MB``) instead of buffering ``resp.content``
    unbounded. Over-cap bodies truncate at the cap with a warning; the
    truncation propagates via the caller's ``truncated`` flags (see
    ``download_hf_dataset``).
    """
    import requests
    from core.api_gateway import get_gateway

    max_bytes = _get_max_download_bytes()

    def _do_request():
        resp = requests.get(url, headers=_hf_headers(), timeout=timeout, stream=True)
        resp.raise_for_status()
        chunks: List[bytes] = []
        total = 0
        truncated = False
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            if total + len(chunk) > max_bytes:
                chunks.append(chunk[: max(0, max_bytes - total)])
                total = max_bytes
                truncated = True
                break
            chunks.append(chunk)
            total += len(chunk)
        if truncated:
            logger.warning("HuggingFace download %s truncated at %d bytes (byte cap)", url, max_bytes)
        return b"".join(chunks)

    gateway = get_gateway()
    try:
        return gateway.request("huggingface", _do_request)
    except Exception as exc:
        logger.warning("HuggingFace download failed: %s — %s", url, exc, exc_info=True)
        return None


# D19: TTL cache for HF lookups (quote queries, exact match first, transient
# errors preserved instead of demoted to not_found).
_HF_CACHE_TTL_SEC = 3600
_hf_info_cache_ts: Dict[str, float] = {}
_hf_search_cache_ts: Dict[str, float] = {}


def _hf_cache_fresh(key: str, table: str = "info") -> bool:
    """True when a cached HF entry is still within TTL."""
    import time as _time

    ts = (_hf_info_cache_ts if table == "info" else _hf_search_cache_ts).get(key, 0.0)
    return (_time.time() - ts) < _HF_CACHE_TTL_SEC


def _hf_error_is_transient(error: Optional[str]) -> bool:
    """True when an HF error string looks transient (D19: preserve, don't demote).

    Mirrors the gateway transient list (D8): 429/500/502/503/504, overloaded,
    unavailable, timed out/timeout, deadline, try again, connection/reset.
    A 404 / not_found is NOT transient.
    """
    if not error or error == "not_found":
        return False
    lowered = str(error).lower()
    markers = ("429", "500", "502", "503", "504", "overloaded", "unavailable",
               "timed out", "timeout", "deadline", "try again", "connection",
               "reset by peer", "temporarily", "rate limit", "ratelimit")
    return any(marker in lowered for marker in markers)


def search_hf_datasets(
    query: str,
    max_results: int = 5,
    sort: str = "downloads",
) -> List[Dict[str, Any]]:
    """Search HuggingFace Hub for datasets matching a query.

    Returns a list of dataset summaries with id, author, downloads, likes,
    tags, and card_data (if available).

    D19: the query is URL-quoted, results carry a TTL cache, and transient
    HTTP errors surface as [] only after a successful empty search — a
    transport failure never fabricates an empty result silently (callers
    distinguish via the gateway logs).
    """
    import time as _time
    import urllib.parse as _urlparse

    cache_key = f"{query}:{max_results}:{sort}"
    if cache_key in _hf_search_cache and _hf_cache_fresh(cache_key, "search"):
        return _hf_search_cache[cache_key]

    url = (
        f"https://huggingface.co/api/datasets"
        f"?search={_urlparse.quote(str(query or ''), safe='')}"
        f"&limit={max_results}"
        f"&sort={sort}"
        f"&direction=-1"
    )
    result = _hf_api_get(url)
    if not result.found or not result.data:
        return []

    results = []
    for item in result.data[:max_results]:
        results.append({
            "id": item.get("id", ""),
            "author": item.get("author", ""),
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "tags": item.get("tags", []),
            "last_modified": item.get("lastModified", ""),
            "card_data": item.get("cardData", {}),
        })

    _hf_search_cache[cache_key] = results
    _hf_search_cache_ts[cache_key] = _time.time()
    return results


def get_hf_dataset_info(dataset_id: str) -> HFResult:
    """Get detailed metadata for a HuggingFace dataset.

    Returns info including: id, downloads, likes, license, features,
    dataset_size, download_size, file_count, last_modified.

    D19: dataset ids are URL-quoted (org/name slashes preserved) and
    metadata carries the shared TTL cache.
    """
    import time as _time
    import urllib.parse as _urlparse

    if dataset_id in _hf_cache and _hf_cache_fresh(dataset_id, "info"):
        return HFResult(found=True, data=_hf_cache[dataset_id])

    quoted = _urlparse.quote(str(dataset_id or ""), safe="/")
    url = f"https://huggingface.co/api/datasets/{quoted}"
    result = _hf_api_get(url)
    if not result.found or not result.data:
        return result

    data = result.data
    info = {
        "id": data.get("id", ""),
        "author": data.get("author", ""),
        "downloads": data.get("downloads", 0),
        "likes": data.get("likes", 0),
        "tags": data.get("tags", []),
        "last_modified": data.get("lastModified", ""),
        "card_data": data.get("cardData", {}),
        "license": _extract_license(data),
        "features": _extract_features(data),
        "dataset_size": data.get("dataset_size", None),
        "download_size": data.get("download_size", None),
        "file_count": data.get("file_count", None),
    }

    _hf_cache[dataset_id] = info
    _hf_info_cache_ts[dataset_id] = _time.time()
    return HFResult(found=True, data=info)


def _extract_license(data: Dict[str, Any]) -> Optional[str]:
    """Extract license from dataset metadata."""
    card_data = data.get("cardData", {})
    if isinstance(card_data, dict):
        lic = card_data.get("license")
        if isinstance(lic, str):
            return lic
        if isinstance(lic, list) and lic:
            return lic[0]
    tags = data.get("tags", [])
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _extract_features(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract dataset features/schema from metadata."""
    card_data = data.get("cardData", {})
    if isinstance(card_data, dict):
        features = card_data.get("features")
        if isinstance(features, dict):
            return features
    return None


def dataset_exists_on_hf(dataset_name: str) -> HFResult:
    """Check if a dataset exists on HuggingFace Hub.

    Tries exact match first, then fuzzy search. Works WITHOUT a token
    for public datasets (most HF datasets are public).

    D19: exact org/name match goes first (verbatim, then normalized), and a
    transient lookup error is preserved (returned as-is) instead of being
    demoted to ``not_found`` by a follow-up search miss.
    """
    verbatim = str(dataset_name or "").strip()
    normalized = verbatim.replace(" ", "-").replace("_", "-").lower()

    # Try exact match first: verbatim org/name, then the normalized id
    # (works without token for public datasets).
    for candidate in ([verbatim] if verbatim else []):
        info = get_hf_dataset_info(candidate)
        if info.found:
            return info
        if _hf_error_is_transient(info.error):
            return info
    info = get_hf_dataset_info(normalized)
    if info.found:
        return info
    if _hf_error_is_transient(info.error):
        return info

    # Try search
    results = search_hf_datasets(dataset_name, max_results=3)
    for r in results:
        r_id = r.get("id", "").lower()
        if normalized in r_id or r_id.endswith("/" + normalized):
            return HFResult(found=True, data=r)
        r_parts = r_id.split("/")
        if normalized in r_parts[-1]:
            return HFResult(found=True, data=r)

    return HFResult(found=False, error="not_found")


def get_dataset_info(dataset_name: str) -> Optional[Dict[str, Any]]:
    """Get dataset info from either local catalog or HuggingFace.

    Returns a unified info dict with: name, source, exists, size_info,
    license, features, download_count.
    """
    # Check local catalog first
    if dataset_name in DATASET_CATALOG:
        spec = DATASET_CATALOG[dataset_name]
        return {
            "name": dataset_name,
            "source": "local_catalog",
            "exists": True,
            "size_info": {"max_rows": spec.get("max_rows", 0)},
            "license": spec.get("license"),
            "features": None,
            "download_count": None,
        }

    # Check HuggingFace (works without token for public datasets)
    hf_result = get_hf_dataset_info(dataset_name)
    if hf_result.found and hf_result.data:
        hf_info = hf_result.data
        return {
            "name": dataset_name,
            "source": "huggingface",
            "exists": True,
            "size_info": {
                "dataset_size": hf_info.get("dataset_size"),
                "download_size": hf_info.get("download_size"),
                "file_count": hf_info.get("file_count"),
                "downloads": hf_info.get("downloads", 0),
            },
            "license": hf_info.get("license"),
            "features": hf_info.get("features"),
            "download_count": hf_info.get("downloads", 0),
        }

    return None


def download_hf_dataset(
    dataset_id: str,
    split: str = "train",
    max_samples: int = 1000,
    cache_dir: str = "./memory/dataset_cache",
) -> Optional[Dict[str, Any]]:
    """Download a small/medium dataset from HuggingFace.

    Uses the `datasets` library. Enforces MAX_DOWNLOAD_BYTES safety cap.
    Returns: {"rows": [...], "row_count": int, "features": dict, "split": str}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets library not installed — pip install datasets")
        return None

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_key = hashlib.md5(
        f"{dataset_id}:{split}:{max_samples}".encode(),
        usedforsecurity=False,
    ).hexdigest()
    cache_file = cache_path / f"{cache_key}.json"

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            logger.info("Using cached dataset: %s (split=%s, rows=%d)", dataset_id, split, len(cached.get("rows", [])))
            return cached
        except (json.JSONDecodeError, OSError):
            pass

    max_bytes = _get_max_download_bytes()
    try:
        # D18: never execute remote dataset code (trust_remote_code=False);
        # truncation at the byte cap propagates via truncated/total_bytes.
        ds = load_dataset(dataset_id, split=split, streaming=True, trust_remote_code=False)

        rows = []
        features = None
        total_bytes = 0
        truncated = False
        for i, example in enumerate(ds):
            if i >= max_samples:
                break
            # Estimate size of this row
            row_json = json.dumps({k: _serialize_value(v) for k, v in example.items()}, default=str)
            total_bytes += len(row_json.encode("utf-8"))
            if total_bytes > max_bytes:
                truncated = True
                logger.warning(
                    "Dataset %s exceeded %d MB limit after %d rows — stopping download",
                    dataset_id, max_bytes // (1024 * 1024), len(rows),
                )
                break
            if features is None:
                features = {k: str(type(v).__name__) for k, v in example.items()}
            rows.append({k: _serialize_value(v) for k, v in example.items()})

        result = {
            "rows": rows,
            "row_count": len(rows),
            "features": features,
            "split": split,
            "dataset_id": dataset_id,
            "truncated": truncated,
            "total_bytes": total_bytes,
        }

        try:
            cache_file.write_text(json.dumps(result, default=str), encoding="utf-8")
        except OSError:
            pass

        logger.info("Downloaded dataset: %s (split=%s, rows=%d, bytes=%d)", dataset_id, split, len(rows), total_bytes)
        return result

    except Exception as exc:
        logger.warning("Failed to download dataset %s: %s", dataset_id, exc, exc_info=True)
        return None


def _serialize_value(v: Any) -> Any:
    """Serialize a dataset value to a JSON-compatible type."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, list):
        serialized = [_serialize_value(item) for item in v[:10]]
        if len(v) > 10:
            return {"_truncated": True, "sample": serialized, "total": len(v)}
        return serialized
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in list(v.items())[:20]}
    return str(v)


# ---------------------------------------------------------------------------
# Unified admissibility check
# ---------------------------------------------------------------------------


def dataset_is_admissible(
    dataset_plan: str,
    local_catalog: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Check if a dataset is admissible — available locally or on HuggingFace.

    This is the unified entry point that agents should call.
    Returns True if the dataset can be used for experiments.
    Works WITHOUT a token for public HuggingFace datasets.
    """
    if not dataset_plan or not dataset_plan.strip():
        return True

    plan = dataset_plan.strip()

    # Synthetic/bundled always pass
    if plan.startswith("synthetic") or plan.startswith("bundled"):
        return True

    # Check local catalog
    if local_catalog:
        catalog_names = {d.get("name", "").lower() for d in local_catalog}
        if plan.lower() in catalog_names:
            return True

    # Check known public datasets (fast path, no API call)
    plan_lower = plan.lower()
    for kw in KNOWN_PUBLIC_DATASETS:
        if kw in plan_lower:
            return True

    # Check HuggingFace — works WITHOUT token for public datasets
    hf_result = dataset_exists_on_hf(plan)
    if hf_result.found:
        return True

    # If HF check failed with error (not 404), fail open
    if hf_result.error and hf_result.error != "not_found":
        logger.info("HF check error for '%s': %s — failing open", plan, hf_result.error)
        return True

    return False
