"""
Hard verification checks: citation resolution + statistical validity.
These are deterministic — not LLM vibe scores.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import config
from .llm import call_llm
from .utils import call_llm_json, parse_json_from_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Citation grounding
# ---------------------------------------------------------------------------

DOI_RE = re.compile(
    r"(?:doi[:\s]*)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)
ARXIV_RE = re.compile(
    # A bare arXiv id is valid, but do not mistake the tail of a DOI such as
    # 10.5555/3295222.3295349 for one.
    r"(?<![\d.])(?:arxiv[:\s]*)?(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
YEAR_CITE_RE = re.compile(
    r"\b([A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?)?)\s*,?\s*(\d{4})\b"
)

CLAIM_TYPES = {"literature_reference", "method_definition", "planned_test", "empirical_result"}

# Quantitative-claim detector shared by the citation gate (L20), the ledger,
# and the release referee: a number with % or a measurement keyword.
QUANTITATIVE_CLAIM_RE = re.compile(
    r"\d+(?:\.\d+)?\s*%|\b(?:mean|std|accuracy|f1|mse|p\s*[<=>])\b",
    re.IGNORECASE,
)

# Paper types with no experiment artifacts (proposal §3.3 recipes). Gates that
# require engineer outputs are inapplicable for these types and waive instead
# of terminal-failing (R2/R8/R12 per-type applicability slice, M1).
NON_EMPIRICAL_TYPES = frozenset({"analytical", "descriptive", "argumentative", "review"})

# M3 (R2 slice): field types with experiments:[] plans and no engineer
# outputs likewise waive outcomes_reported — survey coverage tables and case
# triangulation matrices are the artifacts, not experiment outcomes.
# NOTE (2026-10-06, run d5b5a1db): "empirical" was removed here. Empirical
# manuscripts carry Experiments/Results sections and REQUIRE a bounded MVE
# (bundled_synthetic logistic_regression vs majority_class); waiving it
# produced plan.yaml experiments:[] -> engineering skipped -> empty paper.
FIELD_WAIVER_TYPES = frozenset({"survey", "case_study"})

# Paper types whose release rule requires a verified thesis (§7.1–§7.2).
THESIS_GATE_TYPES = frozenset({"analytical", "argumentative", "descriptive", "review"})

# v5 (f537f638): template filler that must never ship as manuscript prose.
FAKE_PROSE_PHRASES = (
    "verified experimental results are required before making claims",
    "our approach to research on",
    "involves deterministic bundled-synthetic baseline",
    "comprehensive methodology that addresses the key challenges",
    "analysis reveals important insights about the effectiveness",
    "provides guidance for future research directions",
    "standard benchmarks and evaluation metrics to ensure",
    "reproducibility and fair comparison with existing methods",
)

# v5: Methods must name its protocol, not gesture at it.
METHODS_REQUIRED_TOKENS = (
    "resampling unit",
    "bootstrap replicates",
    "ci type",
    "clients",
    "seeds",
    "dataset",
    "model",
    "baseline",
)

# v5: topics making inferential stats claims.
STATISTICAL_TOPIC_RE = re.compile(
    r"bootstrap|confidence interval|\bCI\b|hypothesis test|t-test|t test|p-value|p value|statistical",
    re.IGNORECASE,
)

# v5: comparative adjectives that require a measured baseline + difference CI.
COMPARATIVE_CLAIM_RE = re.compile(
    r"wider than (previously )?reported|often include zero|significantly (better|wider|higher|lower)|outperform",
    re.IGNORECASE,
)


def is_statistical_topic(topic: Any) -> bool:
    if isinstance(topic, dict):
        text = " ".join(str(topic.get(k) or "") for k in ("title", "description", "rationale"))
    else:
        text = str(topic or "")
    return bool(STATISTICAL_TOPIC_RE.search(text))


def _arxiv_id_plausible(aid: str) -> bool:
    """Fail-closed arXiv plausibility: format + real YYMM + not future-dated."""
    m = re.fullmatch(r"(\d{2})(\d{2})\.(\d{4,5})(v\d+)?", str(aid or "").strip())
    if not m:
        return False
    yy, mm = int(m.group(1)), int(m.group(2))
    if not 1 <= mm <= 12:
        return False
    if yy < 7 or yy > 26:  # arXiv YY range 2007..2026; beyond current year is future
        return False
    if yy == 26 and mm > 10:  # cutoff Oct 2026
        return False
    return True


def filler_prose_hits(text: str) -> List[str]:
    lowered = (text or "").lower()
    return [p for p in FAKE_PROSE_PHRASES if p in lowered]


def methods_completeness_gaps(text: str) -> List[str]:
    """Missing protocol tokens in a Methods section (v5: no gesture-methods)."""
    lowered = (text or "").lower()
    gaps = []
    checks = {
        "resampling unit": ("resampl" in lowered and "unit" in lowered) or "resampling unit" in lowered,
        "bootstrap replicates": bool(re.search(r"\bb\s*=\s*\d+|replicates|bootstrap", lowered)),
        "ci type": ("percentile" in lowered or "bca" in lowered or "ci type" in lowered),
        "clients": ("client" in lowered),
        "seeds": ("seed" in lowered),
        "dataset": ("dataset" in lowered or "synthetic" in lowered or "femnist" in lowered or "cifar" in lowered),
        "model": ("model" in lowered or "fedavg" in lowered or "fedprox" in lowered or "scaffold" in lowered or "logistic" in lowered),
        "baseline": ("baseline" in lowered or "comparison" in lowered),
    }
    for token, ok in checks.items():
        if not ok:
            gaps.append(token)
    return gaps


def validate_experiment_evidence(
    engineer_outputs: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Blocking evidence defects: fallback, degenerate variance, underpowered, missing CI/arms."""
    findings: List[Dict[str, Any]] = []
    outputs = engineer_outputs or {}
    if not outputs:
        return findings
    for name, output in outputs.items():
        if not isinstance(output, dict):
            continue
        if str(output.get("outcome") or "") == "fallback_synthetic":
            findings.append({
                "check": "exploratory_only",
                "blocking": True,
                "details": f"{name}: fallback_synthetic is exploratory plumbing, not publishable evidence",
            })
            continue
        metrics = output.get("aggregate_metrics") or {}
        for metric, agg in metrics.items():
            if not isinstance(agg, dict):
                continue
            n = int(agg.get("n") or 0)
            std = agg.get("std")
            values = agg.get("values") or []
            try:
                std_f = float(std) if std is not None else None
            except (TypeError, ValueError):
                std_f = None
            if values and len(set(float(v) for v in values if isinstance(v, (int, float)))) <= 1:
                findings.append({
                    "check": "degenerate_variance",
                    "blocking": True,
                    "details": f"{name}.{metric}: identical values across seeds (std=0, no seed variation)",
                })
            elif std_f == 0.0 and n <= 3:
                findings.append({
                    "check": "degenerate_variance",
                    "blocking": True,
                    "details": f"{name}.{metric}: std=0 with n={n}, cannot support interval inference",
                })
            if n and n < 3:
                findings.append({
                    "check": "underpowered",
                    "blocking": True,
                    "details": f"{name}.{metric}: n={n} below minimum 3 for uncertainty estimates",
                })
    # Statistical plans must carry comparison arms + CI artifacts.
    experiments = (plan or {}).get("experiments", []) if isinstance(plan, dict) else []
    topic_text = str((plan or {}).get("title") or "") if isinstance(plan, dict) else ""
    if any(isinstance(e, dict) and re.search(r"bootstrap|confidence|accuracy_diff", str(e), re.I) for e in experiments) or STATISTICAL_TOPIC_RE.search(topic_text):
        names = [e.get("name") for e in experiments if isinstance(e, dict)]
        if len(names) < 1:
            findings.append({"check": "missing_comparison_arms", "blocking": True, "details": "statistical claim requires >=1 experiment with difference CIs"})
        for name in names:
            out = outputs.get(name) or {}
            boot = (out.get("bootstrap") or {}) if isinstance(out, dict) else {}
            has_ci = bool(boot.get("ci_low") is not None and boot.get("ci_high") is not None)
            metrics = (out.get("aggregate_metrics") or {}) if isinstance(out, dict) else {}
            metric_ci = any(isinstance(v, dict) and ("ci_low" in v or "confidence_interval_95" in v or "ci" in v) for v in metrics.values())
            if isinstance(out, dict) and out.get("success") and not (has_ci or metric_ci):
                findings.append({
                    "check": "missing_confidence_interval",
                    "blocking": True,
                    "details": f"{name}: statistical topic requires a computed CI artifact (bootstrap ci_low/ci_high), not prose",
                })
    return findings


PREVALENCE_CLAIM_RE = re.compile(
    r"(rarely|never|seldom|no (papers?|studies?|work) (report|use|contain)|absence of|do not contain|does not contain).{0,80}(confidence interval|hypothesis test|statistical|p-value|uncertainty)",
    re.IGNORECASE,
)


def validate_gap_prevalence_claims(
    sections: Dict[str, str],
    topic: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    corpus: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """v5: 'literature rarely reports X' requires a counted extraction table, not vibes."""
    text = "\n".join(v for v in (sections or {}).values() if isinstance(v, str))
    if not PREVALENCE_CLAIM_RE.search(text or ""):
        return []
    blob = json.dumps({"topic": topic or {}, "plan": plan or {}, "corpus": corpus or {}}, sort_keys=True, default=str).lower()
    has_counts = bool(re.search(r"screened|extraction|prisma|n\s*=\s*\d+.*(ci|test)|%.*report", blob))
    extraction_rows = 0
    if isinstance(corpus, dict):
        rows = corpus.get("extraction")
        if isinstance(rows, list):
            extraction_rows = len(rows)
    if has_counts or extraction_rows >= 5:
        return []
    return [{
        "check": "unquantified_gap_claim",
        "blocking": True,
        "details": "prevalence claim about literature ('rarely/never reports CIs') requires a counted extraction table (screened N, % with CI/test)",
    }]


def validate_bootstrap_claims(
    sections: Dict[str, str],
    engineer_outputs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Comparative adjectives require measured artifacts (v5: no free 'wider/often')."""
    findings: List[Dict[str, Any]] = []
    text = "\n".join(v for v in (sections or {}).values() if isinstance(v, str))
    if not COMPARATIVE_CLAIM_RE.search(text or ""):
        return findings
    outputs = engineer_outputs or {}
    has_baseline_width = False
    has_include_zero_rate = False
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        blob = json.dumps(output, sort_keys=True, default=str).lower()
        if "baseline_width" in blob or "width_ratio" in blob or "prior_width" in blob:
            has_baseline_width = True
        if "includes_zero" in blob or "include_zero" in blob or "covers_zero" in blob:
            has_include_zero_rate = True
        boot = output.get("bootstrap") or {}
        if isinstance(boot, dict) and boot.get("ci_low") is not None:
            has_include_zero_rate = has_include_zero_rate or True
    if re.search(r"wider than (previously )?reported", text, re.I) and not has_baseline_width:
        findings.append({
            "check": "unmeasured_baseline",
            "blocking": True,
            "details": "'wider than reported' requires a measured baseline interval width artifact",
        })
    if re.search(r"often include zero", text, re.I) and not has_include_zero_rate:
        findings.append({
            "check": "unquantified_zero_coverage",
            "blocking": True,
            "details": "'often include zero' requires a quantified includes-zero rate over computed difference CIs",
        })
    return findings


def extract_citation_ids(text: str) -> Dict[str, List[str]]:
    dois = list({m.group(1).rstrip(".,;)") for m in DOI_RE.finditer(text)})
    arxiv_ids = list({m.group(1) for m in ARXIV_RE.finditer(text)})
    author_year = [f"{m.group(1)} {m.group(2)}" for m in YEAR_CITE_RE.finditer(text)]
    return {"dois": dois, "arxiv_ids": arxiv_ids, "author_year": author_year}


def extract_citation_metadata(text: str) -> Dict[str, Dict[str, Any]]:
    """Extract nearby title/author fields from prose and BibTeX-like entries."""
    metadata: Dict[str, Dict[str, Any]] = {}
    blocks = re.split(r"(?=@\w+\{)|\n\s*\n", text)
    for block in blocks:
        identifiers = extract_citation_ids(block)
        title_match = re.search(r"(?:title\s*[=:]\s*\{?|\"|')([^}\"'\n]+)", block, re.I)
        author_match = re.search(r"author\s*[=:]\s*\{?([^}\n]+)", block, re.I)
        title = title_match.group(1).strip() if title_match else ""
        authors = [part.strip() for part in re.split(r"\s+and\s+|,", author_match.group(1)) if part.strip()] if author_match else []
        for identifier in identifiers["dois"] + identifiers["arxiv_ids"]:
            if title or authors:
                metadata[identifier] = {"title": title, "authors": authors}
    return metadata


def resolve_doi(doi: str) -> Dict[str, Any]:
    """Resolve DOI via CrossRef. Returns {resolved: bool, ...}.

    R5 (§6.1): the HTTP fetch is routed through the API gateway
    (``crossref`` provider: rate limiting + circuit breaking + retry with
    backoff). A gateway failure degrades to fail-closed ``resolved: False`` —
    never a raise, never a pass.
    """
    def _do_fetch() -> requests.Response:
        url = f"https://api.crossref.org/works/{doi}"
        headers = {"User-Agent": f"ScholarGraph/2.0 (mailto:{config.openalex_email})"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 429:
            from .api_gateway import RateLimitError

            raise RateLimitError("crossref", float(response.headers.get("Retry-After", "5")))
        return response

    try:
        from .api_gateway import get_gateway

        _gw = get_gateway()
        if not _gw.is_available("crossref"):
            return {"resolved": False, "doi": doi, "error": "crossref breaker open (skipped)"}
        r = _gw.request("crossref", _do_fetch, retries=2, backoff_base=2.0)
        if r.status_code == 200:
            msg = r.json().get("message", {})
            title = ""
            if msg.get("title"):
                title = msg["title"][0]
            return {
                "resolved": True,
                "doi": doi,
                "title": title,
                "authors": [author.get("family") or author.get("literal") or "" for author in (msg.get("author") or [])],
                "year": (msg.get("published-print") or msg.get("published-online") or {})
                .get("date-parts", [[None]])[0][0],
                "container": (msg.get("container-title") or [""])[0],
            }
        return {"resolved": False, "doi": doi, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"resolved": False, "doi": doi, "error": str(e)}


# R5 (§6.1): DOI retry budget when the gateway layer is bypassed (e.g. a
# monkeypatched resolver in tests, or a direct call). Only transient-looking
# failures are retried; definitive misses (404 / "not found") fail fast so
# fake-DOI stays fail-closed and offline tests stay fast.
_DOI_MAX_ATTEMPTS = 2
_DOI_RETRY_BACKOFF_SEC = 1.0
_TRANSIENT_DIO_HINTS = (
    "429", "500", "502", "503", "504", "timeout", "timed out", "deadline",
    "connection", "reset", "try again", "overloaded", "unavailable",
    "temporary", "econn", "retry",
)


def _doi_error_is_transient(result: Dict[str, Any]) -> bool:
    error = str((result or {}).get("error") or "").lower()
    if not error:
        return False
    if "404" in error or "not found" in error:
        return False
    return any(hint in error for hint in _TRANSIENT_DIO_HINTS)


def _resolve_doi_with_retry(doi: str) -> Dict[str, Any]:
    """Resolve one DOI with a bounded transient retry (R5).

    Calls the module-global ``resolve_doi`` at call time so test doubles that
    patch ``verification.resolve_doi`` keep working. Fail-closed: an
    unresolvable DOI still returns ``resolved: False``.
    """
    last: Dict[str, Any] = {"resolved": False, "doi": doi, "error": "no attempts made"}
    for attempt in range(max(1, _DOI_MAX_ATTEMPTS)):
        try:
            last = resolve_doi(doi)
        except Exception as e:
            last = {"resolved": False, "doi": doi, "error": str(e)}
        if bool(last.get("resolved")) or not _doi_error_is_transient(last):
            return last
        if attempt + 1 < max(1, _DOI_MAX_ATTEMPTS):
            time.sleep(_DOI_RETRY_BACKOFF_SEC * (attempt + 1))
    return last


def resolve_arxiv(arxiv_id: str) -> Dict[str, Any]:
    """Resolve an arXiv ID via the export API, routed through the API gateway.

    AGENTS inviolable rule: all external calls go through
    ``gateway.request(provider, fn, ...)`` (``arxiv`` provider: 0.33 req/s +
    breaker + AIMD + retry). Degrades to fail-closed ``resolved: False`` —
    never a raise, never a pass.
    """
    def _do_fetch() -> requests.Response:
        url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(url, timeout=15)
        if response.status_code == 429:
            from .api_gateway import RateLimitError

            raise RateLimitError("arxiv", float(response.headers.get("Retry-After", "5")))
        return response

    def _parse(r: requests.Response) -> Dict[str, Any]:
        if r.status_code == 200 and "<entry>" in r.text:
            title_m = re.search(r"<title>(.*?)</title>", r.text, re.DOTALL)
            title = title_m.group(1).strip() if title_m else ""
            if title.lower().startswith("arxiv query"):
                titles = re.findall(r"<title>(.*?)</title>", r.text, re.DOTALL)
                title = titles[1].strip() if len(titles) > 1 else title
            authors = re.findall(r"<name>(.*?)</name>", r.text, re.DOTALL)
            return {"resolved": True, "arxiv_id": arxiv_id, "title": title, "authors": [a.strip() for a in authors]}
        return {"resolved": False, "arxiv_id": arxiv_id, "error": f"HTTP {r.status_code}"}

    try:
        from .api_gateway import get_gateway

        gateway = get_gateway()
        if not gateway.is_available("arxiv"):
            return {"resolved": False, "arxiv_id": arxiv_id, "error": "arxiv breaker open (skipped)"}
        r = gateway.request("arxiv", _do_fetch, retries=2, backoff_base=5.0, backoff_max=60.0)
        return _parse(r)
    except Exception:
        pass
    # Fallback: bounded direct fetch (e.g. gateway unavailable in minimal
    # test doubles). Single attempt, no sleep pacing — fail-closed.
    try:
        url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        r = requests.get(url, timeout=15)
        return _parse(r)
    except Exception as e:
        return {"resolved": False, "arxiv_id": arxiv_id, "error": str(e)}


def _normalize_words(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", (value or "").lower()) if len(word) > 2}


def _title_tokens(value: str) -> set[str]:
    """Title tokens per §8.8.2 (L19/R5): lowercase alphanumeric, no stopword removal."""
    return set(re.findall(r"[a-z0-9]+", (value or "").lower()))


def title_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of title token sets (§8.8.2). Pass at >= 0.8."""
    left, right = _title_tokens(a), _title_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _metadata_diff(result: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    """Compare writer-supplied bibliography metadata with resolved metadata.

    L19/R5 (§8.8.2): title passes at Jaccard >= 0.8 over lowercase
    alphanumeric tokens (no stopword removal). Authors check unchanged
    (wanted subset of actual on surname tokens).
    """
    expected_title = expected.get("title") or ""
    expected_authors = expected.get("authors") or []
    differences: Dict[str, Any] = {}
    if expected_title:
        jaccard = title_jaccard(str(result.get("title", "")), str(expected_title))
        if not _title_tokens(str(expected_title)) or jaccard < 0.8:
            differences["title"] = {
                "expected": expected_title,
                "actual": result.get("title", ""),
                "jaccard": round(jaccard, 3),
            }
    if expected_authors:
        actual = _normalize_words(" ".join(result.get("authors") or []))
        wanted = _normalize_words(" ".join(expected_authors) if isinstance(expected_authors, list) else expected_authors)
        if not wanted or not wanted.issubset(actual):
            differences["authors"] = {
                "expected": expected_authors,
                "actual": result.get("authors") or [],
            }
    return differences


def _metadata_matches(result: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Compatibility predicate for callers that only need a boolean."""
    return not _metadata_diff(result, expected)


def verify_citations(text: str, citation_metadata: Optional[Dict[str, Dict[str, Any]]] = None, quarantine_singles: bool = False) -> Dict[str, Any]:
    """
    Hard check: every DOI/arXiv ID must resolve.
    Author-year citations without IDs are flagged as unverifiable (soft fail).

    quarantine_singles (R5 slice, M1): exactly one failed hard ID is moved to
    ``quarantined`` (repair pass) instead of failing the check, so the run
    continues. Defaults to False — legacy fail-closed behavior preserved.
    """
    ids = extract_citation_ids(text)
    citation_metadata = citation_metadata if citation_metadata is not None else extract_citation_metadata(text)
    resolved = []
    failed = []
    mismatched = []
    unverifiable = []

    for doi in ids["dois"]:
        # R5 (§6.1): bounded transient retry around DOI resolution. Legacy
        # fail-closed default preserved (quarantine_singles still opt-in).
        result = _resolve_doi_with_retry(doi)
        expected = citation_metadata.get(doi) if citation_metadata else None
        if result["resolved"] and expected:
            differences = _metadata_diff(result, expected)
            if differences:
                result = {
                    **result,
                    "error": "resolved metadata does not match claimed bibliography",
                    "metadata_mismatch": True,
                    "metadata_diff": differences,
                    "expected_metadata": expected,
                }
                mismatched.append(result)
        (resolved if result["resolved"] and not result.get("metadata_mismatch") else failed).append(result)

    for aid in ids["arxiv_ids"]:
        if not _arxiv_id_plausible(aid):
            result = {"resolved": False, "arxiv_id": aid, "error": "implausible arXiv ID (format/date out of range)"}
            failed.append(result)
            continue
        result = resolve_arxiv(aid)
        expected = citation_metadata.get(aid) if citation_metadata else None
        if result["resolved"] and expected:
            differences = _metadata_diff(result, expected)
            if differences:
                result = {
                    **result,
                    "error": "resolved metadata does not match claimed bibliography",
                    "metadata_mismatch": True,
                    "metadata_diff": differences,
                    "expected_metadata": expected,
                }
                mismatched.append(result)
        (resolved if result["resolved"] and not result.get("metadata_mismatch") else failed).append(result)

    for ay in ids["author_year"]:
        # Soft: no API proof without DOI
        unverifiable.append({"citation": ay, "reason": "author-year without DOI/arXiv"})

    n_hard = len(ids["dois"]) + len(ids["arxiv_ids"])
    n_failed = len(failed)
    if n_hard == 0:
        # L20: zero hard citations with quantitative claims fails — numbers
        # without a resolvable source are ungrounded by construction.
        if QUANTITATIVE_CLAIM_RE.search(text or ""):
            score = 0.0
            passed = False
            note = "Quantitative claims present with zero DOI/arXiv citations to ground them"
        else:
            score = 5.0 if unverifiable else 10.0  # no citable IDs → neutral/ok
            passed = True
            note = "No DOI/arXiv IDs found to verify"
    else:
        score = max(0.0, 10.0 * (1.0 - n_failed / n_hard))
        passed = n_failed == 0
        note = f"Resolved {n_hard - n_failed}/{n_hard} citation IDs"

    quarantined: List[Dict[str, Any]] = []
    if quarantine_singles and n_failed == 1 and failed:
        quarantined = failed
        failed = []
        passed = True
        note = f"Resolved {n_hard - 1}/{n_hard} citation IDs; 1 citation quarantined to repair pass"

    return {
        "passed": passed,
        "score": score,
        "note": note,
        "resolved": resolved,
        "failed": failed,
        "quarantined": quarantined,
        "metadata_mismatches": mismatched,
        "unverifiable": unverifiable[:20],
        "ids_found": ids,
    }


def classify_claim(text: str, evidence_artifact_ids: Optional[List[str]] = None) -> str:
    """Classify a claim conservatively before it enters the evidence ledger."""
    lowered = text.lower()
    if re.search(r"\b(doi|arxiv|et al\.?|\(\d{4}\))", lowered):
        return "literature_reference"
    if any(token in lowered for token in ("we define", "we use", "algorithm", "methodology", "equation")):
        return "method_definition"
    if any(token in lowered for token in ("will evaluate", "plan to", "we propose to", "future experiment")):
        return "planned_test"
    if evidence_artifact_ids and re.search(r"\d+(?:\.\d+)?\s*%|\b(mean|std|accuracy|f1|mse|p\s*[<=>])\b", lowered):
        return "empirical_result"
    return "method_definition"


def cross_section_numeric_consistency(sections: Dict[str, str], engineer_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Reject artifact or cross-section conflicts for one experiment metric."""
    outputs = engineer_outputs or {}
    expected: Dict[str, Dict[str, float]] = {}
    for experiment, output in outputs.items():
        metrics = (output or {}).get("aggregate_metrics") or (output or {}).get("results", {}).get("metrics") or {}
        expected[experiment] = {key: float(value.get("mean", value) if isinstance(value, dict) else value) for key, value in metrics.items() if isinstance(value, (int, float, dict))}
    claims: Dict[tuple[str, str, float], List[str]] = {}
    for section, content in sections.items():
        for experiment, metrics in expected.items():
            for metric in metrics:
                pattern = rf"\b{re.escape(metric)}\b[^\d%]{{0,40}}(\d+(?:\.\d+)?)\s*(%)?"
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    value = float(match.group(1)) / 100 if match.group(2) else float(match.group(1))
                    claims.setdefault((experiment, metric, value), []).append(section)
    conflicts = []
    for experiment, metrics in expected.items():
        for metric in metrics:
            artifact_value = metrics[metric]
            observed = [(value, sections_seen) for (exp, name, value), sections_seen in claims.items() if exp == experiment and name == metric]
            claim_values = {value for value, _ in observed}
            if observed and any(not math.isclose(value, artifact_value, rel_tol=0.05, abs_tol=1e-9) for value in claim_values):
                observed.append((artifact_value, ["structured_artifact"]))
                conflicts.append({
                    "experiment": experiment,
                    "metric": metric,
                    "claims": [{"value": value, "sections": sections_seen} for value, sections_seen in observed],
                    "reason": "manuscript claim disagrees with structured artifact",
                })
            elif len({value for value, _ in observed}) > 1:
                conflicts.append({
                    "experiment": experiment,
                    "metric": metric,
                    "claims": [{"value": value, "sections": sections_seen} for value, sections_seen in observed],
                    "reason": "sections disagree",
                })
    collected = [
        {"experiment": experiment, "metric": metric, "value": value, "sections": sections_seen}
        for (experiment, metric, value), sections_seen in claims.items()
    ]
    return {
        "passed": not conflicts,
        "claims": collected,
        "conflicts": conflicts,
        "note": "No numeric conflicts" if not conflicts else "Conflicting numeric claims",
    }


def preregister_power(planned_effect_size: float, alpha: float = 0.05, target_power: float = 0.8) -> Dict[str, Any]:
    """Compute a prospective two-group sample-size requirement before execution."""
    from scipy import stats

    if planned_effect_size <= 0 or not 0 < alpha < 1 or not 0 < target_power < 1:
        raise ValueError("effect size must be positive; alpha and target_power must be in (0, 1)")
    z_alpha = float(stats.norm.ppf(1 - alpha / 2))
    z_power = float(stats.norm.ppf(target_power))
    required_n = int(math.ceil(2 * ((z_alpha + z_power) / planned_effect_size) ** 2))
    return {"planned_effect_size": float(planned_effect_size), "alpha": float(alpha), "target_power": float(target_power), "required_n_per_group": required_n, "method": "two-group normal approximation"}


def validate_reviewer_checklist(sections: Dict[str, str], engineer_outputs: Optional[Dict[str, Any]] = None, plan: Optional[Dict[str, Any]] = None, paper_type: str = "experimental") -> Dict[str, Any]:
    """Enforce publication checklist requirements deterministically.

    Per-type applicability (R2 slice): a non-empirical plan (``experiments:
    []`` per §4.5) with no engineer outputs waives ``outcomes_reported`` —
    there is nothing to report. An experimental plan with no outputs still
    fails the check.
    """
    text = "\n".join(sections.values())
    outputs = engineer_outputs or {}
    experiments = [exp for exp in (plan or {}).get("experiments", []) if isinstance(exp, dict)]
    if not experiments and not outputs and paper_type in (NON_EMPIRICAL_TYPES | FIELD_WAIVER_TYPES):
        outcomes_reported: bool = True
    else:
        outcomes_reported = bool(outputs) and all(bool(item.get("outcome") or item.get("aggregate_metrics") or item.get("results")) for item in outputs.values() if isinstance(item, dict))
    checks = {
        "limitations": bool(sections.get("Limitations", "").strip()),
        "baselines_declared": all(bool(exp.get("baselines") or exp.get("baseline_comparison")) for exp in experiments),
        "outcomes_reported": outcomes_reported,
        "uncertainty_or_sample_size": not bool(re.search(r"\d+(?:\.\d+)?\s*%|\b(?:accuracy|f1|mse)\b", text, re.I)) or bool(re.search(r"\b(?:n\s*=|std|confidence interval|ci\b)", text, re.I)),
        "literature_evidence": not bool(re.search(r"\b(?:doi|arxiv|et al\.?|\(\d{4}\))", text, re.I)) or bool(re.search(r"retrieved|abstract|source text|literature evidence", text, re.I)),
    }
    return {"passed": all(checks.values()), "checks": checks, "failed": [name for name, passed in checks.items() if not passed]}


def novelty_overlap_check(
    sections: Dict[str, str],
    topic: Optional[Dict[str, Any]] = None,
    threshold: float = 0.65,
    paper_type: str = "experimental",
) -> Dict[str, Any]:
    """Deterministic novelty-plagiarism screen (Gupta & Pruthi, ACL 2025).

    ~24% of AI-generated research documents borrow heavily from prior work
    without acknowledgment. This compares the manuscript's framing text
    (Abstract + Introduction) against the closest prior work and retrieved
    literature evidence using a content-word overlap coefficient.
    Overlap coefficient = |shared| / min(|draft|, |prior|), which is more
    sensitive than Jaccard to a small draft closely paraphrasing one source.

    R11 (M2): review texts use threshold 0.85+ with a framing exemption —
    overlap that cites its sources (DOI/arXiv present in framing) is
    exempt, since reviews legitimately restate prior framing with attribution.
    """
    # R11: review threshold 0.85+ (never lower a caller-supplied higher bar).
    effective_threshold = float(threshold)
    if (paper_type or "experimental") == "review":
        effective_threshold = max(effective_threshold, 0.85)
    prior_sources: List[Dict[str, str]] = []
    structured = (topic or {}).get("structured_hypothesis") or {}
    closest = structured.get("closest_prior_work") if isinstance(structured, dict) else None
    if isinstance(closest, dict):
        text = " ".join(str(closest.get(key) or "") for key in ("title", "abstract", "summary", "contribution") if closest.get(key))
        if text.strip():
            prior_sources.append({"source": "closest_prior_work", "text": text})
    elif isinstance(closest, str) and closest.strip():
        prior_sources.append({"source": "closest_prior_work", "text": closest})
    for paper in (topic or {}).get("literature_evidence") or []:
        if not isinstance(paper, dict):
            continue
        text = " ".join(
            str(paper.get(key) or "") for key in ("title", "abstract")
        ).strip()
        if text:
            prior_sources.append({"source": paper.get("title") or paper.get("doi") or paper.get("arxiv_id") or "literature", "text": text})

    framing = "\n".join(
        sections.get(name, "") for name in ("Abstract", "Introduction")
        if isinstance(sections.get(name), str)
    )
    if not prior_sources or not framing.strip():
        return {
            "passed": True,
            "max_overlap": 0.0,
            "threshold": threshold,
            "overlaps": [],
            "findings": [],
            "note": "no prior sources to compare against",
        }

    def content_tokens(text: str) -> set:
        return {word for word in re.findall(r"[a-z]{4,}", (text or "").lower())}

    draft_tokens = content_tokens(framing)
    overlaps: List[Dict[str, Any]] = []
    for source in prior_sources:
        source_tokens = content_tokens(source["text"])
        if not source_tokens:
            continue
        shared = draft_tokens & source_tokens
        denominator = min(len(draft_tokens), len(source_tokens))
        coefficient = (len(shared) / denominator) if denominator else 0.0
        overlaps.append({"source": source["source"], "overlap_coefficient": round(coefficient, 3), "shared_terms": len(shared)})
    max_overlap = max((item["overlap_coefficient"] for item in overlaps), default=0.0)
    flagged = [item for item in overlaps if item["overlap_coefficient"] > effective_threshold]
    # R11 framing exemption (review only): overlapping framing that cites its
    # sources is attributed restatement, not unattributed borrowing.
    exempted = False
    if flagged and (paper_type or "experimental") == "review":
        ids = extract_citation_ids(framing)
        if ids["dois"] or ids["arxiv_ids"]:
            exempted = True
            flagged = []
    return {
        "passed": not flagged,
        "max_overlap": max_overlap,
        "threshold": effective_threshold,
        "paper_type": paper_type,
        "framing_exempt": exempted,
        "overlaps": sorted(overlaps, key=lambda item: item["overlap_coefficient"], reverse=True)[:5],
        "note": (
            f"Max framing overlap {max_overlap:.2f} vs threshold {effective_threshold}"
            + (" (review framing exempt: sources cited)" if exempted else "")
            if overlaps else "no prior sources to compare against"
        ),
        "findings": [
            {"source": item["source"], "overlap_coefficient": item["overlap_coefficient"]}
            for item in overlaps if item["overlap_coefficient"] > effective_threshold
        ] if not exempted else [],
    }


def verify_review(
    corpus: Optional[Dict[str, Any]] = None,
    sections: Optional[Dict[str, str]] = None,
    paper_type: str = "review",
) -> Dict[str, Any]:
    """Review release gate (§7.1, M2): reconcile + completeness ≥0.9 + recall ≥4/5 + spot-grounded.

    ``corpus`` is the systematic_review corpus (§3.1): ``{prisma, screen_log,
    extraction, seed_recall}`` as produced by ``ReviewAgent.run_review``.
    Accepts both the full corpus shape and bare counts for offline tests.

    Recall below 4/5 demotes ``systematic → scoping`` + banner (not terminal):
    the run still passes when PRISMA + extraction + synthesis hold, with
    ``mode="scoping"`` recorded for the release formula (§7.2).
    """
    from agents.review_agent import (
        extraction_completeness as _extraction_completeness,
        prisma_reconcile as _prisma_reconcile,
    )

    corpus = dict(corpus or {})
    sections = sections or {}
    findings: List[Dict[str, Any]] = []

    # -- PRISMA reconciliation (must balance exactly) --
    prisma_counts = corpus.get("prisma") if isinstance(corpus.get("prisma"), dict) else corpus
    if not isinstance(prisma_counts, dict) or not prisma_counts:
        # Corpus shape carries counts under "prisma" plus a precomputed flag.
        prisma_counts = corpus.get("prisma") if isinstance(corpus.get("prisma"), dict) else {}
    prisma = _prisma_reconcile(prisma_counts) if prisma_counts else {
        "passed": bool(corpus.get("prisma_reconciled", False)),
        "reconciled": bool(corpus.get("prisma_reconciled", False)),
        "findings": [] if corpus.get("prisma_reconciled") else [
            {"check": "prisma", "blocking": True, "details": "PRISMA counts missing or unreconciled"}],
        "counts": prisma_counts,
    }
    if not prisma["passed"]:
        findings.extend(prisma.get("findings", []))

    # -- Extraction completeness ≥ 0.9 --
    extraction_rows = corpus.get("extraction")
    if isinstance(extraction_rows, list):
        extraction = _extraction_completeness(extraction_rows)
    elif isinstance(corpus.get("extraction_completeness"), (int, float)):
        comp = float(corpus["extraction_completeness"])
        extraction = {"passed": comp >= 0.9, "completeness": round(comp, 3),
                      "findings": [] if comp >= 0.9 else [
                          {"check": "extraction", "blocking": True,
                           "details": f"extraction completeness {comp:.2f} < 0.9"}]}
    else:
        extraction = {"passed": False, "completeness": 0.0,
                      "findings": [{"check": "extraction", "blocking": True,
                                    "details": "no extraction rows"}]}
    if not extraction["passed"]:
        findings.extend(extraction.get("findings", []))

    # -- Seed recall ≥ 4/5 (demote, not terminal) --
    recall_info = corpus.get("seed_recall") if isinstance(corpus.get("seed_recall"), dict) else {}
    recall_val: Optional[float] = None
    if isinstance(recall_info, dict) and isinstance(recall_info.get("recall"), (int, float)):
        recall_val = float(recall_info["recall"])
    elif isinstance(corpus.get("recall"), (int, float)):
        recall_val = float(corpus["recall"])
    demote_to_scoping = False
    banner: Optional[str] = None
    recall_passed = True
    if recall_val is not None:
        recall_passed = recall_val >= 0.8
        if not recall_passed:
            demote_to_scoping = True
            banner = (corpus.get("seed_recall", {}) or {}).get("banner") if isinstance(corpus.get("seed_recall"), dict) else None
            banner = banner or (f"Seed recall {recall_val:.2f} below 4/5 — demoted systematic → scoping "
                                "(banner, not terminal).")
    elif isinstance(recall_info, dict) and recall_info.get("demote_to_scoping"):
        demote_to_scoping = True
        banner = recall_info.get("banner")
        recall_passed = False
    else:
        # Recall never measured: verify stays pass-oriented (recall is a
        # banner, not a blocking finding), but surface an advisory row so the
        # release gate's recall>=4/5 requirement never surprises approve.
        # Mirrors core/release.py's fail-closed absent-recall handling.
        findings.append({"check": "seed_recall", "blocking": False,
                         "details": "seed recall not measured — release requires recall>=4/5"})
        recall_passed = False
    mode = "scoping" if demote_to_scoping else str(corpus.get("mode") or "systematic")

    # -- Synthesis spot-grounding --
    synthesis = corpus.get("synthesis") if isinstance(corpus.get("synthesis"), dict) else {}
    synthesis_ok = True
    synthesis_note = ""
    if synthesis:
        mode_s = str(synthesis.get("mode") or "")
        if mode_s == "meta-analysis":
            synthesis_ok = synthesis.get("i_squared") is not None and int(synthesis.get("n_effects") or 0) >= 3
            synthesis_note = "meta-analysis with I²" if synthesis_ok else "meta-analysis missing I² or <3 effects"
        else:
            synthesis_ok = bool(synthesis.get("narrative")) and bool(synthesis.get("why_not_meta"))
            synthesis_note = "narrative synthesis with why-not-meta" if synthesis_ok else "synthesis missing narrative or why-not-meta"
        # Spot check against the manuscript when sections are supplied: at
        # least one extracted DOI must be cited in the text.
        if synthesis_ok and sections:
            manuscript = "\n".join(v for v in sections.values() if isinstance(v, str))
            dois = [str(r.get("doi") or "") for r in (extraction_rows or []) if isinstance(r, dict) and r.get("doi")]
            if dois and not any(d.lower() in manuscript.lower() for d in dois if d):
                synthesis_ok = False
                synthesis_note = "synthesis spot-grounding failed: no extracted DOI cited in manuscript"
    else:
        # No synthesis recorded — only fail when extraction already holds
        # (a corpus without synthesis cannot ground a review manuscript).
        synthesis_ok = False
        synthesis_note = "no synthesis recorded"
    if not synthesis_ok:
        findings.append({"check": "synthesis", "blocking": True, "details": synthesis_note})

    # Recall demotion is a banner, never a blocking finding (§4.6 step 6).
    passed = prisma["passed"] and extraction["passed"] and synthesis_ok
    return {
        "passed": passed,
        "mode": mode,
        "banner": banner,
        "demote_to_scoping": demote_to_scoping,
        "recall": recall_val,
        "recall_passed": recall_passed,
        "prisma": prisma,
        "extraction": extraction,
        "synthesis_ok": synthesis_ok,
        "synthesis_note": synthesis_note,
        "findings": findings,
    }


def verify_survey(
    analysis: Optional[Dict[str, Any]] = None,
    sections: Optional[Dict[str, str]] = None,
    instrument: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Survey release gate (§7.1, M3): instrument + nonresponse + no-fishing + synthetic screen.

    ``analysis`` is the ``SurveyAgent.run_analysis`` bundle
    (``{validation, descriptives, alpha, nonresponse, fishing, synthetic,
    instrument}``); ``instrument`` may also be passed separately (state
    shape) and wins when the bundle lacks it. Single-source-style
    advisory rows (alpha warn, fishing advisory) never block; a tripped
    synthetic screen, a missing instrument, unreported nonresponse, or a
    leaked fished subgroup blocks release.
    """
    analysis = dict(analysis or {})
    sections = sections or {}
    instrument = dict(instrument or analysis.get("instrument") or {})
    findings: List[Dict[str, Any]] = []

    # -- instrument present (codebook + response hashes) --
    instrument_ok = bool(instrument.get("response_hash")) and bool(
        instrument.get("codebook_hash") or instrument.get("questionnaire"))
    if not instrument_ok:
        findings.append({"check": "instrument", "blocking": True,
                         "details": "survey instrument missing codebook/response hashes"})

    # -- nonresponse reported (item rates + unit rate) --
    nonresponse = analysis.get("nonresponse") if isinstance(analysis.get("nonresponse"), dict) else instrument.get("nonresponse")
    nonresponse_ok = (
        isinstance(nonresponse, dict)
        and isinstance(nonresponse.get("item_rates"), dict) and bool(nonresponse.get("item_rates"))
        and isinstance(nonresponse.get("unit_rate"), (int, float)))
    if not nonresponse_ok:
        findings.append({"check": "nonresponse", "blocking": True,
                         "details": "nonresponse item rates + unit rate must both be reported"})

    # -- synthetic screen (blocking when tripped) --
    synthetic = analysis.get("synthetic") if isinstance(analysis.get("synthetic"), dict) else {}
    synthetic_ok = not bool(synthetic.get("blocking"))
    if not synthetic_ok:
        findings.append({"check": "synthetic_screen", "blocking": True,
                         "details": "synthetic respondents suspected: " + "; ".join(synthetic.get("reasons") or [])})

    # -- no-fishing: withheld subgroups must not leak into the manuscript --
    fishing = analysis.get("fishing") if isinstance(analysis.get("fishing"), dict) else {}
    withheld = [str(w or "") for w in (fishing.get("withheld") or []) if str(w or "").strip()]
    fishing_ok = True
    if withheld and sections:
        manuscript = "\n".join(v for v in sections.values() if isinstance(v, str)).lower()
        leaked = [w for w in withheld if w.lower() in manuscript]
        if leaked:
            fishing_ok = False
            findings.append({"check": "fishing_guard", "blocking": True,
                             "details": f"non-preregistered subgroups reported in manuscript: {leaked}"})

    # -- response-table validity --
    validation = analysis.get("validation") if isinstance(analysis.get("validation"), dict) else {}
    table_ok = bool(validation.get("passed", True)) if validation else True
    if validation and not validation.get("passed"):
        findings.append({"check": "response_table", "blocking": True,
                         "details": "; ".join(validation.get("reasons") or ["response_table invalid"])})

    passed = instrument_ok and nonresponse_ok and synthetic_ok and fishing_ok and table_ok
    return {"passed": passed, "instrument_ok": instrument_ok,
            "nonresponse_ok": nonresponse_ok, "synthetic_ok": synthetic_ok,
            "fishing_ok": fishing_ok, "table_ok": table_ok,
            "withheld": withheld, "findings": findings}


def verify_case(
    protocol: Optional[Dict[str, Any]] = None,
    sections: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Case release gate (§7.1/§4.9, M3): triangulation + rival check.

    Single-source flags are warnings with limitation entries (run
    continues); a failed pack or a missing rival/gap blocks release.
    ``protocol`` is the ``CaseAgent.run_protocol`` bundle.
    """
    protocol = dict(protocol or {})
    sections = sections or {}
    findings: List[Dict[str, Any]] = []

    pack = protocol.get("pack") if isinstance(protocol.get("pack"), dict) else {}
    pack_ok = bool(pack.get("passed", False)) if pack else False
    if not pack_ok:
        reasons = list(pack.get("reasons") or ["case pack invalid"])
        findings.append({"check": "case_pack", "blocking": True, "details": "; ".join(reasons)})

    triang = protocol.get("triangulation") if isinstance(protocol.get("triangulation"), dict) else {}
    flags = list(triang.get("single_source_flags") or [])
    limitations = list(protocol.get("limitations") or [])
    # Every flag needs a visible limitation entry (never silent).
    triangulation_ok = True
    if flags and not limitations:
        triangulation_ok = False
        findings.append({"check": "triangulation", "blocking": True,
                         "details": "single-source flags without limitation entries"})
    for flag in flags:
        findings.append({"check": "single_source_flag", "blocking": False,
                         "details": str(flag.get("claim", ""))[:300]})

    rival = protocol.get("rival") if isinstance(protocol.get("rival"), dict) else {}
    rival_ok = bool(rival.get("passed", False)) if rival else False
    if not rival_ok:
        details = "; ".join(f.get("details", "") for f in (rival.get("findings") or [{"details": "rival check missing"}]))
        findings.append({"check": "rival_check", "blocking": True, "details": details})

    blocking = [f for f in findings if f.get("blocking")]
    passed = pack_ok and rival_ok and triangulation_ok and not blocking
    return {"passed": passed, "pack_ok": pack_ok,
            "triangulation_ok": triangulation_ok, "rival_ok": rival_ok,
            "single_source_flags": flags, "n_flagged": len(flags),
            "findings": findings}


def verify_causal(
    causal_claim: Optional[Dict[str, Any]] = None,
    sections: Optional[Dict[str, str]] = None,
    indep_pass: bool = False,
) -> Dict[str, Any]:
    """Cause-effect release gate (§7.1, M4): audit pass + assumption/sensitivity section + verb governor.

    ``causal_claim`` is the ``CausalAgent.run_audit`` bundle
    (``{id_strategy, verb_level, sensitivity, ...}``). A naive strategy or a
    missing claim blocks release (the workflow downgrades to analytical
    instead — this gate states the block, the node performs the reroute). A
    manuscript that keeps strong causal verbs while the level is
    associational is blocked by the governor; placebo infeasibility is a
    limitation entry, never a block (§4.7).
    """
    from agents.causal_agent import _ASSOCIATIONAL_BANNED

    causal_claim = dict(causal_claim or {})
    sections = sections or {}
    findings: List[Dict[str, Any]] = []

    strategy = str(causal_claim.get("id_strategy") or "naive")
    if not causal_claim:
        findings.append({"check": "causal_audit", "blocking": True,
                         "details": "no causal audit bundle: cause_effect requires CausalAgent.run_audit"})
    elif strategy == "naive":
        findings.append({"check": "identification", "blocking": True,
                         "details": "naive observational design without identification: downgrade to analytical"})

    # Assumption/sensitivity section: the manuscript must show its
    # identifying assumptions and a robustness note (§7.1).
    manuscript = "\n".join(v for v in sections.values() if isinstance(v, str))
    lowered = manuscript.lower()
    sensitivity_ok = ("sensitiv" in lowered) or ("assumption" in lowered and "identif" in lowered)
    if not sensitivity_ok:
        findings.append({"check": "assumption_sensitivity", "blocking": True,
                         "details": "manuscript lacks an assumption/sensitivity discussion for the causal claim"})

    # Verb governor: associational level must not carry strong causal verbs
    # in Results/Abstract prose (§4.7).
    level = str(causal_claim.get("verb_level") or "associational")
    scoped = "\n".join(
        content for name, content in sections.items()
        if isinstance(content, str) and name.lower() in ("results", "abstract")
    ) or manuscript
    gov_hits = [
        term for term in _ASSOCIATIONAL_BANNED
        if re.search(r"\b" + re.escape(term) + r"\b", scoped, re.IGNORECASE)
    ]
    governor_ok = True
    if level != "causal" and gov_hits:
        governor_ok = False
        findings.append({"check": "verb_governor", "blocking": True,
                         "details": "associational evidence with causal verbs: " + ", ".join(sorted(set(gov_hits)))})

    blocking = [f for f in findings if f.get("blocking")]
    passed = not blocking
    return {"passed": passed, "strategy": strategy, "verb_level": level,
            "sensitivity_ok": sensitivity_ok, "governor_ok": governor_ok,
            "placebo_feasible": bool((causal_claim.get("placebo") or {}).get("feasible", False)),
            "findings": findings}


def verify_comparative(
    plan: Optional[Dict[str, Any]] = None,
    engineer_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Comparative release gate (§7.1, M4): baseline matrix + compulsory ablation.

    Every planned experiment must name ≥2 baselines (§4.5) and every
    completed output must carry a non-empty ablation result (ablation=on).
    """
    from core.evidence_gate import validate_comparative_plan

    plan = plan or {}
    outputs = engineer_outputs or {}
    findings: List[Dict[str, Any]] = []

    experiments = [exp for exp in (plan.get("experiments") or []) if isinstance(exp, dict)]
    for error in validate_comparative_plan(experiments):
        findings.append({"check": "baseline_matrix", "blocking": True, "details": error})

    ablation_ok = True
    for exp in experiments:
        name = exp.get("name") or "unnamed"
        output = outputs.get(name)
        if not isinstance(output, dict) or not output.get("success"):
            continue
        ablation = output.get("ablation")
        if not isinstance(ablation, dict) or not ablation:
            ablation_ok = False
            findings.append({"check": "ablation_compulsory", "blocking": True,
                             "details": f"{name} missing compulsory ablation result"})

    blocking = [f for f in findings if f.get("blocking")]
    passed = not blocking
    return {"passed": passed, "baseline_ok": not any(f["check"] == "baseline_matrix" for f in blocking),
            "ablation_ok": ablation_ok, "findings": findings}


def consistency_referee(
    sections: Dict[str, str],
    plan: Optional[Dict[str, Any]] = None,
    engineer_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Use one isolated model pass to find contradictions in the full draft."""
    """Use one isolated model pass to find contradictions in the full draft."""
    prompt = f"""
Read this assembled research manuscript as a consistency referee.
Your only task is to identify internal contradictions in numbers, datasets,
methods, outcomes, or contribution framing. Do not rewrite prose and do not
infer missing evidence. Return JSON only:
{{"findings": [{{"category": "numeric|dataset|method|outcome|contribution", "message": "...", "blocking": true}}]}}

Plan and structured evidence:
{json.dumps({"plan": plan or {}, "engineer_outputs": engineer_outputs or {}}, sort_keys=True, default=str)[:12000]}

Assembled sections:
{json.dumps(sections, sort_keys=True, default=str)[:30000]}
"""
    try:
        parsed = call_llm_json(
            prompt,
            temperature=0.0,
            tier="judge",
            max_tokens=2500,
            attempts=2,
            call_fn=call_llm,
        )
    except Exception as exc:
        # R10 (M6): referee LLM error → non-blocking warning + retry already
        # spent inside call_llm_json (attempts=2). Never fail-closed on an
        # infra error (cause_class=infra, not science).
        logger.warning(
            "Consistency referee unavailable (%s); continuing without model verdict",
            type(exc).__name__,
        )
        return {"passed": True, "warning": "consistency_referee_unavailable", "findings": [{"category": "referee_error", "message": "Consistency referee failed to execute", "blocking": False, "error_type": type(exc).__name__}]}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("findings"), list):
        # R10 (M6): malformed referee output after retry → non-blocking warning.
        logger.warning("Consistency referee returned malformed output; continuing without model verdict")
        return {"passed": True, "warning": "consistency_referee_malformed", "findings": [{"category": "referee_error", "message": "Consistency referee returned malformed output", "blocking": False}]}
    findings = [item for item in parsed["findings"] if isinstance(item, dict)]
    # R10 (M6): require explicit blocking == True; absent/False is advisory.
    blocking = [item for item in findings if item.get("blocking") is True]
    return {"passed": not blocking, "findings": findings}


def final_manuscript_referee(
    sections: Dict[str, str],
    plan: Optional[Dict[str, Any]] = None,
    engineer_outputs: Optional[Dict[str, Any]] = None,
    topic: Optional[Dict[str, Any]] = None,
    paper_type: str = "experimental",
    thesis: Optional[Dict[str, Any]] = None,
    corpus: Optional[Dict[str, Any]] = None,
    instrument: Optional[Dict[str, Any]] = None,
    survey_analysis: Optional[Dict[str, Any]] = None,
    case_protocol: Optional[Dict[str, Any]] = None,
    causal_claim: Optional[Dict[str, Any]] = None,
    indep_pass: bool = False,
) -> Dict[str, Any]:
    """Run release-blocking checks over the assembled manuscript as one document.

    Per-type applicability (R8 slice): only applicable sub-checks run. Thesis
    types (analytical/argumentative/descriptive/review, §7.1) additionally
    require a verified thesis (triad + 100% bridge coverage) instead of
    experiment reproducibility. Review types (§7.1–§7.2) additionally require
    ``verify_review`` (PRISMA reconciles + extraction ≥ 0.9 + recall ≥ 4/5 +
    synthesis spot-grounded) when a corpus is supplied. Survey types require
    ``verify_survey`` (instrument + nonresponse + no-fishing + synthetic
    screen) when a survey bundle is supplied; case types require
    ``verify_case`` (triangulation + rival) when a protocol is supplied.
    Cause-effect runs require ``verify_causal`` (audit pass +
    assumption/sensitivity section + verb governor) when a causal bundle is
    supplied; comparative runs require ``verify_comparative`` (baseline
    matrix + compulsory ablation). Prohibited text is R9-scoped (Results/Abstract + code fences, Methods
    allowlist) for the manuscript gate; ``validate_empirical_claims`` keeps
    its legacy unscoped check.
    """
    text = "\n".join(sections.values())
    numeric = cross_section_numeric_consistency(sections, engineer_outputs)
    citation = verify_citations(text)
    checklist = validate_reviewer_checklist(sections, engineer_outputs, plan, paper_type=paper_type)
    model_referee = consistency_referee(sections, plan, engineer_outputs)
    novelty = novelty_overlap_check(sections, topic, paper_type=paper_type)
    findings = []
    if not citation["passed"]:
        findings.append({"check": "citations", "details": citation.get("failed", [])})
    if not numeric["passed"]:
        findings.append({"check": "numeric_consistency", "details": numeric["conflicts"]})
    if not checklist["passed"]:
        findings.append({"check": "reviewer_checklist", "details": checklist["failed"]})
    if not model_referee["passed"]:
        findings.append({"check": "consistency_referee", "details": model_referee["findings"]})
    advisory: list = []
    if model_referee.get("warning"):
        # R10 re-audit (M6): the referee-error path returns passed=True so the
        # skipped gate stays dashboard-visible as a non-blocking caveat instead
        # of vanishing (the sub-dict alone never reaches verification_findings).
        # Kept out of `findings` so `passed` (blocking-only) is unaffected.
        advisory.append({"check": "consistency_referee", "blocking": False, "details": model_referee.get("warning")})
    thesis_result: Optional[Dict[str, Any]] = None
    if paper_type in THESIS_GATE_TYPES:
        from agents.thesis_agent import verify_thesis as _verify_thesis

        thesis_result = _verify_thesis(thesis, sections)
        if not thesis_result["passed"]:
            findings.append({"check": "thesis", "details": thesis_result["findings"]})
    review_result: Optional[Dict[str, Any]] = None
    if paper_type == "review" and corpus is not None:
        review_result = verify_review(corpus, sections, paper_type=paper_type)
        if not review_result["passed"]:
            findings.append({"check": "review", "details": review_result["findings"]})
    survey_result: Optional[Dict[str, Any]] = None
    if paper_type == "survey" and (survey_analysis is not None or instrument is not None):
        survey_result = verify_survey(survey_analysis, sections, instrument)
        if not survey_result["passed"]:
            findings.append({"check": "survey", "details": survey_result["findings"]})
    case_result: Optional[Dict[str, Any]] = None
    if paper_type == "case_study" and case_protocol is not None:
        case_result = verify_case(case_protocol, sections)
        if not case_result["passed"]:
            findings.append({"check": "case", "details": case_result["findings"]})
    causal_result: Optional[Dict[str, Any]] = None
    if paper_type == "cause_effect" and causal_claim is not None:
        causal_result = verify_causal(causal_claim, sections, indep_pass=indep_pass)
        if not causal_result["passed"]:
            findings.append({"check": "causal", "details": causal_result["findings"]})
    comparative_result: Optional[Dict[str, Any]] = None
    if paper_type == "comparative":
        comparative_result = verify_comparative(plan, engineer_outputs)
        if not comparative_result["passed"]:
            findings.append({"check": "comparative", "details": comparative_result["findings"]})
    if not novelty["passed"]:
        findings.append({"check": "novelty_overlap", "details": novelty["findings"]})
    # v5: evidence + bootstrap-claim gates (f537f638: copy-exact numbers passed
    # while science was vacuous — these check the artifacts, not the prose).
    for item in validate_experiment_evidence(engineer_outputs, plan):
        findings.append({"check": item.get("check", "evidence"), "details": item.get("details")})
    for item in validate_bootstrap_claims(sections, engineer_outputs):
        findings.append({"check": item.get("check", "bootstrap_claim"), "details": item.get("details")})
    for item in validate_gap_prevalence_claims(sections, topic, plan, corpus):
        findings.append({"check": item.get("check", "gap_claim"), "details": item.get("details")})
    for name, content in (sections or {}).items():
        text_c = content if isinstance(content, str) else ""
        for hit in filler_prose_hits(text_c):
            findings.append({"check": "template_filler", "details": f"{name}: '{hit[:60]}...'"})
            break
        if "DRAFT_FAILED" in text_c:
            findings.append({"check": "draft_failed", "details": f"{name}: draft generation failed, blocking release"})
    prohibited = _scoped_prohibited_hits(sections)
    if prohibited:
        findings.append({"check": "harness_diagnostics", "details": prohibited})
    if not any(name.lower() == "limitations" and content.strip() for name, content in sections.items()):
        findings.append({"check": "limitations", "details": "A non-empty Limitations section is required"})
    quantitative = re.search(r"\d+(?:\.\d+)?\s*%|\b(?:accuracy|f1|mse|p\s*[<=>])\b", text, re.I)
    if quantitative and not re.search(r"\b(?:n\s*=|std|confidence interval|ci\b)", text, re.I):
        findings.append({"check": "uncertainty_reporting", "details": "Quantitative claims require n, standard deviation, or confidence interval"})
    declared_datasets = {
        str((experiment.get("dataset") or {}).get("name"))
        for experiment in (plan or {}).get("experiments", [])
        if isinstance(experiment, dict) and (experiment.get("dataset") or {}).get("name")
    }
    if declared_datasets:
        missing = [dataset for dataset in declared_datasets if dataset.lower() not in text.lower()]
        if missing:
            findings.append({"check": "dataset_consistency", "details": missing})
    return {
        "passed": not findings,
        "findings": findings + advisory,
        "citation": citation,
        "numeric": numeric,
        "checklist": checklist,
        "consistency_referee": model_referee,
        "novelty": novelty,
        "thesis": thesis_result,
        "review": review_result,
        "survey": survey_result,
        "case": case_result,
        "causal": causal_result,
        "comparative": comparative_result,
    }


# ---------------------------------------------------------------------------
# Statistical validity (deterministic re-derivation from raw results)
# ---------------------------------------------------------------------------

def _welch_t_p(a: List[float], b: List[float]) -> Optional[float]:
    """Two-sided Welch t-test p-value without scipy dependency if needed."""
    try:
        from scipy import stats

        if len(a) < 2 or len(b) < 2:
            return None
        _, p = stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    except Exception:
        # Manual approximation
        import numpy as np

        if len(a) < 2 or len(b) < 2:
            return None
        a, b = np.array(a, float), np.array(b, float)
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        se = math.sqrt(va / na + vb / nb)
        if se == 0:
            return 1.0
        t = (ma - mb) / se
        # crude two-sided normal approx
        from math import erf, sqrt

        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        return float(p)


def verify_statistics(
    reported: Dict[str, Any],
    raw_results_path: Optional[str] = None,
    raw_data: Optional[Dict[str, Any]] = None,
    rtol: float = 0.05,
) -> Dict[str, Any]:
    """
    Re-derive mean/std (and optional p-values) from stored raw results
    and compare to what the paper/writer reported.
    """
    raw = raw_data
    if raw is None and raw_results_path:
        path = Path(raw_results_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

    if not raw:
        return {
            "passed": False,
            "score": 0.0,
            "note": "No raw results available for statistical verification",
            "mismatches": ["missing_raw_results"],
        }

    aggregate = raw.get("aggregate_metrics") or {}
    mismatches = []
    checks = 0
    passes = 0

    reported_metrics = reported.get("metrics") or reported
    for key, agg in aggregate.items():
        if key not in reported_metrics:
            continue
        checks += 1
        rep_val = reported_metrics[key]
        if isinstance(rep_val, dict):
            rep_mean = rep_val.get("mean", rep_val.get("value"))
            rep_std = rep_val.get("std")
        else:
            rep_mean = rep_val
            rep_std = None

        true_mean = agg.get("mean")
        true_std = agg.get("std")

        if true_mean is not None and rep_mean is not None:
            if abs(float(rep_mean) - float(true_mean)) > max(rtol * abs(true_mean), 1e-3):
                mismatches.append(
                    f"{key}.mean reported={rep_mean} raw={true_mean}"
                )
            else:
                passes += 1

        if rep_std is not None and true_std is not None:
            checks += 1
            if abs(float(rep_std) - float(true_std)) > max(rtol * abs(true_std), 1e-3):
                mismatches.append(f"{key}.std reported={rep_std} raw={true_std}")
            else:
                passes += 1

    # Optional p-value check if both groups present.
    # L18: relative tolerance + an explicit significance-flip check — a
    # recomputed p that crosses the 0.05 line contradicts the reported
    # conclusion even when the absolute gap looks small.
    if "group_a" in raw and "group_b" in raw and "p_value" in reported_metrics:
        checks += 1
        p_true = _welch_t_p(raw["group_a"], raw["group_b"])
        p_rep = float(reported_metrics["p_value"])
        if p_true is None:
            mismatches.append("could_not_compute_p_value")
        else:
            rel_tol = max(rtol * abs(p_true), 1e-6)
            if abs(p_true - p_rep) > rel_tol:
                mismatches.append(f"p_value reported={p_rep} recomputed={p_true}")
            if (p_rep < 0.05) != (p_true < 0.05):
                mismatches.append(
                    f"significance_flip reported_p={p_rep} recomputed_p={p_true} (alpha=0.05)"
                )
            if not any(m.startswith(("p_value", "significance_flip")) for m in mismatches):
                passes += 1

    if checks == 0:
        return {
            "passed": True,
            "score": 7.0,
            "note": "No overlapping metrics to verify against raw data",
            "mismatches": [],
        }

    score = 10.0 * (passes / checks) if checks else 0.0
    return {
        "passed": len(mismatches) == 0,
        "score": score,
        "note": f"Verified {passes}/{checks} reported statistics against raw data",
        "mismatches": mismatches,
    }


def hard_verify_section(
    content: str,
    engineer_outputs: Optional[Dict[str, Any]] = None,
    *,
    section_name: Optional[str] = None,
    content_requirements: Optional[str] = None,
    min_chars: Optional[int] = None,
    quarantine_singles: bool = False,
) -> Dict[str, Any]:
    """Run all hard checks; LLM soft review should only run AFTER this passes."""
    from .utils import is_degenerate_llm_output, strip_markdown_headers

    section_minima = {
        "abstract": 120,
        "introduction": 400,
        "related work": 400,
        "methods": 300,
        "experiments": 250,
        "results": 200,
        "discussion": 250,
        "conclusion": 150,
        "limitations": 120,
    }
    required_chars = min_chars
    if required_chars is None:
        required_chars = section_minima.get((section_name or "").lower(), 200)

    body = strip_markdown_headers(content or "")
    substance_errors: List[str] = []
    if is_degenerate_llm_output(content or "", min_chars=required_chars):
        substance_errors.append(
            f"section content is empty, a safety stub, or below minimum length ({required_chars} chars)"
        )
    elif len(body) < required_chars:
        substance_errors.append(
            f"section body length {len(body)} is below minimum {required_chars}"
        )
    # v5: template filler + failed drafts must fail, not ship.
    for hit in filler_prose_hits(content or ""):
        substance_errors.append(f"template filler detected: '{hit[:60]}...'")
    if "DRAFT_FAILED" in (content or ""):
        substance_errors.append("draft generation failed (DRAFT_FAILED marker); blocking release pending repair")
    # v5: incomplete truncation (ends mid-sentence on long sections).
    stripped = (body or "").strip()
    if len(stripped) >= 500 and stripped[-1] not in ".!?:;»" and not stripped.endswith("```"):
        substance_errors.append("section ends abruptly mid-sentence (truncation, not complete prose)")
    # v5: Methods must name its protocol — experimental runs with outputs
    # only. Review/field protocol sections (PRISMA/instrument, no engineer
    # outputs) carry a different protocol and must not trip this gate.
    if (section_name or "").lower() in ("methods", "method", "experiments") and engineer_outputs:
        lowered_body = (content or "").lower()
        is_review_protocol = "prisma" in lowered_body or "inclusion" in lowered_body
        if not is_review_protocol:
            gaps = methods_completeness_gaps(content or "")
            if len(gaps) >= 5:
                substance_errors.append("methods incomplete, missing protocol: " + ", ".join(gaps))
    if content_requirements:
        req_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z]{4,}", str(content_requirements))
            if token.lower() not in {"this", "that", "with", "from", "section", "should", "include"}
        ][:8]
        lowered = body.lower()
        missing = [token for token in req_tokens if token not in lowered]
        # Only fail when almost none of the requirement tokens appear — avoid brittle exact matching.
        if req_tokens and len(missing) >= max(3, int(0.75 * len(req_tokens))):
            substance_errors.append(
                "section does not address content_requirements (missing key terms: "
                + ", ".join(missing[:5])
                + ")"
            )

    citation = verify_citations(content, quarantine_singles=quarantine_singles)

    stats_results = []
    if engineer_outputs:
        for exp_name, output in engineer_outputs.items():
            raw_path = None
            if isinstance(output, dict):
                raw_path = output.get("raw_results_path")
                reported = output.get("results", {}).get("metrics") or output.get(
                    "aggregate_metrics", {}
                )
                # Also try extracting claimed numbers from text for this experiment
                stats_results.append(
                    {
                        "experiment": exp_name,
                        **verify_statistics(
                            {"metrics": reported} if not isinstance(reported, dict) or "metrics" not in reported else reported,
                            raw_results_path=raw_path,
                            raw_data=output if "aggregate_metrics" in output else output.get("multi_seed"),
                        ),
                    }
                )

    stats_passed = all(s.get("passed", True) for s in stats_results) if stats_results else True
    avg_stats = (
        sum(s.get("score", 10) for s in stats_results) / len(stats_results)
        if stats_results
        else 10.0
    )

    substance_passed = not substance_errors
    hard_passed = citation["passed"] and stats_passed and substance_passed
    # Hard checks dominate: fail hard → cap score
    combined = min(citation["score"], avg_stats) if hard_passed else min(
        citation["score"], avg_stats, 4.0
    )
    if not substance_passed:
        combined = min(combined, 3.0)

    feedback = _format_hard_feedback(citation, stats_results)
    if substance_errors:
        feedback = "SUBSTANCE: " + "; ".join(substance_errors) + "\n" + feedback

    # Machine repair hints (agentic-harness upgrade): each failing gate emits
    # {section, check, expected, detail} so workflow_nodes.section_revision_feedback
    # can route precise fixes instead of re-rolling blind. Additive-only.
    repair_hints: List[Dict[str, Any]] = []
    for err in substance_errors:
        repair_hints.append({
            "section": section_name or "",
            "check": "substance",
            "expected": "complete, non-template prose meeting the section minimum",
            "detail": str(err)[:300],
        })
    for failed in (citation.get("failed") or []):
        repair_hints.append({
            "section": section_name or "",
            "check": "citation",
            "expected": "remove or replace the unresolvable DOI/arXiv id",
            "detail": str(failed.get("doi") or failed.get("arxiv_id") or failed.get("error"))[:200],
        })
    for result in stats_results:
        if not result.get("passed", True):
            repair_hints.append({
                "section": section_name or "",
                "check": "statistics",
                "expected": "copy metrics exactly from the experiment source of truth",
                "detail": str(result.get("experiment") or "")[:120],
            })

    return {
        "passed": hard_passed,
        "score": combined,
        "citation": citation,
        "statistics": stats_results,
        "substance_errors": substance_errors,
        "feedback": feedback,
        "repair_hints": repair_hints,
    }


def _thesis_evidence_for_dossier(
    plan: Optional[Dict[str, Any]],
    thesis: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collect thesis triad evidence for dossier conditioning (R7).

    Accepts the full thesis dict (claim/bridges/steelman/concession) or the
    §4.5 light-plan shape (``thesis_slot`` claim string + ``claims`` bridge
    statements) synthesized by ``_non_empirical_plan``.
    """
    triad = thesis if isinstance(thesis, dict) else None
    if not isinstance(triad, dict) or not str(triad.get("claim") or "").strip():
        embedded = (plan or {}).get("thesis") if isinstance(plan, dict) else None
        if isinstance(embedded, dict) and str(embedded.get("claim") or "").strip():
            triad = embedded
    claim = str((triad or {}).get("claim") or "")
    if not claim.strip() and isinstance(plan, dict):
        claim = str(plan.get("thesis_slot") or "")
    bridges: List[str] = []
    if isinstance(triad, dict):
        for bridge in triad.get("bridges") or []:
            if isinstance(bridge, dict) and str(bridge.get("statement") or "").strip():
                bridges.append(str(bridge["statement"]))
            elif isinstance(bridge, str) and bridge.strip():
                bridges.append(bridge)
    if not bridges and isinstance(plan, dict):
        for item in plan.get("claims") or []:
            if isinstance(item, str) and item.strip():
                bridges.append(item)
    return {"claim": claim, "bridges": bridges}


def _survey_evidence_for_dossier(
    plan: Optional[Dict[str, Any]],
    survey_analysis: Optional[Dict[str, Any]],
    instrument: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collect survey coverage/nonresponse evidence for dossier conditioning."""
    bundle = dict(survey_analysis or {})
    if not bundle and isinstance(plan, dict):
        nested = plan.get("survey_analysis")
        if isinstance(nested, dict):
            bundle = dict(nested)
    instr = dict(instrument or bundle.get("instrument") or {})
    if not instr and isinstance(plan, dict):
        nested = plan.get("instrument")
        if isinstance(nested, dict):
            instr = dict(nested)
    return {"analysis": bundle, "instrument": instr}


def _coverage_tables_ok(analysis: Dict[str, Any], instrument: Dict[str, Any]) -> bool:
    """Coverage-table proof (R7, fail-closed): recomputed descriptives only.

    Accepts an explicit recomputation marker or a non-empty descriptives /
    columns mapping as produced by ``SurveyAgent.run_analysis``. A bare
    truthy bundle is not proof (same rule as ``release._tables_evidence_ok``).
    """
    for carrier in (analysis, instrument):
        if not isinstance(carrier, dict):
            continue
        for key in ("tables_recomputed", "tables_ok", "descriptives_ok", "recomputed", "verified"):
            if carrier.get(key) is True:
                return True
        desc = carrier.get("descriptives")
        if isinstance(desc, dict):
            cols = desc.get("columns")
            if isinstance(cols, dict) and len(cols):
                return True
        cols = carrier.get("columns")
        if isinstance(cols, dict) and len(cols):
            return True
    return False


def _nonresponse_reported_ok(analysis: Dict[str, Any], instrument: Dict[str, Any]) -> bool:
    """Nonresponse proof mirrors ``verify_survey`` (item rates + unit rate)."""
    nonresponse = None
    if isinstance(analysis.get("nonresponse"), dict):
        nonresponse = analysis["nonresponse"]
    elif isinstance(instrument.get("nonresponse"), dict):
        nonresponse = instrument["nonresponse"]
    return bool(
        isinstance(nonresponse, dict)
        and isinstance(nonresponse.get("item_rates"), dict)
        and bool(nonresponse.get("item_rates"))
        and isinstance(nonresponse.get("unit_rate"), (int, float))
    )


def reproducibility_dossier(
    plan: Optional[Dict[str, Any]],
    engineer_outputs: Optional[Dict[str, Any]],
    paper_type: Optional[str] = None,
    thesis: Optional[Dict[str, Any]] = None,
    survey_analysis: Optional[Dict[str, Any]] = None,
    case_protocol: Optional[Dict[str, Any]] = None,
    instrument: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Venue-style disclosure checklist derived from executable artifacts, not prose.

    R7 (§6.1): type-conditioned checklists. ``paper_type`` selects the
    checklist; ``None``/unknown types keep the current strict experimental
    dossier (legacy default — existing callers unchanged):

    - experimental family (experimental/comparative/empirical/cause_effect):
      the current full dossier (predictions, baselines, tests, raw results,
      code, provenance, limitations).
    - thesis types (analytical/argumentative/descriptive/review): the
      proofs/argument artifacts (thesis triad + bridge statements) satisfy
      the dossier instead of experiment artifacts.
    - survey: coverage tables + nonresponse reporting satisfy the dossier.
    - case/case_study: triangulation + rival check satisfy the dossier.

    Non-experimental branches are fail-closed when their evidence is absent
    (same verdict as today — no fabrication, no silent pass).
    """
    ptype = str(paper_type or "").strip().lower() or ""
    if ptype == "case":
        ptype = "case_study"
    if ptype in ("analytical", "argumentative", "descriptive", "review"):
        evidence = _thesis_evidence_for_dossier(plan, thesis)
        checks = {
            "thesis_triad": bool(evidence["claim"].strip()),
            "bridge_statements": bool(evidence["bridges"]),
            "limitations_disclosed": bool(plan),
        }
        return {
            "checks": checks,
            "passed": all(checks.values()),
            "score": round(10 * sum(checks.values()) / len(checks), 2),
            "paper_type": ptype or "thesis",
            "mode": "thesis",
        }
    if ptype == "survey":
        bundle = _survey_evidence_for_dossier(plan, survey_analysis, instrument)
        checks = {
            "coverage_tables": _coverage_tables_ok(bundle["analysis"], bundle["instrument"]),
            "nonresponse_reported": _nonresponse_reported_ok(bundle["analysis"], bundle["instrument"]),
            "limitations_disclosed": bool(plan),
        }
        return {
            "checks": checks,
            "passed": all(checks.values()),
            "score": round(10 * sum(checks.values()) / len(checks), 2),
            "paper_type": ptype,
            "mode": "survey",
        }
    if ptype == "case_study":
        protocol = dict(case_protocol or {})
        if not protocol and isinstance(plan, dict):
            for key in ("case_protocol", "protocol"):
                nested = plan.get(key)
                if isinstance(nested, dict):
                    protocol = dict(nested)
                    break
        triang = protocol.get("triangulation") if isinstance(protocol.get("triangulation"), dict) else {}
        flags = list(triang.get("single_source_flags") or [])
        limitations = list(protocol.get("limitations") or [])
        rival = protocol.get("rival") if isinstance(protocol.get("rival"), dict) else {}
        checks = {
            "triangulation_ok": bool(protocol) and not (flags and not limitations),
            "rival_ok": bool(rival.get("passed", False)) if rival else False,
            "limitations_disclosed": bool(protocol),
        }
        return {
            "checks": checks,
            "passed": all(checks.values()),
            "score": round(10 * sum(checks.values()) / len(checks), 2),
            "paper_type": ptype,
            "mode": "case",
        }
    experiments = (plan or {}).get("experiments") or []
    outputs = engineer_outputs or {}
    has_plan = bool(plan) and bool(experiments)
    has_outputs = bool(outputs)
    checks = {
        "falsifiable_predictions": all(isinstance(e, dict) and bool(e.get("falsifiable_prediction")) for e in experiments) if experiments else False,
        "named_baselines": all(isinstance(e, dict) and bool(e.get("baselines") or e.get("baseline_comparison")) for e in experiments) if experiments else False,
        "statistical_tests": all(isinstance(e, dict) and bool(e.get("statistical_test")) for e in experiments) if experiments else False,
        "multi_seed_raw_results": bool(outputs) and all(isinstance(o, dict) and bool(o.get("raw_results_path") or o.get("multi_seed")) and (not o.get("raw_results_path") or Path(str(o["raw_results_path"])).is_file()) for o in outputs.values()),
        # v5: fallback_synthetic is exploratory plumbing — never reproducible evidence.
        "real_execution": bool(outputs) and all(isinstance(o, dict) and o.get("success", True) is True and str(o.get("outcome") or "") != "fallback_synthetic" for o in outputs.values()),
        "executable_code": bool(outputs) and all(isinstance(o, dict) and bool(o.get("code")) and o.get("success", True) and str(o.get("outcome") or "") != "fallback_synthetic" for o in outputs.values()),
        # Never pass by default when there is nothing to evaluate.
        "contract_provenance": has_outputs and all(
            isinstance(o, dict) and bool(o.get("contract_hash")) for o in outputs.values()
        ),
        "limitations_disclosed": has_plan and has_outputs,
    }
    return {"checks": checks, "passed": all(checks.values()), "score": round(10 * sum(checks.values()) / len(checks), 2)}


PROHIBITED_MANUSCRIPT_TEXT = (
    "tracemalloc",
    "sandbox blocked",
    "import blocked",
    "syntaxerror",
    "traceback",
    "object has no attribute",
    "experiment_failed",
    "plan_revision_requested",
    # M3 (R9): fake-field phrases — fabricating respondents, patients, or
    # interviews is a release-blocker (proposal §1.6). Scoped per R9: the
    # release referee checks Results/Abstract + code-fence context with a
    # Methods allowlist (see _scoped_prohibited_hits); validate_empirical_claims
    # keeps its legacy unscoped check so old callers stay green.
    "synthetic respondents",
    "fabricated respondents",
    "simulated respondents",
    "mock respondents",
    "fabricated interviews",
    "simulated interviews",
    "mock interviews",
    "synthetic interviews",
    "fabricated patients",
    "simulated patients",
)

# Subset of fake-field phrases (M3/R9): fabrication claims about field data.
FAKE_FIELD_PHRASES = tuple(
    phrase for phrase in PROHIBITED_MANUSCRIPT_TEXT
    if any(key in phrase for key in ("respondent", "interview", "patient"))
)


def _code_fence_spans(text: str) -> List[Tuple[int, int]]:
    """Spans of fenced code blocks (```...``` and `...`) in one text."""
    spans: List[Tuple[int, int]] = []
    for match in re.finditer(r"```.*?```|`[^`\n]+`", text or "", re.DOTALL):
        spans.append((match.start(), match.end()))
    return spans


def _scoped_prohibited_hits(sections: Dict[str, str]) -> List[str]:
    """R9-scoped prohibited-text hits (M3): Results/Abstract + code fences.

    Only ``Results``/``Abstract`` prose (any section whose name contains
    ``result`` or ``abstract``) and fenced code spans anywhere are scanned;
    ``Methods``/``Method`` sections carry an allowlist (never scanned), so
    instrument descriptions that name a fabrication guard do not trip the
    gate. Matching stays case-insensitive substring, like the legacy check.
    """
    hits: List[str] = []
    sections = sections or {}
    for name, content in sections.items():
        lowered_name = str(name or "").lower()
        if "method" in lowered_name:
            continue
        text = content if isinstance(content, str) else ""
        if "result" in lowered_name or "abstract" in lowered_name:
            lowered = text.lower()
            hits.extend(p for p in PROHIBITED_MANUSCRIPT_TEXT if p in lowered)
        else:
            for start, end in _code_fence_spans(text):
                lowered = text[start:end].lower()
                hits.extend(p for p in PROHIBITED_MANUSCRIPT_TEXT if p in lowered)
    # Deduplicate, preserving order.
    seen = set()
    unique: List[str] = []
    for hit in hits:
        if hit not in seen:
            seen.add(hit)
            unique.append(hit)
    return unique


def validate_empirical_claims(content: str, engineer_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Reject leaked harness diagnostics from empirical manuscript sections."""
    lowered = content.lower()
    prohibited = [phrase for phrase in PROHIBITED_MANUSCRIPT_TEXT if phrase in lowered]
    successful = [
        name for name, output in (engineer_outputs or {}).items()
        if isinstance(output, dict) and output.get("success")
        or isinstance(output, dict) and output.get("aggregate_metrics")
    ]
    return {
        "passed": not prohibited and bool(successful),
        "prohibited_text": prohibited,
        "successful_experiments": successful,
        "note": "Empirical text is grounded in completed experiment outputs" if successful and not prohibited else "Empirical text contains unsupported harness text or has no completed experiment",
    }


def _format_hard_feedback(citation: Dict, stats_results: List[Dict]) -> str:
    parts = [f"Citations: {citation.get('note')}"]
    if citation.get("quarantined"):
        parts.append(
            "Quarantined to repair pass: "
            + ", ".join(
                str(f.get("doi") or f.get("arxiv_id")) for f in citation["quarantined"]
            )
        )
    if citation.get("failed"):
        parts.append(
            "Unresolved: "
            + ", ".join(
                str(f.get("doi") or f.get("arxiv_id")) for f in citation["failed"]
            )
        )
    for s in stats_results:
        parts.append(f"Stats[{s.get('experiment')}]: {s.get('note')}")
        if s.get("mismatches"):
            parts.append("Mismatches: " + "; ".join(s["mismatches"][:5]))
    return "\n".join(parts)
