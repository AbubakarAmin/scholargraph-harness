"""Deterministic, no-LLM literature signals for topic discovery (v5).

Everything here is pure computation over the already-retrieved paper rows —
zero network, zero LLM cost, fail-open ([] on any surprise). These complement
the LLM miners (contradictions, sparsity) with signals a good researcher
checks that the old pipeline never computed:

- temporal momentum: which keyword themes are heating up (recent vs older)?
- citation concentration: highly-cited but aging papers = foundational work
  with few recent extensions (OpenAlex-native replacement for the S2 citation
  graph when Semantic Scholar is disabled).
- concept co-occurrence gaps: OpenAlex concept pairs that are each frequent
  but never appear together in one paper.
- reviewer-weakness clustering: group OpenReview expert weaknesses by theme
  and count votes, so repeated pain outranks one-off remarks.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z][a-z0-9_-]{2,}")
_STOP = frozenset({
    "and", "for", "the", "are", "but", "not", "with", "this", "that", "from",
    "have", "has", "was", "were", "been", "being", "can", "may", "its", "our",
    "using", "used", "use", "via", "novel", "new", "based", "approach",
    "method", "model", "models", "results", "paper", "study", "learning",
    "neural", "deep",
})


def _tokens(text: str) -> List[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP]


def temporal_momentum(
    papers: List[Dict[str, Any]],
    top_k: int = 5,
    recent_years: int = 2,
) -> List[Dict[str, Any]]:
    """Keyword themes whose recent share dominates their older share.

    momentum = recent_count / max(older_count, 1), reported with both counts
    so consumers can tell "3 vs 0" apart from "30 vs 5".
    """
    try:
        if not papers or len(papers) < 5:
            return []
        current_year = datetime.now().year
        cutoff = current_year - recent_years
        recent: Counter = Counter()
        older: Counter = Counter()
        for p in papers:
            if not isinstance(p, dict):
                continue
            year = p.get("year") or p.get("publication_year") or 0
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = 0
            toks = set(_tokens(f"{p.get('title', '')} {p.get('abstract', '')}"))
            bucket = recent if year >= cutoff else older
            for t in toks:
                bucket[t] += 1
        scored = []
        for term, r in recent.items():
            if r < 3:
                continue
            o = older.get(term, 0)
            momentum = r / max(o, 1)
            if momentum >= 1.5 and r >= o:
                scored.append({
                    "keyword": term,
                    "recent_count": r,
                    "older_count": o,
                    "momentum": round(momentum, 2),
                })
        scored.sort(key=lambda d: (d["momentum"], d["recent_count"]), reverse=True)
        return scored[:max(1, top_k)]
    except Exception as e:
        logger.debug("temporal_momentum failed: %s", e)
        return []


def citation_concentration(
    papers: List[Dict[str, Any]],
    top_k: int = 5,
    min_citations: int = 20,
    age_years: int = 3,
) -> List[Dict[str, Any]]:
    """Foundational-but-aging papers: high cited_by_count, old, Underextended.

    OpenAlex-native stand-in for the S2 citation-graph gap score
    (high in-degree, low recent extensions) that needs no extra network call.
    """
    try:
        if not papers:
            return []
        current_year = datetime.now().year
        scored = []
        for p in papers:
            if not isinstance(p, dict):
                continue
            cites = p.get("cited_by_count") or 0
            try:
                cites = int(cites)
            except (TypeError, ValueError):
                continue
            year = p.get("year") or p.get("publication_year") or 0
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = 0
            if cites >= min_citations and year and (current_year - year) >= age_years:
                scored.append({
                    "title": (p.get("title") or "")[:150],
                    "doi": p.get("doi"),
                    "arxiv_id": p.get("arxiv_id"),
                    "cited_by_count": cites,
                    "year": year,
                    "concentration": round(cites / max(current_year - year, 1), 2),
                })
        scored.sort(key=lambda d: d["concentration"], reverse=True)
        return scored[:max(1, top_k)]
    except Exception as e:
        logger.debug("citation_concentration failed: %s", e)
        return []


def concept_cooccurrence_gaps(
    papers: List[Dict[str, Any]],
    top_k: int = 5,
    min_concept_freq: int = 3,
) -> List[Dict[str, Any]]:
    """OpenAlex concept pairs that are each frequent but never co-occur.

    Uses the `concepts` field OpenAlex already returns (display_name +
    level). Falls back to keyword pseudo-concepts when rows lack concepts.
    """
    try:
        if not papers or len(papers) < 5:
            return []
        concept_of: List[List[str]] = []
        for p in papers:
            if not isinstance(p, dict):
                continue
            names: List[str] = []
            for c in p.get("concepts") or []:
                name = (c.get("display_name") or "").strip().lower() if isinstance(c, dict) else ""
                if name and len(name) > 2:
                    names.append(name)
            if not names:
                toks = _tokens(f"{p.get('title', '')}")
                names = sorted(set(toks))[:4]
            if names:
                concept_of.append(sorted(set(names)))
        if len(concept_of) < 5:
            return []
        freq: Counter = Counter()
        cooccur: Counter = Counter()
        for names in concept_of:
            for n in names:
                freq[n] += 1
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    cooccur[tuple(sorted((names[i], names[j])))] += 1
        established = {n for n, c in freq.items() if c >= min_concept_freq}
        gaps = []
        est = sorted(established)
        for i in range(len(est)):
            for j in range(i + 1, len(est)):
                pair = tuple(sorted((est[i], est[j])))
                if cooccur.get(pair, 0) == 0:
                    gaps.append({
                        "concept_a": est[i],
                        "concept_b": est[j],
                        "freq_a": freq[est[i]],
                        "freq_b": freq[est[j]],
                        "pair_frequency": 0,
                    })
        gaps.sort(key=lambda d: d["freq_a"] + d["freq_b"], reverse=True)
        return gaps[:max(1, top_k)]
    except Exception as e:
        logger.debug("concept_cooccurrence_gaps failed: %s", e)
        return []


def cluster_reviewer_weaknesses(
    signals: List[Dict[str, Any]],
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """Cluster OpenReview weaknesses by token overlap; repeated pain first.

    Returns theme clusters with vote counts + example paper titles, so the
    gap prompt and ranker can weight expert-stated pain by frequency.
    """
    try:
        items: List[Tuple[str, str]] = []
        for s in signals or []:
            if not isinstance(s, dict):
                continue
            title = (s.get("title") or "")[:150]
            weaknesses = s.get("weaknesses") or []
            if isinstance(weaknesses, str):
                weaknesses = [weaknesses]
            for w in weaknesses:
                text = str(w or "").strip()
                if len(text) >= 20:
                    items.append((title, text))
        if not items:
            return []
        clusters: List[Dict[str, Any]] = []
        for title, text in items:
            toks = set(_tokens(text))
            placed = False
            for c in clusters:
                inter = toks & c["tokens"]
                union = toks | c["tokens"]
                if union and len(inter) / len(union) >= 0.25:
                    c["votes"] += 1
                    c["tokens"] |= toks
                    c["examples"].append({"title": title, "excerpt": text[:240]})
                    placed = True
                    break
            if not placed:
                clusters.append({
                    "tokens": set(toks),
                    "votes": 1,
                    "examples": [{"title": title, "excerpt": text[:240]}],
                })
        out = []
        for c in clusters:
            keywords = sorted(c["tokens"])[:8]
            out.append({
                "theme": " ".join(keywords[:5]),
                "keywords": keywords,
                "votes": c["votes"],
                "examples": c["examples"][:3],
            })
        out.sort(key=lambda d: d["votes"], reverse=True)
        for d in out:
            d.pop("tokens", None)
        return out[:max(1, top_k)]
    except Exception as e:
        logger.debug("cluster_reviewer_weaknesses failed: %s", e)
        return []
