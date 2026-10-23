"""Method × Domain sparsity matrix for gap detection."""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, List

from core.llm import call_llm
from core.utils import parse_json_from_llm

logger = logging.getLogger(__name__)


def find_sparse_cells(
    papers: List[Dict[str, Any]],
    min_method_freq: int = 3,
    min_domain_freq: int = 3,
    max_results: int = 5,
    max_papers: int = 60,
    abstract_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """Extract (method, application_domain) pairs from paper titles/abstracts,
    build an in-memory count matrix, and find sparse cells.

    A "sparse cell" = a well-established method (freq >= 3) paired with a
    well-established domain (freq >= 3) but the specific pair appears 0 or 1
    times.

    v5: wider windows (60 papers × 1200 chars) over the full corpus instead
    of the old 30 × 400 keyhole. Fail-open: LLM extraction failure or
    <5 papers → return [].
    """
    if not papers or len(papers) < 5:
        return []

    # Build a batch prompt for all papers
    paper_texts = []
    for p in papers[:max(5, max_papers)]:
        title = p.get("title", "")[:150]
        abstract = p.get("abstract", "")[:max(200, abstract_chars)]
        if title or abstract:
            paper_texts.append(f"Title: {title}\nAbstract: {abstract}")

    if not paper_texts:
        return []

    prompt = f"""Extract (method, application_domain) pairs actually described in these titles/abstracts.
A "method" is a specific technique, algorithm, or approach (e.g. "transformer", "random forest", "diffusion model").
An "application_domain" is the problem area or setting (e.g. "medical imaging", "speech recognition", "protein folding").

Return JSON: {{"pairs": [{{"method": "...", "domain": "..."}}]}}

Papers:
{json.dumps(paper_texts[:max_papers], indent=2)[:16000]}"""

    try:
        raw = call_llm(prompt, temperature=0.3, tier="cheap")
        parsed = parse_json_from_llm(raw) or {}
        pairs = parsed.get("pairs") or []
        pairs = [p for p in pairs if isinstance(p, dict) and p.get("method") and p.get("domain")]
    except Exception as e:
        logger.debug("find_sparse_cells: LLM extraction failed: %s", e)
        return []

    if len(pairs) < 5:
        return []

    # Build count matrix
    method_counts: Counter = Counter()
    domain_counts: Counter = Counter()
    pair_counts: Counter = Counter()

    for pair in pairs:
        method = pair["method"].strip().lower()
        domain = pair["domain"].strip().lower()
        method_counts[method] += 1
        domain_counts[domain] += 1
        pair_counts[(method, domain)] += 1

    # Find sparse cells: well-established method × well-established domain
    # but the specific pair is rare
    sparse = []
    for (method, domain), pair_freq in pair_counts.items():
        if pair_freq > 1:
            continue
        method_freq = method_counts.get(method, 0)
        domain_freq = domain_counts.get(domain, 0)
        if method_freq >= min_method_freq and domain_freq >= min_domain_freq:
            sparse.append({
                "method": method,
                "domain": domain,
                "method_frequency": method_freq,
                "domain_frequency": domain_freq,
                "pair_frequency": pair_freq,
            })

    # Sort by combined frequency (most established combos first)
    sparse.sort(key=lambda x: x["method_frequency"] + x["domain_frequency"], reverse=True)
    return sparse[:max_results]
