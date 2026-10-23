"""Contradiction mining from paper abstracts."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.llm import call_llm
from core.utils import parse_json_from_llm

logger = logging.getLogger(__name__)


def find_contradictions(
    papers: List[Dict[str, Any]],
    max_papers: int = 40,
    abstract_chars: int = 1500,
) -> List[Dict[str, Any]]:
    """Identify pairs of papers that make opposing empirical claims about the
    same method/metric/dataset.

    v5: wider windows (40 papers × 1500 chars) so genuine disagreements in a
    200-paper corpus are actually visible; chunked into two LLM passes when
    the corpus exceeds one prompt. Fail-open: any failure → return [].
    """
    if not papers or len(papers) < 2:
        return []

    # Cap prompt size
    candidates = papers[:max(2, max_papers)]

    def _block(idxs):
        texts = []
        for i in idxs:
            p = candidates[i]
            title = p.get("title", "")[:150]
            abstract = p.get("abstract", "")[:max(200, abstract_chars)]
            if title or abstract:
                texts.append(f"[{i}] Title: {title}\nAbstract: {abstract}")
        return texts

    # e0ec9e91: a single [:8000] dump showed the LLM only ~5 abstracts of a
    # 40-paper window, so contradictions read 0-1 every seed. Chunk into two
    # passes (first half / second half, each capped) and merge.
    n = len(candidates)
    chunks = [list(range(n))] if n <= 20 else [list(range(0, n // 2)), list(range(n // 2, n))]
    contradictions = []
    for chunk in chunks:
        abstracts_text = _block(chunk)
        if len(abstracts_text) < 2:
            continue
        prompt = f"""Identify pairs of papers among these abstracts that make opposing empirical claims
about the same method, metric, or dataset (e.g. one says X improves Y, another says X does
not improve Y or hurts it). Focus on genuine empirical disagreements, not just different topics.
Quote paper titles EXACTLY as given (including the [index] is not needed, but the title text must match verbatim).

Return JSON:
{{
  "contradictions": [
    {{
      "paper_a_title": "...",
      "paper_b_title": "...",
      "claim_a": "...",
      "claim_b": "...",
      "shared_subject": "..."
    }}
  ]
}}

Abstracts:
{json.dumps(abstracts_text, indent=2)[:16000]}"""
        try:
            raw = call_llm(prompt, temperature=0.3, tier="cheap")
            parsed = parse_json_from_llm(raw) or {}
            contradictions.extend(parsed.get("contradictions") or [])
        except Exception as e:
            logger.debug("find_contradictions: LLM call failed: %s", e)
            continue

    # Validate both titles actually match input papers (drop hallucinated pairs).
    # e0ec9e91: exact-match validation dropped every paraphrased pair.
    # Fuzzy token overlap >=0.60 against a real input title counts; the stored
    # titles are normalized to the canonical input titles.
    try:
        from core.utils import title_token_overlap as _overlap
    except Exception:
        _overlap = None
    input_titles = [(p.get("title", "") or "") for p in candidates if p.get("title")]
    lowered = {t.strip().lower(): t for t in input_titles}
    validated = []
    for c in contradictions:
        if not isinstance(c, dict):
            continue
        pa = (c.get("paper_a_title") or "").strip().lower()
        pb = (c.get("paper_b_title") or "").strip().lower()
        if pa in lowered and pb in lowered and pa != pb:
            validated.append({
                "paper_a_title": lowered[pa],
                "paper_b_title": lowered[pb],
                "claim_a": str(c.get("claim_a", "")),
                "claim_b": str(c.get("claim_b", "")),
                "shared_subject": str(c.get("shared_subject", "")),
            })
            continue
        if _overlap and pa and pb and pa != pb:
            try:
                ma = max(input_titles, key=lambda t: _overlap(pa, t.strip().lower()))
                mb = max(input_titles, key=lambda t: _overlap(pb, t.strip().lower()))
                if _overlap(pa, ma.strip().lower()) >= 0.60 and _overlap(pb, mb.strip().lower()) >= 0.60 and ma != mb:
                    validated.append({
                        "paper_a_title": ma,
                        "paper_b_title": mb,
                        "claim_a": str(c.get("claim_a", "")),
                        "claim_b": str(c.get("claim_b", "")),
                        "shared_subject": str(c.get("shared_subject", "")),
                    })
            except Exception:
                pass
    return validated
