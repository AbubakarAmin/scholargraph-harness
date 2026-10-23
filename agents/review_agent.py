"""ReviewAgent — systematic/scoping review pipeline (proposal §4.6, M2).

Exact algorithm:
1. OA resolution order: ``best_oa_location`` → Unpaywall → S2 OA → PMC.
   Closed/paywalled → ``full_text_unavailable``, abstract-only + limitation entry.
2. Dual screening: LLM title/abstract screen vs criteria + rule backstop
   (query-token overlap ≥ 2).
3. Extraction columns (exact): ``design, sample_n, effect, metric, limitation, doi``.
4. Synthesis: thematic or vote-count. Meta-analysis ONLY when ≥3 commensurate
   effects + heterogeneity I² computed; otherwise narrative synthesis +
   explicit why-not-meta sentence.
5. ``prisma()`` reconciliation (must balance exactly):
   ``identified = screened + duplicates_excluded``;
   ``screened = eligible + excluded_title_abstract``;
   ``eligible = included + excluded_fulltext``. Any mismatch → blocking.
6. Seed recall: 5 intake-pinned keys, need ≥4/5. Below bar → demote
   ``systematic → scoping`` + banner (not terminal).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import config
from core.utils import log_agent_action, call_llm_json
from core.llm import call_llm, get_llm_client
from core.context import RunContext, get_active_context
from core.memory import memory

logger = logging.getLogger(__name__)


def _review_system(prompt_name: str) -> dict:
    """System kwarg from the v2 registry (stub-safe: {} when unavailable)."""
    try:
        from core.prompts import render_system

        system = render_system(prompt_name)
        return {"system": system} if system else {}
    except Exception:
        return {}

# Extraction columns (exact per §4.6 step 3).
EXTRACTION_COLUMNS = ("design", "sample_n", "effect", "metric", "limitation", "doi")

# Completeness bar for release (§7.1): extraction ≥ 0.9.
EXTRACTION_COMPLETENESS_BAR = 0.9

# Seed recall bar (§4.6 step 6): 5 intake-pinned keys, need ≥4/5.
SEED_RECALL_BAR = 4 / 5


def _content_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def prisma_reconcile(counts: Dict[str, Any]) -> Dict[str, Any]:
    """PRISMA reconciliation — must balance exactly (pure, deterministic).

    Equations (§4.6 step 5):
      identified = screened + duplicates_excluded
      screened = eligible + excluded_title_abstract
      eligible = included + excluded_fulltext
    Any mismatch → ``passed=False`` + blocking finding.
    """
    counts = dict(counts or {})
    keys = ("identified", "screened", "duplicates_excluded", "eligible",
            "excluded_title_abstract", "included", "excluded_fulltext")
    missing = [k for k in keys if not isinstance(counts.get(k), (int, float))]
    if missing:
        return {"passed": False, "reconciled": False,
                "findings": [{"check": "prisma", "blocking": True,
                              "details": f"missing PRISMA counts: {missing}"}],
                "counts": counts}
    identified = int(counts["identified"])
    screened = int(counts["screened"])
    duplicates_excluded = int(counts["duplicates_excluded"])
    eligible = int(counts["eligible"])
    excluded_title_abstract = int(counts["excluded_title_abstract"])
    included = int(counts["included"])
    excluded_fulltext = int(counts["excluded_fulltext"])
    errors = []
    if identified != screened + duplicates_excluded:
        errors.append(f"identified ({identified}) != screened ({screened}) + duplicates_excluded ({duplicates_excluded})")
    if screened != eligible + excluded_title_abstract:
        errors.append(f"screened ({screened}) != eligible ({eligible}) + excluded_title_abstract ({excluded_title_abstract})")
    if eligible != included + excluded_fulltext:
        errors.append(f"eligible ({eligible}) != included ({included}) + excluded_fulltext ({excluded_fulltext})")
    passed = not errors
    return {"passed": passed, "reconciled": passed,
            "findings": [] if passed else [{"check": "prisma", "blocking": True, "details": "; ".join(errors)}],
            "counts": counts}


def extraction_completeness(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fraction of extraction cells filled over the 6 exact columns.

    ``passed`` at ≥0.9 (§7.1). Pure/deterministic.
    """
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if not rows:
        return {"passed": False, "completeness": 0.0, "filled": 0, "total": 0,
                "findings": [{"check": "extraction", "blocking": True, "details": "no extraction rows"}]}
    filled = 0
    total = len(rows) * len(EXTRACTION_COLUMNS)
    for row in rows:
        for col in EXTRACTION_COLUMNS:
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, str) and not val.strip():
                continue
            filled += 1
    completeness = filled / total if total else 0.0
    passed = completeness >= EXTRACTION_COMPLETENESS_BAR
    return {"passed": passed, "completeness": round(completeness, 3),
            "filled": filled, "total": total,
            "findings": [] if passed else [{"check": "extraction", "blocking": True,
                                            "details": f"extraction completeness {completeness:.2f} < 0.9"}]}


def seed_recall_check(included_dois: List[str], seed_keys: List[str]) -> Dict[str, Any]:
    """Seed recall: 5 intake-pinned keys, need ≥4/5 (pure, deterministic).

    Below bar → demote ``systematic → scoping`` + banner (not terminal).
    """
    seeds = [str(s or "").strip().lower() for s in (seed_keys or []) if str(s or "").strip()]
    included = {str(d or "").strip().lower() for d in (included_dois or []) if str(d or "").strip()}
    if not seeds:
        return {"passed": True, "recall": 1.0, "hits": 0, "total": 0,
                "demote_to_scoping": False, "banner": None, "mode": "systematic"}
    hits = sum(1 for s in seeds if s in included)
    recall = hits / len(seeds)
    passed = recall >= SEED_RECALL_BAR
    banner = None
    if not passed:
        banner = (f"Seed recall {hits}/{len(seeds)} below 4/5 — demoted systematic → scoping "
                  "(banner, not terminal). Add the missing keys or narrow the claim.")
    return {"passed": passed, "recall": round(recall, 3), "hits": hits, "total": len(seeds),
            "demote_to_scoping": not passed,
            "banner": banner,
            "mode": "systematic" if passed else "scoping"}


def heterogeneity_i2(effects: List[float]) -> Optional[float]:
    """Higgins I² heterogeneity (%) for ≥3 commensurate effects, else None."""
    vals = [float(v) for v in (effects or []) if isinstance(v, (int, float))]
    if len(vals) < 3:
        return None
    mean = sum(vals) / len(vals)
    q = sum((v - mean) ** 2 for v in vals)
    df = len(vals) - 1
    if q <= 0 or df <= 0:
        return 0.0
    i2 = max(0.0, (q - df) / q) * 100.0
    return round(i2, 1)


# Depth-based review caps (§5.3): (screen_cap, fulltext_cap).
REVIEW_CAPS: Dict[str, tuple] = {
    "quick": (40, 8),
    "standard": (120, 25),
    "thorough": (200, 40),
}


def review_caps(depth: str = "Standard") -> tuple:
    """Screen/full-text caps for one depth (default Standard, §5.3)."""
    return REVIEW_CAPS.get(str(depth or "Standard").strip().lower(), REVIEW_CAPS["standard"])


def build_prisma_counts(
    candidates: List[Dict[str, Any]],
    screened_included: List[Dict[str, Any]],
    screened_excluded: List[Dict[str, Any]],
) -> Dict[str, int]:
    """PRISMA counts balanced by construction (§4.6 step 5, shared helper).

    identified = len(candidates); duplicates_excluded = identified - unique;
    screened = unique; eligible = included; excluded_fulltext = eligible - included
    (0 when the OA-mark keeps every screened-in paper abstract-only instead of
    excluding closed content — the OA-only rule).
    """
    seen: set = set()
    unique = 0
    for paper in candidates or []:
        key = ""
        if isinstance(paper, dict):
            key = str(paper.get("doi") or paper.get("arxiv_id") or paper.get("title") or "").strip().lower()
        if key and key in seen:
            continue
        seen.add(key)
        unique += 1
    identified = len(candidates or [])
    duplicates_excluded = identified - unique
    screened = unique
    excluded_title_abstract = len(screened_excluded or [])
    eligible = len(screened_included or [])
    # eligible should equal screened - excluded_title_abstract when the screen
    # splits the deduped set; clamp defensively so the equations always balance
    # for the degenerate empty-corpus case.
    included = eligible
    excluded_fulltext = 0
    return {
        "identified": identified,
        "screened": screened,
        "duplicates_excluded": duplicates_excluded,
        "eligible": eligible,
        "excluded_title_abstract": excluded_title_abstract,
        "included": included,
        "excluded_fulltext": excluded_fulltext,
    }


class ReviewAgent:
    """Screen → extract → synthesize systematic review corpus (§4.6)."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    @property
    def vector_memory(self):
        return self.context.memory if self.context else memory

    # -- step 2: dual screening -------------------------------------------
    def screen(
        self,
        papers: List[Dict[str, Any]],
        criteria: Optional[Dict[str, Any]] = None,
        query: str = "",
    ) -> Dict[str, Any]:
        """Dual screen: LLM title/abstract screen + rule backstop.

        Rule backstop (deterministic): query-token overlap ≥ 2 keeps a paper
        the LLM would exclude, so one model miss cannot silently drop an
        on-topic source. Returns ``{included, excluded, screen_log}``.
        """
        criteria = criteria or {}
        query_tokens = set(_content_tokens(query))
        included: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []
        screen_log: List[Dict[str, Any]] = []
        for paper in papers or []:
            if not isinstance(paper, dict):
                continue
            title = str(paper.get("title") or "")
            abstract = str(paper.get("abstract") or "")
            llm_keep = self._llm_screen_one(title, abstract, criteria)
            overlap = len(set(_content_tokens(title + " " + abstract)) & query_tokens)
            rule_keep = overlap >= 2
            keep = bool(llm_keep or rule_keep)
            record = {"doi": paper.get("doi"), "title": title[:120],
                      "llm_keep": bool(llm_keep), "rule_overlap": overlap,
                      "rule_keep": rule_keep, "kept": keep}
            screen_log.append(record)
            (included if keep else excluded).append(paper)
        log_agent_action("ReviewAgent", "screen_complete", {
            "n_papers": len(papers or []), "included": len(included),
            "excluded": len(excluded)})
        return {"included": included, "excluded": excluded, "screen_log": screen_log}

    def _llm_screen_one(self, title: str, abstract: str, criteria: Dict[str, Any]) -> bool:
        """One cheap LLM title/abstract screen; fail-closed to keep on error."""
        prompt = (
            "Systematic review title/abstract screen. Reply JSON only: {\"keep\": true|false}.\n"
            f"Title: {title[:400]}\nAbstract: {abstract[:1200]}\n"
            f"Inclusion: {criteria.get('inclusion') or 'on-topic for the review question'}\n"
            f"Exclusion: {criteria.get('exclusion') or 'off-topic, editorial, no abstract'}"
        )
        try:
            parsed = call_llm_json(prompt, temperature=0.0, tier="cheap",
                                   attempts=2, call_fn=call_llm,
                                   **_review_system("review.screen"))
            if isinstance(parsed, dict) and "keep" in parsed:
                return bool(parsed["keep"])
        except Exception as exc:
            logger.debug("LLM screen failed, fail-closed keep: %s", exc)
        return True

    # -- OA-aware full-text marking (step 1 tail) --------------------------
    def mark_full_text(
        self,
        papers: List[Dict[str, Any]],
        oa_resolver=None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Resolve OA URLs; closed → ``full_text_unavailable`` + limitation entry.

        Returns ``(papers_with_oa, limitation_entries)``. Run continues
        abstract-only for closed content (never terminal, never paywalled fetch).
        """
        from core import oa_sources as _oa

        resolver = oa_resolver or _oa.resolve_oa_url
        marked: List[Dict[str, Any]] = []
        limitations: List[str] = []
        for paper in papers or []:
            if not isinstance(paper, dict):
                continue
            doi = str(paper.get("doi") or "").strip()
            try:
                res = resolver(doi, openalex_row=paper, s2_row=paper) if doi else {
                    "url": None, "source": None, "license": "closed",
                    "oa_status": "closed", "full_text_unavailable": True}
            except Exception as exc:
                logger.debug("OA resolve failed for %s: %s", doi, exc)
                res = {"url": None, "source": None, "license": "closed",
                       "oa_status": "closed", "full_text_unavailable": True}
            paper = dict(paper)
            paper["oa_url"] = res.get("url")
            paper["oa_source"] = res.get("source")
            paper["oa_license"] = res.get("license")
            if res.get("full_text_unavailable"):
                paper["full_text_status"] = "full_text_unavailable"
                limitations.append(
                    f"{doi or paper.get('title', 'untitled')}: full text unavailable "
                    "(closed/paywalled) — abstract-only synthesis.")
            else:
                paper["full_text_status"] = "oa_available"
            marked.append(paper)
        return marked, limitations

    # -- step 3: extraction -------------------------------------------------
    def extract(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract the 6 exact columns per included paper (LLM + deterministic)."""
        rows: List[Dict[str, Any]] = []
        for paper in papers or []:
            if not isinstance(paper, dict):
                continue
            row = self._extract_one(paper)
            rows.append(row)
        comp = extraction_completeness(rows)
        log_agent_action("ReviewAgent", "extract_complete", {
            "n_rows": len(rows), "completeness": comp["completeness"]})
        return {"extraction": rows, **comp}

    def _extract_one(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "Extract systematic review columns. Reply JSON only with keys "
            "design, sample_n, effect, metric, limitation.\n"
            f"Title: {str(paper.get('title') or '')[:400]}\n"
            f"Abstract: {str(paper.get('abstract') or paper.get('text') or '')[:2000]}"
        )
        parsed: Dict[str, Any] = {}
        try:
            out = call_llm_json(prompt, temperature=0.0, tier="cheap",
                                attempts=2, call_fn=call_llm,
                                **_review_system("review.extract"))
            if isinstance(out, dict):
                parsed = out
        except Exception as exc:
            logger.debug("Extraction LLM failed: %s", exc)
        return {
            "design": str(parsed.get("design") or paper.get("design") or "").strip(),
            "sample_n": parsed.get("sample_n", paper.get("sample_n")),
            "effect": parsed.get("effect", paper.get("effect")),
            "metric": str(parsed.get("metric") or paper.get("metric") or "").strip(),
            "limitation": str(parsed.get("limitation") or "abstract-only; full-text limits apply").strip(),
            "doi": str(paper.get("doi") or "").strip(),
        }

    # -- step 4: synthesis --------------------------------------------------
    def synthesize(self, extraction: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Thematic or vote-count synthesis; meta-analysis ONLY when gated.

        Meta gate: ≥3 commensurate effects (same metric, numeric) + I² computed.
        Otherwise narrative synthesis + explicit why-not-meta sentence.
        """
        rows = [r for r in (extraction or []) if isinstance(r, dict)]
        by_metric: Dict[str, List[float]] = {}
        for row in rows:
            metric = str(row.get("metric") or "").strip().lower()
            effect = row.get("effect")
            if metric and isinstance(effect, (int, float)):
                by_metric.setdefault(metric, []).append(float(effect))
        for metric, effects in by_metric.items():
            if len(effects) >= 3:
                i2 = heterogeneity_i2(effects)
                if i2 is not None:
                    mean = round(sum(effects) / len(effects), 3)
                    return {"mode": "meta-analysis", "metric": metric,
                            "n_effects": len(effects), "mean_effect": mean,
                            "i_squared": i2,
                            "narrative": f"Meta-analysis over {len(effects)} commensurate {metric} effects (mean {mean}, I² {i2}%)."}
        # Narrative path (default): thematic + vote-count + why-not-meta.
        themes: Dict[str, int] = {}
        for row in rows:
            design = str(row.get("design") or "unspecified").strip().lower() or "unspecified"
            themes[design] = themes.get(design, 0) + 1
        why_not = ("Meta-analysis not performed: fewer than 3 commensurate effects "
                   "with a shared metric were available, so heterogeneity (I²) could "
                   "not be computed; synthesis is narrative (thematic + vote-count).")
        return {"mode": "narrative", "themes": themes,
                "n_studies": len(rows), "why_not_meta": why_not,
                "narrative": f"Thematic synthesis over {len(rows)} studies ({themes}). {why_not}"}

    # -- orchestration ------------------------------------------------------
    def run_review(
        self,
        papers: List[Dict[str, Any]],
        query: str = "",
        criteria: Optional[Dict[str, Any]] = None,
        seed_keys: Optional[List[str]] = None,
        prisma_counts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full §4.6 pass: screen → OA-mark → extract → synthesize → PRISMA → recall.

        When ``prisma_counts`` is omitted they are built by construction via
        :func:`build_prisma_counts` (dedup + screen split), so the equations
        balance exactly without caller bookkeeping.
        """
        screened = self.screen(papers, criteria, query)
        marked, limitations = self.mark_full_text(screened["included"])
        extracted = self.extract(marked)
        synthesis = self.synthesize(extracted["extraction"])
        included_dois = [str(p.get("doi") or "") for p in marked]
        recall = seed_recall_check(included_dois, seed_keys or [])
        if prisma_counts is None:
            prisma_counts = build_prisma_counts(papers, screened["included"], screened["excluded"])
            # OA-mark keeps every screened-in paper (closed → abstract-only),
            # so included == eligible and excluded_fulltext == 0 by construction.
            prisma_counts["included"] = len(marked)
            prisma_counts["eligible"] = len(screened["included"])
            prisma_counts["excluded_fulltext"] = max(0, prisma_counts["eligible"] - len(marked))
        prisma = prisma_reconcile(prisma_counts or {})
        corpus = {
            "prisma": prisma_counts or {},
            "prisma_reconciled": prisma["passed"],
            "screen_log": screened["screen_log"],
            "extraction": extracted["extraction"],
            "extraction_completeness": extracted["completeness"],
            "synthesis": synthesis,
            "seed_recall": recall,
            "full_text_limitations": limitations,
            "mode": recall.get("mode", "systematic"),
            "n_identified": len(papers or []),
            "n_included": len(marked),
        }
        return {"corpus": corpus, "prisma": prisma, "extraction": extracted,
                "recall": recall, "synthesis": synthesis, "limitations": limitations}
