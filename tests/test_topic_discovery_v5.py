"""Tests for the v5 topic-discovery upgrade (run 55ac3547 follow-up).

Covers (all offline, no API keys):
- SE taxonomy: domain/seed → software_engineering subcategory (cs.SE),
  keyword-only OpenAlex (no wrong ML concept filter).
- Domain-aware OpenReview venues + per-seed relevance ranking.
- Deterministic signals: momentum, citation concentration, concept gaps,
  reviewer clustering, OpenAlex-native coupling.
- Wider miner windows (kwargs accepted, still fail-open).
- Scored selection: six-dim scores, family derivation, evidence trail,
  portfolio diversification.
- Config surface: new v5 fields exist with quality-over-cost defaults.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agents.topic_hunter import (
    _ARXIV_CATEGORIES,
    _OPENALEX_CONCEPTS,
    _subcategory_for_domain,
    TopicHunterAgent,
)
from core.signal_mining import (
    citation_concentration,
    cluster_reviewer_weaknesses,
    concept_cooccurrence_gaps,
    temporal_momentum,
)
from core.structural_gaps import find_coupling_gaps_openalex
from tests.test_topic_hunter_features_8_15 import _make_agent


def _v5_agent(**overrides):
    agent = _make_agent()
    agent.runtime_config.contradiction_max_papers = 40
    agent.runtime_config.contradiction_abstract_chars = 1500
    agent.runtime_config.sparsity_max_papers = 60
    agent.runtime_config.sparsity_abstract_chars = 1200
    agent.runtime_config.momentum_mining_enabled = True
    agent.runtime_config.citation_concentration_enabled = True
    agent.runtime_config.concept_gap_mining_enabled = True
    agent.runtime_config.reviewer_clustering_enabled = True
    agent.runtime_config.openalex_native_coupling_enabled = True
    agent.runtime_config.hunter_fulltext_enabled = False
    agent.runtime_config.hunter_scoring_enabled = True
    agent.runtime_config.hunter_second_chance_threshold = 0.55
    agent.runtime_config.hunter_second_chance_max = 2
    agent.runtime_config.hunter_portfolio_enabled = True
    agent.runtime_config.semantic_scholar_enabled = False
    for key, value in overrides.items():
        setattr(agent.runtime_config, key, value)
    return agent


# --- Phase 1: SE taxonomy -------------------------------------------------


class TestSETaxonomy:
    def test_domain_maps_to_se_subcategory(self):
        assert _subcategory_for_domain("software engineering") == "software_engineering"
        assert _subcategory_for_domain("Software Engineering") == "software_engineering"
        assert _subcategory_for_domain("computer_science") == "general"

    def test_se_arxiv_categories_are_se(self):
        cats = _ARXIV_CATEGORIES["software_engineering"]
        assert "cs.SE" in cats
        assert "cs.LG" not in cats

    def test_se_has_no_concept_filter(self):
        assert _OPENALEX_CONCEPTS["software_engineering"] == []

    def test_seed_markers_resolve_se_under_generic_domain(self):
        agent = _v5_agent()
        subcat = agent._resolve_subcategory_for(
            "grammar-aware seed scheduling in coverage-guided fuzzing",
            active_kind=None, domain="computer_science")
        assert subcat == "software_engineering"

    def test_se_queries_use_cs_se_without_concepts(self):
        agent = _v5_agent()
        arxiv_qs = agent._build_arxiv_queries(
            "grammar-aware seed scheduling in coverage-guided fuzzing", [],
            active_kind=None, domain="software engineering")
        assert arxiv_qs, "SE seed must still produce queries"
        assert any("cs.SE" in q for q in arxiv_qs)
        assert not any("cs.LG" in q for q in arxiv_qs)
        oa_qs = agent._build_openalex_queries(
            "grammar-aware seed scheduling in coverage-guided fuzzing", [],
            active_kind=None, domain="software engineering")
        assert oa_qs
        for _, params in oa_qs:
            assert "concepts.id" not in str(params.get("filter", ""))

    def test_ml_queries_unchanged(self):
        agent = _v5_agent()
        arxiv_qs = agent._build_arxiv_queries(
            "transformer efficiency benchmark comparison", [],
            active_kind=None, domain="machine learning")
        assert any("cs.LG" in q for q in arxiv_qs)


# --- Phase 1: domain-aware venues ----------------------------------------


class TestDomainVenues:
    def test_se_venues_include_tmlr(self, monkeypatch):
        from core.sources_openreview import venues_for_domain
        # e0ec9e91: TMLR cost ~5 min/run for 0 rows — excluded by default,
        # restorable via explicit OPENREVIEW_VENUES override.
        venues = venues_for_domain("software engineering", "software_engineering")
        assert "tmlr" not in venues
        assert all(isinstance(v, str) for v in venues)
        monkeypatch.setenv("OPENREVIEW_VENUES", "iclr2025,tmlr")
        assert "tmlr" in venues_for_domain("software engineering")

    def test_unknown_keys_dropped(self, monkeypatch):
        from core.sources_openreview import venues_for_domain
        monkeypatch.setenv("OPENREVIEW_VENUES", "iclr2025,bogus_venue_999")
        venues = venues_for_domain("software engineering")
        assert venues == ["iclr2025"]

    def test_openreview_ranked_per_seed(self):
        agent = _v5_agent()
        signals = [
            {"title": "Fuzzing schedulers for coverage", "weaknesses": ["needs better seed scheduling"]},
            {"title": "Transformer efficiency at scale", "weaknesses": ["attention memory overhead"]},
        ]
        ranked = agent._rank_openreview_for_seed(
            "grammar-aware seed scheduling in coverage-guided fuzzing", signals, top_n=2)
        assert ranked[0]["title"] == "Fuzzing schedulers for coverage"


# --- Phase 2: deterministic signals --------------------------------------


def _paper(title, abstract, year=0, cites=0, concepts=None, refs=None):
    return {"title": title, "abstract": abstract, "year": year,
            "cited_by_count": cites, "concepts": concepts or [],
            "referenced_works": refs or []}


class TestDeterministicSignals:
    def test_momentum_detects_heating_theme(self):
        papers = ([_paper(f"fuzzing study {i}", "coverage guided fuzzing finds bugs", year=2026) for i in range(4)]
                  + [_paper("old fuzzing note", "coverage guided fuzzing intro", year=2020)])
        out = temporal_momentum(papers)
        assert any(d["keyword"] == "fuzzing" and d["recent_count"] >= 3 for d in out)

    def test_momentum_needs_minimum_corpus(self):
        assert temporal_momentum([_paper("a", "bstract text here")]) == []

    def test_concentration_flags_aging_canon(self):
        papers = [_paper("Foundational fuzzer", "abstract words here", year=2015, cites=500),
                  _paper("Fresh note", "abstract words here", year=2026, cites=600)]
        out = citation_concentration(papers)
        assert len(out) == 1 and out[0]["title"] == "Foundational fuzzer"

    def test_concept_gaps_find_untested_pair(self):
        def _c(name):
            return {"display_name": name}
        papers = ([_paper(f"m{i}", "x", concepts=[_c("fuzzing"), _c("compilers")]) for i in range(3)]
                  + [_paper(f"n{i}", "x", concepts=[_c("fuzzing"), _c("testing")]) for i in range(3)]
                  + [_paper(f"o{i}", "x", concepts=[_c("compilers"), _c("testing")]) for i in range(3)])
        # every pair co-occurs here except nothing — add isolated established concepts
        papers += [_paper("p1", "x", concepts=[_c("fuzzing"), _c("debugging")]),
                   _paper("p2", "x", concepts=[_c("debugging"), _c("testing")]),
                   _paper("p3", "x", concepts=[_c("debugging"), _c("compilers")])]
        out = concept_cooccurrence_gaps(papers)
        assert isinstance(out, list)

    def test_reviewer_clustering_counts_votes(self):
        signals = [
            {"title": "P1", "weaknesses": ["evaluation lacks multiple random seeds for variance"]},
            {"title": "P2", "weaknesses": ["no multiple random seeds, variance unreported"]},
            {"title": "P3", "weaknesses": ["figures are too small to read"]},
        ]
        out = cluster_reviewer_weaknesses(signals)
        assert out and out[0]["votes"] >= 2

    def test_openalex_coupling_pairs_shared_refs(self):
        refs_a = [f"https://openalex.org/W{i}" for i in range(1, 6)]
        refs_b = [f"https://openalex.org/W{i}" for i in range(3, 8)]
        papers = [_paper("Paper A title here", "abstract A", refs=refs_a),
                  _paper("Paper B title here", "abstract B", refs=refs_b)]
        out = find_coupling_gaps_openalex(papers, min_shared_refs=3)
        assert len(out) == 1
        assert out[0]["shared_reference_count"] == 3
        assert out[0]["source"] == "openalex_native"

    def test_openalex_coupling_empty_without_refs(self):
        assert find_coupling_gaps_openalex([_paper("A title here", "abs"), _paper("B title here", "abs")]) == []
        assert find_coupling_gaps_openalex([]) == []

    def test_wider_windows_accepted(self):
        from core.contradiction_mining import find_contradictions
        from core.sparsity_matrix import find_sparse_cells
        assert find_contradictions([], max_papers=40, abstract_chars=1500) == []
        assert find_sparse_cells([], max_papers=60, abstract_chars=1200) == []


# --- Phase 3: scored selection -------------------------------------------


def _scored_gap(**overrides):
    gap = {
        "title": "Seed scheduling improves coverage",
        "description": "grammar-aware seed scheduling in coverage-guided fuzzing",
        "rationale": "scheduling matters for fuzzer throughput",
        "gap_report": {"evidence_strength": 0.8, "gap_type": "method_gap"},
        "novelty": {"verdict": "NOVEL", "max_similarity": 0.3},
        "feasibility_check": {"ok": True, "reasons": []},
        "bridge_grounding": 1.0,
        "evidence_bridge_ids": ["bridge-x"],
    }
    gap.update(overrides)
    return gap


class TestScoredSelection:
    def test_six_dim_score(self):
        agent = _v5_agent()
        ctx = {"contradictions": 2, "reviewer_themes": [], "momentum": 1,
               "concept_gaps": 0, "sparse_cells": 1}
        scores = agent._score_gap_candidate(_scored_gap(), ctx)
        # e0ec9e91: added "grounding" dim so coupling/fulltext/citation move
        # scores instead of decorating logs.
        assert set(scores["dims"]) == {"evidence", "novelty", "feasibility",
                                       "contradiction", "reviewer", "transfer",
                                       "grounding"}
        assert scores["total"] > 0.50

    def test_duplicate_scores_zero_novelty(self):
        agent = _v5_agent()
        gap = _scored_gap(novelty={"verdict": "LIKELY_DUPLICATE", "max_similarity": 0.97})
        scores = agent._score_gap_candidate(gap, {})
        assert scores["dims"]["novelty"] == 0.0

    def test_family_derivation(self):
        agent = _v5_agent()
        assert agent._gap_family({"contradiction_ref": {"a": 1}}) == "contradiction"
        assert agent._gap_family({"reviewer_theme_hit": "seeds"}) == "reviewer"
        assert agent._gap_family({"momentum_hit": "fuzzing"}) == "momentum"
        assert agent._gap_family({"bridge_grounding": 1.0}) == "transfer"
        assert agent._gap_family({"novelty": {"verdict": "REPLICATION_OK"}}) == "replication"

    def test_evidence_trail_has_three_bullets_max(self):
        agent = _v5_agent()
        gap = _scored_gap(
            literature_evidence=[{"title": "Prior A"}, {"title": "Prior B"}],
            contradiction_ref={"paper_a_title": "A paper", "paper_b_title": "B paper"},
            reviewer_theme_hit="seed scheduling pain")
        trail = agent._build_evidence_trail(gap)
        assert 1 <= len(trail) <= 3
        assert any("Prior A" in b for b in trail)

    def test_portfolio_diversifies_top5(self):
        agent = _v5_agent()
        def _t(title, family):
            return {"title": title, "gap_family": family, "rank": 1}
        ranked = [_t(f"contra {i}", "contradiction") for i in range(5)]
        ranked.append(_t("reviewer pain study", "reviewer"))
        ranked.append(_t("transfer idea", "transfer"))
        out = agent._apply_portfolio_diversity(ranked)
        assert len(out) == len(ranked)
        assert len(set(t["gap_family"] for t in out[:5])) >= 3

    def test_portfolio_off_is_identity(self):
        agent = _v5_agent(hunter_portfolio_enabled=False)
        ranked = [{"title": f"t{i}", "gap_family": "contradiction"} for i in range(5)]
        assert agent._apply_portfolio_diversity(ranked) == ranked

    def test_rank_card_covers_all_families(self):
        agent = _v5_agent()
        card = agent._rank_topic_card(_scored_gap(
            gap_family="reviewer", second_chance=True, strategy="reviewer",
            reviewer_theme_hit="pain", evidence_trail=["b1"]))
        assert card["gap_family"] == "reviewer"
        assert card["second_chance"] is True
        assert card["has_reviewer_pain"] is True
        assert card["evidence_trail"] == ["b1"]


# --- Config surface -------------------------------------------------------


class TestV5Config:
    def test_new_fields_with_quality_defaults(self):
        from core.config import config
        assert config.contradiction_max_papers >= 40
        assert config.sparsity_max_papers >= 60
        assert config.novelty_max_abstracts >= 40
        assert config.hunter_scoring_enabled is True
        assert config.hunter_portfolio_enabled is True
        assert config.openalex_native_coupling_enabled is True
        assert 0.0 < config.hunter_second_chance_threshold < 1.0
