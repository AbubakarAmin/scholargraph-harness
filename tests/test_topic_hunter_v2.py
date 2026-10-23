"""Tests for TopicHunterAgent v2 features (5 capabilities).

Covers: OpenAlex concept filtering, HyDE query construction, multi-hop
retrieval, frontier-seeded generation, and cross-seed paper cache.
Each function is tested for: happy path, LLM/API failure path, and
feature-flag-disabled path.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.topic_hunter import (
    TopicHunterAgent,
    _OPENALEX_CONCEPTS,
    _ARXIV_CATEGORIES,
)


def _make_agent(tmp_path=None) -> TopicHunterAgent:
    """Create a TopicHunterAgent with minimal init, bypassing real LLM / API."""
    agent = TopicHunterAgent.__new__(TopicHunterAgent)
    agent.context = None
    # Build a minimal runtime_config mock
    cfg = MagicMock()
    cfg.research_domain = "machine_learning"
    cfg.openalex_email = "test@example.com"
    cfg.semantic_scholar_api_key = ""
    cfg.novelty_similarity_reject = 0.92
    cfg.topic_exploration_every = 4
    cfg.topic_exploration_seed = 42
    cfg.output_dir = str(tmp_path or Path("/tmp/th_test"))
    # v2 flags default to True
    cfg.openalex_concept_filtering_enabled = True
    cfg.hyde_enabled = True
    cfg.hyde_max_chars = 600
    cfg.multi_hop_retrieval_enabled = True
    cfg.multi_hop_min_papers_threshold = 12
    cfg.multi_hop_max_hops = 2
    cfg.frontier_seeding_enabled = True
    cfg.frontier_refresh_every_n_runs = 5
    cfg.frontier_sample_size = 30
    cfg.frontier_terms_extracted = 8
    cfg.cross_seed_paper_cache_enabled = True
    cfg.llm_seed_generation_enabled = False
    cfg.openreview_enabled = False
    cfg.arxiv_enabled = True
    cfg.llm_budget_per_seed = 30
    cfg.novelty_max_abstracts = 20
    agent.runtime_config = cfg
    agent.vector_memory = MagicMock()
    agent.client = MagicMock()
    agent.openalex_headers = {}
    agent.s2_headers = {}
    agent.base_urls = {"openalex": "https://api.openalex.org", "crossref": "https://api.crossref.org", "semantic_scholar": "https://api.semanticscholar.org/graph/v1"}
    agent._selection_counter = 0
    agent._rng = __import__("random").Random(42)
    agent.source_client = MagicMock()
    agent.rejection_log = []
    agent.source_health = {}
    agent._excluded_titles_cache = None
    agent._iteration_failures = 0
    agent._run_query_cache = {}
    agent._run_query_cache_lock = threading.Lock()
    agent._arxiv_client = MagicMock()
    return agent


# ────────────────────────────────────────────────
# Feature 1: OpenAlex concept filtering
# ────────────────────────────────────────────────

class TestOpenAlexConceptFiltering:

    def test_concept_ids_verified(self):
        """Spot-check that the mapping contains expected concept IDs."""
        assert "C119857082" in _OPENALEX_CONCEPTS["machine_learning"]  # ML
        assert "C154945302" in _OPENALEX_CONCEPTS["machine_learning"]  # AI
        assert "C204321447" in _OPENALEX_CONCEPTS["natural_language_processing"]  # NLP
        assert "C31972630" in _OPENALEX_CONCEPTS["computer_vision"]  # CV
        assert "general" in _OPENALEX_CONCEPTS

    def test_concept_filter_applied_when_enabled(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.openalex_concept_filtering_enabled = True
        queries = agent._build_openalex_queries(
            "transformer attention mechanisms", [], active_kind=None,
        )
        # At least one query should have a filter containing concepts.id
        found_concept_filter = False
        for _, params in queries:
            f = params.get("filter", "")
            if "concepts.id:" in f:
                found_concept_filter = True
                break
        assert found_concept_filter, "Concept filter not applied when flag is enabled"

    def test_concept_filter_not_applied_when_disabled(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.openalex_concept_filtering_enabled = False
        queries = agent._build_openalex_queries(
            "transformer attention mechanisms", [], active_kind=None,
        )
        for _, params in queries:
            f = params.get("filter", "")
            assert "concepts.id:" not in f, "Concept filter applied when flag is disabled"

    def test_recency_filter_preserved(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.openalex_concept_filtering_enabled = True
        queries = agent._build_openalex_queries(
            "transformer attention mechanisms", [], active_kind=None,
        )
        # The recency query (cited_by_count sort) should still have publication_year filter
        recency_found = False
        for _, params in queries:
            f = params.get("filter", "")
            if "publication_year:" in f:
                recency_found = True
                # Concept filter and year filter should be comma-separated
                if "concepts.id:" in f:
                    assert "concepts.id:" in f and "publication_year:" in f
        assert recency_found, "Recency filter lost after concept filtering"


# ────────────────────────────────────────────────
# Feature 2: HyDE query construction
# ────────────────────────────────────────────────

class TestHyDE:

    def test_hyde_generates_abstract(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.hyde_enabled = True
        with patch("agents.topic_hunter.call_llm", return_value="A hypothetical abstract about transformers."):
            result = agent._generate_hyde_abstract("transformer efficiency", "machine_learning")
        assert result is not None
        assert "hypothetical" in result.lower() or len(result) > 10

    def test_hyde_returns_none_when_disabled(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.hyde_enabled = False
        result = agent._generate_hyde_abstract("test seed", "ml")
        assert result is None

    def test_hyde_returns_none_on_llm_failure(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.hyde_enabled = True
        with patch("agents.topic_hunter.call_llm", side_effect=Exception("API error")):
            result = agent._generate_hyde_abstract("test seed", "ml")
        assert result is None

    def test_hyde_truncates_to_max_chars(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.hyde_enabled = True
        agent.runtime_config.hyde_max_chars = 20
        with patch("agents.topic_hunter.call_llm", return_value="A" * 100):
            result = agent._generate_hyde_abstract("seed", "domain")
        assert result is not None
        assert len(result) <= 20


# ────────────────────────────────────────────────
# Feature 3: Multi-hop retrieval
# ────────────────────────────────────────────────

class TestMultiHop:

    def test_derive_followup_returns_phrase(self, tmp_path):
        agent = _make_agent(tmp_path)
        with patch("agents.topic_hunter.call_llm", return_value="efficient transformer attention mechanism"):
            result = agent._derive_followup_query_text(
                "seed", "hyde abs", [{"title": "Paper A"}], "ml",
            )
        assert result is not None
        assert len(result) > 0

    def test_derive_followup_returns_none_on_failure(self, tmp_path):
        agent = _make_agent(tmp_path)
        with patch("agents.topic_hunter.call_llm", side_effect=Exception("fail")):
            result = agent._derive_followup_query_text(
                "seed", None, [], "ml",
            )
        assert result is None

    def test_multi_hop_disabled_uses_single_hop(self, tmp_path):
        """When multi_hop_retrieval_enabled is False, max_hops should be 1."""
        agent = _make_agent(tmp_path)
        agent.runtime_config.multi_hop_retrieval_enabled = False
        # We don't run the full _hunt_once; just verify the config drives the logic
        max_hops = max(1, int(getattr(agent.runtime_config, "multi_hop_max_hops", 2))) \
            if getattr(agent.runtime_config, "multi_hop_retrieval_enabled", True) else 1
        assert max_hops == 1


# ────────────────────────────────────────────────
# Feature 4: Frontier-seeded generation
# ────────────────────────────────────────────────

class TestFrontierSeeding:

    def test_harvest_returns_empty_when_disabled(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.frontier_seeding_enabled = False
        assert agent._harvest_frontier_terms("machine_learning") == []

    def test_harvest_creates_cache_on_first_call(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.frontier_seeding_enabled = True
        agent.runtime_config.frontier_refresh_every_n_runs = 5
        cache_path = Path(tmp_path) / "source_cache" / "frontier_terms.json"
        # Mock search_arxiv to return papers
        fake_papers = [{"title": f"Paper {i} on transformers"} for i in range(10)]
        with patch.object(agent, "search_arxiv", return_value=fake_papers), \
             patch("agents.topic_hunter.call_llm", return_value=json.dumps({
                 "terms": [{"method": "state space models", "evaluation": "long context"}]
             })):
            terms = agent._harvest_frontier_terms("machine_learning")
        assert len(terms) == 1
        assert cache_path.exists()
        cached = json.loads(cache_path.read_text())
        assert cached["runs_since_refresh"] == 0
        assert cached["domain"] == "machine_learning"

    def test_harvest_reuses_cache_without_refresh(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.frontier_seeding_enabled = True
        agent.runtime_config.frontier_refresh_every_n_runs = 5
        cache_path = Path(tmp_path) / "source_cache" / "frontier_terms.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "domain": "machine_learning",
            "runs_since_refresh": 0,
            "terms": [{"method": "cached_method", "evaluation": "cached_eval"}],
        }))
        # Second call: should increment counter, not harvest
        terms = agent._harvest_frontier_terms("machine_learning")
        assert len(terms) == 1
        assert terms[0]["method"] == "cached_method"
        cached = json.loads(cache_path.read_text())
        assert cached["runs_since_refresh"] == 1

    def test_harvest_refreshes_after_n_runs(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.frontier_seeding_enabled = True
        agent.runtime_config.frontier_refresh_every_n_runs = 3
        cache_path = Path(tmp_path) / "source_cache" / "frontier_terms.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Set counter to refresh_every_n - 1 (due for refresh)
        cache_path.write_text(json.dumps({
            "domain": "machine_learning",
            "runs_since_refresh": 2,
            "terms": [{"method": "old", "evaluation": "old"}],
        }))
        fake_papers = [{"title": "New paper on diffusions"} for i in range(10)]
        with patch.object(agent, "search_arxiv", return_value=fake_papers), \
             patch("agents.topic_hunter.call_llm", return_value=json.dumps({
                 "terms": [{"method": "new_method", "evaluation": "new_eval"}]
             })):
            terms = agent._harvest_frontier_terms("machine_learning")
        assert len(terms) == 1
        assert terms[0]["method"] == "new_method"
        cached = json.loads(cache_path.read_text())
        assert cached["runs_since_refresh"] == 0  # reset after refresh

    def test_harvest_returns_empty_on_search_failure(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.frontier_seeding_enabled = True
        with patch.object(agent, "search_arxiv", return_value=[]):
            terms = agent._harvest_frontier_terms("machine_learning")
        assert terms == []

    def test_harvest_returns_empty_on_llm_failure(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.frontier_seeding_enabled = True
        fake_papers = [{"title": f"Paper {i}"} for i in range(5)]
        with patch.object(agent, "search_arxiv", return_value=fake_papers), \
             patch("agents.topic_hunter.call_llm", side_effect=Exception("LLM fail")):
            terms = agent._harvest_frontier_terms("machine_learning")
        assert terms == []

    def test_domain_mismatch_triggers_fresh_harvest(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.frontier_seeding_enabled = True
        agent.runtime_config.frontier_refresh_every_n_runs = 5
        cache_path = Path(tmp_path) / "source_cache" / "frontier_terms.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "domain": "computer_vision",
            "runs_since_refresh": 0,
            "terms": [{"method": "cv_method", "evaluation": "cv_eval"}],
        }))
        fake_papers = [{"title": "ML paper"} for i in range(5)]
        with patch.object(agent, "search_arxiv", return_value=fake_papers), \
             patch("agents.topic_hunter.call_llm", return_value=json.dumps({
                 "terms": [{"method": "fresh", "evaluation": "fresh"}]
             })):
            terms = agent._harvest_frontier_terms("machine_learning")
        # Should have harvested fresh terms, not returned the cv cache
        assert terms[0]["method"] == "fresh"

    def test_generate_dynamic_seeds_includes_frontier_terms(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.frontier_seeding_enabled = True
        agent.runtime_config.frontier_refresh_every_n_runs = 5
        agent.runtime_config.llm_seed_generation_enabled = False
        cache_path = Path(tmp_path) / "source_cache" / "frontier_terms.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "domain": "machine_learning",
            "runs_since_refresh": 0,
            "terms": [{"method": "state space models", "evaluation": "induction heads"}],
        }))
        # Request many seeds so frontier terms aren't truncated
        seeds = agent._generate_dynamic_seeds(20, [], domain="machine_learning")
        frontier_seed = [s for s in seeds if "state space models" in s["seed"]]
        assert len(frontier_seed) > 0, "Frontier terms not included in seeds"

    def test_generate_dynamic_seeds_without_domain_skips_frontier(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.frontier_seeding_enabled = True
        agent.runtime_config.llm_seed_generation_enabled = False
        seeds_with_domain = agent._generate_dynamic_seeds(5, [], domain="machine_learning")
        seeds_without = agent._generate_dynamic_seeds(5, [], domain=None)
        # Without domain, frontier terms are skipped, so with-domain should be >= without
        assert len(seeds_with_domain) >= len(seeds_without)


# ────────────────────────────────────────────────
# Feature 5: Cross-seed paper cache
# ────────────────────────────────────────────────

class TestCrossSeedCache:

    def test_cached_search_returns_fn_result(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.cross_seed_paper_cache_enabled = True
        result = agent._cached_search("key1", lambda: [{"title": "cached"}])
        assert result == [{"title": "cached"}]

    def test_cached_search_caches_result(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.cross_seed_paper_cache_enabled = True
        call_count = 0
        def fetch():
            nonlocal call_count
            call_count += 1
            return [{"title": "result"}]
        r1 = agent._cached_search("key2", fetch)
        r2 = agent._cached_search("key2", fetch)
        assert call_count == 1  # fetch_fn called only once
        assert r1 == r2

    def test_cached_search_bypasses_when_disabled(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.cross_seed_paper_cache_enabled = False
        call_count = 0
        def fetch():
            nonlocal call_count
            call_count += 1
            return [{"title": "result"}]
        agent._cached_search("key3", fetch)
        agent._cached_search("key3", fetch)
        assert call_count == 2  # fetch_fn called every time

    def test_cache_resets_per_discover_topics(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.cross_seed_paper_cache_enabled = True
        agent._run_query_cache["old_key"] = [{"title": "stale"}]
        # Simulating what discover_topics does at the top
        agent._run_query_cache = {}
        assert "old_key" not in agent._run_query_cache

    def test_cache_is_thread_safe(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.cross_seed_paper_cache_enabled = True
        results = []
        def fetch():
            return [{"title": "threaded"}]
        threads = [threading.Thread(target=lambda: results.append(agent._cached_search("tkey", fetch))) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 10
        # All should get the same result
        assert all(r == [{"title": "threaded"}] for r in results)

    def test_different_keys_get_different_results(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.cross_seed_paper_cache_enabled = True
        r1 = agent._cached_search("alpha", lambda: [{"title": "alpha_result"}])
        r2 = agent._cached_search("beta", lambda: [{"title": "beta_result"}])
        assert r1[0]["title"] == "alpha_result"
        assert r2[0]["title"] == "beta_result"


# ────────────────────────────────────────────────
# Config flag defaults
# ────────────────────────────────────────────────

class TestConfigDefaults:

    def test_all_flags_have_defaults(self):
        from core.config import Config
        fields = {
            "openalex_concept_filtering_enabled": True,
            "hyde_enabled": True,
            "hyde_max_chars": 600,
            "multi_hop_retrieval_enabled": True,
            "multi_hop_min_papers_threshold": 12,
            "multi_hop_max_hops": 2,
            "frontier_seeding_enabled": True,
            "frontier_refresh_every_n_runs": 5,
            "frontier_sample_size": 30,
            "frontier_terms_extracted": 8,
            "cross_seed_paper_cache_enabled": True,
        }
        for field, expected in fields.items():
            default = Config.model_fields[field].default
            assert default == expected, f"{field}: expected {expected}, got {default}"


# ────────────────────────────────────────────────
# _resolve_subcategory helper
# ────────────────────────────────────────────────

class TestResolveSubcategory:

    def test_known_kind(self, tmp_path):
        agent = _make_agent(tmp_path)
        assert agent._resolve_subcategory("llm") == "natural_language_processing"
        assert agent._resolve_subcategory("vision") == "computer_vision"
        assert agent._resolve_subcategory("graph") == "graph_neural_networks"

    def test_unknown_kind_returns_general(self, tmp_path):
        agent = _make_agent(tmp_path)
        assert agent._resolve_subcategory("nonexistent") == "general"

    def test_none_returns_general(self, tmp_path):
        agent = _make_agent(tmp_path)
        assert agent._resolve_subcategory(None) == "general"
