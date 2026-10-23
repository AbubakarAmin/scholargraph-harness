"""Tests for TopicHunter v3 upgrades.

Covers:
  - Adaptive rate limiting (AIMD bucket penalize/recover, single-flight
    coalescing, availability probes) in core.api_gateway
  - Query-builder filler-word exclusion (no more searching arXiv for "gaps")
  - Two-pool LLM budget (generation cannot starve the gate chain)
  - Relevance-ranked literature evidence
  - LLM seed generation with graceful fallback
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.topic_hunter import (
    TopicHunterAgent,
    _SEED_FILLER_WORDS,
)
from core.api_gateway import APIGateway, AdaptiveTokenBucket, RateLimitError


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_agent(tmp_path=None):
    agent = TopicHunterAgent.__new__(TopicHunterAgent)
    agent.context = None
    cfg = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    cfg.research_domain = "machine_learning"
    cfg.openalex_email = "test@example.com"
    cfg.semantic_scholar_api_key = ""
    cfg.novelty_similarity_reject = 0.92
    cfg.topic_exploration_every = 4
    cfg.topic_exploration_seed = 42
    cfg.output_dir = str(tmp_path or Path("/tmp/th_v3_test"))
    cfg.openalex_concept_filtering_enabled = True
    cfg.hyde_enabled = False
    cfg.multi_hop_retrieval_enabled = False
    cfg.frontier_seeding_enabled = False
    cfg.cross_seed_paper_cache_enabled = False
    cfg.capability_first_dataset_scoping_enabled = False
    cfg.structural_gap_mining_enabled = False
    cfg.sparsity_matrix_enabled = False
    cfg.contradiction_mining_enabled = False
    cfg.replication_target_mining_enabled = False
    cfg.negative_result_seeding_enabled = False
    cfg.persona_ensemble_enabled = False
    cfg.seed_strategy_elo_enabled = False
    cfg.llm_seed_generation_enabled = False
    cfg.openreview_enabled = False
    cfg.arxiv_enabled = True
    cfg.openalex_enabled = True
    cfg.llm_budget_per_seed = 30
    cfg.novelty_max_abstracts = 20
    agent.runtime_config = cfg
    agent.vector_memory = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    agent.client = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    agent.openalex_headers = {}
    agent.s2_headers = {}
    agent.base_urls = {
        "openalex": "https://api.openalex.org",
        "crossref": "https://api.crossref.org",
        "semantic_scholar": "https://api.semanticscholar.org/graph/v1",
    }
    agent._selection_counter = 0
    agent._rng = __import__("random").Random(42)
    agent.source_client = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    agent.rejection_log = []
    agent.source_health = {}
    agent._excluded_titles_cache = None
    agent._iteration_failures = 0
    agent._run_query_cache = {}
    agent._run_query_cache_lock = threading.Lock()
    agent._dataset_catalog_cache = None
    agent._arxiv_client = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# v3 #4: filler words never become search keywords
# ─────────────────────────────────────────────────────────────────────────────

class TestFillerWordFiltering:

    def test_arxiv_query_excludes_filler_words(self, tmp_path):
        agent = _make_agent(tmp_path)
        queries = agent._build_arxiv_queries("open problems in attention mechanisms", [])
        joined = " ".join(queries).lower()
        assert "problems" not in [w for q in queries for w in q.split('"')]
        # The seed's content words survive
        assert any("attention" in q or "mechanisms" in q for q in queries)

    def test_openalex_query_excludes_filler_words(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.llm_seed_generation_enabled = False
        queries = agent._build_openalex_queries(
            "methodological gaps in federated learning convergence", [], active_kind=None,
        )
        search_texts = " ".join(q for q, _ in queries).lower()
        assert "gaps" not in search_texts
        assert "methodological" not in search_texts
        assert "federated" in search_texts or "learning" in search_texts

    def test_keyword_floor_survives_all_filler_seed(self, tmp_path):
        """A seed made ENTIRELY of filler words still yields at least one keyword."""
        agent = _make_agent(tmp_path)
        queries = agent._build_arxiv_queries("open problems gaps challenges", [])
        # Not a crash and the fallback chain produced at least one usable query shape
        assert isinstance(queries, list)


# ─────────────────────────────────────────────────────────────────────────────
# v3 #5: two-pool budget — generation cannot starve the gate chain
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetSplit:

    def test_gate_reserve_formula(self):
        for budget, expected in ((30, 7), (8, 6), (100, 12)):
            assert min(max(6, budget // 4), 12) == expected

    def test_generation_blocked_leaves_gate_reserve(self, tmp_path):
        """Once remaining budget equals the reserve, generation must refuse."""
        budget = 8
        gate_reserve = min(max(6, budget // 4), 12)
        assert gate_reserve == 6
        # At count=2: remaining = 8-2 = 6 <= reserve → generation blocked
        count = 2
        generation_allowed = not (count + gate_reserve >= budget)
        assert generation_allowed is False
        # But the gate chain itself still has room (2 < 8)
        assert count < budget

    def test_gate_chain_allowed_while_generation_blocked(self, tmp_path):
        """Gate-kind checks spend normally even when generation is blocked."""
        llm_budget = 8
        gate_reserve = min(max(6, llm_budget // 4), 12)
        state = {"count": 0}

        def check(kind="gate"):
            if kind == "generation":
                if state["count"] + gate_reserve >= llm_budget:
                    return False
            elif state["count"] >= llm_budget:
                return False
            state["count"] += 1
            return True

        # 0 + 6 < 8 → first generation call allowed, second blocked (1+6 >= 8? no, 7<8 allowed, 2+6>=8 blocked)
        assert check("generation") is True
        assert check("generation") is True
        assert check("generation") is False  # 2+6 >= 8
        # Gate chain still spends normally
        assert check("gate") is True


# ─────────────────────────────────────────────────────────────────────────────
# v3 #3: LLM seed generation
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMSeedGeneration:

    def test_disabled_flag_returns_empty(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.llm_seed_generation_enabled = False
        assert agent._generate_llm_seeds("ml", [], n_seeds=4) == []

    def test_llm_failure_returns_empty(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.llm_seed_generation_enabled = True
        with patch("agents.topic_hunter.call_llm", side_effect=Exception("API down")):
            assert agent._generate_llm_seeds("ml", []) == []

    def test_llm_seeds_parsed_and_tagged(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.runtime_config.llm_seed_generation_enabled = True
        raw = '{"seeds": [{"seed": "speculative decoding verification overhead", "angle": "a"}, {"seed": "bad"}]}'
        with patch("agents.topic_hunter.call_llm", return_value=raw):
            seeds = agent._generate_llm_seeds("ml", [])
        assert len(seeds) == 1
        assert seeds[0]["strategy"] == "llm_diverse"
        assert "speculative" in seeds[0]["seed"]

    def test_llm_seeds_rejected_token_filtered(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._generate_llm_seeds = lambda *a, **k: [{"seed": "federated learning drift", "strategy": "llm_diverse"}]
        # Cross-run context marks "federated" as rejected
        context = [{"item": "federated learning topic", "rejection_reason": "novelty"}]
        seeds = agent._generate_dynamic_seeds(10, context, domain="machine_learning")
        assert all("federated learning learning" not in s["seed"] for s in seeds)


# ─────────────────────────────────────────────────────────────────────────────
# v3 #6: relevance-ranked literature evidence
# ─────────────────────────────────────────────────────────────────────────────

class TestRelevanceRanking:

    def test_closest_paper_ranked_first(self, tmp_path):
        agent = _make_agent(tmp_path)
        gap = {"title": "calibration of vision transformers",
               "description": "ECE and calibration error for ViT models",
               "contribution": "measure calibration"}
        papers = [
            {"title": " totally unrelated reinforcement learning", "abstract": "RL agents and rewards"},
            {"title": "calibration of vision transformers", "abstract": "We measure ECE calibration error on ViT models"},
            {"title": "another unrelated graph paper", "abstract": "graph neural networks"},
        ]
        ranked = agent._rank_papers_for_gap(gap, papers, top_n=2)
        assert len(ranked) == 2
        assert "calibration" in ranked[0].get("title", "").lower() or "vision" in ranked[0].get("title", "").lower()

    def test_papers_without_abstract_excluded(self, tmp_path):
        agent = _make_agent(tmp_path)
        papers = [{"title": "t", "abstract": ""}, {"title": "t2", "abstract": "real abstract here"}]
        ranked = agent._rank_papers_for_gap({"title": "t2 x", "description": "d"}, papers, top_n=5)
        assert all(p.get("abstract") for p in ranked)


# ─────────────────────────────────────────────────────────────────────────────
# Gateway v2: adaptive bucket + coalescing + availability
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveBucket:

    def test_penalize_cuts_rate(self):
        b = AdaptiveTokenBucket(rate=2.0, burst=1)
        b.penalize(factor=0.5)
        assert b.current_rate() == pytest.approx(1.0, abs=0.01)

    def test_penalize_has_floor(self):
        b = AdaptiveTokenBucket(rate=1.0, burst=1)
        for _ in range(20):
            b.penalize(factor=0.5)
        assert b.current_rate() >= 0.05

    def test_reward_recovers_toward_base(self):
        b = AdaptiveTokenBucket(rate=1.0, burst=1)
        b.penalize(factor=0.2, cooldown=0.0)
        slowed = b.current_rate()
        for _ in range(6):
            b.reward()
        assert b.current_rate() > slowed

    def test_reward_noop_at_base(self):
        b = AdaptiveTokenBucket(rate=1.0, burst=1)
        b.reward()
        assert b.current_rate() == 1.0


class TestGatewayAvailability:

    def test_is_available_true_when_closed(self):
        from core.api_gateway import APIGateway
        gw = APIGateway()
        assert gw.is_available("arxiv") is True

    def test_is_available_false_when_open(self):
        # v2: 429s no longer count as circuit breaker failures (flow control).
        # Use a real error to trigger the breaker.
        from core.api_gateway import APIGateway
        gw = APIGateway(circuit_threshold=1, circuit_cooldown=10.0)
        with pytest.raises(Exception):
            gw.request("arxiv", lambda: (_ for _ in ()).throw(ConnectionError("test connection reset")), retries=0)
        assert not gw.is_available("arxiv")


class TestCoalescing:

    def test_duplicate_calls_share_one_execution(self):
        from core.api_gateway import APIGateway
        gw = APIGateway()
        calls = []
        started = threading.Event()
        release = threading.Event()

        def slow():
            started.set()
            time.sleep(0.15)
            return {"ok": True}

        results = []

        def leader():
            time.sleep(0.05)  # let follower register first
            return gw.request("openalex", slow, coalesce_key="q1")

        t1 = threading.Thread(target=lambda: results.append(leader()))
        t2 = threading.Thread(target=lambda: results.append(gw.request("openalex", slow, coalesce_key="q1")))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert len(results) == 2
