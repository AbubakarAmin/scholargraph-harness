"""Regression tests for run 95c46c6f / 60271359 P1+P2 fixes (all offline)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from agents.topic_hunter import (
    TopicHunterAgent,
    _normalize_seed_text,
    effective_discovery_timeout,
)
from core.capabilities import SANDBOX_CAPABILITY_MANIFEST, check_plan_feasibility
from core.run_log import CrossRunMemory


def _hunter_mini_plan() -> dict:
    """The exact mini_plan shape discover_topics screens for fallbacks."""
    return {
        "title": "Exploratory evaluation of reproducibility factors in speculative decoding",
        "description": (
            "A bounded exploratory study on bundled_synthetic (local-only, no outbound "
            "downloads) generated because strict topic gates yielded no candidate."
        ),
        "methodology": "Deterministic bundled-synthetic baseline comparison with controlled seeds.",
        "experiments": [
            {"name": "minimum_viable_experiment",
             "dataset": {"name": "bundled_synthetic", "access_policy": "local-only"}}
        ],
    }


class TestFeasibilityWordBoundaries:
    def test_reproducibility_title_passes(self):
        assert check_plan_feasibility(_hunter_mini_plan(), SANDBOX_CAPABILITY_MANIFEST) == []

    def test_real_uci_dataset_still_flagged(self):
        plan = _hunter_mini_plan()
        plan["experiments"] = [{"dataset": {"name": "UCI Machine Learning Repository"}}]
        assert check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST) != []

    def test_real_outbound_download_still_flagged(self):
        plan = {
            "title": "External crawl study",
            "description": "A study on a custom web crawl.",
            "methodology": "Download the dataset from the internet and train.",
            "experiments": [{"dataset": {"name": "custom_crawl"}}],
        }
        assert check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST) != []

    def test_gpu_still_flagged(self):
        plan = _hunter_mini_plan()
        plan["methodology"] = "Fine-tune with CUDA on GPUs."
        assert check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST) != []


class TestSeedReasonTag:
    def test_seed_exhausted_tag(self):
        assert CrossRunMemory._reason_tag("seed_exhausted_no_topics") == "seed_exhausted"


class TestDiscoveryAutoscale:
    def test_floor_covers_all_seeds(self):
        assert effective_discovery_timeout(5, 1200, 3600) == 5 * 1200 + 300

    def test_configured_can_exceed_floor(self):
        assert effective_discovery_timeout(2, 100, 3600) == 3600

    def test_old_defaults_would_starve(self):
        # Documents the 95c46c6f failure mode: 900 s window, 180 s seeds.
        assert effective_discovery_timeout(5, 180, 900) > 900


def _canned_seeds(*seeds: str) -> str:
    return json.dumps({"seeds": [{"seed": s, "angle": "test"} for s in seeds]})


class TestSeedExclusion:
    def _agent(self) -> TopicHunterAgent:
        return TopicHunterAgent()

    def test_attempted_phrases_excluded_from_seeds(self):
        agent = self._agent()
        agent._attempted_seeds.add(_normalize_seed_text("speculative decoding verification overhead"))
        with patch("agents.topic_hunter.call_llm", return_value=_canned_seeds(
                "speculative decoding verification overhead",
                "diffusion denoising step scheduling latency")):
            with patch.object(TopicHunterAgent, "_harvest_frontier_terms", return_value=[]):
                with patch.object(TopicHunterAgent, "_load_cached_frontier_terms", return_value=[]):
                    seeds = agent._generate_dynamic_seeds(5, cross_run_context=[], domain="machine_learning")
        texts = [s["seed"] for s in seeds]
        assert "speculative decoding verification overhead" not in texts
        assert "diffusion denoising step scheduling latency" in texts

    def test_llm_prompt_carries_avoid_list(self):
        agent = self._agent()
        captured = {}

        def _fake(prompt, temperature=0.9, tier="cheap", **kwargs):
            captured["prompt"] = prompt
            captured["temperature"] = temperature
            return _canned_seeds("brand new seed phrase here")

        with patch("agents.topic_hunter.call_llm", side_effect=_fake):
            agent._generate_llm_seeds("machine_learning", [], exclude_phrases=["old tried seed"], temperature=1.1)
        assert "old tried seed" in captured["prompt"]
        assert captured["temperature"] == 1.1

    def test_seed_rejections_become_phrase_exclusions_not_word_bans(self):
        agent = self._agent()
        context = [{"kind": "seed", "item": "transformer efficiency benchmark comparison",
                    "rejection_reason": "seed_exhausted"}]
        with patch("agents.topic_hunter.call_llm", return_value=_canned_seeds(
                "transformer scaling long-context benchmark study")):
            with patch.object(TopicHunterAgent, "_harvest_frontier_terms", return_value=[]):
                with patch.object(TopicHunterAgent, "_load_cached_frontier_terms", return_value=[]):
                    seeds = agent._generate_dynamic_seeds(5, cross_run_context=context, domain="machine_learning")
        texts = [s["seed"] for s in seeds]
        # Exact phrase gone, but a different phrase sharing the word
        # "transformer" survives (word-level banning would have dropped it):
        assert "transformer efficiency benchmark comparison" not in texts
        assert "transformer scaling long-context benchmark study" in texts


class TestOpenAlexQueryLegs:
    def test_recency_leg_differs_from_relevance_leg(self):
        agent = TopicHunterAgent()
        queries = agent._build_openalex_queries("transformer efficiency benchmark comparison", [])
        assert len(queries) >= 2
        assert queries[0][0] != queries[1][0]
        assert queries[0][1] != queries[1][1]


class TestPlannerTier0:
    def test_force_tier0_passes_feasibility(self):
        from agents.planner import PlannerAgent

        agent = PlannerAgent()
        plan = {
            "title": "Study with Yahoo Finance download",
            "methodology": "Download from the internet, train on GPUs with CUDA.",
            "compute_budget": "GPU cluster",
            "experiments": [{"name": "e1", "dataset": {"name": "UCI ML repo"}}],
            "contributions": [],
            "expected_contributions": [],
            "sections": [],
            "revision_history": [],
        }
        agent._force_tier0_plan(plan, {"title": "t"}, ["plan requires outbound dataset access"])
        agent._auto_fix_manifest_violations(plan, SANDBOX_CAPABILITY_MANIFEST)
        assert check_plan_feasibility(plan, SANDBOX_CAPABILITY_MANIFEST) == []
        assert plan.get("tier0_coerced") is True
        assert plan["experiments"][0]["dataset"]["name"] == "bundled_synthetic"


class TestExploratoryGate:
    def test_infra_failure_refused(self):
        from core.workflow_nodes import _exploratory_acceptable

        topic = {"title": "t", "fallback_provenance": {"reason": "x"}}
        assert _exploratory_acceptable(topic, SimpleNamespace(infra_failure=True)) is False

    def test_inconclusive_fallback_accepted(self):
        from core.workflow_nodes import _exploratory_acceptable

        topic = {"title": "t", "fallback_provenance": {"reason": "x"}}
        assert _exploratory_acceptable(topic, SimpleNamespace(infra_failure=False)) is True

    def test_non_fallback_never_accepted(self):
        from core.workflow_nodes import _exploratory_acceptable

        assert _exploratory_acceptable({"title": "t"}, SimpleNamespace(infra_failure=False)) is False


class TestResumeImport:
    def test_load_state_from_db_resolves(self):
        import web.app as app

        workspace = {"mode": "full_research", "topics": [{"title": "t"}]}
        fake_runs = [{"run_id": "r-resume-test", "summary_json": json.dumps({"workspace": workspace})}]
        with patch.object(app.research_db, "list_runs", return_value=fake_runs):
            state = app._load_state_from_db("r-resume-test")
        assert state is not None
        assert state["run_id"] == "r-resume-test"
        assert state["topics"] == [{"title": "t"}]

    def test_load_state_from_db_missing_run(self):
        import web.app as app

        with patch.object(app.research_db, "list_runs", return_value=[]):
            assert app._load_state_from_db("nope") is None
