"""Tests for the run-25ed95d4 reliability fixes (4-fix batch).

Fix 1: Gemini-only embeddings + gateway non-retryable 4xx + TF-IDF fallback.
Fix 2: Hunter feasibility enforcement (manifest triggers, admission scale/GPU
       checks, cross-iteration seed exclusion + fuzzy rephrase guard).
Fix 3: Throughput (single formalize repair, leaner retrieval fanout,
       frontier-first seeds).
Fix 4: Terminal best-effort promotion + unified iteration cap.

All offline/mocked — no API keys required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.api_gateway import APIGateway, CircuitOpen, RateLimitError


# ---------------------------------------------------------------------------
# Fix 1a: gateway treats permanent 4xx as non-retryable
# ---------------------------------------------------------------------------

class TestNonRetryableGateway:
    def _gateway(self) -> APIGateway:
        return APIGateway(circuit_threshold=100, circuit_cooldown=60.0)

    def test_405_status_code_raises_immediately_single_attempt(self):
        gw = self._gateway()
        calls = []

        def boom():
            calls.append(1)
            err = RuntimeError("POST /v1/embeddings -> 405 Method Not Allowed")
            err.status_code = 405
            raise err

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                gw.request("embedding", boom, retries=3, backoff_base=0.01)
        assert len(calls) == 1, f"non-retryable error must not be retried, got {len(calls)} calls"
        mock_sleep.assert_not_called()

    def test_message_only_method_not_allowed_is_non_retryable(self):
        gw = self._gateway()
        calls = []

        def boom():
            calls.append(1)
            raise RuntimeError("Method Not Allowed")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                gw.request("embedding", boom, retries=3, backoff_base=0.01)
        assert len(calls) == 1
        mock_sleep.assert_not_called()

    def test_429_rate_limit_error_still_retries(self):
        gw = self._gateway()
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise RateLimitError("embedding", retry_after=0.01)
            return "ok"

        with patch("time.sleep"):
            assert gw.request("embedding", flaky, retries=3, backoff_base=0.01) == "ok"
        assert len(calls) == 2

    def test_503_still_transient(self):
        assert not APIGateway._is_non_retryable(RuntimeError("upstream 503 unavailable"))
        assert APIGateway._is_non_retryable(RuntimeError("404 Not Found"))
        assert APIGateway._is_non_retryable(RuntimeError("Error code: 401 - unauthorized"))


# ---------------------------------------------------------------------------
# Fix 1b: embeddings always route to Gemini
# ---------------------------------------------------------------------------

class TestGeminiOnlyEmbeddings:
    def _client(self, **kwargs):
        from core import llm as llm_mod

        with patch.object(llm_mod.config, "google_api_key", "test-gemini-key"), \
             patch.object(llm_mod.config, "openai_api_key", "test-openai-key"), \
             patch.object(llm_mod.config, "openai_base_url", "https://chat.example/v1"), \
             patch.object(llm_mod.config, "llm_provider", "openai_compatible"):
            llm_mod.reset_embedding_health()
            client = llm_mod.LLMClient(provider="openai_compatible")
            llm_mod.reset_embedding_health()
            return client, llm_mod

    def test_embed_uses_gemini_not_openai_endpoint(self):
        client, llm_mod = self._client()
        with patch.object(client._gemini.models, "embed_content") as mock_gemini, \
             patch("core.llm.get_gateway") as mock_gw:
            # Route the gateway straight through to the raw fn (no sleeps).
            mock_gw.return_value.request.side_effect = lambda _p, fn, *a, **k: fn(*a, **k)
            mock_gemini.return_value = SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.5] * 8)])
            with patch.object(llm_mod.config, "embedding_dimension", 8):
                vec = client.embed("some abstract text")
            mock_gemini.assert_called_once()
            assert len(vec) == 8

    def test_permanent_failure_latches_and_short_circuits(self):
        from core import llm as llm_mod

        client, _ = self._client()
        err = RuntimeError("405 Method Not Allowed")
        err.status_code = 405
        with patch("core.llm.get_gateway") as mock_gw:
            mock_gw.return_value.request.side_effect = err
            mock_gw.return_value.is_available.return_value = True
            first = client.embed("text one")
            calls_after_first = mock_gw.return_value.request.call_count
            second = client.embed("text two")
            calls_after_second = mock_gw.return_value.request.call_count
        assert (first == 0).all() and (second == 0).all()
        assert calls_after_first == 1
        assert calls_after_second == 1, "latched backend must not hit the gateway again"
        assert not llm_mod.is_embedding_available()
        llm_mod.reset_embedding_health()
        assert llm_mod.is_embedding_available()


# ---------------------------------------------------------------------------
# Fix 1c: TF-IDF fallback similarity
# ---------------------------------------------------------------------------

class TestTfidfCosine:
    def test_identical_texts_score_one(self):
        from core.utils import tfidf_cosine

        assert tfidf_cosine("transformer efficiency benchmark latency",
                            "transformer efficiency benchmark latency") == pytest.approx(1.0)

    def test_unrelated_texts_score_low(self):
        from core.utils import tfidf_cosine

        assert tfidf_cosine("transformer efficiency benchmark latency",
                            "histopathology blendshape avatar animation") < 0.10

    def test_empty_inputs_score_zero(self):
        from core.utils import tfidf_cosine

        assert tfidf_cosine("", "something") == 0.0
        assert tfidf_cosine("the and of", "the and of") == 0.0

    def test_calculate_similarity_falls_back_on_zero_vectors(self):
        from core import utils as utils_mod

        with patch.object(utils_mod, "generate_embedding",
                          return_value=np.zeros(8)):
            sim = utils_mod.calculate_similarity("speculative decoding latency",
                                                 "speculative decoding latency")
        assert sim == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fix 1d: novelty gate forces LLM comparison when embeddings are down
# ---------------------------------------------------------------------------

class TestNoveltyEmbeddingFallback:
    def _agent(self):
        from agents.topic_hunter import TopicHunterAgent

        agent = TopicHunterAgent.__new__(TopicHunterAgent)
        agent.runtime_config = SimpleNamespace(novelty_similarity_reject=0.88)
        agent.context = None
        return agent

    def test_zero_vectors_force_llm_comparison_not_auto_novel(self):
        from agents import topic_hunter as th

        agent = self._agent()
        gap = {"title": "Speculative decoding verification overhead",
               "description": "measuring per-token verification latency",
               "contribution": "new reporting metric"}
        papers = [{"title": "Speculative decoding verification overhead study",
                   "abstract": "we measure per-token verification latency in speculative decoding systems",
                   "doi": "10.1/abc"}]
        comparison_json = ('{"comparisons": [{"paper": "x", "overlap": "same metric", '
                           '"difference": "different setup", "remaining_gap": "none", '
                           '"is_novel": true, "verdict": "NOVEL"}], '
                           '"overall_novelty_verdict": "NOVEL", "reject": false}')
        with patch.object(th, "generate_embedding", return_value=np.zeros(8)), \
             patch.object(th, "call_llm", return_value=comparison_json) as mock_llm:
            report = agent.evaluate_layered_novelty(gap, papers)
        assert report["embedding_fallback"] is True
        assert mock_llm.called, "fallback must run the LLM contribution comparison"
        assert len(report["novelty_comparisons"]) == 1
        assert report["reject"] is False
        assert "unverified by vectors" in report["reason"]

    def test_nonzero_vectors_keep_embedding_path(self):
        from agents import topic_hunter as th

        agent = self._agent()
        gap = {"title": "Totally different idea here",
               "description": "unrelated synthetic description words",
               "contribution": "novelty claim"}
        papers = [{"title": "Other paper", "abstract": "zebras quantum photosynthesis",
                   "doi": "10.1/other"}]
        with patch.object(th, "generate_embedding",
                          return_value=np.array([0.1] * 8)), \
             patch.object(th, "call_llm") as mock_llm:
            report = agent.evaluate_layered_novelty(gap, papers)
        assert report["embedding_fallback"] is False
        assert mock_llm.call_count == 0, "no comparison when nothing overlaps"


# ---------------------------------------------------------------------------
# Fix 2: feasibility enforcement
# ---------------------------------------------------------------------------

class TestFeasibilityEnforcement:
    def _agent(self):
        from agents.topic_hunter import TopicHunterAgent

        agent = TopicHunterAgent.__new__(TopicHunterAgent)
        agent.runtime_config = SimpleNamespace(capability_first_dataset_scoping_enabled=False)
        return agent

    def test_gpu_serving_topic_rejected_pre_debate(self):
        agent = self._agent()
        topic = {
            "title": "Single-seed variance in speculative decoding on LLM serving benchmarks",
            "description": ("Measure kernel scheduling variance and memory allocation "
                            "effects across LLM serving benchmark runs with vLLM."),
            "feasibility": 8,
            "dataset_plan": "bundled_synthetic",
        }
        feas = agent.feasibility_filter(topic)
        assert feas["ok"] is False
        assert any("GPU" in r for r in feas["reasons"])

    def test_accelerator_subject_as_cpu_simulation_passes(self):
        """Subject-vs-method: studying accelerator systems via CPU simulation
        must NOT be rejected. Only the experimental method requiring a GPU
        is infeasible — simulation is the sanctioned CPU proxy."""
        agent = self._agent()
        topic = {
            "title": ("Discrete-event simulation of speculative decoding "
                      "draft-verification dynamics"),
            "description": ("Simulate draft acceptance and kernel scheduling effects "
                            "with synthetic timing models on CPU; compare simulated "
                            "memory allocation patterns across draft lengths."),
            "feasibility": 7,
            "dataset_plan": "bundled_synthetic",
        }
        feas = agent.feasibility_filter(topic)
        assert feas["ok"] is True, feas["reasons"]

    def test_llm_subject_without_gpu_method_passes(self):
        agent = self._agent()
        topic = {
            "title": "Bootstrap confidence reporting in LLM benchmark comparisons",
            "description": ("Statistical re-analysis of published LLM serving benchmark "
                            "numbers with bootstrap confidence intervals on CPU."),
            "feasibility": 7,
            "dataset_plan": "bundled_synthetic",
        }
        assert agent.feasibility_filter(topic)["ok"] is True

    def test_negated_gpu_disclaimer_not_rejected(self):
        agent = self._agent()
        topic = {
            "title": "CPU-only calibration study on synthetic data",
            "description": "Runs on CPU only and does not require a gpu cluster.",
            "feasibility": 7,
            "dataset_plan": "bundled_synthetic",
        }
        feas = agent.feasibility_filter(topic)
        assert feas["ok"] is True, feas["reasons"]

    def test_ordinary_cpu_topic_passes(self):
        agent = self._agent()
        topic = {
            "title": "Temperature scaling vs Platt scaling on CIFAR-10-C",
            "description": "Compare calibration with sklearn baselines on CPU.",
            "feasibility": 7,
            "dataset_plan": "bundled_synthetic",
        }
        assert agent.feasibility_filter(topic)["ok"] is True

    def test_admission_rejects_gpu_contract(self):
        from core.evidence_synthesis import validate_topic_admission

        contract = {
            "research_question": "Does X beat Y?",
            "hypothesis": "X beats Y because Z.",
            "falsification_condition": "If X <= Y, falsified.",
            "dependent_variables": ["accuracy"],
            "required_resources": {"cpu": True, "gpu": True, "max_runtime_seconds": 60},
            "minimum_viable_experiment": {
                "dataset": "bundled_synthetic", "models": ["a"],
                "metrics": ["accuracy"], "seeds": 3, "baseline": "b",
                "falsification_test": "t-test",
            },
        }
        result = validate_topic_admission(contract)
        assert result["admitted"] is False
        assert any("GPU" in e for e in result["errors"])

    def test_admission_rejects_thousand_run_sweep(self):
        from core.evidence_synthesis import validate_topic_admission

        contract = {
            "research_question": "Does seed count change variance?",
            "hypothesis": "Single-seed underestimates variance by >40% because nondeterminism compounds.",
            "falsification_condition": "If multi-seed variance is not larger, falsified.",
            "dependent_variables": ["cv of speedup"],
            "required_resources": {"cpu": True, "gpu": False, "max_runtime_seconds": 100},
            "minimum_viable_experiment": {
                "dataset": "bundled_synthetic", "models": ["a", "b", "c"],
                "conditions": ["w1", "w2", "w3"], "metrics": ["latency"],
                "seeds": 30, "baseline": "b", "falsification_test": "t-test",
            },
        }
        result = validate_topic_admission(contract)
        assert result["admitted"] is False
        assert any("measurement runs" in e for e in result["errors"])

    def test_admission_accepts_small_cpu_contract(self):
        from core.evidence_synthesis import validate_topic_admission

        contract = {
            "research_question": "Does X beat Y?",
            "hypothesis": "X beats Y because Z.",
            "falsification_condition": "If X <= Y, falsified.",
            "dependent_variables": ["accuracy"],
            "required_resources": {"cpu": True, "gpu": False, "max_runtime_seconds": 100},
            "minimum_viable_experiment": {
                "dataset": "bundled_synthetic", "models": ["a", "b"],
                "metrics": ["accuracy"], "seeds": 3, "baseline": "b",
                "falsification_test": "t-test",
            },
        }
        assert validate_topic_admission(contract)["admitted"] is True

    def test_manifest_wall_clock_tracks_env(self, monkeypatch):
        from core import capabilities as cap

        monkeypatch.setenv("SANDBOX_TIMEOUT_SEC", "320")
        assert cap.effective_wall_clock_seconds() == 320
        monkeypatch.setenv("SANDBOX_TIMEOUT_SEC", "not-a-number")
        assert cap.effective_wall_clock_seconds() == cap.SANDBOX_CAPABILITY_MANIFEST.max_wall_clock_seconds

    def test_manifest_gpu_method_requirement_rejected(self):
        from core.capabilities import check_plan_feasibility

        errors = check_plan_feasibility(
            {"methodology": "Train the model on 8 GPUs with CUDA."})
        assert any("GPU" in e for e in errors)

    def test_manifest_gpu_disclaimer_passes(self):
        from core.capabilities import check_plan_feasibility

        errors = check_plan_feasibility(
            {"methodology": ("Runs CPU-only and does not require a gpu. "
                             "Uses sklearn on bundled synthetic data.")})
        assert not any("GPU" in e for e in errors), errors

    def test_manifest_gpu_simulation_framing_passes(self):
        from core.capabilities import check_plan_feasibility

        errors = check_plan_feasibility(
            {"methodology": ("We simulate GPU kernel scheduling with a "
                             "discrete-event model on CPU.")})
        assert not any("GPU" in e for e in errors), errors

    def test_manifest_subject_only_passes(self):
        from core.capabilities import check_plan_feasibility

        errors = check_plan_feasibility(
            {"methodology": ("Statistical analysis of published LLM serving "
                             "benchmark latencies with bootstrap intervals.")})
        assert not any("GPU" in e for e in errors), errors

    def test_cpu_envelope_mentions_simulation_and_gpu_false(self):
        from agents.topic_hunter import _CPU_EXECUTION_ENVELOPE

        assert "SIMULATION" in _CPU_EXECUTION_ENVELOPE
        assert "gpu:false" in _CPU_EXECUTION_ENVELOPE
        assert "synthetic" in _CPU_EXECUTION_ENVELOPE.lower()

    def test_formalize_prompt_carries_cpu_rescope_rule(self):
        from agents import topic_hunter as th

        agent = th.TopicHunterAgent.__new__(th.TopicHunterAgent)
        agent.runtime_config = SimpleNamespace()
        candidate = {"title": "T", "description": "D", "rationale": "R",
                     "literature_evidence": []}
        phase1 = ('{"research_question": "Q", "hypothesis": "H", '
                  '"falsification_condition": "F", "dependent_variables": ["m"]}')
        phase2 = ('{"research_question": "Q", "hypothesis": "H", '
                  '"falsification_condition": "F", "dependent_variables": "m", '
                  '"independent_variables": "x-var", '
                  '"baselines": "b", "metrics": "m", '
                  '"minimum_viable_experiment": "bundled_synthetic"}')
        with patch.object(th, "call_llm",
                          side_effect=[phase1, phase2]) as mock_llm:
            # Capture prompts via the mock's call args instead.
            result = agent.formalize_hypothesis(candidate, {}, {})
            prompts = [c.args[0] if c.args else c.kwargs.get("prompt", "")
                       for c in mock_llm.call_args_list]
        assert result is not None
        assert any("CPU rescope rule" in p for p in prompts), \
            "phase-2 prompt must carry the CPU rescope rule"


# ---------------------------------------------------------------------------
# Fix 2d/3c: seed exclusion + frontier-first
# ---------------------------------------------------------------------------

class TestSeedDiversity:
    def _agent(self):
        from agents.topic_hunter import TopicHunterAgent

        agent = TopicHunterAgent.__new__(TopicHunterAgent)
        agent.runtime_config = SimpleNamespace(
            research_domain="machine_learning",
            llm_seed_generation_enabled=False,
            negative_result_seeding_enabled=False,
            frontier_seeding_enabled=False,
            seed_strategy_elo_enabled=False,
        )
        agent.context = None
        agent._attempted_seeds = set()
        return agent

    def test_exact_repeat_seed_excluded(self):
        agent = self._agent()
        seeds = agent._generate_dynamic_seeds(
            5, [], domain="machine_learning",
            exclude_phrases=["transformer efficiency benchmark comparison"])
        texts = [s["seed"] for s in seeds]
        assert "transformer efficiency benchmark comparison" not in texts

    def test_rephrase_of_excluded_seed_dropped(self):
        agent = self._agent()
        seeds = agent._generate_dynamic_seeds(
            5, [], domain="machine_learning",
            exclude_phrases=["transformer efficiency benchmark comparison"])
        for s in seeds:
            from core.utils import title_token_overlap

            assert title_token_overlap(
                s["seed"], "transformer efficiency benchmark comparison") < 0.60

    def test_frontier_terms_replace_generic_fallbacks(self):
        from agents.topic_hunter import TopicHunterAgent

        agent = TopicHunterAgent.__new__(TopicHunterAgent)
        agent.runtime_config = SimpleNamespace(
            research_domain="machine_learning",
            llm_seed_generation_enabled=False,
            negative_result_seeding_enabled=False,
            frontier_seeding_enabled=True,
            seed_strategy_elo_enabled=False,
            output_dir=".",
        )
        agent.context = None
        agent._attempted_seeds = set()
        frontier = [{"method": "traversal verification", "evaluation": "acceptance length"}]
        with patch.object(TopicHunterAgent, "_load_cached_frontier_terms",
                          return_value=frontier), \
             patch.object(TopicHunterAgent, "_harvest_frontier_terms",
                          return_value=frontier), \
             patch("agents.topic_hunter.CrossRunMemory") as MockCRM:
            MockCRM.return_value.get_prompt_context.return_value = []
            MockCRM.return_value.get_negative_result_lessons.return_value = []
            seeds = agent._generate_dynamic_seeds(5, [], domain="machine_learning")
        texts = [s["seed"] for s in seeds]
        assert any("traversal verification" in t for t in texts), texts
        assert "methodological gaps in evaluation" not in texts
        assert not any(t.endswith("open problems") for t in texts)


# ---------------------------------------------------------------------------
# Fix 3a: single formalize repair
# ---------------------------------------------------------------------------

class TestFormalizeSingleRepair:
    def test_phase2_repairs_once_then_gives_up(self):
        from agents import topic_hunter as th

        agent = th.TopicHunterAgent.__new__(th.TopicHunterAgent)
        agent.runtime_config = SimpleNamespace()
        candidate = {"title": "T", "description": "D", "rationale": "R",
                     "literature_evidence": []}
        phase1 = ('{"research_question": "Q", "hypothesis": "H", '
                  '"falsification_condition": "F", "dependent_variables": ["m"]}')
        repairs = []

        def fake_log(agent_name, action, details=None):
            if action == "formalize_hypothesis_repair_attempt":
                repairs.append(details)

        with patch.object(th, "call_llm",
                          side_effect=[phase1, "{}", "{}"]) as mock_llm, \
             patch.object(th, "log_agent_action", side_effect=fake_log):
            assert agent.formalize_hypothesis(candidate, {}, {}) is None
        assert mock_llm.call_count == 3, "phase1 + initial phase2 + exactly one repair"
        assert len(repairs) == 1


# ---------------------------------------------------------------------------
# Fix 4: terminal best-effort promotion
# ---------------------------------------------------------------------------

def _debate_result(topic: str, score: float, passed: bool = False):
    from agents.hypothesis_debate import DebateResult

    return DebateResult(
        topic=topic, proposer_argument="p", challenger_argument="c",
        moderator_decision="FAIL", score=score, passed=passed,
        reasoning="r", rounds=[], unresolved_objections=[],
        infra_failure=False,
    )


def _promotable_state(iteration: int):
    from core.state import initialize_state

    state = initialize_state()
    state["iteration"] = iteration
    state["topics"] = [{
        "title": "Near-miss topic",
        "description": "d",
        "structured_hypothesis": {"hypothesis": "h"},
    }]
    return state


class TestTerminalPromotion:
    def test_last_iteration_promotes_near_miss(self, monkeypatch):
        import core.workflow_nodes as wn
        from core.config import config as cfg

        state = _promotable_state(cfg.max_iterations - 1)

        class FakeDebate:
            runtime_config = SimpleNamespace(debate_pass_threshold=7.5)

            def conduct_with_repair(self, topic):
                return [_debate_result(topic["title"], 7.4)]

        class FakeHunter:
            def __init__(self, *_a):
                pass

            def pre_debate_self_critique(self, topic):
                return topic

        monkeypatch.setattr(wn, "HypothesisDebateSystem", FakeDebate)
        monkeypatch.setattr(wn, "TopicHunterAgent", FakeHunter)
        result = wn.hypothesis_debate_node(state)
        assert result["current_phase"] == "planning"
        # P3: near-miss promotes bounded exploratory any iteration (last-iteration
        # path labels best_effort_terminal; serial exploratory path labels via
        # debate record). Accept either provenance but require planning handoff.
        assert result["hypothesis_passed"] is True
        assert result["current_phase"] == "planning"

    def test_last_iteration_does_not_promote_far_miss(self, monkeypatch):
        import core.workflow_nodes as wn
        from core.config import config as cfg

        state = _promotable_state(cfg.max_iterations - 1)

        class FakeDebate:
            runtime_config = SimpleNamespace(debate_pass_threshold=7.5)

            def conduct_with_repair(self, topic):
                return [_debate_result(topic["title"], 3.0)]

        class FakeHunter:
            def __init__(self, *_a):
                pass

            def pre_debate_self_critique(self, topic):
                return topic

        monkeypatch.setattr(wn, "HypothesisDebateSystem", FakeDebate)
        monkeypatch.setattr(wn, "TopicHunterAgent", FakeHunter)
        result = wn.hypothesis_debate_node(state)
        assert result.get("should_reset") is True
        assert result["current_phase"] != "planning"

    def test_early_iteration_still_resets(self, monkeypatch):
        import core.workflow_nodes as wn

        state = _promotable_state(0)

        class FakeDebate:
            runtime_config = SimpleNamespace(debate_pass_threshold=7.5)

            def conduct_with_repair(self, topic):
                return [_debate_result(topic["title"], 7.4)]

        class FakeHunter:
            def __init__(self, *_a):
                pass

            def pre_debate_self_critique(self, topic):
                return topic

        monkeypatch.setattr(wn, "HypothesisDebateSystem", FakeDebate)
        monkeypatch.setattr(wn, "TopicHunterAgent", FakeHunter)
        result = wn.hypothesis_debate_node(state)
        # P3: near-miss (7.4 vs 7.5) now promotes bounded exploratory even on
        # early iterations instead of resetting with zero artifact.
        assert result["current_phase"] == "planning"
        assert result["hypothesis_passed"] is True
