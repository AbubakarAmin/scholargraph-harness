"""Regression tests for run 0dfc1c58 (2026-10-08).

Root cause: the OpenAI-compatible LLM proxy exhausted its free quota and
returned an HTTP-200 plain-English top-up notice
("Sorry, to prevent abuse of free resources ... topup", 188 chars), which
every caller accepted as valid output:
- debate proposer stored it as the argument (score 0.0 FAIL),
- engineer compiled it as Python -> SyntaxError line 1, 4 attempts x 2
  experiments x 3 engineering cycles (48 wasted calls, 20 plan revisions).

Secondary: the run domain "software engineer" (no -ing) missed the SE
taxonomy map, and Elo-sorted kind_bias seeds pushed the 3 SE LLM seeds out
of the batch, so 5 generic ML templates were hunted for an SE run.

All tests offline (mocked LLM / gateway / memory).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.utils import (
    is_degenerate_llm_output,
    is_provider_quota_text,
    looks_like_python_code,
)

QUOTA_TEXT = (
    "Sorry, to prevent abuse of free resources, accounts that have not been "
    "recharged can only try 10 times. You can increase the free quota after "
    "recharging; https://console.aihubmix.com/topup"
)

REAL_CODE = '''
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X = np.random.randn(200, 4)
y = (X[:, 0] > 0).astype(int)
m = LogisticRegression(max_iter=200)
m.fit(X, y)
print(json.dumps({"metrics": {"accuracy": float(accuracy_score(y, m.predict(X)))}}))
'''


class TestQuotaDetection:
    def test_exact_run_payload_flagged(self):
        assert is_provider_quota_text(QUOTA_TEXT) is True
        assert len(QUOTA_TEXT) == 188  # the "188 chars" in the run log

    def test_quota_variants_flagged(self):
        assert is_provider_quota_text("Error: insufficient credits, please topup your account")
        assert is_provider_quota_text("Payment Required: quota exceeded for this model")
        assert is_provider_quota_text("accounts that have not been recharged can only try 10 times")

    def test_real_outputs_not_flagged(self):
        assert is_provider_quota_text(REAL_CODE) is False
        assert is_provider_quota_text("The hypothesis predicts a 3pp gain because ...") is False
        assert is_provider_quota_text("") is False
        assert is_provider_quota_text(None) is False

    def test_degenerate_rejects_quota_despite_length(self):
        # The 0dfc1c58 proposer bug: 188 chars passed min_chars=30.
        assert is_degenerate_llm_output(QUOTA_TEXT, min_chars=30) is True
        assert is_degenerate_llm_output(QUOTA_TEXT) is True

    def test_degenerate_still_accepts_prose(self):
        assert is_degenerate_llm_output("A" * 200) is False

    def test_looks_like_python(self):
        assert looks_like_python_code(REAL_CODE) is True
        assert looks_like_python_code(QUOTA_TEXT) is False
        assert looks_like_python_code("Sorry, no code here just words") is False


class TestLLMClientQuotaFailsFast:
    def test_chat_raises_llm_unavailable_on_quota(self, monkeypatch):
        from core import llm as llm_mod

        client = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
        client.provider = "openai_compatible"

        class _FakeGateway:
            def is_available(self, _provider):
                return True

            def request(self, _provider, _fn, *args, **kwargs):
                return QUOTA_TEXT

        monkeypatch.setattr(llm_mod, "get_gateway", lambda: _FakeGateway())
        with pytest.raises(llm_mod.LLMUnavailable):
            client.chat("Generate python code", model="dummy-model")


class TestEngineerNonCodeGuards:
    def _agent(self):
        from agents.engineer import EngineerAgent

        agent = EngineerAgent.__new__(EngineerAgent)
        agent.context = SimpleNamespace(
            config=SimpleNamespace(
                engineer_max_attempts=4,
                experiment_seeds=3,
                engineering_wall_clock_sec=60,
                output_dir="/tmp/eng_test",
                raw_results_dir="/tmp/eng_test",
            ),
            research_db=None,
            memory=None,
        )
        agent._plan_revision_requests = []
        agent._broker = None
        agent._manifest = {}
        agent._progress = lambda *a, **k: None  # noqa: E731
        agent._revisions = []
        agent.request_plan_revision = (  # noqa: E731
            lambda reason, exp, detail="": agent._revisions.append(reason) or {"reason": reason}
        )
        return agent

    def test_is_non_code_payload(self):
        from agents.engineer import EngineerAgent

        assert EngineerAgent._is_non_code_payload(QUOTA_TEXT) is True
        assert EngineerAgent._is_non_code_payload(REAL_CODE) is False
        assert EngineerAgent._is_non_code_payload("") is False

    def test_identical_failure_abort(self):
        from agents.engineer import EngineerAgent

        err = "SyntaxError: invalid syntax (<unknown>, line 1)"
        assert EngineerAgent._identical_failure_abort(
            [{"reason": err}, {"reason": err}]) is True  # dead-model fast path
        assert EngineerAgent._identical_failure_abort(
            [{"reason": err}, {"reason": "NameError: foo"}]) is False
        assert EngineerAgent._identical_failure_abort([{"reason": err}]) is False
        assert EngineerAgent._identical_failure_abort([]) is False

    def test_error_hint_for_line1_syntax(self):
        from agents.engineer import EngineerAgent

        hint = EngineerAgent._error_category_hint(
            "SyntaxError: invalid syntax (<unknown>, line 1)")
        assert hint  # was '' in 0dfc1c58, causing blind re-asks
        assert "quota" in hint.lower() or "line 1" in hint.lower()

    def test_run_experiment_aborts_fast_on_quota(self):
        agent = self._agent()
        agent._generate_experiment_code = lambda *a, **k: QUOTA_TEXT
        exp = {"name": "Baseline Comparison", "contract_hash": "abc"}
        out = agent.run_experiment(exp)
        assert out["success"] is False
        assert out["failure_kind"] == "llm_unavailable"
        # Fast abort: 2 attempts, not 4; single repeated-failure revision.
        assert len(out["decision_log"]) == 2
        assert agent._revisions == ["repeated_identical_failure"]


class TestSEDomainHandling:
    def test_software_engineer_variant_maps_to_se(self):
        from agents.topic_hunter import _subcategory_for_domain

        assert _subcategory_for_domain("software engineer") == "software_engineering"
        assert _subcategory_for_domain("software engineering") == "software_engineering"

    def test_domain_gate_accepts_variant(self):
        from agents.topic_hunter import _seed_passes_domain_gate

        assert _seed_passes_domain_gate(
            "coverage-guided fuzzing branch distance benchmark comparison",
            "software engineer") is True
        assert _seed_passes_domain_gate(
            "hierarchical world models visual planning", "software engineer") is False

    def test_subcategory_prefers_se_over_kind(self):
        from tests.test_topic_hunter_features_8_15 import _make_agent

        agent = _make_agent()
        subcat = agent._resolve_subcategory_for(
            "graph neural inductive benchmark comparison",
            active_kind="vision", domain="software engineer")
        assert subcat == "software_engineering"

    def test_llm_seeds_reserved_in_batch(self, monkeypatch):
        from tests.test_topic_hunter_features_8_15 import _make_agent

        agent = _make_agent()
        agent.runtime_config.research_domain = "software engineer"
        agent.runtime_config.seed_strategy_elo_enabled = True
        agent._selection_counter = 1  # avoid forced-exploration branch

        se_llm = [
            {"seed": "coverage-guided fuzzing branch distance benchmark comparison",
             "strategy": "llm_diverse"},
            {"seed": "conformal test prioritization for regression testing",
             "strategy": "llm_diverse"},
            {"seed": "program repair patch correctness benchmark comparison",
             "strategy": "llm_diverse"},
        ]
        monkeypatch.setattr(agent, "_generate_llm_seeds", lambda *a, **k: list(se_llm))
        monkeypatch.setattr(agent, "_load_cached_frontier_terms", lambda *a, **k: [])
        monkeypatch.setattr(agent, "_harvest_frontier_terms", lambda *a, **k: [])

        class _Elo:
            def get(self, key, default=None):
                if "kind_bias" in key:
                    return 1700.0
                if "llm_diverse" in key:
                    return 1200.0
                return 1500.0

            def under_observed_kinds(self, kinds):
                return []

        monkeypatch.setattr("agents.topic_hunter.EloStore", lambda **kw: _Elo())

        seeds = agent._generate_dynamic_seeds(
            n_seeds=5, cross_run_context=[], domain="software engineer")
        strategies = [s["strategy"] for s in seeds]
        assert "llm_diverse" in strategies, f"LLM seeds sliced out again: {seeds}"
        from agents.topic_hunter import _seed_passes_domain_gate

        assert all(_seed_passes_domain_gate(s["seed"], "software engineer") for s in seeds)
