"""Regression: PlannerAgent.revise_plan() must respect capability manifest."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from agents.planner import PlannerAgent
from core.capabilities import SANDBOX_CAPABILITY_MANIFEST
from core.contracts import Plan, RevisionRequest


def _base_plan(manifest: dict | None = None) -> Plan:
    if manifest is None:
        manifest = SANDBOX_CAPABILITY_MANIFEST.as_dict()
    return {
        "title": "Test Plan",
        "methodology": "Controlled comparison on synthetic data",
        "experiments": [
            {
                "name": "baseline_comparison",
                "baselines": ["logistic_regression"],
                "evaluation_metrics": ["accuracy"],
                "falsifiable_prediction": "Method > baseline",
                "statistical_test": "Welch t-test p<0.05",
                "variants": [],
                "claimed_components": ["main_model"],
            }
        ],
        "contributions": [
            {
                "claim": "Better accuracy",
                "falsifiable_prediction": "Accuracy > 0.8",
                "statistical_test": "t-test",
                "components": ["main_model"],
            }
        ],
        "capability_manifest": manifest,
        "revision_history": [],
    }


def _revision_request(reason: str, detail: str = "") -> RevisionRequest:
    return {"reason": reason, "experiment": "baseline_comparison", "detail": detail}


class TestRevisePlanManifestAware:
    """revise_plan() must include manifest in prompt and reject violations."""

    def _mock_llm_reviser(self, violate: bool = False):
        """Return a mock call_llm that produces compliant or violating plans."""
        manifest = SANDBOX_CAPABILITY_MANIFEST.as_dict()

        def _side_effect(prompt, temperature=0.4, tier="strong", **kwargs):
            if violate:
                return json.dumps({
                    "methodology": "Use GPU clusters with CUDA for training",
                    "experiments": [
                        {
                            "name": "gpu_experiment",
                            "baselines": ["deep_neural_network"],
                            "evaluation_metrics": ["accuracy"],
                            "falsifiable_prediction": "GPU model > CPU",
                            "statistical_test": "t-test",
                            "variants": [],
                            "claimed_components": ["gpu_model"],
                        }
                    ],
                    "contributions": [
                        {
                            "claim": "GPU speedup",
                            "falsifiable_prediction": "2x faster",
                            "statistical_test": "t-test",
                            "components": ["gpu_model"],
                        }
                    ],
                    "revision_notes": "switched to GPU",
                })
            return json.dumps({
                "methodology": "CPU-only comparison on synthetic data",
                "experiments": [
                    {
                        "name": "baseline_comparison",
                        "baselines": ["logistic_regression"],
                        "evaluation_metrics": ["accuracy"],
                        "falsifiable_prediction": "Method > baseline",
                        "statistical_test": "Welch t-test p<0.05",
                        "variants": [],
                        "claimed_components": ["main_model"],
                    }
                ],
                "contributions": [
                    {
                        "claim": "Better accuracy",
                        "falsifiable_prediction": "Accuracy > 0.8",
                        "statistical_test": "t-test",
                        "components": ["main_model"],
                    }
                ],
                "revision_notes": "kept compliant",
            })

        return _side_effect

    @patch("agents.planner.call_llm")
    def test_manifest_included_in_prompt(self, mock_llm):
        """The revision prompt must mention the active capability manifest."""
        mock_llm.side_effect = self._mock_llm_reviser(violate=False)
        agent = PlannerAgent()
        plan = _base_plan()
        req = _revision_request("sandbox_blocked_required_api", "import blocked")

        agent.revise_plan(plan, req)

        first_call_prompt = mock_llm.call_args_list[0][0][0]
        assert "gpu_available=False" in first_call_prompt
        assert "outbound_network=False" in first_call_prompt

    @patch("agents.planner.call_llm")
    def test_compliant_revision_accepted(self, mock_llm):
        """A revision that respects the manifest passes feasibility."""
        mock_llm.side_effect = self._mock_llm_reviser(violate=False)
        agent = PlannerAgent()
        plan = _base_plan()
        req = _revision_request("sandbox_blocked_required_api", "import blocked")

        result = agent.revise_plan(plan, req)
        assert result is not None
        assert result.get("capability_manifest") is not None

    @patch("agents.planner.call_llm")
    def test_violating_revision_reprompts_then_fails(self, mock_llm):
        """A revision proposing GPU/network must be re-prompted, then auto-fixed.

        Production policy is repair-over-terminal: the deterministic auto-fix
        sanitizes manifest violations so the run continues with a compliant
        plan (the raise-only path fires solely when auto-fix cannot repair).
        """
        mock_llm.side_effect = self._mock_llm_reviser(violate=True)
        agent = PlannerAgent()
        plan = _base_plan()
        req = _revision_request("sandbox_blocked_required_api", "import blocked")

        result = agent.revise_plan(plan, req)
        # Machine contract: the returned plan must clear the same feasibility
        # gate production enforces (names may echo the mock's labels, but no
        # live GPU/network/download requirement may survive).
        from core.capabilities import check_plan_feasibility

        assert check_plan_feasibility(result) == []
        # First call = initial revision, second call = fix prompt, third call = (potential) retry
        assert mock_llm.call_count >= 2

    @patch("agents.planner.call_llm")
    def test_reprompt_includes_violation_details(self, mock_llm):
        """The fix-up re-prompt must list the specific violated fields."""
        mock_llm.side_effect = self._mock_llm_reviser(violate=True)
        agent = PlannerAgent()
        plan = _base_plan()
        req = _revision_request("max_attempts_exhausted", "code failed")

        agent.revise_plan(plan, req)

        # The second call should be the fix prompt
        fix_prompt = mock_llm.call_args_list[1][0][0]
        assert "violated" in fix_prompt.lower() or "manifest" in fix_prompt.lower()

    @pytest.mark.parametrize(
        "reason",
        ["sandbox_blocked_required_api", "max_attempts_exhausted", "contract_drift"],
    )
    @patch("agents.planner.call_llm")
    def test_violation_rejected_across_revision_reasons(self, mock_llm, reason):
        """GPU/network violations are sanitized regardless of revision reason."""
        mock_llm.side_effect = self._mock_llm_reviser(violate=True)
        agent = PlannerAgent()
        plan = _base_plan()
        req = _revision_request(reason, "test detail")

        result = agent.revise_plan(plan, req)
        from core.capabilities import check_plan_feasibility

        assert check_plan_feasibility(result) == []

    @patch("agents.planner.call_llm")
    def test_no_gpu_or_network_in_output_plan(self, mock_llm):
        """When revision is compliant, the plan content must not contain GPU/network tokens."""
        mock_llm.side_effect = self._mock_llm_reviser(violate=False)
        agent = PlannerAgent()
        plan = _base_plan()
        req = _revision_request("sandbox_blocked_required_api", "import blocked")

        result = agent.revise_plan(plan, req)
        # Exclude capability_manifest — its key names (gpu_available, outbound_network)
        # contain the very tokens we're checking for; we only care about the plan content.
        content = {k: v for k, v in result.items() if k != "capability_manifest"}
        plan_text = json.dumps(content).lower()
        assert "cuda" not in plan_text
        assert "gpu" not in plan_text
