# tests/test_power_and_rescope.py
"""Unit tests for power‑analysis integration and capability‑rescope traceability.
These tests verify that:
1. `preregister_power` computes a sensible sample‑size requirement.
2. `PlannerAgent._apply_capability_rescope` records the expected traceability fields
   in the plan when feasibility errors are present.
"""

import pytest
from core.verification import preregister_power
from agents.planner import PlannerAgent


def test_preregister_power_computation():
    effect = 0.5
    result = preregister_power(effect, alpha=0.05, target_power=0.8)
    assert isinstance(result, dict)
    assert result["planned_effect_size"] == float(effect)
    assert result["alpha"] == 0.05
    assert result["target_power"] == 0.8
    assert isinstance(result["required_n_per_group"], int)
    assert result["required_n_per_group"] > 0


def test_apply_capability_rescope_adds_traceability_fields():
    plan = {"title": "Original Study", "contributions": ["Improve accuracy"]}
    topic = {"title": "Original Study", "dataset_plan": "external_dataset"}
    reasons = ["GPU unavailable", "Outbound download prevented"]
    PlannerAgent._apply_capability_rescope(plan, topic, reasons)
    cap = plan.get("capability_rescope")
    assert isinstance(cap, dict)
    expected_keys = {"from", "original_title", "to", "original_dataset_requirement", "replacement_dataset", "reason", "replacement_dataset_policy"}
    assert expected_keys.issubset(cap.keys())
    assert cap["original_title"] == "Original Study"
    assert "bounded local synthetic" in cap["to"].lower()
    for r in reasons:
        assert r in cap["reason"]
