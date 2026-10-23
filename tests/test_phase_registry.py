"""M0: phase registry, router, refusals, migration, budgets, packs (15 tests)."""

from __future__ import annotations

import pytest

from core.phase_registry import (
    ALL_PAPER_TYPES,
    PIPELINES,
    REFUSALS,
    is_phase_required,
    make_refusal,
    resolve_recipe,
    validate_pack,
)
from core.state import migrate_state_v1_to_v2
from core.pricing import budget_exhausted


@pytest.mark.parametrize("paper_type", sorted(ALL_PAPER_TYPES))
def test_recipe_resolves_for_every_type(paper_type):
    recipe, skipped, evidence, _manifest = resolve_recipe(paper_type)
    assert recipe == PIPELINES[paper_type]
    assert recipe[0] == "intake"
    assert recipe[-1] == "editing"
    assert set(evidence) == {"needs_experiment", "needs_prisma", "needs_pack", "needs_audit", "needs_thesis"}
    assert all("phase" in s and "reason" in s for s in skipped)


def test_auto_tiebreak_logged_and_deterministic(caplog):
    with caplog.at_level("INFO"):
        recipe, _, _, _ = resolve_recipe("auto", {"response_table": {"rows": 10}})
    assert recipe == PIPELINES["survey"]
    assert any("auto-tiebreak" in r.message for r in caplog.records)


def test_every_refusal_literal_reachable():
    assert len(REFUSALS) == 11
    for reason in REFUSALS:
        refusal = make_refusal(reason, provided="p", alternative="a", how_to_proceed="h")
        assert refusal["reason"] == reason
        assert refusal["provided"] == "p"
    with pytest.raises(ValueError):
        make_refusal("not_a_reason")


def test_v1_state_migrates_with_spine():
    migrated = migrate_state_v1_to_v2({})
    assert migrated["checkpoint_schema_version"] == 2
    # P1 deferred binding: typeless v1 states defer routing until evidence.
    assert migrated["paper_type"] == "auto-deferred"
    assert migrated["pipeline_recipe"] == []
    assert migrated["readiness_level"] == "L1"
    # Additive: existing values preserved.
    migrated2 = migrate_state_v1_to_v2({"paper_type": "review", "pipeline_recipe": ["x"]})
    assert migrated2["paper_type"] == "review"
    assert migrated2["pipeline_recipe"] == ["x"]
    # Explicit legacy experimental keeps its spine.
    migrated3 = migrate_state_v1_to_v2({"paper_type": "experimental"})
    assert migrated3["paper_type"] == "experimental"
    assert migrated3["pipeline_recipe"][0] == "intake"


def test_budget_helper_trips_at_limits():
    # Owner-generous caps (2026-10-04): 2000 calls / 8M tokens / $500 / 6h wall.
    exhausted, reason = budget_exhausted(llm_calls=2000, tokens_total=0, usd_spent=0.0, wall_sec=0.0)
    assert exhausted and reason == "budget_exhausted"
    exhausted, _ = budget_exhausted(llm_calls=0, tokens_total=8000000, usd_spent=0.0, wall_sec=0.0)
    assert exhausted
    exhausted, _ = budget_exhausted(llm_calls=0, tokens_total=0, usd_spent=500.0, wall_sec=0.0)
    assert exhausted
    exhausted, _ = budget_exhausted(llm_calls=1, tokens_total=1, usd_spent=0.01, wall_sec=10.0)
    assert not exhausted


def test_bad_pack_returns_named_reason():
    ok, refusal = validate_pack("survey", {})
    assert not ok and refusal["reason"] == "needs_response_table"
    ok, refusal = validate_pack("case_study", {})
    assert not ok and refusal["reason"] == "needs_case_pack"
    ok, refusal = validate_pack("case_study", {"case_pack": {"entries": [{"type": "doc"}]}})
    assert not ok and refusal["reason"] == "needs_pack_invalid"
    ok, _ = validate_pack("experimental", {})
    assert ok


def test_is_phase_required_and_auto_rejected():
    assert is_phase_required("experimental", "engineering") is True
    assert is_phase_required("analytical", "engineering") is False
    assert is_phase_required("review", "systematic_review") is True
    with pytest.raises(ValueError):
        is_phase_required("auto", "engineering")
    with pytest.raises(ValueError):
        resolve_recipe("not_a_type")
