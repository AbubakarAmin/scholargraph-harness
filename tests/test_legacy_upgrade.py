"""Legacy/shared gate upgrades for per-type awareness (R1/R2/R3/R6/R14).

Offline only. Pins the binding directive: shared gates accept
``experiments: []`` / empty outputs for non-empirical + field-waiver types
while experimental / comparative / cause_effect keep legacy-strict behavior
when ``paper_type`` is absent/None.
"""

import pytest

from core.evidence_gate import gate_engineering_outputs, validate_experiments
from core.verification import cross_section_numeric_consistency

NON_EMPIRICAL = ["analytical", "descriptive", "argumentative", "review"]
FIELD_WAIVER = ["survey", "case_study"]
WAIVER_TYPES = NON_EMPIRICAL + FIELD_WAIVER
STRICT_TYPES = ["experimental", "comparative", "cause_effect", "empirical"]


def _passing_referee(monkeypatch):
    monkeypatch.setattr(
        "agents.editor.final_manuscript_referee",
        lambda *args, **kwargs: {"passed": True, "findings": []},
    )


def _bare_editor(tmp_path):
    from agents.editor import EditorAgent

    editor = EditorAgent.__new__(EditorAgent)
    editor.output_dir = str(tmp_path / "out")
    editor.companion_dir = str(tmp_path / "companion")
    return editor


# --- R1: editor empty-output assembly -------------------------------------

def test_r1_non_empirical_empty_outputs_assemble(tmp_path, monkeypatch):
    _passing_referee(monkeypatch)
    editor = _bare_editor(tmp_path)
    plan = {"experiments": [], "plan_mode": "non_empirical", "paper_type": "analytical"}
    paper = editor.create_final_paper(
        {"title": "T", "description": "D"},
        {"Introduction": "analysis", "Limitations": "limits " * 20},
        plan,
        {},
        paper_type="analytical",
        thesis={"claim": "c"},
    )
    assert paper["engineer_outputs"] == {}


def test_r1_non_empirical_ignores_unplanned_stale_failures(tmp_path, monkeypatch):
    _passing_referee(monkeypatch)
    editor = _bare_editor(tmp_path)
    plan = {"experiments": [], "plan_mode": "non_empirical", "paper_type": "survey"}
    paper = editor.create_final_paper(
        {"title": "T", "description": "D"},
        {"Introduction": "analysis", "Limitations": "limits " * 20},
        plan,
        {"stale": {"success": False, "error": "leftover"}},
        paper_type="survey",
    )
    assert paper["engineer_outputs"]["stale"]["success"] is False


def test_r1_experimental_failed_outputs_still_terminal(tmp_path):
    editor = _bare_editor(tmp_path)
    plan = {"experiments": [{"name": "toy", "evaluation_metrics": ["accuracy"]}]}
    with pytest.raises(RuntimeError, match="failed experiments"):
        editor.create_final_paper(
            {"title": "T", "description": "D"},
            {"Results": "No result"},
            plan,
            {"toy": {"success": False, "error": "boom"}},
            paper_type="experimental",
        )


def test_r1_legacy_typeless_failed_outputs_still_terminal(tmp_path):
    # Default paper_type is experimental: legacy call shape stays strict.
    editor = _bare_editor(tmp_path)
    with pytest.raises(RuntimeError, match="failed experiments"):
        editor.create_final_paper(
            {"title": "T", "description": "D"},
            {"Results": "No result"},
            {"experiments": []},
            {"toy": {"success": False}},
        )


# --- R2/R3: empty-experiment acceptance per type --------------------------

@pytest.mark.parametrize("paper_type", WAIVER_TYPES)
def test_r2_r3_empty_experiments_accepted_per_waiver_type(paper_type):
    assert validate_experiments([], paper_type=paper_type) == []


@pytest.mark.parametrize("paper_type", STRICT_TYPES + [None])
def test_r2_r3_empty_experiments_strict_otherwise(paper_type):
    assert validate_experiments([], paper_type=paper_type) == [
        "plan.experiments must be a non-empty list"
    ]


def test_r2_r3_plan_embedded_type_and_mode_waive():
    assert validate_experiments([], plan={"paper_type": "descriptive"}) == []
    # Empirical requires a bounded MVE (d5b5a1db: waiving it produced an
    # empty paper); only the legacy non_empirical plan_mode still waives.
    assert validate_experiments([], plan={"paper_type": "empirical"}) == [
        "plan.experiments must be a non-empty list"
    ]
    assert validate_experiments([], plan={"paper_type": "survey"}) == []
    assert validate_experiments([], plan={"plan_mode": "non_empirical"}) == []
    # Strict types stay strict even with a plan attached.
    assert validate_experiments(
        [], paper_type="experimental", plan={"plan_mode": "bounded_fallback"}
    )


def test_r2_r3_legacy_nonempty_checks_preserved():
    assert validate_experiments([None])
    assert validate_experiments(["experiment"])
    assert validate_experiments(
        [{"name": "toy", "variants": ["bad"]}],
        paper_type="analytical",
    )
    missing_metrics = validate_experiments(
        [{"name": "toy"}], paper_type="experimental"
    )
    assert any("evaluation_metrics" in err for err in missing_metrics)
    dupes = validate_experiments(
        [
            {"name": "toy", "evaluation_metrics": ["accuracy"]},
            {"name": "toy", "evaluation_metrics": ["accuracy"]},
        ],
        paper_type="review",
    )
    assert any("duplicate" in err for err in dupes)


def test_r2_r3_gate_allows_non_empirical_empty_handoff():
    decision = gate_engineering_outputs(
        {"experiments": [], "plan_mode": "non_empirical", "paper_type": "analytical"},
        {},
        {},
    )
    assert decision["allowed"] and not decision["terminal"]


def test_r2_r3_gate_allows_field_waiver_empty_handoff_via_kwarg():
    decision = gate_engineering_outputs(
        {"experiments": []}, {}, {}, paper_type="survey"
    )
    assert decision["allowed"] and not decision["terminal"]


def test_r2_r3_gate_legacy_empty_still_terminal():
    decision = gate_engineering_outputs({"experiments": []}, {}, {})
    assert not decision["allowed"] and decision["terminal"]
    assert decision["reason_code"] == "invalid_experiment_plan"


def test_r2_r3_gate_comparative_schema_still_enforced():
    decision = gate_engineering_outputs(
        {
            "experiments": [
                {
                    "name": "c",
                    "evaluation_metrics": ["accuracy"],
                    "baselines": ["only_one"],
                }
            ],
            "paper_type": "comparative",
        },
        {},
        {},
    )
    assert not decision["allowed"] and decision["terminal"]
    assert decision["reason_code"] == "invalid_experiment_plan"


# --- R3 fallback: paper_type-respecting shapes ----------------------------

def test_r3_comparative_fallback_keeps_two_baselines():
    from core.workflow_nodes import _minimal_fallback_plan

    plan = _minimal_fallback_plan({"title": "t"}, "test", "comparative")
    baselines = (
        plan["experiments"][0].get("baselines")
        or plan["experiments"][0].get("baseline_comparison")
        or []
    )
    assert len(baselines) >= 2


def test_r3_non_empirical_fallback_shape_passes_gate():
    # Non-empirical fallbacks keep experiments:[] + non_empirical mode; the
    # shared gate must not terminal before the per-type artifact map.
    for paper_type in ("analytical", "survey"):
        plan = {
            "experiments": [],
            "plan_mode": "non_empirical",
            "paper_type": paper_type,
        }
        assert validate_experiments(
            plan["experiments"], paper_type=paper_type, plan=plan
        ) == []
        decision = gate_engineering_outputs(plan, {}, {})
        assert decision["allowed"]


# --- R6: skip-if-no-numbers (lives in verification; pinned here) ----------

def test_r6_no_quantitative_tokens_no_outputs_skips():
    sections = {"Introduction": "This work explores ideas conceptually."}
    result = cross_section_numeric_consistency(sections, {})
    assert result["passed"] and result["conflicts"] == []


def test_r6_quantitative_claims_without_outputs_skip():
    sections = {"Results": "Accuracy reached 0.9 with strong gains."}
    result = cross_section_numeric_consistency(sections, {})
    assert result["passed"] and result["conflicts"] == []


def test_r6_conflict_with_outputs_still_fails():
    outputs = {"exp1": {"aggregate_metrics": {"accuracy": {"mean": 0.8}}}}
    sections = {"Results": "The exp1 accuracy 0.2 was observed."}
    result = cross_section_numeric_consistency(sections, outputs)
    assert not result["passed"] and result["conflicts"]


def test_r6_unified_tolerance_boundary():
    outputs = {"exp1": {"aggregate_metrics": {"accuracy": {"mean": 100.0}}}}
    assert cross_section_numeric_consistency(
        {"Results": "accuracy 105"}, outputs
    )["passed"]
    assert not cross_section_numeric_consistency(
        {"Results": "accuracy 106"}, outputs
    )["passed"]


# --- R14: startup probe report never raises offline -----------------------

def test_r14_probe_report_never_raises_offline():
    from main import report_provider_reachability

    report = report_provider_reachability()
    assert isinstance(report, dict)
    assert {"arxiv", "openalex", "semantic_scholar", "llm", "embedding"} <= set(report)
    assert all(isinstance(value, bool) for value in report.values())


def test_r14_probe_report_survives_gateway_failure(monkeypatch):
    import main

    def _boom(**kwargs):
        raise RuntimeError("no gateway offline")

    monkeypatch.setattr("core.api_gateway.get_gateway", _boom)
    assert main.report_provider_reachability() == {}


def test_fallback_plan_respects_non_empirical_types():
    from core.workflow_nodes import _minimal_fallback_plan

    for ptype in ("analytical", "descriptive", "argumentative", "review", "survey", "case_study"):
        plan = _minimal_fallback_plan({"title": "T"}, "reason", ptype)
        assert plan["experiments"] == [], ptype
        assert plan["plan_mode"] == "non_empirical", ptype

    exp = _minimal_fallback_plan({"title": "T"}, "reason", "experimental")
    assert len(exp["experiments"]) == 1  # experimental fallback still executable

    # d5b5a1db: empirical fallback must keep the bounded CPU MVE, never [].
    for ptype in ("empirical", "comparative", "cause_effect"):
        plan = _minimal_fallback_plan({"title": "T"}, "reason", ptype)
        assert len(plan["experiments"]) >= 1, ptype
        assert plan["plan_mode"] != "non_empirical", ptype
    legacy = _minimal_fallback_plan({"title": "T"}, "reason")
    assert len(legacy["experiments"]) == 1  # None preserves legacy shape
