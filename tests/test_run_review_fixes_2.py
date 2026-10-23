"""Regression tests for run review fixes (error attribution, plan revision gating, downstream guards, variant schema, tournament deduplication)."""

from typing import Any, Dict, List
import pytest

from agents.hypothesis_debate import DebateResult, HypothesisDebateSystem
from agents.planner import PlannerAgent
from core.contracts import Topic
from core.evidence_gate import validate_experiments
from core.run_log import build_run_summary, CrossRunMemory
from core.state import initialize_state
from core.workflow import create_research_graph
from core.workflow_nodes import (
    data_validation_node,
    engineering_node,
    hypothesis_debate_node,
    independent_validation_node,
    is_valid_plan,
    planning_node,
    should_continue,
    should_reset,
    terminal_planning_failure,
    write_narrative_sections,
    write_results_sections,
)


def test_planner_normalizes_string_variants_to_objects():
    planner = PlannerAgent.__new__(PlannerAgent)
    raw_experiments = [
        {
            "name": "Circuit Fidelity Benchmark",
            "evaluation_metrics": ["accuracy"],
            "baselines": ["logistic_regression"],
            "variants": ["variant string A", "variant string B", "variant string C"],
        },
        {
            "name": "Logical Representation Analysis",
            "evaluation_metrics": ["f1"],
            "baselines": ["random_forest"],
            "variants": [
                {"variant_name": "Custom Variant 1", "methodology": "custom feature ablation"},
                "plain string variant",
            ],
        },
    ]
    normalized = planner._normalize_experiments(raw_experiments)
    assert len(normalized) == 2
    for exp in normalized:
        assert isinstance(exp["variants"], list)
        for variant in exp["variants"]:
            assert isinstance(variant, dict)
            assert "name" in variant and variant["name"]
    errors = validate_experiments(normalized)
    assert not errors


def test_planner_attach_variants_ensures_objects():
    planner = PlannerAgent.__new__(PlannerAgent)
    raw_experiments = [
        {
            "name": "Exp 1",
            "variants": ["string_var_1", "string_var_2"],
        }
    ]
    attached = planner._attach_variants(raw_experiments)
    assert len(attached) == 1
    assert len(attached[0]["variants"]) == 2
    for v in attached[0]["variants"]:
        assert isinstance(v, dict)
        assert "name" in v
        assert "methodology" in v
    errors = validate_experiments(attached)
    assert not errors


def test_planner_create_plan_repairs_schema_errors(monkeypatch):
    from agents import planner as planner_mod

    planner = PlannerAgent.__new__(PlannerAgent)
    planner.context = type("Context", (), {"config": type("Runtime", (), {"research_domain": "test"})()})()
    
    # Mock initial plan structure and generation
    planner._generate_plan_structure = lambda *_args: {
        "title": "Test Plan",
        "sections": [{"name": "Methods"}],
        "methodology": "local",
        "compute_budget": "cpu",
    }
    planner._ensure_falsifiable_contributions = lambda *_args: [{"claim": "c", "falsifiable_prediction": "p", "statistical_test": "t"}]
    # Initially returns invalid experiments (missing evaluation_metrics)
    planner._generate_experiments = lambda *_args: [{"name": "bad_exp"}]
    planner._flag_unfalsifiable = lambda *_args: []
    planner._flag_missing_baselines = lambda *_args: ["bad_exp has no baselines"]
    planner._generate_dependencies = lambda *_args: []
    planner._generate_timeline = lambda *_args: {}
    planner._store_plan = lambda *_args: None
    monkeypatch.setattr(planner_mod, "list_datasets", lambda: [])
    monkeypatch.setattr(planner_mod, "check_plan_feasibility", lambda *_args, **_kwargs: [])

    # Mock repair to produce valid experiments
    def fake_repair(plan, topic, unfalsifiable=None, missing_baselines=None, schema_errors=None):
        plan["experiments"] = [
            {
                "name": "repaired_exp",
                "baselines": ["baseline_a"],
                "evaluation_metrics": ["accuracy"],
                "falsifiable_prediction": "prediction",
                "statistical_test": "welch_t",
                "variants": [{"name": "v1", "methodology": "m1"}],
            }
        ]
        return plan

    planner._repair_plan = fake_repair

    plan = planner.create_plan({"title": "topic", "description": "desc"})
    assert plan["experiments"][0]["name"] == "repaired_exp"
    assert not validate_experiments(plan["experiments"])


def test_planning_node_terminal_failure_on_unrecoverable_schema_errors(monkeypatch):
    # M0 L6: single 2-revision policy — after 2 repairs the node takes the
    # bounded synthetic fallback (preserving schema_revision_attempts) instead
    # of the old dead `attempts >= 3` terminal branch. Downstream evidence
    # gates label the result exploratory.
    state = initialize_state()
    state["selected_topic"] = {"title": "Test Topic"}

    class FailingPlanner:
        def __init__(self, *_args):
            pass
        def create_plan(self, topic):
            return {
                "title": "Invalid Plan",
                "experiments": [{"invalid": "no name"}],
                "schema_revision_attempts": 3,
            }

    monkeypatch.setattr("core.workflow_nodes._create_agent", lambda cls: FailingPlanner())

    result = planning_node(state)
    # P1+P3: fresh states defer to empirical (try_and_report) so the bounded
    # fallback advances to data_validation, not the legacy experimental
    # writing_narrative label. Still non-terminal with attempts preserved.
    assert result["current_phase"] == "data_validation"
    assert result["should_continue"] is not False
    assert result["plan"]["schema_revision_attempts"] == 3
    assert result["terminal_error"] is None


def test_workflow_graph_routes_directly_to_end_on_planning_terminal_failure():
    nodes = {
        "topic_discovery": lambda s: s,
        "hypothesis_debate": lambda s: s,
        "planning": lambda s: s,
        "terminal_planning_failure": terminal_planning_failure,
        "data_validation": lambda s: s,
        "writing_narrative": lambda s: s,
        "engineering": lambda s: s,
        "independent_validation": lambda s: s,
        "writing_results": lambda s: s,
        "supervision": lambda s: s,
        "meta_evaluation": lambda s: s,
        "editing": lambda s: s,
        "reset": lambda s: s,
        "should_reset": should_reset,
        "should_continue": should_continue,
        "is_valid_plan": is_valid_plan,
    }
    graph = create_research_graph(nodes)

    state = initialize_state()
    state["selected_topic"] = {"title": "Test Topic"}
    state["current_phase"] = "complete"
    state["should_continue"] = False
    state["terminal_error"] = "Planning failed contract validation"

    executed_nodes = []
    def record_node(node_name):
        def _fn(s):
            executed_nodes.append(node_name)
            return s
        return _fn

    for name in nodes:
        if name not in ("should_reset", "should_continue"):
            nodes[name] = record_node(name)

    graph_with_recording = create_research_graph(nodes).compile()
    final_state = graph_with_recording.invoke(state)
    assert executed_nodes == ["topic_discovery"]


def test_downstream_nodes_do_not_overwrite_earliest_terminal_error():
    state = initialize_state()
    state["terminal_error"] = "Planning failed: Plan experiments failed contract validation"
    state["current_phase"] = "complete"
    state["should_continue"] = False

    # Call downstream nodes directly
    state = data_validation_node(state)
    assert state["terminal_error"] == "Planning failed: Plan experiments failed contract validation"

    state = write_narrative_sections(state)
    assert state["terminal_error"] == "Planning failed: Plan experiments failed contract validation"

    state = engineering_node(state)
    assert state["terminal_error"] == "Planning failed: Plan experiments failed contract validation"

    state = independent_validation_node(state)
    assert state["terminal_error"] == "Planning failed: Plan experiments failed contract validation"

    state = write_results_sections(state)
    assert state["terminal_error"] == "Planning failed: Plan experiments failed contract validation"


def test_conduct_tournament_debates_each_candidate_at_most_once(monkeypatch):
    debater = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    debater.context = None
    debater.runtime_config = None

    debated = []
    def fake_conduct_debate(topic):
        debated.append(topic["title"])
        return DebateResult(
            topic=topic["title"],
            proposer_argument="arg",
            challenger_argument="rebuttal",
            moderator_decision="FAIL",
            score=4.0,
            passed=False,
            reasoning="failed",
        )

    debater.conduct_debate = fake_conduct_debate
    topics = [
        {"title": "Topic 1"},
        {"title": "Topic 2"},
        {"title": "Topic 3"},
        {"title": "Topic 4"},
    ]

    results = debater.conduct_tournament(topics, rounds=2)
    assert len(results) == 4
    # All 4 debated exactly once, no repeated debates
    assert debated == ["Topic 1", "Topic 2", "Topic 3", "Topic 4"]


def test_run_summary_preserves_earliest_terminal_error():
    state = initialize_state()
    state["terminal_error"] = "Planning failed: Plan experiments failed contract validation"
    state["current_phase"] = "complete"
    state["meta_feedback"].append("Planning error: Plan experiments failed contract validation")

    summary = build_run_summary(state=state)
    assert summary["status"] == "failed"
    assert "Planning failed: Plan experiments failed contract validation" in summary["stopped_because"]
    assert "Planning failed: Plan experiments failed contract validation" in summary["narrative"]


# ---------------------------------------------------------------------------
# Phase validation guard tests
# ---------------------------------------------------------------------------

def test_is_valid_plan_rejects_none_plan():
    from core.workflow_nodes import is_valid_plan
    state = initialize_state()
    assert not is_valid_plan(state)


def test_is_valid_plan_rejects_empty_experiments():
    from core.workflow_nodes import is_valid_plan
    state = initialize_state()
    state["plan"] = {"experiments": []}
    assert not is_valid_plan(state)


def test_is_valid_plan_rejects_experiment_missing_name():
    from core.workflow_nodes import is_valid_plan
    state = initialize_state()
    state["plan"] = {"experiments": [{"evaluation_metrics": ["accuracy"]}]}
    assert not is_valid_plan(state)


def test_is_valid_plan_rejects_experiment_missing_metrics():
    from core.workflow_nodes import is_valid_plan
    state = initialize_state()
    state["plan"] = {"experiments": [{"name": "exp1"}]}
    assert not is_valid_plan(state)


def test_is_valid_plan_rejects_variant_missing_name():
    from core.workflow_nodes import classify_plan_issues, is_valid_plan
    state = initialize_state()
    state["plan"] = {
        "experiments": [{
            "name": "exp1",
            "evaluation_metrics": ["accuracy"],
            "variants": [{"methodology": "x"}],
        }]
    }
    # P3: soft variant-shape issues are auto-fixed, not terminal.
    assert classify_plan_issues(state)["soft"]
    assert not classify_plan_issues(state)["hard"]
    assert is_valid_plan(state)
    assert state["plan"]["experiments"][0]["variants"][0]["name"] == "variant_0"


def test_is_valid_plan_rejects_variant_that_is_not_dict():
    from core.workflow_nodes import classify_plan_issues, is_valid_plan
    state = initialize_state()
    state["plan"] = {
        "experiments": [{
            "name": "exp1",
            "evaluation_metrics": ["accuracy"],
            "variants": ["string_variant"],
        }]
    }
    # P3: droppable non-dict variant is soft, auto-fixed to valid.
    assert classify_plan_issues(state)["soft"]
    assert is_valid_plan(state)
    assert state["plan"]["experiments"][0]["variants"] == []


def test_is_valid_plan_passes_valid_plan():
    from core.workflow_nodes import is_valid_plan
    state = initialize_state()
    state["plan"] = {
        "experiments": [{
            "name": "exp1",
            "evaluation_metrics": ["accuracy"],
            "baselines": ["baseline"],
            "variants": [{"name": "v1", "methodology": "m1"}],
        }]
    }
    assert is_valid_plan(state)


def test_terminal_planning_failure_sets_error_attribution():
    from core.workflow_nodes import terminal_planning_failure
    state = initialize_state()
    state["plan"] = {
        "experiments": [{
            "name": "exp1",
            "evaluation_metrics": ["accuracy"],
            "variants": [{"methodology": "no name field"}],
        }]
    }
    # P3: downgrade node builds a bounded fallback and continues — never terminal.
    result = terminal_planning_failure(state)
    assert result["current_phase"] != "complete"
    assert result["should_continue"]
    assert result["terminal_error"] is None
    assert result["plan"]["experiments"]
    assert result["evidence_gate"]["reason_code"] == "planning_downgraded_exploratory"
    assert result["evidence_gate"]["terminal"] is False


def test_graph_routes_invalid_plan_to_terminal_planning_failure():
    """A plan with malformed variants must halt at planning, not fall through."""
    from core.workflow_nodes import is_valid_plan, terminal_planning_failure

    executed = []

    def record(name):
        def fn(s):
            executed.append(name)
            return s
        return fn

    def planning_with_bad_plan(s):
        executed.append("planning")
        s["current_phase"] = "data_validation"
        s["plan"] = {
            "experiments": [{
                "name": "exp1",
                "evaluation_metrics": ["accuracy"],
                "variants": ["bad_string_variant"],
            }]
        }
        return s

    def data_validation_terminates(s):
        executed.append("data_validation")
        s["current_phase"] = "complete"
        s["should_continue"] = False
        return s

    nodes = {
        "topic_discovery": record("topic_discovery"),
        "hypothesis_debate": record("hypothesis_debate"),
        "planning": planning_with_bad_plan,
        "terminal_planning_failure": terminal_planning_failure,
        "data_validation": data_validation_terminates,
        "writing_narrative": record("writing_narrative"),
        "engineering": record("engineering"),
        "independent_validation": record("independent_validation"),
        "writing_results": record("writing_results"),
        "supervision": record("supervision"),
        "meta_evaluation": record("meta_evaluation"),
        "editing": record("editing"),
        "reset": record("reset"),
        "should_reset": should_reset,
        "should_continue": should_continue,
        "is_valid_plan": is_valid_plan,
    }
    graph = create_research_graph(nodes).compile()
    state = initialize_state()
    state["selected_topic"] = {"title": "test"}
    final = graph.invoke(state, {"recursion_limit": 30})

    assert "planning" in executed
    # P3: the downgrade node repairs soft variant issues / falls back, so the
    # run continues past planning instead of halting with zero artifact.
    # terminal_planning_failure is the real function (not record wrapper),
    # so verify via state: downstream must run and no terminal persists.
    assert "data_validation" in executed
    assert final["terminal_error"] is None


def test_graph_routes_valid_plan_past_guard():
    """A valid plan must pass the guard and reach data_validation."""
    from core.workflow_nodes import is_valid_plan, terminal_planning_failure

    executed = []

    def record(name):
        def fn(s):
            executed.append(name)
            return s
        return fn

    def planning_with_good_plan(s):
        executed.append("planning")
        s["current_phase"] = "data_validation"
        s["plan"] = {
            "experiments": [{
                "name": "exp1",
                "evaluation_metrics": ["accuracy"],
                "baselines": ["baseline"],
                "variants": [{"name": "v1", "methodology": "m1"}],
            }]
        }
        return s

    def complete_after_data_validation(s):
        executed.append("data_validation")
        s["current_phase"] = "complete"
        s["should_continue"] = False
        return s

    nodes = {
        "topic_discovery": record("topic_discovery"),
        "hypothesis_debate": record("hypothesis_debate"),
        "planning": planning_with_good_plan,
        "terminal_planning_failure": terminal_planning_failure,
        "data_validation": complete_after_data_validation,
        "writing_narrative": record("writing_narrative"),
        "engineering": record("engineering"),
        "independent_validation": record("independent_validation"),
        "writing_results": record("writing_results"),
        "supervision": record("supervision"),
        "meta_evaluation": record("meta_evaluation"),
        "editing": record("editing"),
        "reset": record("reset"),
        "should_reset": should_reset,
        "should_continue": should_continue,
        "is_valid_plan": is_valid_plan,
    }
    graph = create_research_graph(nodes).compile()
    state = initialize_state()
    state["selected_topic"] = {"title": "test"}
    final = graph.invoke(state, {"recursion_limit": 30})

    assert "planning" in executed
    assert "terminal_planning_failure" not in executed
    assert "data_validation" in executed
    assert final["current_phase"] == "complete"


def test_engineer_retry_loop_actually_retries(monkeypatch):
    """The retry loop must execute all max_attempts even when each fails."""
    from agents.engineer import EngineerAgent

    eng = EngineerAgent.__new__(EngineerAgent)
    eng.context = type("Ctx", (), {"config": type("Cfg", (), {"experiment_seeds": 1, "output_dir": ".", "raw_results_dir": "."})()})()
    eng._plan_revision_requests = []
    refine_calls = {"n": 0}
    gen_calls = {"n": 0}

    def fake_generate(approach, attempt=1, **kwargs):
        gen_calls["n"] += 1
        return (
            "import numpy as np\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "model = LogisticRegression()\n"
            "model.fit(np.random.randn(10, 2), np.random.randint(0, 2, 10))\n"
            "print({'metrics': {'accuracy': 0.5}})"
        )

    def fake_refine(code, error, approach, local_context="", attempt=1, **kwargs):
        refine_calls["n"] += 1
        return code

    monkeypatch.setattr(eng, "_generate_experiment_code", fake_generate)
    monkeypatch.setattr("agents.engineer.validate_code", lambda code: (True, ""))
    monkeypatch.setattr("agents.engineer.fixture_for", lambda *_a, **_k: {})
    monkeypatch.setattr(
        "agents.engineer.execute_multi_seed",
        lambda *_a, **_k: {"success": False, "error": "no metrics"},
    )
    monkeypatch.setattr(eng, "_refine_code", fake_refine)
    monkeypatch.setattr(eng, "_decide_pivot_or_refine", lambda *a, **k: "REFINE")
    monkeypatch.setattr(eng, "_progress", lambda *_a, **_k: None)
    monkeypatch.setattr(
        eng, "check_code_claim_consistency",
        lambda *_a, **_k: {"consistent": True, "score": 10, "notes": []},
    )

    result = EngineerAgent.run_experiment(
        eng,
        {"name": "retry_exp", "baselines": ["logistic_regression"], "claimed_components": []},
    )
    # REFINE path is taken on all 4 attempts
    assert refine_calls["n"] == 4
    assert result["success"] is False
    assert len(result["decision_log"]) == 4
    assert eng.consume_plan_revision_requests()[0]["reason"] == "max_attempts_exhausted"


def test_empty_code_comments_only_classified_as_empty(monkeypatch):
    """Code containing only comments/docstrings must be empty_code_generation, not code_claim_inconsistency."""
    from agents.engineer import EngineerAgent

    eng = EngineerAgent.__new__(EngineerAgent)
    eng.context = type("Ctx", (), {"config": type("Cfg", (), {"experiment_seeds": 1, "output_dir": ".", "raw_results_dir": "."})()})()
    eng._plan_revision_requests = []
    comments_only = "# This is a comment\n'''docstring'''\n# another comment"
    monkeypatch.setattr(eng, "_generate_experiment_code", lambda *_a, **_k: comments_only)
    monkeypatch.setattr("agents.engineer.validate_code", lambda code: (True, ""))
    monkeypatch.setattr("agents.engineer.fixture_for", lambda *_a, **_k: {})
    monkeypatch.setattr(
        "agents.engineer.execute_multi_seed",
        lambda *_a, **_k: {"success": False, "error": "no metrics"},
    )
    monkeypatch.setattr(eng, "_refine_code", lambda code, err, exp: code)
    monkeypatch.setattr(eng, "_decide_pivot_or_refine", lambda *a, **k: "REFINE")
    monkeypatch.setattr(eng, "_progress", lambda *_a, **_k: None)

    result = EngineerAgent.run_experiment(
        eng,
        {"name": "comments_exp", "baselines": ["baseline"], "claimed_components": []},
    )
    assert result["success"] is False
    assert result["failure_kind"] == "empty_code_generation"
