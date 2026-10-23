"""Regression tests for immutable experiment handoffs and terminal gates."""

import json

from core.evidence_gate import build_contract, gate_engineering_outputs, validate_dataset_identity, validate_experiments
from core.artifacts import save_results
from core.state import initialize_state
from core.verification import validate_empirical_claims
from core.workflow_nodes import engineering_node, independent_validation_node
from core.workflow_nodes import topic_discovery_node


def test_validate_experiments_rejects_null_and_string_candidates():
    assert validate_experiments([None])
    assert validate_experiments(["experiment"])
    assert validate_experiments([{"name": "toy", "variants": ["bad"]}])


def test_dataset_identity_requires_the_committed_artifact():
    experiment = {"name": "toy", "evaluation_metrics": ["accuracy"], "dataset": {"name": "fixed"}}
    assert validate_dataset_identity([experiment], {})
    assert not validate_dataset_identity(
        [experiment],
        {"fixed": {"content_hash": "abc", "validation": {"passed": True}}},
    )
    assert validate_dataset_identity(
        [{**experiment, "dataset": {"name": "fixed", "content_hash": "changed"}}],
        {"fixed": {"content_hash": "abc", "validation": {"passed": True}}},
    )


def test_engineering_gate_requires_successful_output_bound_to_contract(tmp_path):
    experiment = {"name": "toy", "evaluation_metrics": ["accuracy"]}
    contract = build_contract(experiment)
    raw = tmp_path / "toy.json"
    raw.write_text(json.dumps({"metrics": {"accuracy": 0.2}}), encoding="utf-8")

    decision = gate_engineering_outputs(
        {"experiments": [experiment]},
        {
            "toy": {
                "success": True,
                "contract_hash": contract["contract_hash"],
                "raw_results_path": str(raw),
                "aggregate_metrics": {"accuracy": {"mean": 0.2}},
            }
        },
        {"toy": contract},
    )

    assert decision["allowed"]
    assert not decision["terminal"]


def test_engineering_gate_stops_failed_output_even_with_error_artifact(tmp_path):
    experiment = {"name": "toy", "evaluation_metrics": ["accuracy"]}
    contract = build_contract(experiment)
    raw = tmp_path / "FAIL_toy.json"
    raw.write_text(json.dumps({"error": "syntax error"}), encoding="utf-8")

    decision = gate_engineering_outputs(
        {"experiments": [experiment]},
        {
            "toy": {
                "success": False,
                "contract_hash": contract["contract_hash"],
                "raw_results_path": str(raw),
                "error": "syntax error",
            }
        },
        {"toy": contract},
    )

    assert decision["terminal"]
    assert decision["reason_code"] == "technical_execution_failure"


def test_engineering_node_does_not_run_malformed_experiment(monkeypatch):
    state = initialize_state()
    state.update({
        "plan": {"experiments": [None]},
        "selected_topic": {"title": "test"},
    })

    class ExplodingEngineer:
        def __init__(self, *_args):
            raise AssertionError("engineer must not be created for malformed plans")

    monkeypatch.setattr("core.workflow_nodes.EngineerAgent", ExplodingEngineer)
    result = engineering_node(state)

    assert result["current_phase"] == "planning"
    assert result["should_continue"]
    assert result["plan_revision_requests"]
    assert result["plan_revision_requests"][0]["reason"] == "invalid_experiment_plan"
    assert result["plan"]["schema_revision_attempts"] == 1


def test_engineering_node_normalizes_experiment_name_alias(monkeypatch):
    state = initialize_state()
    state.update({
        "plan": {
            "experiments": [{
                "experiment_name": "toy",
                "evaluation_metrics": ["accuracy"],
                "baselines": ["logistic_regression"],
            }]
        },
        "selected_topic": {"title": "test"},
    })

    class FakeEngineer:
        def __init__(self, *_args):
            self._plan_revision_requests = []

        def run_branching_search(self, *_args, **_kwargs):
            return {"success": True, "experiment_name": "toy"}

        def run_experiment(self, experiment, **_kwargs):
            assert experiment["name"] == "toy"
            return {
                "success": True,
                "code": "print(1)",
                "aggregate_metrics": {"accuracy": {"mean": 0.5}},
                "raw_results_path": "",
            }

        def consume_plan_revision_requests(self):
            return []

    monkeypatch.setattr("core.workflow_nodes.EngineerAgent", FakeEngineer)
    monkeypatch.setattr(
        "core.workflow_nodes.gate_engineering_outputs",
        lambda *_args, **_kwargs: {"allowed": True, "reason_code": "ok"},
    )
    result = engineering_node(state)
    assert result["plan"]["experiments"][0]["name"] == "toy"
    assert result["current_phase"] == "writing_results"


def test_engineering_node_terminals_after_repeated_schema_failures(monkeypatch):
    state = initialize_state()
    state.update({
        "plan": {"experiments": [None], "schema_revision_attempts": 2},
        "selected_topic": {"title": "test"},
    })

    class ExplodingEngineer:
        def __init__(self, *_args):
            raise AssertionError("engineer must not be created for malformed plans")

    monkeypatch.setattr("core.workflow_nodes.EngineerAgent", ExplodingEngineer)
    result = engineering_node(state)

    assert result["current_phase"] == "complete"
    assert not result["should_continue"]
    assert result["evidence_gate"]["reason_code"] == "invalid_experiment_plan"


def test_engineering_node_rejects_contract_drift(monkeypatch):
    original = {"name": "toy", "evaluation_metrics": ["accuracy"]}
    state = initialize_state()
    state.update({
        "plan": {"experiments": [{**original, "evaluation_metrics": ["f1"]}]},
        "selected_topic": {"title": "test"},
        "experiment_contracts": {"toy": build_contract(original)},
    })

    class ExplodingEngineer:
        def __init__(self, *_args):
            raise AssertionError("engineer must not run after contract drift")

    monkeypatch.setattr("core.workflow_nodes.EngineerAgent", ExplodingEngineer)
    result = engineering_node(state)

    assert result["current_phase"] == "complete"
    assert result["evidence_gate"]["reason_code"] == "experiment_contract_drift"


def test_independent_validation_stops_without_code_artifacts():
    state = initialize_state()
    state.update({
        "plan": {"experiments": [{"name": "toy", "evaluation_metrics": ["accuracy"]}]},
        "engineer_outputs": {
            "toy": {"success": False, "error": "runtime failure"},
        },
    })

    result = independent_validation_node(state)

    assert result["current_phase"] == "complete"
    assert not result["should_continue"]
    assert result["terminal_error"]


def test_terminal_run_writes_failure_dossier_without_latex(tmp_path):
    state = initialize_state()
    state.update({
        "run_id": "failed-run",
        "terminal_error": "sandbox runtime failure",
        "evidence_gate": {"terminal": True, "reason_code": "technical_execution_failure"},
        "latex_output": "\\section{Fabricated}",
    })

    save_results(state, str(tmp_path))

    assert (tmp_path / "failure_dossier.json").exists()
    assert not (tmp_path / "paper_output.tex").exists()


def test_discovery_exhaustion_is_a_terminal_failed_run(monkeypatch):
    from core.config import config as _cfg

    state = initialize_state()
    # Exhaustion is relative to MAX_ITERATIONS (unified cap), not a hardcoded 5.
    state["iteration"] = _cfg.max_iterations - 1

    class EmptyTopicHunter:
        def __init__(self, *_args):
            pass

        def discover_topics(self, *_args):
            return []

    monkeypatch.setattr("core.workflow_nodes.TopicHunterAgent", EmptyTopicHunter)
    result = topic_discovery_node(state)

    assert result["current_phase"] == "complete"
    assert not result["should_continue"]
    assert result["terminal_error"]
    assert result["technical_failures"]["topic_discovery"]["reason_code"] == "no_viable_topic_after_retries"


def test_discovery_empty_before_cap_resets(monkeypatch):
    state = initialize_state()
    state["iteration"] = 0

    class EmptyTopicHunter:
        def __init__(self, *_args):
            pass

        def discover_topics(self, *_args):
            return []

    monkeypatch.setattr("core.workflow_nodes.TopicHunterAgent", EmptyTopicHunter)
    result = topic_discovery_node(state)

    assert result["should_reset"] is True
    assert result["current_phase"] != "complete"


def test_empirical_claim_check_rejects_harness_diagnostics():
    result = validate_empirical_claims(
        "The run failed because tracemalloc was blocked by the sandbox.",
        {"toy": {"success": True}},
    )

    assert not result["passed"]
    assert "tracemalloc" in result["prohibited_text"]


def test_editor_refuses_failed_run_assembly():
    from agents.editor import EditorAgent

    editor = EditorAgent.__new__(EditorAgent)
    try:
        editor.create_final_paper(
            {"title": "Toy", "description": "Test"},
            {"Results": "No result"},
            {"experiments": []},
            {"toy": {"success": False}},
        )
    except RuntimeError as exc:
        assert "failed experiments" in str(exc)
    else:
        raise AssertionError("failed runs must not produce a manuscript")


def test_editor_refuses_dataset_drift():
    from agents.editor import EditorAgent

    editor = EditorAgent.__new__(EditorAgent)
    try:
        editor.create_final_paper(
            {"title": "Toy", "description": "Test"},
            {"Results": "Results from the bundled dataset."},
            {"experiments": [{"name": "toy", "dataset": {"name": "Yahoo Finance"}}]},
            {"toy": {"success": True, "aggregate_metrics": {"accuracy": {"mean": 0.8}}}},
        )
    except RuntimeError as exc:
        assert "release referee" in str(exc)
    else:
        raise AssertionError("dataset drift must block assembly")
