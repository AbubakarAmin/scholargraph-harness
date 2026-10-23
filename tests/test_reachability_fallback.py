"""Reachability guarantees for the research harness."""

from agents.topic_hunter import TopicHunterAgent
from core.artifacts import save_results
from core.evidence_gate import build_contract, gate_engineering_outputs
from core.state import initialize_state
from core.workflow_nodes import _minimal_fallback_plan, _run_synthetic_fallback_outputs


def test_bounded_topic_fallback_is_executable():
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    topic = hunter._build_reachable_fallback_topic(
        "calibration under distribution shift",
        "temperature scaling and robustness",
        [{"title": "A retrieved calibration paper"}],
    )

    assert topic["structured_hypothesis"]["minimum_viable_experiment"]["dataset"] == "bundled_synthetic"
    assert topic["topic_admission"]["admitted"] is True
    assert topic["fallback_provenance"]["reason"] == "strict_gates_empty"


def test_unapproved_run_writes_reviewable_candidate(tmp_path):
    state = {
        "mode": "full_research",
        "iteration": 0,
        "latex_output": "\\section{Exploratory Results}\nCandidate",
        "human_approved": False,
        "terminal_error": None,
        "evidence_gate": {},
        "draft_sections": {},
        "supervisor_scores": {},
        "engineer_outputs": {},
        "meta_feedback": [],
    }
    save_results(state, str(tmp_path))
    assert (tmp_path / "paper_candidate.tex").exists()
    assert not (tmp_path / "paper_output.tex").exists()


def test_synthetic_engineering_fallback_remains_exploratory_only(tmp_path, monkeypatch):
    state = initialize_state()
    state["plan"] = _minimal_fallback_plan({"title": "fallback study"}, "test")
    experiment = state["plan"]["experiments"][0]
    state["experiment_contracts"] = {experiment["name"]: build_contract(experiment)}
    monkeypatch.setattr("core.workflow_nodes.config.raw_results_dir", str(tmp_path))

    outputs = _run_synthetic_fallback_outputs(state)
    assert outputs[experiment["name"]]["outcome"] == "fallback_synthetic"
    assert outputs[experiment["name"]]["aggregate_metrics"]
    gate = gate_engineering_outputs(state["plan"], outputs, state["experiment_contracts"])
    assert gate["allowed"] is False
    assert gate["reason_code"] == "exploratory_only"
