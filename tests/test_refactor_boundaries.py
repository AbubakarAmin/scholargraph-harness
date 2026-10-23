"""Offline regression tests for the extracted runtime boundaries."""

import json
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import pytest

from core.artifacts import FilesystemArtifactStore, save_results
from core.context import RunContext, activate_context, get_active_context, reset_context
from core.pipeline import ResearchPipeline
from core.state import initialize_state
from core.workflow import create_research_graph
from core.workflow_nodes import should_continue, should_reset, is_valid_plan
from core.workflow_nodes import independent_validation_node


class FakeTracker:
    def __init__(self):
        self.run_id = "fake-run"
        self.completed = []

    def complete(self, success=True):
        self.completed.append(success)


def test_artifact_service_writes_paper_plan_and_summary(tmp_path):
    state = initialize_state()
    state.update(
        {
            "iteration": 2,
            "selected_topic": {"title": "Typed boundaries"},
            "plan": {"title": "A plan", "sections": [{"name": "Results"}]},
            "draft_sections": {"Results": "Observed result."},
            "supervisor_scores": {"Results": 8.5},
            "engineer_outputs": {"baseline": {"success": True}},
            "meta_feedback": ["kept grounded"],
            "latex_output": "\\section{Results}\nObserved result.",
                "human_approved": True,
        }
    )

    save_results(state, str(tmp_path))

    assert (tmp_path / "paper_output.tex").read_text(encoding="utf-8").startswith("\\section")
    assert (tmp_path / "plan.yaml").exists()
    summary = json.loads((tmp_path / "research_summary.json").read_text(encoding="utf-8"))
    assert summary["iteration"] == 2
    assert summary["sections_written"] == ["Results"]
    assert summary["experiments_run"] == ["baseline"]


def test_filesystem_artifact_store_implements_safe_named_writes(tmp_path):
    store = FilesystemArtifactStore(str(tmp_path / "artifacts"))

    text_path = store.save("nested/result.txt", "raw result")
    json_path = store.save("metrics.json", {"accuracy": 0.85})

    assert Path(text_path).read_text(encoding="utf-8") == "raw result"
    assert json.loads(Path(json_path).read_text(encoding="utf-8"))["accuracy"] == 0.85
    with pytest.raises(ValueError):
        store.save("../outside.txt", "blocked")


def test_llm_compatibility_aliases_have_one_owner():
    from core import llm, utils

    assert callable(llm.setup_gemini)
    assert callable(llm.call_gemini)
    assert not hasattr(utils, "setup_gemini")
    assert not hasattr(utils, "call_gemini")


def test_context_activation_restores_previous_context():
    outer = RunContext(config=None, memory=None, research_db=None)
    inner = RunContext(config=None, memory=None, research_db=None)

    outer_token = activate_context(outer)
    try:
        assert get_active_context() is outer
        inner_token = activate_context(inner)
        try:
            assert get_active_context() is inner
        finally:
            reset_context(inner_token)
        assert get_active_context() is outer
    finally:
        reset_context(outer_token)

    assert get_active_context() is None


def test_pipeline_run_finalizes_and_completes_tracker():
    tracker = FakeTracker()
    finalized = []

    class FakeApp:
        def stream(self, source, config):
            assert source["current_phase"] == "topic_discovery"
            yield {"editing": {"current_phase": "complete", "latex_output": "paper"}}

    class FakeGraph:
        def compile(self, checkpointer):
            assert checkpointer == "checkpoint"
            return FakeApp()

    pipeline = ResearchPipeline(
        lambda: FakeGraph(),
        lambda: "checkpoint",
        context=RunContext(config=None, memory=None, research_db=None, tracker=tracker),
    )
    result = pipeline.run(initialize_state(), "run-1", finalize=finalized.append)

    assert result.nodes_seen == 1
    assert result.state["latex_output"] == "paper"
    assert finalized == [result.state]
    assert tracker.completed == [True]


def test_pipeline_marks_tracker_failed_when_execution_raises():
    tracker = FakeTracker()

    class FakeGraph:
        def compile(self, checkpointer):
            raise RuntimeError("compile failed")

    pipeline = ResearchPipeline(
        lambda: FakeGraph(),
        lambda: "checkpoint",
        context=RunContext(config=None, memory=None, research_db=None, tracker=tracker),
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        pipeline.run(initialize_state(), "run-1")

    assert tracker.completed == [False]


def test_workflow_graph_compiles_with_injected_nodes():
    node_names = {
        "topic_discovery", "hypothesis_debate", "planning", "terminal_planning_failure",
        "data_validation", "writing_narrative",
        "engineering", "independent_validation", "writing_results", "supervision", "meta_evaluation",
        "editing", "reset", "should_reset", "should_continue", "is_valid_plan",
    }
    nodes = {name: (lambda state: state) for name in node_names}
    graph = create_research_graph(nodes)

    compiled = graph.compile()

    assert compiled is not None


def test_offline_workflow_runs_every_phase_to_a_release_candidate():
    """Exercise the actual LangGraph routing without network, models, or APIs."""
    phases = [
        "topic_discovery", "hypothesis_debate", "planning", "data_validation",
        "writing_narrative", "engineering", "independent_validation",
        "writing_results", "supervision", "editing",
    ]
    observed = []

    def node(name, next_phase):
        def run(state):
            observed.append(name)
            state["current_phase"] = next_phase
            return state
        return run

    def planning_node_with_plan(state):
        observed.append("planning")
        state["current_phase"] = "data_validation"
        state["plan"] = {
            "experiments": [{
                "name": "test_exp",
                "evaluation_metrics": ["accuracy"],
                "baselines": ["baseline"],
            }]
        }
        return state

    nodes = {
        "topic_discovery": node("topic_discovery", "hypothesis_debate"),
        "hypothesis_debate": node("hypothesis_debate", "planning"),
        "planning": planning_node_with_plan,
        "terminal_planning_failure": node("terminal_planning_failure", "complete"),
        "data_validation": node("data_validation", "writing_narrative"),
        "writing_narrative": node("writing_narrative", "engineering"),
        "engineering": node("engineering", "independent_validation"),
        "independent_validation": node("independent_validation", "writing_results"),
        "writing_results": node("writing_results", "supervision"),
        "supervision": node("supervision", "editing"),
        "meta_evaluation": node("meta_evaluation", "complete"),
        "editing": node("editing", "complete"),
        "reset": node("reset", "topic_discovery"),
        "should_reset": should_reset,
        "should_continue": should_continue,
        "is_valid_plan": is_valid_plan,
    }
    graph = create_research_graph(nodes).compile()
    result = graph.invoke(initialize_state(), {"recursion_limit": 30})

    assert observed == phases
    assert result["current_phase"] == "complete"


def test_route_selectors_honor_terminal_state():
    state = initialize_state()
    state["current_phase"] = "complete"
    assert should_reset(state) == "end"
    assert should_continue(state) == "end"

    state["current_phase"] = "planning"
    state["should_reset"] = True
    state["should_continue"] = True
    assert should_reset(state) == "reset"
    assert should_continue(state) == "continue"


def test_independent_validation_node_records_clean_artifacts(tmp_path, monkeypatch):
    state = initialize_state()
    state.update({
        "plan": {"experiments": [{"name": "toy", "evaluation_metrics": ["accuracy"]}]},
        "engineer_outputs": {
            "toy": {
                "success": True,
                "code": 'import json\nprint(json.dumps({"metrics": {"accuracy": 0.8}}))',
            }
        },
    })
    monkeypatch.setattr("core.config.config.raw_results_dir", str(tmp_path))
    monkeypatch.setattr("core.config.config.experiment_seeds", 3)

    result = independent_validation_node(state)

    assert result["execution_artifacts"]["toy"]["status"] == "completed"
    assert result["analysis_reports"]["independent"]["metrics"]
    assert result["verification_findings"] == []


def test_primary_agent_boundaries_use_shared_contracts():
    from agents.editor import EditorAgent
    from agents.planner import PlannerAgent
    from agents.writer import WriterAgent
    from core.contracts import ExperimentOutput, Paper, Plan, RevisionRequest, Topic

    planner_create = get_type_hints(PlannerAgent.create_plan)
    planner_revise = get_type_hints(PlannerAgent.revise_plan)
    writer_draft = get_type_hints(WriterAgent.draft_section)
    editor_create = get_type_hints(EditorAgent.create_final_paper)

    assert planner_create["topic"] is Topic
    assert planner_create["return"] is Plan
    assert planner_revise["revision_request"] is RevisionRequest
    assert writer_draft["topic"] is Topic
    assert writer_draft["plan"] is Plan
    assert get_origin(writer_draft["engineer_outputs"]) is dict
    assert get_args(writer_draft["engineer_outputs"]) == (str, ExperimentOutput)
    assert editor_create["topic"] is Topic
    assert editor_create["plan"] is Plan
    assert editor_create["return"] is Paper


def test_remaining_agent_boundaries_use_shared_contracts():
    from agents.engineer import EngineerAgent
    from agents.meta_agent import MetaAgent
    from agents.supervisor import SupervisorAgent
    from agents.topic_hunter import TopicHunterAgent
    from core.contracts import (
        CodeClaimReport,
        ExperimentOutput,
        ExperimentSpec,
        FeasibilityReport,
        MetaChatResult,
        MetaDashboard,
        RevisionRequest,
        Topic,
    )
    from core.state import ResearchState

    engineer_run = get_type_hints(EngineerAgent.run_experiment)
    engineer_branch = get_type_hints(EngineerAgent.run_branching_search)
    engineer_check = get_type_hints(EngineerAgent.check_code_claim_consistency)
    supervisor_eval = get_type_hints(SupervisorAgent.evaluate_section)
    hunter_discover = get_type_hints(TopicHunterAgent.discover_topics)
    hunter_filter = get_type_hints(TopicHunterAgent.feasibility_filter)
    meta_chat = get_type_hints(MetaAgent.chat)
    meta_dashboard = get_type_hints(MetaAgent.get_run_dashboard)

    assert engineer_run["experiment"] is ExperimentSpec
    assert engineer_run["return"] is ExperimentOutput
    assert engineer_branch["return"] is ExperimentOutput
    assert engineer_check["return"] is CodeClaimReport
    assert supervisor_eval["engineer_outputs"] is not None
    assert get_origin(hunter_discover["return"]) is list
    assert get_args(hunter_discover["return"]) == (Topic,)
    assert hunter_filter["return"] is FeasibilityReport
    assert ResearchState in get_args(meta_chat["state"])
    assert meta_chat["return"] is MetaChatResult
    assert meta_dashboard["state"] is ResearchState
    assert meta_dashboard["return"] is MetaDashboard
