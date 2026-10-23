"""Regression tests for run 85598a00 review findings."""

from agents.hypothesis_debate import ChallengerAgent
from agents.planner import PlannerAgent
from agents.topic_hunter import TopicHunterAgent
from core.evidence_gate import validate_experiments
from core.run_log import CrossRunMemory
from core.utils import is_degenerate_llm_output, title_token_overlap
from core.verification import hard_verify_section


def test_planner_normalizes_experiment_name_alias_to_contract_name():
    planner = PlannerAgent.__new__(PlannerAgent)
    experiments = planner._normalize_experiments([
        {
            "experiment_name": "Baseline Comparison",
            "evaluation_metrics": ["accuracy"],
            "baselines": ["logistic_regression"],
            "variants": [{"experiment_name": "alt", "evaluation_metrics": ["accuracy"]}],
        }
    ])
    assert experiments[0]["name"] == "Baseline Comparison"
    assert "experiment_name" not in experiments[0]
    assert experiments[0]["variants"][0]["name"] == "alt"
    assert not validate_experiments(experiments)


def test_planner_create_plan_rejects_schema_invalid_experiments(monkeypatch):
    from agents import planner as planner_mod

    planner = PlannerAgent.__new__(PlannerAgent)
    planner.context = type("Context", (), {"config": type("Runtime", (), {"research_domain": "test"})()})()
    planner._generate_plan_structure = lambda *_args: {
        "title": "t",
        "sections": [],
        "methodology": "local",
        "compute_budget": "cpu",
    }
    planner._ensure_falsifiable_contributions = lambda *_args: []
    planner._generate_experiments = lambda *_args: []
    planner._attach_variants = lambda experiments: experiments
    planner._flag_unfalsifiable = lambda *_args: []
    planner._flag_missing_baselines = lambda *_args: []
    planner._generate_dependencies = lambda *_args: []
    planner._generate_timeline = lambda *_args: []
    planner._repair_plan = lambda plan, *args, **kwargs: plan
    planner._store_plan = lambda *_args: None
    monkeypatch.setattr(planner_mod, "list_datasets", lambda: [])
    monkeypatch.setattr(planner_mod, "check_plan_feasibility", lambda *_args, **_kwargs: [])

    try:
        planner.create_plan({"title": "topic", "description": "d"})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "contract validation" in str(exc)


def test_degenerate_llm_output_detects_safety_stub():
    assert is_degenerate_llm_output("User Safety: safe")
    assert is_degenerate_llm_output("# Related Work\n\n# Related Work\n\nUser Safety: safe")
    assert not is_degenerate_llm_output(
        "This related work section surveys evaluation methodology for robustness under distribution shift "
        "and positions the contribution against prior calibration and domain-adaptation literature."
    )


def test_hard_verify_rejects_stub_related_work():
    result = hard_verify_section(
        "# Related Work\n\n# Related Work\n\nUser Safety: safe",
        section_name="Related Work",
        content_requirements="Survey literature and identify gaps",
    )
    assert not result["passed"]
    assert result["substance_errors"]


def test_challenger_retries_and_fails_closed_on_garbled_output(monkeypatch):
    agent = ChallengerAgent.__new__(ChallengerAgent)
    agent.context = None
    agent.client = None
    agent.vector_memory = type("Mem", (), {"get_prompt_context": lambda *a, **k: []})()
    calls = {"n": 0}

    def fake_llm(*_args, **_kwargs):
        calls["n"] += 1
        return "User Safety: safe"

    monkeypatch.setattr("agents.hypothesis_debate.call_llm", fake_llm)
    monkeypatch.setattr("agents.hypothesis_debate.parse_json_from_llm", lambda *_a, **_k: {})
    monkeypatch.setattr("agents.hypothesis_debate.check_plan_feasibility", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "agents.hypothesis_debate.EloStore",
        lambda **_kwargs: type("E", (), {"get": lambda self, *_a, **_k: 1500})(),
    )
    monkeypatch.setattr("agents.hypothesis_debate.hypothesis_kind", lambda *_a, **_k: "general")

    summary = ChallengerAgent.build_rebuttal(agent, {"title": "t", "description": "d"}, "argument text")
    assert calls["n"] == 2
    # v5 fail-soft: invalid flag stays for audit, but template fallback gives
    # substantive objections (no 25-char guard stub, no severity-5 poison).
    assert agent._challenger_invalid is True
    assert agent._challenger_fallback_used is True
    assert len(summary) > 200
    assert summary != "CHALLENGER_OUTPUT_INVALID"
    assert agent._last_objections
    assert all(o.get("source") == "challenger_template_fallback" for o in agent._last_objections)
    assert {o.get("criterion") for o in agent._last_objections} >= {"falsifiability", "baseline"}


def test_topic_hunter_excludes_failed_debate_near_duplicates(tmp_path):
    memory = CrossRunMemory(path=str(tmp_path / "cross_run.jsonl"))
    memory.record_rejection(
        "topic",
        "Meta-Learning for Robustness to Distributional Shift",
        "failed_hypothesis_debate",
        {"score": 6.0},
    )
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter._excluded_titles_cache = memory.excluded_topic_titles()
    matched = TopicHunterAgent._matches_excluded_topic(
        hunter,
        "Meta Learning for Robustness to Distributional Shift in Models",
    )
    assert matched
    assert title_token_overlap(
        "Meta-Learning for Robustness to Distributional Shift",
        "Meta Learning for Robustness to Distributional Shift in Models",
    ) >= 0.55


def test_engineer_retries_claim_inconsistency_before_failing(monkeypatch):
    from agents.engineer import EngineerAgent

    eng = EngineerAgent.__new__(EngineerAgent)
    eng.context = type("Ctx", (), {"config": type("Cfg", (), {"experiment_seeds": 1, "output_dir": ".", "raw_results_dir": "."})()})()
    eng._plan_revision_requests = []
    calls = {"n": 0}

    def fake_generate(approach, attempt=1, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()\nprint('{}')"
        return (
            "from sklearn.ensemble import GradientBoostingClassifier\n"
            "model = GradientBoostingClassifier()\n"
            "print('{\"metrics\": {\"accuracy\": 0.9}}')"
        )

    monkeypatch.setattr(eng, "_generate_experiment_code", fake_generate)
    monkeypatch.setattr("agents.engineer.validate_code", lambda code: (True, ""))
    monkeypatch.setattr("agents.engineer.fixture_for", lambda *_a, **_k: {})
    monkeypatch.setattr(
        "agents.engineer.execute_multi_seed",
        lambda *_a, **_k: {
            "success": True,
            "aggregate_metrics": {"accuracy": {"mean": 0.9, "std": 0.0}},
        },
    )
    monkeypatch.setattr(eng, "_auto_ablation", lambda *_a, **_k: {})
    monkeypatch.setattr(eng, "_store_raw", lambda *_a, **_k: "raw.json")
    monkeypatch.setattr(eng, "_store_experiment_results", lambda *_a, **_k: None)
    monkeypatch.setattr(eng, "_progress", lambda *_a, **_k: None)

    result = EngineerAgent.run_experiment(
        eng,
        {
            "name": "gb_exp",
            "baselines": ["gradient_boosting"],
            "claimed_components": [],
        },
    )
    assert calls["n"] == 2
    assert result["success"] is True
    assert any("code_claim_inconsistency" in (item.get("reason") or "") for item in result["decision_log"])


def test_engineer_empty_code_is_distinct_failure_kind(monkeypatch):
    from agents.engineer import EngineerAgent

    eng = EngineerAgent.__new__(EngineerAgent)
    eng.context = type("Ctx", (), {"config": type("Cfg", (), {"experiment_seeds": 1, "output_dir": ".", "raw_results_dir": "."})()})()
    eng._plan_revision_requests = []
    monkeypatch.setattr(eng, "_generate_experiment_code", lambda *_a, **_k: "")
    monkeypatch.setattr(eng, "_progress", lambda *_a, **_k: None)
    monkeypatch.setattr("agents.engineer.fixture_for", lambda *_a, **_k: {})

    result = EngineerAgent.run_experiment(eng, {"name": "empty_exp", "baselines": ["logistic_regression"]})
    assert result["success"] is False
    assert result["failure_kind"] == "empty_code_generation"
    assert result["error"] == "empty_code_generation"
    assert len(result["decision_log"]) == 4
    assert eng.consume_plan_revision_requests()[0]["reason"] == "empty_code_generation"


def test_engineering_node_routes_plan_revision_before_terminal(monkeypatch):
    from core.state import initialize_state
    from core.workflow_nodes import engineering_node

    state = initialize_state()
    state.update({
        "plan": {
            "experiments": [{
                "name": "toy",
                "evaluation_metrics": ["accuracy"],
                "baselines": ["logistic_regression"],
            }]
        },
        "selected_topic": {"title": "test"},
    })

    class FakeEngineer:
        def __init__(self, *_args):
            self._plan_revision_requests = [{
                "reason": "code_claim_inconsistency",
                "experiment": "toy",
                "detail": "missing logistic regression",
            }]

        def run_experiment(self, experiment, **_kwargs):
            return {
                "success": False,
                "error": "code_claim_inconsistency: missing logistic regression",
                "failure_kind": "code_claim_inconsistency",
                "experiment_name": experiment["name"],
            }

        def consume_plan_revision_requests(self):
            reqs = list(self._plan_revision_requests)
            self._plan_revision_requests.clear()
            return reqs

    monkeypatch.setattr("core.workflow_nodes.EngineerAgent", FakeEngineer)
    monkeypatch.setattr(
        "core.workflow_nodes.gate_engineering_outputs",
        lambda *_args, **_kwargs: {
            "allowed": False,
            "terminal": True,
            "reason_code": "technical_execution_failure",
            "message": "toy failed",
        },
    )
    result = engineering_node(state)
    assert result["current_phase"] == "planning"
    assert result["should_continue"] is True
    assert result["plan_revision_requests"]
    assert result["plan"]["engineer_revision_attempts"] == 1
    assert result["evidence_gate"]["reason_code"] == "plan_revision_requested"