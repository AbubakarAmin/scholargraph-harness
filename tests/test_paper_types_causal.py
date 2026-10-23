"""M4: comparative + causal + sandbox hardening (§11-M4, 5 tracker cases + wiring).

Tracker cases:
- naive observational → causal verbs blocked + downgraded analytical + banner.
- synthetic-DAG benchmark → passes with `causal` level.
- placebo infeasible → limitation (not block).
- 1-baseline comparative → refused with reason.
- pivot path retains alternatives; `_error_category_hint` no TypeError.

Plus per-type wiring guards (recipes, skeletons, packs, manifests,
supervisor dispatch, node mapping) and the L/D fix pins
(L2/L4/L5/L18, D9–D16) following the M1–M3 offline pattern (pure unit
tests first, no LLM, no network).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _naive_plan():
    return {
        "paper_type": "cause_effect",
        "methodology": "Compare observed outcomes between self-selected groups.",
        "experiments": [{
            "name": "group_comparison",
            "methodology": "Compare observed group means without randomization.",
            "baselines": ["group_b_mean"],
            "evaluation_metrics": ["accuracy"],
            "falsifiable_prediction": "Group A mean exceeds group B mean.",
            "statistical_test": "t-test p<0.05",
            "variants": [{"name": "alt", "methodology": "different split"}],
        }],
        "causal_spec": {
            "dag": {"treatment": "program", "outcome": "score", "confounders": ["income"]},
            "id_strategy": "naive",
        },
    }


def _synthetic_dag_plan():
    return {
        "paper_type": "cause_effect",
        "methodology": "Randomized assignment on a synthetic-DAG benchmark with sensitivity analysis. Sensitivity bounds computed via coefficient stability.",
        "experiments": [{
            "name": "rct_benchmark",
            "methodology": "Random assignment on synthetic DAG data; estimate ATE with covariate adjustment.",
            "baselines": ["unadjusted_diff", "random_guess"],
            "evaluation_metrics": ["ate_error"],
            "falsifiable_prediction": "Adjusted ATE error < unadjusted error.",
            "statistical_test": "permutation test p<0.05",
            "variants": [{"name": "alt", "methodology": "different seed"}],
            "sensitivity": "Coefficient-stability bounds hold under 2x confounding.",
        }],
        "causal_spec": {
            "dag": {"treatment": "program", "outcome": "score", "confounders": ["income"]},
            "id_strategy": "synthetic_benchmark",
            "sensitivity": "Coefficient-stability bounds hold under 2x confounding.",
        },
    }


def _comparative_experiments(n_baselines=2):
    baselines = ["logistic_regression", "random_forest", "random_guess"][:n_baselines]
    return [{
        "name": "model_matrix",
        "methodology": "Train proposed model and baselines on same splits.",
        "baselines": baselines,
        "evaluation_metrics": ["accuracy"],
        "falsifiable_prediction": "Proposed mean accuracy > best baseline mean.",
        "statistical_test": "Welch t-test p<0.05 across 3 seeds",
        "variants": [{"name": "v1", "methodology": "m1"}],
    }]


def _causal_state(paper_type="cause_effect", plan=None):
    from core.state import initialize_state
    from core.phase_registry import resolve_recipe

    recipe, skipped, evidence, _ = resolve_recipe(paper_type, {})
    state = initialize_state()
    state["run_id"] = f"test-{paper_type}"
    state["paper_type"] = paper_type
    state["pipeline_recipe"] = recipe
    state["skipped_phases"] = skipped
    state["evidence_requirements"] = evidence
    state["selected_topic"] = {"title": "Test cause topic",
                               "description": "A bounded causal investigation."}
    state["plan"] = plan if plan is not None else _naive_plan()
    state["engineer_outputs"] = {}
    return state


# ---------------------------------------------------------------------------
# Tracker case 1: naive observational → blocked + downgraded analytical + banner
# ---------------------------------------------------------------------------

def test_naive_observational_blocked_and_downgraded():
    from agents.causal_agent import CausalAgent, classify_id
    from core.verification import verify_causal
    import core.workflow_nodes as wn

    assert classify_id({"methodology": "Compare observed group means."})["strategy"] == "naive"

    audit = CausalAgent().run_audit(_naive_plan(), {})
    assert audit["passed"] is False
    assert audit["id_strategy"] == "naive"
    assert audit["downgrade"] and audit["downgrade"]["to"] == "analytical"
    assert "banner" in audit["downgrade"] and audit["downgrade"]["banner"]

    gate = verify_causal(audit, {"Results": "Groups differ. Sensitivity analysis pending. Identifying assumptions stated."})
    assert gate["passed"] is False
    assert any(f["check"] == "identification" for f in gate["findings"])

    state = wn.causal_audit_node(_causal_state())
    assert state["paper_type"] == "analytical"
    assert state["pipeline_recipe"][0] == "intake"
    assert any("downgrade" in (s.get("reason") or "").lower() or "cause_effect" in (s.get("reason") or "")
               for s in state["skipped_phases"])
    assert any("downgrad" in str(m).lower() for m in state["meta_feedback"])
    assert state.get("thesis") and state["thesis"].get("claim")
    assert state["causal_claim"]["downgrade"]["to"] == "analytical"


# ---------------------------------------------------------------------------
# Tracker case 2: synthetic-DAG benchmark → passes with `causal` level
# ---------------------------------------------------------------------------

def test_synthetic_dag_benchmark_passes_with_causal_level():
    from agents.causal_agent import CausalAgent, classify_id, verb_level
    from core.verification import verify_causal

    plan = _synthetic_dag_plan()
    assert classify_id({"methodology": plan["methodology"], "dag": plan["causal_spec"]["dag"]})["strategy"] in (
        "rct", "synthetic_benchmark")

    audit = CausalAgent().run_audit(plan, {}, indep_pass=True)
    assert audit["passed"] is True
    assert audit["verb_level"] == "causal"
    assert verb_level(audit["id_strategy"], audit["sensitivity"], True) == "causal"
    assert verb_level(audit["id_strategy"], audit["sensitivity"], False) == "associational"
    assert verb_level("naive", "note", True) == "associational"

    sections = {
        "Methods": "Identification via random assignment on a synthetic DAG. Identifying assumptions: ignorability holds by construction.",
        "Results": "The adjusted difference is 0.42 (n=400, 95% CI 0.31-0.53).",
        "Discussion": "Sensitivity bounds hold under 2x confounding.",
    }
    gate = verify_causal({**audit, "verb_level": "causal"}, sections, indep_pass=True)
    assert gate["passed"], gate["findings"]


# ---------------------------------------------------------------------------
# Tracker case 3: placebo infeasible → limitation (not block)
# ---------------------------------------------------------------------------

def test_placebo_infeasible_is_limitation_not_block():
    from agents.causal_agent import CausalAgent, placebo_hint

    hint = placebo_hint({}, code_text="model.fit(X, y)")
    assert hint["feasible"] is False
    assert hint["limitation"]

    hint2 = placebo_hint({}, code_text="permutation test with shuffled labels")
    assert hint2["feasible"] is True
    assert hint2["hint"]

    audit = CausalAgent().run_audit(_synthetic_dag_plan(), {}, indep_pass=True)
    assert any("placebo" in str(item).lower() or "not_feasible" in str(item).lower()
               for item in audit["limitations"])
    assert audit["passed"] is True


# ---------------------------------------------------------------------------
# Tracker case 4: 1-baseline comparative → refused with reason
# ---------------------------------------------------------------------------

def test_single_baseline_comparative_refused():
    from core.evidence_gate import (
        validate_causal_plan,
        validate_comparative_plan,
        validate_experiments,
    )

    errors = validate_experiments(_comparative_experiments(1), paper_type="comparative")
    assert errors and any(">= 2" in e and "baseline" in e.lower() for e in errors)
    assert validate_comparative_plan(_comparative_experiments(1))
    assert validate_comparative_plan(_comparative_experiments(2)) == []
    assert validate_experiments(_comparative_experiments(2), paper_type="comparative") == []
    # Legacy generic path untouched.
    assert validate_experiments(_comparative_experiments(1)) == []

    assert validate_causal_plan({}) and validate_causal_plan({"causal_spec": {"dag": {}}})
    assert validate_causal_plan({"causal_spec": {"dag": {"treatment": "t"}, "id_strategy": "rct"}}) == []
    causal_errors = validate_experiments(
        _comparative_experiments(2), paper_type="cause_effect", plan={})
    assert any("causal_spec" in e for e in causal_errors)


# ---------------------------------------------------------------------------
# Tracker case 5: pivot retains alternatives + re-stamps; hint no TypeError
# ---------------------------------------------------------------------------

def test_pivot_keeps_alternatives_and_restamps_contract(monkeypatch):
    from agents.engineer import EngineerAgent
    from core.evidence_gate import build_contract

    eng = EngineerAgent.__new__(EngineerAgent)
    eng.context = type("Ctx", (), {"config": type(
        "Cfg", (), {"experiment_seeds": 1, "output_dir": ".", "raw_results_dir": ".",
                    "engineer_max_attempts": 4})()})()
    eng._plan_revision_requests = []

    code = (
        "import numpy as np\nimport json\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "X = np.random.randn(60, 4); y = (X[:, 0] > 0).astype(int)\n"
        "m = LogisticRegression(max_iter=200).fit(X, y)\n"
        "print(json.dumps({'metrics': {'accuracy': float(m.score(X, y))}, 'raw': {}}))\n"
    )
    monkeypatch.setattr(eng, "_generate_experiment_code", lambda *_a, **_k: code)
    monkeypatch.setattr("agents.engineer.validate_code", lambda c: (True, ""))
    monkeypatch.setattr("agents.engineer.fixture_for", lambda *_a, **_k: {})
    calls = {"n": 0}

    def _flaky_multi_seed(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"success": False, "error": "no parseable metrics"}
        return {"success": True,
                "aggregate_metrics": {"accuracy": {"mean": 0.8, "std": 0.01, "values": [0.8], "n": 1}}}

    monkeypatch.setattr("agents.engineer.execute_multi_seed", _flaky_multi_seed)
    monkeypatch.setattr(eng, "_refine_code", lambda code, err, exp, local_context="", **kwargs: code)
    monkeypatch.setattr(eng, "_auto_ablation", lambda exp, code: {"main_component": {"success": True}})
    monkeypatch.setattr(eng, "_store_raw", lambda name, multi: "/tmp/raw.json")
    monkeypatch.setattr(eng, "_store_experiment_results", lambda out, exp: None)
    monkeypatch.setattr(eng, "_progress", lambda *_a, **_k: None)
    monkeypatch.setattr(
        eng, "check_code_claim_consistency",
        lambda *_a, **_k: {"consistent": True, "score": 10, "notes": []},
    )

    experiment = {"name": "pivot_exp", "paper_type": "comparative",
                  "baselines": ["logistic_regression", "random_guess"],
                  "contract_hash": build_contract({"name": "pivot_exp"})["contract_hash"]}
    caller_alternatives = [{"name": "pivot_alt", "paper_type": "comparative",
                            "baselines": ["logistic_regression", "random_guess"]}]
    result = EngineerAgent.run_experiment(eng, experiment, alternatives=caller_alternatives)
    assert result["success"] is True
    assert result["pivoted"] is True
    assert result["contract_restamped"] is True
    assert result["contract"]["contract_hash"] != experiment["contract_hash"]
    assert result["ablation_status"] == "ok"
    # L2: the caller's pool is never consumed by the run.
    assert [a["name"] for a in caller_alternatives] == ["pivot_alt"]


def test_error_category_hint_never_typeerrors():
    from agents.engineer import EngineerAgent

    for error in (None, "", "timeout after 120s", "Import blocked: os",
                  "Sandbox timeout after 5s", "tracemalloc was blocked by the sandbox",
                  "identification failed: confounder missing", "baseline mismatch",
                  12345):
        assert isinstance(EngineerAgent._error_category_hint(error), str)
    assert "workload" in EngineerAgent._error_category_hint("timeout").lower()
    assert "adjustment" in EngineerAgent._error_category_hint("confounder missing").lower()
    assert "baseline" in EngineerAgent._error_category_hint("baseline missing").lower()


def test_failure_kind_preserved_for_sandbox_tokens():
    from agents.engineer import EngineerAgent

    assert EngineerAgent._classify_failure_kind("Import blocked: os") == "sandbox_blocked"
    assert EngineerAgent._classify_failure_kind("Sandbox timeout after 5s") == "sandbox_timeout"
    assert EngineerAgent._classify_failure_kind("Container timeout after 5s") == "sandbox_timeout"
    # L4: bare prose matches must NOT steal the sandbox label.
    assert EngineerAgent._classify_failure_kind("tracemalloc was blocked by the sandbox") == "technical"
    assert EngineerAgent._classify_failure_kind("this isn't feasible") == "technical"
    assert EngineerAgent._classify_failure_kind("no parseable metrics") == "missing_metrics"
    assert EngineerAgent._is_plan_level_sandbox_block("Sandbox rejection: Import blocked: os") is True
    assert EngineerAgent._is_plan_level_sandbox_block("tracemalloc was blocked by the sandbox") is False


def test_config_attempt_budget_and_early_pivot(monkeypatch):
    from agents.engineer import EngineerAgent

    eng = EngineerAgent.__new__(EngineerAgent)
    eng.context = type("Ctx", (), {"config": type(
        "Cfg", (), {"experiment_seeds": 1, "output_dir": ".", "raw_results_dir": ".",
                    "engineer_max_attempts": 2})()})()
    eng._plan_revision_requests = []
    monkeypatch.setattr(eng, "_generate_experiment_code", lambda *_a, **_k: "x = 1")
    monkeypatch.setattr("agents.engineer.validate_code", lambda c: (True, ""))
    monkeypatch.setattr("agents.engineer.fixture_for", lambda *_a, **_k: {})
    monkeypatch.setattr("agents.engineer.execute_multi_seed",
                        lambda *_a, **_k: {"success": False, "error": "boom"})
    monkeypatch.setattr(eng, "_refine_code", lambda code, err, exp, local_context="", **kwargs: code)
    monkeypatch.setattr(eng, "_decide_pivot_or_refine", lambda *a, **k: "REFINE")
    monkeypatch.setattr(eng, "_progress", lambda *_a, **_k: None)
    monkeypatch.setattr(eng, "check_code_claim_consistency",
                        lambda *_a, **_k: {"consistent": True, "score": 10, "notes": []})
    result = EngineerAgent.run_experiment(eng, {"name": "budget_exp"})
    # L5: configured budget (2) honored instead of the hardcoded 4.
    assert len(result["decision_log"]) == 2

    decide = EngineerAgent._decide_pivot_or_refine
    assert decide(eng, {"paper_type": "comparative"}, "no parseable metrics", [{"name": "a"}], 1) == "PIVOT"
    assert decide(eng, {"paper_type": "cause_effect"}, "identification failed", [{"name": "a"}], 1) == "PIVOT"
    assert decide(eng, {"paper_type": "experimental"}, "no parseable metrics", [{"name": "a"}], 1) == "REFINE"


# ---------------------------------------------------------------------------
# L18: relative p-value + significance-flip check
# ---------------------------------------------------------------------------

def test_l18_significance_flip_blocked():
    from core.verification import verify_statistics

    # Recomputed p (~0.005) is significant; the reported 0.06 crosses
    # alpha=0.05 — a significance flip even at a modest absolute gap.
    raw = {"group_a": [0.81, 0.82, 0.80, 0.83, 0.81],
           "group_b": [0.79, 0.78, 0.80, 0.77, 0.79],
           "aggregate_metrics": {"accuracy": {"mean": 0.8, "std": 0.02}}}
    reported = {"metrics": {"accuracy": 0.8, "p_value": 0.06}}
    result = verify_statistics(reported, raw_data=raw)
    assert result["passed"] is False
    assert any("significance_flip" in m for m in result["mismatches"])


def test_l18_matching_significant_p_passes():
    from core.verification import verify_statistics, _welch_t_p

    raw = {"group_a": [0.9, 0.91, 0.89, 0.92, 0.90],
           "group_b": [0.5, 0.51, 0.49, 0.52, 0.50],
           "aggregate_metrics": {"accuracy": {"mean": 0.9, "std": 0.01}}}
    p_true = _welch_t_p(raw["group_a"], raw["group_b"])
    assert p_true is not None and p_true < 0.05
    reported = {"metrics": {"accuracy": 0.9, "p_value": p_true}}
    result = verify_statistics(reported, raw_data=raw)
    assert result["passed"], result["mismatches"]


# ---------------------------------------------------------------------------
# Planner per-type schemas (§4.5)
# ---------------------------------------------------------------------------

def test_planner_comparative_schema_and_causal_spec():
    from agents.planner import PlannerAgent

    planner = PlannerAgent.__new__(PlannerAgent)
    exps = planner._ensure_comparative_experiments(
        [{"name": "m", "baselines": ["only_one"], "evaluation_metrics": ["accuracy"]}])
    assert len(exps[0]["baselines"]) >= 2
    assert exps[0]["ablation_required"] is True

    spec = planner._build_causal_spec(
        {"title": "T"},
        {"methodology": "Random assignment trial.", "research_questions": ["Does X improve Y?"]})
    assert spec["dag"] and spec["id_strategy"] == "rct"
    naive = planner._build_causal_spec(
        {"title": "T"}, {"methodology": "Compare observed groups."})
    assert naive["id_strategy"] == "naive"


def test_governor_rewrites_associational_text():
    from agents.causal_agent import govern_text

    assert govern_text("X causes Y (ATE=0.3).", "causal") == "X causes Y (ATE=0.3)."
    governed = govern_text("X causes Y (ATE=0.3).", "associational")
    assert "causes" not in governed and "ATE" not in governed


def test_verify_comparative_ablation_compulsory():
    from core.verification import verify_comparative

    plan = {"experiments": _comparative_experiments(2)}
    assert verify_comparative(plan, {})["passed"] is True
    outputs = {"model_matrix": {"success": True, "aggregate_metrics": {"accuracy": {"mean": 0.8}},
                                "ablation": {}}}
    gate = verify_comparative(plan, outputs)
    assert gate["passed"] is False
    assert any(f["check"] == "ablation_compulsory" for f in gate["findings"])
    outputs["model_matrix"]["ablation"] = {"comp": {"success": True}}
    assert verify_comparative(plan, outputs)["passed"] is True


def test_verify_causal_governor_blocks_causal_verbs():
    from core.verification import verify_causal

    claim = {"id_strategy": "quasi", "verb_level": "associational",
             "sensitivity": "bounds hold", "placebo": {"feasible": False}}
    sections = {
        "Methods": "Identifying assumptions stated. Sensitivity bounds computed.",
        "Results": "The program causes higher scores (ATE=0.3, n=200, 95% CI).",
    }
    gate = verify_causal(claim, sections)
    assert gate["passed"] is False
    assert any(f["check"] == "verb_governor" for f in gate["findings"])
    clean = dict(sections)
    clean["Results"] = "The program is associated with higher scores (n=200, 95% CI 0.1-0.5)."
    assert verify_causal(claim, clean)["passed"] is True


# ---------------------------------------------------------------------------
# Wiring: recipes, skeletons, packs, manifests, dispatch, nodes
# ---------------------------------------------------------------------------

def test_m4_recipes_and_phases():
    from core.phase_registry import PIPELINES, is_phase_required
    from main import research_nodes

    assert PIPELINES["comparative"] == PIPELINES["experimental"]
    assert PIPELINES["cause_effect"] == ["intake", "topic_discovery", "hypothesis_debate", "planning",
                                         "engineering", "causal_audit", "independent_validation",
                                         "writing_results", "supervision", "editing"]
    assert is_phase_required("cause_effect", "causal_audit") is True
    assert is_phase_required("comparative", "causal_audit") is False
    assert is_phase_required("comparative", "engineering") is True
    nodes = research_nodes()
    assert "causal_audit" in nodes


def test_typed_graph_wires_causal_audit_and_engineering():
    """Recipe wiring: causal_audit wired for cause_effect, engineering kept for both."""
    from core.workflow import create_typed_graph
    from core.phase_registry import resolve_recipe
    from main import research_nodes

    nodes = research_nodes()
    recipe, _, _, _ = resolve_recipe("cause_effect", {})
    graph = create_typed_graph(nodes, recipe)
    assert "causal_audit" in graph.nodes, sorted(graph.nodes)
    assert "engineering" in graph.nodes, sorted(graph.nodes)
    recipe, _, _, _ = resolve_recipe("comparative", {})
    graph = create_typed_graph(nodes, recipe)
    assert "engineering" in graph.nodes, sorted(graph.nodes)
    assert "causal_audit" not in graph.nodes, sorted(graph.nodes)


def test_m4_skeletons_pack_manifest_dispatch():
    from pathlib import Path

    from agents.editor import MANUSCRIPT_ORDERS, SKELETON_TOKENS
    from agents.writer import WRITER_PACKS
    from agents.supervisor import SupervisorAgent
    from core.capabilities import DEFAULT_MANIFESTS

    for skeleton in ("latex_comparative.tex", "latex_cause_effect.tex"):
        text = (Path(__file__).resolve().parents[1] / "templates" / skeleton).read_text(encoding="utf-8")
        assert all(token in text for token in SKELETON_TOKENS), skeleton
    assert MANUSCRIPT_ORDERS["comparative"] == MANUSCRIPT_ORDERS["experimental"]
    assert MANUSCRIPT_ORDERS["cause_effect"] == MANUSCRIPT_ORDERS["experimental"]
    for paper_type in ("comparative", "cause_effect"):
        assert paper_type in WRITER_PACKS
        assert WRITER_PACKS[paper_type]["headers"] and WRITER_PACKS[paper_type]["policy"]
        assert paper_type in MANUSCRIPT_ORDERS
    for agent in ("CausalAgent", "CausalAuditAgent"):
        assert agent in DEFAULT_MANIFESTS, f"{agent} manifest missing"
        assert "code.execute" in DEFAULT_MANIFESTS[agent]["forbidden_capabilities"]

    passed, cap, _ = SupervisorAgent._per_type_section_checks(
        SupervisorAgent.__new__(SupervisorAgent),
        "Results", "Accuracy 0.9 with CI. Retrieved evidence.", "comparative", None)
    assert passed and cap == 10.0
    me = SupervisorAgent.__new__(SupervisorAgent)
    ok, _, _ = SupervisorAgent._per_type_section_checks(
        me, "Results", "The program causes gains.", "cause_effect",
        None, causal_claim={"verb_level": "associational"})
    assert ok is False
    ok2, _, _ = SupervisorAgent._per_type_section_checks(
        me, "Results", "The program causes gains.", "cause_effect",
        None, causal_claim={"verb_level": "causal"})
    assert ok2 is True


def test_planning_node_comparative_schema(monkeypatch):
    import core.workflow_nodes as wn
    from core.workflow_nodes import planning_node

    class ComparativePlanner:
        def __init__(self, *_a):
            pass

        def create_plan(self, topic, paper_type=None):
            assert paper_type == "comparative"
            return {
                "title": "T", "paper_type": "comparative",
                "sections": [{"name": "Abstract"}],
                "experiments": [{
                    "name": "m", "baselines": ["a", "b"], "evaluation_metrics": ["accuracy"],
                    "falsifiable_prediction": "p", "statistical_test": "t",
                    "variants": [{"name": "v", "methodology": "m"}]}],
            }

    monkeypatch.setattr(wn, "_create_agent", lambda cls: ComparativePlanner())
    state = _causal_state("comparative", plan=None)
    state["selected_topic"] = {"title": "T"}
    result = planning_node(state)
    assert result["plan"]["paper_type"] == "comparative"
    assert result["current_phase"] == "data_validation"


# ---------------------------------------------------------------------------
# Sandbox D-pins (D9–D16)
# ---------------------------------------------------------------------------

def test_d10_parse_before_truncate():
    from core.sandbox import _parse_json_from_stdout, truncate_to_bytes
    from core.config import config

    payload = '{"metrics": {"accuracy": 0.91}}'
    padding = "x" * (config.sandbox_max_output_bytes + 100)
    stdout = padding + "\n" + payload
    assert _parse_json_from_stdout(stdout)["metrics"]["accuracy"] == 0.91
    cut, flag = truncate_to_bytes(stdout, config.sandbox_max_output_bytes)
    assert flag is True
    assert len(cut.encode("utf-8")) <= config.sandbox_max_output_bytes + len("\n...[truncated]".encode("utf-8"))


def test_d13_shared_preamble_and_seed_param():
    from core.sandbox import build_preamble, run_known_answer_check
    from core.container_sandbox import _build_preamble

    assert _build_preamble(42) == build_preamble(42)
    assert "seed(42)" in build_preamble(42)
    code = "import json\nprint(json.dumps({'metrics': {'a': 1.0}, 'raw': {}}))"
    assert run_known_answer_check(code, {"a": 1.0}, seed=7)["passed"] is True


def test_d14_class_and_super_allowed():
    from core.sandbox import execute_sandboxed, validate_code

    code = ("class Base:\n    def value(self):\n        return 1\n"
            "class Child(Base):\n    def value(self):\n        return super().value() + 1\n"
            "import json\nprint(json.dumps({'metrics': {'v': Child().value()}, 'raw': {}}))")
    assert validate_code(code)[0] is True
    result = execute_sandboxed(code, seed=0)
    assert result["success"] is True
    assert result["parsed"]["metrics"]["v"] == 2


def test_d11_timeout_threads_through_multi_seed(monkeypatch):
    import core.sandbox as sandbox

    seen = {}

    def _fake_execute(code, timeout_sec=None, seed=42):
        seen[seed] = timeout_sec
        return {"success": True, "stdout": '{"metrics": {"a": 1}}',
                "stderr": "", "parsed": {"metrics": {"a": 1}}, "seed": seed}

    monkeypatch.setattr(sandbox, "execute_sandboxed", _fake_execute)
    out = sandbox.run_multi_seed("code", n_seeds=2, timeout_sec=33)
    assert out["success"] is True
    assert seen == {42: 33, 42 + 1009: 33}


def test_d12_timeout_poison_marks_leaked_thread():
    from core.sandbox import execute_sandboxed

    result = execute_sandboxed("import time\ntime.sleep(30)", timeout_sec=1, seed=0)
    assert result["success"] is False
    assert result.get("timeout") is True
    assert result.get("leaked_thread") is True


def test_d9_dispatch_logs_success_not_status(monkeypatch, caplog):
    import logging

    import core.sandbox_dispatch as dispatch

    monkeypatch.setattr(dispatch, "_use_docker", lambda: False)
    with caplog.at_level(logging.INFO, logger="core.sandbox_dispatch"):
        dispatch.execute("import json\nprint(json.dumps({'metrics': {'a': 1}, 'raw': {}}))", seed=0)
    finished = [r for r in caplog.records if "Sandbox execution finished" in r.message]
    assert finished and "outcome=success" in finished[0].message
    assert all("unknown" not in r.message for r in finished)


def test_d16_container_cmd_names_and_caps(monkeypatch):
    import subprocess as _subprocess

    import core.container_sandbox as cs

    captured = {}

    class _FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return ('{"metrics": {"a": 1}}', "")

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(
        cs,
        "ensure_sandbox_image",
        lambda **kwargs: {"ok": True, "backend": "docker"},
    )
    monkeypatch.setattr(_subprocess, "Popen", _fake_popen)
    result = cs.execute_containerized("code", timeout_sec=10, seed=0)
    assert result["success"] is True
    cmd = captured["cmd"]
    assert "--pids-limit" in cmd and "128" in cmd
    name_idx = cmd.index("--name") + 1
    assert cmd[name_idx].startswith("sg_")
    assert result["container_name"] == cmd[name_idx]


def test_engineering_wall_clock_exhaustion_uses_budget_taxonomy(monkeypatch):
    """Phase cap spent → budget_exhausted (locked §1.5/§5.3), partial bundle kept."""
    import core.workflow_nodes as wn
    from core.config import config

    monkeypatch.setattr(config, "engineering_wall_clock_sec", -1)
    state = _causal_state("comparative", plan={
        "paper_type": "comparative",
        "experiments": _comparative_experiments(2),
    })
    result = wn.engineering_node(state)
    assert str(result.get("terminal_error") or "").startswith("budget_exhausted")
    assert (result.get("evidence_gate") or {}).get("reason_code") == "budget_exhausted"
    assert result["current_phase"] == "complete"
    assert result["should_continue"] is False


def test_minimal_fallback_plan_carries_per_type_schema():
    """Bounded fallbacks validate under the comparative/causal schemas."""
    from core.evidence_gate import validate_experiments
    from core.workflow_nodes import _minimal_fallback_plan

    legacy = _minimal_fallback_plan({"title": "t"}, "test")
    assert "paper_type" not in legacy

    comp = _minimal_fallback_plan({"title": "t"}, "test", "comparative")
    assert validate_experiments(comp["experiments"], paper_type="comparative", plan=comp) == []

    causal = _minimal_fallback_plan({"title": "t"}, "test", "cause_effect")
    assert validate_experiments(causal["experiments"], paper_type="cause_effect", plan=causal) == []
    assert causal["causal_spec"]["id_strategy"] == "naive"
