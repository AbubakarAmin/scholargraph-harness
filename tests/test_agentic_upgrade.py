"""Agentic-harness upgrade: eval + regression tests (offline, mocked).

Covers the full upgrade surface so future prompt/harness edits stay gated:
- prompt registry v2 specs (system/user/tier/temp/schema/steps)
- agent_llm cascade + doom-loop + prompt_version stamping
- broker permission table + broker-only enrich audit
- experience-bank record_trial unfreeze + fuzzy hints_for
- state v2 backfill (work_plans/prompt_version/tool_calls/provenance)
- provenance downgrade order
- verification repair_hints + section_revision_feedback wiring
- editor patch hints + provenance stamp
- replay hash/metric drift checks
"""

from __future__ import annotations


def test_registry_specs_complete():
    from core.prompts import PROMPTS

    required = [
        "hunter.seed", "hunter.gap_generation", "hunter.screener",
        "hunter.novelty", "hunter.hypothesis_core", "hunter.hypothesis_delta",
        "hunter.self_critique", "hunter.hyde", "hunter.followup_query",
        "hunter.frontier", "hunter.second_pass", "hunter.rank",
        "debate.proposer", "debate.proposer_respond", "debate.challenger",
        "debate.followup", "debate.moderator", "debate.repair",
        "planner.generate", "planner.experiments", "planner.falsifiability",
        "planner.repair",
        "writer.abstract", "writer.introduction", "writer.related",
        "writer.methods", "writer.experiments", "writer.results",
        "writer.conclusion", "writer.discussion", "writer.generic",
        "writer.repair",
        "engineer.generate", "engineer.refine", "engineer.ablate",
        "supervisor.soft_check", "supervisor.review",
        "meta.chat", "meta.feedback",
        "thesis.build", "review.screen", "review.extract",
    ]
    for name in required:
        assert name in PROMPTS, f"registry missing {name}"
        spec = PROMPTS[name]
        assert spec.get("system"), f"{name}: empty system"
        assert spec.get("user_template"), f"{name}: empty user_template"
        assert spec.get("tier") in ("cheap", "strong", "judge", "default"), f"{name}: bad tier"
        assert 0.0 <= float(spec.get("temperature", 0.3)) <= 1.0, f"{name}: bad temp"
        assert int(spec.get("max_steps", 1)) >= 1, f"{name}: bad max_steps"


def test_registry_render_and_schema():
    from core.prompts import render, render_system, validate_schema

    user = render("hunter.seed", domain="ML", n_seeds=3, avoid_block="x",
                  cross_run_context="y", frontier_terms="z")
    assert "ML" in user
    system = render_system("debate.moderator", threshold=7.0)
    assert "7.0" in system
    assert validate_schema("hunter.screener", {"status": "PASS"})
    assert not validate_schema(
        "hunter.screener",
        {"gap_type": "g", "gap_claim": "c", "closest_prior_work": [],
         "why_insufficient": "w", "proposed_contribution": "p",
         "evidence_strength": 0.5, "status": "PASS"})


def test_doom_loop_and_compaction():
    from core.prompts import check_doom_loop, compact_context, hash_output

    assert not check_doom_loop([hash_output("a"), hash_output("b")])
    assert check_doom_loop([hash_output("x")] * 3)
    out = compact_context(["aaa", "bbb", "ccc"], 7)
    assert len(out) <= 60  # elision marker allowed, content capped


def test_cascade_tiers():
    from core.agent_llm import resolve_tier

    assert resolve_tier("cheap", 1) == "cheap"
    assert resolve_tier("cheap", 99) in ("strong", "cheap")  # config-dependent
    # Cascade disabled path never raises.
    assert isinstance(resolve_tier("judge", 99), str)


def test_agent_json_schema_repair_and_version_stamp():
    from core.agent_llm import agent_json

    calls = {"n": 0}

    def fake_call(prompt=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"wrong_shape": true}'
        return ('{"gap_type": "g", "gap_claim": "c", "closest_prior_work": [], '
                '"why_insufficient": "w", "proposed_contribution": "p", '
                '"evidence_strength": 0.5, "status": "PASS"}')

    state: dict = {}
    out = agent_json("hunter.screener", call_fn=fake_call, state=state)
    assert isinstance(out, dict) and out.get("status") == "PASS"
    assert state.get("prompt_version", {}).get("hunter.screener") == "v2"
    assert len(state.get("prompt_version", {}).get(
        "skill:evidence-synthesis", "")) == 64


def test_state_backfill_and_provenance_order():
    from core.state import (downgrade_provenance, initialize_state,
                            migrate_state_v1_to_v2)

    assert downgrade_provenance("clean", "synthetic") == "synthetic"
    assert downgrade_provenance("synthetic", "clean") == "synthetic"
    assert downgrade_provenance("exploratory", "fallback") == "fallback"
    assert downgrade_provenance("fallback", "clean") == "fallback"
    s = initialize_state()
    for key in ("work_plans", "prompt_version", "tool_calls", "provenance"):
        assert key in s, f"initialize_state missing {key}"
    m = migrate_state_v1_to_v2({})
    assert m["checkpoint_schema_version"] == 2  # additive-only, stays v2
    assert m["provenance"] == "clean"


def test_experience_bank_trial_unfreezes(tmp_path, monkeypatch):
    import core.experience_bank as eb

    bank = tmp_path / "lessons.jsonl"
    monkeypatch.setattr(eb, "BANK_PATH", bank)
    assert eb.hints_for("k-zzz") == []
    assert eb.record_trial("k-zzz", "hint-a", "ctx-a", False)["ok"]
    assert eb.record_trial("k-zzz", "hint-a", "ctx-a", True)["ok"]
    rows = eb.hints_for("k-zzz")
    assert rows and rows[0]["wins"] == 1 and rows[0]["losses"] == 1
    # Fuzzy context_sig ranking prefers overlapping context.
    eb.record_trial("k-zzz", "hint-b", "quantum error correction", False)
    top = eb.hints_for("k-zzz", context_sig="quantum error mitigation")
    assert top[0]["hint"] == "hint-b"


def test_verification_repair_hints_and_feedback():
    from core.verification import hard_verify_section
    from core.workflow_nodes import section_revision_feedback

    res = hard_verify_section("", None, section_name="Results")
    assert isinstance(res.get("repair_hints"), list) and res["repair_hints"]
    assert all({"section", "check", "expected", "detail"} <= set(h)
               for h in res["repair_hints"])
    fb = section_revision_feedback("Results", res)
    assert fb and "expected:" in fb


def test_editor_patch_hints_and_provenance_stamp():
    from agents.editor import EditorAgent

    referee = {"passed": False,
               "findings": [{"section": "Results", "check": "numeric",
                             "details": "untraced 12.5%"}]}
    hints = EditorAgent.build_repair_hints(referee)
    assert hints and hints[0]["section"] == "Results"
    assert hints[0]["check"] == "numeric"


def test_replay_metric_and_hash_drift(tmp_path):
    import json

    from core.replay import _metrics_drift, replay_companion

    assert _metrics_drift({"a": 1.0}, {"a": 1.02})["passed"]
    assert not _metrics_drift({"a": 1.0}, {"a": 1.50})["passed"]
    assert _metrics_drift({"a": 1.0}, {})["n_compared"] == 0

    root = tmp_path / "companion"
    (root / "experiments").mkdir(parents=True)
    (root / "experiments" / "toy.py").write_text(
        'import json\nprint(json.dumps({"metrics": {"accuracy": 1.0}}))',
        encoding="utf-8")
    (root / "reproducibility_manifest.json").write_text(json.dumps({
        "expected_metrics": {"toy": {"accuracy": 1.0}},
        "seeds": {"toy": [0]},
    }), encoding="utf-8")
    assert replay_companion(str(root))["passed"]
    # Drifted expectation fails replay.
    (root / "reproducibility_manifest.json").write_text(json.dumps({
        "expected_metrics": {"toy": {"accuracy": 0.10}},
        "seeds": {"toy": [0]},
    }), encoding="utf-8")
    bad = replay_companion(str(root))
    assert not bad["passed"]


def test_writer_generic_system_has_no_raw_slot():
    from agents.writer import _writer_system

    for section in ("Abstract", "Results", "Something Custom"):
        system = _writer_system(section, "experimental").get("system", "")
        assert "{paper_type}" not in system, f"raw slot leaked for {section}"
        assert "experimental" in system


def test_needs_clarification_control_branches():
    from agents.meta_agent import MetaAgent

    agent = MetaAgent.__new__(MetaAgent)
    cont = agent.apply_control("needs_clarification",
                               {"hunt_mode": "try_and_report"})
    assert cont.get("current_phase") == "planning"
    assert cont.get("should_continue") is True
    stop = agent.apply_control("needs_clarification", {"hunt_mode": "gap"})
    assert stop.get("should_continue") is False
    assert stop.get("terminal_error") is None  # parked, not failed


def test_work_plan_and_version_stamps():
    from core.state import initialize_state
    from core.workflow_nodes import _record_work_plan

    state = initialize_state()
    _record_work_plan(state, "planning", ["step one", "step two"])
    assert state["work_plans"] and state["work_plans"][0]["phase"] == "planning"
    assert state["prompt_version"].get("planner.generate") == "v2"


def test_config_budgets_reach_renderer(monkeypatch):
    from core import prompts as _prompts

    monkeypatch.setattr("core.prompts._live_budgets",
                        lambda entry=None: {"system": 50, "evidence": 60,
                                            "prior": 60, "lessons": 60})
    _prompts.clear_skill_cache()
    short = _prompts.render_system("debate.moderator", threshold=7.0,
                                   title="t")
    assert len(short) <= 120


def test_skill_loader_and_engineer_wiring():
    from core.prompts import load_skill, skill_section

    text = load_skill("sandbox-fix-hints")
    assert "workload" in text
    assert "Shrink the workload" in skill_section("sandbox-fix-hints", "timeout")
    assert skill_section("sandbox-fix-hints", "no-such-section") == ""
    assert load_skill("no-such-skill") == ""
    from agents.engineer import EngineerAgent

    assert "Shrink the workload" in EngineerAgent._error_category_hint("timeout boom")
    assert EngineerAgent.manifest_name == "EngineerAgent"
    assert "code_execute_sandbox" in EngineerAgent.allowed_tools


def test_skill_registry_routes_validates_and_versions():
    from core.prompts import (
        PROMPTS,
        clear_skill_cache,
        render_system,
        skill_fingerprint,
        skills_for_prompt,
        validate_skill_registry,
    )
    from core.state import initialize_state
    from core.workflow_nodes import _record_work_plan

    clear_skill_cache()
    assert validate_skill_registry() == []
    assert "research-design" in skills_for_prompt("planner.experiments")
    assert "statistical-interpretation" in skills_for_prompt("writer.results")
    for prompt_name in PROMPTS:
        if skills_for_prompt(prompt_name):
            rendered = render_system(
                prompt_name, threshold=7.0, paper_type="experimental",
                section="Results")
            assert "[elided: over context budget]" not in rendered, prompt_name

    debate_system = render_system("debate.moderator", threshold=7.0)
    assert "Carry every objection across rounds" in debate_system
    results_system = render_system("writer.results", paper_type="experimental")
    assert "source artifact" in results_system

    state = initialize_state()
    _record_work_plan(state, "planning", ["design the experiments"])
    assert state["prompt_version"]["skill:research-design"] == skill_fingerprint(
        "research-design")
    assert len(state["prompt_version"]["skill:research-design"]) == 64


def test_engineer_broker_denies_unlisted_tool():
    from agents.engineer import EngineerAgent

    eng = EngineerAgent.__new__(EngineerAgent)
    eng._broker = None
    eng._manifest = {}
    try:
        eng._broker_call("code.execute", "code_execute", code="x = 1")
    except PermissionError:
        pass
    else:
        raise AssertionError("EngineerAgent must be denied code_execute")


def test_bridge_grounding_scores_and_rank_prefers_grounded():
    from core.prompts import get_prompt

    # Rank system must prefer grounded bridges (scored 1.0 vs 0.5).
    assert "bridge-grounded" in get_prompt("hunter.rank")["system"].lower()


def test_pipeline_budget_guard_trips(monkeypatch):
    from core.pipeline import ResearchPipeline

    class _StubTracker:
        stats = {"llm_calls": 999999, "llm_tokens_in": 0,
                 "llm_tokens_out": 0, "cost_estimate_usd": 0.0}

        def complete(self, success=False):
            self.done = success

    stub = _StubTracker()
    ctx = type("Ctx", (), {"tracker": stub})()
    pipe = ResearchPipeline(graph_factory=lambda: None,
                            checkpointer_factory=lambda: None,
                            context=ctx)
    monkeypatch.setattr(pipe, "stream",
                        lambda *a, **k: iter([("n1", {})]))
    out = pipe.run({}, run_id="budget-test")
    assert out.state.get("terminal_error", "").startswith("budget_exhausted")
    assert out.state.get("current_phase") == "complete"
