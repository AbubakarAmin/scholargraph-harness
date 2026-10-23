"""Offline regression tests for run e0ec9e91 (no API keys, no network)."""

from agents.topic_hunter import _seed_passes_domain_gate
from core.capabilities import _gpu_required_by_text, check_plan_feasibility
from core.evidence_synthesis import validate_topic_admission


def test_se_domain_gate_vetoes_ml_seeds():
    assert not _seed_passes_domain_gate("hierarchical world models visual planning", "software engineering")
    assert not _seed_passes_domain_gate("on-policy distillation sequence power", "software engineering")
    assert _seed_passes_domain_gate("coverage-guided fuzzing branch distance benchmark comparison", "software engineering")
    assert _seed_passes_domain_gate("program repair patch correctness benchmark comparison", "software engineering")
    # Non-SE domains pass through ungated.
    assert _seed_passes_domain_gate("hierarchical world models visual planning", "machine learning")


def test_gpu_checker_ignores_output_and_false_flag():
    assert not _gpu_required_by_text("probabilistic and coherent output, no explicit post-processing")
    assert not _gpu_required_by_text('required_resources: {"gpu": false}')
    assert _gpu_required_by_text("train with distributed training on 8 GPUs")


def test_structured_hypothesis_gpu_text_gets_sanitized():
    from agents.planner import PlannerAgent
    plan = {
        "title": "t",
        "methodology": "CPU study",
        "structured_hypothesis": {"hypothesis": "Uses distributed training on 8 GPUs"},
        "experiments": [{"name": "mve", "dataset": {"name": "bundled_synthetic"}}],
    }
    PlannerAgent._auto_fix_manifest_violations(plan, __import__("core.capabilities", fromlist=["SANDBOX_CAPABILITY_MANIFEST"]).SANDBOX_CAPABILITY_MANIFEST)
    assert check_plan_feasibility(plan) == []


def test_admission_warns_on_thin_replication():
    out = validate_topic_admission({
        "research_question": "q",
        "hypothesis": "h",
        "dependent_variables": ["accuracy"],
        "falsification_condition": "it gets better",
        "minimum_viable_experiment": {
            "dataset": "bundled_synthetic", "baseline": "majority_class",
            "metrics": ["accuracy"], "falsification_test": "t-test", "seeds": 1,
        },
        "gap_report": {"status": "INSUFFICIENT_EVIDENCE"},
    })
    assert out["admitted"]
    assert any("seed" in w for w in out.get("warnings", []))
    assert any("threshold" in w or "numeric" in w for w in out.get("warnings", []))


def test_repair_prompt_uses_compact_caps():
    import inspect
    from agents import hypothesis_debate as hd
    src = inspect.getsource(hd.HypothesisDebateSystem.revise_topic_from_objections)
    assert "max_tokens=16384" in src
    assert "[:2500]" in src


def test_planning_node_dedupes_revisions():
    import inspect
    from core import workflow_nodes as wn
    src = inspect.getsource(wn.planning_node)
    assert "planning_revision_deduped" in src
    assert "planning_revision_infra_skipped" in src


def test_golden_offline_chain_poisoned_plan_to_executable():
    """P2-14 (golden path, offline): a GPU-poisoned plan like e0ec9e91's
    rejected plan coerces through Tier-0 + sanitizer to a feasible,
    schema-valid plan with no filler variants — no LLM needed."""
    from agents.planner import PlannerAgent
    from core.capabilities import SANDBOX_CAPABILITY_MANIFEST
    from core.evidence_gate import validate_experiments
    planner = PlannerAgent.__new__(PlannerAgent)
    plan = {
        "title": "Replication study",
        "methodology": "End-to-end coherent forecast using distributed training on 8 GPUs",
        "structured_hypothesis": {
            "hypothesis": "GPU-oriented transformer beats baselines",
            "gap_report": {"status": "ok"},
        },
        "sections": [{"name": "Methods", "content_requirements": "CPU study"}],
        "experiments": [{
            "name": "minimum_viable_experiment",
            "methodology": "Train with distributed training on 8 GPUs",
            "dataset": {"name": "bundled_synthetic", "access_policy": "local-only"},
        }],
        "revision_history": [],
    }
    planner._force_tier0_plan(plan, {"title": "Replication study"}, ["gpu"])
    PlannerAgent._auto_fix_manifest_violations(plan, SANDBOX_CAPABILITY_MANIFEST)
    assert check_plan_feasibility(plan) == []
    assert validate_experiments(plan.get("experiments"), plan=plan) == []
    # No filler variants invented for the engineer to probe as real.
    assert plan["experiments"][0].get("variants") in ([], None) or all(
        isinstance(v, dict) and "simpler model" not in str(v.get("description", ""))
        for v in plan["experiments"][0].get("variants", []))


def test_debate_replication_family_threshold():
    """P1-7 family-aware bar: bounded replications pass at 6.0, novel-method
    claims still need the configured 7.0."""
    import inspect
    from agents import hypothesis_debate as hd
    src = inspect.getsource(hd.ModeratorAgent.evaluate_debate)
    assert "_is_replication_family" in src
    assert "min(_threshold, 6.0)" in src


def test_contradiction_miner_chunks_and_fuzzy_matches():
    import inspect
    from core import contradiction_mining as cm
    src = inspect.getsource(cm.find_contradictions)
    assert "chunks" in src
    assert "0.60" in src


def test_gateway_fails_fast_on_gate():
    # Fail-fast lives in LLMClient.chat (pre-check + fallback skip), NOT in
    # APIGateway._request_inner — the gateway must keep executing gated
    # requests so a success probe can clear the gate (pinned by
    # test_success_clears_throttle_gate).
    import inspect
    from core import llm as llm_mod
    src = inspect.getsource(llm_mod.LLMClient.chat)
    assert 'is_available("llm")' in src
    assert "fallbacks skipped" in src or "fallback" in src.lower()
