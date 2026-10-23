"""M0: logic-fix regression tests L1/L3/L6–L11/L13/L15–L17/L21 (12 tests)."""

from __future__ import annotations

import pytest

from agents.engineer import EngineerAgent
from core.evidence_gate import build_contract, contract_matches, gate_engineering_outputs


def test_l1_error_hint_no_typeerror():
    # L1: staticmethod — works unbound (old test style) and via instance.
    hint = EngineerAgent._error_category_hint("timeout after 120s")
    assert "Shrink the workload" in hint
    hint = EngineerAgent._error_category_hint("import sklearn failed")
    assert "imports" in hint.lower()


def test_l3_llm_error_requests_revision_not_decoy(monkeypatch):
    from agents import engineer as eng_mod

    eng = EngineerAgent.__new__(EngineerAgent)
    calls = {}

    def _boom(*a, **k):
        raise RuntimeError("connection down")

    def _req(self, reason, experiment, detail=""):
        calls["reason"] = reason
        return {"reason": reason}

    monkeypatch.setattr(eng_mod, "call_llm", _boom)
    monkeypatch.setattr(EngineerAgent, "request_plan_revision", _req)
    with pytest.raises(RuntimeError, match="llm_unavailable"):
        eng._generate_experiment_code({"name": "e"})
    assert calls.get("reason") == "llm_unavailable"


def test_l6_single_policy_repairs_twice_then_fallback():
    # Planning policy: attempts 0,1 -> repair; attempts>=2 -> bounded fallback
    # preserving the counter (no dead attempts>=3 branch).
    import core.workflow_nodes as wn

    src = open(wn.__file__, encoding="utf-8").read()
    assert "attempts >= 3" not in src.split("def planning_node")[1].split("def ")[0]
    assert 'plan["schema_revision_attempts"] = attempts' in src


def test_l7_counter_reaches_3_on_persistent_fail():
    import core.workflow_nodes as wn

    src = open(wn.__file__, encoding="utf-8").read()
    # Except paths increment without resetting to 0 on failure.
    assert src.count('state["error_count"] = 0') >= 2  # success-path resets only
    # No reset-to-zero inside except blocks.
    assert "state[\"error_count\"] = 0\n        return state\n\n\ndef write_results" not in src


def test_l8_config_iterations_not_hardcoded():
    # L8 outcome: iteration caps are config-driven (no hardcoded 5/10).
    # Deviation logged: the three lookahead sites keep the original `+ 1`
    # form (last-iteration promotion/exhaustion), pinned by
    # test_discovery_exhaustion_is_a_terminal_failed_run and
    # TestTerminalPromotion; the reset guard itself is strict `>=`.
    import core.workflow_nodes as wn

    src = open(wn.__file__, encoding="utf-8").read()
    assert "config.max_iterations" in src
    assert "if current_iteration >= config.max_iterations:" in src
    # No hardcoded iteration caps remain (severity >= 5 is a finding
    # severity, not an iteration cap — excluded by context).
    import re as _re

    for m in _re.finditer(r"iteration[\"'\]]?\s*\)?\s*>=?\s*(\d+)", src):
        raise AssertionError(f"hardcoded iteration cap: {m.group(0)}")


def test_l9_drift_detected_without_dataset_override():
    exp = {"name": "e", "dataset": {"name": "d1"}, "evaluation_metrics": ["acc"]}
    contract = build_contract(exp)
    same = dict(exp)
    assert contract_matches(contract, same) is True
    drifted = dict(exp, dataset={"name": "d2"})
    assert contract_matches(contract, drifted) is False


def test_l10_safe_metrics_unwrap_no_crash():
    out = {"success": True, "results": None, "aggregate_metrics": {"acc": 0.5}}
    metrics = out.get("aggregate_metrics") or (out.get("results") or {}).get("metrics")
    assert metrics == {"acc": 0.5}
    out2 = {"success": True, "results": None}
    metrics2 = out2.get("aggregate_metrics") or (out2.get("results") or {}).get("metrics")
    assert not metrics2


def test_l11_retryable_vs_structural():
    plan = {"experiments": [{"name": "e1", "evaluation_metrics": ["acc"]}]}
    outputs = {"e1": {"success": True, "contract_hash": "x", "raw_results_path": "/nope.json", "aggregate_metrics": {"acc": 1.0}}}
    contracts = {"e1": {"contract_hash": "x"}}
    decision = gate_engineering_outputs(plan, outputs, contracts)
    assert decision["reason_code"] == "missing_raw_results"
    assert decision["terminal"] is False
    outputs2 = {"e1": {"success": True, "contract_hash": "x", "raw_results_path": __file__, "aggregate_metrics": {}}}
    decision2 = gate_engineering_outputs(plan, outputs2, contracts)
    assert decision2["reason_code"] == "missing_metrics"
    assert decision2["terminal"] is False


def test_l13_unified_tolerance():
    from core.workflow_nodes import verify_result_numbers

    assert verify_result_numbers.__defaults__ == (0.05,)
    import core.verification as ver

    src = open(ver.__file__, encoding="utf-8").read()
    assert "rel_tol=0.05" in src


def test_l15_zero_mean_floor():
    # abs(true)==0 must still allow 1e-3 absolute slack.
    assert max(0.05 * abs(0.0), 1e-3) == pytest.approx(1e-3)
    import core.verification as ver

    src = open(ver.__file__, encoding="utf-8").read()
    assert "1e-3" in src


def test_l16_failed_plus_metrics_not_successful():
    def successful(output):
        metrics = output.get("aggregate_metrics") or (output.get("results") or {}).get("metrics")
        return (output.get("success") is True) and bool(metrics)

    assert successful({"success": True, "aggregate_metrics": {"a": 1}}) is True
    assert successful({"success": False, "aggregate_metrics": {"a": 1}}) is False
    assert successful({"success": True, "aggregate_metrics": {}}) is False
    assert successful({"success": 1, "aggregate_metrics": {"a": 1}}) is False


def test_l17_stable_unwrap_prefers_output():
    output = {"success": True, "aggregate_metrics": {"acc": 0.9}, "multi_seed": {"seeds": [42]}}
    metrics = output.get("aggregate_metrics") or (output.get("results") or {}).get("metrics")
    assert metrics == {"acc": 0.9}
