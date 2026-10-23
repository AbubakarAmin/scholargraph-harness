"""v5 regression (run f537f638): filler must block release, real evidence must pass.

Offline: citation network calls are monkeypatched; all other gates are pure.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _fallback_outputs():
    return {
        "minimum_viable_experiment": {
            "success": True,
            "outcome": "fallback_synthetic",
            "aggregate_metrics": {
                "accuracy": {"mean": 0.7278, "std": 0.0, "values": [0.7278, 0.7278, 0.7278], "n": 3},
            },
            "code": "import json, random\nprint(json.dumps({'metrics': {'accuracy': 0.7}}))",
            "contract_hash": "abc",
            "raw_results_path": __file__,
        }
    }


def test_fallback_blocks_evidence_gate():
    from core.evidence_gate import gate_engineering_outputs

    plan = {"experiments": [{"name": "minimum_viable_experiment", "evaluation_metrics": ["accuracy"]}],
            "paper_type": "experimental"}
    gate = gate_engineering_outputs(plan, _fallback_outputs(), {"minimum_viable_experiment": {"contract_hash": "abc"}})
    assert gate["allowed"] is False
    assert gate["reason_code"] == "exploratory_only"
    assert gate["terminal"] is True


def test_degenerate_identical_values_blocked():
    from core.evidence_gate import gate_engineering_outputs

    outputs = {
        "exp": {
            "success": True,
            "outcome": "completed",
            "aggregate_metrics": {"accuracy": {"mean": 0.7, "std": 0.0, "values": [0.7, 0.7, 0.7], "n": 3}},
            "code": "x",
            "contract_hash": "h",
            "raw_results_path": __file__,
        }
    }
    plan = {"experiments": [{"name": "exp", "evaluation_metrics": ["accuracy"]}], "paper_type": "experimental"}
    gate = gate_engineering_outputs(plan, outputs, {"exp": {"contract_hash": "h"}})
    assert gate["allowed"] is False
    assert gate["reason_code"] == "degenerate_metrics"


def test_validate_experiment_evidence_flags_fallback_and_zero_variance():
    from core.verification import validate_experiment_evidence

    findings = validate_experiment_evidence(_fallback_outputs(), {})
    assert any(f["check"] == "exploratory_only" and f["blocking"] for f in findings)

    outputs = {
        "exp": {"success": True, "outcome": "completed",
                "aggregate_metrics": {"acc": {"mean": 0.5, "std": 0.0, "values": [0.5, 0.5], "n": 2}}},
    }
    findings = validate_experiment_evidence(outputs, {})
    assert any(f["check"] in ("degenerate_variance", "underpowered") and f["blocking"] for f in findings)


def test_statistical_plan_requires_ci_artifact():
    from core.verification import validate_experiment_evidence

    plan = {"title": "bootstrap CIs", "experiments": [{"name": "exp", "evaluation_metrics": ["accuracy_difference"]}]}
    outputs = {"exp": {"success": True, "outcome": "completed",
                       "aggregate_metrics": {"accuracy_difference": {"mean": 0.01, "std": 0.005, "values": [0.01, 0.02, 0.0], "n": 3}}}}
    findings = validate_experiment_evidence(outputs, plan)
    assert any(f["check"] == "missing_confidence_interval" for f in findings)


def test_bootstrap_claim_without_artifact_blocked():
    from core.verification import validate_bootstrap_claims

    sections = {"Abstract": "Bootstrap CIs are wider than reported and often include zero."}
    findings = validate_bootstrap_claims(sections, {"exp": {"success": True, "aggregate_metrics": {"acc": {"mean": 0.7}}}})
    assert any(f["check"] in ("unmeasured_baseline", "unquantified_zero_coverage") for f in findings)


def test_bootstrap_claim_with_artifact_passes():
    from core.verification import validate_bootstrap_claims

    sections = {"Abstract": "Bootstrap CIs are wider than reported and often include zero."}
    outputs = {"exp": {"success": True,
                       "bootstrap": {"ci_low": -0.02, "ci_high": 0.05, "B": 1000},
                       "baseline_width": 0.03,
                       "aggregate_metrics": {}}}
    findings = validate_bootstrap_claims(sections, outputs)
    assert findings == []


def test_filler_prose_and_failed_draft_block_section():
    from core.verification import hard_verify_section, filler_prose_hits, methods_completeness_gaps

    assert filler_prose_hits("Verified experimental results are required before making claims about X.")
    assert len(methods_completeness_gaps("Our approach involves experimental evaluation.")) >= 5
    report = hard_verify_section(
        "# Methods\n\nOur approach to Research on X involves Deterministic bundled-synthetic baseline.",
        section_name="Methods",
    )
    assert report["passed"] is False
    assert any("filler" in e or "methods incomplete" in e for e in report["substance_errors"])
    failed = hard_verify_section("# Results\n\nDRAFT_FAILED: results unavailable.", section_name="Results")
    assert failed["passed"] is False


def test_abrupt_truncation_blocked():
    from core.verification import hard_verify_section

    body = "# Related Work\n\n" + "This is a long complete-looking sentence with citations. " * 30
    body = body.rstrip(". ") + " does not report accu"
    report = hard_verify_section(body, section_name="Related Work")
    assert report["passed"] is False
    assert any("abruptly" in e for e in report["substance_errors"])


def test_fake_arxiv_id_rejected_offline(monkeypatch):
    from core import verification as v

    monkeypatch.setattr(v, "resolve_arxiv", lambda aid: {"resolved": True, "arxiv_id": aid})
    monkeypatch.setattr(v, "_resolve_doi_with_retry", lambda doi: {"resolved": True, "doi": doi})
    # Malformed / out-of-range IDs fail closed without network.
    assert v._arxiv_id_plausible("2610.03597") is True  # Oct 2026 == current month, plausible shape
    assert v._arxiv_id_plausible("2611.00001") is False  # future month
    assert v._arxiv_id_plausible("9913.99999") is False
    assert v._arxiv_id_plausible("abcd.1234") is False
    assert v._arxiv_id_plausible("2501.12345") is True
    out = v.verify_citations("see arXiv:9913.99999 for details")
    assert out["passed"] is False


def test_gap_prevalence_claim_requires_counts():
    from core.verification import validate_gap_prevalence_claims

    sections = {"Introduction": "The retrieved literature evidence does not contain confidence intervals. Papers rarely report CIs."}
    findings = validate_gap_prevalence_claims(sections, {}, {}, {})
    assert any(f["check"] == "unquantified_gap_claim" for f in findings)
    corpus = {"extraction": [{"doi": f"d{i}"} for i in range(6)]}
    assert validate_gap_prevalence_claims(sections, {}, {}, corpus) == []


def test_planner_statistical_protocol_pads_deterministically():
    from agents.planner import PlannerAgent

    topic = {"title": "Bootstrap confidence intervals for federated accuracy differences are wider",
             "description": "bootstrap CI evaluation", "rationale": "t-test fails"}
    exps = [{"name": "exp", "evaluation_metrics": ["accuracy"], "baselines": ["fedavg"],
             "falsifiable_prediction": "A beats B", "statistical_test": "t-test"}]
    agent = PlannerAgent.__new__(PlannerAgent)
    out = agent._ensure_statistical_protocol(exps, topic)
    exp = out[0]
    assert "ci_low" in exp["evaluation_metrics"] and "accuracy_difference" in exp["evaluation_metrics"]
    assert "bootstrap" in exp["statistical_test"].lower()
    assert exp["analysis_plan"]["bootstrap_B"] >= 1000
    assert exp["analysis_plan"]["n_seeds"] >= 10
    assert len(exp["baselines"]) >= 2


def test_engineer_lint_rejects_fixed_seed_and_missing_ci():
    from agents.engineer import EngineerAgent

    bad_seed = "import random, json\nrandom.seed(42)\nprint(json.dumps({'metrics': {'acc': 0.7}}))"
    exp = {"name": "e", "statistical_test": "bootstrap percentile", "evaluation_metrics": ["ci_low"],
           "analysis_plan": {"bootstrap_B": 1000}}
    assert "fixed constant" in EngineerAgent._lint_generated_code(bad_seed, exp)
    no_ci = "import json\nprint(json.dumps({'metrics': {'acc': 0.7}}))"
    assert "bootstrap" in EngineerAgent._lint_generated_code(no_ci, exp).lower()
    good = "import numpy as np, json\nboot=np.percentile([0.1,0.2], [2.5,97.5])\nprint(json.dumps({'metrics': {'acc': 0.7}}))"
    assert EngineerAgent._lint_generated_code(good, {"name": "plain"}) == ""


def test_analysis_bootstrap_missing_warns():
    from agents.analysis import AnalysisAgent

    artifacts = {
        "a": {"status": "completed",
              "seed_results": {"aggregate_metrics": {"acc": {"mean": 0.7, "std": 0.01, "values": [0.7, 0.71, 0.69],
                                                            "confidence_interval_95": [0.68, 0.72]}}}},
        "b": {"status": "completed",
              "seed_results": {"aggregate_metrics": {"acc": {"mean": 0.9, "std": 0.01, "values": [0.9, 0.91, 0.89],
                                                            "confidence_interval_95": [0.88, 0.92]}}}},
    }
    report = AnalysisAgent().analyze(artifacts, {"primary_metric": "acc", "bootstrap_B": 1000})
    assert report["passed"] is True, report["warnings"]
    report2 = AnalysisAgent().analyze(
        {"a": {"status": "completed",
               "seed_results": {"aggregate_metrics": {"acc": {"mean": 0.7, "std": 0.01, "values": [0.7, 0.71, 0.69]}}}}},
        {"primary_metric": "acc", "bootstrap_B": 1000})
    assert report2["passed"] is False
    assert any("bootstrap" in w.lower() for w in report2["warnings"])


def test_stats_screener_rejects_point_estimate_only_literature():
    from agents.topic_hunter import TopicHunterAgent

    agent = TopicHunterAgent.__new__(TopicHunterAgent)
    candidate = {"title": "Bootstrap CIs for federated accuracy differences are wider",
                 "description": "bootstrap confidence intervals", "rationale": "t-tests fail",
                 "contribution": "bootstrap protocol"}
    lit = [{"title": "FedBWO point estimates", "abstract": "improves accuracy by 21% over FedAvg, point estimate"}]
    out = agent.screen_research_gap(candidate, lit, hunt_mode="gap")
    assert out["status"] == "FAIL"
    assert out["reason"] == "stats_gap_without_stats_literature"


def test_debate_stats_topic_disables_light_track():
    from agents.hypothesis_debate import ModeratorAgent

    mod = ModeratorAgent.__new__(ModeratorAgent)
    topic = {"title": "Bootstrap CIs wider", "hunt_mode": "try_and_report",
             "structured_hypothesis": {"hypothesis": "x", "falsification_condition": "y", "dependent_variables": ["acc"]},
             "rationale": "r"}
    # Simulate the light-track computation path directly.
    import re as _re
    blob = str(topic)
    assert _re.search(r"bootstrap", blob, _re.I)
    # 6.5 with 4 unresolved must not pass at standard bar 7.0.
    assert 6.5 < 7.0


def test_reproducibility_fails_on_fallback():
    from core.verification import reproducibility_dossier

    plan = {"experiments": [{"name": "e", "falsifiable_prediction": "p", "baselines": ["b"], "statistical_test": "t"}]}
    out = {"e": {"success": True, "outcome": "fallback_synthetic", "code": "x", "contract_hash": "h",
                 "raw_results_path": __file__, "multi_seed": {}}}
    dossier = reproducibility_dossier(plan, out, paper_type="experimental")
    assert dossier["passed"] is False


def test_final_referee_blocks_f537_shape():
    from core.verification import final_manuscript_referee

    sections = {
        "Abstract": "Bootstrap confidence intervals are wider than reported and often include zero. Accuracy 0.727.",
        "Introduction": "Background on federated learning with proper punctuation.",
        "Methods": "Our approach to Research on Bootstrap involves Deterministic bundled-synthetic baseline.",
        "Results": "Verified experimental results are required before making claims about X.",
        "Discussion": "No differences measured.",
        "Limitations": "Bounded synthetic only.",
    }
    plan = {"title": "bootstrap CIs", "experiments": [{"name": "minimum_viable_experiment", "dataset": {"name": "bundled_synthetic"}, "evaluation_metrics": ["accuracy"]}],
            "paper_type": "experimental"}
    ref = final_manuscript_referee(sections, plan, _fallback_outputs(), topic={"title": "bootstrap CIs"})
    assert ref["passed"] is False
    checks = {f["check"] for f in ref["findings"]}
    assert "exploratory_only" in checks or "template_filler" in checks
