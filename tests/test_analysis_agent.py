"""Offline tests for independent statistical analysis."""

from agents.analysis import AnalysisAgent


def execution(name, values):
    return {
        "status": "completed",
        "seed_results": {
            "aggregate_metrics": {
                "accuracy": {
                    "values": values,
                    "mean": sum(values) / len(values),
                    "std": 0.0,
                    "n": len(values),
                }
            }
        },
    }


def test_analysis_agent_computes_interval_and_comparison():
    report = AnalysisAgent().analyze(
        {"baseline": execution("baseline", [0.70, 0.71, 0.69]), "method": execution("method", [0.80, 0.81, 0.79])},
        {"primary_metric": "accuracy", "confidence_level": 0.95, "statistical_test": "Welch t-test"},
    )

    assert report["passed"]
    assert report["metrics"]["method"]["accuracy"]["n"] == 3
    assert len(report["metrics"]["method"]["accuracy"]["confidence_interval_95"]) == 2
    assert report["comparisons"][0]["test"] == "Welch t-test"
    assert report["comparisons"][0]["cohens_d"] > 0
    assert report["comparisons"][0]["required_n_per_group"] >= 1
    assert "observed_power_estimate" in report["comparisons"][0]


def test_analysis_agent_warns_on_insufficient_seeds():
    report = AnalysisAgent().analyze(
        {"only": execution("only", [0.8, 0.9])},
        {"primary_metric": "accuracy"},
    )

    assert not report["passed"]
    assert any("insufficient seeds" in warning for warning in report["warnings"])
    assert any("missing comparison" in warning for warning in report["warnings"])
