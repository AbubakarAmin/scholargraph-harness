import json


def test_dataset_catalog_is_local_only():
    from core.datasets import list_datasets, load_local_dataset, resolve_dataset

    assert list_datasets()
    assert resolve_dataset("bundled_synthetic")["access_policy"] == "local-only"
    assert load_local_dataset("sklearn_iris")["row_count"] > 0


def test_replay_companion_runs_experiment(tmp_path):
    from core.replay import replay_companion

    root = tmp_path / "companion"
    (root / "experiments").mkdir(parents=True)
    (root / "reproducibility_manifest.json").write_text(json.dumps({"git_commit": "test"}), encoding="utf-8")
    (root / "experiments" / "toy.py").write_text(
        'import json\nprint(json.dumps({"metrics": {"accuracy": 1.0}}))', encoding="utf-8"
    )
    result = replay_companion(str(root))
    assert result["passed"]


def test_forensic_report_contains_claim_lineage(tmp_path):
    from core.forensics import build_forensic_report
    from core.research_db import ResearchDatabase

    db = ResearchDatabase(str(tmp_path / "ledger.sqlite"))
    db.create_run("r1", "2026-01-01T00:00:00Z")
    db.record_artifact("r1", "raw", "raw.json", {"artifact_id": "a1"})
    db.record_claim("r1", "Results", "accuracy 1.0", "empirical_result", "verified", {"artifact_ids": ["a1"]})
    report = build_forensic_report("r1", db)
    assert report["summary"]["claim_count"] == 1
    assert report["lineage"][0]["artifacts"]


def test_known_answer_fixture_registry():
    from core.known_answers import fixture_for

    fixture = fixture_for({"known_answer_type": "identity_transform"})
    assert fixture["metrics"]["identity_error"] == 0.0


def test_power_preregistration_and_checklist():
    from core.verification import preregister_power, validate_reviewer_checklist

    power = preregister_power(0.5)
    assert power["required_n_per_group"] > 0
    checklist = validate_reviewer_checklist(
        {"Limitations": "Limited scope.", "Results": "accuracy n=3 std=0.1"},
        {"toy": {"success": True, "outcome": "negative", "aggregate_metrics": {"accuracy": {"mean": 0.5}}}},
        {"experiments": [{"baselines": ["baseline"]}]},
    )
    assert checklist["passed"]


def test_historical_report_handles_empty_ledger(tmp_path):
    from core.forensics import build_historical_report
    from core.research_db import ResearchDatabase

    report = build_historical_report(ResearchDatabase(str(tmp_path / "empty.sqlite")))
    assert report["failed_run"] is None
    assert report["completed_run"] is None