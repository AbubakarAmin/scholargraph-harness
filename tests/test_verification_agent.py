"""Offline tests for independent verification."""

import hashlib
import json

from agents.verification import VerificationAgent


def test_verifier_accepts_matching_raw_statistics(tmp_path):
    raw = {"aggregate_metrics": {"accuracy": {"mean": 0.8, "std": 0.1, "values": [0.7, 0.8, 0.9], "n": 3}}}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    findings = VerificationAgent().verify(
        {"toy": {"status": "completed", "raw_results_path": str(path), "content_hash": digest}},
        {"toy": {"metrics": {"accuracy": {"mean": 0.8, "std": 0.1}}}},
    )

    assert findings == []


def test_verifier_blocks_hash_and_statistical_mismatches(tmp_path):
    raw = {"aggregate_metrics": {"accuracy": {"mean": 0.8, "std": 0.1, "values": [0.7, 0.8, 0.9], "n": 3}}}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    findings = VerificationAgent().verify(
        {"toy": {"status": "completed", "raw_results_path": str(path), "content_hash": "wrong"}},
        {"toy": {"metrics": {"accuracy": {"mean": 0.99, "std": 0.1}}}},
    )

    assert len(findings) == 2
    assert all(finding["blocking"] for finding in findings)