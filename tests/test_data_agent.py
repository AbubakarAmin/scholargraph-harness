"""Offline tests for independent dataset stewardship."""

from agents.data import DataAgent


def test_data_agent_validates_and_hashes_csv(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")

    artifact = DataAgent().validate_dataset(
        str(path),
        {"name": "toy", "target": "target", "features": ["feature"]},
    )

    assert artifact["validation"]["passed"]
    assert artifact["row_count"] == 2
    assert artifact["content_hash"] == artifact["provenance"]["content_hash"]
    assert artifact["schema"]["columns"] == ["feature", "target"]


def test_data_agent_flags_target_leakage_and_missing_target(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("feature\n1\n2\n", encoding="utf-8")

    artifact = DataAgent().validate_dataset(
        str(path),
        {"name": "bad", "target": "target", "features": ["target"]},
    )

    assert not artifact["validation"]["passed"]
    assert any("target" in issue for issue in artifact["validation"]["mismatches"])


def test_data_agent_does_not_accept_unknown_formats(tmp_path):
    path = tmp_path / "dataset.txt"
    path.write_text("not tabular", encoding="utf-8")

    try:
        DataAgent().validate_dataset(str(path), {"name": "text"})
    except ValueError as exc:
        assert "Supported dataset formats" in str(exc)
    else:
        raise AssertionError("unknown dataset format was accepted")
