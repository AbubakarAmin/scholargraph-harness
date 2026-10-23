"""
Tests for the OpenReview calibration dataset catalog entry.

Run: python -m pytest tests/test_datasets_openreview.py -v
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_entry_exists():
    """openreview_calibration is registered in the dataset catalog."""
    from core.datasets import DATASET_CATALOG

    assert "openreview_calibration" in DATASET_CATALOG
    entry = DATASET_CATALOG["openreview_calibration"]
    assert entry["name"] == "openreview_calibration"
    assert entry["access_policy"] == "local-only"
    assert entry["license"] == "CC-BY-4.0"
    assert entry["source"].startswith("OpenReview")


def test_list_datasets_includes_calibration():
    """list_datasets() returns the new entry."""
    from core.datasets import list_datasets

    names = {d["name"] for d in list_datasets()}
    assert "openreview_calibration" in names


def test_resolve_dataset_calibration():
    """resolve_dataset returns the calibration entry."""
    from core.datasets import resolve_dataset

    spec = resolve_dataset("openreview_calibration")
    assert spec["name"] == "openreview_calibration"
    assert "path" in spec


def test_load_calibration_empty_dir(tmp_path):
    """load_local_dataset returns empty rows when data dir doesn't exist."""
    from core.datasets import load_local_dataset

    # The real data dir likely doesn't exist in CI; the function should handle this
    result = load_local_dataset("openreview_calibration")
    assert "rows" in result
    assert "row_count" in result
    assert "manifest" in result
    # row_count may be 0 if data hasn't been ingested yet
    assert isinstance(result["rows"], list)
    assert isinstance(result["row_count"], int)


def test_load_calibration_with_papers(tmp_path, monkeypatch):
    """load_local_dataset reads JSON paper files from the calibration directory."""
    from core import datasets

    # Create a fake calibration directory structure
    cal_dir = tmp_path / "review_calibration"
    venue_dir = cal_dir / "ICLR_2025"
    venue_dir.mkdir(parents=True)

    # Write a sample paper
    paper = {
        "paper_id": "test123",
        "forum_id": "test123",
        "number": 1,
        "venue_id": "ICLR.cc/2025/Conference",
        "year": 2025,
        "title": "Test Paper",
        "authors": ["Alice", "Bob"],
        "abstract": "A test abstract.",
        "keywords": ["test"],
        "venueid": "ICLR.cc/2025/Conference/Poster",
        "reviews": [
            {
                "review_id": "rev1",
                "scores": {"rating": 8.0, "confidence": 4.0},
                "review_text": "Good paper.",
                "timestamp": 1234567890,
            }
        ],
        "meta_reviews": [],
        "decision": "Accept",
        "decision_note_id": "dec1",
    }
    (venue_dir / "test123.json").write_text(json.dumps(paper), encoding="utf-8")

    # Write a manifest
    manifest = {
        "description": "Test manifest",
        "venues": [{"label": "ICLR_2025", "paper_count": 1}],
        "total_papers": 1,
    }
    (cal_dir / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Patch the catalog path to use our tmp directory
    monkeypatch.setitem(
        datasets.DATASET_CATALOG,
        "openreview_calibration",
        {
            **datasets.DATASET_CATALOG["openreview_calibration"],
            "path": str(cal_dir),
        },
    )

    result = datasets.load_local_dataset("openreview_calibration")
    assert result["row_count"] == 1
    assert result["rows"][0]["paper_id"] == "test123"
    assert result["rows"][0]["decision"] == "Accept"
    assert result["rows"][0]["reviews"][0]["scores"]["rating"] == 8.0
    assert result["manifest"]["total_papers"] == 1


def test_load_calibration_skips_non_json(tmp_path, monkeypatch):
    """load_local_dataset skips non-JSON files in the calibration directory."""
    from core import datasets

    cal_dir = tmp_path / "review_calibration"
    venue_dir = cal_dir / "ICLR_2025"
    venue_dir.mkdir(parents=True)

    # Write a valid paper
    paper = {"paper_id": "valid", "title": "Valid"}
    (venue_dir / "valid.json").write_text(json.dumps(paper), encoding="utf-8")

    # Write a non-JSON file
    (venue_dir / "README.md").write_text("# Not a paper", encoding="utf-8")

    monkeypatch.setitem(
        datasets.DATASET_CATALOG,
        "openreview_calibration",
        {
            **datasets.DATASET_CATALOG["openreview_calibration"],
            "path": str(cal_dir),
        },
    )

    result = datasets.load_local_dataset("openreview_calibration")
    assert result["row_count"] == 1
    assert result["rows"][0]["paper_id"] == "valid"


def test_existing_catalog_unchanged():
    """Existing catalog entries are not affected by the new addition."""
    from core.datasets import DATASET_CATALOG

    assert "bundled_synthetic" in DATASET_CATALOG
    assert "sklearn_iris" in DATASET_CATALOG
    assert "sklearn_digits" in DATASET_CATALOG
    assert DATASET_CATALOG["bundled_synthetic"]["access_policy"] == "local-only"
    assert DATASET_CATALOG["sklearn_iris"]["path"] == "data/catalog/iris.csv"
