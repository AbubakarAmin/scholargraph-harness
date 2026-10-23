"""Regression: _parse_json_from_stdout must handle multi-line JSON and warnings."""

from __future__ import annotations

import json

import pytest

from core.sandbox import _parse_json_from_stdout


class TestParseJsonFromStdout:
    """Robust JSON extraction from experiment stdout."""

    def test_single_line_json(self):
        stdout = '{"metrics": {"accuracy": 0.85, "f1": 0.82}}\n'
        result = _parse_json_from_stdout(stdout)
        assert result["metrics"]["accuracy"] == 0.85

    def test_json_with_warning_prefix(self):
        stdout = (
            "WARNING: something sklearn\n"
            "FutureWarning: deprecated\n"
            '{"metrics": {"accuracy": 0.91}}\n'
        )
        result = _parse_json_from_stdout(stdout)
        assert result["metrics"]["accuracy"] == 0.91

    def test_multiline_indented_json(self):
        """LLM-generated code using print(json.dumps(..., indent=2))."""
        data = {"metrics": {"accuracy": 0.87, "f1": 0.84}}
        stdout = json.dumps(data, indent=2) + "\n"
        result = _parse_json_from_stdout(stdout)
        assert result["metrics"]["accuracy"] == 0.87

    def test_multiline_json_with_warning_prefix(self):
        data = {"metrics": {"accuracy": 0.79}}
        stdout = (
            "Loading library...\n"
            "ConvergenceWarning: did not converge\n"
            + json.dumps(data, indent=2)
            + "\n"
        )
        result = _parse_json_from_stdout(stdout)
        assert result["metrics"]["accuracy"] == 0.79

    def test_json_with_trailing_text(self):
        stdout = '{"metrics": {"accuracy": 0.88}}\nExperiment complete.\n'
        result = _parse_json_from_stdout(stdout)
        assert result["metrics"]["accuracy"] == 0.88

    def test_empty_stdout(self):
        assert _parse_json_from_stdout("") == {}
        assert _parse_json_from_stdout("   ") == {}

    def test_no_json_in_stdout(self):
        stdout = "Just some plain text output\nNo JSON here\n"
        assert _parse_json_from_stdout(stdout) == {}

    def test_json_without_metrics_wrapper(self):
        """Some code may print flat JSON without 'metrics' key."""
        stdout = '{"accuracy": 0.85, "f1": 0.82}\n'
        result = _parse_json_from_stdout(stdout)
        assert result["accuracy"] == 0.85

    def test_multiline_json_with_other_text_between(self):
        """Multi-line JSON preceded by interleaved warnings and progress."""
        data = {"metrics": {"accuracy": 0.72}}
        stdout = (
            "Step 1/5 complete\n"
            "WARNING: something\n"
            "Step 2/5 complete\n"
            "DeprecationWarning: old API\n"
            + json.dumps(data, indent=2)
            + "\n"
        )
        result = _parse_json_from_stdout(stdout)
        assert result["metrics"]["accuracy"] == 0.72

    def test_single_line_json_takes_precedence_over_multiline(self):
        """If both single-line and multi-line JSON exist, prefer single-line (more specific)."""
        stdout = (
            '{"metrics": {"accuracy": 0.95}}\n'
            + json.dumps({"metrics": {"accuracy": 0.10}}, indent=2)
            + "\n"
        )
        result = _parse_json_from_stdout(stdout)
        # Last single-line JSON wins (scanned from bottom)
        assert result["metrics"]["accuracy"] == 0.95

    def test_deeply_nested_json(self):
        data = {"metrics": {"accuracy": 0.81, "details": {"by_class": {"A": 0.9, "B": 0.7}}}}
        stdout = json.dumps(data, indent=2) + "\n"
        result = _parse_json_from_stdout(stdout)
        assert result["metrics"]["accuracy"] == 0.81
        assert result["metrics"]["details"]["by_class"]["A"] == 0.9

    def test_compact_json_last_line(self):
        """JSON is on the last line, preceded by other output."""
        stdout = "Starting...\nProcessing...\n" + '{"metrics": {"accuracy": 0.88}}' + "\n"
        result = _parse_json_from_stdout(stdout)
        assert result["metrics"]["accuracy"] == 0.88
