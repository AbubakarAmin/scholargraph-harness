"""Tests for core.sandbox.extract_local_code_structure.

Verifies that the function correctly:
- Walks traceback frames and filters to generated-code frames only
- Extracts the full source of implicated custom functions
- Excludes library/stdlib frames
- Handles edge cases (top-level error, no matching functions, None traceback)
"""

from __future__ import annotations

import sys
import traceback

import pytest

from core.sandbox import execute_sandboxed, extract_local_code_structure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_and_get_tb(code: str, filename: str = "<test>"):
    """Execute *code* in-process and return (tb_obj, source_code).

    The code should raise an exception.  Returns the traceback object and
    the source code string so tests can call ``extract_local_code_structure``.
    """
    try:
        exec(compile(code, filename, "exec"), {})
    except Exception as exc:
        return exc.__traceback__, code
    raise RuntimeError("code did not raise — test is misconfigured")


# ---------------------------------------------------------------------------
# 1. Simple two-function chain
# ---------------------------------------------------------------------------

class TestSimpleFunctionChain:
    """Traceback spans two custom functions: outer -> inner -> error."""

    SOURCE = (
        "def outer(data):\n"
        "    return inner(data)\n"
        "\n"
        "def inner(data):\n"
        "    raise ValueError('bad data')\n"
        "\n"
        "outer([])\n"
    )

    def test_includes_both_functions(self):
        tb, src = _run_and_get_tb(self.SOURCE)
        result = extract_local_code_structure(tb, src, "<test>", preamble_offset=0)

        assert "=== Traceback ===" in result
        assert "=== Implicated Custom Functions ===" in result
        assert "--- outer() ---" in result
        assert "--- inner() ---" in result

    def test_full_function_source_present(self):
        tb, src = _run_and_get_tb(self.SOURCE)
        result = extract_local_code_structure(tb, src, "<test>", preamble_offset=0)

        # outer() body must include the call to inner()
        assert "return inner(data)" in result
        # inner() body must include the raise
        assert "raise ValueError('bad data')" in result

    def test_traceback_text_also_present(self):
        tb, src = _run_and_get_tb(self.SOURCE)
        result = extract_local_code_structure(tb, src, "<test>", preamble_offset=0)

        # The formatted traceback should mention the exception type
        assert "ValueError" in result


# ---------------------------------------------------------------------------
# 2. Deep three-function chain
# ---------------------------------------------------------------------------

class TestDeepFunctionChain:
    """Traceback spans three custom functions: a -> b -> c -> error."""

    SOURCE = (
        "def a():\n"
        "    return b()\n"
        "\n"
        "def b():\n"
        "    return c()\n"
        "\n"
        "def c():\n"
        "    raise RuntimeError('deep error')\n"
        "\n"
        "a()\n"
    )

    def test_all_three_functions_extracted(self):
        tb, src = _run_and_get_tb(self.SOURCE)
        result = extract_local_code_structure(tb, src, "<test>", preamble_offset=0)

        assert "--- a() ---" in result
        assert "--- b() ---" in result
        assert "--- c() ---" in result

    def test_full_source_of_each_function(self):
        tb, src = _run_and_get_tb(self.SOURCE)
        result = extract_local_code_structure(tb, src, "<test>", preamble_offset=0)

        assert "return b()" in result
        assert "return c()" in result
        assert "raise RuntimeError('deep error')" in result


# ---------------------------------------------------------------------------
# 3. Mix of custom and library frames
# ---------------------------------------------------------------------------

class TestMixedCustomAndLibrary:
    """Traceback includes both custom functions and stdlib frames (e.g. json)."""

    SOURCE = (
        "import json\n"
        "\n"
        "def parse_data(raw):\n"
        "    return json.loads(raw)\n"
        "\n"
        "def process():\n"
        "    raw = '{bad json'\n"
        "    return parse_data(raw)\n"
        "\n"
        "process()\n"
    )

    def test_custom_functions_included(self):
        tb, src = _run_and_get_tb(self.SOURCE)
        result = extract_local_code_structure(tb, src, "<test>", preamble_offset=0)

        assert "--- parse_data() ---" in result
        assert "--- process() ---" in result

    def test_library_frame_excluded_from_custom_section(self):
        tb, src = _run_and_get_tb(self.SOURCE)
        result = extract_local_code_structure(tb, src, "<test>", preamble_offset=0)

        # The "Implicated Custom Functions" section should NOT contain
        # a function named "json" or "loads" — only user-defined functions.
        custom_section = result.split("=== Implicated Custom Functions ===")[-1]
        assert "--- json() ---" not in custom_section
        assert "--- loads() ---" not in custom_section


# ---------------------------------------------------------------------------
# 4. Sibling functions (error in first, second also on stack)
# ---------------------------------------------------------------------------

class TestSiblingFunctions:
    """Two sibling functions called from a common caller."""

    SOURCE = (
        "def caller():\n"
        "    a()\n"
        "    b()\n"
        "\n"
        "def a():\n"
        "    pass\n"
        "\n"
        "def b():\n"
        "    raise TypeError('oops')\n"
        "\n"
        "caller()\n"
    )

    def test_only_implicated_functions_included(self):
        tb, src = _run_and_get_tb(self.SOURCE)
        result = extract_local_code_structure(tb, src, "<test>", preamble_offset=0)

        # b() and caller() are on the stack; a() is NOT
        assert "--- caller() ---" in result
        assert "--- b() ---" in result
        assert "--- a() ---" not in result


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_none_traceback_returns_empty(self):
        result = extract_local_code_structure(None, "x = 1", "<test>", preamble_offset=0)
        assert result == ""

    def test_syntax_error_in_source_returns_raw_tb(self):
        """If source_code can't be parsed, fall back to raw traceback text."""
        bad_source = "def foo(:\n  pass\n"
        try:
            compile(bad_source, "<test>", "exec")
        except SyntaxError as exc:
            tb = exc.__traceback__
            result = extract_local_code_structure(tb, bad_source, "<test>", preamble_offset=0)
        # Should not raise — just returns some traceback text
        assert isinstance(result, str)
        assert len(result) > 0

    def test_top_level_error_shows_context_window(self):
        """Error at module top-level (not inside a function) shows context lines."""
        source = "x = 1\ny = 2\nz = x + 'incompatible'\n"
        tb, src = _run_and_get_tb(source)
        result = extract_local_code_structure(tb, src, "<test>", preamble_offset=0)

        # Should include source context since no function encloses the error
        assert "Source context" in result or "Implicated Custom Functions" in result

    def test_preamble_offset_adjusts_line_numbers(self):
        """With a preamble offset, the function maps traceback lines correctly.

        The preamble adds N lines before the user's code, so traceback line
        numbers are shifted by N.  The function must subtract the offset to
        map back to source_code line numbers.
        """
        preamble = "x = 1\ny = 2\nz = 3\na = 4\nb = 5\n"
        source = (
            "def helper():\n"
            "    raise RuntimeError('offset test')\n"
            "\n"
            "helper()\n"
        )
        full_code = preamble + source
        preamble_lines = preamble.count("\n")  # 5

        try:
            exec(compile(full_code, "<sandbox>", "exec"), {})
        except RuntimeError as exc:
            tb = exc.__traceback__
            result = extract_local_code_structure(tb, source, "<sandbox>", preamble_offset=preamble_lines)

        # With correct offset, helper() should be found
        assert "--- helper() ---" in result


# ---------------------------------------------------------------------------
# 6. Integration with execute_sandboxed
# ---------------------------------------------------------------------------

class TestIntegrationWithSandbox:
    """Verify that execute_sandboxed returns local_code_context on failure."""

    def test_failed_execution_includes_local_code_context(self):
        code = (
            "def compute():\n"
            "    return 1 / 0\n"
            "\n"
            "compute()\n"
        )
        result = execute_sandboxed(code, seed=0)
        assert not result["success"]
        assert result.get("local_code_context")
        assert "=== Traceback ===" in result["local_code_context"]
        assert "--- compute() ---" in result["local_code_context"]

    def test_successful_execution_no_local_code_context(self):
        code = (
            "import json\n"
            "print(json.dumps({'metrics': {'acc': 0.5}}))\n"
        )
        result = execute_sandboxed(code, seed=0)
        assert result["success"]
        # Successful runs don't have local_code_context (no error to extract)
        assert not result.get("local_code_context")

    def test_syntax_error_no_local_code_context(self):
        """Syntax errors are caught before execution — no traceback object."""
        code = "def foo(:\n  pass\n"
        result = execute_sandboxed(code, seed=0)
        assert not result["success"]
        # validate_code catches SyntaxError before execution, so no traceback
        assert not result.get("local_code_context")
