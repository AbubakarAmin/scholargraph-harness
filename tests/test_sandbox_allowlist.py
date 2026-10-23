"""Regression: sandbox validate_code must allow safe stdlib and statsmodels."""

from __future__ import annotations

import pytest

from core.sandbox import validate_code, ALLOWED_IMPORT_ROOTS, FORBIDDEN_MODULES


class TestSandboxAllowlist:
    """Safe modules must pass AST validation without burning retry attempts."""

    @pytest.mark.parametrize(
        "module",
        [
            "sys",
            "io",
            "json",
            "warnings",
            "random",
            "math",
            "itertools",
            "functools",
            "collections",
            "dataclasses",
            "typing",
            "re",
            "statistics",
            "copy",
            "time",
            "datetime",
        ],
    )
    def test_stdlib_imports_allowed(self, module):
        ok, err = validate_code(f"import {module}")
        assert ok, f"import {module} blocked: {err}"

    @pytest.mark.parametrize(
        "module",
        [
            "numpy",
            "scipy",
            "pandas",
            "sklearn",
            "statsmodels",
        ],
    )
    def test_science_imports_allowed(self, module):
        ok, err = validate_code(f"import {module}")
        assert ok, f"import {module} blocked: {err}"

    @pytest.mark.parametrize(
        "module",
        [
            "statsmodels.api",
            "statsmodels.regression.linear_model",
            "statsmodels.stats.outliers_influence",
        ],
    )
    def test_statsmodels_submodules_allowed(self, module):
        ok, err = validate_code(f"import {module}")
        assert ok, f"import {module} blocked: {err}"

    @pytest.mark.parametrize(
        "stmt",
        [
            "from sys import version_info",
            "from io import StringIO",
            "from statsmodels.api import OLS",
            "from statsmodels.regression.linear_model import OLS",
            "from statsmodels.stats.outliers_influence import variance_inflation_factor",
        ],
    )
    def test_from_imports_allowed(self, stmt):
        ok, err = validate_code(stmt)
        assert ok, f"'{stmt}' blocked: {err}"

    def test_forbidden_modules_still_blocked(self):
        for mod in ("subprocess", "os", "shutil", "socket", "ctypes", "multiprocessing"):
            ok, err = validate_code(f"import {mod}")
            assert not ok, f"import {mod} should be blocked but was allowed"

    def test_allowed_roots_include_statsmodels(self):
        assert "statsmodels" in ALLOWED_IMPORT_ROOTS

    def test_allowed_roots_include_sys(self):
        assert "sys" in ALLOWED_IMPORT_ROOTS

    def test_allowed_roots_include_io(self):
        assert "io" in ALLOWED_IMPORT_ROOTS

    def test_sys_not_in_forbidden_modules(self):
        assert "sys" not in FORBIDDEN_MODULES


class TestStatsmodelsInManifest:
    """statsmodels must appear in the default capability manifest available_libraries."""

    def test_statsmodels_in_manifest(self):
        from core.capabilities import SANDBOX_CAPABILITY_MANIFEST
        assert "statsmodels" in SANDBOX_CAPABILITY_MANIFEST.available_libraries

    def test_manifest_libraries_cover_allowlist_core(self):
        from core.capabilities import SANDBOX_CAPABILITY_MANIFEST
        manifest_libs = set(SANDBOX_CAPABILITY_MANIFEST.available_libraries)
        # Core data-science libs must be in both
        for lib in ("numpy", "scipy", "pandas", "sklearn", "statsmodels"):
            assert lib in manifest_libs, f"{lib} missing from manifest"
            assert lib in ALLOWED_IMPORT_ROOTS, f"{lib} missing from sandbox allowlist"
