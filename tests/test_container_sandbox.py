"""Tests for the Docker-based container sandbox backend.

Requires Docker daemon running.  Tests skip gracefully when Docker is
unavailable or the sandbox image has not been built.
"""

from __future__ import annotations

import subprocess
import time

import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from core.sandbox import execute_sandboxed

IMAGE = "scholargraph-sandbox:latest"


def _docker_available() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _image_exists() -> bool:
    r = subprocess.run(
        ["docker", "images", "-q", IMAGE],
        capture_output=True, text=True, timeout=10,
    )
    return bool((r.stdout or "").strip())


def _running_containers() -> list[str]:
    r = subprocess.run(
        ["docker", "ps", "-q"],
        capture_output=True, text=True, timeout=10,
    )
    return [c for c in (r.stdout or "").strip().splitlines() if c]


needs_docker = pytest.mark.skipif(
    not _docker_available() or not _image_exists(),
    reason="Docker daemon not running or sandbox image not built",
)

SIMPLE_SUCCESS_CODE = """
import json, numpy as np
np.random.seed(0)
x = np.random.randn(100)
metrics = {"mean": float(x.mean()), "std": float(x.std())}
print(json.dumps({"metrics": metrics}))
"""

FAILING_CODE = """
raise ValueError("intentional test failure")
"""

NO_METRICS_CODE = """
print("no json here")
"""

TIMEOUT_CODE = """
import time
time.sleep(600)
"""


@needs_docker
def test_containerized_success():
    from core.container_sandbox import execute_containerized

    result = execute_containerized(SIMPLE_SUCCESS_CODE, seed=0)
    assert result["success"], result.get("error")
    assert "parsed" in result
    assert "metrics" in result["parsed"]
    assert "mean" in result["parsed"]["metrics"]
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)
    assert result["seed"] == 0


@needs_docker
def test_containerized_failure_shape():
    from core.container_sandbox import execute_containerized

    ast_result = execute_sandboxed(FAILING_CODE, seed=0)
    container_result = execute_containerized(FAILING_CODE, seed=0)

    assert not container_result["success"]
    assert "error" in container_result
    assert isinstance(container_result["stdout"], str)
    assert isinstance(container_result["stderr"], str)
    assert container_result["seed"] == 0

    # Shape must match AST backend's failure shape
    for key in ("success", "error", "stdout", "stderr", "seed"):
        assert key in container_result, f"missing key: {key}"


@needs_docker
def test_containerized_timeout():
    from core.container_sandbox import execute_containerized

    result = execute_containerized(TIMEOUT_CODE, timeout_sec=5, seed=0)
    assert not result["success"]
    assert result.get("timeout") is True
    assert "timeout" in (result.get("error") or "").lower() or "timeout" in (result.get("stderr") or "").lower()
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)


@needs_docker
def test_no_container_leak_on_success():
    from core.container_sandbox import execute_containerized

    before = _running_containers()
    execute_containerized(SIMPLE_SUCCESS_CODE, seed=0)
    after = _running_containers()
    # No new containers left running
    new = set(after) - set(before)
    assert not new, f"Leaked containers after success: {new}"


@needs_docker
def test_no_container_leak_on_failure():
    from core.container_sandbox import execute_containerized

    before = _running_containers()
    execute_containerized(FAILING_CODE, seed=0)
    after = _running_containers()
    new = set(after) - set(before)
    assert not new, f"Leaked containers after failure: {new}"


@needs_docker
def test_no_container_leak_on_timeout():
    from core.container_sandbox import execute_containerized

    before = _running_containers()
    execute_containerized(TIMEOUT_CODE, timeout_sec=3, seed=0)
    time.sleep(2)
    after = _running_containers()
    new = set(after) - set(before)
    assert not new, f"Leaked containers after timeout: {new}"


@needs_docker
def test_container_multi_seed():
    from core.container_sandbox import container_multi_seed

    code = """
import numpy as np, json, random
x = [random.random() for _ in range(50)]
print(json.dumps({"metrics": {"acc": float(sum(x)/len(x))}}))
"""
    out = container_multi_seed(code, n_seeds=2)
    assert out["success"], out.get("error")
    assert out["n_success"] == 2
    assert "acc" in out["aggregate_metrics"]
    assert "mean" in out["aggregate_metrics"]["acc"]


@needs_docker
def test_dispatch_routes_to_docker(monkeypatch):
    monkeypatch.setattr("core.sandbox_dispatch.config", type("C", (), {
        "sandbox_backend": "docker",
        "sandbox_docker_memory": "1g",
        "sandbox_docker_cpus": "1.0",
        "sandbox_timeout_sec": 30,
        "experiment_seeds": 2,
        "sandbox_max_output_bytes": 1048576,
    })())

    from core import sandbox_dispatch
    result = sandbox_dispatch.execute(SIMPLE_SUCCESS_CODE, seed=0)
    assert result["success"], result.get("error")
    assert "metrics" in result.get("parsed", {})


@needs_docker
def test_dispatch_routes_to_ast(monkeypatch):
    monkeypatch.setattr("core.sandbox_dispatch.config", type("C", (), {
        "sandbox_backend": "ast",
        "sandbox_docker_memory": "1g",
        "sandbox_docker_cpus": "1.0",
        "sandbox_timeout_sec": 30,
        "experiment_seeds": 2,
        "sandbox_max_output_bytes": 1048576,
    })())

    from core import sandbox_dispatch
    result = sandbox_dispatch.execute(SIMPLE_SUCCESS_CODE, seed=0)
    assert result["success"], result.get("error")
    assert "metrics" in result.get("parsed", {})
