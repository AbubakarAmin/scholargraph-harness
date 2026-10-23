"""Docker-based execution backend for Engineer experiments.

Provides stronger isolation than the AST/builtins sandbox by running
generated code inside a container with no network access, configurable
memory/CPU limits, and automatic cleanup.

The returned dict shapes are identical to ``core.sandbox.execute_sandboxed``
and ``core.sandbox.run_multi_seed`` so callers do not need to change.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config
from .sandbox import build_preamble, truncate_to_bytes

logger = logging.getLogger(__name__)

IMAGE_NAME = "scholargraph-sandbox:latest"

# Project root + Dockerfile location (cwd-independent: core/ -> root).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE_PATH = _PROJECT_ROOT / "sandbox" / "Dockerfile"

# Cached ensure verdict so repeated experiment calls don't re-probe
# `docker images` on every seed. Reset via ensure_sandbox_image(force=True).
_ensure_cache: dict | None = None

DEFAULT_MEMORY_LIMIT = "2g"
DEFAULT_CPU_LIMIT = "1.0"

# D16: each container run gets a unique name so timeout cleanup kills exactly
# its own container (never a sibling run matched by image ancestor filter).
CONTAINER_NAME_PREFIX = "sg_"
CONTAINER_PIDS_LIMIT = "128"


def _resource_limits(
    memory: Optional[str] = None,
    cpus: Optional[str] = None,
) -> Dict[str, str]:
    return {
        "memory": memory or DEFAULT_MEMORY_LIMIT,
        "cpus": cpus or DEFAULT_CPU_LIMIT,
    }


def _build_preamble(seed: int) -> str:
    """Seed preamble (D13): the shared AST template, never a local copy."""
    return build_preamble(seed)


def _parse_json_from_stdout(stdout: str) -> Dict[str, Any]:
    """Same logic as core.sandbox._parse_json_from_stdout."""
    text = (stdout or "").strip()
    if not text:
        return {}
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    last_close = text.rfind("}")
    if last_close >= 0:
        depth = 0
        for i in range(last_close, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[i : last_close + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return {}


def _container_running(container_id: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
            capture_output=True, text=True, timeout=5,
        )
        return "true" in (r.stdout or "").lower()
    except Exception:
        return False


def docker_available(timeout: int = 5) -> bool:
    """True when the Docker daemon answers `docker info`."""
    try:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0
    except Exception:
        return False


def image_exists(timeout: int = 10) -> bool:
    """True when the sandbox image is present locally."""
    import time as _time

    for attempt in (1, 2):
        try:
            r = subprocess.run(
                ["docker", "images", "-q", IMAGE_NAME],
                capture_output=True, text=True, timeout=timeout,
            )
            if bool((r.stdout or "").strip()):
                return True
            # Empty on first try can be a cold-daemon flake — retry once.
            if attempt == 1:
                _time.sleep(2)
        except Exception:
            if attempt == 1:
                _time.sleep(2)
    return False


def build_sandbox_image(timeout: int = 600) -> dict:
    """Build the sandbox image from sandbox/Dockerfile.

    Returns {"ok": bool, "output": str}. Takes a few minutes on first
    build (~900MB with numpy/pandas/sklearn/scipy). Safe to re-run —
    Docker caches layers.
    """
    global _ensure_cache
    if not DOCKERFILE_PATH.exists():
        return {"ok": False, "output": f"Dockerfile not found at {DOCKERFILE_PATH}"}
    try:
        r = subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, "-f", str(DOCKERFILE_PATH),
             str(_PROJECT_ROOT / "sandbox")],
            capture_output=True, text=True, timeout=timeout,
        )
        ok = r.returncode == 0 and image_exists()
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-4000:]
        if ok:
            _ensure_cache = {"ok": True, "backend": "docker",
                             "image": IMAGE_NAME, "built_now": True}
        return {"ok": ok, "output": tail}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": f"docker build timed out after {timeout}s"}
    except FileNotFoundError:
        return {"ok": False, "output": "Docker not found on PATH — start Docker Desktop first"}
    except Exception as e:
        return {"ok": False, "output": str(e)}


def ensure_sandbox_image(auto_build: bool = True, force: bool = False) -> dict:
    """Ensure Docker + sandbox image are ready. Cached per process.

    Returns {"ok", "backend" ("docker"|"ast_fallback"), "reason", ...}.
    When the image is missing and auto_build=True, builds it once
    (first build takes minutes). Callers that can't wait should pass
    auto_build=False and fall back to the AST sandbox.
    """
    global _ensure_cache
    if _ensure_cache is not None and not force:
        return _ensure_cache
    if not docker_available():
        _ensure_cache = {"ok": False, "backend": "ast_fallback",
                         "reason": "docker_unreachable",
                         "hint": "Start Docker Desktop, then retry. AST sandbox will be used meanwhile."}
        return _ensure_cache
    if image_exists():
        _ensure_cache = {"ok": True, "backend": "docker",
                         "image": IMAGE_NAME, "built_now": False}
        return _ensure_cache
    if not auto_build:
        _ensure_cache = {"ok": False, "backend": "ast_fallback",
                         "reason": "image_missing",
                         "hint": f"Build it: docker build -t {IMAGE_NAME} -f sandbox/Dockerfile sandbox/"}
        return _ensure_cache
    logger.warning("Sandbox image %s missing — building now (one-time, takes minutes)...", IMAGE_NAME)
    build = build_sandbox_image()
    if build["ok"]:
        _ensure_cache = {"ok": True, "backend": "docker",
                         "image": IMAGE_NAME, "built_now": True}
    else:
        _ensure_cache = {"ok": False, "backend": "ast_fallback",
                         "reason": "image_build_failed",
                         "hint": f"Build manually: docker build -t {IMAGE_NAME} -f sandbox/Dockerfile sandbox/",
                         "build_output": build["output"][-2000:]}
    return _ensure_cache


def sandbox_status() -> dict:
    """Machine-readable sandbox readiness for /api/health and CLI checks."""
    status: dict = {
        "image": IMAGE_NAME,
        "dockerfile": str(DOCKERFILE_PATH),
        "ephemeral": True,
        "note": "No persistent container: each experiment runs `docker run --rm "
                f"{IMAGE_NAME}` and is removed on exit. `docker ps` is empty between runs by design.",
    }
    status["docker_daemon"] = docker_available()
    status["image_present"] = image_exists() if status["docker_daemon"] else False
    ensure = ensure_sandbox_image(auto_build=False)
    status["ready"] = bool(ensure.get("ok"))
    status["backend"] = ensure.get("backend", "ast_fallback")
    if not status["ready"]:
        status["hint"] = ensure.get("hint", "")
        status["reason"] = ensure.get("reason", "")
    return status


def _force_remove(container_id: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass


def execute_containerized(
    code: str,
    timeout_sec: Optional[int] = None,
    seed: int = 42,
    memory: Optional[str] = None,
    cpus: Optional[str] = None,
    auto_build: bool = True,
) -> Dict[str, Any]:
    """Execute code inside a Docker container.

    Returns the same dict shape as ``core.sandbox.execute_sandboxed``.

    Ephemeral model: every call does ``docker run --rm`` with a unique
    container name and removes it on exit — there is intentionally NO
    long-running container, so ``docker ps`` is empty between runs.
    If the image is missing it is built once automatically; pass
    ``auto_build=False`` to fail fast instead (dispatch uses the
    fallback path).
    """
    timeout_sec = timeout_sec or config.sandbox_timeout_sec
    ensure = ensure_sandbox_image(auto_build=auto_build)
    if not ensure.get("ok"):
        reason = ensure.get("reason", "not_ready")
        hint = ensure.get("hint", "")
        if reason == "docker_unreachable":
            return {
                "success": False,
                "error": "Docker daemon unreachable — start Docker Desktop (AST fallback recommended meanwhile)",
                "stdout": "",
                "stderr": hint,
                "seed": seed,
                "backend": "ast_fallback",
            }
        return {
            "success": False,
            "error": f"Sandbox image {IMAGE_NAME} missing ({reason}). {hint}",
            "stdout": "",
            "stderr": hint,
            "seed": seed,
            "backend": "ast_fallback",
        }
    limits = _resource_limits(memory, cpus)
    preamble = _build_preamble(seed)
    full_code = preamble + "\n" + code

    tmpdir = tempfile.mkdtemp(prefix="sg_container_")
    code_path = os.path.join(tmpdir, "experiment.py")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        Path(code_path).write_text(full_code, encoding="utf-8")
        container_id = None
        # D16: unique container name for exact cleanup + pids cap.
        container_name = CONTAINER_NAME_PREFIX + uuid.uuid4().hex[:12]
        try:
            cmd = [
                "docker", "run",
                "--rm",
                "--name", container_name,
                "--network", "none",
                "--memory", limits["memory"],
                "--cpus", limits["cpus"],
                "--pids-limit", CONTAINER_PIDS_LIMIT,
                "--read-only",
                "--tmpfs", "/tmp:size=64m",
                "-v", f"{code_path}:/sandbox/experiment.py:ro",
                "-v", f"{output_dir}:/sandbox/output",
                IMAGE_NAME,
                "python", "/sandbox/experiment.py",
            ]
            t0 = time.monotonic()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_sec)
                elapsed = time.monotonic() - t0
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                # Kill the docker run process (this stops the container)
                proc.kill()
                try:
                    stdout, stderr = proc.communicate(timeout=10)
                except Exception:
                    stdout = ""
                    stderr = ""
                # D16: kill by exact container name first (never a sibling);
                # fall back to the image-ancestor sweep for strays.
                _force_remove(container_name)
                try:
                    inspect = subprocess.run(
                        ["docker", "ps", "-q", "--filter", "ancestor=" + IMAGE_NAME],
                        capture_output=True, text=True, timeout=5,
                    )
                    for cid in (inspect.stdout or "").strip().splitlines():
                        if cid:
                            _force_remove(cid)
                except Exception:
                    pass
                return {
                    "success": False,
                    "error": f"Container timeout after {timeout_sec}s",
                    "stdout": stdout,
                    "stderr": f"timeout:{timeout_sec}",
                    "seed": seed,
                    "timeout": True,
                    "container_name": container_name,
                }

            if exit_code != 0:
                return {
                    "success": False,
                    "error": f"Container exited with code {exit_code}",
                    "stdout": stdout,
                    "stderr": stderr,
                    "seed": seed,
                }

            # D10: parse the FULL stdout first, then truncate for transport
            # under the byte budget (never truncate-then-parse).
            parsed = _parse_json_from_stdout(stdout)
            stdout_truncated, _ = truncate_to_bytes(stdout, config.sandbox_max_output_bytes)
            return {
                "success": True,
                "stdout": stdout_truncated,
                "stderr": stderr,
                "parsed": parsed,
                "seed": seed,
                "container_name": container_name,
            }

        except FileNotFoundError:
            return {
                "success": False,
                "error": "Docker not found on PATH — install Docker and ensure daemon is running",
                "stdout": "",
                "stderr": "docker binary not found",
                "seed": seed,
            }
        except Exception as e:
            if container_id and _container_running(container_id):
                _force_remove(container_id)
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": "",
                "stderr": str(e),
                "seed": seed,
            }
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": str(e),
            "seed": seed,
        }


def container_multi_seed(
    code: str,
    n_seeds: Optional[int] = None,
    base_seed: int = 42,
    memory: Optional[str] = None,
    cpus: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    wall_clock_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the same experiment with multiple seeds inside containers;
    aggregate mean +/- std. Returns the same dict shape as
    ``core.sandbox.run_multi_seed``.

    D11: ``timeout_sec`` threads through to every per-seed container run;
    ``wall_clock_sec`` shares one deadline across the batch (seeds that
    cannot start in time are recorded as skips).
    """
    import numpy as np

    n_seeds = n_seeds or config.experiment_seeds
    deadline = (time.monotonic() + float(wall_clock_sec)) if wall_clock_sec else None
    runs: List[Dict[str, Any]] = []
    skipped: List[int] = []
    for i in range(n_seeds):
        seed = base_seed + i * 1009
        per_seed_timeout = timeout_sec
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                skipped.append(seed)
                continue
            if per_seed_timeout is None or per_seed_timeout > remaining:
                per_seed_timeout = max(1, int(remaining))
        result = execute_containerized(
            code, timeout_sec=per_seed_timeout, seed=seed, memory=memory, cpus=cpus
        )
        runs.append(result)

    successes = [r for r in runs if r.get("success")]
    if not successes:
        error_details = []
        for i, r in enumerate(runs):
            err = r.get("error") or "unknown"
            stderr = r.get("stderr") or ""
            detail = f"seed_{i}: {err}"
            if stderr:
                detail += f" | stderr: {stderr[:300]}"
            error_details.append(detail)
        combined_error = "All seeded runs failed. Details:\n" + "\n".join(error_details)
        return {
            "success": False,
            "error": combined_error,
            "runs": runs,
            "aggregate_metrics": {},
            "wall_clock_skipped_seeds": skipped,
        }

    metric_series: Dict[str, List[float]] = {}
    for r in successes:
        metrics = (r.get("parsed") or {}).get("metrics") or {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_series.setdefault(k, []).append(float(v))

    if not metric_series:
        return {
            "success": False,
            "error": "Code executed but produced no parseable metrics - check stdout JSON format",
            "runs": runs,
            "aggregate_metrics": {},
        }

    aggregate = {}
    for k, vals in metric_series.items():
        arr = np.array(vals, dtype=float)
        aggregate[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": vals,
            "n": len(vals),
        }

    return {
        "success": True,
        "n_seeds": n_seeds,
        "n_success": len(successes),
        "runs": runs,
        "aggregate_metrics": aggregate,
        "raw_results": [r.get("parsed") for r in successes],
        "wall_clock_skipped_seeds": skipped,
    }


if __name__ == "__main__":
    import sys as _sys

    args = set(_sys.argv[1:])
    if "--build" in args:
        print(f"Building {IMAGE_NAME} from {DOCKERFILE_PATH} ...")
        res = build_sandbox_image()
        print("BUILD OK" if res["ok"] else "BUILD FAILED")
        print(res["output"][-2000:])
        raise SystemExit(0 if res["ok"] else 1)
    # Default: status check (no build). Use --build to build, --ensure to build-if-missing.
    if "--ensure" in args:
        import json as _json
        print(_json.dumps(ensure_sandbox_image(auto_build=True, force=True), indent=2))
    else:
        import json as _json
        print(_json.dumps(sandbox_status(), indent=2))
