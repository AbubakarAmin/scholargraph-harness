"""Single-point dispatch for sandbox backend selection.

Routes ``execute`` and ``run_multi_seed`` calls to either the AST/builtins
sandbox (fast, in-process) or the Docker container sandbox (hard isolation)
based on ``config.sandbox_backend``.

All callers that need to respect the backend switch should call these
functions instead of importing from ``core.sandbox`` or
``core.container_sandbox`` directly.

Robustness (v3): when the Docker backend is configured but the daemon is
unreachable (e.g. Docker Desktop not running), dispatch falls back to the
AST sandbox with a one-time warning instead of failing every run.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from typing import Any, Dict, List, Optional

from .config import config
from .sandbox import (
    execute_sandboxed,
    run_multi_seed,
    validate_code,
    run_known_answer_check,
)

logger = logging.getLogger(__name__)

_docker_probe_result: bool | None = None
_docker_probe_lock = threading.Lock()
_docker_fallback_warned = False


def _docker_daemon_reachable() -> bool:
    """Probe the Docker daemon once per process; cache the verdict."""
    global _docker_probe_result, _docker_fallback_warned
    if _docker_probe_result is not None:
        return _docker_probe_result
    with _docker_probe_lock:
        if _docker_probe_result is not None:
            return _docker_probe_result
        docker_exe = shutil.which("docker")
        if not docker_exe:
            _docker_probe_result = False
        else:
            try:
                probe = subprocess.run(
                    [docker_exe, "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    timeout=5,
                )
                _docker_probe_result = probe.returncode == 0
            except Exception:
                _docker_probe_result = False
        if not _docker_probe_result and not _docker_fallback_warned:
            _docker_fallback_warned = True
            logger.warning(
                "SANDBOX_BACKEND=docker but the Docker daemon is unreachable — "
                "falling back to the AST sandbox for this process."
            )
    return _docker_probe_result


def _use_docker() -> bool:
    if (config.sandbox_backend or "ast").lower() != "docker":
        return False
    return _docker_daemon_reachable()


def _fallback_to_ast(reason: str) -> None:
    global _docker_fallback_warned
    if not _docker_fallback_warned:
        _docker_fallback_warned = True
        logger.warning(
            "Docker backend unavailable (%s) — falling back to the AST "
            "sandbox for this process. No manual `docker run` needed: "
            "experiments use ephemeral `docker run --rm` containers automatically "
            "once Docker + the scholargraph-sandbox image are ready. "
            "Check with: python -m core.container_sandbox",
            reason,
        )


def execute(
    code: str,
    timeout_sec: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute one seed run using the configured backend.

    No manual container management: the Docker backend spawns an
    ephemeral ``docker run --rm`` per call (unique name, auto-removed).
    If Docker/the image isn't ready, falls back to AST with a warning.
    """
    use_docker = _use_docker()
    backend = "docker" if use_docker else "ast"
    logger.info("Sandbox execute started (backend=%s, seed=%s, timeout=%s)", backend, seed, timeout_sec)

    try:
        if use_docker:
            from .container_sandbox import execute_containerized

            result = execute_containerized(
                code,
                timeout_sec=timeout_sec,
                seed=seed,
                memory=config.sandbox_docker_memory,
                cpus=config.sandbox_docker_cpus,
            )
            if isinstance(result, dict) and result.get("backend") == "ast_fallback":
                _fallback_to_ast(str(result.get("error") or "not_ready")[:160])
                result = execute_sandboxed(code, timeout_sec=timeout_sec, seed=seed)
                backend = "ast"
        else:
            result = execute_sandboxed(code, timeout_sec=timeout_sec, seed=seed)
    except TimeoutError:
        logger.info("Sandbox execution timed out (backend=%s, seed=%s, timeout=%s)", backend, seed, timeout_sec)
        raise
    except Exception as exc:
        logger.info("Sandbox execution failed (backend=%s, seed=%s, error=%s)", backend, seed, exc)
        raise

    # D9: both backends report `success`/`error` — the old `status` lookup
    # always logged "unknown".
    if isinstance(result, dict):
        outcome = "success" if result.get("success") is True else "error"
        detail = "" if result.get("success") is True else str(result.get("error") or "")[:200]
    else:
        outcome, detail = "error", "non-dict result"
    logger.info("Sandbox execution finished (backend=%s, seed=%s, outcome=%s %s)", backend, seed, outcome, detail)
    return result


def execute_multi_seed(
    code: str,
    n_seeds: Optional[int] = None,
    base_seed: int = 42,
    timeout_sec: Optional[int] = None,
    wall_clock_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Run N seeds and aggregate metrics using the configured backend.

    D11: ``timeout_sec`` (per-seed) and ``wall_clock_sec`` (whole batch)
    thread through to whichever backend runs — the independent-validation
    ``resource_limits`` timeout actually reaches the sandbox now.
    """
    if _use_docker():
        from .container_sandbox import container_multi_seed, ensure_sandbox_image

        ensure = ensure_sandbox_image(auto_build=True)
        if not ensure.get("ok"):
            _fallback_to_ast(str(ensure.get("reason") or "not_ready"))
        else:
            return container_multi_seed(
                code,
                n_seeds=n_seeds,
                base_seed=base_seed,
                memory=config.sandbox_docker_memory,
                cpus=config.sandbox_docker_cpus,
                timeout_sec=timeout_sec,
                wall_clock_sec=wall_clock_sec,
            )
    return run_multi_seed(
        code, n_seeds=n_seeds, base_seed=base_seed,
        timeout_sec=timeout_sec, wall_clock_sec=wall_clock_sec,
    )
