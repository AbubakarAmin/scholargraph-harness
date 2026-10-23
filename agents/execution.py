"""Independent reproducible experiment execution worker."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.capabilities import DEFAULT_MANIFESTS
from core.context import RunContext, get_active_context
from core.contracts import CodeArtifact, ExecutionArtifact, ExecutionRequest
from core.sandbox import validate_code
from core.sandbox_dispatch import execute_multi_seed

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """Execute approved code and preserve raw, replayable outputs."""

    capability_manifest = DEFAULT_MANIFESTS["ExecutionAgent"]

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        runtime_config = self.context.config if self.context else None
        output_dir = runtime_config.raw_results_dir if runtime_config else "./output/raw_results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        code_artifact: CodeArtifact,
        request: ExecutionRequest,
    ) -> ExecutionArtifact:
        code = code_artifact.get("source_code") or code_artifact.get("code") or ""
        experiment_name = request.get("experiment_name") or code_artifact.get("experiment_name", "experiment")
        seeds = request.get("seeds") or self._default_seeds()
        arena_generated = code_artifact.get("execution_backend") == "arena"
        now = datetime.now(timezone.utc).isoformat()
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "seeds": seeds,
            "sandbox": "Docker (arena-required)" if arena_generated else "core.sandbox_dispatch.execute_multi_seed",
        }
        ok, error = validate_code(code)
        if not ok:
            return self._artifact(
                experiment_name,
                request,
                environment,
                "failed",
                error=f"Code rejected before execution: {error}",
                created_at=now,
            )

        # D11: the caller's resource_limits timeout actually reaches the
        # sandbox now (previously the multi-seed path always used the
        # default and the configured timeout was silently dropped).
        limits = request.get("resource_limits") or {}
        timeout_sec = limits.get("timeout_seconds")
        wall_clock_sec = limits.get("wall_clock_seconds")
        if arena_generated:
            expected_seeds = [seeds[0] + index * 1009 for index in range(len(seeds))]
            if seeds != expected_seeds:
                result = {
                    "success": False,
                    "error": f"Arena validation seeds must follow the trusted seed schedule {expected_seeds}",
                }
            else:
                try:
                    from core.container_sandbox import (
                        container_multi_seed,
                        ensure_sandbox_image,
                    )

                    ready = ensure_sandbox_image(auto_build=True)
                    if not ready.get("ok"):
                        result = {
                            "success": False,
                            "error": "Docker execution is required for arena code; "
                                     f"sandbox unavailable: {ready.get('reason') or ready.get('hint')}",
                        }
                    else:
                        from core.config import config as sandbox_config

                        result = container_multi_seed(
                            code,
                            n_seeds=len(seeds),
                            base_seed=seeds[0],
                            memory=sandbox_config.sandbox_docker_memory,
                            cpus=sandbox_config.sandbox_docker_cpus,
                            timeout_sec=timeout_sec,
                            wall_clock_sec=wall_clock_sec,
                        )
                        runs = result.get("runs") or []
                        actual_seeds = [
                            run.get("seed") for run in runs if isinstance(run, dict)
                        ]
                        if (
                            result.get("success") is not True
                            or result.get("n_seeds") != len(seeds)
                            or result.get("n_success") != len(seeds)
                            or actual_seeds != seeds
                            or result.get("wall_clock_skipped_seeds")
                            or any(run.get("backend") == "ast_fallback" for run in runs)
                        ):
                            result["success"] = False
                            result["error"] = (
                                result.get("error")
                                or "Docker validation did not complete every requested seed"
                            )
                        result_errors = self._arena_result_errors(result, seeds)
                        if result_errors:
                            result["success"] = False
                            prior_error = result.get("error")
                            result["error"] = "; ".join(
                                item for item in [prior_error, *result_errors] if item
                            )
                except Exception as exc:
                    logger.exception("Docker execution failed for arena-generated code")
                    result = {
                        "success": False,
                        "error": f"Docker execution required for arena code failed: {exc}",
                    }
        else:
            result = execute_multi_seed(code, n_seeds=len(seeds), base_seed=seeds[0],
                                        timeout_sec=timeout_sec, wall_clock_sec=wall_clock_sec)
        if result.get("success") and len(seeds) != result.get("n_seeds"):
            result["requested_seeds"] = seeds
        status = "completed" if result.get("success") else "failed"
        return self._artifact(
            experiment_name,
            request,
            environment,
            status,
            result=result,
            error=result.get("error"),
            created_at=now,
        )

    @staticmethod
    def _arena_result_errors(result: dict[str, Any], seeds: list[int]) -> list[str]:
        errors: list[str] = []
        runs = result.get("runs")
        if not isinstance(runs, list) or len(runs) != len(seeds):
            return [f"Docker returned {len(runs) if isinstance(runs, list) else 0} seed runs; expected {len(seeds)}"]
        actual_seeds = [run.get("seed") for run in runs if isinstance(run, dict)]
        if actual_seeds != seeds:
            errors.append(f"Docker seed list {actual_seeds} does not match requested {seeds}")
        expected_keys: Optional[set[str]] = None
        for index, run in enumerate(runs):
            if not isinstance(run, dict) or run.get("success") is not True:
                errors.append(f"seed {seeds[index]} did not complete successfully")
                continue
            if run.get("backend") == "ast_fallback":
                errors.append(f"seed {seeds[index]} was not executed in Docker")
            parsed = run.get("parsed")
            metrics = parsed.get("metrics") if isinstance(parsed, dict) else None
            if not isinstance(metrics, dict) or not metrics:
                errors.append(f"seed {seeds[index]} returned no metrics")
                continue
            keys: set[str] = set()
            for name, value in metrics.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append(f"seed {seeds[index]} metric '{name}' is not numeric")
                    continue
                try:
                    finite = math.isfinite(float(value))
                except (OverflowError, ValueError):
                    finite = False
                if not finite:
                    errors.append(f"seed {seeds[index]} metric '{name}' is not finite")
                    continue
                keys.add(str(name))
            if expected_keys is None:
                expected_keys = keys
            elif expected_keys != keys:
                errors.append(f"seed {seeds[index]} returned inconsistent metric keys")
        if result.get("wall_clock_skipped_seeds"):
            errors.append(f"Docker skipped seeds: {result['wall_clock_skipped_seeds']}")
        return errors

    def _default_seeds(self) -> list[int]:
        runtime_config = self.context.config if self.context else None
        count = runtime_config.experiment_seeds if runtime_config else 3
        return [42 + index * 1009 for index in range(count)]

    def _artifact(
        self,
        experiment_name: str,
        request: ExecutionRequest,
        environment: dict[str, Any],
        status: str,
        *,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        created_at: str,
    ) -> ExecutionArtifact:
        payload = result or {"error": error}
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        raw_path = self.output_dir / f"{experiment_name}_{digest[:12]}.json"
        raw_path.write_bytes(encoded)
        return {
            "request": request,
            "raw_results_path": str(raw_path),
            "environment": environment,
            "seed_results": payload,
            "status": status,
            "error": error or "",
            "content_hash": digest,
            "provenance": {
                "artifact_type": "execution",
                "producer": "ExecutionAgent",
                "content_hash": digest,
                "created_at": created_at,
                "status": status,
                "environment": environment,
                "limitations": (
                    ["Arena code validation requires Docker; no AST fallback is allowed."]
                    if environment.get("sandbox") == "Docker (arena-required)"
                    else ["Execution uses the configured sandbox backend."]
                ),
            },
        }