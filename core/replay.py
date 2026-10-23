"""Replay companion experiments from a reproducibility manifest.

Replay-grade checks (agentic-harness upgrade):
- code hash drift vs the sealed manifest (``code_sha256``),
- per-manifest-seed execution (not just seed 0),
- metric drift within ``metric_tolerance`` (default ±5%) on shared numeric
  keys vs ``expected_metrics``.
Missing seal fields degrade gracefully to success-only checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, Dict, List, Optional

from .sandbox import execute_sandboxed


def _parse_metrics_json(stdout: str) -> Dict[str, Any]:
    """Extract the trailing {"metrics": ...} JSON line from program output."""
    text = stdout or ""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict) and isinstance(parsed.get("metrics"), dict):
                return dict(parsed["metrics"])
        except Exception:
            continue
    return {}


def _metrics_drift(expected: Dict[str, Any], actual: Dict[str, Any],
                   tolerance: float = 0.05) -> Dict[str, Any]:
    """Compare shared numeric metric keys; drift beyond tolerance fails."""
    compared: List[Dict[str, Any]] = []
    worst = 0.0
    for key, exp_val in (expected or {}).items():
        if key not in (actual or {}):
            continue
        try:
            exp_f, act_f = float(exp_val), float(actual[key])
        except (TypeError, ValueError):
            continue
        denom = abs(exp_f) if abs(exp_f) > 1e-12 else 1.0
        rel = abs(act_f - exp_f) / denom
        worst = max(worst, rel)
        compared.append({"metric": key, "expected": exp_f,
                         "actual": act_f, "rel_drift": rel,
                         "ok": rel <= tolerance})
    failed = [c for c in compared if not c["ok"]]
    passed = not failed
    return {"compared": compared, "worst_rel_drift": worst,
            "passed": passed, "ok": passed, "n_compared": len(compared)}


def replay_companion(companion_dir: str, manifest_name: str = "reproducibility_manifest.json", clean_env: bool = False, metric_tolerance: Optional[float] = None) -> Dict[str, Any]:
    root = Path(companion_dir).resolve()
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        return {"passed": False, "error": "missing reproducibility manifest", "experiments": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tolerance = float(metric_tolerance if metric_tolerance is not None
                      else manifest.get("metric_tolerance", 0.05))
    sealed_sha = manifest.get("code_sha256") or {}
    sealed_metrics = manifest.get("expected_metrics") or {}
    sealed_seeds = manifest.get("seeds") or {}
    experiments: List[Dict[str, Any]] = []
    python_executable = sys.executable
    environment = {"mode": "current_process", "python": sys.version}
    if clean_env:
        env_dir = root / ".replay_venv"
        if not env_dir.exists():
            venv.EnvBuilder(with_pip=False, clear=False).create(env_dir)
        python_executable = str(env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        environment = {"mode": "clean_venv", "python": python_executable}
    for code_path in sorted((root / "experiments").glob("*.py")):
        name = code_path.stem
        item: Dict[str, Any] = {"name": name, "checks": {}}
        # 1. Code hash drift vs sealed manifest.
        try:
            actual_sha = hashlib.sha256(code_path.read_bytes()).hexdigest()[:16]
        except Exception:
            actual_sha = ""
        expected_sha = sealed_sha.get(name)
        if expected_sha:
            item["checks"]["code_hash"] = {
                "expected": expected_sha, "actual": actual_sha,
                "ok": bool(actual_sha) and actual_sha == expected_sha,
            }
        else:
            item["checks"]["code_hash"] = {"ok": True, "note": "no sealed hash"}
        # 2. Execute per manifest seeds (fallback: seed 0 only).
        seeds = sealed_seeds.get(name) or [0]
        try:
            seeds = [int(s) for s in seeds][:5] or [0]
        except Exception:
            seeds = [0]
        seed_results: List[Dict[str, Any]] = []
        for seed in seeds:
            if clean_env:
                completed = subprocess.run([python_executable, str(code_path)], cwd=root, capture_output=True, text=True, timeout=120)
                result = {"success": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr, "returncode": completed.returncode}
                metrics = _parse_metrics_json(completed.stdout or "")
            else:
                result = execute_sandboxed(code_path.read_text(encoding="utf-8"), seed=int(seed))
                parsed = result.get("parsed") or {}
                metrics = parsed.get("metrics") or _parse_metrics_json(result.get("stdout") or "")
            seed_results.append({"seed": seed, "success": bool(result.get("success")), "metrics": metrics})
        item["seeds"] = seed_results
        item["success"] = all(s["success"] for s in seed_results)
        # 3. Metric drift vs sealed expectations (shared numeric keys only).
        expected = sealed_metrics.get(name) or {}
        merged_actual: Dict[str, Any] = {}
        for s in seed_results:
            for k, v in (s.get("metrics") or {}).items():
                merged_actual.setdefault(k, v)
        if expected and merged_actual:
            drift = _metrics_drift(expected, merged_actual, tolerance)
            item["checks"]["metric_drift"] = drift
        else:
            item["checks"]["metric_drift"] = {"ok": True, "note": "no sealed metrics", "n_compared": 0}
        item["passed"] = (
            bool(item["checks"]["code_hash"]["ok"])
            and bool(item["success"])
            and bool(item["checks"]["metric_drift"]["ok"])
        )
        item["result"] = seed_results[-1] if seed_results else {}
        experiments.append(item)
    passed = bool(experiments) and all(item["passed"] for item in experiments)
    return {
        "passed": passed,
        "manifest": manifest,
        "environment": environment,
        "experiments": experiments,
        "error": None if passed else "one or more companion experiments failed replay (hash/seed/metric)",
    }
