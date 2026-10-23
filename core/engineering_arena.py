"""
Engineering Arena — delegate experiment-code authoring to a headless opencode agent.

Why this exists (run 211505fb): the legacy Engineer generates one code blob
per attempt via the shared LLM gateway. When the provider 503s, the circuit
breaker opens, ``chat()`` returns ``""`` for allow_empty callers, and all 4
attempts burn in ~0.3s with zero useful work. An opencode agent instead owns
a code-authoring loop with its own provider session, independent of the
ScholarGraph gateway. Generated code is executed only by our Docker runner.

Directory layout (per-run isolation = same-paper memory, cross-paper amnesia)::

    engineering_arena/
      <run_id>/
        manifest.json            # run_id, topic, created, arena version
        _shared/
          datasets.md            # how to reach project datasets
          run_memory.md          # rolling summaries of this run's experiments
        <exp_slug>/<invocation_id>/
          task.md                # the experiment contract + output protocol
          context.md             # same-run memory: summaries of prior experiments
          feedback.md            # validator feedback for retry rounds (append-only)
          questions.md           # needs_input questions surfaced to us (fallback channel)
          plan_update_request.json  # optional: agent-proposed plan change
          opencode.json          # headless permission policy (scoped by cwd)
          experiment.py          # written by the agent; executed by Docker
          metrics.json           # written by ScholarGraph from Docker results
          STATUS.json            # written by the agent: ready|needs_input|failed
          stdout.log             # captured `opencode run` output per round
          session.json           # our record: rounds, return codes, durations

Interaction protocol (the back-and-forth):
  * ScholarGraph writes task.md/context.md, spawns
    ``opencode run --dir <taskdir> --format json -f task.md ...``.
  * The agent MUST finish by writing STATUS.json:
       done        -> experiment.py is ready for trusted Docker execution
      needs_input -> STATUS.json carries question/context/options. This is the
                     fallback-to-us channel: we surface it as a plan-revision
                     request (and, when an answer file is provided, resume).
      failed      -> STATUS.json carries error + notes.
  * The agent only writes experiment.py and STATUS.json; it cannot execute
    shell commands. ScholarGraph validates the code, runs every seed in
    Docker, checks the contract and outputs, and writes the trusted metrics.
    Failures become feedback.md and the agent is re-spawned, bounded by
    ARENA_MAX_ROUNDS.

Cross-paper isolation: each run gets a fresh ``<run_id>/`` dir, opencode's
cwd is the task dir (never the project root), and stale run dirs are swept
by retention. Same-paper memory: per-experiment summaries accumulate in
``_shared/run_memory.md`` and are injected into the next task's context.md.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ARENA_VERSION = 1

STATUS_DONE = "done"
STATUS_NEEDS_INPUT = "needs_input"
STATUS_FAILED = "failed"

METRICS_FILENAME = "metrics.json"
STATUS_FILENAME = "STATUS.json"
CODE_FILENAME = "experiment.py"
TASK_FILENAME = "task.md"
CONTEXT_FILENAME = "context.md"
FEEDBACK_FILENAME = "feedback.md"
QUESTIONS_FILENAME = "questions.md"
PLAN_UPDATE_FILENAME = "plan_update_request.json"
SESSION_FILENAME = "session.json"
STDOUT_FILENAME = "stdout.log"
SHARED_DIRNAME = "_shared"
RUN_MEMORY_FILENAME = "run_memory.md"
DATASETS_FILENAME = "datasets.md"
MANIFEST_FILENAME = "manifest.json"


class ArenaError(Exception):
    """Base class for arena failures."""


class ArenaUnavailable(ArenaError):
    """opencode binary missing or cannot be spawned — caller may fall back."""


class ArenaTimeout(ArenaError):
    """A session round exceeded its wall-clock budget."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Paths + naming
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", str(name or "experiment")).strip("_")
    return (slug or "experiment")[:80]


def arena_root() -> str:
    """Arena root, resolved from config at call time (test-friendly)."""
    try:
        from core.config import config as _cfg

        root = str(getattr(_cfg, "arena_dir", "./engineering_arena") or "./engineering_arena")
    except Exception:
        root = "./engineering_arena"
    return os.path.abspath(root)


def run_dir(run_id: str) -> str:
    return os.path.join(arena_root(), slugify(run_id or "adhoc"))


def task_dir(run_id: str, experiment_name: str) -> str:
    return os.path.join(run_dir(run_id), slugify(experiment_name))


def resolve_run_id(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(explicit)
    try:
        from core.run_log import get_tracker as _get_tracker

        tracker = _get_tracker()
        if tracker and getattr(tracker, "run_id", None):
            return str(tracker.run_id)
    except Exception:
        pass
    return "adhoc"


# ---------------------------------------------------------------------------
# Run lifecycle: manifest, shared memory, stale sweep (cross-paper isolation)
# ---------------------------------------------------------------------------

def ensure_run(run_id: str, topic: Optional[str] = None) -> str:
    """Create the per-run arena dir + shared files. Sweeps stale runs first."""
    rdir = run_dir(run_id)
    shared = os.path.join(rdir, SHARED_DIRNAME)
    os.makedirs(shared, exist_ok=True)
    manifest_path = os.path.join(rdir, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "topic": topic or "",
                "arena_version": ARENA_VERSION,
                "created": _utcnow(),
            }, f, indent=2)
    mem_path = os.path.join(shared, RUN_MEMORY_FILENAME)
    if not os.path.exists(mem_path):
        with open(mem_path, "w", encoding="utf-8") as f:
            f.write(f"# Arena run memory — run {run_id}\n\n")
            f.write("Summaries of completed experiments in THIS run only. "
                    "No other paper or run is visible here.\n\n")
    ds_path = os.path.join(shared, DATASETS_FILENAME)
    if not os.path.exists(ds_path):
        with open(ds_path, "w", encoding="utf-8") as f:
            f.write(render_datasets_doc())
    try:
        from core.config import config as _cfg

        if str(getattr(_cfg, "arena_cleanup_stale", "true")).lower() == "true":
            retention = int(getattr(_cfg, "arena_retention_days", 7) or 7)
            sweep_stale_runs(retention_days=retention, keep_run_id=run_id)
    except Exception as exc:
        logger.debug("arena stale sweep skipped: %s", exc)
    return rdir


def sweep_stale_runs(retention_days: int = 7, keep_run_id: Optional[str] = None) -> List[str]:
    """Delete arena run dirs older than retention_days (cross-paper amnesia).

    Returns the list of removed run_ids. The live run (keep_run_id) is never
    touched. Fail-open: errors are logged, never raised.
    """
    removed: List[str] = []
    root = arena_root()
    if not os.path.isdir(root):
        return removed
    cutoff = time.time() - max(0, int(retention_days)) * 86400.0
    keep_slug = slugify(keep_run_id) if keep_run_id else None
    try:
        entries = os.listdir(root)
    except OSError as exc:
        logger.debug("arena sweep: cannot list %s: %s", root, exc)
        return removed
    for entry in entries:
        if entry == keep_slug or entry.startswith("."):
            continue
        path = os.path.join(root, entry)
        if not os.path.isdir(path) or entry == SHARED_DIRNAME:
            continue
        try:
            if os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(entry)
                logger.info("arena sweep: removed stale run dir %s", entry)
        except OSError as exc:
            logger.debug("arena sweep: cannot remove %s: %s", entry, exc)
    return removed


def read_run_memory(run_id: str) -> str:
    try:
        with open(os.path.join(run_dir(run_id), SHARED_DIRNAME, RUN_MEMORY_FILENAME),
                  encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def record_experiment_summary(run_id: str, experiment_name: str, summary: str) -> None:
    """Append one experiment's outcome to same-run memory (same-paper recall)."""
    try:
        rdir = ensure_run(run_id)
        with open(os.path.join(rdir, SHARED_DIRNAME, RUN_MEMORY_FILENAME),
                  "a", encoding="utf-8") as f:
            f.write(f"\n## {experiment_name}\n{summary.strip()}\n")
    except OSError as exc:
        logger.debug("arena record summary failed: %s", exc)


# ---------------------------------------------------------------------------
# Task rendering — the specific instructions the agent receives
# ---------------------------------------------------------------------------

def render_datasets_doc(project_root: Optional[str] = None) -> str:
    root = os.path.abspath(project_root or os.getcwd())
    return f"""# Dataset access for arena experiments

Project root on the agent host: `{root}`

The trusted Docker runner currently mounts only `experiment.py` and a temporary
output directory. It has no network access and cannot read this host's catalog
or download datasets. Use synthetic data only when it satisfies the experiment
contract. If a named real dataset is required, do not substitute synthetic or
unrelated data; report `needs_input` so the workflow can arrange an explicit,
read-only dataset input.
"""


def _fmt_list(items: Any) -> str:
    if not items:
        return "none specified"
    if isinstance(items, str):
        return items
    try:
        return "; ".join(str(x) for x in list(items))
    except Exception:
        return str(items)


def render_task_md(
    experiment: Dict[str, Any],
    seeds: int = 3,
    prior_context: str = "",
    dataset_plan: str = "",
    project_root: Optional[str] = None,
) -> str:
    """Render the exact instruction file the opencode agent must follow."""
    exp_name = experiment.get("name", "<unnamed>")
    analysis_plan = experiment.get("analysis_plan") or {}
    try:
        analysis_json = json.dumps(analysis_plan, indent=2, default=str)[:2000]
    except Exception:
        analysis_json = str(analysis_plan)[:2000]
    prior_block = (f"\n## Prior experiments in THIS run (same paper memory)\n{prior_context.strip()}\n"
                   if prior_context and prior_context.strip() else "")
    dataset_block = f"\n## Dataset for this experiment\n{dataset_plan.strip()}\n" if dataset_plan and dataset_plan.strip() else ""
    return f"""# Engineering task: {exp_name}

You are the experiment engineer for a research pipeline. Your job: implement
this ONE experiment and write the requested code and status files. ScholarGraph
will run your code in an isolated Docker container for each requested seed,
inspect the results, and send you any errors to fix. Do not run or execute
experiment.py yourself.

## Working-directory rule (hard)
- Do ALL work inside THIS directory only. Never read, write, or execute
  anything outside it.
- Other experiments, other papers, and the main project tree are NOT your
  concern. Only `context.md` (prior summaries for this same paper) is yours.

## Experiment contract (do not renegotiate silently)
- Purpose: {experiment.get("purpose", "N/A")}
- Methodology: {experiment.get("methodology", "N/A")}
- Baselines (implement EVERY one on the SAME data/splits): {_fmt_list(experiment.get("baselines") or experiment.get("baseline_comparison"))}
- Claimed components: {_fmt_list(experiment.get("claimed_components") or experiment.get("components"))}
- Required metric keys: {_fmt_list(experiment.get("evaluation_metrics"))}
- Falsifiable prediction: {experiment.get("falsifiable_prediction", "N/A")}
- Statistical test you MUST actually compute: {experiment.get("statistical_test", "N/A")}
- Analysis plan:
```json
{analysis_json}
```
- Seeds to run and aggregate: {seeds} (ScholarGraph supplies each seed)
- contract_hash (validated and recorded by ScholarGraph): `{experiment.get("contract_hash", "none")}`
{prior_block}{dataset_block}
## Data & environment
- See `_shared/datasets.md` (one level up) for data constraints. Synthetic
  data matching `data_requirements` is acceptable only when the contract allows it.
- Data requirements: {experiment.get("data_requirements", "synthetic data matching the contract")}
- Python 3 with numpy/pandas/matplotlib/sklearn/scipy available. Do not install
  packages; the runner has no network access.
## Code constraints (hard — your script is re-validated by an AST gate)
- Allowed imports ONLY: numpy, pandas, matplotlib, sklearn, scipy, math,
  statistics, random, json, re, collections, seaborn, networkx, sympy.
  Anything else (os, sys, subprocess,
  pathlib, socket, requests, open() for writing, eval, exec, exit) FAILS
  validation and burns a round — do not use them, even if they run fine here.
- Every sklearn/scipy symbol you use MUST be explicitly imported in YOUR
  script (e.g. `from sklearn.datasets import make_classification`).
  Bare names without an import WILL NameError downstream.
- No `pass  # TODO` / `NotImplemented` stubs; no hardcoded-literal metrics —
  every reported number must be computed by the script.
- The trusted runner injects integer `ARENA_SEED` and seeds Python's and
  NumPy's global RNG before executing your script. Use those seeded RNGs;
  do not set a constant seed or use an unseeded `default_rng()`.
- Run once per invocation and print exactly one final JSON line:
  `{{"metrics": {{"<metric>": 0.0}}, "raw": {{...}}}}`. Each metric must be
  a finite number computed for this run only; do not aggregate across seeds.
- If the contract needs a bootstrap CI, compute it (B from analysis_plan,
  percentile/BCa, resampling the right unit) and report ci_low/ci_high.

## Required output files (exact names, in THIS directory)
1. `experiment.py` — the complete runnable script.
2. `STATUS.json` — EXACTLY one of:
   - `{{"status": "done", "notes": "implementation is ready for trusted execution"}}`
   - `{{"status": "needs_input", "question": "<what you need>", "context": "<why>",
      "options": ["A", "B"], "blocked_on": "<contract item>"}}`
     Use this when the contract is ambiguous, data is unreachable, or the plan
     itself must change. ALSO write `plan_update_request.json`
     (`{{"reason": ..., "experiment": ..., "detail": ...}}`) when you propose
     a plan change — the orchestrator reads it and may resume you with an answer.
   - `{{"status": "failed", "error": "<root cause>", "notes": "<what you tried>"}}`
     Only after genuinely exhausting fixes. Never fail on the first error.

## Think-cycle
Write code → inspect it for consistency with the contract → write STATUS.json.
The trusted runner executes each seed; any traceback or contract mismatch is
returned to you in feedback.md. Never use shell, install packages, or execute
the experiment yourself.
"""


def render_context_md(prior_summaries: str, feedback: str = "") -> str:
    parts = ["# Context for this experiment\n"]
    if prior_summaries and prior_summaries.strip():
        parts.append("## What sibling experiments in this run already learned\n")
        parts.append(prior_summaries.strip() + "\n")
    else:
        parts.append("(No prior experiments in this run yet — you are first.)\n")
    if feedback and feedback.strip():
        parts.append("\n## Validator feedback on your previous round (MUST address)\n")
        parts.append(feedback.strip() + "\n")
    return "\n".join(parts)


def write_task_files(
    tdir: str,
    experiment: Dict[str, Any],
    seeds: int,
    prior_summaries: str,
    dataset_plan: str = "",
    feedback: str = "",
) -> None:
    """Write task.md / context.md / opencode.json into the task dir."""
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, TASK_FILENAME), "w", encoding="utf-8") as f:
        f.write(render_task_md(experiment, seeds=seeds,
                               prior_context=prior_summaries,
                               dataset_plan=dataset_plan))
    with open(os.path.join(tdir, CONTEXT_FILENAME), "w", encoding="utf-8") as f:
        f.write(render_context_md(prior_summaries, feedback=feedback))
    # The arena agent authors code only. In particular, it must not execute
    # generated code on the host; execution belongs to the Docker runner.
    policy = {"permission": {
        "read": "allow", "edit": "allow", "write": "allow",
        "glob": "allow", "grep": "allow", "bash": "deny",
    }}
    with open(os.path.join(tdir, "opencode.json"), "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)


# ---------------------------------------------------------------------------
# Spawning opencode
# ---------------------------------------------------------------------------

def find_opencode_binary() -> Optional[str]:
    """Locate a directly-executable opencode binary.

    npm on Windows installs a `.cmd` shim that CreateProcess cannot execute
    (and `cmd /c` mangles its quoting). The shim delegates to a real exe
    under node_modules — prefer that so no shell is ever involved.
    """
    found = shutil.which("opencode")
    if not found:
        return None
    if str(found).lower().endswith((".cmd", ".bat", ".ps1")):
        base = os.path.dirname(os.path.abspath(found))
        for candidate in (
            os.path.join(base, "node_modules", "opencode-ai", "bin", "opencode.exe"),
            os.path.join(base, "opencode.exe"),
        ):
            if os.path.isfile(candidate):
                return candidate
    return found


def _build_command(
    binary: str,
    tdir: str,
    brief: str,
    model: str = "",
    agent: str = "",
) -> List[str]:
    # NOTE: the brief (positional message) MUST come before any --file
    # flags, and files must use the `--file=<path>` form. yargs greedily
    # consumes bare positionals following an array flag, which turns the
    # brief into "File not found: <brief>" (opencode 1.18).
    cmd = [binary, "run", brief, "--dir", tdir, "--format", "json",
           "--title", f"arena:{os.path.basename(tdir)}"]
    if model:
        cmd += ["--model", model]
    if agent:
        cmd += ["--agent", agent]
    cmd += [f"--file={os.path.join(tdir, TASK_FILENAME)}",
            f"--file={os.path.join(tdir, CONTEXT_FILENAME)}"]
    return cmd


_SESSION_ID_RE = re.compile(r'"sessionID"\s*:\s*"([^"]+)"')


def spawn_session(
    tdir: str,
    brief: str,
    timeout_sec: int = 1800,
    model: str = "",
    agent: str = "",
) -> Dict[str, Any]:
    """Spawn one headless `opencode run` round. Returns a spawn record.

    Raises ArenaUnavailable when the binary is missing or cannot be started.
    A timeout kills the child and is reported (not raised) so the caller can
    still harvest partial artifacts (STATUS.json written before the hang).
    """
    binary = find_opencode_binary()
    if not binary:
        raise ArenaUnavailable(
            "opencode binary not found on PATH — install opencode or use "
            "ENGINEERING_BACKEND=legacy")
    cmd = _build_command(binary, tdir, brief, model=model, agent=agent)
    # Last-resort Windows path: only a .cmd/.bat shim was found (no real
    # exe). Route via `cmd /s /c` with the whole command double-quoted —
    # plain `cmd /c` strips the quotes around a spaced binary path.
    use_cmd_prefix = os.name == "nt" and str(binary).lower().endswith((".cmd", ".bat"))
    started = time.monotonic()
    try:
        if use_cmd_prefix:
            quoted = '"' + subprocess.list2cmdline(cmd) + '"'
            proc = subprocess.run(
                ["cmd", "/s", "/c", quoted],
                capture_output=True, text=True, timeout=timeout_sec,
                cwd=os.path.dirname(os.path.abspath(tdir)),
            )
        else:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_sec,
                cwd=os.path.dirname(os.path.abspath(tdir)),
            )
        timed_out = False
        returncode: Optional[int] = proc.returncode
        stdout, stderr = proc.stdout or "", proc.stderr or ""
    except FileNotFoundError as exc:
        raise ArenaUnavailable(f"cannot start opencode ({exc})") from exc
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
    except OSError as exc:
        raise ArenaUnavailable(f"cannot start opencode ({exc})") from exc
    elapsed = time.monotonic() - started
    session_id: Optional[str] = None
    try:
        match = _SESSION_ID_RE.search(stdout or "")
        if match:
            session_id = match.group(1)
    except Exception:
        pass
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_sec": round(elapsed, 1),
        "session_id": session_id,
        "stdout_chars": len(stdout or ""),
        "stderr_chars": len(stderr or ""),
        "stdout": stdout or "",
        "stderr_tail": (stderr or "")[-2000:],
    }


# ---------------------------------------------------------------------------
# Artifact protocol: STATUS.json + metrics.json
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_status(tdir: str) -> Dict[str, Any]:
    return _read_json(os.path.join(tdir, STATUS_FILENAME))


def read_metrics(tdir: str) -> Dict[str, Any]:
    return _read_json(os.path.join(tdir, METRICS_FILENAME))


def read_code(tdir: str) -> str:
    try:
        with open(os.path.join(tdir, CODE_FILENAME), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def normalize_metrics(raw: Dict[str, Any], seeds: int) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Normalize metrics.json -> aggregate {key: {mean,std,n}} + error list."""
    errors: List[str] = []
    if not isinstance(raw, dict) or not raw:
        return {}, ["metrics.json is missing or not a JSON object"]
    metrics = raw.get("metrics", raw)
    if not isinstance(metrics, dict) or not metrics:
        return {}, ["metrics.json carries no 'metrics' mapping"]
    agg: Dict[str, Dict[str, float]] = {}
    for key, val in metrics.items():
        try:
            if isinstance(val, dict):
                mean = float(val.get("mean", val.get("value")))
                std = float(val.get("std", 0.0) or 0.0)
                n = int(val.get("n", seeds) or seeds)
            else:
                mean, std, n = float(val), 0.0, int(seeds)
            agg[str(key)] = {"mean": mean, "std": std, "n": n}
        except (TypeError, ValueError):
            errors.append(f"metric '{key}' is not numeric: {str(val)[:80]}")
    if not agg:
        errors.append("no parseable numeric metrics")
    return agg, errors


def _expected_seed_list(seeds: int, base_seed: int = 42) -> List[int]:
    return [base_seed + index * 1009 for index in range(seeds)]


def _metric_key_matches(required: str, actual: str) -> bool:
    required_key = required.strip().lower()
    actual_key = actual.strip().lower()
    return bool(required_key and actual_key and (
        required_key == actual_key or required_key in actual_key or actual_key in required_key
    ))


def _validate_runner_results(
    result: Dict[str, Any],
    experiment: Dict[str, Any],
    seeds: int,
) -> Tuple[List[str], Dict[str, Dict[str, Any]], List[int]]:
    errors: List[str] = []
    expected_seeds = _expected_seed_list(seeds)
    runs = result.get("runs")
    if not isinstance(runs, list) or len(runs) != seeds:
        return [f"Docker runner returned {len(runs) if isinstance(runs, list) else 0} runs; expected {seeds}"], {}, expected_seeds

    actual_seeds = [run.get("seed") for run in runs if isinstance(run, dict)]
    if actual_seeds != expected_seeds:
        errors.append(f"Docker runner seeds {actual_seeds} do not match expected seeds {expected_seeds}")
    if result.get("n_seeds") != seeds or result.get("n_success") != seeds:
        errors.append(
            f"Docker runner reported n_seeds={result.get('n_seeds')} "
            f"and n_success={result.get('n_success')}; expected {seeds} each"
        )
    if result.get("wall_clock_skipped_seeds"):
        errors.append(f"Docker runner skipped seeds: {result['wall_clock_skipped_seeds']}")

    metric_rows: List[Dict[str, float]] = []
    expected_keys: Optional[set[str]] = None
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or run.get("success") is not True:
            errors.append(f"seed {expected_seeds[index]} failed: {run.get('error', 'invalid runner record') if isinstance(run, dict) else 'invalid runner record'}")
            continue
        if run.get("backend") == "ast_fallback":
            errors.append(
                f"seed {expected_seeds[index]} was not executed: Docker runner reported AST fallback"
            )
        parsed = run.get("parsed")
        metrics = parsed.get("metrics") if isinstance(parsed, dict) else None
        if not isinstance(metrics, dict) or not metrics:
            errors.append(f"seed {expected_seeds[index]} returned no metrics mapping")
            continue
        row: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"seed {expected_seeds[index]} metric '{key}' is not a finite number")
                continue
            try:
                numeric_value = float(value)
            except (OverflowError, ValueError):
                errors.append(f"seed {expected_seeds[index]} metric '{key}' is not a finite number")
                continue
            if not math.isfinite(numeric_value):
                errors.append(f"seed {expected_seeds[index]} metric '{key}' is not a finite number")
                continue
            row[str(key)] = numeric_value
        keys = set(row)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            errors.append(
                f"seed {expected_seeds[index]} metric keys {sorted(keys)} "
                f"do not match {sorted(expected_keys)}"
            )
        metric_rows.append(row)

    if metric_rows and len(metric_rows) == seeds and not errors:
        required = experiment.get("evaluation_metrics") or []
        if isinstance(required, str):
            required = [required]
        metric_keys = list(metric_rows[0])
        for required_key in (str(item) for item in required):
            if required_key.strip() and not any(
                _metric_key_matches(required_key, key) for key in metric_keys
            ):
                errors.append(f"required metric '{required_key}' missing from Docker results")
        aggregate: Dict[str, Dict[str, Any]] = {}
        for key in metric_keys:
            values = [row[key] for row in metric_rows]
            try:
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
            except (OverflowError, ValueError) as exc:
                errors.append(f"metric '{key}' aggregation failed: {exc}")
                continue
            if not math.isfinite(mean) or not math.isfinite(std):
                errors.append(f"metric '{key}' aggregate is not finite")
                continue
            aggregate[key] = {"mean": mean, "std": std, "values": values, "n": len(values)}
    else:
        aggregate = {}
    if result.get("success") is not True:
        errors.append(f"Docker runner failed: {result.get('error') or 'unknown runner failure'}")
    return errors, aggregate, expected_seeds


def validate_arena_artifacts(
    tdir: str,
    experiment: Dict[str, Any],
    seeds: int,
    runner_result: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], str, Dict[str, Dict[str, Any]]]:
    """Validate agent code and, when supplied, independently collected run results.

    The agent does not provide trusted metrics. The application derives them
    from every successful Docker seed run and writes the canonical metrics file.
    """
    errors: List[str] = []
    code = read_code(tdir)
    if not (code or "").strip():
        errors.append("experiment.py is missing or empty")
    else:
        try:
            from core.sandbox import validate_code as _validate

            ok, err = _validate(code)
            if not ok:
                errors.append(f"sandbox validation: {err}"[:400])
        except Exception as exc:
            errors.append(f"validator unavailable: {exc}"[:200])
        try:
            from agents.engineer import EngineerAgent as _Eng

            lint = _Eng._lint_generated_code(code, experiment)
            if lint:
                errors.append(f"code lint: {lint}"[:400])
        except Exception as exc:
            errors.append(f"code lint unavailable: {exc}"[:200])
    agg: Dict[str, Dict[str, Any]] = {}
    if runner_result is not None:
        runner_errors, agg, expected_seeds = _validate_runner_results(
            runner_result, experiment, seeds
        )
        errors.extend(runner_errors)
        contract_hash = str(experiment.get("contract_hash") or "").strip()
        if not contract_hash:
            errors.append("experiment contract_hash is missing")
        if not errors:
            trusted_metrics = {
                "metrics": agg,
                "contract_hash": contract_hash,
                "seeds": expected_seeds,
                "notes": "Generated from trusted Docker runs by ScholarGraph.",
            }
            try:
                with open(os.path.join(tdir, METRICS_FILENAME), "w", encoding="utf-8") as f:
                    json.dump(trusted_metrics, f, indent=2, allow_nan=False)
                persisted = read_metrics(tdir)
                if persisted.get("contract_hash") != contract_hash:
                    errors.append("trusted metrics contract_hash does not match experiment contract")
                if persisted.get("seeds") != expected_seeds:
                    errors.append("trusted metrics seed list does not match Docker runs")
                persisted_metrics = persisted.get("metrics")
                if not isinstance(persisted_metrics, dict) or persisted_metrics != agg:
                    errors.append("trusted metrics do not match the Docker-run aggregate")
            except (OSError, ValueError) as exc:
                errors.append(f"could not persist trusted Docker metrics: {exc}")
    return (not errors), errors, code, agg


# ---------------------------------------------------------------------------
# Orchestration: rounds, feedback, needs_input fallback, memory record
# ---------------------------------------------------------------------------

def _append_feedback(tdir: str, text: str) -> None:
    try:
        with open(os.path.join(tdir, FEEDBACK_FILENAME), "a", encoding="utf-8") as f:
            f.write(f"\n## Round feedback ({_utcnow()})\n{text.strip()}\n")
    except OSError as exc:
        logger.debug("arena feedback append failed: %s", exc)


def _append_stdout_log(tdir: str, round_no: int, record: Dict[str, Any]) -> None:
    try:
        with open(os.path.join(tdir, STDOUT_FILENAME), "a", encoding="utf-8") as f:
            f.write(f"\n===== round {round_no} ({_utcnow()}) "
                    f"rc={record.get('returncode')} timed_out={record.get('timed_out')} "
                    f"elapsed={record.get('elapsed_sec')}s session={record.get('session_id')} =====\n")
            f.write(record.get("stdout") or "")
            tail = record.get("stderr_tail") or ""
            if tail:
                f.write(f"\n----- stderr tail -----\n{tail}\n")
    except OSError as exc:
        logger.debug("arena stdout log append failed: %s", exc)


def _record_session(tdir: str, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = os.path.join(tdir, SESSION_FILENAME)
    history: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            history = data
    except (OSError, ValueError):
        history = []
    history.append(entry)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
    except OSError as exc:
        logger.debug("arena session record failed: %s", exc)
    return history


def _arena_cfg() -> Dict[str, Any]:
    try:
        from core.config import config as _cfg

        return {
            "timeout": int(getattr(_cfg, "arena_session_timeout_sec", 1800) or 1800),
            "rounds": int(getattr(_cfg, "arena_max_rounds", 3) or 3),
            "model": str(getattr(_cfg, "arena_model", "") or ""),
            "agent": str(getattr(_cfg, "arena_agent", "build") or ""),
        }
    except Exception:
        return {"timeout": 1800, "rounds": 3, "model": "", "agent": "build"}


def run_arena_experiment(
    engineer: Any,
    experiment: Dict[str, Any],
    method_description: str = "",
    run_id: Optional[str] = None,
    alternatives: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute one experiment via a headless opencode agent in the arena.

    Returns an ExperimentOutput-shaped dict (success or failure), reusing the
    caller's validators and storage so downstream gates see a familiar shape.
    Raises ArenaUnavailable when opencode cannot be started at all (the
    caller decides legacy fallback vs hard fail).
    """
    from core.utils import log_agent_action

    exp_name = experiment.get("name", "<unnamed>")
    rid = resolve_run_id(run_id or experiment.get("run_id"))
    cfg = _arena_cfg()
    try:
        seeds = int(getattr(engineer.runtime_config, "experiment_seeds", 3) or 3)
    except Exception:
        seeds = 3
    if seeds < 1:
        return engineer._fail(
            experiment, "arena_invalid_seed_count: experiment_seeds must be positive",
            [], failure_kind="arena_invalid_seed_count",
            original_name=experiment.get("name"),
        )
    if not str(experiment.get("contract_hash") or "").strip():
        return engineer._fail(
            experiment, "arena_contract_missing: experiment contract_hash is required",
            [], failure_kind="arena_contract_missing",
            original_name=experiment.get("name"),
        )
    try:
        from core.container_sandbox import ensure_sandbox_image

        docker_ready = ensure_sandbox_image(auto_build=True)
    except Exception as exc:
        logger.exception("arena: Docker sandbox readiness check failed")
        docker_ready = {"ok": False, "reason": str(exc)}
    if not docker_ready.get("ok"):
        reason = (
            f"arena_sandbox_unavailable: Docker execution is required; "
            f"{docker_ready.get('reason') or docker_ready.get('hint') or 'sandbox image unavailable'}"
        )
        logger.error("%s", reason)
        return engineer._fail(
            experiment, reason, [], failure_kind="arena_sandbox_unavailable",
            original_name=experiment.get("name"),
        )
    ensure_run(rid, topic=str(experiment.get("topic") or ""))
    tdir = os.path.join(task_dir(rid, exp_name), uuid.uuid4().hex[:8])
    prior = read_run_memory(rid)
    dataset_plan = str(experiment.get("dataset_plan")
                       or (experiment.get("dataset") or {}).get("plan", "")
                       if isinstance(experiment.get("dataset"), dict)
                       else (experiment.get("dataset") or ""))
    if not dataset_plan:
        dataset_plan = str(experiment.get("data_requirements") or "")
    write_task_files(tdir, experiment, seeds, prior, dataset_plan=dataset_plan)
    logger.info("arena: task ready for '%s' in %s (rounds<=%d, timeout=%ds)",
                exp_name, tdir, cfg["rounds"], cfg["timeout"])
    log_agent_action("EngineerAgent", "arena_task_ready",
                     {"experiment": exp_name, "task_dir": tdir, "run_id": rid})
    try:
        engineer._progress("arena_start", {"experiment": exp_name, "task_dir": tdir})
    except Exception:
        pass

    decision_log: List[Dict[str, Any]] = []
    last_error = ""
    last_question: Optional[Dict[str, Any]] = None
    spawn_ok_once = False
    terminal_failure_kind = ""

    for round_no in range(1, cfg["rounds"] + 1):
        if round_no > 1:
            for artifact_name in (STATUS_FILENAME, METRICS_FILENAME):
                artifact_path = os.path.join(tdir, artifact_name)
                try:
                    os.remove(artifact_path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    last_error = f"arena_artifact_reset_failed: cannot clear {artifact_name}: {exc}"
                    terminal_failure_kind = "arena_artifact_reset_failed"
                    logger.error("%s", last_error)
                    break
            if terminal_failure_kind:
                break
        feedback = ""
        try:
            with open(os.path.join(tdir, FEEDBACK_FILENAME), encoding="utf-8") as f:
                feedback = f.read()
        except OSError:
            feedback = ""
        if round_no > 1:
            # Re-render context so the retry carries validator feedback.
            with open(os.path.join(tdir, CONTEXT_FILENAME), "w", encoding="utf-8") as f:
                f.write(render_context_md(prior, feedback=feedback))
        brief = (f"Complete the engineering task in {TASK_FILENAME} (context in "
                 f"{CONTEXT_FILENAME}). Work until {STATUS_FILENAME} is written. "
                 f"This is round {round_no}.")
        try:
            record = spawn_session(tdir, brief, timeout_sec=cfg["timeout"],
                                   model=cfg["model"], agent=cfg["agent"])
        except ArenaUnavailable:
            if not spawn_ok_once and round_no == 1:
                raise
            last_error = "arena_unavailable: opencode spawn failed mid-run"
            decision_log.append({"attempt": round_no, "decision": "REFINE", "reason": last_error})
            break
        spawn_ok_once = True
        _append_stdout_log(tdir, round_no, record)
        try:
            os.remove(os.path.join(tdir, METRICS_FILENAME))
        except FileNotFoundError:
            pass
        except OSError as exc:
            last_error = f"arena_untrusted_metrics_cleanup_failed: {exc}"
            terminal_failure_kind = "arena_artifact_reset_failed"
            logger.error("%s", last_error)
            break
        _record_session(tdir, {
            "round": round_no, "returncode": record.get("returncode"),
            "timed_out": record.get("timed_out"), "elapsed_sec": record.get("elapsed_sec"),
            "session_id": record.get("session_id"), "at": _utcnow(),
        })
        logger.info("arena: '%s' round %d finished rc=%s timed_out=%s elapsed=%ss",
                    exp_name, round_no, record.get("returncode"),
                    record.get("timed_out"), record.get("elapsed_sec"))
        try:
            engineer._progress("arena_round", {
                "experiment": exp_name, "round": round_no,
                "returncode": record.get("returncode"),
                "timed_out": record.get("timed_out")})
        except Exception:
            pass

        if record.get("timed_out"):
            last_error = (f"arena_timeout: session exceeded {cfg['timeout']}s "
                          f"(round {round_no})")
            decision_log.append({"attempt": round_no, "decision": "REFINE", "reason": last_error})
            _append_feedback(tdir, last_error + " — continue from existing files, "
                             "shrink per-run work, finish STATUS.json.")
            continue

        status = read_status(tdir)
        state = str(status.get("status") or "").strip().lower()
        if not state:
            last_error = (f"arena_no_artifacts: round {round_no} ended "
                          f"(rc={record.get('returncode')}) with no STATUS.json")
            decision_log.append({"attempt": round_no, "decision": "REFINE", "reason": last_error})
            _append_feedback(tdir, last_error + " — you MUST write STATUS.json "
                             f"(done|needs_input|failed) per {TASK_FILENAME}.")
            continue
        if state == STATUS_NEEDS_INPUT:
            question = {
                "question": status.get("question", ""),
                "context": status.get("context", ""),
                "options": status.get("options", []),
                "blocked_on": status.get("blocked_on", ""),
            }
            last_question = question
            try:
                with open(os.path.join(tdir, QUESTIONS_FILENAME), "a", encoding="utf-8") as f:
                    f.write(f"\n## Round {round_no} ({_utcnow()})\n"
                            f"Q: {question['question']}\nContext: {question['context']}\n"
                            f"Options: {question['options']}\nBlocked on: {question['blocked_on']}\n")
            except OSError:
                pass
            # Fallback-to-us channel: surface as a first-class revision
            # request so Planner/orchestrator (or the operator on resume) can
            # answer, then record the stop.
            detail = (f"Q: {question['question']} | Context: {question['context']} | "
                      f"Options: {question['options']} | Blocked on: {question['blocked_on']}")
            try:
                engineer.request_plan_revision("arena_needs_input", experiment, detail=detail[:800])
            except Exception:
                pass
            last_error = f"arena_needs_input: {question['question']}"[:500]
            decision_log.append({"attempt": round_no, "decision": "REFINE", "reason": last_error})
            logger.info("arena: '%s' needs input: %s", exp_name, question["question"][:200])
            try:
                engineer._progress("arena_needs_input", {"experiment": exp_name, **question})
            except Exception:
                pass
            break
        if state == STATUS_FAILED:
            last_error = f"arena_agent_failed: {status.get('error') or status.get('notes') or 'unknown'}"[:500]
            decision_log.append({"attempt": round_no, "decision": "REFINE", "reason": last_error})
            _append_feedback(tdir, f"Your STATUS was failed: {last_error}. "
                             "Re-examine the root cause and try a different approach.")
            continue
        if state != STATUS_DONE:
            last_error = f"arena_bad_status: unknown STATUS '{state}' (round {round_no})"
            decision_log.append({"attempt": round_no, "decision": "REFINE", "reason": last_error})
            _append_feedback(tdir, last_error + f" — status must be one of done|needs_input|failed.")
            continue

        # "done" means code is ready; only our Docker runner can establish
        # whether the experiment succeeded and what its metrics are.
        code_ok, code_errors, code, _ = validate_arena_artifacts(
            tdir, experiment, seeds
        )
        if code_ok:
            try:
                consistency = engineer.check_code_claim_consistency(experiment, code)
            except Exception as exc:
                consistency = {"consistent": False, "score": 0.0,
                               "notes": [f"consistency check unavailable: {exc}"[:200]]}
            if consistency.get("consistent") is not True:
                detail = "; ".join(consistency.get("notes") or ["code-claim mismatch"])
                _append_feedback(tdir, f"Code-claim gate rejected your experiment.py: {detail}. "
                                 "Implement the claimed baselines/components exactly, then re-run.")
                last_error = f"code_claim_inconsistency: {detail}"[:500]
                decision_log.append({"attempt": round_no, "decision": "REFINE", "reason": last_error})
                continue
            try:
                from core.container_sandbox import container_multi_seed
                from core.config import config as sandbox_config

                runner_result = container_multi_seed(
                    code,
                    n_seeds=seeds,
                    base_seed=42,
                    memory=getattr(sandbox_config, "sandbox_docker_memory", None),
                    cpus=getattr(sandbox_config, "sandbox_docker_cpus", None),
                    timeout_sec=getattr(sandbox_config, "sandbox_timeout_sec", 600),
                    wall_clock_sec=getattr(sandbox_config, "engineering_wall_clock_sec", None),
                )
            except Exception as exc:
                logger.exception("arena: trusted Docker runner raised an exception")
                runner_result = {"success": False, "error": f"Docker runner error: {exc}"}
            fallback_runs = [
                run for run in runner_result.get("runs", [])
                if isinstance(run, dict) and run.get("backend") == "ast_fallback"
            ]
            if fallback_runs:
                last_error = (
                    "arena_sandbox_unavailable: Docker became unavailable during "
                    "seed execution; AST fallback was rejected"
                )
                terminal_failure_kind = "arena_sandbox_unavailable"
                decision_log.append({
                    "attempt": round_no, "decision": "STOP", "reason": last_error,
                })
                break
            ok, gate_errors, code, agg = validate_arena_artifacts(
                tdir, experiment, seeds, runner_result=runner_result
            )
            if not ok:
                last_error = "arena_invalid_results: " + "; ".join(gate_errors)[:600]
                decision_log.append({"attempt": round_no, "decision": "REFINE", "reason": last_error})
                _append_feedback(
                    tdir,
                    "Trusted Docker execution rejected this attempt:\n- "
                    + "\n- ".join(gate_errors)[:2000]
                    + "\nFix the code and mark STATUS.json done; ScholarGraph will rerun every seed.",
                )
                continue
            logger.info("arena: '%s' validated OK — metrics=%s", exp_name, sorted(agg))
            try:
                record_experiment_summary(
                    rid, exp_name,
                    f"status=SUCCESS metrics={json.dumps(agg, default=str)[:500]} "
                    f"rounds={len(decision_log)} task_dir={tdir} "
                    f"notes={(status.get('notes') or '')[:200]}")
            except Exception:
                pass
            return engineer._build_arena_success_output(
                experiment, code, agg, decision_log, tdir,
                consistency=consistency, original_name=experiment.get("name"),
                multi_seed=runner_result)
        last_error = "arena_invalid_artifacts: " + "; ".join(code_errors)[:600]
        decision_log.append({"attempt": round_no, "decision": "REFINE", "reason": last_error})
        _append_feedback(tdir, "Validator rejected your artifacts:\n- "
                         + "\n- ".join(code_errors)[:2000]
                         + "\nFix experiment.py and write STATUS.json with status=done.")

    # Rounds exhausted (or needs_input stop): record same-run memory, fail
    # through the standard path so Planner/workflow can respond.
    failure_kind = (
        "arena_needs_input" if last_question
        else terminal_failure_kind or "arena_failed"
    )
    if not last_error:
        last_error = "arena_failed: rounds exhausted with no valid artifacts"
    try:
        record_experiment_summary(
            rid, exp_name,
            f"status=FAILED kind={failure_kind} error={last_error[:300]} "
            f"rounds={len(decision_log)} task_dir={tdir}")
    except Exception:
        pass
    try:
        engineer.request_plan_revision(failure_kind, experiment, detail=last_error[:500])
    except Exception:
        pass
    out = engineer._fail(experiment, last_error, decision_log, code=read_code(tdir),
                         failure_kind=failure_kind,
                         original_name=experiment.get("name"))
    out["arena_task_dir"] = tdir
    if last_question:
        out["arena_question"] = last_question
    return out
