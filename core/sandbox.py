"""
Restricted Python sandbox for Engineer experiments.
Blocks exit(), subprocess, os.system, and other dangerous calls.
Does NOT rely on the model behaving — AST + builtins lockdown.

Trust boundary (D15): the AST validator + restricted builtins are a
policy/lint filter, NOT a security boundary — they catch the documented
failure modes (exit, subprocess, host installs) but cannot contain
untrusted code the way OS isolation can. The Docker backend
(``core.container_sandbox``) is the trust boundary for untrusted
execution; treat AST results as lint-checked, not contained.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import sys
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Set, Tuple

from .config import config

# D13: single shared seed-preamble template. The container backend imports
# this (never a copy) so AST and Docker runs seed identically.
SANDBOX_PREAMBLE_TEMPLATE = (
    "ARENA_SEED = {seed}\n"
    "import random as _sg_random\n"
    "import numpy as _sg_np\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    # e0ec9e91: LLM emits bare make_classification / DummyClassifier assuming
    # implicit imports. Pre-import the common symbols so both bare names and
    # module-qualified names work; explicit imports in generated code still win.
    "try:\n"
    "    from sklearn.datasets import make_classification as make_classification\n"
    "    from sklearn.datasets import make_regression as make_regression\n"
    "    from sklearn.dummy import DummyClassifier as DummyClassifier\n"
    "    from sklearn.dummy import DummyRegressor as DummyRegressor\n"
    "except Exception:\n"
    "    pass\n"
    "_sg_random.seed({seed})\n"
    "_sg_np.random.seed({seed})\n"
)


def build_preamble(seed: int) -> str:
    """Render the shared sandbox seed preamble (D13)."""
    return SANDBOX_PREAMBLE_TEMPLATE.format(seed=int(seed))


def truncate_to_bytes(text: str, max_bytes: int) -> Tuple[str, bool]:
    """Truncate to a byte budget (D10): compare encoded length, cut bytes.

    Returns (possibly truncated text, truncated_flag). Decoding uses
    ``errors="replace"`` so a cut never splits a multi-byte character.
    """
    raw = (text or "").encode("utf-8", errors="replace")
    limit = int(max_bytes)
    if len(raw) <= limit:
        return text or "", False
    return raw[:limit].decode("utf-8", errors="replace") + "\n...[truncated]", True


# D12: leaked worker threads (Windows cannot hard-kill them) keep writing to
# their buffers after a soft timeout. All post-timeout reads take this lock
# so a racing writer cannot corrupt the snapshot, and every timeout result
# is poison-marked (leaked_thread=True) so callers never mistake a partial
# buffer for a complete run.
_BUFFER_LOCK = threading.Lock()
LEAKED_THREAD_COUNT = 0
_LEAK_COUNT_LOCK = threading.Lock()

FORBIDDEN_NAMES: Set[str] = {
    "exit",
    "quit",
    "open",  # replaced with safe_open for read-only of allowed paths if needed
    "exec",
    "eval",
    "compile",
    "__import__",
    "breakpoint",
    "input",
    "help",
}

FORBIDDEN_MODULES: Set[str] = {
    "subprocess",
    "multiprocessing",
    "ctypes",
    "socket",
    "http",
    "urllib",
    "requests",
    "os",
    "shutil",
    "pathlib",
    "importlib",
    "pty",
    "fcntl",
    "signal",
    "pickle",
    "shelve",
    "tempfile",
}

ALLOWED_IMPORT_ROOTS: Set[str] = {
    "numpy",
    "np",
    "pandas",
    "pd",
    "matplotlib",
    "plt",
    "sklearn",
    "scipy",
    "math",
    "statistics",
    "random",
    "json",
    "re",
    "collections",
    "itertools",
    "functools",
    "typing",
    "dataclasses",
    "copy",
    "time",
    "datetime",
    "hashlib",
    "decimal",
    "fractions",
    "string",
    "warnings",
    "textwrap",
    "heapq",
    "bisect",
    "array",
    "struct",
    "operator",
    "seaborn",
    "networkx",
    "sympy",
    "sys",
    "io",
    "statsmodels",
}


class SandboxViolation(Exception):
    pass


class SandboxASTValidator(ast.NodeVisitor):
    """Reject dangerous AST patterns before execution."""

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_MODULES or root not in ALLOWED_IMPORT_ROOTS:
                raise SandboxViolation(f"Import blocked: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_MODULES or root not in ALLOWED_IMPORT_ROOTS:
                raise SandboxViolation(f"Import blocked: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Block exit()/quit() and attribute calls like os.system / subprocess.run
        if isinstance(node.func, ast.Name) and node.func.id in ("exit", "quit", "exec", "eval", "compile", "__import__"):
            raise SandboxViolation(f"Call blocked: {node.func.id}()")
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("system", "popen", "remove", "rmdir", "unlink", "chdir", "kill"):
                raise SandboxViolation(f"Attribute call blocked: .{node.func.attr}()")
            if isinstance(node.func.value, ast.Name) and node.func.value.id in FORBIDDEN_MODULES:
                raise SandboxViolation(f"Module call blocked: {node.func.value.id}.{node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr not in ("__name__", "__doc__", "__class__", "__dict__", "__len__", "__iter__", "__getitem__"):
                # Allow common dunders used by libraries but block __builtins__ etc.
                if node.attr in ("__builtins__", "__import__", "__subclasses__", "__globals__", "__code__"):
                    raise SandboxViolation(f"Dunder access blocked: {node.attr}")
        self.generic_visit(node)


def validate_code(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        SandboxASTValidator().visit(tree)
        return True, ""
    except SandboxViolation as e:
        return False, str(e)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def extract_local_code_structure(
    tb: Any,
    source_code: str,
    generated_code_path: str = "<sandbox>",
    preamble_offset: int = 10,
) -> str:
    """Walk a traceback and extract the full source of custom functions implicated in it.

    Args:
        tb: A traceback object (from ``sys.exc_info()[2]`` or ``e.__traceback__``).
        source_code: The user-generated code (without the sandbox preamble).
        generated_code_path: The filename passed to ``compile()``; frames from
            other paths (stdlib, installed packages) are excluded.
        preamble_offset: Number of lines the sandbox preamble occupies in the
            compiled code.  Line numbers from the traceback are adjusted by
            subtracting this value before mapping into *source_code*.

    Returns:
        A formatted string combining the traceback text with the extracted
        function sources, clearly labelled.  Returns an empty string when
        *tb* is ``None``.
    """
    import traceback as _tb

    if tb is None:
        return ""

    # -- Parse source_code to locate every FunctionDef / AsyncFunctionDef ----
    func_map: Dict[int, Tuple[str, str]] = {}  # lineno -> (name, full_source)
    try:
        source_lines = source_code.splitlines()
        tree = ast.parse(source_code)
    except SyntaxError:
        # Can't parse — fall back to raw traceback text
        return "".join(_tb.format_tb(tb))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            func_source = "\n".join(source_lines[start - 1 : end])
            for ln in range(start, end + 1):
                func_map[ln] = (node.name, func_source)

    # -- Walk traceback frames, collect implicated custom functions -----------
    collected: Dict[str, str] = {}  # name -> source (deduplicates)
    for frame in _tb.extract_tb(tb):
        if frame.filename != generated_code_path:
            continue
        adjusted = frame.lineno - preamble_offset
        if adjusted in func_map:
            name, src = func_map[adjusted]
            collected[name] = src

    # -- Build output --------------------------------------------------------
    parts: List[str] = ["=== Traceback ==="]
    parts.extend(_tb.format_tb(tb))

    if collected:
        parts.append("\n=== Implicated Custom Functions ===")
        for name in sorted(collected):
            parts.append(f"\n--- {name}() ---")
            parts.append(collected[name])
    else:
        # No function-level match (error at module top-level or preamble).
        # Include a short window around the failing line for context.
        last_frame = _tb.extract_tb(tb)[-1] if _tb.extract_tb(tb) else None
        if last_frame and last_frame.filename == generated_code_path:
            adj = last_frame.lineno - preamble_offset
            if 1 <= adj <= len(source_lines):
                lo = max(0, adj - 4)
                hi = min(len(source_lines), adj + 3)
                parts.append("\n=== Source context (around failing line) ===")
                for i in range(lo, hi):
                    marker = ">>>" if i + 1 == adj else "   "
                    parts.append(f"{marker} {i + 1:4d} | {source_lines[i]}")

    return "\n".join(parts)


def _safe_builtins() -> Dict[str, Any]:
    # D14: __build_class__ + super are required for ordinary class/super
    # definitions; without them valid object-oriented experiment code fails.
    allowed = {
        "abs", "all", "any", "bool", "bytes", "callable", "chr", "complex",
        "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter",
        "len", "list", "map", "max", "min", "next", "oct", "ord", "pow",
        "print", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "type", "zip", "True", "False", "None",
        "super", "__build_class__",
        "Exception", "ValueError", "TypeError", "RuntimeError", "AssertionError",
        "StopIteration", "KeyError", "IndexError", "AttributeError",
    }
    ns = {name: getattr(builtins, name) for name in allowed if hasattr(builtins, name)}
    # Provide a restricted __import__ that only allows allowlisted modules
    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in FORBIDDEN_MODULES or root not in ALLOWED_IMPORT_ROOTS:
            raise SandboxViolation(f"Import blocked at runtime: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    ns["__import__"] = _restricted_import
    return ns


def execute_sandboxed(
    code: str,
    timeout_sec: int = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute code in-process with restricted builtins and AST validation.
    Note: true OS-level isolation would require containers; this blocks the
    documented Agent Laboratory failure modes (exit, subprocess, host installs).
    """
    timeout_sec = timeout_sec or config.sandbox_timeout_sec
    ok, err = validate_code(code)
    if not ok:
        return {"success": False, "error": f"Sandbox rejection: {err}", "stdout": "", "stderr": err}

    # Inject seed preamble (D13: shared template — container backend uses
    # the same builder). MPLBACKEND=Agg is set in config.py at import time
    # so matplotlib uses the non-interactive backend inside sandbox threads,
    # preventing Tcl_AsyncDelete on Windows.
    #
    # Also inject common aliases (np, pd, plt) so generated code that uses
    # these standard shorthand names works even if the generated import line
    # is malformed or missing in cheap_mode probes.
    preamble = build_preamble(seed)
    full_code = preamble + "\n" + code

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    # D14: __name__ belongs in globals — every class body reads it to set
    # __module__ (without it, valid class/super code dies at definition).
    local_ns: Dict[str, Any] = {"__name__": "__sandbox__"}
    global_ns: Dict[str, Any] = {"__builtins__": _safe_builtins(), "__name__": "__sandbox__"}

    try:
        # Soft wall-clock timeout (threads cannot be hard-killed on Windows, but
        # the graph stops waiting so engineering no longer hangs forever).
        compiled = compile(full_code, "<sandbox>", "exec")

        def _run():
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compiled, global_ns, local_ns)

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run)
            try:
                fut.result(timeout=timeout_sec)
            except FuturesTimeout:
                # D12: soft timeout — the worker thread leaks (Windows cannot
                # hard-kill threads). Snapshot the buffers under the lock and
                # poison-mark the result so the partial output is never
                # mistaken for a complete run.
                global LEAKED_THREAD_COUNT
                with _BUFFER_LOCK:
                    partial_stdout = stdout_buf.getvalue()
                    partial_stderr = stderr_buf.getvalue()
                with _LEAK_COUNT_LOCK:
                    LEAKED_THREAD_COUNT += 1
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "Sandbox timeout after %ss (seed=%s): worker thread leaked, "
                    "partial output marked, run continues without it",
                    timeout_sec, seed,
                )
                return {
                    "success": False,
                    "error": (
                        f"Sandbox timeout after {timeout_sec}s — experiment code ran too long "
                        "(worker thread continues in background; output below is partial)"
                    ),
                    "stdout": partial_stdout,
                    "stderr": f"timeout:{timeout_sec}",
                    "seed": seed,
                    "timeout": True,
                    "leaked_thread": True,
                }

        with _BUFFER_LOCK:
            stdout_full = stdout_buf.getvalue()
            stderr_full = stderr_buf.getvalue()
        # D10: parse the FULL stdout first, then truncate for transport.
        # Truncating before parsing silently drops the metrics line it was
        # meant to carry; the cap is a byte budget, not a char count.
        parsed = _parse_json_from_stdout(stdout_full)
        stdout, _ = truncate_to_bytes(stdout_full, config.sandbox_max_output_bytes)
        return {
            "success": True,
            "stdout": stdout,
            "stderr": stderr_full,
            "parsed": parsed,
            "seed": seed,
        }
    except SandboxViolation as e:
        return {
            "success": False,
            "error": f"Sandbox violation: {e}",
            "stdout": stdout_buf.getvalue(),
            "stderr": str(e),
            "seed": seed,
        }
    except Exception as e:
        preamble_offset = preamble.count("\n") + 1  # +1 for the separator line
        local_ctx = extract_local_code_structure(
            e.__traceback__, code, "<sandbox>", preamble_offset
        )
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "local_code_context": local_ctx,
            "stdout": stdout_buf.getvalue(),
            "stderr": stdout_buf.getvalue() + "\n" + traceback.format_exc(),
            "seed": seed,
        }


def run_known_answer_check(
    code: str,
    expected_metrics: Dict[str, float],
    tolerance: float = 1e-3,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run a small known-answer probe before accepting generated scientific code.

    D13: the probe seed is a parameter (default 0 preserves legacy calls).
    """
    result = execute_sandboxed(code, seed=seed)
    if not result.get("success"):
        return {"passed": False, "reason": result.get("error", "known-answer execution failed"), "result": result}
    metrics = (result.get("parsed") or {}).get("metrics") or {}
    mismatches = {
        key: {"expected": expected, "actual": metrics.get(key)}
        for key, expected in expected_metrics.items()
        if not isinstance(metrics.get(key), (int, float)) or abs(float(metrics[key]) - float(expected)) > tolerance
    }
    return {"passed": not mismatches, "mismatches": mismatches, "result": result}


def _parse_json_from_stdout(stdout: str) -> Dict[str, Any]:
    """Extract a JSON dict from stdout, tolerating warnings/preamble before the JSON.

    Strategy (ordered by specificity):
    1. Scan lines bottom-up for a single-line JSON object (fast path).
    2. Find the last '}' in the full text, then scan backward for the matching '{'.
    3. Try the entire stripped stdout as a single JSON object.
    """
    text = (stdout or "").strip()
    if not text:
        return {}

    # Fast path: scan lines bottom-up for single-line JSON
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # Handle multi-line JSON (e.g. indented print): find the outermost '{...}' pair.
    last_close = text.rfind("}")
    if last_close >= 0:
        # Scan backward from last_close to find the matching opening '{'
        depth = 0
        for i in range(last_close, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[i:last_close + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # wrong '{', keep looking

    # Last resort: try the whole output
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    return {}


def run_multi_seed(
    code: str,
    n_seeds: int = None,
    base_seed: int = 42,
    timeout_sec: int = None,
    wall_clock_sec: float = None,
) -> Dict[str, Any]:
    """Run the same experiment with multiple seeds; aggregate mean ± std.

    D11: ``timeout_sec`` threads through to every per-seed execution
    (previously the multi-seed path always used the default). When
    ``wall_clock_sec`` is set, the whole batch shares one deadline: each
    seed receives the remaining budget, and seeds that cannot start in time
    are recorded as wall-clock skips instead of hanging the run.
    """
    import time as _time

    import numpy as np

    n_seeds = n_seeds or config.experiment_seeds
    deadline = (_time.monotonic() + float(wall_clock_sec)) if wall_clock_sec else None
    runs: List[Dict[str, Any]] = []
    skipped: List[int] = []
    for i in range(n_seeds):
        seed = base_seed + i * 1009
        per_seed_timeout = timeout_sec
        if deadline is not None:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                skipped.append(seed)
                continue
            if per_seed_timeout is None or per_seed_timeout > remaining:
                per_seed_timeout = max(1, int(remaining))
        result = execute_sandboxed(code, timeout_sec=per_seed_timeout, seed=seed)
        runs.append(result)

    successes = [r for r in runs if r.get("success")]
    if not successes:
        # Collect actual error details from failed runs for debugging
        error_details = []
        for i, r in enumerate(runs):
            err = r.get("error") or "unknown"
            tb = r.get("traceback") or ""
            stderr = r.get("stderr") or ""
            detail = f"seed_{i}: {err}"
            if stderr:
                detail += f" | stderr: {stderr[:300]}"
            if tb:
                detail += f" | traceback: {tb[:300]}"
            error_details.append(detail)
        combined_error = "All seeded runs failed. Details:\n" + "\n".join(error_details)
        # Propagate the richest local_code_context from the failed runs
        # (last run is typically the most representative).
        local_ctx = ""
        for r in reversed(runs):
            if r.get("local_code_context"):
                local_ctx = r["local_code_context"]
                break
        return {
            "success": False,
            "error": combined_error,
            "runs": runs,
            "aggregate_metrics": {},
            "local_code_context": local_ctx,
            "wall_clock_skipped_seeds": skipped,
        }

    # Aggregate numeric metrics from parsed JSON
    metric_series: Dict[str, List[float]] = {}
    for r in successes:
        metrics = (r.get("parsed") or {}).get("metrics") or {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_series.setdefault(k, []).append(float(v))

    # If successes exist but produced no parseable metrics, treat as failure
    # to prevent infinite REFINE loops where code runs but produces no output
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
