"""Engineering Arena tests (all offline — subprocess is stubbed, never spawned)."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

from core import engineering_arena as arena


GOOD_CODE = """import json
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=200, n_features=8, random_state=42)
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)
acc = float(accuracy_score(y, clf.predict(X)))
print(json.dumps({"metrics": {"accuracy": acc}, "raw": {}}))
"""

GOOD_METRICS = {
    "metrics": {"accuracy": {"mean": 0.91, "std": 0.01, "n": 2}},
    "contract_hash": "abc",
    "seeds": [42, 43],
}

EXPERIMENT = {
    "name": "E1 Probe Test",
    "purpose": "probe",
    "methodology": "synthetic",
    "baselines": ["LogisticRegression"],
    "claimed_components": [],
    "evaluation_metrics": ["accuracy"],
    "falsifiable_prediction": "acc > 0.5",
    "statistical_test": "none",
    "analysis_plan": {},
    "data_requirements": "synthetic",
    "contract_hash": "abc",
}


def _use_tmp_arena(monkeypatch, tmp_path):
    monkeypatch.setattr("core.engineering_arena.arena_root", lambda: str(tmp_path / "arena"))
    return str(tmp_path / "arena")


class _StubEngineer:
    """Duck-typed EngineerAgent: records calls, returns canned outputs."""

    def __init__(self, seeds=2):
        self.runtime_config = SimpleNamespace(experiment_seeds=seeds)
        self.progress_events = []
        self.revisions = []
        self.stored = []

    def _progress(self, event_type, data):
        self.progress_events.append((event_type, data))

    def request_plan_revision(self, reason, experiment, detail=""):
        req = {"reason": reason, "experiment": experiment.get("name"), "detail": detail}
        self.revisions.append(req)
        return req

    def check_code_claim_consistency(self, experiment, code):
        return {"consistent": True, "score": 10.0, "notes": []}

    def _build_arena_success_output(self, experiment, code, agg, decision_log,
                                    tdir, consistency=None, original_name=None,
                                    multi_seed=None):
        out = {"success": True, "experiment_name": experiment.get("name"),
               "aggregate_metrics": agg, "code": code,
               "results": {"metrics": {k: v["mean"] for k, v in agg.items()}},
               "decision_log": decision_log, "arena_task_dir": tdir,
               "multi_seed": multi_seed}
        self.stored.append(out)
        return out

    def _fail(self, experiment, error, decision_log, code="", failure_kind="technical",
              original_name=None):
        return {"success": False, "experiment_name": experiment.get("name"),
                "error": str(error), "decision_log": decision_log,
                "failure_kind": failure_kind}


def _write_artifacts(tdir, status, metrics=None, code=GOOD_CODE):
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, arena.CODE_FILENAME), "w", encoding="utf-8") as f:
        f.write(code)
    if metrics is not None:
        with open(os.path.join(tdir, arena.METRICS_FILENAME), "w", encoding="utf-8") as f:
            json.dump(metrics, f)
    with open(os.path.join(tdir, arena.STATUS_FILENAME), "w", encoding="utf-8") as f:
        json.dump(status, f)


# --- task rendering ---------------------------------------------------------

def test_task_md_renders_contract_and_protocol():
    md = arena.render_task_md(EXPERIMENT, seeds=2, prior_context="prior summary",
                              dataset_plan="synthetic")
    for needle in ("E1 Probe Test", "LogisticRegression", "accuracy",
                   "STATUS.json", "Docker", "experiment.py", "ARENA_SEED",
                   "needs_input", "plan_update_request.json",
                   "contract_hash", "prior summary", "Allowed imports ONLY",
                   "Do not run or execute"):
        assert needle in md, needle


def test_task_files_written_with_context_and_policy(tmp_path, monkeypatch):
    _use_tmp_arena(monkeypatch, tmp_path)
    tdir = os.path.join(str(tmp_path), "arena", "run1", "e1")
    arena.write_task_files(tdir, EXPERIMENT, seeds=2, prior_summaries="mem",
                           dataset_plan="synthetic", feedback="fix x")
    assert os.path.exists(os.path.join(tdir, "task.md"))
    ctx = open(os.path.join(tdir, "context.md"), encoding="utf-8").read()
    assert "mem" in ctx
    policy = json.load(open(os.path.join(tdir, "opencode.json"), encoding="utf-8"))
    assert "permission" in policy
    assert policy["permission"]["bash"] == "deny"


# --- run lifecycle: isolation + same-run memory ------------------------------

def test_run_isolation_and_same_run_memory(tmp_path, monkeypatch):
    _use_tmp_arena(monkeypatch, tmp_path)
    arena.ensure_run("runA", topic="paper one")
    arena.ensure_run("runB", topic="paper two")
    arena.record_experiment_summary("runA", "E1", "accuracy 0.91")
    mem_a = arena.read_run_memory("runA")
    mem_b = arena.read_run_memory("runB")
    assert "accuracy 0.91" in mem_a
    assert "accuracy 0.91" not in mem_b
    assert os.path.isdir(arena.run_dir("runA"))
    assert os.path.isdir(arena.run_dir("runB"))


def test_sweep_removes_only_stale(tmp_path, monkeypatch):
    root = _use_tmp_arena(monkeypatch, tmp_path)
    arena.ensure_run("fresh")
    arena.ensure_run("old")
    arena.ensure_run("pinned")
    old_mtime = time.time() - 30 * 86400
    os.utime(arena.run_dir("old"), (old_mtime, old_mtime))
    os.utime(arena.run_dir("pinned"), (old_mtime, old_mtime))
    removed = arena.sweep_stale_runs(retention_days=7, keep_run_id="pinned")
    assert "old" in removed
    assert "pinned" not in removed
    assert os.path.isdir(arena.run_dir("fresh"))
    assert os.path.isdir(arena.run_dir("pinned"))
    assert not os.path.exists(arena.run_dir("old"))


# --- spawning ----------------------------------------------------------------

def test_spawn_unavailable_without_binary(monkeypatch):
    monkeypatch.setattr(arena.shutil, "which", lambda *a, **k: None)
    with pytest.raises(arena.ArenaUnavailable):
        arena.spawn_session("/tmp/nowhere", "hi", timeout_sec=5)


def test_spawn_invokes_opencode_with_dir_and_format(monkeypatch, tmp_path):
    monkeypatch.setattr(arena.shutil, "which", lambda *a, **k: "opencode")
    captured = {}

    class _Proc:
        returncode = 0
        stdout = '{"type":"session.created","sessionID":"sess-123"}\n{"type":"text"}\n'
        stderr = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(arena.subprocess, "run", _fake_run)
    rec = arena.spawn_session(str(tmp_path), "do it", timeout_sec=60,
                              model="m", agent="build")
    cmd = captured["cmd"]
    flat = " ".join(cmd) if isinstance(cmd, list) else cmd
    assert "--dir" in flat and "--format" in flat and "json" in flat
    assert "--file=" in flat and "task.md" in flat
    assert "--auto" not in flat
    assert rec["returncode"] == 0 and rec["session_id"] == "sess-123"


# --- metrics normalization ----------------------------------------------------

def test_normalize_metrics_flat_and_nested():
    agg, errs = arena.normalize_metrics({"metrics": {"f1": 0.9}}, seeds=3)
    assert not errs and agg["f1"] == {"mean": 0.9, "std": 0.0, "n": 3}
    agg, errs = arena.normalize_metrics(GOOD_METRICS, seeds=2)
    assert not errs and agg["accuracy"]["mean"] == 0.91
    agg, errs = arena.normalize_metrics({}, seeds=3)
    assert not agg and errs


# --- artifact validation (real deterministic gates) ----------------------------

def test_validate_arena_artifacts_good_and_bad(tmp_path):
    good = str(tmp_path / "good")
    _write_artifacts(good, {"status": "done"}, GOOD_METRICS, GOOD_CODE)
    ok, errors, code, agg = arena.validate_arena_artifacts(good, EXPERIMENT, seeds=2)
    assert ok, errors
    assert not agg
    assert "LogisticRegression" in code

    bad = str(tmp_path / "bad")
    _write_artifacts(bad, {"status": "done"}, {}, "")
    ok, errors, _, _ = arena.validate_arena_artifacts(bad, EXPERIMENT, seeds=2)
    assert not ok


def _enable_docker(monkeypatch):
    monkeypatch.setattr(
        "core.container_sandbox.ensure_sandbox_image",
        lambda auto_build=True: {"ok": True, "backend": "docker"},
    )


def _docker_result(seeds=(42, 1051), values=(0.9, 0.95)):
    return {
        "success": True,
        "n_seeds": len(seeds),
        "n_success": len(seeds),
        "aggregate_metrics": {
            "accuracy": {"mean": 0.01, "std": 0.0, "n": len(seeds)},
        },
        "wall_clock_skipped_seeds": [],
        "runs": [
            {
                "success": True,
                "seed": seed,
                "parsed": {"metrics": {"accuracy": value}},
            }
            for seed, value in zip(seeds, values)
        ],
    }


def test_runner_results_are_seed_checked_and_trusted(tmp_path):
    good_dir = str(tmp_path / "runner-good")
    _write_artifacts(good_dir, {"status": "done"}, {"metrics": {"accuracy": 0.01}}, GOOD_CODE)
    ok, errors, _, agg = arena.validate_arena_artifacts(
        good_dir, EXPERIMENT, seeds=2, runner_result=_docker_result()
    )
    assert ok, errors
    assert agg["accuracy"]["values"] == [0.9, 0.95]
    assert agg["accuracy"]["mean"] == pytest.approx(0.925)
    trusted = arena.read_metrics(good_dir)
    assert trusted["contract_hash"] == EXPERIMENT["contract_hash"]
    assert trusted["seeds"] == [42, 1051]
    assert trusted["metrics"] == agg

    bad_dir = str(tmp_path / "runner-bad")
    _write_artifacts(bad_dir, {"status": "done"}, {}, GOOD_CODE)
    wrong_seeds = _docker_result(seeds=(42, 43))
    ok, errors, _, _ = arena.validate_arena_artifacts(
        bad_dir, EXPERIMENT, seeds=2, runner_result=wrong_seeds
    )
    assert not ok
    assert any("do not match expected seeds" in error for error in errors) and errors


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (_docker_result(values=(0.9, float("nan"))), "not a finite number"),
        (_docker_result(values=(0.9, float("inf"))), "not a finite number"),
        (_docker_result(values=(0.9, True)), "not a finite number"),
    ],
)
def test_runner_rejects_nonfinite_or_non_numeric_metrics(tmp_path, result, message):
    tdir = str(tmp_path / "invalid-metric")
    _write_artifacts(tdir, {"status": "done"}, None, GOOD_CODE)
    ok, errors, _, _ = arena.validate_arena_artifacts(
        tdir, EXPERIMENT, seeds=2, runner_result=result
    )
    assert not ok
    assert any(message in error for error in errors)


def test_runner_requires_contract_hash(tmp_path):
    tdir = str(tmp_path / "missing-contract")
    _write_artifacts(tdir, {"status": "done"}, None, GOOD_CODE)
    experiment = {**EXPERIMENT, "contract_hash": ""}
    ok, errors, _, _ = arena.validate_arena_artifacts(
        tdir, experiment, seeds=2, runner_result=_docker_result()
    )
    assert not ok
    assert any("contract_hash is missing" in error for error in errors) and errors


# --- orchestration -------------------------------------------------------------

def _stub_spawn_factory(tdir_writer):
    def _spawn(tdir, brief, timeout_sec=1800, model="", agent=""):
        tdir_writer(tdir)
        return {"returncode": 0, "timed_out": False, "elapsed_sec": 1.0,
                "session_id": "s", "stdout_chars": 10, "stderr_chars": 0,
                "stdout": "", "stderr_tail": ""}
    return _spawn


def test_done_round_returns_success_output(tmp_path, monkeypatch):
    _use_tmp_arena(monkeypatch, tmp_path)
    _enable_docker(monkeypatch)
    monkeypatch.setattr(arena, "spawn_session",
                        _stub_spawn_factory(
                            lambda tdir: _write_artifacts(
                                tdir, {"status": "done", "metrics": {"accuracy": 0.01}},
                                GOOD_METRICS, GOOD_CODE)))
    monkeypatch.setattr(
        "core.container_sandbox.container_multi_seed",
        lambda code, **kwargs: _docker_result(),
    )
    eng = _StubEngineer()
    out = arena.run_arena_experiment(eng, dict(EXPERIMENT), run_id="runDone")
    assert out["success"] is True
    assert out["results"]["metrics"]["accuracy"] == pytest.approx(0.925)
    assert out["multi_seed"]["n_success"] == 2
    assert any(e[0] == "arena_start" for e in eng.progress_events)
    trusted = arena.read_metrics(os.path.join(
        arena.task_dir("runDone", "E1 Probe Test"),
        os.listdir(arena.task_dir("runDone", "E1 Probe Test"))[0],
    ))
    assert trusted["metrics"]["accuracy"]["mean"] == pytest.approx(0.925)
    mem = arena.read_run_memory("runDone")
    assert "E1 Probe Test" in mem  # failures record; success path stores via engineer


def test_needs_input_surfaces_question_and_revision(tmp_path, monkeypatch):
    _use_tmp_arena(monkeypatch, tmp_path)
    _enable_docker(monkeypatch)
    question = {"status": "needs_input", "question": "Which dataset?",
                "context": "catalog miss", "options": ["A", "B"],
                "blocked_on": "dataset"}
    monkeypatch.setattr(arena, "spawn_session",
                        _stub_spawn_factory(lambda tdir: _write_artifacts(tdir, question)))
    eng = _StubEngineer()
    out = arena.run_arena_experiment(eng, dict(EXPERIMENT), run_id="runQ")
    assert out["success"] is False
    assert out["failure_kind"] == "arena_needs_input"
    assert out["arena_question"]["question"] == "Which dataset?"
    assert eng.revisions and eng.revisions[0]["reason"] == "arena_needs_input"
    qpath = os.path.join(out["arena_task_dir"], arena.QUESTIONS_FILENAME)
    assert "Which dataset?" in open(qpath, encoding="utf-8").read()


def test_no_status_retries_bounded_then_fails(tmp_path, monkeypatch):
    from core.config import config as _cfg

    _use_tmp_arena(monkeypatch, tmp_path)
    _enable_docker(monkeypatch)
    monkeypatch.setattr(_cfg, "arena_max_rounds", 2)
    calls = []

    def _spawn(tdir, brief, timeout_sec=1800, model="", agent="", auto=False):
        calls.append(brief)
        os.makedirs(tdir, exist_ok=True)  # no STATUS.json written
        return {"returncode": 0, "timed_out": False, "elapsed_sec": 1.0,
                "session_id": "s", "stdout_chars": 0, "stderr_chars": 0,
                "stdout": "", "stderr_tail": ""}

    monkeypatch.setattr(arena, "spawn_session", _spawn)
    eng = _StubEngineer()
    out = arena.run_arena_experiment(eng, dict(EXPERIMENT), run_id="runEmpty")
    assert out["success"] is False
    assert len(calls) == 2  # bounded by arena_max_rounds
    assert "arena_no_artifacts" in out["error"]
    fb = open(os.path.join(out["arena_task_dir"], arena.FEEDBACK_FILENAME), encoding="utf-8").read()
    assert "STATUS.json" in fb


def test_invalid_runner_results_become_feedback_and_retry(tmp_path, monkeypatch):
    from core.config import config as _cfg

    _use_tmp_arena(monkeypatch, tmp_path)
    _enable_docker(monkeypatch)
    monkeypatch.setattr(_cfg, "arena_max_rounds", 1)
    monkeypatch.setattr(arena, "spawn_session",
                        _stub_spawn_factory(
                            lambda tdir: _write_artifacts(
                                tdir, {"status": "done"}, None, GOOD_CODE)))
    monkeypatch.setattr(
        "core.container_sandbox.container_multi_seed",
        lambda code, **kwargs: _docker_result(seeds=(42,), values=(0.9,)),
    )
    eng = _StubEngineer()
    out = arena.run_arena_experiment(eng, dict(EXPERIMENT), run_id="runBad")
    assert out["success"] is False
    assert "arena_invalid_results" in out["error"]


def test_docker_unavailable_fails_closed_before_agent_spawn(tmp_path, monkeypatch):
    _use_tmp_arena(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "core.container_sandbox.ensure_sandbox_image",
        lambda auto_build=True: {"ok": False, "reason": "docker_unreachable"},
    )
    monkeypatch.setattr(
        arena, "spawn_session",
        lambda *args, **kwargs: pytest.fail("opencode must not start without Docker"),
    )
    out = arena.run_arena_experiment(_StubEngineer(), dict(EXPERIMENT), run_id="noDocker")
    assert out["success"] is False
    assert out["failure_kind"] == "arena_sandbox_unavailable"
    assert "Docker execution is required" in out["error"]


# --- engineer wiring ------------------------------------------------------------

def test_engineer_dispatches_to_arena(monkeypatch):
    from agents.engineer import EngineerAgent
    from core.config import config as _cfg

    monkeypatch.setattr("agents.engineer.get_llm_client", lambda: SimpleNamespace())
    monkeypatch.setattr(_cfg, "arena_backend", "opencode")
    seen = {}

    def _fake_arena(engineer, experiment, method_description="", run_id=None,
                    alternatives=None):
        seen["experiment"] = experiment.get("name")
        return {"success": True, "experiment_name": experiment.get("name")}

    monkeypatch.setattr("core.engineering_arena.run_arena_experiment", _fake_arena)
    agent = EngineerAgent()
    out = agent.run_experiment(dict(EXPERIMENT))
    assert out["success"] is True
    assert seen["experiment"] == "E1 Probe Test"


def test_engineer_auto_falls_back_to_legacy(monkeypatch):
    from agents.engineer import EngineerAgent
    from core.config import config as _cfg

    monkeypatch.setattr("agents.engineer.get_llm_client", lambda: SimpleNamespace())
    monkeypatch.setattr(_cfg, "arena_backend", "auto")

    def _no_arena(*a, **k):
        raise arena.ArenaUnavailable("no binary")

    monkeypatch.setattr("core.engineering_arena.run_arena_experiment", _no_arena)
    # Legacy path, fully stubbed below the dispatch.
    monkeypatch.setattr(EngineerAgent, "_generate_experiment_code",
                        lambda self, exp, attempt=1: GOOD_CODE)
    monkeypatch.setattr("agents.engineer.execute_multi_seed",
                        lambda code, n_seeds=3, wall_clock_sec=1800: {
                            "success": True,
                            "aggregate_metrics": {"accuracy": {"mean": 0.9, "std": 0.0}},
                            "results": {"metrics": {"accuracy": 0.9}}})
    monkeypatch.setattr(EngineerAgent, "_auto_ablation", lambda self, e, c: {})
    monkeypatch.setattr(EngineerAgent, "_store_raw", lambda self, n, m: "raw.json")
    monkeypatch.setattr(EngineerAgent, "_store_experiment_results",
                        lambda self, o, e: None)
    agent = EngineerAgent()
    out = agent.run_experiment(dict(EXPERIMENT))
    assert out["success"] is True
    assert out["results"]["metrics"]["accuracy"] == 0.9
