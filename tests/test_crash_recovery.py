"""Regression tests for crash durability.

Covers the Sep-30 run that died 39 minutes in (LLM outage mid-debate) with
zero durable output: the pipeline now supports an on_crash hook, the CLI/web
wiring persists a failure dossier with a resume hint, phase snapshots preserve
expensive discovery/debate work, and the append-only logs rotate.
"""

import json


def test_pipeline_on_crash_receives_partial_state_and_reraises():
    from core.pipeline import ResearchPipeline

    seen = {}

    class ExplodingStream:
        def stream(self, source, config):
            yield {"topic_discovery": {"topics": [{"title": "T"}], "current_phase": "debate"}}
            raise RuntimeError("connection died mid-debate")

    class FakeGraph:
        def compile(self, checkpointer=None):
            return ExplodingStream()

    def on_crash(state, exc):
        seen["state"] = dict(state)
        seen["exc"] = exc

    pipeline = ResearchPipeline(
        graph_factory=lambda: FakeGraph(),
        checkpointer_factory=lambda: None,
    )
    try:
        pipeline.run(
            {"mode": "full_research", "run_id": "crash1"},
            "crash1",
            on_crash=on_crash,
        )
    except RuntimeError as e:
        assert "connection died" in str(e)
    else:  # pragma: no cover
        raise AssertionError("original exception must propagate")
    assert seen["state"]["topics"] == [{"title": "T"}]
    assert isinstance(seen["exc"], RuntimeError)


def test_pipeline_on_crash_failure_never_masks_original():
    from core.pipeline import ResearchPipeline

    class ExplodingStream:
        def stream(self, source, config):
            raise RuntimeError("original boom")

    class FakeGraph:
        def compile(self, checkpointer=None):
            return ExplodingStream()

    def bad_handler(state, exc):
        raise ValueError("handler bug")

    pipeline = ResearchPipeline(
        graph_factory=lambda: FakeGraph(),
        checkpointer_factory=lambda: None,
    )
    try:
        pipeline.run({"run_id": "crash2"}, "crash2", on_crash=bad_handler)
    except RuntimeError as e:
        assert "original boom" in str(e)
    else:  # pragma: no cover
        raise AssertionError("handler failure must not mask the original error")


def test_save_phase_snapshot_writes_json_safe_subset(tmp_path):
    from core.artifacts import save_phase_snapshot

    class FakeResult:
        topic = "T1"
        passed = True
        score = 8.0
        moderator_decision = "PASS"

    state = {
        "run_id": "snap1",
        "mode": "full_research",
        "current_phase": "planning",
        "iteration": 0,
        "topics": [{"title": "T1"}, {"title": "T2"}],
        "selected_topic": {"title": "T1"},
        "debate_results": [FakeResult()],
        "plan": {"title": "P", "experiments": [{"name": "exp1"}]},
        "terminal_error": None,
    }
    path = save_phase_snapshot(state, "snap1", "hypothesis_debate", output_dir=str(tmp_path))
    assert path is not None
    snapshot = json.loads(open(path, encoding="utf-8").read())
    assert snapshot["run_id"] == "snap1"
    assert snapshot["node"] == "hypothesis_debate"
    assert snapshot["topics"] == ["T1", "T2"]
    assert snapshot["debate_outcomes"][0]["passed"] is True
    assert snapshot["plan_experiments"] == ["exp1"]


def test_failure_dossier_carries_resume_hint(tmp_path):
    from core.artifacts import save_failure_dossier

    state = {
        "mode": "full_research",
        "run_id": "resume9",
        "terminal_error": "Unhandled crash in debate: boom",
        "evidence_gate": {},
        "experiment_contracts": {},
        "technical_failures": {},
        "engineer_outputs": {},
        "execution_artifacts": {},
        "verification_findings": [],
        "meta_feedback": [],
        "draft_sections": {},
        "supervisor_scores": {},
        "results_verification": {},
        "reproducibility": {},
        "plan": {},
        "debate_results": [],
        "plan_revision_requests": [],
        "current_phase": "debate",
        "iteration": 0,
    }
    dossier_path = save_failure_dossier(state, str(tmp_path))
    dossier = json.loads(open(dossier_path, encoding="utf-8").read())
    assert dossier["resume_hint"] == "python main.py --resume resume9"
    assert "phase_snapshots" in (dossier["phase_snapshots_dir"] or "")


def test_prune_jsonl_file_trims_only_when_oversized(tmp_path):
    from core.run_log import prune_jsonl_file

    small = tmp_path / "small.jsonl"
    small.write_text('{"a": 1}\n', encoding="utf-8")
    assert prune_jsonl_file(str(small), max_bytes=10**9) == 0
    assert small.read_text(encoding="utf-8") == '{"a": 1}\n'

    big = tmp_path / "big.jsonl"
    lines = [f'{{"n": {i}}}\n' for i in range(100)]
    big.write_text("".join(lines), encoding="utf-8")
    assert prune_jsonl_file(str(big), max_bytes=10, keep_lines=10) == 90
    kept = big.read_text(encoding="utf-8").splitlines()
    assert len(kept) == 10
    assert json.loads(kept[-1]) == {"n": 99}


def test_test_session_does_not_touch_production_logs(tmp_path):
    """The conftest isolation fixture must redirect all writable artifact paths."""
    from pathlib import Path

    from core.config import config

    for attr in (
        "run_events_path",
        "run_log_path",
        "cross_run_memory_path",
        "research_db_path",
        "elo_ratings_path",
        "debate_log_path",
        "feedback_log_path",
    ):
        assert Path(getattr(config, attr)).parent == tmp_path, attr


def test_corrupt_log_files_load_as_empty(tmp_path):
    """Empty / garbage debate & feedback logs must not crash startup loading."""
    from core.memory import _load_json_list

    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert _load_json_list(str(empty), "empty.json") == []

    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")
    assert _load_json_list(str(garbage), "garbage.json") == []

    wrong_shape = tmp_path / "object.json"
    wrong_shape.write_text('{"a": 1}', encoding="utf-8")
    assert _load_json_list(str(wrong_shape), "object.json") == []

    good = tmp_path / "good.json"
    good.write_text('[{"a": 1}]', encoding="utf-8")
    assert _load_json_list(str(good), "good.json") == [{"a": 1}]


def test_atomic_write_never_leaves_partial_file(tmp_path):
    """A failed dump must leave the previous complete file intact (the 0-byte bug)."""
    from core.memory import _atomic_write_json

    target = tmp_path / "debate_log.json"
    target.write_text('[{"old": true}]', encoding="utf-8")
    circular = []
    circular.append(circular)  # json.dump raises ValueError on this
    try:
        _atomic_write_json(str(target), circular)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected the circular write to fail")
    assert target.read_text(encoding="utf-8") == '[{"old": true}]'
    assert list(tmp_path.glob("*.tmp")) == []

    _atomic_write_json(str(target), [{"new": 1}])
    assert json.loads(target.read_text(encoding="utf-8")) == [{"new": 1}]
