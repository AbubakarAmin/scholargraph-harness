"""M6 hardening (§11-M6 steps 1–2): parallel fan-out plan + step accounting.

1. PARALLEL_PHASES_ENABLED opt-in fan-out (Methods ∥ engineering) + soak;
   default stays False, flag-off behavior sequential and unchanged.
2. Per-phase step accounting (phase_steps) vs recursion budget with a
   budget_exhausted terminal (§1.5 taxonomy) + budget_check emit (§10.4).

All offline/mocked — no API keys, no threads, no network.
"""

from core.config import config
from core.phase_registry import (
    PIPELINES,
    apply_step_budget,
    check_step_budget,
    parallel_plan_for_recipe,
    record_phase_step,
    total_phase_steps,
)
from core.state import ResearchState, initialize_state, migrate_state_v1_to_v2
from core.workflow import (
    DEFAULT_RECURSION_LIMIT,
    _after_phase,
    create_typed_graph,
)


# ---------------------------------------------------------------------------
# M6.1 — parallel defaults off (sequential unchanged)
# ---------------------------------------------------------------------------

def test_parallel_default_flag_is_false():
    assert config.parallel_phases_enabled is False, \
        "PARALLEL_PHASES_ENABLED default must stay false (§9)"


def test_parallel_defaults_off_sequential_unchanged(monkeypatch):
    monkeypatch.setattr(config, "parallel_phases_enabled", False)
    assert parallel_plan_for_recipe(PIPELINES["experimental"]) == []
    assert parallel_plan_for_recipe(PIPELINES["comparative"]) == []
    assert parallel_plan_for_recipe(None) == []
    # No ThreadPoolExecutor may back external-API fan-out (AGENTS rule).
    import pathlib

    for rel in ("core/phase_registry.py", "core/workflow.py", "core/pipeline.py"):
        src = pathlib.Path(rel).read_text(encoding="utf-8")
        assert "from concurrent.futures import" not in src, f"{rel} must not fan out with threads"
        assert "import concurrent.futures" not in src, f"{rel} must not fan out with threads"


# ---------------------------------------------------------------------------
# M6.1 — opt-in returns Methods ∥ engineering only when both present
# ---------------------------------------------------------------------------

def test_parallel_opt_in_methods_engineering_pair(monkeypatch):
    monkeypatch.setattr(config, "parallel_phases_enabled", True)
    assert parallel_plan_for_recipe(PIPELINES["experimental"]) == [
        ("writing_narrative", "engineering")
    ]
    # Explicit override wins over the flag.
    assert parallel_plan_for_recipe(PIPELINES["experimental"], enabled=False) == []


def test_parallel_opt_in_requires_both_phases(monkeypatch):
    monkeypatch.setattr(config, "parallel_phases_enabled", True)
    # analytical: narrative but no engineering → no pair.
    assert parallel_plan_for_recipe(PIPELINES["analytical"]) == []
    # cause_effect: engineering but no writing_narrative → no pair.
    assert parallel_plan_for_recipe(PIPELINES["cause_effect"]) == []
    # survey: neither → no pair.
    assert parallel_plan_for_recipe(PIPELINES["survey"]) == []


def test_parallel_soak_deterministic_across_recipes(monkeypatch):
    """Soak: plan is a pure function of (flag, recipe) — stable across runs."""
    monkeypatch.setattr(config, "parallel_phases_enabled", True)
    first = {t: parallel_plan_for_recipe(r) for t, r in PIPELINES.items()}
    second = {t: parallel_plan_for_recipe(r) for t, r in PIPELINES.items()}
    assert first == second
    paired = sorted(t for t, pairs in first.items() if pairs)
    # Only recipes containing BOTH writing_narrative and engineering.
    assert paired == ["comparative", "experimental"]
    for pairs in first.values():
        for pair in pairs:
            assert pair == ("writing_narrative", "engineering")


# ---------------------------------------------------------------------------
# M6.2 — per-phase step accounting
# ---------------------------------------------------------------------------

def test_state_carries_phase_steps():
    state = initialize_state()
    assert state["phase_steps"] == {}
    assert "phase_steps" in ResearchState.__annotations__
    legacy = {"mode": "full_research", "iteration": 0}
    assert migrate_state_v1_to_v2(legacy)["phase_steps"] == {}


def test_step_accounting_increments_per_phase():
    state: dict = {}
    record_phase_step(state, "planning")
    record_phase_step(state, "planning")
    record_phase_step(state, "engineering")
    assert state["phase_steps"] == {"planning": 2, "engineering": 1}
    assert total_phase_steps(state) == 3
    assert total_phase_steps({}) == 0


def test_step_budget_trips_at_limit():
    state: dict = {"phase_steps": {"a": 2, "b": 1}}
    tripped, refusal = check_step_budget(state, recursion_limit=3)
    assert tripped is True
    assert refusal is not None
    assert refusal["reason"] == "budget_exhausted"
    for key in ("reason", "provided", "alternative", "how_to_proceed"):
        assert refusal[key], f"§1.5 refusal must carry {key}"

    tripped, refusal = check_step_budget(state, recursion_limit=4)
    assert (tripped, refusal) == (False, None)
    tripped, _ = check_step_budget({}, recursion_limit=DEFAULT_RECURSION_LIMIT)
    assert tripped is False


def test_step_budget_terminal_and_budget_check_emit(monkeypatch):
    events: list = []

    def _recorder(event_type, data, run_id=None, agent=None):
        events.append({"type": event_type, "data": data, "run_id": run_id, "agent": agent})

    import core.run_log as run_log

    monkeypatch.setattr(run_log, "emit_event", _recorder)
    state: dict = {"phase_steps": {"planning": 5}, "run_id": "r-m6", "evidence_gate": {}}
    assert apply_step_budget(state, recursion_limit=5) is True
    assert str(state["terminal_error"]).startswith("budget_exhausted")
    assert state["current_phase"] == "complete"
    assert state["should_continue"] is False
    assert state["evidence_gate"]["reason_code"] == "budget_exhausted"
    budget_checks = [e for e in events if e["type"] == "budget_check"]
    assert len(budget_checks) == 1, "trip must emit exactly one budget_check (§10.4)"
    assert budget_checks[0]["data"]["cause_class"] == "budget"


def test_step_budget_no_trip_no_emit(monkeypatch):
    events: list = []
    import core.run_log as run_log

    monkeypatch.setattr(run_log, "emit_event", lambda *a, **k: events.append((a, k)))
    state: dict = {"phase_steps": {"planning": 1}}
    assert apply_step_budget(state, recursion_limit=1000) is False
    assert "terminal_error" not in state
    assert events == []


# ---------------------------------------------------------------------------
# Legacy-path regression: routing unchanged, wrapper records without harm
# ---------------------------------------------------------------------------

def test_after_phase_routing_unchanged_below_budget():
    assert _after_phase([], {"should_continue": True}) == "continue"
    assert _after_phase([], {"should_continue": False}) == "end"
    assert _after_phase([], {"should_continue": True, "terminal_error": "x"}) == "end"
    assert _after_phase([], {"should_continue": True, "should_reset": True}) == "reset"
    # Tripped step budget routes to end even without a stamped terminal.
    big = {"should_continue": True, "phase_steps": {"loop": DEFAULT_RECURSION_LIMIT}}
    assert _after_phase([], big) == "end"


def test_wrapped_typed_graph_records_steps_offline():
    def _a(state):
        out = dict(state)
        out["current_phase"] = "mid"
        return out

    def _b(state):
        out = dict(state)
        out["current_phase"] = "complete"
        out["should_continue"] = False
        return out

    graph = create_typed_graph({"a": _a, "b": _b}, ["a", "b"])
    final = graph.compile().invoke(
        {"should_continue": True, "phase_steps": {}},
        {"recursion_limit": 30},
    )
    assert final["phase_steps"] == {"a": 1, "b": 1}
    assert not str(final.get("terminal_error") or "").startswith("budget_exhausted")


# ---------------------------------------------------------------------------
# M6 steps 5+7 — R10: consistency-referee error path is non-blocking
# (proposal §6.1 R10: referee LLM-error → non-blocking warning + retry,
# require explicit blocking == True)
# ---------------------------------------------------------------------------

def test_r10_referee_llm_exception_is_non_blocking_warning(monkeypatch):
    from core import verification

    def boom(*_args, **_kwargs):
        raise RuntimeError("llm_unavailable")

    monkeypatch.setattr(verification, "call_llm", boom)
    result = verification.consistency_referee({"Results": "clean"})
    assert result["passed"] is True
    assert result["findings"][0]["category"] == "referee_error"
    assert result["findings"][0]["blocking"] is False


def test_r10_referee_malformed_output_is_non_blocking_warning(monkeypatch):
    from core import verification

    monkeypatch.setattr(verification, "call_llm", lambda *_a, **_k: "not json")
    result = verification.consistency_referee({"Results": "No contradictions"})
    assert result["passed"] is True
    assert result["findings"][0]["category"] == "referee_error"
    assert result["findings"][0]["blocking"] is False


def test_r10_referee_explicit_blocking_true_still_blocks(monkeypatch):
    from core import verification

    monkeypatch.setattr(
        verification,
        "call_llm",
        lambda *_a, **_k: '{"findings": [{"category": "numeric", "message": "mismatch", "blocking": true}]}',
    )
    result = verification.consistency_referee({"Methods": "A", "Results": "B"})
    assert result["passed"] is False
    assert result["findings"][0]["blocking"] is True


def test_r10_referee_missing_blocking_flag_is_advisory(monkeypatch):
    # Explicit ``blocking == True`` is required; an absent flag no longer
    # fail-closes via ``.get("blocking", True)``.
    from core import verification

    monkeypatch.setattr(
        verification,
        "call_llm",
        lambda *_a, **_k: '{"findings": [{"category": "method", "message": "style drift"}]}',
    )
    result = verification.consistency_referee({"Methods": "A"})
    assert result["passed"] is True


def test_r10_referee_explicit_blocking_false_passes(monkeypatch):
    from core import verification

    monkeypatch.setattr(
        verification,
        "call_llm",
        lambda *_a, **_k: '{"findings": [{"category": "method", "message": "minor", "blocking": false}]}',
    )
    result = verification.consistency_referee({"Methods": "A"})
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# M6 steps 5+7 — R12: independent-validation skip for non-experimental
# empty outputs (proposal §6.1 R12); legacy experimental path unchanged
# ---------------------------------------------------------------------------

def _r12_skip_state(paper_type):
    state = initialize_state()
    state.update({
        "paper_type": paper_type,
        "plan": {"experiments": [], "plan_mode": "non_empirical"},
        "engineer_outputs": {},
        "draft_sections": {"Introduction": "Background on widgets (Smith 2021)."},
    })
    return state


def test_r12_skip_on_empty_outputs_non_experimental():
    from core.workflow_nodes import independent_validation_node

    for paper_type in ("analytical", "review", "survey", "case_study", "descriptive"):
        state = _r12_skip_state(paper_type)
        result = independent_validation_node(state)
        assert result.get("terminal_error") in (None, ""), paper_type
        assert result["current_phase"] == "writing_results", paper_type
        assert result["analysis_reports"]["independent"]["skipped"] is True, paper_type
        assert any(
            f.get("check") == "independent_validation" and f.get("blocking") is False
            for f in result["verification_findings"]
        ), paper_type


def test_r12_no_skip_experimental_empty_outputs():
    from core.workflow_nodes import independent_validation_node

    state = initialize_state()
    state.update({
        "paper_type": "experimental",
        "plan": {"experiments": [{"name": "toy", "evaluation_metrics": ["accuracy"]}]},
        "engineer_outputs": {},
    })
    result = independent_validation_node(state)
    assert result["current_phase"] == "complete"
    assert result["terminal_error"]
    assert result["evidence_gate"]["reason_code"] == "missing_code_artifact"


def test_r12_no_skip_comparative_or_causal_empty_outputs():
    from core.workflow_nodes import independent_validation_node

    for paper_type in ("comparative", "cause_effect"):
        state = initialize_state()
        state.update({
            "paper_type": paper_type,
            "plan": {"experiments": [{"name": "toy", "evaluation_metrics": ["accuracy"]}]},
            "engineer_outputs": {},
        })
        result = independent_validation_node(state)
        assert result["current_phase"] == "complete", paper_type
        assert result["terminal_error"], paper_type


# ---------------------------------------------------------------------------
# M6.3 — ledger backup/rotation (proposal §11-M6 step 3 + §12 migration)
# ---------------------------------------------------------------------------

def test_ledger_backup_creates_bak_and_leaves_live(tmp_path):
    from core.ledger_backup import backup_ledger

    live = tmp_path / "research_ledger.sqlite"
    live.write_bytes(b"ledger-bytes")
    dest = backup_ledger(db_path=live, backup_dir=tmp_path, keep=5)
    assert dest is not None and dest.exists()
    assert ".bak." in dest.name
    assert live.read_bytes() == b"ledger-bytes"


def test_ledger_backup_rotation_keeps_newest(tmp_path):
    from core.ledger_backup import backup_ledger, rotate_backups

    live = tmp_path / "research_ledger.sqlite"
    live.write_bytes(b"x")
    for _ in range(7):
        import time

        backup_ledger(db_path=live, backup_dir=tmp_path, keep=5)
        time.sleep(1.01)
    survivors = rotate_backups(live, tmp_path, keep=5)
    assert len(survivors) == 5
    assert live.exists()


def test_ledger_backup_missing_ledger_returns_none(tmp_path):
    from core.ledger_backup import backup_ledger

    assert backup_ledger(db_path=tmp_path / "absent.sqlite", backup_dir=tmp_path) is None


def test_ledger_backup_keep_zero_never_wipes_history(tmp_path):
    from core.ledger_backup import backup_ledger, rotate_backups

    live = tmp_path / "research_ledger.sqlite"
    live.write_bytes(b"x")
    first = backup_ledger(db_path=live, backup_dir=tmp_path, keep=5)
    assert first is not None
    survivors = rotate_backups(live, tmp_path, keep=0)
    assert len(survivors) >= 1  # clamped to keep>=1


def test_ledger_backup_rotation_ignores_sidecars(tmp_path):
    from core.ledger_backup import backup_ledger, rotate_backups, _backup_siblings

    live = tmp_path / "research_ledger.sqlite"
    live.write_bytes(b"x")
    dest = backup_ledger(db_path=live, backup_dir=tmp_path, keep=5)
    assert dest is not None
    (dest.parent / f"{dest.name}-wal").write_bytes(b"sidecar")
    siblings = _backup_siblings(live, tmp_path)
    assert all(not s.name.endswith(("-wal", "-shm")) for s in siblings)
    assert dest in siblings


def test_ledger_backup_same_second_does_not_collide(tmp_path):
    from core.ledger_backup import backup_ledger

    live = tmp_path / "research_ledger.sqlite"
    live.write_bytes(b"x")
    first = backup_ledger(db_path=live, backup_dir=tmp_path, keep=5)
    second = backup_ledger(db_path=live, backup_dir=tmp_path, keep=5)
    assert first is not None and second is not None
    assert first != second and first.exists() and second.exists()


# ---------------------------------------------------------------------------
# M6.6 — §9 parity spot-check + D13/U10 pins
# ---------------------------------------------------------------------------

def test_m6_config_parity_locked_defaults():
    # Owner-generous caps (2026-10-04 directive: time/cost/tokens are a backstop,
    # not a target) + TYPED_GRAPH_ENABLED=true (full upgrade live).
    assert config.run_max_llm_calls == 2000
    assert config.run_max_tokens == 8000000
    assert config.run_max_usd == 500.0
    # 119ce81c wall-trip fix: discovery autoscale (27300s) exceeded the old
    # 6h run wall, killing the run right after topic discovery. Wall raised
    # to 12h; discovery capped to wall-reserve in code.
    assert config.run_wall_clock_sec == 43200
    assert config.engineer_max_attempts == 4
    assert config.engineering_wall_clock_sec == 7200
    assert config.typed_graph_enabled is True
    assert config.parallel_phases_enabled is False
    assert config.paper_type_default == "experimental"
    assert config.allow_synthetic_respondents is False
    assert config.embedding_dimension == 3072
    assert config.llm_timeout_sec == 1800
    assert config.sandbox_timeout_sec == 600


def test_d13_preamble_pin():
    from core.sandbox import SANDBOX_PREAMBLE_TEMPLATE, build_preamble, run_known_answer_check
    import inspect

    assert "numpy" in SANDBOX_PREAMBLE_TEMPLATE
    assert "numpy" in build_preamble(seed=0)
    assert "seed" in inspect.signature(run_known_answer_check).parameters


def test_u10_frontend_pin():
    import pathlib

    src = pathlib.Path("web/static/admin/admin.js").read_text(encoding="utf-8")
    assert "const phases" in src  # single phase source
    assert "_copyMap" in src or "new Map()" in src  # Map-based copy
    assert "?." in src and "Number(" in src  # null guards


def test_custom_recursion_limit_threads_into_wrapped_graph():
    from core.workflow import _after_phase, create_typed_graph

    def _a(state):
        return {"current_phase": "a", "phase_steps": dict(state.get("phase_steps") or {})}

    def _b(state):
        return {"current_phase": "b", "phase_steps": dict(state.get("phase_steps") or {})}

    graph = create_typed_graph({"a": _a, "b": _b}, ["a", "b"], recursion_limit=2)
    final = graph.compile().invoke(
        {"current_phase": "start", "phase_steps": {"a": 1, "b": 1}},
        {"recursion_limit": 30},
    )
    assert str(final.get("terminal_error", "")).startswith("budget_exhausted")
    assert _after_phase(["a", "b"], {"terminal_error": "budget_exhausted"}, recursion_limit=2) == "end"


def test_companion_bundle_has_models_decision_log_cost_report(tmp_path, monkeypatch):
    import json
    from agents.editor import EditorAgent

    monkeypatch.setattr(EditorAgent, "companion_dir", str(tmp_path / "companion"), raising=False)
    ed = EditorAgent.__new__(EditorAgent)
    ed.companion_dir = str(tmp_path / "companion")
    outputs = {"toy": {"success": True, "decision_log": [{"attempt": 0, "decision": "REFINE", "reason": "tune"}]}}
    paths = ed._write_companion_repo(
        {"title": "T"}, {}, outputs, run_stats={"llm_calls": 7, "cost_estimate_usd": 0.02},
    )
    manifest = json.loads((tmp_path / "companion" / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    assert manifest["models"]["embedding_dimension"] == 3072
    assert manifest["models"]["llm_provider"]
    decisions = json.loads((tmp_path / "companion" / "decision_log.json").read_text(encoding="utf-8"))
    assert decisions[0]["experiment"] == "toy"
    cost = json.loads((tmp_path / "companion" / "cost_report.json").read_text(encoding="utf-8"))
    assert cost["llm_calls"] == 7 and cost["complete"] is True
    assert set(("decision_log", "cost_report")) <= set(paths)


def test_companion_cost_report_without_stats_is_labelled_partial(tmp_path, monkeypatch):
    import json
    from agents.editor import EditorAgent

    ed = EditorAgent.__new__(EditorAgent)
    ed.companion_dir = str(tmp_path / "companion")
    ed._write_companion_repo({"title": "T"}, {}, {})
    cost = json.loads((tmp_path / "companion" / "cost_report.json").read_text(encoding="utf-8"))
    assert cost["complete"] is False and cost["llm_calls"] == 0
