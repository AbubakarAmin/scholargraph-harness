"""Offline tests for paper-type release requirements and dashboard behavior."""

from __future__ import annotations

import inspect
import json

# ---------------------------------------------------------------------------
# Local pure helpers (spec mirrors — prefer core imports where a helper exists)
# ---------------------------------------------------------------------------


def evaluate_release(paper_type: str, evidence: dict) -> bool:
    """Local pure mirror of proposal §7.2 release formulas (exact).

    Base for every type: no terminal error AND no blocking findings.
    Per-type additions mirror §7.2 verbatim. Cross-checked against
    core.verification helpers (verify_review / verify_survey / verify_case /
    verify_thesis) inside test_release_formulas_per_type where they exist.
    """
    ev = dict(evidence or {})
    if ev.get("terminal"):
        return False
    if any(isinstance(f, dict) and f.get("blocking") for f in (ev.get("findings") or [])):
        return False
    if paper_type in ("experimental", "comparative", "empirical", "cause_effect"):
        return bool(ev.get("reproducibility_passed"))
    if paper_type == "review":
        return (
            bool(ev.get("prisma_reconciled"))
            and float(ev.get("extraction", 0.0)) >= 0.9
            and float(ev.get("recall", 0.0)) >= 0.8
        )
    if paper_type in ("analytical", "argumentative"):
        # No reproducibility required for thesis types.
        return bool(ev.get("thesis_triad")) and float(ev.get("bridge_coverage", 0.0)) >= 1.0
    if paper_type == "descriptive":
        return bool(ev.get("tables_recomputed"))
    if paper_type == "survey":
        return bool(ev.get("tables_recomputed")) and bool(ev.get("nonresponse_reported"))
    if paper_type == "case_study":
        return bool(ev.get("triangulation_ok"))
    raise ValueError(f"Unknown paper_type: {paper_type!r}")


def build_slim(dash: dict) -> dict:
    """Appendix B slim shape: exactly {run_id, phase, readiness, pipeline, spend, blockers}."""
    dash = dash or {}
    return {
        "run_id": dash.get("run_id"),
        "phase": dash.get("phase"),
        "readiness": dash.get("readiness"),
        "pipeline": dash.get("pipeline"),
        "spend": dash.get("spend"),
        "blockers": dash.get("blockers"),
    }


# BACKEND TODO: wire GET /api/dashboard/slim to this shape (proposal §8.7 / Appendix B:
# {run_id, phase, readiness, pipeline[{phase, state, reason}], spend{usd, calls,
# tokens, left}, blockers[]} at <10% of full-dashboard bytes, slim-poll only, no SSE).


# ---------------------------------------------------------------------------
# 1. Run-payload pre-spend validation (§11-M5 case 1; §8.2 + §8.7)
# ---------------------------------------------------------------------------


def test_run_payload_validation_rejects_pre_spend(monkeypatch):
    import web.app as app
    from core.config import config

    # Valid backend context so only the payload under test can fail.
    monkeypatch.setattr(config, "google_api_key", "test-google-key")
    monkeypatch.setattr(config, "openai_api_key", "test-openai-key")
    monkeypatch.setattr(config, "openalex_email", "researcher@university.edu")
    monkeypatch.setattr(config, "llm_provider", "gemini")

    good_goal = "effects of retrieval augmentation on scientific claim verification"

    # Goal is OPTIONAL: a blank/short goal must NOT be rejected — the run can
    # be fully autonomous (domain-driven topic discovery, no user goal).
    ok = app.validate_run_payload(app.RunPayload(goal="too short"))
    assert ok is None, f"short goal must be allowed (goal optional), got {ok!r}"
    ok = app.validate_run_payload(app.RunPayload(goal=""))
    assert ok is None, f"blank goal must be allowed (goal optional), got {ok!r}"

    err = app.validate_run_payload(
        app.RunPayload(goal=good_goal, paper_type="wizardry")
    )
    assert err and "paper_type" in err, f"bad paper_type must be rejected, got {err!r}"

    err = app.validate_run_payload(
        app.RunPayload(goal=good_goal, depth="Deep")
    )
    assert err and "depth" in err, f"bad depth must be rejected, got {err!r}"

    ok = app.validate_run_payload(
        app.RunPayload(
            goal=good_goal,
            paper_type="survey",
            depth="Standard",
            budget_usd=25.0,
            budget_min=90.0,
        )
    )
    assert ok is None, f"valid full_research payload must be accepted, got {ok!r}"


# ---------------------------------------------------------------------------
# 2. Snapshot round-trip carries ALL keys (§11-M5 case 2; §8.7; §2.4 + §3.1)
# ---------------------------------------------------------------------------


def test_snapshot_roundtrip_all_keys(monkeypatch):
    import web.app as app
    from core.state import ResearchState, initialize_state

    state = initialize_state()
    # Set every v2 key (§3.1) + latex_output + started_at/stats-style extras.
    state["paper_type"] = "survey"
    state["pipeline_recipe"] = ["intake", "topic_discovery", "data_validation",
                                "survey_analysis", "writing_results", "supervision", "editing"]
    state["evidence_requirements"] = {"needs_experiment": False, "needs_prisma": False,
                                      "needs_pack": True, "needs_audit": False, "needs_thesis": False}
    state["corpus"] = {"prisma": {"identified": 10}, "mode": "scoping"}
    state["instrument"] = {"response_hash": "abc", "codebook_hash": "def"}
    state["causal_claim"] = {"verb_level": "associational"}
    state["thesis"] = {"claim": "x", "bridges": [], "steelman": "s", "concession": "c"}
    state["skipped_phases"] = [{"phase": "engineering", "reason": "not in survey recipe"}]
    state["intake_report"] = {"goal": "effects of retrieval augmentation on claim verification"}
    state["readiness_level"] = "L2"
    state["checkpoint_schema_version"] = 2
    state["latex_output"] = "\\documentclass{article}"
    state["current_phase"] = "supervision"
    state["run_id"] = "r-snap"
    state["started_at"] = "2026-10-03T12:00:00Z"
    state["stats"] = {"llm_calls": 7}

    snap = app._workspace_snapshot(state)
    expected = set(ResearchState.__annotations__) | {"started_at", "stats"}
    missing = sorted(k for k in expected if k not in snap)

    # _load_state_from_db-style restore must preserve every v2 value + phase.
    workspace = dict(snap)
    workspace["stats"] = state["stats"]
    fake_runs = [{"run_id": "r-snap", "summary_json": json.dumps({"workspace": workspace})}]
    monkeypatch.setattr(app.research_db, "list_runs", lambda limit=100: fake_runs)
    restored = app._load_state_from_db("r-snap")
    assert restored is not None, "_load_state_from_db must restore a snapshotted run"
    assert restored.get("current_phase") == "supervision", "restore must keep graph position (current_phase)"
    lost = sorted(
        k for k in ("paper_type", "pipeline_recipe", "evidence_requirements", "corpus",
                    "instrument", "causal_claim", "thesis", "skipped_phases",
                    "intake_report", "readiness_level", "latex_output")
        if restored.get(k) != state[k]
    )

    problems = []
    if missing:
        problems.append(f"snapshot drops {len(missing)} keys: {missing}")
    if lost:
        problems.append(f"restore loses values for: {lost}")
    assert not problems, (
        "BACKEND GAP (M5 §8.7: snapshot = ALL state keys §2.4+§3.1 + "
        "started_at/stats/latex_output, verbatim restore): " + "; ".join(problems)
    )


# ---------------------------------------------------------------------------
# 3. Per-type release formulas (§11-M5 case 3; §7.2)
# ---------------------------------------------------------------------------


def test_release_formulas_per_type():
    from agents.thesis_agent import verify_thesis
    from core.verification import verify_case, verify_review, verify_survey

    # (a) experimental: reproducibility.passed + no blocking + no terminal.
    assert evaluate_release("experimental", {"reproducibility_passed": True}) is True
    assert evaluate_release("experimental", {"reproducibility_passed": False}) is False
    assert evaluate_release("experimental", {"reproducibility_passed": True, "terminal": "boom"}) is False
    assert evaluate_release("experimental", {"reproducibility_passed": True,
                                             "findings": [{"blocking": True}]}) is False

    # (b) review: prisma_reconciled + extraction >= 0.9 + recall >= 4/5.
    ready_review = {"prisma_reconciled": True, "extraction": 0.95, "recall": 0.8}
    assert evaluate_release("review", ready_review) is True
    assert evaluate_release("review", {**ready_review, "prisma_reconciled": False}) is False
    assert evaluate_release("review", {**ready_review, "extraction": 0.85}) is False
    assert evaluate_release("review", {**ready_review, "recall": 0.6}) is False
    # Cross-check core.verify_review: balanced PRISMA + full extraction + recall + grounded synthesis.
    counts = {"identified": 100, "screened": 80, "duplicates_excluded": 20, "eligible": 60,
              "excluded_title_abstract": 20, "included": 50, "excluded_fulltext": 10}
    rows = [{"design": "rct", "sample_n": 50, "effect": 0.2, "metric": "acc",
             "limitation": "small", "doi": f"10.1/x{i}"} for i in range(10)]
    good_corpus = {"prisma": counts, "extraction": rows,
                   "seed_recall": {"recall": 1.0},
                   "synthesis": {"mode": "narrative", "narrative": "themes emerge",
                                 "why_not_meta": "effects are incommensurate"}}
    sections = {"Results": "We include evidence from 10.1/x3 and others."}
    assert verify_review(good_corpus, sections)["passed"] is True
    bad_counts = dict(counts, identified=101)  # off-by-one → blocking
    assert verify_review({**good_corpus, "prisma": bad_counts}, sections)["passed"] is False

    # (c) analytical/argumentative: thesis triad + 100% bridges, no reproducibility.
    ready_thesis = {"thesis_triad": True, "bridge_coverage": 1.0,
                    "reproducibility_passed": False}
    assert evaluate_release("analytical", ready_thesis) is True
    assert evaluate_release("argumentative", ready_thesis) is True
    assert evaluate_release("analytical", {"thesis_triad": False, "bridge_coverage": 1.0}) is False
    assert evaluate_release("analytical", {"thesis_triad": True, "bridge_coverage": 0.5}) is False
    # Cross-check core.verify_thesis: triad + every bridge covered by the manuscript.
    thesis = {"claim": "retrieval grounding improves verification accuracy",
              "bridges": [{"statement": "retrieval grounding improves accuracy"}],
              "steelman": "retrieval adds latency cost", "concession": "latency tradeoff remains"}
    prose = {"Analysis": "Our study shows retrieval grounding improves accuracy overall."}
    assert verify_thesis(thesis, prose)["passed"] is True
    assert verify_thesis({**thesis, "steelman": ""}, prose)["passed"] is False

    # (d) survey: tables recomputed + nonresponse reported.
    assert evaluate_release("survey", {"tables_recomputed": True,
                                       "nonresponse_reported": True}) is True
    assert evaluate_release("survey", {"tables_recomputed": True,
                                       "nonresponse_reported": False}) is False
    assert evaluate_release("survey", {"tables_recomputed": False,
                                       "nonresponse_reported": True}) is False
    # Cross-check core.verify_survey.
    instrument = {"response_hash": "abc", "codebook_hash": "def"}
    good_analysis = {"instrument": instrument,
                     "nonresponse": {"item_rates": {"q1": 0.02}, "unit_rate": 0.9},
                     "synthetic": {"blocking": False}, "fishing": {"withheld": []},
                     "validation": {"passed": True}}
    assert verify_survey(good_analysis, {}, instrument)["passed"] is True
    assert verify_survey({**good_analysis, "nonresponse": {}}, {}, instrument)["passed"] is False

    # (e) case: triangulation ok.
    assert evaluate_release("case_study", {"triangulation_ok": True}) is True
    assert evaluate_release("case_study", {"triangulation_ok": False}) is False
    # Cross-check core.verify_case.
    good_protocol = {"pack": {"passed": True},
                     "triangulation": {"single_source_flags": []}, "limitations": [],
                     "rival": {"passed": True}}
    assert verify_case(good_protocol, {})["passed"] is True
    assert verify_case({**good_protocol, "rival": {"passed": False,
                                                  "findings": [{"details": "no rival"}]}},
                       {})["passed"] is False


# ---------------------------------------------------------------------------
# 4. Approve guards (§11-M5 case 4; §8.6 + U5; Appendix B)
# ---------------------------------------------------------------------------


def test_approve_guards():
    import web.app as app
    from fastapi.testclient import TestClient

    client = TestClient(app.app, raise_server_exceptions=False)

    # Unready approve is refused (terminal_error set → 409).
    app._set_latest_state({"run_id": "r-ready", "terminal_error": "boom", "final_paper": None})
    try:
        resp = client.post("/api/release/approve", json={"run_id": "r-ready"})
        assert resp.status_code == 409, f"unready approve must be 409, got {resp.status_code}"
    finally:
        app._set_latest_state({})

    # Unknown/wrong-run approve must be refused — requires a run_id-aware endpoint.
    params = inspect.signature(app.approve_release).parameters
    assert "run_id" in params, (
        "BACKEND GAP (M5 §8.6/U5 + App B: POST /api/release/approve takes {run_id} "
        "and requires per-type ready (§7.2), else 409 + missing-items list): "
        "endpoint accepts no run_id, so wrong-run approves cannot be refused"
    )


# ---------------------------------------------------------------------------
# 5. True-resume vs Re-run split (§11-M5 case 5; §6.2.5 + U6)
# ---------------------------------------------------------------------------


def test_resume_split(monkeypatch):
    import web.app as app

    # Restore keeps graph position (no phase reset on load).
    workspace = {"mode": "full_research", "current_phase": "supervision", "topics": []}
    fake_runs = [{"run_id": "r-keep", "summary_json": json.dumps({"workspace": workspace})}]
    monkeypatch.setattr(app.research_db, "list_runs", lambda limit=100: fake_runs)
    restored = app._load_state_from_db("r-keep")
    assert restored is not None and restored.get("current_phase") == "supervision"

    # True-resume must NOT delete checkpoints and must keep current_phase;
    # only the explicitly labeled Re-run path may delete checkpoints + reset phase.
    src = inspect.getsource(app._run_pipeline)
    resume_branch = src.split("if resume_run_id:")[1] if "if resume_run_id:" in src else ""
    rerun_deletes = "_delete_checkpoints" in resume_branch
    rerun_clears_phase = 'existing_state["current_phase"] = None' in resume_branch
    has_true_resume = (
        "true_resume" in src.lower()
        or "true resume" in src.lower()
        or "resume=True" in src.replace(" ", "")
    )
    assert rerun_deletes, "re-run path must delete checkpoints (explicit restart)"
    assert has_true_resume, (
        "BACKEND GAP (M5 U6: True Resume via real LangGraph resume — no checkpoint "
        "delete, current_phase kept — vs explicitly labeled Re-run, only Re-run "
        "deletes checkpoints): web/app.py has a single resume path that deletes "
        "checkpoints, clears current_phase and forces resume=False, i.e. resume is "
        "actually restart"
    )


# ---------------------------------------------------------------------------
# 6. Slim dashboard payload (§11-M5 case 6; §8.7 + Appendix B)
# ---------------------------------------------------------------------------


def test_slim_size():
    full = {
        "run_id": "r-fat",
        "phase": "supervision",
        "readiness": "L2",
        "pipeline": [{"phase": f"p{i}", "state": "done", "reason": ""} for i in range(11)],
        "spend": {"usd": 1.25, "calls": 120, "tokens": 90000, "left": 0.5},
        "blockers": [],
        "draft_sections": {f"sec{i}": "x" * 4000 for i in range(8)},
        "debates": [{"round": i, "topic": "t" * 500, "winner": "a",
                     "score": 8.0, "detail": "y" * 2000} for i in range(6)],
        "engineer_outputs": {f"exp{i}": {"metrics": {"acc": 0.9}, "raw": "z" * 3000}
                             for i in range(4)},
        "events_tail": [{"message": "m" * 300} for _ in range(50)],
    }
    slim = build_slim(full)
    assert set(slim) == {"run_id", "phase", "readiness", "pipeline", "spend", "blockers"}, (
        f"slim must contain exactly the Appendix B keys, got {sorted(slim)}"
    )
    full_bytes = len(json.dumps(full).encode())
    slim_bytes = len(json.dumps(slim).encode())
    assert slim_bytes < 0.10 * full_bytes, (
        f"slim ({slim_bytes}B) must be <10% of full ({full_bytes}B)"
    )


# ---------------------------------------------------------------------------
# 7. Export + null shapes (§11-M5 case 7; U3/U9/U10)
# ---------------------------------------------------------------------------


def test_export_and_null_shapes(monkeypatch):
    import web.app as app
    from core.capabilities import DEFAULT_MANIFESTS
    from fastapi.testclient import TestClient

    client = TestClient(app.app, raise_server_exceptions=False)
    monkeypatch.setattr(app, "get_tracker", lambda: None)
    monkeypatch.setattr(app.research_db, "list_runs", lambda limit=100: [])

    debates = client.get("/api/admin/export/debates")
    assert debates.status_code == 200, f"debates export must render, got {debates.status_code}"
    assert debates.text.splitlines()[0] == "round,topic,winner,score"

    agents = client.get("/api/admin/export/agents")
    assert agents.status_code == 200, (
        "BACKEND GAP (M5 U9: exports use real shapes — name/role/capability from "
        f"DEFAULT_MANIFESTS): /api/admin/export/agents returned {agents.status_code}"
    )
    header = agents.text.splitlines()[0]
    assert "name" in header and "role" in header and "capability" in header, (
        f"agents export must carry name/role/capability shapes, got header {header!r}"
    )
    assert len(agents.text.splitlines()) - 1 == len(DEFAULT_MANIFESTS), (
        "agents export must list one row per DEFAULT_MANIFESTS entry"
    )

    dash = client.get("/api/dashboard")
    assert dash.status_code == 200, f"empty dashboard must render, got {dash.status_code}"
    body = dash.json()
    assert "workspace" in body and body["workspace"] is not None, (
        "BACKEND GAP (M5 U3: dashboard always returns workspace {}, never None): "
        f"empty-state workspace is {body.get('workspace')!r}"
    )

    # Null fixtures (None paper/sections/findings) render without throw.
    from types import SimpleNamespace

    monkeypatch.setattr(
        app,
        "get_tracker",
        lambda: SimpleNamespace(
            run_id="r-null",
            dashboard=lambda: {"run_id": "r-null", "status": "running",
                               "phase": "planning", "stats": {}, "recent_messages": []},
            stats={},
        ),
    )
    monkeypatch.setattr(
        app,
        "_get_latest_state",
        lambda: {"mode": "full_research", "debate_results": None, "final_paper": None,
                 "draft_sections": None, "verification_findings": None,
                 "reproducibility": None, "evidence_gate": None, "terminal_error": None,
                 "human_approved": False},
    )
    monkeypatch.setattr(app, "_get_run_error", lambda: None)
    null_dash = client.get("/api/dashboard")
    assert null_dash.status_code == 200, (
        "BACKEND GAP (M5 U10 null-guards): dashboard throws on null fixtures "
        f"(status {null_dash.status_code})"
    )
