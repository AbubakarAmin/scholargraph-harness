"""
ScholarGraph Control Deck — FastAPI backend.
God's-eye view of runs, keys, events, scratchpad, cross-run memory.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import config, apply_runtime_keys, validate_config, sync_env_file
from core.llm import reset_llm_client, get_llm_client, call_llm
from core.run_log import (
    read_events,
    read_scratchpad,
    get_tracker,
    start_run,
    CrossRunMemory,
    emit_event,
)
from core.memory import memory
from core.research_db import research_db
from core.capabilities import DEFAULT_MANIFESTS
from core.datasets import DATASET_CATALOG
from core.release import evaluate_release

logger = logging.getLogger(__name__)

from fastapi.responses import JSONResponse
import math


class SafeJSONResponse(JSONResponse):
    """Last line of defense: scrub NaN/±Inf before serialization.

    Starlette's JSONResponse renders with ``allow_nan=False``, so a single
    ``float("nan")`` anywhere in a payload raises ``ValueError: Out of range
    float values are not JSON compliant`` and 500s the endpoint. Degenerate
    statistics (empty group, zero-variance t-test) produce exactly that, so
    every response body is sanitized here regardless of which endpoint built
    it. Non-finite floats become ``null``.
    """

    @staticmethod
    def _scrub(value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {k: SafeJSONResponse._scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [SafeJSONResponse._scrub(v) for v in value]
        return value

    def render(self, content) -> bytes:
        return json.dumps(
            self._scrub(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


app = FastAPI(title="ScholarGraph Control Deck", version="2.0")
app.response_class = SafeJSONResponse
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Normalize HTTP errors to the flat {"detail", "type"} shape (U8).

    Some ``raise HTTPException`` sites predate the M5 contract and carry a
    bare-string ``detail`` (no ``type``) or a nested ``{"detail","type"}``
    dict. Both are flattened here so every error body matches Appendix B.
    """
    detail = exc.detail
    if isinstance(detail, dict) and "detail" in detail:
        content = {"detail": str(detail.get("detail")), "type": str(detail.get("type") or "http_error")}
        for key, value in detail.items():
            if key not in content:
                content[key] = value
    elif isinstance(detail, dict):
        content = {"detail": str(detail.get("detail", detail)), "type": str(detail.get("type", "http_error"))}
    else:
        content = {"detail": str(detail), "type": "http_error"}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log every unhandled exception with full traceback."""
    logger.exception("Unhandled exception in %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )

STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)

_run_thread: Optional[threading.Thread] = None
_run_error: Optional[str] = None
_run_lock = threading.Lock()
_latest_state: Dict[str, Any] = {}


class KeysPayload(BaseModel):
    keys: Dict[str, Any]


class RunPayload(BaseModel):
    domain: Optional[str] = None
    provider: Optional[str] = None
    # M0 §8.7 deltas (one-button flow contract; enforced synchronously).
    goal: Optional[str] = None
    paper_type: Optional[str] = "auto"
    pack_id: Optional[str] = None
    depth: Optional[str] = "Standard"
    budget_usd: Optional[float] = None
    budget_min: Optional[float] = None
    hunt_mode: Optional[str] = "try_and_report"


def _validation_error(message: str, err_type: str = "validation_error"):
    from fastapi.responses import JSONResponse as _JR

    return _JR(status_code=400, content={"detail": message, "type": err_type})


def validate_run_payload(payload: RunPayload) -> Optional[str]:
    """Synchronous pre-spend validation (§8.7). Returns error message or None."""
    from core.phase_registry import ALL_PAPER_TYPES

    # Goal is OPTIONAL — a run can be fully autonomous (domain-driven topic
    # discovery with no user goal). When provided it is stored verbatim.
    goal = (payload.goal or "")
    paper_type = (payload.paper_type or "auto")
    if paper_type != "auto" and paper_type not in ALL_PAPER_TYPES:
        return f"paper_type must be auto or one of {sorted(ALL_PAPER_TYPES)} (got {paper_type!r})"
    hunt_mode = (payload.hunt_mode or "try_and_report")
    if hunt_mode not in ("gap", "try_and_report", "replication"):
        return f"hunt_mode must be gap|try_and_report|replication (got {hunt_mode!r})"
    if payload.depth and payload.depth not in ("Quick", "Standard", "Thorough"):
        return f"depth must be Quick|Standard|Thorough (got {payload.depth!r})"
    # Provider key present (no spend before this passes).
    provider = (payload.provider or config.llm_provider or "").lower()
    if provider == "gemini" and not config.google_api_key:
        return "GOOGLE_API_KEY is required when LLM_PROVIDER=gemini"
    if provider in ("openai", "openai_compatible") and not config.openai_api_key:
        return "OPENAI_API_KEY is required for openai providers"
    email = (config.openalex_email or "").strip().lower()
    if not email or "@" not in email or email in {"you@example.com", "researcher@example.com", "test@example.com"}:
        return "OPENALEX_EMAIL must be a real (non-placeholder) email"
    return None


class ResetPayload(BaseModel):
    confirmation: str


class ApprovePayload(BaseModel):
    run_id: Optional[str] = None


class RerunPayload(BaseModel):
    note: Optional[str] = None


# Depth → run caps (§5.3). Tokens cap stays global (no per-depth token in
# the spec); an explicit budget_usd/budget_min (>0) overrides the depth default.
DEPTH_RUN_CAPS = {
    "Quick": {"calls": 750, "usd": 50.0, "minutes": 225},
    "Standard": {"calls": 2000, "usd": 125.0, "minutes": 450},
    "Thorough": {"calls": 4000, "usd": 300.0, "minutes": 900},
}


def _keys_path() -> Path:
    # Same resolution as core.config so UI and CLI share one keys file
    from core.config import _resolve_keys_path

    return _resolve_keys_path()


def load_keys() -> Dict[str, Any]:
    path = _keys_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load keys from %s: %s", path, e)
            return {}
    return {}


def _is_masked_secret(value: Any) -> bool:
    s = str(value or "")
    return ("…" in s) or s.startswith("••") or ("..." in s and len(s) <= 16)


def save_keys(keys: Dict[str, Any]) -> None:
    path = _keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Mask nothing on disk — user asked to store keys; file is local
    path.write_text(json.dumps(keys, indent=2), encoding="utf-8")
    apply_runtime_keys(keys)
    sync_env_file(keys)  # keep .env in sync with UI changes
    reset_llm_client()


@app.on_event("startup")
def _startup():
    keys = load_keys()
    if keys:
        apply_runtime_keys(keys)
        reset_llm_client()


@app.get("/")
def index():
    """Chat-first landing — Control Deck remains at /dashboard."""
    chat_path = STATIC / "chat" / "index.html"
    if chat_path.exists():
        return FileResponse(chat_path)
    # Fallback to admin if chat UI is missing.
    index_path = STATIC / "admin" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>ScholarGraph</h1><p>static/chat/index.html missing</p>")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = STATIC / "img" / "favicon.ico"
    if path.exists():
        return FileResponse(path, media_type="image/x-icon")
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/apple-touch-icon.png", include_in_schema=False)
def apple_touch_icon():
    path = STATIC / "img" / "apple-touch-icon.png"
    if path.exists():
        return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="apple touch icon not found")


@app.get("/dashboard")
def dashboard_page():
    """Full Control Deck (shared harness state with chat)."""
    index_path = STATIC / "admin" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>ScholarGraph</h1><p>static/admin/index.html missing</p>")


@app.get("/admin")
def admin_alias():
    """Legacy alias for the Control Deck."""
    return dashboard_page()


@app.get("/api/health")
def health():
    try:
        from core.container_sandbox import sandbox_status as _sandbox_status

        sandbox = _sandbox_status()
    except Exception as exc:
        sandbox = {"ready": False, "reason": f"sandbox check failed: {exc}"}
    return {
        "ok": True,
        "provider": config.llm_provider,
        "model": config.resolve_model("default"),
        "has_google_key": bool(config.google_api_key),
        "has_openai_key": bool(config.openai_api_key),
        "sandbox_backend": config.sandbox_backend,
        "sandbox": sandbox,
    }


@app.get("/api/config")
def get_config():
    values = {
        "llm_provider": config.llm_provider,
        "gemini_model": config.gemini_model,
        "gemma_thinking_level": config.gemma_thinking_level,
        "openai_model": config.openai_model,
        "openai_base_url": config.openai_base_url,
        "research_domain": config.research_domain,
        "supervisor_threshold": config.supervisor_threshold,
        "debate_pass_threshold": config.debate_pass_threshold,
        "experiment_seeds": config.experiment_seeds,
        "max_iterations": config.max_iterations,
        "debate_min_rounds": config.debate_min_rounds,
        "debate_max_rounds": config.debate_max_rounds,
        "novelty_similarity_reject": config.novelty_similarity_reject,
        "experiment_branch_count": config.experiment_branch_count,
        "openalex_email": config.openalex_email,
        "gemini_embedding_model": config.gemini_embedding_model,
        "openai_embedding_model": config.openai_embedding_model,
        "llm_model_cheap": config.llm_model_cheap,
        "llm_model_strong": config.llm_model_strong,
        "llm_model_judge": config.llm_model_judge,
        "llm_model_fallback": config.llm_model_fallback,
        "llm_fallback_provider": config.llm_fallback_provider,
        "llm_fallback_model": config.llm_fallback_model,
        "llm_timeout_sec": config.llm_timeout_sec,
        "llm_max_output_tokens": config.llm_max_output_tokens,
        "llm_persona_max_tokens": config.llm_persona_max_tokens,
        "llm_price_table_json": config.llm_price_table_json,
        "ensemble_judge_models": config.ensemble_judge_models,
        "semantic_scholar_enabled": config.semantic_scholar_enabled,
        "research_db_path": config.research_db_path,
        "checkpoint_path": config.checkpoint_path,
        "gemini_embedding_model": config.gemini_embedding_model,
        "openai_embedding_model": config.openai_embedding_model,
        "debug_mode": config.debug_mode,
        "log_level": config.log_level,
        "vector_db_path": config.vector_db_path,
        "memory_size": config.memory_size,
        "cross_run_memory_path": config.cross_run_memory_path,
        "run_log_path": config.run_log_path,
        "run_events_path": config.run_events_path,
        "elo_ratings_path": config.elo_ratings_path,
        "output_dir": config.output_dir,
        "draft_versions_dir": config.draft_versions_dir,
        "debate_log_path": config.debate_log_path,
        "feedback_log_path": config.feedback_log_path,
        "raw_results_dir": config.raw_results_dir,
        "companion_repo_dir": config.companion_repo_dir,
        "sandbox_timeout_sec": config.sandbox_timeout_sec,
        "sandbox_max_output_bytes": config.sandbox_max_output_bytes,
        "sandbox_backend": config.sandbox_backend,
        "engineer_max_attempts": config.engineer_max_attempts,
        "engineering_wall_clock_sec": config.engineering_wall_clock_sec,
        "topic_seed_timeout_sec": config.topic_seed_timeout_sec,
        "topic_discovery_timeout_sec": config.topic_discovery_timeout_sec,
        "topic_fallback_enabled": config.topic_fallback_enabled,
        "topic_fallback_max_candidates": config.topic_fallback_max_candidates,
        "seed_sequential": config.seed_sequential,
        "persona_sequential": config.persona_sequential,
        "persona_count": config.persona_count,
        "llm_budget_per_seed": config.llm_budget_per_seed,
        "novelty_max_abstracts": config.novelty_max_abstracts,
        "llm_seed_generation_enabled": config.llm_seed_generation_enabled,
        "typed_graph_enabled": config.typed_graph_enabled,
        "paper_type_default": config.paper_type_default,
        "parallel_phases_enabled": config.parallel_phases_enabled,
        "run_max_llm_calls": config.run_max_llm_calls,
        "run_max_tokens": config.run_max_tokens,
        "run_max_usd": config.run_max_usd,
        "run_wall_clock_sec": config.run_wall_clock_sec,
        "arxiv_enabled": config.arxiv_enabled,
        "openalex_enabled": config.openalex_enabled,
        "openreview_enabled": config.openreview_enabled,
        "openreview_username": config.openreview_username,
        "unpaywall_enabled": config.unpaywall_enabled,
        "pmc_enabled": config.pmc_enabled,
        "doaj_guard_enabled": config.doaj_guard_enabled,
        "huggingface_max_download_mb": config.huggingface_max_download_mb,
        "allow_synthetic_respondents": config.allow_synthetic_respondents,
        "web_host": config.web_host,
        "web_port": config.web_port,
        "keys_store_path": config.keys_store_path,
    }
    return values


@app.get("/api/sandbox")
def sandbox_info():
    """Sandbox readiness: ephemeral docker-run-per-experiment model.

    No persistent container exists by design — `docker ps` is empty
    between runs. Only Docker Desktop + the scholargraph-sandbox image
    need to exist; experiments spawn `docker run --rm` automatically.
    """
    from core.container_sandbox import sandbox_status as _sandbox_status

    return _sandbox_status()


@app.post("/api/sandbox/build")
def sandbox_build():
    """Build (or rebuild) the sandbox image. One-time, takes minutes."""
    from core.container_sandbox import build_sandbox_image

    result = build_sandbox_image()
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("output", "build failed")[-2000:])
    return {"ok": True, "output": result.get("output", "")[-2000:]}


@app.get("/api/keys")
def get_keys():
    """Return stored keys with secrets masked for display.

    Shows EFFECTIVE values: keys.json first, falling back to the process
    environment (.env) so the dashboard reflects what the running engine
    actually uses even if a key was never saved through the UI. Secrets are
    always masked in the response; posting a masked/empty value never
    overwrites the stored secret (see post_keys).
    """
    keys = load_keys()
    effective = dict(keys)
    for k in _SECRET_KEYS:
        if not effective.get(k):
            env_val = os.environ.get(k, "")
            if env_val:
                effective[k] = env_val
    # Non-secret login names are also shown from the environment when the UI
    # never stored them (clear text is safe for usernames, never passwords).
    if not effective.get("OPENREVIEW_USERNAME") and os.environ.get("OPENREVIEW_USERNAME"):
        effective["OPENREVIEW_USERNAME"] = os.environ["OPENREVIEW_USERNAME"]
    masked = {}
    for k, v in effective.items():
        if k in _SECRET_KEYS and v:
            s = str(v)
            masked[k] = (s[:4] + "…" + s[-4:]) if len(s) > 8 else "••••"
        else:
            masked[k] = v
    return {"keys": masked, "raw_present": {k: bool(effective.get(k)) for k in _SECRET_KEYS}}



_SECRET_KEYS = {
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "HUGGINGFACE_TOKEN",
    "OPENREVIEW_PASSWORD",
}


@app.post("/api/keys")
def post_keys(payload: KeysPayload):
    existing = load_keys()
    # Frontend sends lowercase ids (e.g. research_domain); backend mapping,
    # keys store and .env are all UPPER. Normalize so saves actually apply
    # live via apply_runtime_keys + sync_env_file (incl. mid-run).
    existing = {(k or "").strip().upper(): v for k, v in existing.items()}
    incoming = {(k or "").strip().upper(): v for k, v in payload.keys.items()}
    secret_keys = _SECRET_KEYS
    # Don't overwrite secrets with masked placeholders or empty fields
    for k, v in incoming.items():
        if v is None or v == "":
            continue
        if k in secret_keys and _is_masked_secret(v):
            continue
        if _is_masked_secret(v) and k in existing:
            continue
        existing[k] = v
    save_keys(existing)
    return {
        "ok": True,
        "saved": list(payload.keys.keys()),
        "path": str(_keys_path()),
        "env_path": str(ROOT / ".env"),
        "has_openai_key": bool(existing.get("OPENAI_API_KEY")),
        "has_google_key": bool(existing.get("GOOGLE_API_KEY")),
        "provider": existing.get("LLM_PROVIDER") or config.llm_provider,
    }


@app.post("/api/keys/test")
def test_llm():
    try:
        validate_config()
        text = (call_llm("Reply with exactly: OK", temperature=0, tier="cheap", max_tokens=16) or "").strip()
        if not text:
            raise RuntimeError("Provider returned an empty response")
        return {
            "ok": True,
            "response": text[:200],
            "provider": config.llm_provider,
            "model": config.resolve_model("cheap"),
        }
    except Exception as e:
        logger.error("LLM test failed: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dashboard")
def dashboard(run_id: Optional[str] = None):
    tracker = get_tracker()
    target_run_id = run_id or (tracker.run_id if tracker else None)
    if tracker and (run_id is None or run_id == tracker.run_id):
        dash = tracker.dashboard()
    else:
        # Check historical run from db
        runs = research_db.list_runs(limit=100)
        found = next((r for r in runs if r["run_id"] == target_run_id), None) if target_run_id else None
        if found:
            summary = {}
            try:
                summary = json.loads(found.get("summary_json") or "{}")
            except Exception as e:
                logger.warning("Failed to parse run summary for %s: %s", target_run_id, e)
            hist_workspace = summary.get("workspace") or {}
            if not isinstance(hist_workspace, dict):
                hist_workspace = {}
            hist_stats = summary.get("stats") or hist_workspace.get("stats") or {}
            if not isinstance(hist_stats, dict):
                hist_stats = {}
            dash = {
                "run_id": found["run_id"],
                "status": found["status"],
                "phase": found["phase"] or "completed",
                "stats": hist_stats,
                "recent_messages": [],
                "workspace": hist_workspace,
            }
        else:
            dash = {
                "run_id": target_run_id,
                "status": "idle",
                "phase": "idle",
                "stats": {},
                "recent_messages": [],
            }
    dash["events_tail"] = read_events(limit=50, run_id=target_run_id)
    dash["cross_run"] = {
        "rejections": CrossRunMemory().load("rejection", limit=10),
        "pivots": CrossRunMemory().load("pivot", limit=5),
        "runs": CrossRunMemory().load("run_summary", limit=5),
    }
    dash["config"] = {
        "provider": config.llm_provider,
        "model": config.resolve_model("default"),
        "domain": config.research_domain,
        "research_db_path": config.research_db_path,
    }
    dash["tracker_stats"] = tracker.stats if tracker else {}
    state = _get_latest_state() if tracker and (run_id is None or run_id == tracker.run_id) else {}

    if state:
        dash["workspace"] = _workspace_snapshot(state)
    # U3: workspace is always a dict, never None.
    workspace = dash.get("workspace")
    if not isinstance(workspace, dict):
        workspace = {}
        dash["workspace"] = workspace
    # U3/U4: typed-harness keys always present (backfill for old snapshots).
    for _k, _default in (
        ("paper_type", "experimental"),
        ("pipeline_recipe", []),
        ("evidence_requirements", {}),
        ("corpus", None),
        ("instrument", None),
        ("causal_claim", None),
        ("thesis", None),
        ("skipped_phases", []),
        ("intake_report", None),
        ("readiness_level", "L1"),
        ("paper_candidates", []),
        ("selected_artifacts", []),
        ("hunt_mode", "try_and_report"),
        ("routing_provenance", None),
    ):
        if _k not in workspace:
            workspace[_k] = list(_default) if isinstance(_default, list) else (
                dict(_default) if isinstance(_default, dict) else _default
            )
    dash["evidence_trace"] = research_db.claims(dash.get("run_id"))
    workspace = dash.get("workspace")
    if not isinstance(workspace, dict):
        workspace = {}
        dash["workspace"] = workspace
    terminal = bool(workspace.get("terminal_error") or (workspace.get("evidence_gate") or {}).get("terminal"))
    # M5: per-type release rules (§7.2) via pure evaluate_release.
    # Unrouted runs (auto/auto-deferred/blank — intake hasn't bound a type yet)
    # evaluate as experimental (strictest: needs reproducibility). Otherwise an
    # evidence-free fresh state hits the base-gate-only path for unknown types
    # and shows "pending_human_approval" before any science exists.
    paper_type = str(workspace.get("paper_type") or "experimental")
    if paper_type.strip().lower() in ("auto", "auto-deferred", ""):
        paper_type = "experimental"
    term_flag, term_msg, blocking, repro_passed, extra = _release_inputs(workspace)
    result = evaluate_release(
        paper_type,
        terminal=term_flag,
        blocking_findings=blocking,
        reproducibility_passed=repro_passed,
        extra=extra,
    )
    missing = list(result.get("missing") or [])
    human_approved = bool(workspace.get("human_approved", False))
    if term_flag:
        release_status = "blocked"
    elif not result.get("ready"):
        release_status = "incomplete"
    elif not human_approved:
        # L3 gate: internally ready but still awaiting human approval.
        release_status = "pending_human_approval"
    else:
        release_status = "ready"
    release_reason = term_msg or (f"missing: {', '.join(missing)}" if missing else "")
    if release_status == "ready":
        readiness = "L3"
    elif release_status == "pending_human_approval":
        readiness = "L2"
    else:
        readiness = workspace.get("readiness_level") or "L1"
    dash["release"] = {
        "status": release_status,
        "reason": release_reason,
        "missing": missing,
        "readiness": readiness,
        # Explicit UI flags (additive): needs_approval drives the approve
        # button visibility; ready means fully released (L3). Idle/early runs
        # are incomplete (never pending) thanks to the unrouted-type mapping.
        "needs_approval": release_status == "pending_human_approval",
        "ready": release_status == "ready",
    }
    # Top-level consumer contract (U3/U4/U10): readiness/spend/blockers/
    # pipeline are always present on the full snapshot too (not only slim),
    # so clients never read undefined after a full refresh.
    try:
        _slim_view = build_slim(dash)
    except Exception:
        _slim_view = {}
    if _slim_view:
        dash.setdefault("readiness", _slim_view.get("readiness"))
        dash.setdefault("spend", _slim_view.get("spend"))
        dash.setdefault("blockers", _slim_view.get("blockers"))
        dash.setdefault("pipeline", _slim_view.get("pipeline"))
    # Surface the current run error so the UI can display it prominently.
    run_err = _get_run_error()
    if run_err:
        dash["release"]["error"] = run_err
        if not release_reason:
            dash["release"]["reason"] = run_err
    dash["capabilities"] = DEFAULT_MANIFESTS
    return dash


@app.get("/api/dashboard/slim")
def dashboard_slim(run_id: Optional[str] = None):
    """Slim-poll dashboard payload (Appendix B, U1 single transport).

    ``{run_id, phase, readiness, pipeline[{phase,state,reason}],
    spend{usd,calls,tokens,left}, blockers[]}`` — a small fraction of the
    full dashboard bytes. Error bodies are ``{"detail", "type"}`` (U8).
    """
    from fastapi.responses import JSONResponse as _JR

    try:
        dash = dashboard(run_id=run_id)
    except Exception as exc:
        logger.error("Slim dashboard failed: %s", exc, exc_info=True)
        return _JR(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})
    try:
        return build_slim(dash)
    except Exception as exc:
        logger.error("Slim build failed: %s", exc, exc_info=True)
        return _JR(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})


@app.post("/api/release/approve")
def approve_release(payload: ApprovePayload = ApprovePayload(), run_id: Optional[str] = None):
    """Record the explicit human checkpoint before a run is publishable.

    M5 U5: requires ``{"run_id"}`` (body, or ``?run_id=``) matching the
    selected/latest run plus per-type readiness via ``evaluate_release``
    (§7.2). Error bodies are always ``{"detail", "type"}`` (U8); unready
    approvals return 409 with the missing-items list.
    """
    from fastapi.responses import JSONResponse as _JR

    want = ((payload.run_id if payload else None) or run_id or "").strip()
    if not want:
        return _JR(status_code=400, content={"detail": "run_id is required", "type": "validation_error"})
    latest_state = _get_latest_state()
    tracker = get_tracker()
    candidate_ids = {i for i in [latest_state.get("run_id"), tracker.run_id if tracker else None] if i}
    state = None
    if latest_state.get("run_id") == want:
        state = latest_state
    else:
        state = _load_state_from_db(want)
        if state is None:
            if candidate_ids:
                return _JR(
                    status_code=409,
                    content={
                        "detail": f"run_id {want!r} does not match the active run",
                        "type": "wrong_run",
                    },
                )
            return _JR(status_code=404, content={"detail": f"Run {want} not found", "type": "not_found"})
    ws = state
    paper_type = str(ws.get("paper_type") or "experimental")
    # Same unrouted-type rule as dashboard(): an unbound type must not sneak
    # through the base-gate-only path to approval with zero evidence.
    if paper_type.strip().lower() in ("auto", "auto-deferred", ""):
        paper_type = "experimental"
    term_flag, _term_msg, blocking, repro_passed, extra = _release_inputs(ws)
    result = evaluate_release(
        paper_type,
        terminal=term_flag,
        blocking_findings=blocking,
        reproducibility_passed=repro_passed,
        extra=extra,
    )
    if not result.get("ready"):
        return _JR(
            status_code=409,
            content={
                "detail": f"Run {want} is not ready for approval (paper_type={paper_type})",
                "type": "not_ready",
                "missing": list(result.get("missing") or []),
            },
        )
    # Approve ONLY the requested run: never clobber live in-memory state and
    # never stamp the active tracker run when the operator approves a
    # historical run (approving run B must not approve/clobber run A).
    approved = dict(state)
    approved["human_approved"] = True
    if isinstance(approved.get("final_paper"), dict):
        approved["final_paper"] = {**approved["final_paper"], "publishable": True}
    if latest_state.get("run_id") == want:
        _set_latest_state(approved)
    research_db.update_run_summary(want, {"human_approved": True, "publishable": True})
    from core.artifacts import save_results
    save_results(approved)
    return {"ok": True, "publishable": True, "run_id": want}


@app.get("/api/capabilities")
def capabilities():
    """Return the role/tool contract used by the current runtime."""
    return {"capabilities": DEFAULT_MANIFESTS}


@app.get("/api/artifacts")
def artifacts(run_id: Optional[str] = None):
    """Return durable artifact records for the selected run."""
    target_run_id = run_id or (get_tracker().run_id if get_tracker() else None)
    return {"artifacts": research_db.artifacts(target_run_id)}


@app.get("/api/events")
def events(limit: int = 100, run_id: Optional[str] = None):
    return {"events": read_events(limit=limit, run_id=run_id)}


@app.get("/api/scratchpad")
def scratchpad(limit: int = 50, run_id: Optional[str] = None):
    return {"entries": read_scratchpad(limit=limit, run_id=run_id)}


@app.get("/api/memory/feedback")
def feedback(limit: int = 20):
    return {"feedback": memory.get_recent_feedback(limit=limit)}


@app.get("/api/admin/export/{data_type}")
def export_data(data_type: str):
    """Export debates or agents data as CSV.
    No authentication required as per user request.
    """
    from fastapi.responses import JSONResponse as _JR

    # Determine payload based on data_type
    if data_type == "debates":
        dash = dashboard()
        items = dash.get("workspace", {}).get("debates", []) or []
        headers = ["round", "topic", "winner", "score"]
        rows = []
        for d in items:
            if not isinstance(d, dict):
                rows.append(["", "", "", ""])
                continue
            # Canonical debate_result fields first, raw keys as fallback.
            round_v = d.get("round", d.get("round_number", d.get("round_idx", d.get("index", ""))))
            topic_v = d.get("topic", d.get("topic_title", d.get("title", d.get("hypothesis", ""))))
            winner_v = d.get("winner", d.get("winner_title", d.get("selected", d.get("passed", ""))))
            score_v = d.get("score", d.get("mean_score", d.get("final_score", d.get("ensemble_mean", ""))))
            if isinstance(topic_v, dict):
                topic_v = topic_v.get("title", topic_v.get("hypothesis", ""))
            rows.append([round_v, topic_v, winner_v, score_v])
    elif data_type == "agents":
        # Real capability-manifest shapes from DEFAULT_MANIFESTS.
        headers = ["name", "role", "allowed", "forbidden", "capability_artifacts"]
        rows = []
        for name, manifest in (DEFAULT_MANIFESTS or {}).items():
            if not isinstance(manifest, dict):
                rows.append([name, "", "", "", ""])
                continue
            allowed = manifest.get("allowed_capabilities", [])
            forbidden = manifest.get("forbidden_capabilities", [])
            allowed_s = ";".join(allowed) if isinstance(allowed, list) else str(allowed)
            forbidden_s = ";".join(forbidden) if isinstance(forbidden, list) else str(forbidden)
            artifacts = list(manifest.get("input_artifact_types", []) or []) + list(
                manifest.get("output_artifact_types", []) or []
            )
            rows.append([
                name,
                manifest.get("role", manifest.get("agent", "")),
                allowed_s,
                forbidden_s,
                ";".join(artifacts),
            ])
    else:
        return _JR(status_code=400, content={"detail": "Unsupported export type", "type": "validation_error"})

    # Build CSV string (headers always present, never empty-header CSV).
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    csv_bytes = output.getvalue().encode("utf-8")
    from fastapi.responses import StreamingResponse
    return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={data_type}_export.csv"})



@app.post("/api/data/reset/outputs")
def reset_outputs(payload: ResetPayload):
    """Clear generated paper artifacts without touching durable history."""
    global _latest_state, _run_error
    if payload.confirmation != "RESET_OUTPUTS":
        raise HTTPException(status_code=400, detail="Type RESET_OUTPUTS to confirm output reset")
    if _run_thread and _run_thread.is_alive():
        raise HTTPException(status_code=409, detail="Stop the active run before resetting outputs")
    protected = {Path(config.run_events_path).resolve(), Path(config.run_log_path).resolve()}
    candidates = {
        Path(config.output_dir),
        Path(config.draft_versions_dir),
        Path(config.raw_results_dir),
        Path(config.companion_repo_dir),
    }
    removed = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in protected or not resolved.exists():
            continue
        if resolved.is_dir():
            for child in resolved.iterdir():
                if child.resolve() in protected:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed.append(str(child))
        else:
            resolved.unlink()
            removed.append(str(resolved))
    _set_latest_state({})
    _set_run_error(None)
    return {"ok": True, "scope": "outputs", "removed": removed, "history_preserved": True}


@app.post("/api/data/reset/catalog")
def reset_dataset_catalog(payload: ResetPayload):
    """Clear checked-in catalog assets only after a separate hard confirmation."""
    if payload.confirmation != "DELETE_DATASET_CATALOG":
        raise HTTPException(status_code=400, detail="Type DELETE_DATASET_CATALOG to confirm catalog reset")
    catalog_root = (ROOT / "data" / "catalog").resolve()
    removed = []
    if catalog_root.exists():
        for child in catalog_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed.append(str(child))
    DATASET_CATALOG.clear()
    return {"ok": True, "scope": "dataset_catalog", "removed": removed, "history_preserved": True}


@app.post("/api/data/clear")
def clear_all_data():
    """Retained as an explicit refusal so clients cannot wipe history by accident."""
    raise HTTPException(status_code=410, detail="Unscoped data clearing is disabled; use a scoped reset")


@app.delete("/api/runs/{run_id}")
def delete_single_run(run_id: str):
    cleaned = research_db.delete_run_comprehensive(run_id)
    memory.delete_run_vectors(run_id)
    return {"ok": True, "message": f"Run {run_id} deleted.", "cleaned": cleaned}


@app.get("/api/runs/{run_id}/lineage")
def run_claim_lineage(run_id: str):
    return {"run_id": run_id, "lineage": research_db.claim_lineage(run_id)}

# ALL ResearchState keys (§2.4 43 keys + §3.1 11 v2 keys + M6 phase_steps). Snapshots carry every
# key so the dashboard round-trips without loss (M5 U4, §8.7).
_ALL_STATE_KEYS = (
    "iteration", "current_phase", "should_reset", "should_continue",
    "error_count", "topics", "selected_topic", "debate_results",
    "hypothesis_passed", "plan", "data_artifacts", "data_validation",
    "execution_artifacts", "analysis_reports", "verification_findings",
    "draft_sections", "current_section", "engineer_outputs",
    "supervisor_scores", "supervisor_feedback", "meta_feedback", "final_paper",
    "latex_output", "plan_revision_requests", "run_id", "results_redraft_count",
    "narrative_revision_count", "editor_repair_count", "editor_repair_findings",
    "results_verification", "reproducibility", "terminal_error",
    "experiment_contracts", "experiment_outcomes", "technical_failures",
    "evidence_gate", "human_approved", "outcome_calibration",
    "literature_context", "user_query",
    "paper_type", "pipeline_recipe", "evidence_requirements", "corpus",
    "instrument", "causal_claim", "thesis", "skipped_phases",
    "intake_report", "readiness_level", "checkpoint_schema_version",
    "phase_steps",
    "paper_candidates", "selected_artifacts", "hunt_mode",
    "routing_provenance",
    "messages", "claims", "tool_ledger_ids",
    "work_plans", "prompt_version", "tool_calls", "provenance",
)


def _release_inputs(workspace: Dict[str, Any]):
    """Split a workspace/state dict into evaluate_release inputs (§7.2)."""
    ws = workspace or {}
    eg = ws.get("evidence_gate")
    if not isinstance(eg, dict):
        eg = {}
    terminal_error = ws.get("terminal_error")
    terminal_flag = terminal_error or eg.get("terminal")
    if isinstance(terminal_flag, str):
        terminal_msg = terminal_flag.strip()
        terminal_flag = terminal_flag.strip() or False
    elif isinstance(terminal_flag, dict):
        terminal_msg = str(
            terminal_flag.get("message") or terminal_flag.get("reason") or "terminal"
        )
    elif terminal_flag:
        terminal_msg = str(terminal_error or eg.get("message") or "terminal")
    else:
        terminal_flag = False
        terminal_msg = ""
    findings = ws.get("verification_findings") or []
    blocking = [
        f for f in findings
        if isinstance(f, dict) and f.get("blocking")
    ]
    repro = ws.get("reproducibility")
    repro_passed = bool(repro.get("passed")) if isinstance(repro, dict) else bool(repro)
    extra: Dict[str, Any] = {}
    for _k in (
        "corpus", "thesis", "instrument", "causal_claim", "analysis_reports",
        "draft_sections", "sections", "intake_report", "evidence_gate",
        "skipped_phases", "pipeline_recipe", "downgrade", "downgraded_from",
        "downgraded_to", "refusal", "refusals", "banner", "banners",
        "prisma_reconciled", "extraction_completeness", "recall",
        "tables_recomputed", "nonresponse_reported", "triangulation_ok",
        "thesis_triad", "bridge_coverage",
    ):
        if ws.get(_k) is not None:
            extra[_k] = ws.get(_k)
    # Survey/case bundles live under analysis_reports — surface them for the gate.
    reports = ws.get("analysis_reports")
    if isinstance(reports, dict):
        for _k in ("survey", "case_protocol"):
            if reports.get(_k) is not None and _k not in extra:
                extra[_k] = reports[_k]
    return terminal_flag, terminal_msg, blocking, repro_passed, extra


def _workspace_snapshot(state: Dict[str, Any], tracker: Any = None) -> Dict[str, Any]:
    """Keep enough completed state for the admin panel to survive a refresh.

    M5 U4: carries ALL ResearchState keys (§2.4 + §3.1) verbatim, plus
    ``started_at``/``stats``/``latex_output``. Legacy ``debates``/``paper``
    aliases are kept for older frontend/export readers.
    """
    import copy as _copy

    state = state or {}
    snap: Dict[str, Any] = {}
    for key in _ALL_STATE_KEYS:
        if key == "debate_results":
            snap[key] = [
                item.__dict__ if hasattr(item, "__dict__") else item
                for item in (state.get(key) or [])
            ]
        elif key in state:
            try:
                snap[key] = _copy.deepcopy(state[key])
            except Exception:
                snap[key] = state[key]
    # Pack carriers (B8): response_table/codebook/case_pack/corpus_override live
    # as untyped state extras (set by intake/CLI). Carry them verbatim so a
    # resumed survey/case run keeps its pack refs.
    for _pack_key in ("response_table", "codebook", "case_pack",
                      "corpus_override", "pack_id"):
        if state.get(_pack_key) is not None:
            try:
                import copy as _copy2

                snap[_pack_key] = _copy2.deepcopy(state[_pack_key])
            except Exception:
                snap[_pack_key] = state[_pack_key]
    # Legacy aliases (older UI/export code reads these names).
    snap["debates"] = snap.get("debate_results", [])
    snap["paper"] = snap.get("final_paper") or {"sections": snap.get("draft_sections", {})}
    if tracker is not None:
        snap["started_at"] = getattr(tracker, "started_at", None)
        try:
            snap["stats"] = dict(getattr(tracker, "stats", {}) or {})
        except Exception:
            snap["stats"] = {}
    else:
        if "started_at" in state:
            snap["started_at"] = state["started_at"]
        if "stats" in state:
            snap["stats"] = state["stats"]
    return snap


def _load_state_from_db(run_id: str) -> Optional[Dict[str, Any]]:
    """Load historical workspace state from the research database for resume."""
    from core.state import initialize_state, migrate_state_v1_to_v2  # local import: keeps web boot light; mirrors _run_pipeline

    runs = research_db.list_runs(limit=100)
    found = next((r for r in runs if r["run_id"] == run_id), None)
    if not found:
        return None
    try:
        summary = json.loads(found.get("summary_json") or "{}")
    except Exception:
        return None
    workspace = summary.get("workspace", {})
    if not workspace:
        return None
    state = initialize_state()
    state["run_id"] = run_id  # ledger row id is authoritative (old snapshots lack it)
    for key in _ALL_STATE_KEYS:
        if key in workspace:
            state[key] = workspace[key]
    # Legacy aliases from pre-M5 snapshots.
    if "debate_results" not in workspace and workspace.get("debates"):
        state["debate_results"] = workspace["debates"]
    if "final_paper" not in workspace and workspace.get("paper") is not None:
        state["final_paper"] = workspace["paper"]
    # started_at/stats/latex_output restore verbatim (U4).
    if workspace.get("started_at") is not None:
        state["started_at"] = workspace["started_at"]
    if workspace.get("stats") is not None:
        state["stats"] = workspace["stats"]
    # M0 schema v2: restore typed-harness keys when present, backfill when not.
    for _k in ("paper_type", "pipeline_recipe", "evidence_requirements", "corpus",
               "instrument", "causal_claim", "thesis", "skipped_phases",
               "intake_report", "readiness_level", "checkpoint_schema_version"):
        if workspace.get(_k) is not None:
            state[_k] = workspace[_k]
    # Pack carriers (B8): restore verbatim when the snapshot carries them.
    for _k in ("response_table", "codebook", "case_pack",
               "corpus_override", "pack_id"):
        if workspace.get(_k) is not None:
            state[_k] = workspace[_k]
    return migrate_state_v1_to_v2(state)


def build_slim(dash: Dict[str, Any]) -> Dict[str, Any]:
    """Build the slim-poll dashboard payload (Appendix B, U1 transport).

    Pure builder (no I/O) so tests can import it: ``{run_id, phase,
    readiness, pipeline[{phase, state, reason}], spend{usd, calls, tokens,
    left}, blockers[]}``. Deliberately <10% of full dashboard bytes.
    """
    dash = dash or {}
    workspace = dash.get("workspace")
    if not isinstance(workspace, dict):
        workspace = {}
    run_id = dash.get("run_id")
    phase = dash.get("phase") or workspace.get("current_phase") or "idle"
    release = dash.get("release")
    if not isinstance(release, dict):
        release = {}
    readiness = release.get("readiness") or workspace.get("readiness_level") or "L1"

    # Pipeline strip from recipe + skips + current phase.
    recipe = workspace.get("pipeline_recipe") or []
    skipped = workspace.get("skipped_phases") or []
    skip_reason: Dict[str, str] = {}
    for entry in skipped:
        if isinstance(entry, dict) and entry.get("phase"):
            skip_reason[str(entry["phase"])] = str(entry.get("reason") or "")
    current = workspace.get("current_phase") or dash.get("phase")
    try:
        current_idx = list(recipe).index(current) if current in list(recipe) else -1
    except Exception:
        current_idx = -1
    pipeline = []
    for idx, name in enumerate(list(recipe)):
        name = str(name)
        if name in skip_reason:
            reason = skip_reason[name]
            st = "refused" if "refus" in reason.lower() else "skipped"
            pipeline.append({"phase": name, "state": st, "reason": reason})
        elif idx < current_idx or (current_idx == -1 and False):
            pipeline.append({"phase": name, "state": "done", "reason": ""})
        elif name == current:
            pipeline.append({"phase": name, "state": "active", "reason": ""})
        else:
            pipeline.append({"phase": name, "state": "pending", "reason": ""})
    for name, reason in skip_reason.items():
        if name not in list(recipe):
            st = "refused" if "refus" in str(reason).lower() else "skipped"
            pipeline.append({"phase": name, "state": st, "reason": str(reason)})

    # Spend from tracker stats when present, else zeros; `left` from config caps.
    stats = dash.get("tracker_stats") or dash.get("stats") or {}
    if not isinstance(stats, dict):
        stats = {}
    try:
        calls = int(stats.get("llm_calls") or 0)
    except Exception:
        calls = 0
    try:
        tokens = int(stats.get("llm_tokens_in") or 0) + int(stats.get("llm_tokens_out") or 0)
    except Exception:
        tokens = 0
    try:
        usd = float(stats.get("cost_estimate_usd") or 0.0)
    except Exception:
        usd = 0.0
    try:
        max_calls = int(config.run_max_llm_calls)
        max_tokens = int(config.run_max_tokens)
        max_usd = float(config.run_max_usd)
    except Exception:
        max_calls, max_tokens, max_usd = 2000, 8000000, 500.0  # owner-generous caps (2026-10-04)
    spend = {
        "usd": round(usd, 4),
        "calls": calls,
        "tokens": tokens,
        "left": {
            "calls": max(0, max_calls - calls),
            "tokens": max(0, max_tokens - tokens),
            "usd": round(max(0.0, max_usd - usd), 4),
        },
    }

    # Blockers: terminal error + blocking findings + terminal evidence gate.
    blockers: list = []
    if workspace.get("terminal_error"):
        blockers.append(str(workspace.get("terminal_error")))
    eg = workspace.get("evidence_gate")
    if isinstance(eg, dict) and eg.get("terminal") and eg.get("message"):
        blockers.append(str(eg.get("message")))
    for finding in workspace.get("verification_findings") or []:
        if isinstance(finding, dict) and finding.get("blocking"):
            blockers.append(str(finding.get("details") or finding.get("check") or finding))

    return {
        "run_id": run_id,
        "phase": phase,
        "status": dash.get("status"),
        "readiness": readiness,
        "pipeline": pipeline,
        "spend": spend,
        "blockers": blockers,
    }


def _set_latest_state(state: Dict[str, Any]):
    """Thread-safe update of _latest_state (deep copy: the live run keeps
    mutating nested dicts/lists after the snapshot is taken)."""
    import copy as _copy

    global _latest_state
    with _run_lock:
        try:
            _latest_state = _copy.deepcopy(state)
        except Exception:
            _latest_state = dict(state)


def _get_latest_state() -> Dict[str, Any]:
    """Thread-safe read of _latest_state."""
    global _latest_state
    with _run_lock:
        return dict(_latest_state)


def _set_run_error(error: Optional[str]):
    """Thread-safe update of _run_error."""
    global _run_error
    with _run_lock:
        _run_error = error


def _get_run_error() -> Optional[str]:
    """Thread-safe read of _run_error."""
    global _run_error
    with _run_lock:
        return _run_error


def _run_pipeline(domain: Optional[str] = None, resume_run_id: Optional[str] = None,
                   precreated_run_id: Optional[str] = None,
                   intake: Optional[Dict[str, Any]] = None,
                   true_resume: bool = False):
    global _run_error, _latest_state, _run_thread
    with _run_lock:
        _run_error = None
        _latest_state = {}
    tracker = None
    try:
        if domain:
            apply_runtime_keys({"RESEARCH_DOMAIN": domain})
        validate_config()
        import os
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.draft_versions_dir, exist_ok=True)
        from core.logging_config import setup_logging
        setup_logging()
        from core.state import initialize_state
        from main import create_checkpointer, create_research_graph, save_results
        from core.context import create_run_context
        from core.pipeline import ResearchPipeline

        if resume_run_id and true_resume:
            # TRUE resume (U6): never delete checkpoints, never force reset.
            # LangGraph continues from the stored graph position (resume=True);
            # the DB snapshot only seeds the live UI state, unmutated.
            tracker = start_run(resume_run_id)
            from core.logging_config import attach_run_log
            attach_run_log(tracker.run_id)
            existing_state = _load_state_from_db(resume_run_id)
            state = existing_state if existing_state else initialize_state()
            _set_latest_state(dict(state))
            resume_flag = True
        elif resume_run_id:
            # Explicit Re-run: fresh start. Delete stale checkpoints so the
            # fresh run starts clean and old terminal_error / __end__ state
            # never haunts a re-run.
            research_db._delete_checkpoints(resume_run_id)
            tracker = start_run(resume_run_id)
            from core.logging_config import attach_run_log
            attach_run_log(tracker.run_id)
            existing_state = _load_state_from_db(resume_run_id)
            # Re-runs always start fresh: clear terminal_error and signal
            # reset so the graph re-enters topic_discovery instead of
            # terminating immediately from a stale checkpoint state.
            if existing_state:
                existing_state["terminal_error"] = None
                existing_state["should_reset"] = True
                existing_state["current_phase"] = None
            state = existing_state if existing_state else initialize_state()
            resume_flag = False
        elif precreated_run_id:
            # M0 U2: endpoint already minted + ledgered this id synchronously.
            # Attach to it (no checkpoint delete: fresh id has none).
            tracker = start_run(precreated_run_id)
            from core.logging_config import attach_run_log
            attach_run_log(tracker.run_id)
            state = initialize_state()
        else:
            tracker = start_run()
            from core.logging_config import attach_run_log
            attach_run_log(tracker.run_id)
            state = initialize_state()
        pipeline = ResearchPipeline(
            create_research_graph,
            create_checkpointer,
            context=create_run_context(tracker),
        )
        state["run_id"] = tracker.run_id
        if intake:
            # Seed intake so intake_node routes without an extra round-trip.
            if intake.get("goal"):
                state["user_query"] = intake["goal"]
            state["paper_type"] = intake.get("paper_type") or state.get("paper_type")
            state["hunt_mode"] = intake.get("hunt_mode") or state.get("hunt_mode") or "try_and_report"
            state["intake_report"] = {
                "goal": intake.get("goal", ""),
                "domain": domain or config.research_domain,
                "requested": intake.get("paper_type") or "auto",
                "hunt_mode": intake.get("hunt_mode") or "try_and_report",
                "pack_manifest": {"pack_id": intake.get("pack_id")} if intake.get("pack_id") else {},
                "depth": intake.get("depth") or "Standard",
                "budgets": {
                    "budget_usd": intake.get("budget_usd"),
                    "budget_min": intake.get("budget_min"),
                },
            }
        _set_latest_state(state)
        from core.artifacts import PHASE_SNAPSHOT_NODES, save_failure_dossier, save_phase_snapshot

        # Budget thresholds (§10.4: budget_check at 50/80/100%). Caps come
        # from the intake depth map (§5.3) with explicit budget_usd/budget_min
        # overrides; a blank/zero budget means "depth default", never zero.
        import time as _time

        _run_t0 = _time.time()
        _budget_depth = ((state.get("intake_report") or {}).get("depth") or "Standard")
        _depth_caps = DEPTH_RUN_CAPS.get(_budget_depth, DEPTH_RUN_CAPS["Standard"])
        _intake_budgets = ((state.get("intake_report") or {}).get("budgets")) or {}
        _cap_calls = float(_depth_caps["calls"])
        _cap_usd = float(_intake_budgets.get("budget_usd") or _depth_caps["usd"])
        _cap_tokens = float(config.run_max_tokens)
        _cap_sec = float(_intake_budgets.get("budget_min") or _depth_caps["minutes"]) * 60.0
        _budget_fired: set = set()

        def _maybe_emit_budget_check():
            try:
                stats = getattr(tracker, "stats", {}) or {}
                calls = float(stats.get("llm_calls") or 0)
                tokens = float(stats.get("llm_tokens_in") or 0) + float(stats.get("llm_tokens_out") or 0)
                usd = float(stats.get("cost_estimate_usd") or 0.0)
                elapsed = _time.time() - _run_t0
                frac = max(
                    calls / _cap_calls if _cap_calls > 0 else 0.0,
                    tokens / _cap_tokens if _cap_tokens > 0 else 0.0,
                    usd / _cap_usd if _cap_usd > 0 else 0.0,
                    elapsed / _cap_sec if _cap_sec > 0 else 0.0,
                )
                for threshold in (0.5, 0.8, 1.0):
                    if frac >= threshold and threshold not in _budget_fired:
                        _budget_fired.add(threshold)
                        emit_event(
                            "budget_check",
                            {"threshold": threshold, "actual": round(frac, 3),
                             "calls": calls, "tokens": tokens, "usd": round(usd, 4),
                             "cap_calls": _cap_calls, "cap_tokens": _cap_tokens,
                             "cap_usd": _cap_usd, "cap_minutes": round(_cap_sec / 60.0, 1)},
                            run_id=tracker.run_id,
                            agent="BudgetGuard",
                        )
            except Exception as e:
                logger.warning("budget_check emit failed: %s", e)

        def on_node(node_name, node_output):
            _set_latest_state(dict(node_output))
            tracker.message(f"{node_name} -> {node_output.get('current_phase')}")
            if node_name in PHASE_SNAPSHOT_NODES:
                save_phase_snapshot(node_output, tracker.run_id, node_name)
            _maybe_emit_budget_check()

        def on_crash(crash_state, exc):
            try:
                if isinstance(crash_state, dict) and not crash_state.get("terminal_error"):
                    crash_state["terminal_error"] = f"Unhandled crash: {exc}"
                dossier = save_failure_dossier(crash_state, config.output_dir)
                tracker.message(f"Crash dossier saved to {dossier}", level="error")
            except Exception:
                logger.exception("Failed to write crash dossier (web)")

        # True resume continues the stored graph position; every other path
        # starts fresh (resume=False) to avoid a stale checkpoint causing
        # immediate termination via terminal_error or __end__.
        result = pipeline.run(
            state, tracker.run_id, resume=resume_flag if resume_run_id else False,
            on_node=on_node, finalize=save_results, on_crash=on_crash,
        )
        last = result.state
        research_db.update_run_summary(tracker.run_id, {"workspace": _workspace_snapshot(last, tracker=tracker)})
        terminal_error = last.get("terminal_error")
        _set_run_error(terminal_error)
        if terminal_error:
            tracker.message(terminal_error, level="error")
    except Exception as e:
        logger.exception("Pipeline crashed")
        _set_run_error(str(e))
        if tracker:
            tracker.message(f"ERROR: {e}", level="error")
            tracker.complete(success=False)
        else:
            # Emit error event even without a tracker so the UI surfaces it.
            run_id_for_event = resume_run_id or "unknown"
            emit_event("error", {"message": str(e)}, run_id=run_id_for_event, agent="system")
    finally:
        with _run_lock:
            _run_thread = None


@app.post("/api/run")
def start_research(payload: RunPayload = RunPayload()):
    global _run_thread
    with _run_lock:
        if _run_thread and _run_thread.is_alive():
            raise HTTPException(status_code=409, detail={"detail": "A run is already in progress", "type": "run_in_progress"})
        # M0 U2: synchronous validation before spend + {run_id} return.
        err = validate_run_payload(payload)
        if err:
            from fastapi.responses import JSONResponse as _JR

            return _JR(status_code=400, content={"detail": err, "type": "validation_error"})
        if payload.provider:
            apply_runtime_keys({"LLM_PROVIDER": payload.provider})
            reset_llm_client()
        # Mint run_id synchronously (ledger row now; thread attaches via start_run).
        import uuid as _uuid

        run_id = str(_uuid.uuid4())[:8]
        try:
            from core.research_db import research_db as _db
            from datetime import datetime, timezone as _tz

            _db.create_run(run_id, datetime.now(_tz.utc).isoformat().replace("+00:00", "Z"))
        except Exception:
            pass
        goal = (payload.goal or "")
        intake_seed = {
            "goal": goal,
            "paper_type": (payload.paper_type or "auto"),
            "depth": (payload.depth or "Standard"),
            "pack_id": payload.pack_id,
            "budget_usd": payload.budget_usd,
            "budget_min": payload.budget_min,
            "hunt_mode": (payload.hunt_mode or "try_and_report"),
        }
        _run_thread = threading.Thread(
            target=_run_pipeline,
            args=(payload.domain, None),
            kwargs={"precreated_run_id": run_id, "intake": intake_seed},
            daemon=True,
        )
        _run_thread.start()
    return {"run_id": run_id}


@app.post("/api/run/resume/{run_id}")
def resume_run(run_id: str):
    """TRUE resume (U6): continue the LangGraph graph position from checkpoints.

    Must NOT delete checkpoints, must NOT force reset/should_reset, and runs
    the pipeline with ``resume=True``. For an explicit fresh start, use
    ``POST /api/run/rerun/{run_id}`` — restarts are never disguised as resumes.
    """
    global _run_thread
    from fastapi.responses import JSONResponse as _JR

    with _run_lock:
        if _run_thread and _run_thread.is_alive():
            return _JR(status_code=409, content={"detail": "A run is already in progress", "type": "run_in_progress"})
        runs = research_db.list_runs(limit=200)
        prev = next((r for r in runs if r["run_id"] == run_id), None)
        if not prev:
            return _JR(status_code=404, content={"detail": f"Run {run_id} not found", "type": "not_found"})
        _run_thread = threading.Thread(
            target=_run_pipeline, args=(None, run_id),
            kwargs={"true_resume": True}, daemon=True,
        )
        _run_thread.start()
    return {"ok": True, "message": f"Resuming run {run_id} from checkpoint"}


@app.post("/api/run/rerun/{run_id}")
def rerun_run(run_id: str, payload: RerunPayload = RerunPayload()):
    """Explicit Re-run (U6): delete checkpoints, clear terminal_error, force reset.

    This is the labelled restart path (previous resume-endpoint behavior lives
    here). Never call this a resume. An optional ``{"note"}`` body is recorded
    to cross-run memory (the final-review "Request re-run with note" control
    promises this feed — the note must never be silently discarded).
    """
    global _run_thread
    from fastapi.responses import JSONResponse as _JR

    note = ""
    try:
        note = str((payload.note if payload else "") or "").strip()[:2000]
    except Exception:
        note = ""
    if note:
        try:
            CrossRunMemory().record(
                "rerun_note",
                {"run_id": run_id, "note": note, "outcome_status": "setup",
                 "retrieval_eligible": False},
            )
        except Exception as e:
            logger.warning("Failed to record re-run note for %s: %s", run_id, e)

    with _run_lock:
        if _run_thread and _run_thread.is_alive():
            return _JR(status_code=409, content={"detail": "A run is already in progress", "type": "run_in_progress"})
        runs = research_db.list_runs(limit=200)
        prev = next((r for r in runs if r["run_id"] == run_id), None)
        if not prev:
            return _JR(status_code=404, content={"detail": f"Run {run_id} not found", "type": "not_found"})
        _run_thread = threading.Thread(
            target=_run_pipeline, args=(None, run_id),
            kwargs={"true_resume": False}, daemon=True
        )
        _run_thread.start()
    return {"ok": True, "message": f"Re-running run {run_id} from scratch"}


def _sweep_stale_running_runs() -> int:
    """Mark ledger 'running' rows as interrupted when no run thread is alive.

    Mirrors the CLI sweep (main.py): a crash/kill leaves status='running'
    forever, and the dashboard would show it as running. Only sweeps when no
    pipeline thread is alive in this process, so a live run is never touched.
    Returns the number of rows swept.
    """
    try:
        with _run_lock:
            alive = bool(_run_thread and _run_thread.is_alive())
        if alive:
            return 0
        stale = research_db.get_stale_running_runs()
        count = 0
        for row in stale or []:
            try:
                research_db.finish_run(row["run_id"], "interrupted", "unknown", {
                    "terminal_error": "Process terminated before completion (swept by web)",
                })
                count += 1
            except Exception:
                pass
        if count:
            logger.info("Swept %d stale running run(s) → interrupted", count)
        return count
    except Exception as exc:
        logger.debug("Stale-run sweep failed: %s", exc)
        return 0


@app.get("/api/run/status")
def run_status():
    alive = bool(_run_thread and _run_thread.is_alive())
    if not alive:
        _sweep_stale_running_runs()
    tracker = get_tracker()
    return {
        "running": alive,
        "run_id": tracker.run_id if tracker else None,
        "error": _get_run_error(),
    }


@app.get("/api/runs")
def list_runs(limit: int = 50):
    """List all recorded runs for the admin panel to select from."""
    runs = research_db.list_runs(limit=limit)
    return {"runs": runs}


@app.post("/api/packs")
async def upload_pack(request: Request):
    """Upload field packs → manifest + hash (§8.7/Appendix B, M3).

    Multipart files: ``response_table`` (.csv/.json/.jsonl ≤50MB),
    ``codebook`` (.csv), ``case_pack`` (.zip ≤200MB),
    ``corpus_override`` (.json). Returns ``{pack_id,
    manifest[{name, bytes, sha256, rows|entries}], codebook_match}``
    with named invalid-pack reasons (400 + ``{"detail","type"}``).
    """
    from fastapi.responses import JSONResponse as _JR

    try:
        form = await request.form()
    except Exception as exc:
        return _JR(status_code=400, content={"detail": f"multipart parse failed: {exc}", "type": "validation_error"})
    from core.packs import (
        CASE_PACK_MAX_BYTES,
        RESPONSE_TABLE_MAX_BYTES,
        build_pack_manifest,
        packs_dir,
        save_pack,
        validate_case_pack_file,
        validate_codebook_file,
        validate_corpus_override,
        validate_response_table_file,
    )

    try:
        staging = packs_dir(config.output_dir) / "_staging"
        staging.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return _JR(status_code=500, content={"detail": f"pack store unavailable: {exc}", "type": "infra_error"})
    response_result = codebook_result = case_result = None
    corpus_result = None
    saved_names: list = []
    try:
        for field in ("response_table", "codebook", "case_pack", "corpus_override"):
            upload = form.get(field)
            if upload is None:
                continue
            filename = str(getattr(upload, "filename", "") or field)
            max_bytes = CASE_PACK_MAX_BYTES if field == "case_pack" else RESPONSE_TABLE_MAX_BYTES
            known_size = getattr(upload, "size", None)
            if isinstance(known_size, int) and known_size > max_bytes:
                return _JR(status_code=413, content={
                    "detail": f"{field} exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
                    "type": "needs_pack_invalid",
                })
            dest = staging / f"{field}_{filename.replace('/', '_').replace(chr(92), '_')}"
            try:
                content = await upload.read(max_bytes + 1)
            except Exception as exc:
                return _JR(status_code=400, content={"detail": f"{field} read failed: {exc}", "type": "validation_error"})
            if len(content) > max_bytes:
                return _JR(status_code=413, content={
                    "detail": f"{field} exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
                    "type": "needs_pack_invalid",
                })
            try:
                dest.write_bytes(content)
            except Exception as exc:
                return _JR(status_code=500, content={"detail": f"{field} store failed: {exc}", "type": "infra_error"})
            saved_names.append(dest.name)
            if field == "response_table":
                response_result = validate_response_table_file(dest)
                if response_result.get("manifest"):
                    response_result["manifest"]["name"] = filename
                if not response_result.get("ok"):
                    return _JR(status_code=400, content={
                        "detail": "; ".join(response_result.get("reasons") or ["invalid response_table"]),
                        "type": "needs_pack_invalid"})
            elif field == "codebook":
                codebook_result = validate_codebook_file(dest)
                if codebook_result.get("manifest"):
                    codebook_result["manifest"]["name"] = filename
                if not codebook_result.get("ok"):
                    return _JR(status_code=400, content={
                        "detail": "; ".join(codebook_result.get("reasons") or ["invalid codebook"]),
                        "type": "needs_pack_invalid"})
            elif field == "case_pack":
                case_result = validate_case_pack_file(dest)
                if isinstance(case_result.get("manifest"), dict):
                    case_result["manifest"]["name"] = filename
                if not case_result.get("ok"):
                    reason = str((case_result.get("reasons") or ["invalid case_pack"])[0])
                    err_type = "needs_case_pack" if reason.startswith("needs_case_pack") else "needs_pack_invalid"
                    return _JR(status_code=400, content={"detail": reason, "type": err_type})
            elif field == "corpus_override":
                import json as _json

                try:
                    corpus_result = validate_corpus_override(_json.loads(dest.read_text(encoding="utf-8")))
                except Exception as exc:
                    return _JR(status_code=400, content={
                        "detail": f"needs_pack_invalid: corpus_override parse failed ({exc})",
                        "type": "needs_pack_invalid"})
                if not corpus_result.get("ok"):
                    return _JR(status_code=400, content={
                        "detail": "; ".join(corpus_result.get("reasons") or ["invalid corpus_override"]),
                        "type": "needs_pack_invalid"})
        if response_result is None and codebook_result is None and case_result is None and corpus_result is None:
            return _JR(status_code=400, content={
                "detail": "no pack file supplied (response_table/codebook/case_pack/corpus_override)",
                "type": "validation_error"})
        manifest = build_pack_manifest(response_result, codebook_result, case_result, corpus_result)
        pack_id, _ = save_pack(manifest, config.output_dir)
        # Move staged files under the pack id for audit (best-effort).
        moved: dict = {}
        try:
            pack_files = packs_dir(config.output_dir) / pack_id
            pack_files.mkdir(parents=True, exist_ok=True)
            for name in saved_names:
                src = staging / name
                if src.is_file():
                    dest = pack_files / name
                    src.replace(dest)
                    moved[name] = str(dest)
        except Exception:
            pass
        # Persist local paths so survey/case nodes can read the pack files.
        if moved:
            try:
                import json as _json2

                def _attach(result, key):
                    if not isinstance(result, dict):
                        return
                    manifest_block = result.get("manifest") if isinstance(result.get("manifest"), dict) else None
                    for name, path in moved.items():
                        if manifest_block and manifest_block.get("name") and manifest_block["name"] in name:
                            manifest[key]["path"] = path
                            sub = manifest[key].get("manifest")
                            if isinstance(sub, dict):
                                sub["path"] = path

                _attach(response_result, "response_table")
                _attach(codebook_result, "codebook")
                _attach(case_result, "case_pack")
                pack_file = packs_dir(config.output_dir) / f"{pack_id}.json"
                pack_file.write_text(_json2.dumps({"pack_id": pack_id, "manifest": manifest}, default=str, indent=2), encoding="utf-8")
            except Exception:
                pass
        return {"pack_id": pack_id, "manifest": manifest.get("files", []),
                "codebook_match": bool(manifest.get("codebook_match", False)),
                "pack": manifest}
    finally:
        try:
            for name in saved_names:
                leftover = staging / name
                if leftover.is_file():
                    leftover.unlink()
        except Exception:
            pass


# Chat control plane (session history separate; run state shared with dashboard).
from web.chat_api import router as chat_router

app.include_router(chat_router)


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def main():
    import uvicorn

    uvicorn.run(
        "web.app:app",
        host=config.web_host,
        port=config.web_port,
        reload=False,
        access_log=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
