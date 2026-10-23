"""Chat control-plane API — UI layer on top of the ScholarGraph harness.

Shares run state with the dashboard (tracker / research_db / /api/dashboard).
Owns only chat session history and skill-routing for conversational control.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from web import chat_session
from web.chat_skills import (
    SKILL_KINDS,
    build_run_overview,
    classify_intent,
    extract_config_patches,
    help_text,
    parse_domain_topic_pairs,
    summarize_run_fallback,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])
CHAT_UPLOAD_MAX_BYTES = 50 * 1024 * 1024
CHAT_CASE_PACK_MAX_BYTES = 200 * 1024 * 1024
CHAT_MEDIA_MAX_BYTES = 20 * 1024 * 1024


async def _read_upload_limited(file: UploadFile, size_limit: int) -> bytes:
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await file.read(min(1024 * 1024, size_limit - total + 1))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > size_limit:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {size_limit // (1024 * 1024)} MB limit",
            )
        chunks.append(chunk)


class ChatMessagePayload(BaseModel):
    message: str = ""
    session_id: Optional[str] = None
    focus_run_id: Optional[str] = None
    approve: Optional[bool] = None
    reject: Optional[bool] = None
    run_id: Optional[str] = None


class SessionCreatePayload(BaseModel):
    focus_run_id: Optional[str] = None


def _ok(data: Dict[str, Any], status: int = 200):
    return JSONResponse(status_code=status, content=data)


def _err(detail: str, err_type: str = "chat_error", status: int = 400):
    return JSONResponse(status_code=status, content={"detail": detail, "type": err_type})


def _harness_snapshot(run_id: Optional[str] = None) -> Dict[str, Any]:
    """Pull shared harness state via the same builders the dashboard uses."""
    from web import app as webapp

    try:
        dash = webapp.dashboard(run_id=run_id)
    except Exception as exc:
        logger.warning("dashboard snapshot failed: %s", exc)
        dash = {"run_id": run_id, "status": "idle", "phase": "idle", "workspace": {}}
    try:
        status = webapp.run_status()
    except Exception:
        status = {"running": False, "run_id": None, "error": None}
    try:
        runs_payload = webapp.list_runs(limit=30)
        runs = runs_payload.get("runs") if isinstance(runs_payload, dict) else []
    except Exception:
        runs = []
    overview = build_run_overview(dash if isinstance(dash, dict) else {}, status)
    # Only surface truly live / paused work — stale DB "running" rows (crashed
    # processes) must not flood the chat chrome.
    live_id = status.get("run_id") if status.get("running") else None
    active = []
    if live_id:
        active.append({
            "run_id": live_id,
            "status": "active",
            "phase": overview.get("phase"),
        })
    for r in runs or []:
        rid = r.get("run_id")
        st = str(r.get("status") or "").lower()
        if rid and rid == live_id:
            continue
        # "interrupted" is historical (crash/kill), not live work — omit it.
        if st in ("paused", "awaiting_approval"):
            active.append({
                "run_id": rid,
                "status": st,
                "phase": r.get("phase"),
                "updated_at": r.get("updated_at") or r.get("finished_at") or r.get("started_at"),
            })
    active = active[:8]
    return {
        "overview": overview,
        "dashboard": dash if isinstance(dash, dict) else {},
        "status": status,
        "runs": runs or [],
        "active_runs": active,
    }


def _llm_reply(message: str, history: List[Dict[str, Any]], snap: Dict[str, Any]) -> str:
    """Gemini-backed conversational reply with harness context."""
    try:
        from core.llm import call_llm
    except Exception as exc:
        return f"(LLM unavailable: {exc})\n\n{_status_blurb(snap)}"

    overview = snap.get("overview") or {}
    hist_txt = "\n".join(
        f"{m.get('role')}: {m.get('content', '')[:400]}"
        for m in history[-10:]
        if m.get("role") in ("user", "assistant")
    )
    active = snap.get("active_runs") or []
    prompt = f"""You are the ScholarGraph chat control plane — a friendly UI layer on top of the research harness.
You can see shared run state (same as the dashboard) and help the operator drive runs end-to-end.
Be concrete: name stages, agents, and run ids. Keep answers scannable (short paragraphs, bullets).
Do not invent completed experiments. If idle, say so and offer to start a run.
No auth — the operator may start runs freely.

CURRENT RUN OVERVIEW:
{json.dumps(overview, default=str)[:5000]}

ACTIVE RUNS:
{json.dumps(active, default=str)[:1500]}

CHAT HISTORY:
{hist_txt}

OPERATOR:
{message}
"""
    try:
        return call_llm(prompt, temperature=0.35, tier="judge", max_tokens=1400) or _status_blurb(snap)
    except Exception as exc:
        logger.warning("chat LLM failed: %s", exc)
        return f"{_status_blurb(snap)}\n\n_(Model call failed: {exc})_"


def _status_blurb(snap: Dict[str, Any]) -> str:
    ov = snap.get("overview") or {}
    rid = ov.get("run_id") or "—"
    return (
        f"**Run `{rid}`** — {ov.get('status') or 'idle'} · phase `{ov.get('phase') or 'idle'}`\n"
        f"Topic: {ov.get('topic') or '—'}\n"
        f"Active agent: {ov.get('active_agent') or '—'}\n"
        f"Task: {ov.get('active_task') or '—'}"
    )


def _cards_for_snapshot(snap: Dict[str, Any], include_approval: bool = True) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    ov = snap.get("overview") or {}
    if ov.get("run_id"):
        cards.append({"kind": "run_overview", "data": ov})
    if ov.get("pipeline"):
        cards.append({"kind": "stage_rail", "data": {"pipeline": ov["pipeline"], "phase": ov.get("phase")}})
    if include_approval and (ov.get("release") or {}).get("needs_approval"):
        cards.append({
            "kind": "approval",
            "data": {
                "run_id": ov.get("run_id"),
                "reason": (ov.get("release") or {}).get("reason") or "Release gate awaiting human approval",
                "missing": (ov.get("release") or {}).get("missing") or [],
            },
        })
    active = snap.get("active_runs") or []
    if active:
        cards.append({"kind": "active_runs", "data": {"runs": active}})
    return cards


def _assistant_message(
    content: str,
    *,
    skill: str = "chat",
    cards: Optional[List[Dict[str, Any]]] = None,
    thinking: Optional[str] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "skill": skill if skill in SKILL_KINDS else "chat",
        "cards": cards or [],
        "thinking": thinking,
        "artifacts": artifacts or [],
        "actions": actions or [],
    }


@router.post("/session")
def create_chat_session(payload: SessionCreatePayload = SessionCreatePayload()):
    session = chat_session.create_session()
    if payload.focus_run_id:
        chat_session.set_focus_run(session["id"], payload.focus_run_id)
        session["focus_run_id"] = payload.focus_run_id
    snap = _harness_snapshot(payload.focus_run_id)
    return {
        "session": session,
        "harness": {
            "overview": snap["overview"],
            "active_runs": snap["active_runs"],
            "status": snap["status"],
        },
    }


@router.get("/sessions")
def list_chat_sessions():
    sessions = []
    for session in chat_session.list_sessions():
        messages = session.get("messages") or []
        if not messages:
            continue
        first_user = next(
            (message for message in messages if message.get("role") == "user"),
            None,
        )
        attachments = (first_user or {}).get("attachments") or []
        title = (first_user or {}).get("content") or ""
        if not title and attachments:
            title = attachments[0].get("name") or ""
        sessions.append({
            "id": session["id"],
            "title": title.strip()[:72] or "New chat",
            "updated_at": session["updated_at"],
            "message_count": len(messages),
        })
    return {"sessions": sessions}


@router.get("/session/{session_id}")
def get_chat_session(session_id: str):
    session = chat_session.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail={"detail": "Session not found", "type": "not_found"})
    snap = _harness_snapshot(session.get("focus_run_id"))
    return {
        "session": session,
        "messages": chat_session.list_messages(session_id),
        "harness": {
            "overview": snap["overview"],
            "active_runs": snap["active_runs"],
            "status": snap["status"],
        },
    }


@router.get("/harness")
def chat_harness_view(run_id: Optional[str] = None):
    """Shared harness snapshot for the chat chrome (stage rail, active runs)."""
    snap = _harness_snapshot(run_id)
    return {
        "overview": snap["overview"],
        "active_runs": snap["active_runs"],
        "status": snap["status"],
        "runs": [
            {
                "run_id": r.get("run_id"),
                "status": r.get("status"),
                "phase": r.get("phase"),
            }
            for r in (snap.get("runs") or [])[:20]
        ],
    }


@router.post("/message")
async def chat_message(payload: ChatMessagePayload):
    session = chat_session.ensure_session(payload.session_id)
    sid = session["id"]
    focus = payload.focus_run_id or session.get("focus_run_id")
    if payload.focus_run_id:
        chat_session.set_focus_run(sid, payload.focus_run_id)
        focus = payload.focus_run_id

    # Inline approval / reject from chat buttons
    if payload.approve is True or payload.reject is True:
        return await _handle_approval(
            sid,
            run_id=payload.run_id or focus,
            approve=bool(payload.approve),
        )

    text = (payload.message or "").strip()
    if not text:
        return _err("message is required")

    chat_session.append_message(sid, {"role": "user", "content": text})
    skill, params = classify_intent(text)
    if params.get("run_id"):
        focus = params["run_id"]
        chat_session.set_focus_run(sid, focus)

    reply = await _dispatch_skill(sid, skill, params, text=text, focus=focus)
    stored = chat_session.append_message(sid, reply)
    return {
        "session_id": sid,
        "message": stored,
        "harness": _public_harness(focus),
    }


@router.post("/upload")
async def chat_upload(
    session_id: Optional[str] = Form(None),
    message: Optional[str] = Form(""),
    focus_run_id: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
):
    uploads = files or ([file] if file else [])
    if not uploads:
        raise HTTPException(status_code=400, detail="Attach at least one file")

    session = chat_session.ensure_session(session_id)
    sid = session["id"]
    if focus_run_id:
        chat_session.set_focus_run(sid, focus_run_id)
    user_note = (message or "").strip()
    prompt_was_provided = bool(user_note)
    prepared = []
    for upload in uploads:
        name = upload.filename or "upload.bin"
        suffix = Path(name.replace("\\", "/")).suffix.lower()
        mime_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
        is_media = (
            suffix == ".pdf"
            or suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".heif"}
            or mime_type.startswith(("image/", "audio/", "video/"))
        )
        size_limit = (
            CHAT_CASE_PACK_MAX_BYTES
            if suffix == ".zip"
            else CHAT_MEDIA_MAX_BYTES
            if is_media
            else CHAT_UPLOAD_MAX_BYTES
        )
        raw = await _read_upload_limited(upload, size_limit)
        is_text = (
            suffix in {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".xml", ".html", ".py", ".log"}
            or mime_type.startswith("text/")
            or mime_type == "application/json"
        )
        text_body = ""
        if is_text:
            try:
                text_body = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text_body = raw.decode("latin-1")
        prepared.append({
            "name": Path(name.replace("\\", "/")).name or "upload.bin",
            "raw": raw,
            "mime_type": mime_type or _mime_for_suffix(suffix),
            "text": text_body,
        })

    user_note = user_note or "Please review the attached files."
    chat_session.append_message(
        sid,
        {
            "role": "user",
            "content": user_note,
            "attachments": [
                {"name": item["name"], "bytes": len(item["raw"]), "mime": item["mime_type"]}
                for item in prepared
            ],
        },
    )

    first_text = next((item["text"] for item in prepared if item["text"]), "")
    pairs = parse_domain_topic_pairs(first_text) if first_text else []
    explicit_topic_list = (
        len(prepared) == 1
        and _is_domain_topic_list_upload(first_text, user_note, pairs)
        and (
            not prompt_was_provided
            or any(marker in user_note.lower() for marker in ("topic list", "start a run", "research these"))
        )
    )
    auto_pack_upload = not prompt_was_provided and all(
        Path(item["name"]).suffix.lower() in {".zip", ".csv", ".json", ".jsonl"}
        for item in prepared
    )
    if explicit_topic_list:
        reply = await _kickoff_from_pairs(sid, pairs, note=user_note)
    elif auto_pack_upload:
        responses = [
            await _upload_as_pack(sid, item["name"], item["raw"], item["mime_type"])
            for item in prepared
        ]
        reply = _combine_upload_responses(responses)
    else:
        reply = await _interpret_uploads(user_note, prepared)

    stored = chat_session.append_message(sid, reply)
    return {
        "session_id": sid,
        "message": stored,
        "harness": _public_harness(focus_run_id or session.get("focus_run_id")),
        "parsed_pairs": pairs[:50],
    }


def _mime_for_suffix(suffix: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
    }.get(suffix, "application/octet-stream")


def _combine_upload_responses(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(responses) == 1:
        return responses[0]
    return _assistant_message(
        "\n\n".join(response.get("content", "File processed.") for response in responses),
        skill="upload_pack",
        artifacts=[
            artifact
            for response in responses
            for artifact in response.get("artifacts", [])
        ],
    )


def _has_domain_topic_header(text: str) -> bool:
    first_line = next(
        (
            line.strip()
            for line in (text or "").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ),
        "",
    )
    if not first_line:
        return False
    for delimiter in (",", "\t", "|"):
        if delimiter not in first_line:
            continue
        columns = [part.strip().strip('"').lower() for part in first_line.split(delimiter, 1)]
        if len(columns) == 2 and columns[0] in {"domain", "research_domain"} and columns[1] in {"topic", "goal", "research_question"}:
            return True
    return False


def _is_domain_topic_list_upload(
    text: str,
    note: str,
    pairs: List[Dict[str, str]],
) -> bool:
    if not pairs:
        return False
    if _has_domain_topic_header(text):
        return True
    lower_note = (note or "").lower()
    return any(
        marker in lower_note
        for marker in ("topic list", "domain/topic", "domain and topic", "domain,topic")
    )


async def _handle_approval(session_id: str, run_id: Optional[str], approve: bool):
    from web import app as webapp

    if not run_id:
        msg = _assistant_message(
            "No run id to approve. Open a run first or pass a run id.",
            skill="approve",
        )
        stored = chat_session.append_message(session_id, msg)
        return {"session_id": session_id, "message": stored, "harness": _public_harness(None)}

    chat_session.append_message(
        session_id,
        {"role": "user", "content": ("Approve" if approve else "Reject") + f" release for `{run_id}`"},
    )
    if not approve:
        msg = _assistant_message(
            f"Rejection recorded in chat for `{run_id}`. The harness release gate stays closed "
            f"(no publishable stamp). Re-run or continue work from the dashboard when ready.",
            skill="reject",
            cards=_cards_for_snapshot(_harness_snapshot(run_id), include_approval=False),
        )
        stored = chat_session.append_message(session_id, msg)
        return {"session_id": session_id, "message": stored, "harness": _public_harness(run_id)}

    try:
        result = webapp.approve_release(webapp.ApprovePayload(run_id=run_id), run_id=run_id)
        if isinstance(result, JSONResponse):
            body = json.loads(result.body.decode("utf-8"))
            detail = body.get("detail") or "Approval failed"
            msg = _assistant_message(
                f"Could not approve `{run_id}`: {detail}",
                skill="approve",
                cards=_cards_for_snapshot(_harness_snapshot(run_id)),
            )
        else:
            msg = _assistant_message(
                f"Approved release for `{run_id}`. Publishable stamp recorded.",
                skill="approve",
                cards=_cards_for_snapshot(_harness_snapshot(run_id), include_approval=False),
            )
    except Exception as exc:
        msg = _assistant_message(f"Approval error: {exc}", skill="approve")
    stored = chat_session.append_message(session_id, msg)
    return {"session_id": session_id, "message": stored, "harness": _public_harness(run_id)}


def _public_harness(run_id: Optional[str]) -> Dict[str, Any]:
    snap = _harness_snapshot(run_id)
    return {
        "overview": snap["overview"],
        "active_runs": snap["active_runs"],
        "status": snap["status"],
    }


async def _dispatch_skill(
    sid: str,
    skill: str,
    params: Dict[str, Any],
    *,
    text: str,
    focus: Optional[str],
) -> Dict[str, Any]:
    from web import app as webapp

    if skill == "help":
        return _assistant_message(help_text(), skill="help")

    if skill == "list_runs":
        snap = _harness_snapshot(focus)
        lines = ["**Recent runs**"]
        for r in (snap.get("runs") or [])[:15]:
            lines.append(
                f"- `{r.get('run_id')}` — {r.get('status')} · `{r.get('phase') or '—'}`"
            )
        if snap.get("active_runs"):
            lines.append("\n**Active now**")
            for r in snap["active_runs"]:
                lines.append(f"- `{r.get('run_id')}` · `{r.get('phase')}`")
        return _assistant_message(
            "\n".join(lines) if len(lines) > 1 else "No runs recorded yet.",
            skill="list_runs",
            cards=_cards_for_snapshot(snap),
        )

    if skill in ("status", "inspect_run"):
        rid = params.get("run_id") or focus
        snap = _harness_snapshot(rid)
        if rid:
            chat_session.set_focus_run(sid, rid)
        ov = snap["overview"]
        body = (
            f"### Run `{ov.get('run_id') or '—'}`\n"
            f"{_status_blurb(snap)}\n\n"
            f"Planned path: "
            + ", ".join(
                f"`{p.get('phase')}`" for p in (ov.get("pipeline") or [])[:10]
            )
        )
        if ov.get("plan_experiments"):
            body += "\n\n**Experiments / plan**\n" + "\n".join(
                f"- {e.get('title')} ({e.get('status')})" for e in ov["plan_experiments"]
            )
        return _assistant_message(
            body,
            skill="inspect_run",
            thinking=f"Inspecting harness state for {rid or 'latest'}",
            cards=_cards_for_snapshot(snap),
        )

    if skill == "start_run" or skill == "research":
        payload = webapp.RunPayload(
            domain=params.get("domain"),
            goal=params.get("goal") or text,
            paper_type=params.get("paper_type") or "auto",
            depth=params.get("depth") or "Standard",
            hunt_mode=params.get("hunt_mode") or "try_and_report",
        )
        err = webapp.validate_run_payload(payload)
        if err:
            return _assistant_message(
                f"Can't start yet: {err}\n\nFix config in chat (`set RESEARCH_DOMAIN=…`) "
                f"or open the dashboard settings.",
                skill="start_run",
            )
        try:
            result = webapp.start_research(payload)
            if isinstance(result, JSONResponse):
                body = json.loads(result.body.decode("utf-8"))
                return _assistant_message(
                    f"Start failed: {body.get('detail')}",
                    skill="start_run",
                )
            run_id = result.get("run_id")
            chat_session.set_focus_run(sid, run_id)
            snap = _harness_snapshot(run_id)
            return _assistant_message(
                f"Started run `{run_id}`.\n"
                f"Goal: {payload.goal or '—'}\n"
                f"Domain: {payload.domain or '(config default)'} · depth {payload.depth}",
                skill="start_run",
                thinking="Queued research pipeline",
                cards=_cards_for_snapshot(snap),
                actions=[{"label": "Open dashboard", "href": "/dashboard"}],
            )
        except HTTPException as exc:
            detail = exc.detail
            if isinstance(detail, dict):
                detail = detail.get("detail") or detail
            return _assistant_message(f"Start failed: {detail}", skill="start_run")
        except Exception as exc:
            return _assistant_message(f"Start failed: {exc}", skill="start_run")

    if skill == "resume_run":
        rid = params.get("run_id") or focus
        if not rid:
            return _assistant_message("Which run should I resume? Pass an 8-char run id.", skill="resume_run")
        try:
            result = webapp.resume_run(rid)
            if isinstance(result, JSONResponse):
                body = json.loads(result.body.decode("utf-8"))
                return _assistant_message(f"Resume failed: {body.get('detail')}", skill="resume_run")
            chat_session.set_focus_run(sid, rid)
            snap = _harness_snapshot(rid)
            return _assistant_message(
                f"Resuming `{rid}` from checkpoint.",
                skill="resume_run",
                cards=_cards_for_snapshot(snap),
            )
        except Exception as exc:
            return _assistant_message(f"Resume failed: {exc}", skill="resume_run")

    if skill == "approve":
        return (
            await _handle_approval(sid, params.get("run_id") or focus, approve=True)
        )["message"]

    if skill == "reject":
        return (
            await _handle_approval(sid, params.get("run_id") or focus, approve=False)
        )["message"]

    if skill == "update_config":
        patches = extract_config_patches(params.get("raw") or text)
        if not patches:
            return _assistant_message(
                "Tell me what to change, e.g. `GEMINI_MODEL=gemini-2.5-flash` "
                "or `set domain to computer science`.",
                skill="update_config",
            )
        try:
            result = webapp.post_keys(webapp.KeysPayload(keys=patches))
            return _assistant_message(
                f"Updated harness config: {', '.join(f'`{k}`' for k in patches)}\n"
                f"Provider now: {(result or {}).get('provider') if isinstance(result, dict) else '—'}",
                skill="update_config",
            )
        except Exception as exc:
            try:
                from core.config import apply_runtime_keys, sync_env_file
                from core.llm import reset_llm_client

                apply_runtime_keys(patches)
                sync_env_file(patches)
                reset_llm_client()
                return _assistant_message(
                    f"Applied runtime keys: {', '.join(patches)}",
                    skill="update_config",
                )
            except Exception as exc2:
                return _assistant_message(
                    f"Config update failed: {exc} / {exc2}",
                    skill="update_config",
                )

    if skill == "generate_doc":
        rid = params.get("run_id") or focus
        snap = _harness_snapshot(rid)
        ov = snap["overview"]
        fmt = params.get("format") or "markdown"
        body = summarize_run_fallback(ov)
        # Enrich with Gemini when available
        try:
            from core.llm import call_llm

            enriched = call_llm(
                "Write a clear operator-facing research run summary document in "
                f"{fmt}. Use only this JSON context:\n{json.dumps(ov, default=str)[:8000]}",
                temperature=0.3,
                tier="strong",
                max_tokens=2500,
            )
            if enriched and len(enriched) > 80:
                body = enriched
        except Exception:
            pass
        artifact = {
            "name": f"run-{(ov.get('run_id') or 'summary')}.{ 'md' if fmt == 'markdown' else fmt }",
            "format": fmt,
            "content": body,
        }
        return _assistant_message(
            f"Generated a **{fmt}** summary for `{ov.get('run_id') or '—'}`.",
            skill="generate_doc",
            artifacts=[artifact],
            cards=_cards_for_snapshot(snap, include_approval=False),
        )

    # Default conversational path
    snap = _harness_snapshot(focus)
    history = chat_session.list_messages(sid)
    reply_text = _llm_reply(text, history, snap)
    return _assistant_message(
        reply_text,
        skill="chat",
        cards=_cards_for_snapshot(snap),
        thinking="Reading shared harness state",
    )


async def _kickoff_from_pairs(sid: str, pairs: List[Dict[str, str]], note: str) -> Dict[str, Any]:
    from web import app as webapp

    started = []
    errors = []
    # Only one live run at a time in this harness — start the first, queue note for rest.
    first, rest = pairs[0], pairs[1:]
    payload = webapp.RunPayload(
        domain=first["domain"],
        goal=first["topic"],
        paper_type="auto",
        depth="Standard",
        hunt_mode="try_and_report",
    )
    err = webapp.validate_run_payload(payload)
    if err:
        return _assistant_message(
            f"Parsed {len(pairs)} domain/topic pairs, but can't start: {err}",
            skill="upload_pack",
        )
    try:
        result = webapp.start_research(payload)
        if isinstance(result, JSONResponse):
            body = json.loads(result.body.decode("utf-8"))
            errors.append(body.get("detail"))
        else:
            run_id = result.get("run_id")
            started.append({"run_id": run_id, **first})
            chat_session.set_focus_run(sid, run_id)
    except Exception as exc:
        errors.append(str(exc))

    lines = [f"Parsed **{len(pairs)}** domain/topic pairs from upload."]
    if started:
        lines.append(f"Started `{started[0]['run_id']}` for **{first['domain']}** / {first['topic']}.")
    if rest:
        lines.append(
            f"{len(rest)} additional pairs are attached in `remaining-pairs.json`. "
            "This harness starts one run at a time; start those pairs individually after this run."
        )
    if errors:
        lines.append("Errors: " + "; ".join(str(e) for e in errors))
    snap = _harness_snapshot(started[0]["run_id"] if started else None)
    return _assistant_message(
        "\n".join(lines),
        skill="upload_pack",
        thinking="Processing domain/topic list",
        cards=_cards_for_snapshot(snap),
        artifacts=[{
            "name": "remaining-pairs.json",
            "format": "json",
            "content": json.dumps(rest, indent=2),
        }] if rest else [],
    )


async def _upload_as_pack(sid: str, name: str, raw: bytes, content_type: str) -> Dict[str, Any]:
    """Persist upload via the existing packs endpoint contract."""
    from core.config import config
    from core.packs import (
        build_pack_manifest,
        packs_dir,
        save_pack,
        validate_case_pack_file,
        validate_codebook_file,
        validate_corpus_override,
        validate_response_table_file,
    )

    safe_name = Path((name or "upload.bin").replace("\\", "/")).name[:255] or "upload.bin"
    lower = safe_name.lower()
    if lower.endswith(".zip"):
        field = "case_pack"
        size_limit = CHAT_CASE_PACK_MAX_BYTES
    elif lower.endswith(".json") and "corpus" in lower:
        field = "corpus_override"
        size_limit = CHAT_UPLOAD_MAX_BYTES
    elif "codebook" in lower:
        field = "codebook"
        size_limit = CHAT_UPLOAD_MAX_BYTES
    elif lower.endswith((".csv", ".json", ".jsonl")):
        field = "response_table"
        size_limit = CHAT_UPLOAD_MAX_BYTES
    else:
        return _assistant_message(
            f"Unsupported pack file `{safe_name}`. Upload a response table, codebook, corpus JSON, or case-pack ZIP.",
            skill="upload_pack",
        )
    if len(raw) > size_limit:
        return _assistant_message(
            f"Upload `{safe_name}` exceeds the {size_limit // (1024 * 1024)} MB limit.",
            skill="upload_pack",
        )

    stored_path: Optional[Path] = None
    pack_dir: Optional[Path] = None
    registry_path: Optional[Path] = None
    try:
        pack_root = packs_dir(config.output_dir)
        pack_id = uuid.uuid4().hex[:8]
        pack_dir = pack_root / pack_id
        pack_dir.mkdir(parents=True, exist_ok=False)
        stored_path = pack_dir / f"{uuid.uuid4().hex}{Path(safe_name).suffix.lower()}"
        stored_path.write_bytes(raw)

        if field == "response_table":
            response_result = validate_response_table_file(stored_path)
            if not response_result.get("ok"):
                raise ValueError("; ".join(response_result.get("reasons") or ["invalid response table"]))
            response_result["manifest"]["name"] = safe_name
            response_result["manifest"]["path"] = str(stored_path)
            pack_manifest = build_pack_manifest(response_table=response_result)
        elif field == "codebook":
            codebook_result = validate_codebook_file(stored_path)
            if not codebook_result.get("ok"):
                raise ValueError("; ".join(codebook_result.get("reasons") or ["invalid codebook"]))
            codebook_result["manifest"]["name"] = safe_name
            codebook_result["manifest"]["path"] = str(stored_path)
            pack_manifest = build_pack_manifest(codebook=codebook_result)
        elif field == "case_pack":
            case_result = validate_case_pack_file(stored_path)
            if not case_result.get("ok"):
                raise ValueError("; ".join(case_result.get("reasons") or ["invalid case pack"]))
            case_result["manifest"]["name"] = safe_name
            case_result["manifest"]["path"] = str(stored_path)
            pack_manifest = build_pack_manifest(case_pack=case_result)
        else:
            corpus_result = validate_corpus_override(json.loads(raw.decode("utf-8")))
            if not corpus_result.get("ok"):
                raise ValueError("; ".join(corpus_result.get("reasons") or ["invalid corpus override"]))
            pack_manifest = build_pack_manifest(corpus_override=corpus_result)
            pack_manifest["corpus_override"]["path"] = str(stored_path)

        _, registry_path = save_pack(pack_manifest, config.output_dir, pack_id=pack_id)
        return _assistant_message(
            f"Stored and validated `{safe_name}` ({len(raw)} bytes) as pack material "
            f"(field `{field}`). Use pack id `{pack_id}` when starting a run.",
            skill="upload_pack",
            artifacts=[{"name": safe_name, "bytes": len(raw), "pack_id": pack_id, "manifest": pack_manifest}],
        )
    except Exception as exc:
        logger.warning("Chat pack upload failed for %s: %s", safe_name, exc, exc_info=True)
        if registry_path and registry_path.is_file():
            registry_path.unlink()
        if stored_path and stored_path.is_file():
            stored_path.unlink()
        if pack_dir and pack_dir.is_dir():
            pack_dir.rmdir()
        return _assistant_message(
            f"Could not register `{safe_name}` as a pack: {exc}",
            skill="upload_pack",
        )


async def _interpret_uploads(note: str, files: List[Dict[str, Any]]) -> Dict[str, Any]:
    text_sections = []
    media_parts = []
    for item in files:
        name = item["name"]
        mime_type = item["mime_type"]
        raw = item["raw"]
        if item["text"]:
            text_sections.append(f"--- {name} ---\n{item['text'][:12000]}")
            continue
        is_image = mime_type in {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/gif",
            "image/heic",
            "image/heif",
        }
        is_pdf = mime_type == "application/pdf" or Path(name).suffix.lower() == ".pdf"
        is_audio_or_video = mime_type.startswith(("audio/", "video/"))
        if not (is_image or is_pdf or is_audio_or_video):
            return _assistant_message(
                f"I couldn't analyze `{name}` ({mime_type or 'unknown file type'}). "
                "Attach an image, PDF, audio/video file, or text-based file.",
                skill="upload_pack",
            )
        if len(raw) > CHAT_MEDIA_MAX_BYTES:
            return _assistant_message(
                f"`{name}` is larger than the {CHAT_MEDIA_MAX_BYTES // (1024 * 1024)} MB inline media limit.",
                skill="upload_pack",
            )
        media_parts.append({"data": raw, "mime_type": "application/pdf" if is_pdf else mime_type})

    prompt = (
        "You are ScholarGraph chat. Answer the operator's request using the attached files as evidence. "
        "Be clear about uncertainty and do not claim to have read file content that was not provided.\n"
        f"Operator request: {note}\n"
        "Attached files:\n"
        + "\n".join(f"- {item['name']} ({item['mime_type']})" for item in files)
        + "\n"
    )
    if text_sections:
        prompt += "\nTEXT FILE CONTENTS:\n" + "\n\n".join(text_sections)
    try:
        from core.llm import call_llm

        reply = call_llm(
            prompt,
            temperature=0.3,
            tier="cheap",
            max_tokens=900,
            allow_empty=False,
            media_parts=media_parts or None,
        )
        return _assistant_message(
            reply,
            skill="upload_pack",
            thinking=f"Reading {len(files)} attached file{'s' if len(files) != 1 else ''}",
        )
    except Exception as exc:
        return _assistant_message(
            f"I couldn't analyze the attached file{'s' if len(files) != 1 else ''}: {exc}",
            skill="upload_pack",
        )
