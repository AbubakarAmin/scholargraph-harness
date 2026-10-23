"""Chat session store — separate from dashboard view state.

Chat history and UI session live here. Underlying harness run state is still
owned by run_log / research_db / the live pipeline thread (shared with the
Control Deck dashboard).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional


_lock = threading.Lock()
_sessions: Dict[str, Dict[str, Any]] = {}


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def create_session() -> Dict[str, Any]:
    sid = _new_id()
    session = {
        "id": sid,
        "created_at": time.time(),
        "updated_at": time.time(),
        "messages": [],
        "focus_run_id": None,
        "pending_approvals": [],
    }
    with _lock:
        _sessions[sid] = session
    return dict(session)


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        s = _sessions.get(session_id)
        return dict(s) if s else None


def list_sessions() -> List[Dict[str, Any]]:
    with _lock:
        return [
            dict(session)
            for session in sorted(
                _sessions.values(),
                key=lambda item: item["updated_at"],
                reverse=True,
            )
        ]


def ensure_session(session_id: Optional[str] = None) -> Dict[str, Any]:
    if session_id:
        existing = get_session(session_id)
        if existing:
            return existing
    return create_session()


def append_message(session_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
    with _lock:
        s = _sessions.get(session_id)
        if not s:
            raise KeyError(session_id)
        msg = dict(message)
        msg.setdefault("id", _new_id())
        msg.setdefault("ts", time.time())
        s["messages"].append(msg)
        # Cap history so sessions stay light.
        if len(s["messages"]) > 200:
            s["messages"] = s["messages"][-160:]
        s["updated_at"] = time.time()
        return dict(msg)


def set_focus_run(session_id: str, run_id: Optional[str]) -> None:
    with _lock:
        s = _sessions.get(session_id)
        if not s:
            raise KeyError(session_id)
        s["focus_run_id"] = run_id
        s["updated_at"] = time.time()


def list_messages(session_id: str, limit: int = 80) -> List[Dict[str, Any]]:
    with _lock:
        s = _sessions.get(session_id)
        if not s:
            return []
        return list(s["messages"][-limit:])


def set_pending_approvals(session_id: str, approvals: List[Dict[str, Any]]) -> None:
    with _lock:
        s = _sessions.get(session_id)
        if not s:
            raise KeyError(session_id)
        s["pending_approvals"] = list(approvals)
        s["updated_at"] = time.time()
