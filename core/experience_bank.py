"""Procedural memory: ExperienceBank for failure->fix rules.

Stores {failure_kind, context_sig, hint, wins, losses} in
memory/lessons.jsonl (append-only JSONL). Engineers and Planners read the
top hints for a failure_kind; wins promote, losses demote. This is the
Critic Experience Bank lite: counts + reusable hints, never raw narrative.

Fail-closed: missing/corrupt file -> empty bank, never a raise in hot path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

BANK_PATH = Path("memory/lessons.jsonl")


def _load_all() -> List[Dict[str, Any]]:
    if not BANK_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        for line in BANK_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    except Exception as exc:
        logger.debug("ExperienceBank load failed: %s", exc)
    return rows


def record_lesson(failure_kind: str = "", hint: str = "", context_sig: str = "") -> Dict[str, Any]:
    """Append one lesson. Never raises."""
    try:
        BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "failure_kind": (failure_kind or "unknown")[:120],
            "hint": (hint or "")[:800],
            "context_sig": (context_sig or "")[:200],
            "wins": 0,
            "losses": 0,
        }
        with open(BANK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def _sig_overlap(a: str, b: str) -> float:
    """Jaccard overlap of alphanumeric tokens (fuzzy context match)."""
    import re as _re

    def _toks(value: str) -> set:
        return {t for t in _re.findall(r"[a-z0-9]+", (value or "").lower()) if len(t) > 2}

    left, right = _toks(a), _toks(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def hints_for(failure_kind: str, limit: int = 3, context_sig: str = "") -> List[Dict[str, Any]]:
    """Top hints for a failure kind, ranked by (wins - losses).

    When ``context_sig`` is supplied, hints whose stored context_sig
    token-overlaps the query rank first (fuzzy match instead of the old
    exact-kind-only lookup).
    """
    rows = [r for r in _load_all() if r.get("failure_kind") == failure_kind]

    def _rank(row: Dict[str, Any]) -> tuple:
        base = int(row.get("wins", 0)) - int(row.get("losses", 0))
        bonus = _sig_overlap(context_sig, str(row.get("context_sig") or "")) if context_sig else 0.0
        return (bonus, base)

    rows.sort(key=_rank, reverse=True)
    return rows[: max(0, int(limit or 3))]


def record_trial(failure_kind: str, hint: str, context_sig: str, won: bool) -> Dict[str, Any]:
    """Record one trial outcome (unfreezes the bank).

    Appends a lesson row when no exact (failure_kind, hint) row exists,
    then bumps wins/losses on the matching row. Node-level retry
    success/failure must call this — otherwise wins/losses stay frozen
    at zero and cross-run learning never happens.
    """
    try:
        kind = (failure_kind or "unknown")[:120]
        h = (hint or "")[:800]
        rows = _load_all()
        match = next(
            (r for r in rows
             if r.get("failure_kind") == kind and r.get("hint") == h),
            None,
        )
        if match is None:
            rec = record_lesson(kind, h, context_sig)
            if not rec.get("ok"):
                return rec
        return record_outcome(kind, h, won)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def record_outcome(failure_kind: str, hint: str, won: bool) -> Dict[str, Any]:
    """Bump wins/losses for a matching hint (best-effort rewrite)."""
    try:
        rows = _load_all()
        updated = False
        for r in rows:
            if r.get("failure_kind") == failure_kind and r.get("hint") == hint:
                r["wins"] = int(r.get("wins", 0)) + (1 if won else 0)
                r["losses"] = int(r.get("losses", 0)) + (0 if won else 1)
                updated = True
                break
        if updated:
            BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
            BANK_PATH.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return {"ok": updated}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}
