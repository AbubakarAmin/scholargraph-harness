"""
Append-only run scratchpad, progress events (for UI), and cross-run memory.
Replaces lossy JSON handoffs with durable source-of-truth logs.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config
from .research_db import research_db

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    """Recursively replace NaN/±Infinity floats with None.

    Scratchpads carry raw analysis dicts where a degenerate statistic (empty
    group, zero-variance t-test) yields ``float("nan")``. Python's ``json``
    happily *writes and reads* the non-standard ``NaN`` literal, but Starlette's
    ``JSONResponse`` serializes with ``allow_nan=False`` — so any endpoint
    serving such a row dies with ``ValueError: Out of range float values are
    not JSON compliant: nan``. Sanitizing at the write *and* read boundary keeps
    the JSONL strict-JSON and the API serializable (NaN becomes ``null``).
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(record), default=str)
    with _lock:
        with open(p, "a", encoding="utf-8") as f:
            f.write(payload + "\n")


#: Rotation caps for the append-only JSONL logs. These files accumulate across
#: runs (and previously across test sessions); without a cap, every forensic
#: read pays for the full history. Pruning keeps the newest records — the only
#: ones any consumer reads (read_events/read_scratchpad both tail the file).
EVENTS_PRUNE_MAX_BYTES = 2_000_000
EVENTS_PRUNE_KEEP_LINES = 5000


def prune_jsonl_file(
    path: str,
    max_bytes: int = EVENTS_PRUNE_MAX_BYTES,
    keep_lines: int = EVENTS_PRUNE_KEEP_LINES,
) -> int:
    """Trim a JSONL log to its newest *keep_lines* when over *max_bytes*.

    Returns the number of lines dropped (0 when untouched or on any error —
    rotation must never break a run). Callers invoke this at run start so
    the per-write path stays O(1).
    """
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size <= max_bytes:
            return 0
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) <= keep_lines:
            return 0
        dropped = len(lines) - keep_lines
        with _lock:
            with open(p, "w", encoding="utf-8") as f:
                f.writelines(lines[-keep_lines:])
        logger.info("Rotated %s: dropped %d old lines, kept %d", path, dropped, keep_lines)
        return dropped
    except Exception as exc:
        logger.debug("JSONL rotation skipped for %s: %s", path, exc)
        return 0


def filter_jsonl_by_run_id(path: str, run_id: str) -> int:
    """Remove all lines matching *run_id* from a JSONL file. Returns count removed."""
    p = Path(path)
    if not p.exists():
        return 0
    removed = 0
    with _lock:
        kept: List[str] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                    if record.get("run_id") == run_id:
                        removed += 1
                        continue
                except json.JSONDecodeError:
                    pass
                kept.append(line)
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(kept)
    return removed


class RunTracker:
    """Per-run observability + raw scratchpad shared by all agents."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.started_at = _now()
        self.ended_at: Optional[str] = None
        self.stats: Dict[str, Any] = {
            "rejected_topics": 0,
            "debate_rounds": 0,
            "sections_bounced": 0,
            "pivots": 0,
            "refines": 0,
            "plan_revisions": 0,
            "hard_check_fails": 0,
            "llm_calls": 0,
            "llm_failures": 0,
            "llm_tokens_in": 0,
            "llm_tokens_out": 0,
            "cost_estimate_usd": 0.0,
        }
        self.phase = "idle"
        self.status = "idle"  # idle | running | completed | failed
        self.messages: List[str] = []
        research_db.create_run(self.run_id, self.started_at)
        emit_event(
            "run_created",
            {"run_id": self.run_id, "started_at": self.started_at},
            run_id=self.run_id,
        )

    def set_phase(self, phase: str, detail: Optional[Dict[str, Any]] = None):
        self.phase = phase
        self.status = "running"
        emit_event(
            "phase_change",
            {"phase": phase, **(detail or {})},
            run_id=self.run_id,
            agent="Orchestrator",
        )

    def scratch(self, agent: str, kind: str, content: Any, meta: Optional[Dict] = None):
        """Append raw reasoning/code/results — not a compressed summary."""
        # Sanitize once so both the JSONL file and the ledger stay strict-JSON
        # (a nan stat would otherwise 500 every API endpoint that serves it).
        content = json_safe(content)
        meta = json_safe(meta) if meta else {}
        record = {
            "ts": _now(),
            "run_id": self.run_id,
            "agent": agent,
            "kind": kind,
            "content": content,
            "meta": meta or {},
        }
        _append_jsonl(config.run_log_path, record)
        research_db.record_scratch(self.run_id, agent, kind, content, meta or {})
        emit_event(
            "scratchpad",
            {
                "agent": agent,
                "kind": kind,
                "preview": str(content)[:240],
                "meta": meta or {},
            },
            run_id=self.run_id,
            agent=agent,
        )

    def bump(self, key: str, amount: int = 1):
        if key in self.stats:
            self.stats[key] += amount
        else:
            self.stats[key] = amount
        emit_event("stat", {"key": key, "value": self.stats.get(key)}, run_id=self.run_id)

    def message(self, text: str, level: str = "info"):
        self.messages.append(text)
        emit_event("message", {"text": text, "level": level}, run_id=self.run_id)

    def complete(self, success: bool = True):
        self.status = "completed" if success else "failed"
        self.ended_at = _now()
        emit_event(
            "run_complete",
            {"success": success, "stats": self.stats},
            run_id=self.run_id,
        )
        CrossRunMemory().record_run(
            {
                "run_id": self.run_id,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "success": success,
                "outcome_status": "released" if success else "unsupported",
                "stats": self.stats,
                "phase": self.phase,
            }
        )
        research_db.finish_run(self.run_id, self.status, self.phase, {"success": success, "stats": self.stats})

    def dashboard(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "phase": self.phase,
            "status": self.status,
            "stats": self.stats,
            "recent_messages": self.messages[-20:],
        }


_active_tracker: Optional[RunTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> Optional[RunTracker]:
    with _tracker_lock:
        return _active_tracker


def start_run(run_id: Optional[str] = None) -> RunTracker:
    """Start a new run or resume an existing one.

    When *run_id* is provided and matches a prior run in the research ledger,
    cumulative statistics (llm_calls, debate_rounds, pivots, etc.) are restored
    so that resumed runs continue counting from where they left off instead of
    resetting to zero.
    """
    global _active_tracker
    # Bound the append-only logs before this run adds to them.
    prune_jsonl_file(config.run_events_path)
    prune_jsonl_file(config.run_log_path)
    with _tracker_lock:
        _active_tracker = RunTracker(run_id=run_id)
        # On resume, restore cumulative stats from the prior run so accounting
        # (llm_calls, debate_rounds, pivots, etc.) continues correctly.
        if run_id:
            prior_run = research_db.list_runs(1000)
            match = next((r for r in prior_run if r.get("run_id") == run_id), None)
            if match:
                prior_stats = match.get("stats") or {}
                for key in _active_tracker.stats:
                    if key in prior_stats:
                        _active_tracker.stats[key] = prior_stats[key]
    return _active_tracker


def set_tracker(tracker: Optional[RunTracker]) -> None:
    global _active_tracker
    with _tracker_lock:
        _active_tracker = tracker


def emit_event(
    event_type: str,
    data: Dict[str, Any],
    run_id: Optional[str] = None,
    agent: Optional[str] = None,
):
    if run_id is None:
        from .context import get_active_context

        active_context = get_active_context()
        if active_context is not None:
            run_id = active_context.run_id or (
                active_context.tracker.run_id if active_context.tracker else None
            )
    # Guard: never write events without a run_id to the production log.
    # Test code that doesn't set up a tracker will have run_id=None; writing
    # those events into the shared JSONL pollutes forensics and historical
    # report reconstruction.
    if run_id is None:
        return
    record = {
        "ts": _now(),
        "type": event_type,
        "run_id": run_id,
        "agent": agent,
        "data": data,
    }
    _append_jsonl(config.run_events_path, record)
    research_db.record_event(event_type, data, run_id=run_id, agent=agent)


def read_events(limit: int = 200, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    db_rows = research_db.recent_events(run_id, limit)
    if db_rows:
        # Legacy rows were written before the write-side sanitizer existed.
        return [json_safe(row) for row in db_rows]
    path = Path(config.run_events_path)
    if not path.exists():
        return []

    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                if run_id and ev.get("run_id") != run_id:
                    continue
                events.append(json_safe(ev))
            except json.JSONDecodeError:
                continue
    return events[-limit:]


def read_scratchpad(limit: int = 100, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    path = Path(config.run_log_path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if run_id and row.get("run_id") != run_id:
                    continue
                # Legacy lines may hold NaN stats written before json_safe();
                # they parse (Python allows the NaN literal) but break the API.
                rows.append(json_safe(row))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def build_run_summary(
    state: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Operator-facing summary of what was tried, where we got, and why we stopped."""
    state = state or {}
    tracker = get_tracker()
    run_id = state.get("run_id") or (tracker.run_id if tracker else None)
    events = read_events(limit=80, run_id=run_id)
    scratch = read_scratchpad(limit=40, run_id=run_id)
    topic = state.get("selected_topic") or {}
    plan = state.get("plan") or {}
    eng = state.get("engineer_outputs") or {}
    exp_rows = []
    for name, out in eng.items():
        if not isinstance(out, dict):
            continue
        exp_rows.append({
            "name": name,
            "success": bool(out.get("success")),
            "error": (out.get("error") or "")[:300],
            "metrics": out.get("aggregate_metrics") or out.get("results", {}).get("metrics"),
            "decision_log": out.get("decision_log") or [],
            "raw_results_path": out.get("raw_results_path"),
            "branch_search": out.get("branch_search"),
        })
    phase = state.get("current_phase") or (tracker.phase if tracker else "idle")
    terminal = error or state.get("terminal_error")
    last_feedback = (state.get("meta_feedback") or [None])[-1]
    if isinstance(last_feedback, dict):
        import json as _json
        last_feedback = _json.dumps(last_feedback, default=str)
    stopped_because = (
        terminal
        or last_feedback
        or ("completed with paper" if state.get("latex_output") else None)
        or "run ended"
    )
    summary = {
        "run_id": run_id,
        "phase": phase,
        "status": "failed" if terminal or (tracker and tracker.status == "failed") else (
            "completed" if state.get("latex_output") else (tracker.status if tracker else "idle")
        ),
        "stopped_because": stopped_because,
        "topic": {
            "title": topic.get("title") or topic.get("name"),
            "gap": topic.get("gap") or topic.get("research_gap"),
            "feasibility": topic.get("feasibility"),
        },
        "plan": {
            "contributions": (plan.get("contributions") or plan.get("expected_contributions") or [])[:5],
            "experiments_planned": [
                (e.get("name") if isinstance(e, dict) else str(e))
                for e in (plan.get("experiments") or [])
            ],
        },
        "what_we_tried": {
            "iterations": state.get("iteration", 0),
            "debates": len(state.get("debate_results") or []),
            "sections_drafted": list((state.get("draft_sections") or {}).keys()),
            "experiments": exp_rows,
            "plan_revisions": state.get("plan_revision_requests") or [],
            "stats": tracker.stats if tracker else {},
        },
        "where_we_got": {
            "supervisor_scores": state.get("supervisor_scores") or {},
            "results_verification": state.get("results_verification") or {},
            "reproducibility": state.get("reproducibility") or {},
            "has_latex": bool(state.get("latex_output")),
            "has_paper": bool(state.get("final_paper") or state.get("draft_sections")),
        },
        "recent_events": [
            {
                "ts": e.get("ts"),
                "agent": e.get("agent"),
                "type": e.get("type"),
                "data": e.get("data"),
            }
            for e in events[-15:]
        ],
        "recent_scratch": [
            {"agent": s.get("agent"), "kind": s.get("kind"), "content": s.get("content")}
            for s in scratch[-10:]
        ],
        "meta_feedback": [str(item) for item in (state.get("meta_feedback") or [])],
        "error": terminal,
    }
    # Human paragraph for chat / notice banner
    lines = [
        f"Run `{run_id or '?'}` stopped in phase **{phase}**.",
        f"Reason: {stopped_because}",
    ]
    if summary["topic"].get("title"):
        lines.append(f"Topic: {summary['topic']['title']}")
    if exp_rows:
        ok_n = sum(1 for x in exp_rows if x["success"])
        lines.append(f"Experiments: {ok_n}/{len(exp_rows)} succeeded.")
        for x in exp_rows[:4]:
            if x["success"]:
                lines.append(f"  ✓ {x['name']}: {x.get('metrics')}")
            else:
                lines.append(f"  ✕ {x['name']}: {x.get('error') or 'failed'}")
    elif summary["plan"]["experiments_planned"]:
        lines.append("Experiments were planned but none completed successfully.")
    sections = summary["what_we_tried"]["sections_drafted"]
    if sections:
        lines.append(f"Draft sections: {', '.join(sections)}")
    summary["narrative"] = "\n".join(lines)
    return summary


class CrossRunMemory:
    """Persists failures/rejections across runs for Topic Hunter / Planner."""

    _instance: Optional["CrossRunMemory"] = None
    _init_lock = threading.Lock()

    def __init__(self, path: Optional[str] = None):
        with CrossRunMemory._init_lock:
            self.path = Path(path or config.cross_run_memory_path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write_lock = threading.Lock()

    def record(self, category: str, payload: Dict[str, Any]):
        entry = {
            "ts": _now(),
            "category": category,
            **payload,
        }
        with self._write_lock:
            _append_jsonl(str(self.path), entry)

    def record_rejection(self, kind: str, item: str, reason: str, meta: Optional[Dict] = None):
        self.record(
            "rejection",
            {"kind": kind, "item": item, "reason": reason, "rejection_reason": self._reason_tag(reason), "meta": meta or {}, "content_class": "structured_signal", "retrieval_eligible": True, "outcome_status": "rejected"},
        )

    def is_intra_run_collision(self, kind: str, item: str, current_run_id: str) -> bool:
        """Check if a rejection was already recorded in the SAME run (self-collision)."""
        if not current_run_id:
            return False
        for row in reversed(self.load("rejection", limit=50)):
            if row.get("kind") != kind:
                continue
            row_item = str(row.get("item") or "").strip()
            row_run = str((row.get("meta") or {}).get("run_id") or "").strip()
            if row_item == item and row_run == current_run_id:
                return True
        return False

    def excluded_topic_titles(self, limit: int = 100) -> List[str]:
        """Titles previously rejected or failed in debate — TopicHunter must not resurface them.

        P0-1: infra rejections (infra_llm_unavailable) are explicitly excluded:
        the topic never got a fair debate, so excluding it would permanently
        burn a valid hypothesis because the provider was down.
        """
        titles: List[str] = []
        seen = set()
        for row in self.load("rejection", limit=limit):
            if row.get("kind") != "topic":
                continue
            reason = str(row.get("reason") or "")
            if "infra" in reason.lower() or "llm_unavailable" in reason.lower():
                continue
            meta = row.get("meta") or {}
            if meta.get("infra_failure") is True:
                continue
            item = str(row.get("item") or "").strip()
            key = item.lower()
            if not item or key in seen:
                continue
            seen.add(key)
            titles.append(item)
        return titles

    def record_pivot(self, experiment: str, reason: str, meta: Optional[Dict] = None):
        self.record("pivot", {"experiment": experiment, "reason": reason, "failure_category": self._reason_tag(reason), "meta": meta or {}, "content_class": "structured_signal", "retrieval_eligible": True, "outcome_status": "inconclusive"})

    def record_plan_revision(self, reason: str, meta: Optional[Dict] = None):
        self.record("plan_revision", {"reason": reason, "revision_reason": self._reason_tag(reason), "meta": meta or {}, "content_class": "structured_signal", "retrieval_eligible": True, "outcome_status": "revised"})

    @staticmethod
    def _reason_tag(reason: str) -> str:
        lowered = str(reason or "").lower()
        if any(token in lowered for token in ("debate", "hypothesis")):
            return "failed_debate"
        if any(token in lowered for token in ("feasib", "sandbox", "gpu", "download")):
            return "feasibility"
        if any(token in lowered for token in ("novel", "similar", "saturated")):
            return "novelty"
        if any(token in lowered for token in ("claim", "metric", "result", "schema", "invalid_experiment")):
            return "evidence_consistency"
        if any(token in lowered for token in ("api", "dependency", "import")):
            return "dependency_or_api"
        if "missing_structured_hypothesis" in lowered or "missing" in lowered:
            return "formalize_failure"
        if "dataset_not_catalogued" in lowered or "dataset" in lowered:
            return "dataset_gap"
        if "previously_failed_or_rejected" in lowered or "duplicate" in lowered:
            return "self_collision"
        if "unsupported_research_gap" in lowered or "corpus" in lowered or "literature" in lowered:
            return "unsupported_gap"
        if "bridge" in lowered:
            return "bridge_mismatch"
        if "seed_exhausted" in lowered:
            return "seed_exhausted"
        return "other"

    def record_run(self, summary: Dict[str, Any]):
        self.record("run_summary", summary)

    def load(self, category: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if category and row.get("category") != category:
                        continue
                    rows.append(row)
                except json.JSONDecodeError:
                    continue
        return rows[-limit:]

    def get_prompt_context(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Return structured cross-run tags without raw reasoning text."""
        context = []
        for entry in self.load(limit=limit * 3):
            if entry.get("content_class") != "structured_signal" or not entry.get("retrieval_eligible"):
                continue
            signal = {
                "category": entry.get("category"),
                "kind": entry.get("kind"),
                "item": entry.get("item"),
                "experiment": entry.get("experiment"),
                "rejection_reason": entry.get("rejection_reason"),
                "failure_category": entry.get("failure_category"),
                "revision_reason": entry.get("revision_reason"),
                "outcome_status": entry.get("outcome_status"),
            }
            context.append({key: value for key, value in signal.items() if value is not None})
            if len(context) >= limit:
                break
        return context

    def lessons_for_prompt(self, limit: int = 15) -> str:
        """Compatibility formatter over structured tags, never raw reasons."""
        rejections = self.load("rejection", limit=limit)
        pivots = self.load("pivot", limit=8)
        revisions = self.load("plan_revision", limit=8)
        lines = ["## Lessons from prior runs (avoid repeating these failures)"]
        for r in rejections[-10:]:
            tag = r.get("rejection_reason", "unknown")
            prefix = "FAILED DEBATE" if tag == "failed_debate" else "REJECTED"
            lines.append(f"- {prefix} {r.get('kind')}: {r.get('item')} — tag={tag}")
        for p in pivots[-5:]:
            lines.append(f"- PIVOT on {p.get('experiment')}: tag={p.get('failure_category', 'unknown')}")
        for rev in revisions[-5:]:
            lines.append(f"- PLAN REVISION: tag={rev.get('revision_reason', 'unknown')}")
        if len(lines) == 1:
            return "No prior-run lessons yet."
        return "\n".join(lines)

    def clear(self):
        with _lock:
            if self.path.exists():
                self.path.write_text("", encoding="utf-8")

    def get_negative_result_lessons(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Feature 6: query rejected topics with structured hypotheses, returning
        only structured hypothesis fields — never raw draft/narrative text.

        Reads from rejection records (which now include structured_hypothesis
        from the topic that failed debate). Fail-closed: any query error → return [].
        """
        try:
            entries = self.load("rejection", limit=limit * 5)
            negative = []
            for entry in entries:
                if entry.get("kind") != "topic":
                    continue
                meta = entry.get("meta") or {}
                hyp = meta.get("structured_hypothesis") or {}
                if not hyp:
                    continue
                negative.append({
                    "hypothesis_kind": hyp.get("hypothesis_kind", ""),
                    "research_question": hyp.get("research_question", ""),
                    "falsification_condition": hyp.get("falsification_condition", ""),
                    "dependent_variables": hyp.get("dependent_variables", []),
                    "outcome_status": "unsupported",
                })
                if len(negative) >= limit:
                    break
            return negative
        except Exception as e:
            logger.warning("Failed to load negative result lessons: %s", e)
            return []

