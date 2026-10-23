"""Durable SQLite ledger for runs, provenance, and evidence-backed claims."""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config

logger = logging.getLogger(__name__)


class ResearchDatabase:
    """Small transactional store; JSONL remains an export, not the source of truth."""
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or config.research_db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._migrate()

    def _connect(self):
        con = sqlite3.connect(str(self.path), timeout=15)
        con.row_factory = sqlite3.Row
        return con

    def _migrate(self):
        with self._connect() as con:
            con.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS run_events (
              id INTEGER PRIMARY KEY, ts TEXT NOT NULL, run_id TEXT, agent TEXT,
              event_type TEXT NOT NULL, payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_run_ts ON run_events(run_id, id DESC);
            CREATE TABLE IF NOT EXISTS evidence_claims (
              id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, section_name TEXT NOT NULL,
              claim_text TEXT NOT NULL, claim_type TEXT NOT NULL, status TEXT NOT NULL,
              evidence_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claims_run ON evidence_claims(run_id, id DESC);
            CREATE TABLE IF NOT EXISTS research_artifacts (
              id INTEGER PRIMARY KEY, run_id TEXT, artifact_type TEXT NOT NULL,
              location TEXT, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_runs (
              run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
              status TEXT NOT NULL, phase TEXT, summary_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS run_scratchpad (
              id INTEGER PRIMARY KEY, ts TEXT NOT NULL, run_id TEXT NOT NULL,
              agent TEXT NOT NULL, kind TEXT NOT NULL, content_json TEXT NOT NULL,
              metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scratch_run ON run_scratchpad(run_id, id DESC);
            """)

    @staticmethod
    def _now(): return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_event(self, event_type: str, payload: Dict[str, Any], run_id: Optional[str] = None, agent: Optional[str] = None) -> bool:
        # D20-core: bool return + BUSY retry + surfaced counter.
        for attempt in range(3):
            try:
                with self.lock, self._connect() as con:
                    con.execute("INSERT INTO run_events(ts,run_id,agent,event_type,payload_json) VALUES(?,?,?,?,?)",
                                (self._now(), run_id, agent, event_type, json.dumps(payload, default=str)))
                return True
            except sqlite3.OperationalError as e:
                if "busy" in str(e).lower() and attempt < 2:
                    import time as _t

                    _t.sleep(0.05 * (attempt + 1))
                    continue
                logger.warning("DB record_event failed: %s", e, exc_info=True)
                self._bump_ledger_failures()
                return False
            except Exception as e:
                logger.warning("DB record_event failed: %s", e, exc_info=True)
                self._bump_ledger_failures()
                return False
        self._bump_ledger_failures()
        return False

    @staticmethod
    def _bump_ledger_failures() -> None:
        try:
            from .run_log import get_tracker

            tracker = get_tracker()
            if tracker:
                tracker.bump("ledger_failures")
        except Exception:
            pass

    def create_run(self, run_id: str, started_at: str) -> bool:
        try:
            with self.lock, self._connect() as con:
                existing = con.execute("SELECT summary_json FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
                if existing:
                    con.execute("UPDATE research_runs SET started_at=?, status=? WHERE run_id=?", (started_at, "running", run_id))
                else:
                    con.execute("INSERT INTO research_runs(run_id,started_at,status,summary_json) VALUES(?,?,?,?)", (run_id, started_at, "running", "{}"))
            return True
        except Exception as e:
            logger.warning("DB create_run failed: %s", e, exc_info=True)
            self._bump_ledger_failures()
            return False

    def finish_run(self, run_id: str, status: str, phase: str, summary: Dict[str, Any]) -> bool:
        try:
            with self.lock, self._connect() as con:
                con.execute("UPDATE research_runs SET ended_at=?, status=?, phase=?, summary_json=? WHERE run_id=?", (self._now(), status, phase, json.dumps(summary, default=str), run_id))
            return True
        except Exception as e:
            logger.warning("DB finish_run failed: %s", e, exc_info=True)
            self._bump_ledger_failures()
            return False

    def get_stale_running_runs(self) -> List[Dict[str, Any]]:
        """Find runs stuck in 'running' status (from prior crashes/kills)."""
        with self.lock, self._connect() as con:
            rows = con.execute(
                "SELECT run_id, started_at, summary_json FROM research_runs WHERE status='running'"
            ).fetchall()
            return [dict(r) for r in rows]

    def update_run_summary(self, run_id: str, updates: Dict[str, Any]):
        """Merge additional JSON-safe details into a completed run summary."""
        try:
            with self.lock, self._connect() as con:
                row = con.execute("SELECT summary_json FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
                if not row:
                    logger.debug("update_run_summary: run_id %s not found", run_id)
                    return
                try:
                    summary = json.loads(row["summary_json"] or "{}")
                except json.JSONDecodeError:
                    summary = {}
                summary.update(updates)
                con.execute(
                    "UPDATE research_runs SET summary_json=? WHERE run_id=?",
                    (json.dumps(summary, default=str), run_id),
                )
        except Exception as e:
            logger.warning("DB update_run_summary failed: %s", e)

    def claim_lineage(self, run_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Return claim-to-artifact mappings for operator diagnosis and export."""
        claims = self.claims(run_id, limit)
        artifacts = self.artifacts(run_id, limit)
        by_id = {str(item.get("metadata", {}).get("artifact_id")): item for item in artifacts}
        lineage = []
        for claim in claims:
            evidence = claim.get("evidence") or {}
            ids = evidence.get("artifact_ids") or evidence.get("artifact_id") or []
            if isinstance(ids, str):
                ids = [ids]
            lineage.append({**claim, "artifacts": [by_id[identifier] for identifier in ids if identifier in by_id]})
        return lineage

    def outcome_calibration(self, limit: int = 1000) -> Dict[str, Any]:
        """Summarize recorded outcomes for reward calibration and monitoring."""
        counts = {"positive": 0, "negative": 0, "inconclusive": 0}
        for run in self.list_runs(limit):
            summary = json.loads(run.get("summary_json") or "{}")
            outcomes = summary.get("outcomes") if isinstance(summary.get("outcomes"), list) else []
            for outcome in outcomes:
                if outcome in counts:
                    counts[outcome] += 1
        total = sum(counts.values())
        return {"counts": counts, "total": total, "rates": {key: value / total if total else 0.0 for key, value in counts.items()}, "reward_policy": "validated outcomes receive equal base credit"}

    def _record_with_retry(self, sql: str, args: tuple, what: str) -> bool:
        """Shared D20 insert path: 3x BUSY retry + surfaced counter, bool return."""
        import sqlite3 as _sqlite3

        for attempt in range(3):
            try:
                with self.lock, self._connect() as con:
                    con.execute(sql, args)
                return True
            except _sqlite3.OperationalError as e:
                if "busy" in str(e).lower() and attempt < 2:
                    import time as _t

                    _t.sleep(0.05 * (attempt + 1))
                    continue
                logger.warning("DB %s failed: %s", what, e, exc_info=True)
                self._bump_ledger_failures()
                return False
            except Exception as e:
                logger.warning("DB %s failed: %s", what, e, exc_info=True)
                self._bump_ledger_failures()
                return False
        self._bump_ledger_failures()
        return False

    def record_scratch(self, run_id: str, agent: str, kind: str, content: Any, metadata: Dict[str, Any]) -> bool:
        return self._record_with_retry(
            "INSERT INTO run_scratchpad(ts,run_id,agent,kind,content_json,metadata_json) VALUES(?,?,?,?,?,?)",
            (self._now(), run_id, agent, kind, json.dumps(content, default=str), json.dumps(metadata, default=str)),
            "record_scratch",
        )

    def record_artifact(self, run_id: Optional[str], artifact_type: str, location: Optional[str], metadata: Dict[str, Any]) -> bool:
        return self._record_with_retry(
            "INSERT INTO research_artifacts(run_id,artifact_type,location,metadata_json,created_at) VALUES(?,?,?,?,?)",
            (run_id, artifact_type, location, json.dumps(metadata, default=str), self._now()),
            "record_artifact",
        )

    def record_claim(self, run_id: str, section: str, text: str, claim_type: str, status: str, evidence: Dict[str, Any]) -> bool:
        return self._record_with_retry(
            "INSERT INTO evidence_claims(run_id,section_name,claim_text,claim_type,status,evidence_json,created_at) VALUES(?,?,?,?,?,?,?)",
            (run_id, section, text, claim_type, status, json.dumps(evidence, default=str), self._now()),
            "record_claim",
        )

    def recent_events(self, run_id: Optional[str], limit: int = 100) -> List[Dict[str, Any]]:
        query, args = ("SELECT * FROM run_events WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)) if run_id else ("SELECT * FROM run_events ORDER BY id DESC LIMIT ?", (limit,))
        with self._connect() as con:
            rows = con.execute(query, args).fetchall()
        return [{"ts": r["ts"], "type": r["event_type"], "run_id": r["run_id"], "agent": r["agent"], "data": json.loads(r["payload_json"])} for r in reversed(rows)]

    def claims(self, run_id: Optional[str], limit: int = 200) -> List[Dict[str, Any]]:
        if not run_id: return []
        with self._connect() as con:
            rows = con.execute("SELECT * FROM evidence_claims WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)).fetchall()
        return [{"section": r["section_name"], "claim": r["claim_text"], "type": r["claim_type"], "status": r["status"], "evidence": json.loads(r["evidence_json"])} for r in reversed(rows)]

    def artifacts(self, run_id: Optional[str], limit: int = 200) -> List[Dict[str, Any]]:
        if not run_id:
            return []
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM research_artifacts WHERE run_id=? ORDER BY id DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
        return [
            {
                "artifact_type": row["artifact_type"],
                "location": row["location"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]


    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM research_runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def delete_run(self, run_id: str):
        try:
            with self.lock, self._connect() as con:
                con.execute("DELETE FROM research_runs WHERE run_id=?", (run_id,))
                con.execute("DELETE FROM run_events WHERE run_id=?", (run_id,))
                con.execute("DELETE FROM run_scratchpad WHERE run_id=?", (run_id,))
                con.execute("DELETE FROM evidence_claims WHERE run_id=?", (run_id,))
                con.execute("DELETE FROM research_artifacts WHERE run_id=?", (run_id,))
        except Exception as e:
            logger.warning("DB delete_run failed: %s", e)

    def delete_run_comprehensive(self, run_id: str) -> dict:
        """Delete every trace of a run across all data stores.

        Returns a summary dict of what was cleaned.
        """
        from .run_log import filter_jsonl_by_run_id

        cleaned: dict = {}

        # 1. Research ledger (5 tables)
        self.delete_run(run_id)
        cleaned["research_ledger"] = True

        # 2. Checkpoints (LangGraph SqliteSaver tables)
        cleaned["checkpoints"] = self._delete_checkpoints(run_id)

        # 3. JSONL event/scratchpad logs
        cleaned["run_events_jsonl"] = filter_jsonl_by_run_id(config.run_events_path, run_id)
        cleaned["run_scratchpad_jsonl"] = filter_jsonl_by_run_id(config.run_log_path, run_id)

        # 4. Cross-run memory
        cleaned["cross_run_jsonl"] = filter_jsonl_by_run_id(config.cross_run_memory_path, run_id)

        # 5. Debate and feedback logs (JSON arrays)
        cleaned["debate_log"] = self._filter_json_array(config.debate_log_path, run_id)
        cleaned["feedback_log"] = self._filter_json_array(config.feedback_log_path, run_id)

        # 6. Raw results directory
        cleaned["raw_results"] = self._delete_raw_results(run_id)

        return cleaned

    def _delete_checkpoints(self, run_id: str) -> bool:
        """Delete LangGraph checkpoint rows for a given thread_id (= run_id)."""
        ckpt_path = Path(config.checkpoint_path)
        if not ckpt_path.exists():
            return False
        try:
            import sqlite3 as _sqlite3
            con = _sqlite3.connect(str(ckpt_path), timeout=10)
            try:
                cur = con.execute("DELETE FROM checkpoints WHERE thread_id=?", (run_id,))
                writes_deleted = cur.rowcount
                con.execute("DELETE FROM writes WHERE thread_id=?", (run_id,))
                con.commit()
                return writes_deleted > 0 or con.total_changes > 0
            finally:
                con.close()
        except Exception:
            return False

    @staticmethod
    def _filter_json_array(path_str: str, run_id: str) -> int:
        """Remove entries matching run_id from a JSON-array file. Returns count removed."""
        p = Path(path_str)
        if not p.exists():
            return 0
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return 0
            original_len = len(data)
            filtered = [entry for entry in data if entry.get("run_id") != run_id]
            removed = original_len - len(filtered)
            if removed > 0:
                p.write_text(json.dumps(filtered, indent=2, default=str), encoding="utf-8")
            return removed
        except (json.JSONDecodeError, OSError):
            return 0

    @staticmethod
    def _delete_raw_results(run_id: str) -> int:
        """Remove raw_results directories that reference this run_id."""
        raw_dir = Path(config.raw_results_dir)
        if not raw_dir.exists():
            return 0
        removed = 0
        for child in raw_dir.iterdir():
            if not child.is_dir():
                continue
            # Match directories named after the run_id or containing it
            if run_id in child.name:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
                continue
            # Check metadata files inside the directory for run_id references
            for meta_file in child.glob("*.json"):
                try:
                    text = meta_file.read_text(encoding="utf-8")
                    if run_id in text:
                        shutil.rmtree(child, ignore_errors=True)
                        removed += 1
                        break
                except OSError:
                    continue
        return removed

    def clear_all(self):
        with self.lock, self._connect() as con:
            con.execute("DELETE FROM research_runs")
            con.execute("DELETE FROM run_events")
            con.execute("DELETE FROM run_scratchpad")
            con.execute("DELETE FROM evidence_claims")
            con.execute("DELETE FROM research_artifacts")


research_db = ResearchDatabase()
