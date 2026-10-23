"""Build an auditable run report from the durable research ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .research_db import ResearchDatabase, research_db


def build_forensic_report(run_id: str, db: ResearchDatabase = research_db) -> Dict[str, Any]:
    runs = [run for run in db.list_runs(1000) if run.get("run_id") == run_id]
    run = runs[0] if runs else None
    claims = db.claims(run_id, 10000)
    artifacts = db.artifacts(run_id, 10000)
    events = db.recent_events(run_id, 10000)
    return {
        "run_id": run_id,
        "run": run,
        "events": events,
        "claims": claims,
        "artifacts": artifacts,
        "lineage": db.claim_lineage(run_id, 10000),
        "unresolved_claims": [claim for claim in claims if claim.get("status") not in {"verified", "passed"}],
        "summary": {
            "event_count": len(events),
            "claim_count": len(claims),
            "artifact_count": len(artifacts),
        },
    }


def write_forensic_report(run_id: str, output_path: str, db: ResearchDatabase = research_db) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_forensic_report(run_id, db), indent=2, default=str), encoding="utf-8")
    return str(path)


def build_historical_report(db: ResearchDatabase = research_db) -> Dict[str, Any]:
    """Reconstruct the latest failed and completed runs for incident review."""
    runs = db.list_runs(1000)
    failed = next((item for item in runs if item.get("status") in {"failed", "technical_failure"}), None)
    completed = next((item for item in runs if item.get("status") in {"completed", "success"}), None)
    return {
        "failed_run": build_forensic_report(failed["run_id"], db) if failed else None,
        "completed_run": build_forensic_report(completed["run_id"], db) if completed else None,
        "replay_status": "available through replay_run.py when a companion repository exists",
    }


def write_historical_report(output_path: str, db: ResearchDatabase = research_db) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_historical_report(db), indent=2, default=str), encoding="utf-8")
    return str(path)
