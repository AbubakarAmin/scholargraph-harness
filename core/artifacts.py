"""Filesystem artifact serialization for completed research runs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from .config import config
from .ports import ArtifactPort
from .state import ResearchState

logger = logging.getLogger(__name__)


#: Graph nodes after which a small durable snapshot is written. These are the
#: expensive, non-reproducible phases (live literature + LLM debate); if the
#: process dies later, the snapshots preserve the useful work so far.
PHASE_SNAPSHOT_NODES = ("topic_discovery", "hypothesis_debate", "planning")


def _json_safe(value: Any) -> Any:
    """Round-trip through JSON so dataclasses/objects degrade to strings."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def save_phase_snapshot(
    state: ResearchState,
    run_id: Optional[str] = None,
    node_name: str = "unknown",
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """Persist a small JSON-safe snapshot after a phase-boundary node.

    Complements the LangGraph sqlite checkpointer (which enables --resume)
    with a human-inspectable record of topics / debate / plan produced so
    far. Cheap by design: titles only, bounded lists, never raw drafts.
    Returns the snapshot path, or None on failure (never raises).
    """
    try:
        if hasattr(state, "value"):
            state = state.value
        elif hasattr(state, "__dict__"):
            state = state.__dict__
        run_id = run_id or state.get("run_id") or "unknown"
        target_dir = Path(output_dir or config.output_dir) / "phase_snapshots" / str(run_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        topics = state.get("topics") or []
        snapshot = {
            "run_id": run_id,
            "node": node_name,
            "current_phase": state.get("current_phase"),
            "iteration": state.get("iteration", 0),
            "topic_count": len(topics),
            "topics": [
                t.get("title") if isinstance(t, dict) else str(t)
                for t in topics[:10]
            ],
            "selected_topic": _json_safe(state.get("selected_topic")),
            "debate_outcomes": [
                {
                    "topic": getattr(d, "topic", None) if not isinstance(d, dict) else d.get("topic"),
                    "passed": getattr(d, "passed", None) if not isinstance(d, dict) else d.get("passed"),
                    "score": getattr(d, "score", None) if not isinstance(d, dict) else d.get("score"),
                    "decision": getattr(d, "moderator_decision", None) if not isinstance(d, dict) else d.get("moderator_decision"),
                }
                for d in (state.get("debate_results") or [])[-5:]
            ],
            "plan_title": (state.get("plan") or {}).get("title") if isinstance(state.get("plan"), dict) else None,
            "plan_experiments": [
                e.get("name") if isinstance(e, dict) else str(e)
                for e in ((state.get("plan") or {}).get("experiments") or [])
            ],
            "terminal_error": state.get("terminal_error"),
        }
        path = target_dir / f"{node_name}.json"
        path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
        logger.info("Phase snapshot saved to %s", path)
        return str(path)
    except Exception as exc:
        logger.warning("Phase snapshot failed for node %s: %s", node_name, exc)
        return None


class FilesystemArtifactStore(ArtifactPort):
    """Store named artifacts below one controlled filesystem root."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, content: Any) -> str:
        target = (self.root / name).resolve()
        if self.root not in target.parents:
            raise ValueError("Artifact path must remain inside the configured root")
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        elif isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            with target.open("w", encoding="utf-8") as handle:
                json.dump(content, handle, indent=2, default=str)
        return str(target)


def save_results(state: ResearchState, output_dir: Optional[str] = None) -> None:
    """Write the final paper, plan, and operator-facing run summary."""
    try:
        target_dir = output_dir or config.output_dir
        os.makedirs(target_dir, exist_ok=True)

        if hasattr(state, "value"):
            state = state.value
        elif hasattr(state, "__dict__"):
            state = state.__dict__

        paper_path = None

        if state.get("terminal_error") or state.get("evidence_gate", {}).get("terminal"):
            dossier = save_failure_dossier(state, target_dir)
            logger.warning("Run stopped before release; failure dossier saved to %s", dossier)
            return

        # Machine-generated work is a candidate artifact even before a human
        # approval decision.  Previously this gate made successful graph runs
        # appear empty on disk, which made end-to-end operation impossible.
        # Keep the approval distinction: only approved work gets the canonical
        # release name, while every non-terminal completed run gets a reviewable
        # candidate.
        if state.get("latex_output"):
            latex_file = os.path.join(
                target_dir,
                "paper_output.tex" if state.get("human_approved", False) else "paper_candidate.tex",
            )
            with open(latex_file, "w", encoding="utf-8") as handle:
                handle.write(state["latex_output"])
            paper_path = latex_file
            logger.info("LaTeX output saved to %s", latex_file)

        if state.get("plan"):
            import yaml

            plan_file = os.path.join(target_dir, "plan.yaml")
            with open(plan_file, "w", encoding="utf-8") as handle:
                yaml.dump(state["plan"], handle, default_flow_style=False)
            logger.info("Plan saved to %s", plan_file)

        summary = {
            "iteration": state.get("iteration", 0),
            "selected_topic": state.get("selected_topic"),
            "sections_written": list(state.get("draft_sections", {}).keys()),
            "supervisor_scores": state.get("supervisor_scores", {}),
            "experiments_run": list(state.get("engineer_outputs", {}).keys()),
            "meta_feedback": [str(item) for item in state.get("meta_feedback", [])],
            "publishable": bool(state.get("human_approved", False)),
            "artifact_status": "released" if state.get("human_approved", False) else ("candidate" if paper_path else "none"),
            "paper_path": paper_path,
        }
        summary_file = os.path.join(target_dir, "research_summary.json")
        with open(summary_file, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Research summary saved to %s", summary_file)
    except Exception as exc:
        logger.error("Error saving results: %s", exc, exc_info=True)


def save_failure_dossier(state: ResearchState, output_dir: str) -> str:
    """Persist an operator-readable failure report without releasing a paper."""
    from .run_log import build_run_summary, get_tracker
    from .research_db import research_db

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary = build_run_summary(state, error=state.get("terminal_error"))
    run_id = state.get("run_id") or summary.get("run_id")
    dossier = {
        "status": "failed",
        "run_id": run_id,
        "terminal_error": state.get("terminal_error"),
        # A crash is recoverable: the sqlite checkpointer holds per-node
        # state under this thread id, and phase_snapshots/ holds the
        # expensive discovery/debate output. Tell the operator exactly that.
        "resume_hint": f"python main.py --resume {run_id}" if run_id else None,
        "phase_snapshots_dir": str(Path(output_dir) / "phase_snapshots" / str(run_id)) if run_id else None,
        "evidence_gate": state.get("evidence_gate", {}),
        "experiment_contracts": state.get("experiment_contracts", {}),
        "technical_failures": state.get("technical_failures", {}),
        "engineer_outputs": state.get("engineer_outputs", {}),
        "execution_artifacts": state.get("execution_artifacts", {}),
        "verification_findings": state.get("verification_findings", []),
        "summary": summary,
    }
    dossier_path = target_dir / "failure_dossier.json"
    dossier_path.write_text(json.dumps(dossier, indent=2, default=str), encoding="utf-8")
    summary_path = target_dir / "research_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    tracker = get_tracker()
    if tracker:
        research_db.record_artifact(
            tracker.run_id,
            "failure_dossier",
            str(dossier_path),
            {"reason": state.get("terminal_error"), "gate": state.get("evidence_gate", {})},
        )
        research_db.update_run_summary(
            tracker.run_id,
            {"status": "failed", "terminal_error": state.get("terminal_error"), "failure_dossier": str(dossier_path)},
        )
    return str(dossier_path)
