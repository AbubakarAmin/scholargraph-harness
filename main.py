"""Composition root for the multi-agent research system."""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from langgraph.graph import StateGraph

from core.artifacts import save_results as write_results
from core.config import config, validate_config
from core.context import create_run_context, get_active_context
from core.memory import memory
from core.pipeline import ResearchPipeline
from core.run_log import get_tracker, start_run
from core.state import ResearchState, initialize_state
from core.utils import log_agent_action
from core.verification import reproducibility_dossier
from core.workflow import create_research_graph as build_research_graph
from core import workflow_nodes
from core.workflow_nodes import (
    case_protocol_node,
    causal_audit_node,
    editing_node,
    data_validation_node,
    engineering_node,
    hypothesis_debate_node,
    independent_validation_node,
    intake_node,
    is_valid_plan,
    meta_evaluation_node,
    planning_node,
    reset_node,
    should_continue,
    should_reset,
    supervision_node,
    survey_analysis_node,
    systematic_review_node,
    terminal_planning_failure,
    thesis_argument_node,
    verify_result_numbers,
    write_narrative_sections,
)
from agents.editor import EditorAgent
from agents.engineer import EngineerAgent
from agents.hypothesis_debate import HypothesisDebateSystem
from agents.meta_agent import MetaAgent
from agents.planner import PlannerAgent
from agents.supervisor import SupervisorAgent
from agents.topic_hunter import ResearchSourceUnavailable, TopicHunterAgent
from agents.writer import WriterAgent

from core.logging_config import setup_logging, attach_run_log, detach_run_log

setup_logging(config.log_level)
logger = logging.getLogger(__name__)


def topic_discovery_node(state: ResearchState) -> ResearchState:
    """Compatibility wrapper that keeps monkeypatchable agent imports working."""
    _original = workflow_nodes.TopicHunterAgent
    workflow_nodes.TopicHunterAgent = TopicHunterAgent
    try:
        return workflow_nodes.topic_discovery_node(state)
    finally:
        workflow_nodes.TopicHunterAgent = _original


def write_results_sections(state: ResearchState) -> ResearchState:
    """Compatibility wrapper that keeps monkeypatchable agent imports working."""
    _original = workflow_nodes.WriterAgent
    workflow_nodes.WriterAgent = WriterAgent
    try:
        return workflow_nodes.write_results_sections(state)
    finally:
        workflow_nodes.WriterAgent = _original


def research_nodes() -> dict:
    """Phase node mapping for graph construction (M1: includes thesis_argument).

    New recipe phases wire up here as they land (M1–M4); the typed builder
    skips recipe phases still missing from this mapping.
    """
    return {
        "intake": intake_node,
        "topic_discovery": topic_discovery_node,
        "hypothesis_debate": hypothesis_debate_node,
        "planning": planning_node,
        "terminal_planning_failure": terminal_planning_failure,
        "data_validation": data_validation_node,
        "writing_narrative": write_narrative_sections,
        "engineering": engineering_node,
        "independent_validation": independent_validation_node,
        "writing_results": write_results_sections,
        "supervision": supervision_node,
        "meta_evaluation": meta_evaluation_node,
        "editing": editing_node,
        "reset": reset_node,
        "thesis_argument": thesis_argument_node,
        "systematic_review": systematic_review_node,
        "survey_analysis": survey_analysis_node,
        "case_protocol": case_protocol_node,
        "causal_audit": causal_audit_node,
        "should_reset": should_reset,
        "should_continue": should_continue,
        "is_valid_plan": is_valid_plan,
    }


def create_research_graph() -> StateGraph:
    """Create the LangGraph workflow from the extracted phase nodes."""
    return build_research_graph(research_nodes())


def create_checkpointer():
    """Create the on-disk checkpoint store used by both CLI and web runs.

    M0: WAL + timeout + explicit close discipline. Callers should use as
    context manager or call .close() via close_checkpointer().
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "Durable checkpoints require langgraph-checkpoint-sqlite; install requirements.txt."
        ) from exc
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3
    conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False, timeout=15.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=15000;")
    except Exception:
        pass
    return SqliteSaver(conn)


def close_checkpointer(saver) -> None:
    """Close the underlying sqlite connection when the saver exposes it."""
    for attr in ("conn", "_conn", "connection"):
        conn = getattr(saver, attr, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            return


def save_results(state: ResearchState, output_dir: str = None):
    """Compatibility wrapper for the artifact service."""
    write_results(state, output_dir)


def report_provider_reachability() -> dict:
    """Warn-only startup probe report: one line per provider (R14 §6.1).

    Uses only ``gateway.is_available()`` (circuit-breaker state, no network
    I/O beyond what that already does). Never raises — safe keyless/offline.
    Returns ``{provider: reachable_bool}`` (``{}`` when even the gateway is
    unavailable).
    """
    try:
        from core.api_gateway import get_gateway

        gateway = get_gateway()
        providers = list(getattr(gateway, "DEFAULT_RATES", {}) or {})
        if not providers:
            providers = ["arxiv", "openalex", "semantic_scholar", "llm", "embedding"]
        report = {}
        for provider in providers:
            try:
                reachable = bool(gateway.is_available(provider))
            except Exception:
                reachable = False
            report[provider] = reachable
            status = "reachable" if reachable else "unavailable"
            logger.info("provider reachability: %s=%s", provider, status)
            print(f"   Provider {provider}: {status}")
        return report
    except Exception as exc:
        logger.warning("Provider reachability probe skipped: %s", exc)
        return {}


def handle_crash(state: ResearchState, exc: BaseException) -> None:
    """Persist a crash dossier when the graph raises mid-run.

    Without this, an outage 39 minutes in (e.g. LLM connection death during
    debate) leaves zero durable output. The sqlite checkpointer already holds
    per-node state for --resume; the dossier records where we stopped and
    how to continue. Never raises.
    """
    try:
        from core.artifacts import save_failure_dossier
        if isinstance(state, dict) and not state.get("terminal_error"):
            phase = state.get("current_phase") or "unknown phase"
            state["terminal_error"] = f"Unhandled crash in {phase}: {exc}"
        run_id = state.get("run_id") if isinstance(state, dict) else None
        dossier = save_failure_dossier(state, config.output_dir)
        logger.error("Crash dossier saved to %s", dossier)
        if run_id:
            logger.error("Resume with: python main.py --resume %s", run_id)
    except Exception:
        logger.exception("Failed to write crash dossier")


def main():
    """Run the research system from the command line."""
    import signal
    import atexit

    # Parse CLI flags first (before banner/config): --help must work without
    # keys and on consoles that cannot render the banner emoji (M3).
    parser = argparse.ArgumentParser(description="Run or resume a ScholarGraph research workflow")
    parser.add_argument("--resume", metavar="RUN_ID", help="resume a durable checkpoint by run id")
    # M3 pack intake (§5.2 + §11-M3 step 4): field packs from disk.
    parser.add_argument("--response-table", dest="response_table", metavar="PATH",
                        help="survey response_table (.csv/.json/.jsonl, <=50MB)")
    parser.add_argument("--codebook", dest="codebook", metavar="PATH",
                        help="survey codebook.csv (item,target,scale,subgroup_preregistered)")
    parser.add_argument("--case-pack", dest="case_pack", metavar="PATH",
                        help="case_pack.zip (<=200MB, >=2 source types)")
    parser.add_argument("--paper-type", dest="paper_type", metavar="TYPE",
                        help="paper type (auto + 10 IDs); default from config")
    # M6 ledger backup/rotation (§11-M6 step 3, §12): offline, no keys needed.
    parser.add_argument("--backup-ledger", dest="backup_ledger", action="store_true",
                        help="copy research ledger to .bak.<ts> + rotate (keep --backup-keep newest)")
    parser.add_argument("--backup-keep", dest="backup_keep", metavar="N", type=int, default=5,
                        help="backups to keep with --backup-ledger (default 5)")
    args = parser.parse_args()
    if getattr(args, "backup_ledger", False):
        from core.ledger_backup import backup_ledger as _backup_ledger
        dest = _backup_ledger(keep=args.backup_keep)
        print(f"Ledger backup: {dest}" if dest else "No live ledger yet - nothing to back up")
        return 0

    print("🤖 Multi-Agent AI Research System")
    print("=" * 50)
    try:
        validate_config()
        print("✅ Configuration validated")
        # R14 (§6.1): warn-only startup probe report — breaker state only,
        # never fatal (stays silent-safe offline/keyless).
        try:
            report_provider_reachability()
        except Exception as exc:  # belt-and-suspenders; the probe never raises by contract
            logger.warning("Provider reachability probe failed: %s", exc)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.draft_versions_dir, exist_ok=True)
        os.makedirs("memory/vector_db", exist_ok=True)
        print("✅ Output directories created")

        # Fix 10: Detect and clean stale 'running' runs from prior crashes
        try:
            from core.research_db import research_db
            stale_runs = research_db.get_stale_running_runs()
            if stale_runs:
                print(f"⚠️  Found {len(stale_runs)} stale 'running' run(s) from prior sessions — marking as interrupted")
                for sr in stale_runs:
                    research_db.finish_run(sr["run_id"], "interrupted", "unknown", {
                        "terminal_error": "Process terminated before completion",
                    })
        except Exception as e:
            logger.debug("Stale run check failed: %s", e)

        tracker = start_run(args.resume)
        attach_run_log(tracker.run_id)

        # Fix 9: Graceful shutdown — mark run as interrupted on SIGINT/SIGTERM
        _shutdown_marked = {"done": False}
        def _shutdown_handler(signum, frame):
            if _shutdown_marked["done"]:
                return
            _shutdown_marked["done"] = True
            sig_name = signal.Signals(signum).name
            logger.warning("Received %s — marking run as interrupted", sig_name)
            try:
                from core.research_db import research_db
                research_db.update_run_summary(tracker.run_id, {
                    "status": "interrupted",
                    "terminal_error": f"Received {sig_name}",
                })
            except Exception:
                pass
            detach_run_log(tracker.run_id)
            print(f"\n⚠️  Run interrupted by {sig_name}. State saved for resume.")
            sys.exit(1)
        signal.signal(signal.SIGINT, _shutdown_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _shutdown_handler)
        pipeline = ResearchPipeline(
            create_research_graph,
            create_checkpointer,
            context=create_run_context(tracker),
        )
        initial_state = initialize_state()
        initial_state["run_id"] = args.resume or tracker.run_id
        # M3 pack intake: validate CLI pack files now (named reasons, no
        # spend) and stash lightweight refs for intake_node to merge.
        cli_pack: dict = {}
        if getattr(args, "response_table", None):
            from core.packs import validate_response_table_file as _validate_rt
            rt = _validate_rt(args.response_table)
            if not rt.get("ok"):
                print(f"❌ Response table invalid: {'; '.join(rt.get('reasons') or ['invalid'])}")
                return 1
            cli_pack["response_table"] = {"path": str(args.response_table),
                                          "manifest": rt.get("manifest")}
            print(f"✅ Response table: {rt.get('rows')} rows (truncated={rt.get('truncated')})")
        if getattr(args, "codebook", None):
            from core.packs import validate_codebook_file as _validate_cb
            cb = _validate_cb(args.codebook)
            if not cb.get("ok"):
                print(f"❌ Codebook invalid: {'; '.join(cb.get('reasons') or ['invalid'])}")
                return 1
            cli_pack["codebook"] = {"path": str(args.codebook), "manifest": cb.get("manifest"),
                                    "targets": cb.get("targets"),
                                    "preregistered_subgroups": cb.get("preregistered_subgroups")}
            print(f"✅ Codebook: {len(cb.get('items') or [])} items")
        if getattr(args, "case_pack", None):
            from core.packs import validate_case_pack_file as _validate_cp
            cp = _validate_cp(args.case_pack)
            if not cp.get("ok"):
                print(f"❌ Case pack invalid: {'; '.join(cp.get('reasons') or ['invalid'])}")
                return 1
            cli_pack["case_pack"] = {"path": str(args.case_pack), "manifest": cp.get("manifest"),
                                     "entries": cp.get("entries"),
                                     "distinct_types": cp.get("distinct_types")}
            print(f"✅ Case pack: {len(cp.get('entries') or [])} entries {cp.get('distinct_types')}")
        if cli_pack:
            initial_state["intake_report"] = dict(initial_state.get("intake_report") or {})
            initial_state["intake_report"]["pack_manifest"] = cli_pack
            for key in ("response_table", "codebook", "case_pack"):
                if key in cli_pack:
                    initial_state[key] = cli_pack[key]  # type: ignore[index]
        if getattr(args, "paper_type", None):
            initial_state["paper_type"] = args.paper_type
        print(f"✅ Initial state created (run_id={tracker.run_id})")
        print("\n🚀 Starting research workflow...")
        print("=" * 50)

        def print_node_progress(node_name, node_output):
            state = node_output
            print(f"\n📋 {node_name.upper()}")
            print(f"   Phase: {state['current_phase']}")
            print(f"   Iteration: {state['iteration']}")
            if state["selected_topic"]:
                print(f"   Topic: {state['selected_topic']['title']}")
            if state["topics"] and node_name == "topic_discovery":
                print(f"   Topics Found: {len(state['topics'])}")
                print(f"   Best Topic: {state['topics'][0]['title']}")
            if state["debate_results"] and node_name == "hypothesis_debate":
                last_debate = state["debate_results"][-1]
                print(f"   Debates Completed: {len(state['debate_results'])}")
                print(f"   Last Debate: {'PASSED' if last_debate.passed else 'FAILED'}")
                if last_debate.passed:
                    print(f"   Selected Topic: {state['selected_topic']['title']}")
                else:
                    print(f"   Topics Remaining: {len(state['topics'])}")
            if node_name == "reset":
                print(f"   System Reset - New Iteration: {state['iteration']}")
            if state["supervisor_scores"]:
                average = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
                print(f"   Avg Score: {average:.2f}")
            if state["draft_sections"]:
                print(f"   Sections: {len(state['draft_sections'])}")

        from core.artifacts import PHASE_SNAPSHOT_NODES, save_phase_snapshot

        def on_node_with_snapshot(node_name, full_state):
            print_node_progress(node_name, full_state)
            if node_name in PHASE_SNAPSHOT_NODES:
                save_phase_snapshot(full_state, initial_state["run_id"], node_name)

        result = pipeline.run(
            initial_state,
            initial_state["run_id"],
            resume=bool(args.resume),
            on_node=on_node_with_snapshot,
            finalize=save_results,
            on_crash=handle_crash,
        )
        state = result.state
        print("\n" + "=" * 50)
        print("🎉 Research workflow completed!")
        if state["latex_output"]:
            paper_name = "paper_output.tex" if state.get("human_approved", False) else "paper_candidate.tex"
            print(f"📄 LaTeX paper generated: {config.output_dir}/{paper_name}")
        if state["supervisor_scores"]:
            average = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
            print(f"📊 Final average score: {average:.2f}")
        print(f"📁 All outputs saved to: {config.output_dir}")
    except Exception as exc:
        logger.error(f"Research system failed: {exc}", exc_info=True)
        print(f"❌ Error: {exc}")
        return 1
    finally:
        if tracker:
            detach_run_log(tracker.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
