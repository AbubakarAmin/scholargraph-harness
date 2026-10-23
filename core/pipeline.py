"""Shared execution service for CLI and web research runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from .context import RunContext, activate_context, create_run_context, reset_context
from .state import ResearchState

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Final state and node count produced by one pipeline execution."""

    state: ResearchState
    nodes_seen: int


class ResearchPipeline:
    """Compile and stream a research graph with durable checkpoint settings."""

    def __init__(
        self,
        graph_factory: Callable[[], Any],
        checkpointer_factory: Callable[[], Any],
        recursion_limit: int = 1000,
        context: Optional[RunContext] = None,
    ) -> None:
        self._graph_factory = graph_factory
        self._checkpointer_factory = checkpointer_factory
        self._recursion_limit = recursion_limit
        self.context = context or create_run_context()

    def run(
        self,
        initial_state: Optional[ResearchState],
        run_id: str,
        resume: bool = False,
        on_node: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
        finalize: Optional[Callable[[ResearchState], None]] = None,
        on_crash: Optional[Callable[[Dict[str, Any], BaseException], None]] = None,
    ) -> PipelineResult:
        """Execute the graph, finalize artifacts, and complete the run tracker.

        *on_crash* runs with the accumulated state when the graph raises
        (e.g. an LLM outage mid-debate), so callers can persist a crash
        dossier instead of leaving zero durable output. It must never mask
        the original exception — failures inside it are logged and swallowed.
        """
        full_state = dict(initial_state) if initial_state else {}
        nodes_seen = 0
        _migrated = False
        try:
            for node_name, node_output in self.stream(initial_state, run_id, resume=resume):
                nodes_seen += 1
                full_state.update(node_output)
                # B4: resumed checkpoints predate schema v2 (no paper_type /
                # pipeline_recipe / phase_steps). Backfill once on the first
                # streamed output so old runs resume on the right recipe
                # instead of KeyError/wrong-spine defaults.
                if resume and not _migrated:
                    try:
                        from .state import migrate_state_v1_to_v2

                        if int(full_state.get("checkpoint_schema_version") or 1) < 2:
                            full_state = migrate_state_v1_to_v2(full_state)
                    except Exception:
                        pass
                    _migrated = True
                # M6 step accounting (§11-M6 step 2 + §8.5): graph-built phases
                # already carry phase_steps via the workflow wrapper; for
                # unwrapped graphs count here so every executed phase is
                # accounted exactly once.
                try:
                    if not isinstance(node_output, Mapping) or "phase_steps" not in node_output:
                        from .phase_registry import record_phase_step

                        record_phase_step(full_state, node_name)
                except Exception:
                    pass
                # Hard stop on recursion-budget exhaustion: stamp the legible
                # budget_exhausted terminal instead of running to LangGraph's
                # opaque recursion error (§6.2.5). Partial bundle retained.
                try:
                    from .phase_registry import apply_step_budget

                    if apply_step_budget(full_state, recursion_limit=self._recursion_limit):
                        logger.error(
                            "Step budget exhausted after %s node(s); stopping run %s",
                            nodes_seen, run_id,
                        )
                        break
                except Exception:
                    pass
                # Run-budget guard (§5.3, agentic-harness upgrade): tracker
                # counters were write-only — now llm_calls/tokens/usd/wall
                # trip a legible budget_exhausted terminal per node instead
                # of running past RUN_MAX_* silently.
                try:
                    from .pricing import budget_exhausted

                    try:
                        from .config import config as _cfg

                        _max_calls = int(getattr(_cfg, "run_max_llm_calls", 2000) or 2000)
                        _max_tokens = int(getattr(_cfg, "run_max_tokens", 8000000) or 8000000)
                        _max_usd = float(getattr(_cfg, "run_max_usd", 500.0) or 500.0)
                        _max_wall = int(getattr(_cfg, "run_wall_clock_sec", 21600) or 21600)
                    except Exception:
                        from .pricing import RUN_MAX_LLM_CALLS as _max_calls  # type: ignore[no-redef]
                        from .pricing import RUN_MAX_TOKENS as _max_tokens  # type: ignore[no-redef]
                        from .pricing import RUN_MAX_USD as _max_usd  # type: ignore[no-redef]
                        from .pricing import RUN_WALL_CLOCK_SEC as _max_wall  # type: ignore[no-redef]
                    _tracker = self.context.tracker if self.context else None
                    _stats = dict(getattr(_tracker, "stats", None) or {})
                    _calls = int(_stats.get("llm_calls") or 0)
                    _tokens = int(_stats.get("llm_tokens_in") or 0) + int(_stats.get("llm_tokens_out") or 0)
                    _usd = float(_stats.get("cost_estimate_usd") or 0.0)
                    try:
                        import time as _time

                        from .run_log import get_tracker as _get_tracker

                        _active = _get_tracker()
                        _start = getattr(_active, "started_at", None) if _active else None
                        from datetime import datetime as _dt, timezone as _tz

                        _wall = ((_dt.now(_tz.utc) - _dt.fromisoformat(str(_start).replace("Z", "+00:00"))).total_seconds()
                                 if _start else 0.0)
                    except Exception:
                        _wall = 0.0
                    _tripped, _reason = budget_exhausted(
                        llm_calls=_calls, tokens_total=_tokens, usd_spent=_usd,
                        wall_sec=_wall, max_calls=_max_calls,
                        max_tokens=_max_tokens, max_usd=_max_usd,
                        max_wall_sec=_max_wall)
                    if _tripped:
                        # 119ce81c: all four dims shared one generic label, so a
                        # wall-clock trip printed as calls/tokens/usd and looked
                        # like a call-budget trip. Name the tripped dimension(s)
                        # explicitly; keep the locked `budget_exhausted` prefix
                        # so taxonomy tests (startswith) still pass.
                        _dims = []
                        if _calls >= _max_calls:
                            _dims.append(f"calls={_calls}/{_max_calls}")
                        if _tokens >= _max_tokens:
                            _dims.append(f"tokens={_tokens}/{_max_tokens}")
                        if _usd >= _max_usd:
                            _dims.append(f"usd={_usd:.2f}/{_max_usd:.2f}")
                        if _wall >= _max_wall:
                            _dims.append(f"wall={_wall:.0f}s/{_max_wall}s")
                        _dim_str = ",".join(_dims) if _dims else (_reason or "run cap")
                        full_state["terminal_error"] = (
                            f"budget_exhausted ({_dim_str}): "
                            f"calls={_calls} tokens={_tokens} usd={_usd:.2f} wall={_wall:.0f}s"
                        )
                        full_state["current_phase"] = "complete"
                        full_state["should_continue"] = False
                        logger.error("Run budget exhausted (%s), stopping run %s", _dim_str, run_id)
                        break
                except Exception:
                    pass
                if on_node:
                    on_node(node_name, full_state)
            if finalize:
                finalize(full_state)
            if self.context.tracker:
                # A candidate paper is a valid product outcome even when it
                # still needs human approval.  Approval controls release naming,
                # not whether the harness produced a usable artifact.
                self.context.tracker.complete(
                    success=bool(full_state.get("latex_output"))
                    and not full_state.get("terminal_error")
                )
            return PipelineResult(full_state, nodes_seen)
        except Exception as exc:
            if self.context.tracker:
                self.context.tracker.complete(success=False)
            if on_crash is not None:
                try:
                    on_crash(full_state, exc)
                except Exception:
                    logger.exception("on_crash handler failed; original error preserved")
            raise

    def stream(
        self,
        initial_state: Optional[ResearchState],
        run_id: str,
        resume: bool = False,
    ) -> Iterator[tuple[str, Mapping[str, Any]]]:
        """Yield each workflow node output for a new or resumed run."""
        self.context.run_id = run_id
        token = activate_context(self.context)
        try:
            graph = self._graph_factory()
            app = graph.compile(checkpointer=self._checkpointer_factory())
            config = {
                "configurable": {"thread_id": run_id},
                "recursion_limit": self._recursion_limit,
            }
            source = None if resume else initial_state
            for event in app.stream(source, config):
                for node_name, node_output in event.items():
                    if node_name != "__end__":
                        yield node_name, node_output
        finally:
            reset_context(token)
