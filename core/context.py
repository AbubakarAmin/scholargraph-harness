"""Runtime dependencies shared by one research run."""

from __future__ import annotations

from dataclasses import dataclass
from contextvars import ContextVar, Token
from typing import Any, Dict, Optional

from .config import Config, config as default_config
from .memory import memory as default_memory
from .research_db import research_db as default_research_db
from .ports import ResearchLedgerPort, VectorMemoryPort
from .run_log import RunTracker
from .capabilities import DEFAULT_MANIFESTS


@dataclass
class RunContext:
    """Dependencies and identity for a single pipeline execution."""

    config: Config
    memory: VectorMemoryPort
    research_db: ResearchLedgerPort
    tracker: Optional[RunTracker] = None
    run_id: Optional[str] = None
    capability_manifests: Dict[str, Dict[str, Any]] | None = None


_active_context: ContextVar[Optional[RunContext]] = ContextVar(
    "scholargraph_run_context",
    default=None,
)


def create_run_context(tracker: Optional[RunTracker] = None) -> RunContext:
    """Create a context using the current application adapters."""
    return RunContext(
        config=default_config,
        memory=default_memory,
        research_db=default_research_db,
        tracker=tracker,
        run_id=tracker.run_id if tracker else None,
        capability_manifests=dict(DEFAULT_MANIFESTS),
    )


def activate_context(context: RunContext) -> Token:
    """Make a run context available to code created inside the pipeline."""
    return _active_context.set(context)


def reset_context(token: Token) -> None:
    """Restore the previous context after a pipeline stream completes."""
    _active_context.reset(token)


def get_active_context() -> Optional[RunContext]:
    """Return the context for the current pipeline execution, if any."""
    return _active_context.get()
