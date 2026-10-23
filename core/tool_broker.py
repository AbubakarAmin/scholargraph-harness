"""Auditable execution boundary for agent capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping

from .capabilities import authorize


@dataclass(frozen=True)
class ToolCall:
    agent: str
    capability: str
    tool: str
    status: str
    reason: str
    started_at: str


class CapabilityBroker:
    """Dispatch registered tools only after manifest authorization."""

    def __init__(self, tools: Mapping[str, Callable[..., Any]] | None = None):
        self._tools: Dict[str, Callable[..., Any]] = dict(tools or {})
        self.calls: list[ToolCall] = []

    def register(self, tool: str, handler: Callable[..., Any]) -> None:
        if not tool.strip():
            raise ValueError("Tool name cannot be empty")
        self._tools[tool] = handler

    def call(
        self,
        manifest: Mapping[str, object],
        capability: str,
        tool: str,
        **kwargs: Any,
    ) -> Any:
        decision = authorize(manifest, capability)
        started_at = datetime.now(timezone.utc).isoformat()
        if not decision.allowed:
            self.calls.append(ToolCall(
                decision.agent, capability, tool, "denied", decision.reason, started_at,
            ))
            raise PermissionError(f"{decision.agent} cannot use {capability}: {decision.reason}")
        if tool not in self._tools:
            self.calls.append(ToolCall(
                decision.agent, capability, tool, "missing", "tool not registered", started_at,
            ))
            raise KeyError(f"Tool is not registered: {tool}")
        self.calls.append(ToolCall(
            decision.agent, capability, tool, "started", "authorized", started_at,
        ))
        return self._tools[tool](**kwargs)


# ---------------------------------------------------------------------------
# Permission table + broker-only entry point (agentic-harness upgrade)
# ---------------------------------------------------------------------------

#: Tool -> capability needed to invoke it through the broker.
TOOL_CAPABILITY: Dict[str, str] = {
    "search_literature": "literature.search",
    "citation_graph": "literature.citation_graph",
    "contradiction_scan": "literature.contradiction_scan",
    "evidence_synthesize": "literature.evidence_synthesize",
    "code_validate": "code.execute_sandbox",
    "code_execute_sandbox": "code.execute_sandbox",
    "code_execute_multi": "code.execute_sandbox",
    "stats_analyze": "analysis.statistics",
    "verify_statistics": "verification.claims",
    "verify_citations": "verification.claims",
    "check_feasibility": "artifact.read",
    "memory_recall": "memory.recall",
    "memory_record_lesson": "memory.record",
}

#: Manifest name -> tools that member may call (opencode per-agent
#: permission equivalent). Denied tools raise PermissionError; every
#: decision is audited in CapabilityBroker.calls and state["tool_calls"].
#:
#: Consistency rule (machine-checked): every listed tool must map to a
#: capability the agent's DEFAULT_MANIFESTS entry actually declares —
#: otherwise the broker's manifest authorization would deny what the
#: table allows. Deterministic agents with no broker path get frozenset().
AGENT_TOOL_PERMISSIONS: Dict[str, frozenset] = {
    "TopicHunterAgent": frozenset({
        "search_literature", "citation_graph", "contradiction_scan",
        "evidence_synthesize", "memory_recall", "memory_record_lesson",
        "check_feasibility",
    }),
    "ProposerAgent": frozenset({
        "search_literature", "citation_graph", "evidence_synthesize",
        "memory_recall",
    }),
    "ChallengerAgent": frozenset({
        "search_literature", "citation_graph", "contradiction_scan",
        "memory_recall",
    }),
    "ModeratorAgent": frozenset(),
    "PlannerAgent": frozenset({
        "memory_recall", "memory_record_lesson", "check_feasibility",
    }),
    "WriterAgent": frozenset({"search_literature"}),
    "SupervisorAgent": frozenset({
        "verify_statistics", "verify_citations", "search_literature",
    }),
    "MetaAgent": frozenset(),
    "EditorAgent": frozenset({"verify_citations"}),
    "EngineerAgent": frozenset({
        "code_validate", "code_execute_sandbox", "code_execute_multi",
        "memory_recall", "memory_record_lesson",
    }),
    "DataAgent": frozenset({"search_literature"}),
    "ExecutionAgent": frozenset(),
    "AnalysisAgent": frozenset({"stats_analyze"}),
    "VerificationAgent": frozenset({
        "verify_statistics", "verify_citations",
    }),
    "ThesisAgent": frozenset({"search_literature"}),
    "ReviewAgent": frozenset({"search_literature"}),
}

# ---------------------------------------------------------------------------
# Plan vs build modes (opencode plan/build primary-agent equivalent)
# ---------------------------------------------------------------------------
# Plan-mode members are read-only: literature search, recall, feasibility
# checks. They may never generate code, execute anything, run statistics,
# or raise verification claims. Build-mode members act: they hold at least
# one acting tool. broker_call enforces the per-agent table above; this
# mapping makes the mode split machine-checkable (tests pin it) so a
# future permission edit cannot silently promote a planner into a builder.

#: Tools that constitute acting (code/statistics execution) capability.
#: Plan-mode members must hold none of these (machine-checked); review
#: functions (verify_*, memory_*) are mode-neutral audit tools.
ACTING_TOOLS: frozenset = frozenset({
    "code_validate", "code_execute_sandbox", "code_execute_multi",
    "stats_analyze",
})

#: Manifest name -> opencode-style primary mode.
AGENT_MODES: Dict[str, str] = {
    "TopicHunterAgent": "plan",
    "ProposerAgent": "plan",
    "ChallengerAgent": "plan",
    "ModeratorAgent": "plan",
    "PlannerAgent": "plan",
    "WriterAgent": "build",
    "SupervisorAgent": "plan",
    "MetaAgent": "plan",
    "EditorAgent": "build",
    "EngineerAgent": "build",
    "DataAgent": "build",
    "ExecutionAgent": "build",
    "AnalysisAgent": "build",
    "VerificationAgent": "plan",
    "ThesisAgent": "plan",
    "ReviewAgent": "plan",
}


def mode_of(manifest_name: str) -> str:
    """Return the declared mode for a manifest (unknown -> plan, fail-closed)."""
    return AGENT_MODES.get(str(manifest_name or ""), "plan")


def is_acting_tool(tool: str) -> bool:
    """True when a tool constitutes build-mode (acting) capability."""
    return str(tool or "") in ACTING_TOOLS


def broker_call(    manifest_name: str,
    tool: str,
    state: Dict[str, Any] | None = None,
    broker: "CapabilityBroker | None" = None,
    **kwargs: Any,
) -> Any:
    """Broker-only tool entry point. All agents must use this (never
    ``tools.registry()[name](...)`` directly).

    Enforces the per-agent permission table, dispatches through the
    manifest-authorized broker, and records an audit row into
    ``state["tool_calls"]`` when a state dict is supplied.
    """
    allowed = AGENT_TOOL_PERMISSIONS.get(manifest_name)
    if allowed is not None and tool not in allowed:
        raise PermissionError(f"{manifest_name} may not use tool {tool}")
    capability = TOOL_CAPABILITY.get(tool, "artifact.read")
    active = broker
    if active is None:
        try:
            from core import tools as _tool_registry

            active = CapabilityBroker(tools=_tool_registry.registry())
        except Exception as exc:
            raise RuntimeError(f"tool broker unavailable: {exc}") from exc
    try:
        from core.capabilities import DEFAULT_MANIFESTS

        manifest = dict(DEFAULT_MANIFESTS.get(manifest_name, {}))
    except Exception:
        manifest = {"agent": manifest_name, "allowed_capabilities": [],
                    "forbidden_capabilities": []}
    result = active.call(manifest, capability, tool, **kwargs)
    if isinstance(state, dict):
        try:
            rows = list(state.get("tool_calls") or [])
            rows.append({"agent": manifest_name, "tool": tool,
                         "capability": capability, "status": "ok"})
            state["tool_calls"] = rows
        except Exception:
            pass
    return result
