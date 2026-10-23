"""Lab society primitives: BaseLabMember + budgets + messages + claims.

This is the ground-up agent foundation for the ScholarGraph lab upgrade.
Every lab member (Hunter, Debater, Planner, Engineer, Writer, Supervisor...)
inherits BaseLabMember instead of calling call_llm with a bare persona prompt.

Design:
- perceive() builds a slim, scored observation (never a full state dump).
- act() is a ReAct step: Thought + ToolCall | Handoff | AskHelp | Final.
- observe() grounds tool results (parse, hash, ledger append).
- Budgets abort to fallback_used + limitation, never silent empty.
- All tool use goes through CapabilityBroker, never direct imports.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.utils import log_agent_action

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

@dataclass
class AgentBudget:
    """Per-agent execution budget. Trips legibly instead of hanging a run."""

    max_iters: int = 6
    max_llm_calls: int = 12
    max_tool_calls: int = 20
    timeout_s: float = 600.0

    llm_calls: int = 0
    tool_calls: int = 0
    started_at: float = field(default_factory=time.time)

    def check(self) -> Optional[str]:
        if self.llm_calls >= self.max_llm_calls:
            return f"llm budget exhausted ({self.llm_calls}/{self.max_llm_calls})"
        if self.tool_calls >= self.max_tool_calls:
            return f"tool budget exhausted ({self.tool_calls}/{self.max_tool_calls})"
        if (time.time() - self.started_at) > self.timeout_s:
            return f"wall-clock budget exhausted ({self.timeout_s}s)"
        return None

    def bump_llm(self) -> None:
        self.llm_calls += 1

    def bump_tool(self) -> None:
        self.tool_calls += 1


# ---------------------------------------------------------------------------
# Messages + claims (society bus)
# ---------------------------------------------------------------------------

MESSAGE_TYPES = frozenset({
    "propose", "challenge", "rebut", "handoff", "help_request",
    "help_offer", "review_report", "repair_request", "concede",
    "standup", "verification_report",
})

CLAIM_STATUSES = frozenset({"proposed", "supported", "falsified", "withdrawn"})


def make_message(
    frm: str,
    to: str,
    mtype: str,
    payload: Optional[Dict[str, Any]] = None,
    obs_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a lab-bus message. mtype must be in MESSAGE_TYPES."""
    if mtype not in MESSAGE_TYPES:
        raise ValueError(f"Unknown message type: {mtype}")
    return {
        "from": frm,
        "to": to,
        "type": mtype,
        "payload": dict(payload or {}),
        "obs_ids": list(obs_ids or []),
    }


def make_claim(
    claim_id: str,
    text: str,
    owner: str,
    evidence_ids: Optional[List[str]] = None,
    status: str = "proposed",
) -> Dict[str, Any]:
    if status not in CLAIM_STATUSES:
        raise ValueError(f"Unknown claim status: {status}")
    return {
        "claim_id": claim_id,
        "text": text,
        "owner": owner,
        "evidence_ids": list(evidence_ids or []),
        "status": status,
    }


def _obs_id(payload: Any) -> str:
    h = hashlib.sha256(repr(payload).encode("utf-8", errors="replace")).hexdigest()
    return f"obs_{h[:12]}"


# ---------------------------------------------------------------------------
# Base member
# ---------------------------------------------------------------------------

class BaseLabMember:
    """Shared ReAct + broker + ledger foundation for every lab member.

    Subclasses set:
      name: str (e.g. "Hunter")
      manifest_name: str (key into DEFAULT_MANIFESTS, e.g. "TopicHunterAgent")
      tools: list of tool names this member may call via the broker
    and implement perceive() / build_prompt().
    """

    name: str = "BaseMember"
    manifest_name: str = ""
    allowed_tools: List[str] = []

    def __init__(self, context=None, budget: Optional[AgentBudget] = None):
        from core.context import get_active_context

        self.context = context or get_active_context()
        self.budget = budget or AgentBudget()
        self._broker = None
        self._manifest: Dict[str, Any] = {}
        self._obs_log: List[Dict[str, Any]] = []
        self._init_manifest()

    # -- setup -------------------------------------------------------------
    def _init_manifest(self) -> None:
        try:
            from core.capabilities import DEFAULT_MANIFESTS

            if self.context and getattr(self.context, "capability_manifests", None):
                manifests = self.context.capability_manifests
            else:
                manifests = DEFAULT_MANIFESTS
            self._manifest = dict(manifests.get(self.manifest_name, {}))
            if not self._manifest and self.manifest_name:
                logger.warning("%s: no manifest found for %s", self.name, self.manifest_name)
        except Exception as exc:
            logger.debug("%s manifest load failed: %s", self.name, exc)
            self._manifest = {}

    def _broker_for_run(self):
        if self._broker is not None:
            return self._broker
        try:
            from core.tool_broker import CapabilityBroker
            from core import tools as _tool_registry

            handlers = _tool_registry.registry()
            self._broker = CapabilityBroker(tools=handlers)
        except Exception as exc:
            logger.debug("%s broker init failed: %s", self.name, exc)
            self._broker = None
        return self._broker

    # -- core loop helpers -------------------------------------------------
    def perceive(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build a slim observation. Subclasses override to select + score."""
        # Default: expose counts + selected topic title only, never full dumps.
        topics = state.get("topics") or []
        return {
            "n_topics": len(topics),
            "selected_topic": (state.get("selected_topic") or {}).get("title", ""),
            "phase": state.get("current_phase", ""),
            "messages_pending": len(state.get("messages") or []),
            "claims_open": len([c for c in (state.get("claims") or []) if c.get("status") == "proposed"]),
        }

    def call_tool(self, capability: str, tool: str, **kwargs: Any) -> Any:
        """Broker-enforced tool call. All lab members must use this."""
        reason = self.budget.check()
        if reason:
            raise RuntimeError(f"{self.name} budget tripped: {reason}")
        broker = self._broker_for_run()
        if broker is None:
            raise RuntimeError(f"{self.name}: tool broker unavailable")
        if self.allowed_tools and tool not in self.allowed_tools:
            raise PermissionError(f"{self.name} may not use tool {tool}")
        self.budget.bump_tool()
        result = broker.call(self._manifest, capability, tool, **kwargs)
        obs_id = _obs_id(result)
        self._obs_log.append({"obs_id": obs_id, "tool": tool, "capability": capability})
        log_agent_action(self.name, "tool_call", {"tool": tool, "obs_id": obs_id})
        return result

    def ask_help(self, state: Dict[str, Any], to: str, question: str, context_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        msg = make_message(self.name, to, "help_request", {"question": question}, context_ids)
        msgs = list(state.get("messages") or [])
        msgs.append(msg)
        state["messages"] = msgs
        log_agent_action(self.name, "ask_help", {"to": to, "q": question[:160]})
        self._persist_bus_event("lab_help_request", {"to": to, "question": question[:500]})
        return msg

    def handoff(self, state: Dict[str, Any], to: str, payload: Dict[str, Any], obs_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        msg = make_message(self.name, to, "handoff", payload, obs_ids)
        msgs = list(state.get("messages") or [])
        msgs.append(msg)
        state["messages"] = msgs
        log_agent_action(self.name, "handoff", {"to": to, "keys": sorted(payload.keys())})
        self._persist_bus_event("lab_handoff", {"to": to, "keys": sorted(payload.keys())})
        return msg

    def post_standup(self, state: Dict[str, Any], progress: str, blocker: str = "", need: str = "") -> Dict[str, Any]:
        msg = make_message(self.name, "Lab", "standup", {"progress": progress, "blocker": blocker, "need": need})
        msgs = list(state.get("messages") or [])
        msgs.append(msg)
        state["messages"] = msgs
        self._persist_bus_event("lab_standup", {"progress": progress[:500], "blocker": blocker[:200]})
        return msg

    def _persist_bus_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Fire-and-forget ledger write for lab-bus traffic."""
        try:
            from core.research_db import research_db
            from core.run_log import get_tracker

            tracker = get_tracker()
            research_db.record_event(
                event_type, payload,
                run_id=tracker.run_id if tracker else None, agent=self.name)
        except Exception as exc:
            logger.debug("%s bus persist failed: %s", self.name, exc)

    def inbox(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [m for m in (state.get("messages") or []) if m.get("to") in (self.name, "Lab", "All")]

    def obs_ids(self) -> List[str]:
        return [o["obs_id"] for o in self._obs_log]
