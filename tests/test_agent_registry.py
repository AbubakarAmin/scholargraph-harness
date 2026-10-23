"""Agent-registry completeness: every Agent class must be registered for the dashboard.

The Agents view renders ``GET /api/dashboard`` → ``capabilities`` which is
``core.capabilities.DEFAULT_MANIFESTS``. If a new agent class is added under
``agents/`` without a manifest entry, it is invisible in the UI and
un-auditable — this test fails the build in that case (proposal §4.3, M5 UX).
Offline, no API keys, no network.
"""

from __future__ import annotations

import importlib
import pkgutil

import agents as _agents_pkg
from core.capabilities import CAPABILITIES, DEFAULT_MANIFESTS


def _agent_class_names() -> set[str]:
    names: set[str] = set()
    for mod in pkgutil.iter_modules(_agents_pkg.__path__):
        module = importlib.import_module(f"agents.{mod.name}")
        for attr, obj in vars(module).items():
            if (
                isinstance(obj, type)
                and attr.endswith("Agent")
                and obj.__module__ == module.__name__
            ):
                names.add(attr)
    # Documented role aliases (``XxxAgent = XxxAgent`` re-exports point at the
    # same class object, so they are invisible to the scan above).
    for alias in ("SurveyAnalysisAgent", "CaseProtocolAgent", "CausalAuditAgent"):
        names.add(alias)
    return names


def test_every_agent_class_has_a_manifest():
    missing = sorted(n for n in _agent_class_names() if n not in DEFAULT_MANIFESTS)
    assert not missing, (
        f"Agent classes without a DEFAULT_MANIFESTS entry (invisible in dashboard): {missing}"
    )


def test_manifests_have_dashboard_shapes():
    for name, manifest in DEFAULT_MANIFESTS.items():
        assert isinstance(manifest, dict), f"{name}: manifest must be a dict"
        for key in ("agent", "role", "allowed_capabilities", "forbidden_capabilities"):
            assert key in manifest, f"{name}: manifest missing {key!r}"
        allowed = set(manifest["allowed_capabilities"] or [])
        forbidden = set(manifest["forbidden_capabilities"] or [])
        assert allowed & forbidden == set(), f"{name}: capability both allowed and forbidden"
        unknown = (allowed | forbidden) - set(CAPABILITIES)
        assert not unknown, f"{name}: unknown capabilities {sorted(unknown)}"


def test_only_execution_agent_may_execute():
    holders = [
        name
        for name, manifest in DEFAULT_MANIFESTS.items()
        if "code.execute" in (manifest.get("allowed_capabilities") or [])
    ]
    assert holders == ["ExecutionAgent"], (
        f"only ExecutionAgent may hold code.execute (§4.3), got {holders}"
    )


def test_scoped_tokens_disjoint_and_sole_held():
    """Post-M6 review decision: development tokens are disjoint from their
    validation-grade counterparts, so no agent can iterate and self-validate.
    """
    from core.capabilities import CONFLICTING_CAPABILITY_PAIRS

    for scoped, validation_grade in CONFLICTING_CAPABILITY_PAIRS:
        for name, manifest in DEFAULT_MANIFESTS.items():
            allowed = set(manifest.get("allowed_capabilities") or [])
            assert not ({scoped, validation_grade} <= allowed), (
                f"{name} holds both {scoped} and {validation_grade}"
            )
    sandbox_holders = [
        name
        for name, manifest in DEFAULT_MANIFESTS.items()
        if "code.execute_sandbox" in (manifest.get("allowed_capabilities") or [])
    ]
    assert sandbox_holders == ["EngineerAgent"], (
        f"only EngineerAgent may hold code.execute_sandbox, got {sandbox_holders}"
    )
    brokered_holders = [
        name
        for name, manifest in DEFAULT_MANIFESTS.items()
        if "dataset.download_brokered" in (manifest.get("allowed_capabilities") or [])
    ]
    assert brokered_holders == ["EngineerAgent"], (
        f"only EngineerAgent may hold dataset.download_brokered, got {brokered_holders}"
    )


def test_expected_milestone_agents_registered():    # M1–M4 specialist roles must be visible in the dashboard registry.
    for name in (
        "ThesisAgent",
        "ReviewAgent",
        "SurveyAgent",
        "SurveyAnalysisAgent",
        "CaseAgent",
        "CaseProtocolAgent",
        "CausalAgent",
        "CausalAuditAgent",
    ):
        assert name in DEFAULT_MANIFESTS, f"{name} missing from dashboard registry"


def test_tool_capability_map_covers_registry():
    """Every broker tool must map to a known capability (no untyped dispatch)."""
    from core.capabilities import CAPABILITIES
    from core.tool_broker import TOOL_CAPABILITY
    from core.tools import registry

    for tool in registry():
        assert tool in TOOL_CAPABILITY, f"tool {tool!r} has no capability mapping"
        assert TOOL_CAPABILITY[tool] in CAPABILITIES, (
            f"tool {tool!r} maps to unknown capability {TOOL_CAPABILITY[tool]!r}"
        )


def test_broker_call_enforces_permission_table():
    """ModeratorAgent (no tools) is denied; audit row lands in state."""
    from core.tool_broker import broker_call

    state: dict = {}
    try:
        broker_call("ModeratorAgent", "search_literature", state, query="x")
    except PermissionError:
        pass
    else:
        raise AssertionError("ModeratorAgent must be denied search_literature")
    assert state.get("tool_calls", []) == [], "denied calls must not audit ok-rows"


def test_broker_only_enrich_audits_tool_calls():
    """Hunter enrich routes through the broker: state['tool_calls'] grows."""
    from agents.lab_society import enrich_topics_with_lab_graph

    state: dict = {}
    topics = [{"title": "t", "literature_evidence": [{"title": "p", "doi": "10/x"}]}]
    out = enrich_topics_with_lab_graph(topics, state)
    assert isinstance(out, list) and out, "enrich must return topics best-effort"
    tools_used = {r.get("tool") for r in (state.get("tool_calls") or [])}
    assert {"evidence_synthesize", "contradiction_scan", "citation_graph"} <= tools_used, (
        f"broker audit missing enrich calls, got {sorted(tools_used)}"
    )


def test_permission_table_matches_manifests():
    """Every broker-table tool must map to a capability its manifest declares."""
    from core.capabilities import DEFAULT_MANIFESTS
    from core.tool_broker import AGENT_TOOL_PERMISSIONS, TOOL_CAPABILITY

    for agent, tools in AGENT_TOOL_PERMISSIONS.items():
        assert agent in DEFAULT_MANIFESTS, f"{agent} has tools but no manifest"
        allowed = set(DEFAULT_MANIFESTS[agent].get("allowed_capabilities") or [])
        for tool in tools:
            cap = TOOL_CAPABILITY[tool]
            assert cap in allowed, (
                f"{agent}: table allows {tool} ({cap}) but manifest does not declare it"
            )


def test_plan_mode_holds_no_acting_tools():
    """Plan-mode members are read-only: no code/stats execution tools."""
    from core.tool_broker import ACTING_TOOLS, AGENT_MODES, AGENT_TOOL_PERMISSIONS, mode_of

    for agent, mode in AGENT_MODES.items():
        assert mode in ("plan", "build"), f"{agent}: unknown mode {mode}"
        assert agent in AGENT_TOOL_PERMISSIONS, f"{agent}: mode without tool entry"
        if mode == "plan":
            acting = set(AGENT_TOOL_PERMISSIONS[agent]) & set(ACTING_TOOLS)
            assert not acting, f"plan-mode {agent} holds acting tools {sorted(acting)}"
    assert mode_of("NoSuchAgent") == "plan"  # fail-closed default
    assert mode_of("EngineerAgent") == "build"


def test_debate_subagent_identities_match_broker_table():
    """Proposer/Challenger/Moderator class attrs must mirror the broker table."""
    from agents.hypothesis_debate import (ChallengerAgent, ModeratorAgent,
                                          ProposerAgent)
    from core.tool_broker import AGENT_TOOL_PERMISSIONS

    for cls in (ProposerAgent, ChallengerAgent, ModeratorAgent):
        name = cls.manifest_name
        assert name in AGENT_TOOL_PERMISSIONS, f"{name} missing from broker table"
        assert set(cls.allowed_tools) == set(AGENT_TOOL_PERMISSIONS[name]), (
            f"{name}: class tools {sorted(cls.allowed_tools)} != "
            f"table {sorted(AGENT_TOOL_PERMISSIONS[name])}"
        )
