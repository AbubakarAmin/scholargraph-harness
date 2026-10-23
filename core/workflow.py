"""LangGraph graph construction for the ScholarGraph workflow.

Node implementations remain injectable so the composition root can migrate
without changing the state-machine behavior or public entry points.
"""

from typing import Callable, Mapping

from langgraph.graph import END, StateGraph

from .state import ResearchState

Node = Callable[[ResearchState], ResearchState]

# LangGraph stream cap (§2.2 "Recursion limit: 1000"). Mirrors
# ResearchPipeline's default; the per-phase step guard trips legibly just
# before the framework would fail opaquely (§6.2.5).
DEFAULT_RECURSION_LIMIT: int = 1000

# Keys in the nodes mapping that are router callbacks, not graph phases —
# never wrapped with step accounting.
_ROUTER_HELPERS: frozenset[str] = frozenset({"should_reset", "should_continue", "is_valid_plan"})


def _with_phase_accounting(name: str, fn: Node, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> Node:
    """Wrap a phase node with M6 per-phase step accounting (§11-M6 step 2).

    Records one step per successful execution into state["phase_steps"] and
    stamps a `budget_exhausted` terminal (via apply_step_budget) when the
    recursion budget is exceeded. Routing is otherwise untouched: flag-off
    runs record counts but take identical edges.
    """
    def _wrapped(state: ResearchState) -> ResearchState:
        out = fn(state)
        target = out if isinstance(out, dict) else state
        try:
            from .phase_registry import apply_step_budget, record_phase_step

            record_phase_step(target, name)
            apply_step_budget(target, recursion_limit=recursion_limit)
        except Exception:
            pass
        return out

    _wrapped.__name__ = getattr(fn, "__name__", f"accounted_{name}")
    return _wrapped  # type: ignore[return-value]


def _accounted_nodes(nodes: Mapping[str, Node], recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> dict[str, Node]:
    """Return the nodes mapping with phase nodes step-accounted."""
    wrapped: dict[str, Node] = {}
    for key, fn in nodes.items():
        if key in _ROUTER_HELPERS:
            wrapped[key] = fn
        else:
            try:
                wrapped[key] = _with_phase_accounting(key, fn, recursion_limit=recursion_limit)
            except Exception:
                wrapped[key] = fn
    return wrapped

# Canonical experimental spine (§3.3). create_research_graph shims here
# when TYPED_GRAPH_ENABLED=true; legacy wiring preserved when false.
EXPERIMENTAL_RECIPE: list[str] = [
    "intake",
    "topic_discovery",
    "hypothesis_debate",
    "planning",
    "data_validation",
    "writing_narrative",
    "engineering",
    "independent_validation",
    "writing_results",
    "supervision",
    "editing",
]

# Full canonical phase order covering every §3.3 recipe as a subsequence.
# Used by create_superset_graph so one production graph serves all 10 paper
# types: each phase routes at runtime to the next phase present in the
# state's pipeline_recipe (sequential-with-skip), instead of hardcoding the
# experimental spine. Verified subsequence for all PIPELINES entries.
FULL_CANONICAL_ORDER: list[str] = [
    "intake",
    "topic_discovery",
    "systematic_review",
    "thesis_argument",
    "hypothesis_debate",
    "planning",
    "data_validation",
    "survey_analysis",
    "case_protocol",
    "writing_narrative",
    "engineering",
    "causal_audit",
    "independent_validation",
    "writing_results",
    "supervision",
    "meta_evaluation",
    "editing",
]


def _next_recipe_phase(recipe: list[str] | None, current: str) -> str | None:
    """Next phase after `current` in the runtime recipe (None = run ends)."""
    order = list(recipe) if recipe else list(FULL_CANONICAL_ORDER)
    if current in order:
        idx = order.index(current)
        if idx + 1 < len(order):
            return order[idx + 1]
        return None
    # Phase outside the recipe (e.g. legacy state): fall back to canonical order.
    if current in FULL_CANONICAL_ORDER:
        idx = FULL_CANONICAL_ORDER.index(current)
        for nxt in FULL_CANONICAL_ORDER[idx + 1:]:
            if nxt in order:
                return nxt
    return None


def _dynamic_route(state: ResearchState, current: str) -> str:
    """Recipe-aware router for the production superset graph.

    Returns a node name, "end", "reset", or "terminal_planning_failure".
    Universal kill-switch first (mirrors _after_phase); then per-phase
    repair/back-edges preserved from the legacy wiring; otherwise advance to
    the next recipe phase (sequential-with-skip).
    """
    if (
        state.get("current_phase") == "complete"
        or state.get("terminal_error")
        or not state.get("should_continue", True)
    ):
        return "end"
    if state.get("should_reset"):
        return "reset"
    try:
        from .phase_registry import check_step_budget

        tripped, _ = check_step_budget(state)
        if tripped:
            return "end"
    except Exception:
        pass
    recipe = state.get("pipeline_recipe") or None

    def _advance() -> str:
        nxt = _next_recipe_phase(recipe, current)
        return nxt if nxt else "end"

    if current == "planning":
        if state.get("current_phase") == "planning" and state.get("plan_revision_requests"):
            return "planning"
        try:
            from .workflow_nodes import is_valid_plan

            if not is_valid_plan(state):
                return "terminal_planning_failure"
        except Exception:
            pass
        return _advance()
    if current == "engineering":
        if state.get("current_phase") == "planning":
            return "planning"
        return _advance()
    if current == "writing_results":
        if state.get("current_phase") == "writing_results":
            return "writing_results"
        return _advance()
    if current == "supervision":
        if state.get("current_phase") == "editing":
            return "editing"
        # Legacy behavior routes supervision through meta_evaluation unless
        # the node already approved into editing. No §3.3 recipe lists
        # meta_evaluation, but the node ships in every production mapping, so
        # preserve the loop unconditionally here (create_superset_graph always
        # wires it when provided; see the guard there).
        return "meta_evaluation"
    if current == "meta_evaluation":
        try:
            from .workflow_nodes import should_continue as _should_continue

            verdict = _should_continue(state)
        except Exception:
            verdict = "end"
        if verdict != "continue":
            return "end"
        # Re-enter drafting at the first drafting phase present in the recipe
        # (survey has no writing_narrative — stranding it there was a B2 bug).
        order = list(recipe) if recipe else list(FULL_CANONICAL_ORDER)
        for candidate in ("writing_narrative", "writing_results"):
            if candidate in order:
                return candidate
        return "end"
    if current == "editing":
        if (
            state.get("current_phase") == "writing_results"
            and (state.get("editor_repair_count") or 0) >= 1
            and (state.get("editor_repair_count") or 0) < 3
        ):
            return "writing_results"
        return "end"
    if current == "reset":
        order = list(recipe) if recipe else list(FULL_CANONICAL_ORDER)
        return "topic_discovery" if "topic_discovery" in order else (order[0] if order else "end")
    if current == "terminal_planning_failure":
        # P3: downgrade node now produces a bounded fallback plan and clears
        # the terminal — continue down the recipe instead of ending the run.
        if state.get("terminal_error") or state.get("current_phase") == "complete":
            return "end"
        nxt = _next_recipe_phase(recipe, "planning")
        return nxt if nxt else "end"
    return _advance()


def create_superset_graph(nodes: Mapping[str, Node], recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> StateGraph:
    """Build the production graph: all phases, recipe-aware dynamic routing.

    One static graph serves every paper type. Each phase's conditional edge
    consults the runtime ``pipeline_recipe`` (set by intake) and jumps to the
    next recipe phase, so non-experimental runs actually traverse their own
    recipes (thesis_argument, systematic_review, survey_analysis,
    case_protocol, causal_audit, meta loop) instead of the experimental spine.
    States without a recipe fall back to canonical order (legacy-compatible).
    """
    nodes = _accounted_nodes(nodes, recursion_limit=recursion_limit)
    present = [p for p in FULL_CANONICAL_ORDER if p in nodes]
    if not present:
        raise ValueError("create_superset_graph: no recipe phases present in nodes")
    workflow = StateGraph(ResearchState)
    for phase in present:
        workflow.add_node(phase, nodes[phase])
    for extra in ("reset", "terminal_planning_failure"):
        if extra in nodes and extra not in present:
            workflow.add_node(extra, nodes[extra])
            present.append(extra)
    if "reset" in nodes:
        workflow.add_edge("reset", present[0])

    # The supervision router always names meta_evaluation (legacy loop). If a
    # caller passes a mapping without it, alias the name to the recipe
    # successor so the edge target always exists. (Local name — never shadows
    # the module-global _dynamic_route.)
    _route_fn = _dynamic_route
    if "meta_evaluation" not in present:
        _base_route = _dynamic_route

        def _route_fn(state: ResearchState, current: str) -> str:
            verdict = _base_route(state, current)
            if verdict == "meta_evaluation":
                nxt = _next_recipe_phase(state.get("pipeline_recipe") or None, current)
                return nxt if nxt else "end"
            return verdict

    entry = "intake" if "intake" in present else present[0]
    workflow.set_entry_point(entry)
    # Every phase (and the terminal sink) can route anywhere: next recipe
    # phase, repair self-loops, reset, terminal sink, or END.
    targets: dict[str, str] = {p: p for p in present}
    targets["end"] = END
    for phase in [p for p in present if p not in ("reset", "terminal_planning_failure")]:
        def _route(state: ResearchState, _current: str = phase) -> str:
            return _route_fn(state, _current)

        workflow.add_conditional_edges(phase, _route, targets)
    if "terminal_planning_failure" in nodes:
        workflow.add_edge("terminal_planning_failure", END)
    return workflow


def _after_phase(recipe: list[str], state: ResearchState, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> str:
    """Generic per-phase router with universal kill-switch (§3.3).

    Returns "end" | "reset" | "continue". M6: a tripped step budget also
    routes to "end" (read-only check; the budget_exhausted terminal itself
    is stamped by the phase wrapper / pipeline guard).
    """
    if (
        state.get("current_phase") == "complete"
        or state.get("terminal_error")
        or not state.get("should_continue", True)
    ):
        return "end"
    if state.get("should_reset"):
        return "reset"
    try:
        from .phase_registry import check_step_budget

        tripped, _ = check_step_budget(state, recursion_limit=recursion_limit)
        if tripped:
            return "end"
    except Exception:
        pass
    return "continue"


def create_typed_graph(nodes: Mapping[str, Node], recipe: list[str] | None = None, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> StateGraph:
    """Build a recipe-scoped graph wiring only recipe edges (§3.3).

    Sequential execution only (PARALLEL_PHASES_ENABLED=false until M6).
    Universal: supervision → editing → END, * → reset, terminal → END.
    Nodes missing from `nodes` are skipped (new phases wire up in M1–M4).
    """
    active = list(recipe) if recipe else list(EXPERIMENTAL_RECIPE)
    # M6: step-accounted wrappers (recording only; routing unchanged).
    nodes = _accounted_nodes(nodes, recursion_limit=recursion_limit)
    # Keep only phases present in the nodes mapping (new M1–M4 phases wire up
    # as their nodes land); core known set always passes through.
    present = [p for p in active if p in nodes]
    workflow = StateGraph(ResearchState)
    for phase in present:
        workflow.add_node(phase, nodes[phase])
    # Ensure reset/terminal sinks exist when provided.
    for extra in ("reset", "terminal_planning_failure"):
        if extra in nodes and extra not in present:
            workflow.add_node(extra, nodes[extra])
            present.append(extra)

    if not present:
        raise ValueError("create_typed_graph: no recipe phases present in nodes")

    entry = present[0]
    workflow.set_entry_point(entry)

    # Linear chain through recipe order; each step honors kill-switch/reset.
    for i, phase in enumerate(present):
        if phase in ("reset", "terminal_planning_failure"):
            continue
        nxt = present[i + 1] if i + 1 < len(present) else END
        # Find next non-terminal phase for linear advance.
        j = i + 1
        while j < len(present) and present[j] in ("reset", "terminal_planning_failure"):
            j += 1
        nxt = present[j] if j < len(present) else END

        def _route(state: ResearchState, _recipe: list[str] = active) -> str:
            verdict = _after_phase(_recipe, state)
            if verdict == "end":
                return "end"
            if verdict == "reset":
                return "reset"
            return "continue"

        targets: dict[str, str] = {"continue": nxt if nxt != END else END, "end": END}
        if "reset" in nodes:
            targets["reset"] = "reset"
        else:
            # No reset node: collapse reset into end.
            def _route_no_reset(state: ResearchState, _recipe: list[str] = active) -> str:
                verdict = _after_phase(_recipe, state)
                return "end" if verdict in ("end", "reset") else "continue"

            workflow.add_conditional_edges(phase, _route_no_reset, {"continue": targets["continue"], "end": END})
            continue
        workflow.add_conditional_edges(phase, _route, targets)

    if "terminal_planning_failure" in nodes:
        workflow.add_edge("terminal_planning_failure", END)
    if "reset" in nodes:
        # reset loops to entry (topic_discovery when available, else entry).
        back = "topic_discovery" if "topic_discovery" in nodes else entry
        workflow.add_edge("reset", back)
    return workflow


def _typed_graph_enabled() -> bool:
    try:
        from .config import config as _cfg

        return bool(getattr(_cfg, "typed_graph_enabled", False))
    except Exception:
        return False


def create_research_graph(nodes: Mapping[str, Node], recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> StateGraph:
    """Build the research graph from its phase node implementations.

    Shim (§3.3): when TYPED_GRAPH_ENABLED=true delegates to the
    recipe-aware superset graph (all phases, dynamic per-run routing);
    when false preserves the legacy 13-node wiring byte-identically
    (M0 acceptance). create_typed_graph remains for explicit-recipe
    callers (tests, per-type matrix).
    """
    if _typed_graph_enabled() and "intake" in nodes:
        return create_superset_graph(nodes, recursion_limit=recursion_limit)
    # M6: step-accounted wrappers (recording only; legacy wiring unchanged).
    nodes = _accounted_nodes(nodes, recursion_limit=recursion_limit)
    workflow = StateGraph(ResearchState)

    workflow.add_node("topic_discovery", nodes["topic_discovery"])
    workflow.add_node("hypothesis_debate", nodes["hypothesis_debate"])
    workflow.add_node("planning", nodes["planning"])
    workflow.add_node("terminal_planning_failure", nodes["terminal_planning_failure"])
    workflow.add_node("data_validation", nodes["data_validation"])
    workflow.add_node("writing_narrative", nodes["writing_narrative"])
    workflow.add_node("engineering", nodes["engineering"])
    workflow.add_node("independent_validation", nodes["independent_validation"])
    workflow.add_node("writing_results", nodes["writing_results"])
    workflow.add_node("supervision", nodes["supervision"])
    workflow.add_node("meta_evaluation", nodes["meta_evaluation"])
    workflow.add_node("editing", nodes["editing"])
    workflow.add_node("reset", nodes["reset"])

    workflow.set_entry_point("topic_discovery")

    workflow.add_conditional_edges(
        "topic_discovery",
        nodes["should_reset"],
        {"reset": "reset", "continue": "hypothesis_debate", "end": END},
    )
    workflow.add_conditional_edges(
        "hypothesis_debate",
        nodes["should_reset"],
        {"reset": "reset", "continue": "planning", "end": END},
    )

    def _after_planning(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("should_reset"):
            return "reset"
        # planning_node may have requested a bounded schema repair.  Route back
        # to planning instead of sending the still-invalid state to the
        # terminal node immediately; otherwise the repair counter is dead code.
        if state.get("current_phase") == "planning" and state.get("plan_revision_requests"):
            return "repair"
        if not nodes["is_valid_plan"](state):
            return "invalid_plan"
        return "continue"

    workflow.add_conditional_edges(
        "planning",
        _after_planning,
        {"continue": "data_validation", "repair": "planning", "invalid_plan": "terminal_planning_failure", "reset": "reset", "end": END},
    )
    workflow.add_edge("terminal_planning_failure", END)

    def _after_data_validation(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "data_validation",
        _after_data_validation,
        {"continue": "writing_narrative", "reset": "reset", "end": END},
    )

    def _after_writing_narrative(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "writing_narrative",
        _after_writing_narrative,
        {"continue": "engineering", "reset": "reset", "end": END},
    )

    def _after_engineering(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("current_phase") == "planning":
            return "planning"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "engineering",
        _after_engineering,
        {"planning": "planning", "continue": "independent_validation", "reset": "reset", "end": END},
    )

    def _after_independent_validation(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "independent_validation",
        _after_independent_validation,
        {"continue": "writing_results", "reset": "reset", "end": END},
    )

    def _after_writing_results(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("current_phase") == "writing_results":
            return "redraft"
        if state.get("should_reset"):
            return "reset"
        return "continue"

    workflow.add_conditional_edges(
        "writing_results",
        _after_writing_results,
        {"redraft": "writing_results", "continue": "supervision", "reset": "reset", "end": END},
    )

    def _after_supervision(state: ResearchState) -> str:
        if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
            return "end"
        if state.get("current_phase") == "editing":
            return "editing"
        if state.get("should_reset"):
            return "reset"
        return "meta_evaluation"

    workflow.add_conditional_edges(
        "supervision",
        _after_supervision,
        {"editing": "editing", "meta_evaluation": "meta_evaluation", "reset": "reset", "end": END},
    )
    workflow.add_conditional_edges(
        "meta_evaluation",
        nodes["should_continue"],
        {"continue": "writing_narrative", "end": END},
    )
    def _after_editing(state: ResearchState) -> str:
        if (
            state.get("current_phase") == "writing_results"
            and (state.get("editor_repair_count") or 0) >= 1
            and (state.get("editor_repair_count") or 0) < 3
        ):
            return "repair"
        return "done"

    workflow.add_conditional_edges(
        "editing",
        _after_editing,
        {"repair": "writing_results", "done": END},
    )
    workflow.add_edge("reset", "topic_discovery")

    return workflow
