"""Production-graph regression pins (post-review hardening).

Covers the critical finding that the TYPED_GRAPH_ENABLED production path
hardcoded the experimental recipe: the superset graph must serve every §3.3
recipe with dynamic per-run routing, and legacy waivers must hold.
"""

from core.phase_registry import PIPELINES
from core.state import initialize_state
from core.workflow import (
    FULL_CANONICAL_ORDER,
    _dynamic_route,
    _next_recipe_phase,
    create_superset_graph,
)


def _state(recipe):
    st = initialize_state()
    st["pipeline_recipe"] = list(recipe)
    st["plan_revision_requests"] = []
    return st


def test_every_recipe_is_canonical_subsequence():
    for paper_type, recipe in PIPELINES.items():
        idx = [FULL_CANONICAL_ORDER.index(p) for p in recipe]
        assert idx == sorted(idx), paper_type


def test_dynamic_routing_per_type():
    assert _dynamic_route(_state(PIPELINES["survey"]), "data_validation") == "survey_analysis"
    assert _dynamic_route(_state(PIPELINES["review"]), "systematic_review") == "thesis_argument"
    assert _dynamic_route(_state(PIPELINES["cause_effect"]), "engineering") == "causal_audit"
    assert _dynamic_route(_state(PIPELINES["analytical"]), "topic_discovery") == "thesis_argument"
    assert _dynamic_route(_state(PIPELINES["experimental"]), "supervision") == "meta_evaluation"


def test_superset_graph_builds_from_production_nodes():
    import main

    graph = main.create_research_graph()
    assert graph is not None
    # New phases must be wired in production (the old experimental-only build
    # silently dropped them).
    for phase in ("thesis_argument", "systematic_review", "survey_analysis",
                  "case_protocol", "causal_audit", "meta_evaluation"):
        assert phase in main.research_nodes(), phase


def test_create_superset_graph_compiles():
    import main

    app = create_superset_graph(main.research_nodes()).compile()
    assert app is not None


def test_is_valid_plan_waives_non_empirical():
    from core.workflow_nodes import is_valid_plan

    st = initialize_state()
    st["paper_type"] = "analytical"
    st["plan"] = {"experiments": [], "plan_mode": "non_empirical", "sections": []}
    assert is_valid_plan(st) is True
    st2 = initialize_state()
    st2["paper_type"] = "experimental"
    st2["plan"] = {"experiments": [], "sections": []}
    assert is_valid_plan(st2) is False


def test_reset_preserves_type_routing():
    from core.workflow_nodes import reset_node

    st = initialize_state()
    st["iteration"] = 0
    st["paper_type"] = "survey"
    st["pipeline_recipe"] = list(PIPELINES["survey"])
    st["intake_report"] = {"goal": "x" * 30}
    out = reset_node(st)
    assert out["paper_type"] == "survey"
    assert out["pipeline_recipe"] == list(PIPELINES["survey"])
    assert out["iteration"] == 1


def test_engineering_waiver_skip():
    from core.workflow_nodes import engineering_node

    st = initialize_state()
    st["paper_type"] = "analytical"
    st["plan"] = {"experiments": [], "plan_mode": "non_empirical", "sections": []}
    out = engineering_node(st)
    assert out.get("terminal_error") in (None, "")
    assert out["analysis_reports"]["engineering"]["skipped"] is True


def test_next_recipe_phase_fallback():
    assert _next_recipe_phase(None, "writing_narrative") == "engineering"
    assert _next_recipe_phase([], "editing") is None
