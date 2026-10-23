"""P1-P3 regression tests: deferred routing, try-and-report, no-terminal planning."""

from core.phase_registry import DEFERRED_TYPE, infer_candidates
from core.state import initialize_state, migrate_state_v1_to_v2
from core.workflow_nodes import (
    _exploratory_acceptable,
    autofix_soft_plan_issues,
    classify_plan_issues,
    is_valid_plan,
    resolve_deferred_routing,
    terminal_planning_failure,
)


class _R:
    def __init__(self, score=6.0, infra=False):
        self.score = score
        self.infra_failure = infra


def test_intake_defers_auto_to_candidates():
    cands = infer_candidates({})
    types = [c["type"] for c in cands]
    assert "experimental" in types
    assert len(cands) >= 3  # never collapses to a single ID


def test_fresh_state_is_deferred():
    s = initialize_state()
    assert s["paper_type"] == "auto-deferred"
    assert s["pipeline_recipe"] == []
    assert s["hunt_mode"] == "try_and_report"


def test_migrate_v1_stays_deferred():
    m = migrate_state_v1_to_v2({"foo": 1})
    assert m["paper_type"] == "auto-deferred"


def test_post_evidence_routing_binds_from_pack():
    s = initialize_state()
    s["intake_report"] = {"pack_manifest": {"response_table": {"path": "r.csv"}}}
    s["selected_topic"] = {"title": "t", "structured_hypothesis": {"falsification_condition": "x"}}
    out = resolve_deferred_routing(s)
    assert out["paper_type"] == "survey"
    assert "survey_analysis" in out["pipeline_recipe"]
    assert out["routing_provenance"]["stage"] == "post-evidence"


def test_post_evidence_routing_try_without_literature():
    s = initialize_state()
    s["intake_report"] = {"pack_manifest": {}}
    s["selected_topic"] = {
        "title": "try X on Y",
        "hunt_mode": "try_and_report",
        "structured_hypothesis": {"falsification_condition": "acc drops"},
    }
    s["hunt_mode"] = "try_and_report"
    out = resolve_deferred_routing(s)
    # Honest empirical label, not a fake gap claim.
    assert out["paper_type"] in ("empirical", "experimental")
    assert out["pipeline_recipe"]


def test_try_screener_passes_without_literature():
    from agents.topic_hunter import TopicHunterAgent

    ag = TopicHunterAgent.__new__(TopicHunterAgent)
    rep = ag.screen_research_gap(
        {"title": "try X", "description": "try logistic on synthetic, report outcome"},
        [],
        hunt_mode="try_and_report",
    )
    assert rep["status"] == "PASS"
    assert rep["reason"] == "try_and_report_without_literature"


def test_gap_screener_still_fails_without_literature():
    from agents.topic_hunter import TopicHunterAgent

    ag = TopicHunterAgent.__new__(TopicHunterAgent)
    rep = ag.screen_research_gap({"title": "gap", "description": "d"}, [], hunt_mode="gap")
    assert rep["status"] == "FAIL"


def test_pregate_try_skips_regex():
    from agents.topic_hunter import TopicHunterAgent

    ag = TopicHunterAgent.__new__(TopicHunterAgent)
    ok, _ = ag._formalize_pregate(
        {"title": "Simple characterization of optimizer behavior on tiny data",
         "description": "run optimizer variants on synthetic blobs and report curves plainly"},
        hunt_mode="try_and_report",
    )
    assert ok is True


def test_novelty_replication_ok_not_reject():
    from agents.topic_hunter import TopicHunterAgent

    ag = TopicHunterAgent.__new__(TopicHunterAgent)
    # No abstracts -> auto-novel path untouched; direct logic tested via hunt_mode plumbing
    out = ag.evaluate_layered_novelty({"title": "t", "description": "d"}, [], hunt_mode="replication")
    assert out["reject"] is False


def test_soft_variant_autofix_counts_valid():
    s = initialize_state()
    s["paper_type"] = "experimental"
    s["plan"] = {
        "paper_type": "experimental",
        "experiments": [{
            "name": "mve", "evaluation_metrics": ["accuracy"],
            "variants": [{"method": "lr"}, {"name": "base"}],
        }],
    }
    assert classify_plan_issues(s)["hard"] == []
    assert classify_plan_issues(s)["soft"]
    assert autofix_soft_plan_issues(s) is True
    assert is_valid_plan(s) is True


def test_terminal_node_downgrades_not_kills():
    s = initialize_state()
    s["paper_type"] = "experimental"
    s["pipeline_recipe"] = ["planning", "data_validation", "engineering"]
    s["selected_topic"] = {"title": "t"}
    s["plan"] = {"experiments": [{"name": "", "evaluation_metrics": []}]}
    out = terminal_planning_failure(s)
    assert out["terminal_error"] is None
    assert out["should_continue"] is True
    assert out["plan"]["experiments"]
    assert out["evidence_gate"]["reason_code"] == "planning_downgraded_exploratory"


def test_exploratory_accepts_near_miss_any_iteration():
    topic = {"title": "t", "structured_hypothesis": {"falsification_condition": "x"}}
    assert _exploratory_acceptable(topic, _R(score=5.5)) is True
    assert _exploratory_acceptable(topic, _R(score=5.5, infra=True)) is False


def test_funnel_counts_engineering_reach():
    from core.funnel import funnel_summary

    states = [
        {"current_phase": "engineering", "terminal_error": None, "topics": [1], "selected_topic": {}, "plan": {}, "engineer_outputs": {}},
        {"current_phase": "complete", "terminal_error": "no_viable_topic", "topics": [], "selected_topic": None, "plan": None, "engineer_outputs": {}},
    ]
    summ = funnel_summary(states)
    assert summ["total"] == 2
    assert summ["reached_engineering"] == 1
