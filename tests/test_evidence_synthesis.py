from core.evidence_synthesis import build_cross_paper_evidence_map, validate_candidate_bridge_claim, validate_topic_admission


def test_cross_paper_synthesis_requires_two_distinct_grounded_sources():
    papers = [
        {
            "id": "method-paper",
            "title": "Robustness validation for learning systems",
            "abstract": "We develop a simulation and validation algorithm for robustness evaluation of learning systems.",
        },
        {
            "id": "setting-paper",
            "title": "IoT healthcare networks",
            "abstract": "Healthcare IoT networks require reliable sensor coordination and security monitoring.",
        },
    ]

    result = build_cross_paper_evidence_map(papers)

    assert result["schema_version"] == "cross-paper-evidence-map/v1"
    assert result["bridges"]
    bridge = result["bridges"][0]
    assert set(bridge["source_paper_ids"]) == {"method-paper", "setting-paper"}
    assert bridge["status"] == "candidate_requires_literature_screening"
    assert all(item["excerpt"] for item in bridge["evidence"])

    grounded = validate_candidate_bridge_claim(
        {"title": "Robustness validation for healthcare IoT", "evidence_bridge_ids": [bridge["bridge_id"]]},
        result,
    )
    ungrounded = validate_candidate_bridge_claim({"title": "Unrelated topic"}, result)
    assert grounded["valid"]
    # Soft validation: ungrounded candidates get a warning but are not blocked
    assert ungrounded["valid"]
    assert "soft_warning" in ungrounded["reason"]


def test_cross_paper_synthesis_does_not_create_a_bridge_without_roles():
    result = build_cross_paper_evidence_map([
        {"id": "one", "title": "A short note", "abstract": "A historical discussion of archives."},
        {"id": "two", "title": "Another note", "abstract": "A philosophical discussion of history."},
    ])

    assert result["bridges"] == []


def test_topic_admission_requires_an_executable_falsifiable_mve():
    accepted = validate_topic_admission({
        "research_question": "Does calibration improve reliability?",
        "hypothesis": "The calibrated model reduces error.",
        "dependent_variables": ["accuracy"],
        "falsification_condition": "No improvement across seeds.",
        "minimum_viable_experiment": {
            "dataset": "sklearn_iris",
            "baseline": "logistic_regression",
            "metrics": ["accuracy"],
            "falsification_test": "paired bootstrap confidence interval",
            "seeds": 3,
        },
    })
    rejected = validate_topic_admission({"research_question": "Unspecified"})

    assert accepted["admitted"]
    # v2: core hypothesis fields (hypothesis, dependent_variables,
    # falsification_condition) are still required — only MVE is auto-fixed.
    assert not rejected["admitted"]
    assert any("hypothesis" in error for error in rejected["errors"])


def test_repair_loop_is_bounded_to_one_revised_contract(monkeypatch):
    from agents.hypothesis_debate import DebateResult, HypothesisDebateSystem

    system = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    first = DebateResult("topic", "", "", "FAIL", 5.0, False, "", objections=[{"criterion": "baseline", "status": "unresolved"}])
    second = DebateResult("topic", "", "", "PASS", 8.0, True, "")
    calls = []

    def debate(topic):
        calls.append(topic.get("revision"))
        return second if topic.get("revision") else first

    system.conduct_debate = debate
    system.revise_topic_from_objections = lambda topic, result: {**topic, "revision": 1}

    topic = {"title": "topic"}
    results = system.conduct_with_repair(topic)

    assert results == [first, second]
    assert calls == [None, 1]
    assert topic["revision"] == 1


def test_serial_workflow_uses_the_bounded_repair_protocol(monkeypatch):
    from agents.hypothesis_debate import DebateResult
    from core.state import initialize_state
    from core.workflow_nodes import hypothesis_debate_node

    state = initialize_state()
    state["topics"] = [{"title": "Repairable topic"}]
    repaired = DebateResult("Repairable topic", "", "", "PASS", 8.0, True, "")

    class FakeDebater:
        def __init__(self, *_args):
            self.called = False

        def conduct_with_repair(self, topic):
            self.called = True
            topic["repaired"] = True
            return [DebateResult(topic["title"], "", "", "FAIL", 5.0, False, ""), repaired]

    monkeypatch.setattr("core.workflow_nodes.HypothesisDebateSystem", FakeDebater)
    outcome = hypothesis_debate_node(state)

    assert outcome["hypothesis_passed"]
    assert outcome["selected_topic"]["repaired"]
    assert outcome["debate_results"][-1] is repaired
