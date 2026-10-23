"""M1: no-experiment paper types — analytical / descriptive / argumentative.

Proposal §11-M1 acceptance: 3 x L2 mocked E2E (engineering call count == 0)
plus verb-redraft, triad, zero-cite-fail and quarantine cases.
"""

from __future__ import annotations

import json
import re

import pytest

DOI_1 = "10.1234/scholargraph01"
DOI_2 = "10.1234/scholargraph02"
ARXIV_1 = "2301.00001"

EVIDENCE = [
    {
        "title": "Citation velocity indicators for emerging topics",
        "abstract": (
            "This study examines citation velocity and keyword bursts across large "
            "bibliographic databases to flag emerging topics early for database curators."
        ),
        "doi": DOI_1,
        "arxiv_id": None,
        "year": 2023,
    },
    {
        "title": "Calibration of probabilistic forecasts",
        "abstract": (
            "We evaluate probabilistic forecasts from ensemble models, documenting "
            "miscalibration patterns and recalibration procedures for forecast consumers."
        ),
        "doi": DOI_2,
        "arxiv_id": ARXIV_1,
        "year": 2024,
    },
]

BRIDGE_1 = "Prior coupling studies map research fronts through shared references, bounding synthesis scope."
BRIDGE_2 = (
    "Calibration literature documents overconfident estimates under distribution shift, "
    "motivating cautious synthesis."
)
STEELMAN = (
    "The strongest counter-reading notes evidence remains insufficient for generalized "
    "claims pending independent replication."
)
CONCESSION = (
    "This thesis concedes its scope is limited to published applied investigations "
    "and excludes unpublished reports."
)
CLAIM = "Integrative synthesis workflows improve cumulative insight across fragmented applied domains."

QUANT = (
    "Prior work reports mean accuracy 0.82 with n=120 and std=0.04 "
    "(95% confidence interval 0.74 to 0.90), as recorded in doi:10.1234/scholargraph01."
)

GROUNDING = " All background statements below draw on retrieved literature evidence."

# Framing vocabulary avoids the evidence abstracts' content words so the
# novelty screen stays green (checked by the tests themselves).


def _intro() -> str:
    return (
        "# Introduction\n\nThis paper investigates how synthesis methods organize fragmented "
        "findings in applied investigation domains. The problem persists: isolated inquiries "
        "accumulate without cumulative insight, as shown in doi:10.1234/scholargraph01. "
        "Retrieved literature evidence indicates repeated calls for integrative synthesis. "
        "Prior coupling studies map research fronts through shared references, so a reproducible "
        "synthesis workflow is timely. Our contribution organizes the workflow into explicit stages "
        "with traceable provenance, per arXiv:2301.00001. The remainder of this paper surveys context, "
        "presents the synthesis, discusses implications, and states limitations with unusual candor "
        "about coverage boundaries and unresolved questions in this investigation domain."
        + GROUNDING
    )


def _abstract() -> str:
    return (
        "# Abstract\n\nWe investigate synthesis workflows that organize fragmented findings in "
        "applied investigation domains, motivated by doi:10.1234/scholargraph01. Retrieved literature "
        "evidence supports integrative synthesis with traceable provenance. Our synthesis maps stages, "
        "states scope boundaries, and records limitations for future investigations in this domain."
        + GROUNDING
    )


def _background() -> str:
    return (
        "# Background\n\nContext for this synthesis comes from retrieved literature evidence, "
        "including doi:10.1234/scholargraph01 and arXiv:2301.00001. "
        + BRIDGE_1 + " " + BRIDGE_2 + " Together these sources bound the synthesis scope: "
        "we synthesize published investigations with explicit provenance chains and record every "
        "coverage gap as a limitation for future work in this investigation domain."
        + GROUNDING
    )


def _analysis(quant: bool = True) -> str:
    body = (
        "# Analysis\n\nThe synthesis proceeds in three stages. First, sources are collected with "
        "provenance. Second, bridge statements connect each source to the central claim. "
        + BRIDGE_1 + " " + BRIDGE_2 + " Third, scope boundaries are recorded. "
    )
    if quant:
        body += QUANT + " "
    body += (
        "Retrieved literature evidence underwrites each stage, and every interpretive move is "
        "labelled as synthesis rather than measurement in this investigation domain." + GROUNDING
    )
    return body


def _discussion() -> str:
    return (
        "# Discussion\n\nThe synthesis implies that provenance-first workflows reduce ambiguity about "
        "what is established, per doi:10.1234/scholargraph02. "
        + BRIDGE_2 + " Limits include dependence on published investigations and sparse coverage of "
        "null findings. Retrieved literature evidence suggests future synthesis should preregister "
        "inclusion rules for this investigation domain." + GROUNDING
    )


def _conclusion() -> str:
    return (
        "# Conclusion\n\nThis synthesis organizes fragmented findings into a traceable workflow, "
        "grounded in doi:10.1234/scholargraph01 and arXiv:2301.00001. Retrieved literature evidence "
        "supports its stages. Limitations bound its scope, and future investigations can extend the "
        "workflow to adjacent domains with care." + GROUNDING
    )


def _findings_causal() -> str:
    return (
        "# Findings\n\nObserved profiles are summarized here. The program causes higher completion "
        "rates, and exposure leads to stronger retention. Retrieved literature evidence frames the "
        "profile vocabulary, per doi:10.1234/scholargraph01. Group contrasts are reported with care "
        "for sampling variation in this investigation domain." + GROUNDING
    )


def _findings_clean() -> str:
    return (
        "# Findings\n\nObserved profiles are summarized here. Higher completion rates are associated "
        "with program participation, with 95% confidence interval 0.74 to 0.90 (n=120, std=0.04), "
        "per doi:10.1234/scholargraph01. Stronger retention is observed alongside exposure. Retrieved "
        "literature evidence frames the profile vocabulary. No directional reading is warranted beyond "
        "the reported intervals in this investigation domain." + GROUNDING
    )


def _thesis_section() -> str:
    return (
        "# Thesis\n\nThe central claim is: " + CLAIM + " Retrieved literature evidence supports "
        "this claim through explicit bridges, per doi:10.1234/scholargraph01 and "
        "doi:10.1234/scholargraph02. " + BRIDGE_1 + " " + BRIDGE_2 + " Scope boundaries are "
        "stated alongside the claim for this investigation domain." + GROUNDING
    )


def _evidence_section() -> str:
    return (
        "# Evidence\n\nEach bridge is grounded in retrieved literature evidence. " + BRIDGE_1 + " "
        "Source doi:10.1234/scholargraph01 records the mapping approach. " + BRIDGE_2 + " Source "
        "doi:10.1234/scholargraph02 records the calibration concern. " + QUANT + " Together the "
        "bridges support the claim within stated scope for this investigation domain." + GROUNDING
    )


def _counterargument() -> str:
    return (
        "# Counterargument\n\nThe strongest counter-reading notes evidence remains insufficient "
        "for generalized claims pending independent replication. We answer that the claim is scoped "
        "to published investigations, conceding coverage limits per doi:10.1234/scholargraph02. "
        "The concession test is met by one cited source whose stated finding directly entails the "
        "scoped claim, and replication remains future work for this investigation domain. Retrieved "
        "literature evidence is cited for each rebuttal step." + GROUNDING
    )


PROSE = {
    "abstract": _abstract(),
    "introduction": _intro(),
    "background": _background(),
    "analysis": _analysis(),
    "discussion": _discussion(),
    "conclusion": _conclusion(),
    "findings": _findings_clean(),
    "thesis": _thesis_section(),
    "evidence": _evidence_section(),
    "counterargument": _counterargument(),
}

CANNED_THESIS = {
    "claim": CLAIM,
    "bridges": [
        {"statement": BRIDGE_1, "source": DOI_1},
        {"statement": BRIDGE_2, "source": DOI_2},
    ],
    "steelman": STEELMAN,
    "concession": CONCESSION,
    "fallback_used": False,
}


def _mock_resolve_doi(doi):
    return {"resolved": True, "doi": doi, "title": f"Title for {doi}", "authors": ["Smith"]}


def _mock_resolve_arxiv(aid):
    return {"resolved": True, "arxiv_id": aid, "title": f"Title for {aid}", "authors": ["Smith"]}


@pytest.fixture()
def noexp_env(monkeypatch, tmp_path):
    """Offline harness: resolved citations, canned thesis, routed LLM, tmp outputs."""
    from core import verification
    from core.config import config
    import agents.writer as writer_mod
    import agents.editor as editor_mod
    import agents.supervisor as supervisor_mod
    import agents.thesis_agent as thesis_mod

    monkeypatch.setattr(verification, "resolve_doi", _mock_resolve_doi)
    monkeypatch.setattr(verification, "resolve_arxiv", _mock_resolve_arxiv)
    monkeypatch.setattr(thesis_mod, "call_llm_json", lambda *a, **k: dict(CANNED_THESIS))
    monkeypatch.setattr(verification, "call_llm_json", lambda *a, **k: {"findings": []})

    def _soft_review(*a, **k):
        return {"overall_score": 9.0, "scores": {}, "weaknesses": [], "checklist": {}}

    def _soft_hall(*a, **k):
        return {"score": 9, "issues": [], "recommendations": []}

    def _review_router(prompt, *a, **k):
        if "peer reviewer" in prompt:
            return _soft_review()
        return _soft_hall()

    monkeypatch.setattr(supervisor_mod, "call_llm_json", _review_router)
    monkeypatch.setattr(writer_mod.WriterAgent, "_store_section", lambda self, *a, **k: None)
    monkeypatch.setattr(editor_mod, "generate_embedding", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(config, "output_dir", str(tmp_path / "output"))
    monkeypatch.setattr(config, "companion_repo_dir", str(tmp_path / "companion"))
    calls: list = []

    def _writer_llm(prompt, *a, **k):
        calls.append(str(prompt))
        lowered = str(prompt).lower()
        if "rewrite the" in lowered and "associationally" in lowered:
            return _findings_clean()
        m = re.search(r"write (?:a|an|the) (\w+) section", lowered)
        if m:
            return PROSE.get(m.group(1), PROSE["analysis"])
        if "concise abstract" in lowered:
            return PROSE["abstract"]
        return PROSE["analysis"]

    monkeypatch.setattr(writer_mod, "call_llm", _writer_llm)
    return {"calls": calls}


def _findings_causal_env(noexp_env):
    """First Findings draft carries causal verbs (redraft path); later calls are clean."""
    import agents.writer as writer_mod

    seen = {"n": 0}
    base = writer_mod.call_llm

    def _first_causal(prompt, *a, **k):
        lowered = str(prompt).lower()
        if "write a findings section" in lowered and "rewrite the" not in lowered and seen["n"] == 0:
            seen["n"] += 1
            noexp_env["calls"].append(str(prompt))
            return _findings_causal()
        return base(prompt, *a, **k)

    return _first_causal


def _make_state(paper_type):
    """Production-path state: NO hand-built plan — thesis_argument must synthesize it.

    This exercises the real no-experiment path (§4.5): recipes without a
    planning phase get their light plan (thesis_slot/sources/claims,
    experiments: [], manuscript sections) from the thesis node.
    """
    from core.state import initialize_state
    from core.phase_registry import resolve_recipe

    recipe, skipped, evidence, _ = resolve_recipe(paper_type, {})
    state = initialize_state()
    state["run_id"] = f"test-{paper_type}"
    state["paper_type"] = paper_type
    state["pipeline_recipe"] = recipe
    state["skipped_phases"] = skipped
    state["evidence_requirements"] = evidence
    state["selected_topic"] = {
        "title": f"Test {paper_type} topic",
        "description": "A synthesis investigation for offline harness verification.",
        "literature_evidence": [dict(p) for p in EVIDENCE],
    }
    state["plan"] = None
    state["engineer_outputs"] = {}
    return state


def _run_noexp_pipeline(state):
    import core.workflow_nodes as wn

    state = wn.thesis_argument_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    state = wn.write_narrative_sections(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    state = wn.write_results_sections(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    state = wn.supervision_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    assert state.get("current_phase") == "editing", state.get("supervisor_scores")
    state = wn.editing_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    return state


def _assert_l2(state, paper_type):
    from core.verification import final_manuscript_referee

    # Production-path plan: synthesized by thesis_argument (§4.5), never hand-built.
    plan = state.get("plan") or {}
    assert plan.get("experiments") == [], plan
    assert plan.get("plan_mode") == "non_empirical", plan.get("plan_mode")
    assert len(plan.get("sections") or []) >= 6, plan.get("sections")
    assert state.get("final_paper"), "no final paper assembled"
    assert state.get("latex_output"), "no latex output"
    sections = state["final_paper"]["sections"]
    assert sections.get("Limitations") and len(sections["Limitations"]) >= 80
    scores = state["supervisor_scores"]
    assert scores and min(scores.values()) >= 8.5, scores
    referee = final_manuscript_referee(
        sections, state["plan"], state.get("engineer_outputs") or {},
        topic=state["selected_topic"], paper_type=paper_type, thesis=state.get("thesis"),
    )
    assert referee["passed"], referee["findings"]
    assert state["final_paper"].get("paper_type") == paper_type


# --- M1 cases ---------------------------------------------------------------

def test_analytical_e2e_no_engineering_l2(noexp_env, monkeypatch):
    """Analytical E2E mocked: engineering never runs, L2 evidence bundle reached."""
    import core.workflow_nodes as wn

    monkeypatch.setattr(wn, "engineering_node", lambda state: (_ for _ in ()).throw(AssertionError("engineering must not run")))
    state = _make_state("analytical")
    state = _run_noexp_pipeline(state)
    _assert_l2(state, "analytical")
    assert state["thesis"]["claim"] == CLAIM
    assert len(state["thesis"]["bridges"]) == 2


def test_descriptive_e2e_verb_redraft_ci(noexp_env, monkeypatch):
    """Descriptive E2E mocked: causal verbs trigger a bounded redraft, CIs present."""
    import agents.writer as writer_mod
    import core.workflow_nodes as wn

    monkeypatch.setattr(writer_mod, "call_llm", _findings_causal_env(noexp_env))
    monkeypatch.setattr(wn, "engineering_node", lambda state: (_ for _ in ()).throw(AssertionError("engineering must not run")))
    state = _make_state("descriptive")
    state = _run_noexp_pipeline(state)
    _assert_l2(state, "descriptive")
    prompts = noexp_env["calls"]
    assert any("associationally" in p for p in prompts), "verb-lint redraft never fired"
    findings = state["draft_sections"]["Findings"]
    assert "causes" not in findings.lower() and "leads to" not in findings.lower()
    assert "confidence interval" in findings.lower()


def test_argumentative_e2e_triad_steelman(noexp_env, monkeypatch):
    """Argumentative E2E mocked: triad present, steelman non-empty (thesis + debate mode)."""
    import core.workflow_nodes as wn
    from agents.hypothesis_debate import ChallengerAgent

    monkeypatch.setattr(wn, "engineering_node", lambda state: (_ for _ in ()).throw(AssertionError("engineering must not run")))
    state = _make_state("argumentative")
    state = _run_noexp_pipeline(state)
    _assert_l2(state, "argumentative")
    thesis = state["thesis"]
    assert thesis["claim"] and thesis["bridges"] and thesis["steelman"] and thesis["concession"]

    # Steelman debate mode: challenger produces strongest counter + concession test.
    import agents.hypothesis_debate as debate_mod

    def _steelman_llm(prompt, *a, **k):
        assert "steelman" in str(prompt).lower()
        return json.dumps({
            "steelman_argument": "The position overstates what its cited sources establish.",
            "concession_test": "Produce one source whose stated finding entails the claim.",
            "objections": [{"criterion": "soundness", "objection": "scope overstatement",
                            "severity": 3, "status": "unresolved"}],
            "summary_rebuttal": "scope objection",
        })

    monkeypatch.setattr(debate_mod, "call_llm", _steelman_llm)
    challenger = ChallengerAgent.__new__(ChallengerAgent)
    challenger.context = None
    challenger.client = None
    challenger.vector_memory = type("Mem", (), {"get_prompt_context": lambda *a, **k: []})()
    summary = ChallengerAgent.build_rebuttal(
        challenger, {"title": "t", "description": "d"}, "position text", mode="steelman")
    assert challenger._last_steelman, "steelman empty"
    assert challenger._last_concession_test, "concession test empty"
    assert any(o.get("source") == "steelman_audit" for o in challenger._last_objections)
    assert "STEELMAN" in summary


def test_zero_cite_quantitative_fails(noexp_env):
    """L20: quantitative claims with zero DOI/arXiv citations fail."""
    from core import verification

    text = "Treatment improved mean accuracy from 0.70 to 0.85 with p < 0.01."
    result = verification.verify_citations(text)
    assert result["passed"] is False
    # Control: the same claims grounded by a resolving citation pass.
    grounded = text + " See doi:10.1234/scholargraph01."
    ok = verification.verify_citations(grounded)
    assert ok["passed"] is True
    # Control: qualitative prose without IDs still passes (no L20 trigger).
    plain = verification.verify_citations("This synthesis organizes prior investigations.")
    assert plain["passed"] is True


def test_single_cite_failure_quarantined_run_continues(noexp_env, monkeypatch):
    """R5 slice: one failed citation quarantines to the repair pass; run continues."""
    from core import verification
    from agents.supervisor import SupervisorAgent

    def _one_bad(doi):
        if doi == "10.9999/fake.doi.123":
            return {"resolved": False, "doi": doi, "error": "not found"}
        return {"resolved": True, "doi": doi, "title": "Real", "authors": ["Smith"]}

    monkeypatch.setattr(verification, "resolve_doi", _one_bad)
    text = ("Background per doi:10.1234/scholargraph01 and doi:10.9999/fake.doi.123. " * 20
            + " Retrieved literature evidence frames this investigation domain.")
    legacy = verification.verify_citations(text)
    assert legacy["passed"] is False  # legacy default stays fail-closed
    result = verification.verify_citations(text, quarantine_singles=True)
    assert result["passed"] is True
    assert len(result["quarantined"]) == 1
    assert result["quarantined"][0]["doi"] == "10.9999/fake.doi.123"

    # Supervisor level (analytical auto-quarantines): section continues with a note.
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.context = None
    agent.client = None
    agent.ledger = type("L", (), {"record_claim": lambda *a, **k: None})()
    agent.feedback_memory = type("M", (), {"add_feedback_entry": lambda *a, **k: None})()
    from agents.supervisor import MathChecker, CodeChecker, ReviewerBot
    agent.math_checker = MathChecker()
    agent.code_checker = CodeChecker()
    agent.reviewer_bot = ReviewerBot()
    thesis = dict(CANNED_THESIS)
    score, feedback = SupervisorAgent.evaluate_section(
        agent, "Background", text, engineer_outputs={}, paper_type="analytical", thesis=thesis)
    assert "Quarantined" in feedback or "quarantined" in feedback


def test_verify_thesis_triad_and_bridge_coverage():
    """Thesis gate unit: triad + 100% bridge coverage passes; gaps fail."""
    from agents.thesis_agent import verify_thesis

    sections = {"Background": _background(), "Analysis": _analysis(), "Discussion": _discussion()}
    ok = verify_thesis(dict(CANNED_THESIS), sections)
    assert ok["passed"] is True, ok["findings"]
    assert ok["bridge_coverage"] == 1.0

    thin = {"Background": "Unrelated filler about laboratory glassware procurement budgets."}
    missing = verify_thesis(dict(CANNED_THESIS), thin)
    assert missing["passed"] is False
    assert missing["uncovered"], "expected uncovered bridges"

    no_triad = verify_thesis({"claim": "", "bridges": [], "steelman": "", "concession": ""}, sections)
    assert no_triad["passed"] is False
    assert set(no_triad["triad"]) == {"claim", "bridges", "steelman_concession"}


# --- Extra regression guards -------------------------------------------------

def test_supervision_min_gate_thesis_types(noexp_env, monkeypatch):
    """L12: thesis types gate on the weakest section (min), not the mean."""
    import core.workflow_nodes as wn
    from core.state import initialize_state

    # v5 (f537f638: mean 2.19 hid five 0.0 sections behind one 9.14): all
    # types gate on the weakest section (min), not the mean.
    # 9.5 + 8.0: mean (8.75) clears 8.5 but min (8.0) does not.
    for paper_type, expected in (("analytical", "meta_evaluation"), ("experimental", "meta_evaluation")):
        state = initialize_state()
        state["paper_type"] = paper_type
        state["plan"] = {"experiments": []}
        state["draft_sections"] = {"A": "x", "B": "y"}
        state["engineer_outputs"] = {}
        scores = iter([9.5, 8.0])

        class _Sup:
            def evaluate_section(self, *a, **k):
                return next(scores), "fb"

        monkeypatch.setattr(wn, "_create_agent", lambda cls: _Sup())
        out = wn.supervision_node(state)
        assert out["current_phase"] == expected, (paper_type, out["supervisor_scores"])


def test_verify_result_numbers_year_allowlist():
    """L14: calendar years are grounding-exempt; real mismatches still fail."""
    from core.workflow_nodes import verify_result_numbers

    assert verify_result_numbers.__defaults__ == (0.05,)
    outputs = {"e": {"aggregate_metrics": {"accuracy": {"mean": 0.82, "std": 0.04}}}}
    year_only = verify_result_numbers("Published in 2024 with methods from 2019.", outputs)
    assert year_only["passed"] is True, year_only["mismatches"]
    bad = verify_result_numbers("Accuracy reached 0.99 in 2024.", outputs)
    assert bad["passed"] is False
    assert 0.99 in bad["mismatches"] and 2024 not in bad["mismatches"]


def test_typed_graph_wires_thesis_skips_engineering():
    """Recipe wiring: thesis_argument wired for thesis types, engineering excluded."""
    from core.workflow import create_typed_graph
    from core.phase_registry import resolve_recipe
    from main import research_nodes

    nodes = research_nodes()
    assert "thesis_argument" in nodes, "M1 must wire thesis_argument into the node mapping"
    for paper_type in ("analytical", "argumentative", "descriptive"):
        recipe, _, _, _ = resolve_recipe(paper_type, {})
        graph = create_typed_graph(nodes, recipe)
        assert "thesis_argument" in graph.nodes or paper_type == "descriptive", (
            paper_type, sorted(graph.nodes))
        assert "engineering" not in graph.nodes, (paper_type, sorted(graph.nodes))
    recipe, _, _, _ = resolve_recipe("experimental", {})
    graph = create_typed_graph(nodes, recipe)
    assert "engineering" in graph.nodes


def test_debate_phase_advance_follows_recipe(noexp_env, monkeypatch):
    """Phase labels follow the recipe: argumentative debate -> writing_narrative."""
    import core.workflow_nodes as wn
    from core.state import initialize_state
    from core.phase_registry import resolve_recipe
    from agents.hypothesis_debate import DebateResult

    passed = DebateResult("t", "arg", "reb", "PASS", 8.0, True, "ok")

    class FakeDebater:
        runtime_config = type("C", (), {"debate_pass_threshold": 7.0})()

        def conduct_with_repair(self, topic, **kw):
            assert (kw.get("mode", "standard")) == ("steelman" if topic.get("steelman") else "standard")
            return [passed]

    class FakeHunter:
        def pre_debate_self_critique(self, topic):
            return topic

    def _fake_create(agent_class):
        if agent_class.__name__ == "HypothesisDebateSystem":
            return FakeDebater()
        return FakeHunter()

    monkeypatch.setattr(wn, "_create_agent", _fake_create)
    for paper_type, steelman, expected in (
        ("argumentative", True, "writing_narrative"),
        ("experimental", False, "planning"),
    ):
        state = initialize_state()
        state["paper_type"] = paper_type
        recipe, _, _, _ = resolve_recipe(paper_type, {})
        state["pipeline_recipe"] = recipe
        topic = {"title": "t", "description": "d"}
        if steelman:
            topic["steelman"] = True
        state["topics"] = [topic]
        out = wn.hypothesis_debate_node(state)
        assert out["current_phase"] == expected, (paper_type, out["current_phase"])
