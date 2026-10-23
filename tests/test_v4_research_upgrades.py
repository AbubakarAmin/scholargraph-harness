"""Offline tests for the v4 research-upgrade batch.

1. Self-correcting JSON calls (control-data flow separation).
2. Feedback-aware writer revision (reviewer-guided > blind re-roll).
3. Literature-grounded introduction/abstract drafting.
4. Results-prompt hardening (copy-exact + uncertainty).
5. Novelty-plagiarism gate in the final manuscript referee.
6. Bounded editor repair routing (WARA-style artifact repair).
7. Targeted results redraft carrying check failures into prompts.
8. Narrative revision loop on meta-continue.
9. LLM failure visibility in run stats.
10. Graph editing routing (repair → writing_results; done → END).
11. Generic section revision feedback reach.
"""


# ---------------------------------------------------------------------------
# 1. call_llm_json self-correcting structured output
# ---------------------------------------------------------------------------

def test_call_llm_json_repairs_malformed_json():
    from core.utils import call_llm_json

    prompts = []
    responses = iter(["not json at all", '{"score": 7}'])

    def fake_call(prompt, **_kwargs):
        prompts.append(prompt)
        return next(responses)

    parsed = call_llm_json("give json", attempts=3, call_fn=fake_call)
    assert parsed == {"score": 7}
    assert len(prompts) == 2
    assert "could not be parsed as JSON" in prompts[1]


def test_call_llm_json_returns_none_after_exhausting_attempts():
    from core.utils import call_llm_json

    calls = []

    def fake_call(prompt, **_kwargs):
        calls.append(prompt)
        return "still not json"

    result = call_llm_json("give json", attempts=2, call_fn=fake_call)
    assert result is None
    assert len(calls) == 2


def test_call_llm_json_single_valid_response_short_circuits():
    from core.utils import call_llm_json

    calls = []

    def fake_call(prompt, **_kwargs):
        calls.append(prompt)
        return '{"answer": 42}'

    parsed = call_llm_json("prompt", call_fn=fake_call)
    assert parsed == {"answer": 42}
    assert len(calls) == 1


def test_consistency_referee_uses_self_correcting_parse(monkeypatch):
    """The referee must survive one malformed response by re-asking."""
    from core import verification

    responses = ["first attempt is not json", '{"findings": []}']

    def fake_call(*_args, **_kwargs):
        return responses.pop(0)

    monkeypatch.setattr(verification, "call_llm", fake_call)
    result = verification.consistency_referee({"Results": "clean"})
    assert result["passed"], result


def test_consistency_referee_non_blocking_after_repeated_malformation(monkeypatch):
    # R10 (M6): malformed referee output after retry → non-blocking warning.
    from core import verification

    monkeypatch.setattr(verification, "call_llm", lambda *_a, **_k: "not json")
    result = verification.consistency_referee({"Results": "No contradictions"})
    assert result["passed"]
    assert result["findings"][0]["category"] == "referee_error"
    assert result["findings"][0]["blocking"] is False


# ---------------------------------------------------------------------------
# Writer revision feedback + literature grounding
# ---------------------------------------------------------------------------

def _bare_writer():
    from agents.writer import WriterAgent

    writer = WriterAgent.__new__(WriterAgent)
    writer.context = None
    writer.client = None
    writer._active_revision_feedback = None
    return writer


def test_writer_revision_feedback_enters_prompt(monkeypatch):
    import agents.writer as writer_mod

    captured = {}

    def fake_call(prompt, **_kwargs):
        captured["prompt"] = prompt
        return "# Abstract\n\n" + "We study a bounded synthetic benchmark with verified results. " * 12

    monkeypatch.setattr(writer_mod, "call_llm", fake_call)
    writer = _bare_writer()
    writer.draft_section(
        "Abstract",
        {"title": "T", "description": "d"},
        {"research_questions": [], "expected_contributions": []},
        {},
        revision_feedback="FIX THE NUMBERS",
    )
    assert "REVISION REQUIRED" in captured["prompt"]
    assert "FIX THE NUMBERS" in captured["prompt"]

    captured.clear()
    writer.draft_section("Abstract", {"title": "T", "description": "d"}, {"expected_contributions": []}, {})
    assert "REVISION REQUIRED" not in captured.get("prompt", "")


def test_writer_introduction_uses_literature_evidence(monkeypatch):
    import agents.writer as writer_mod

    captured = {}

    def fake_call(prompt, **_kwargs):
        captured["prompt"] = prompt
        return "# Introduction\n\n" + "Grounded background with citations. " * 40

    monkeypatch.setattr(writer_mod, "call_llm", fake_call)
    writer = _bare_writer()
    topic = {
        "title": "Attention calibration",
        "description": "desc",
        "literature_evidence": [
            {"title": "Prior Work A", "abstract": "Attention efficiency limits", "doi": "10.1234/abc"}
        ],
    }
    writer.draft_section("Introduction", topic, {}, {})
    assert "Retrieved literature evidence" in captured["prompt"]
    assert "Attention" in captured["prompt"]
    assert "Never invent" in captured["prompt"]

    writer.draft_section("Abstract", topic, {"research_questions": []}, {})
    assert "Retrieved literature evidence" in captured["prompt"]
    assert "Never invent" in captured["prompt"]


def test_writer_results_prompt_requires_exact_numbers(monkeypatch):
    import agents.writer as writer_mod

    captured = {}

    def fake_call(prompt, **_kwargs):
        captured["prompt"] = prompt
        return "# Results\n\n" + "n=3 seeds, std 0.02, accuracy 0.85. " * 10

    monkeypatch.setattr(writer_mod, "call_llm", fake_call)
    writer = _bare_writer()
    writer.draft_section(
        "Results",
        {"title": "T"},
        {"expected_contributions": []},
        {"exp": {"aggregate_metrics": {"accuracy": {"mean": 0.85, "std": 0.02}}}},
    )
    prompt = captured["prompt"]
    assert "EXACTLY" in prompt
    assert "n=" in prompt
    assert "standard deviation" in prompt or "confidence interval" in prompt
    assert "falsifiable prediction" in prompt


def test_writer_literature_block_handles_missing_evidence():
    writer = _bare_writer()
    block = writer._literature_block({"title": "T", "literature_evidence": []})
    assert "Do not cite any paper" in block
    block2 = writer._literature_block({
        "literature_evidence": [{"title": "A", "abstract": "Neural scaling laws", "arxiv_id": "2401.00001"}]
    })
    assert "Neural" in block2


# ---------------------------------------------------------------------------
# Novelty-plagiarism gate
# ---------------------------------------------------------------------------

PRIOR_ABSTRACT = (
    "We study scaling laws for retrieval-augmented language models with sparse memory "
    "and show that adaptive retrieval improves accuracy on knowledge intensive tasks."
)


def test_novelty_overlap_flags_near_copy():
    from core.verification import novelty_overlap_check

    copied = (
        "We study scaling laws for retrieval-augmented language models with adaptive "
        "retrieval improving accuracy on knowledge intensive tasks"
    )
    sections = {
        "Abstract": copied,
        "Introduction": "We extend scaling laws for retrieval augmented models.",
    }
    topic = {
        "literature_evidence": [{"title": "Scaling Retrieval", "abstract": PRIOR_ABSTRACT}],
        "structured_hypothesis": {"closest_prior_work": {"title": "Scaling laws for retrieval"}},
    }
    result = novelty_overlap_check(sections, topic)
    assert not result["passed"]
    assert result["findings"]
    assert result["max_overlap"] > result["threshold"]


def test_novelty_overlap_passes_original_framing():
    from core.verification import novelty_overlap_check

    sections = {
        "Abstract": "We measure phase transitions in tiny synthetic classifiers under seed perturbations.",
        "Introduction": "This study evaluates stability of decision boundaries under bounded compute.",
    }
    topic = {
        "literature_evidence": [
            {"title": "Ocean current prediction", "abstract": "Drifting buoys measure salinity gradients across basins"}
        ]
    }
    result = novelty_overlap_check(sections, topic)
    assert result["passed"]


def test_novelty_overlap_skips_when_no_topic():
    from core.verification import novelty_overlap_check

    result = novelty_overlap_check({"Abstract": "text"}, None)
    assert result["passed"]
    assert result["max_overlap"] == 0.0


def test_final_manuscript_referee_flags_novelty_overlap(monkeypatch):
    from core import verification

    monkeypatch.setattr(verification, "call_llm", lambda *_a, **_k: '{"findings": []}')
    prior = (
        "Adaptive gradient clipping stabilizes deep network training under noisy labels "
        "with bounded memory"
    )
    sections = {
        "Abstract": (
            "# Abstract\n\nAdaptive gradient clipping stabilizes deep network training "
            "under noisy labels with bounded memory across many architectures."
        ),
        "Introduction": "We study adaptive gradient clipping for noisy label training with stability guarantees.",
        "Limitations": "Findings are limited to the bounded local protocol.",
    }
    topic = {"literature_evidence": [{"title": "Clipping", "abstract": prior}]}
    referee = verification.final_manuscript_referee(sections, plan={}, topic=topic)
    checks = [finding["check"] for finding in referee["findings"]]
    assert "novelty_overlap" in checks


def test_final_manuscript_referee_without_topic_has_no_novelty_finding(monkeypatch):
    from core import verification

    monkeypatch.setattr(verification, "call_llm", lambda *_a, **_k: '{"findings": []}')
    sections = {
        "Abstract": "Stability of bounded classifiers under perturbations.",
        "Introduction": "Synthetic study of seed perturbation effects.",
        "Limitations": "Bounded local protocol only.",
    }
    referee = verification.final_manuscript_referee(sections, plan={}, topic=None)
    checks = [finding["check"] for finding in referee["findings"]]
    assert "novelty_overlap" not in checks
    assert "novelty" in referee


# ---------------------------------------------------------------------------
# Editor repair routing (WARA-style artifact repair)
# ---------------------------------------------------------------------------

def test_editor_repair_route_pure_logic():
    from core.workflow_nodes import editor_repair_route

    referee_fail = 'Cannot assemble manuscript: release referee failed: {"findings": []}'
    assert editor_repair_route(referee_fail, 0) == "repair"
    assert editor_repair_route(referee_fail, 1) == "repair"  # §5.3/R8: second repair allowed (bound 2)
    assert editor_repair_route(referee_fail, 2) == "terminal"
    assert editor_repair_route("Cannot assemble manuscript from failed experiments: x", 0) == "terminal"


class _RefereeFailingEditor:
    def __init__(self, context=None):
        pass

    def create_final_paper(self, *args, **kwargs):
        raise RuntimeError('Cannot assemble manuscript: release referee failed: {"findings": [{"check": "citations"}]}')


class _FailedExperimentEditor:
    def __init__(self, context=None):
        pass

    def create_final_paper(self, *args, **kwargs):
        raise RuntimeError("Cannot assemble manuscript from failed experiments: toy")


def test_editing_node_routes_referee_failure_to_repair(monkeypatch):
    import core.workflow_nodes as wn
    from core.state import initialize_state

    monkeypatch.setattr(wn, "EditorAgent", _RefereeFailingEditor)

    state = initialize_state()
    state.update({
        "selected_topic": {"title": "T"},
        "plan": {"sections": []},
        "draft_sections": {"Results": "clean"},
        "engineer_outputs": {},
    })
    state = wn.editing_node(state)
    assert state["current_phase"] == "writing_results"
    assert state["editor_repair_count"] == 1
    assert state["editor_repair_findings"]
    assert not state.get("terminal_error")

    # Second referee failure → second (final) repair pass (§5.3/R8 bound 2).
    state["current_phase"] = "editing"
    state = wn.editing_node(state)
    assert state["current_phase"] == "writing_results"
    assert state["editor_repair_count"] == 2
    assert not state.get("terminal_error")

    # Third referee failure → terminal.
    state["current_phase"] = "editing"
    state = wn.editing_node(state)
    assert state["current_phase"] == "complete"
    assert state.get("terminal_error")
    assert state["editor_repair_count"] == 2


def test_editing_node_failed_experiments_stay_terminal(monkeypatch):
    import core.workflow_nodes as wn
    from core.state import initialize_state

    monkeypatch.setattr(wn, "EditorAgent", _FailedExperimentEditor)
    state = initialize_state()
    state.update({
        "selected_topic": {"title": "T"},
        "plan": {"sections": []},
        "draft_sections": {"Results": "x"},
        "engineer_outputs": {},
    })
    state = wn.editing_node(state)
    assert state["current_phase"] == "complete"
    assert state.get("terminal_error")
    assert state["editor_repair_count"] == 0


# ---------------------------------------------------------------------------
# Targeted redraft + narrative revision loop
# ---------------------------------------------------------------------------

def test_results_redraft_carries_check_failures_into_prompt(monkeypatch):
    """A failing results section is re-drafted with its specific failure feedback."""
    import main as main_module
    from core.state import initialize_state

    feedbacks = []
    n_calls = [0]

    class FakeWriter:
        def __init__(self, context=None):
            pass

        def draft_section(self, section, *args):
            revision = args[3] if len(args) >= 4 else None
            feedbacks.append(revision)
            if revision is None and feedbacks.count(None) == 1:
                return "# Results\n\nAccuracy was 95.0% with n=3 and std 0.02."
            return "# Results\n\nAccuracy was 0.85 with n=3 seeds and standard deviation 0.02."

    monkeypatch.setattr(main_module, "WriterAgent", FakeWriter)
    state = initialize_state()
    state.update({
        "selected_topic": {"title": "T", "description": "d"},
        "plan": {"sections": ["Results"]},
        "engineer_outputs": {"exp": {"aggregate_metrics": {"accuracy": {"mean": 0.85, "std": 0.02}}}},
    })
    result = main_module.write_results_sections(state)
    assert result["results_redraft_count"] == 1
    # First draft call has no feedback; the redraft call carries the failure.
    assert feedbacks[0] is None
    redraft_feedbacks = [fb for fb in feedbacks[1:] if fb]
    assert redraft_feedbacks and all(
        "Untraceable numeric claims" in (fb or "") for fb in redraft_feedbacks
    )
    # The re-check happens on the next pass through the node.
    result = main_module.write_results_sections(result)
    check = result["results_verification"]["Results"]
    assert check["passed"], result.get("meta_feedback")
    assert result["current_phase"] == "supervision"


def test_narrative_revision_loop_injects_supervisor_feedback(monkeypatch):
    import core.workflow_nodes as wn
    from core.state import initialize_state

    calls = []

    class FakeWriter:
        def __init__(self, context=None):
            pass

        def draft_section(self, section, *args):
            feedback = args[3] if len(args) >= 4 else None
            calls.append({"section": section, "feedback": feedback})
            return "# Introduction\n\n" + ("Revised prose with specifics. " * 60) + (feedback or "")

    monkeypatch.setattr(wn, "WriterAgent", FakeWriter)

    state = initialize_state()
    state.update({
        "selected_topic": {"title": "T", "description": "d"},
        "plan": {"sections": [{"name": "Introduction"}]},
        "draft_sections": {"Introduction": "# Introduction\n\nOld draft text that scored poorly."},
        "iteration": 1,
        "supervisor_scores": {"Introduction": 4.0},
        "supervisor_feedback": {"Introduction": "BE MORE SPECIFIC"},
        "narrative_revision_count": 0,
    })
    state = wn.write_narrative_sections(state)
    assert len(calls) == 1
    assert "BE MORE SPECIFIC" in calls[0]["feedback"]
    assert state["narrative_revision_count"] == 1

    # Lab upgrade: revision budget is 3 (was 1). Second + third passes still
    # re-draft; the fourth is blocked.
    state["current_phase"] = "writing_narrative"
    state = wn.write_narrative_sections(state)
    assert len(calls) == 2
    assert state["narrative_revision_count"] == 2
    state["current_phase"] = "writing_narrative"
    state = wn.write_narrative_sections(state)
    assert len(calls) == 3
    assert state["narrative_revision_count"] == 3
    state["current_phase"] = "writing_narrative"
    state = wn.write_narrative_sections(state)
    assert len(calls) == 3
    assert state["narrative_revision_count"] == 3


def test_narrative_first_pass_does_not_redraft():
    import core.workflow_nodes as wn
    from core.state import initialize_state

    calls = []

    class FakeWriter:
        def __init__(self, context=None):
            pass

        def draft_section(self, section, *args):
            calls.append({"section": section, "feedback": args[3] if len(args) >= 4 else None})
            return "# Introduction\n\n" + "First-pass draft content. " * 40

    import core.workflow_nodes as wn_mod
    original = wn_mod.WriterAgent
    wn_mod.WriterAgent = FakeWriter
    try:
        state = initialize_state()
        state.update({
            "selected_topic": {"title": "T", "description": "d"},
            "plan": {"sections": [{"name": "Introduction"}]},
            "draft_sections": {},
            "iteration": 0,
        })
        state = wn.write_narrative_sections(state)
        assert len(calls) == 1
        assert calls[0]["feedback"] is None
        assert state["narrative_revision_count"] == 0
    finally:
        wn_mod.WriterAgent = original


# ---------------------------------------------------------------------------
# LLM failure visibility
# ---------------------------------------------------------------------------

def test_run_tracker_stats_include_llm_failures():
    from core.run_log import RunTracker

    tracker = RunTracker(run_id="llmfail-test")
    assert "llm_failures" in tracker.stats
    tracker.bump("llm_failures")
    assert tracker.stats["llm_failures"] == 1


# ---------------------------------------------------------------------------
# Round 2: engineer guidance, supervisor checklist, screener self-correction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Round 2: engineer guidance, supervisor checklist, screener self-correction
# ---------------------------------------------------------------------------

def test_engineer_refine_hint_maps_failure_categories():
    from agents.engineer import EngineerAgent

    timeout_hint = EngineerAgent._error_category_hint("Execution timed out after 30 seconds")
    assert "time budget" in timeout_hint
    import_hint = EngineerAgent._error_category_hint("ImportError: No module named 'requests'")
    assert "Only these imports are allowed" in import_hint
    assert EngineerAgent._error_category_hint("unrelated failure") == ""


def test_engineer_code_prompt_carries_refine_feedback(monkeypatch):
    import agents.engineer as engineer_mod
    from agents.engineer import EngineerAgent

    captured = {}

    def fake_call(prompt, **_kwargs):
        captured["prompt"] = prompt
        return "x = 1"

    monkeypatch.setattr(engineer_mod, "call_llm", fake_call)
    eng = EngineerAgent.__new__(EngineerAgent)
    eng._generate_experiment_code({"name": "toy", "refine_feedback": "PRIOR RUN TIMED OUT"})
    assert "PRIOR RUN" in captured["prompt"].upper()
    assert "Refine feedback" in captured["prompt"]


def test_supervisor_checklist_requires_falsifiability_verdict():
    from agents.supervisor import REVIEW_CHECKLIST

    assert any("falsifiable prediction" in item and "verdict" in item for item in REVIEW_CHECKLIST)


class _FakeVectorMemory:
    def get_prompt_context(self, namespace=None, k=8, **_kwargs):
        return [
            {
                "run_id": "r1",
                "outcome_status": "released",
                "signal": {"objection_type": "feasibility", "severity": 3, "resolution_status": "unresolved"},
            }
        ]


def test_proposer_argument_prompt_includes_evidence_and_prior_tags(monkeypatch):
    import agents.hypothesis_debate as hd
    from agents.hypothesis_debate import ProposerAgent

    captured = {}

    def fake_call(prompt, **_kwargs):
        captured["prompt"] = prompt
        return "Proposer builds a grounded argument with a falsifiable prediction and cited evidence."

    monkeypatch.setattr(hd, "call_llm", fake_call)
    proposer = ProposerAgent.__new__(ProposerAgent)
    proposer.context = None
    proposer.client = None
    proposer.vector_memory = _FakeVectorMemory()
    topic = {
        "title": "Attention calibration",
        "description": "Study attention calibration on synthetic data",
        "literature_evidence": [
            {"title": "Calibration Prior", "abstract": "Temperature scaling limits attention calibration"}
        ],
        "structured_hypothesis": {"research_question": "Does calibration help?"},
    }
    argument = proposer.build_argument(topic)
    assert argument
    assert "Calibration Prior" in captured["prompt"]
    assert "preempt" in captured["prompt"].lower()
    assert "Structured Hypothesis Contract" in captured["prompt"]


# ---------------------------------------------------------------------------
# 10. Graph editing → writing_results repair routing
# ---------------------------------------------------------------------------

def test_graph_routes_editing_repair_to_writing_results():
    from core.workflow import create_research_graph
    from core.state import initialize_state

    executed = []

    def record_node(name):
        def _node(state):
            executed.append(name)
            return state
        return _node

    def route_always_continue(state):
        return "continue"

    nodes = {name: record_node(name) for name in [
        "topic_discovery", "hypothesis_debate", "planning",
        "terminal_planning_failure", "data_validation",
        "writing_narrative", "engineering", "independent_validation",
        "writing_results", "supervision", "meta_evaluation",
        "editing", "reset", "write_plan",
    ]}
    nodes["should_reset"] = route_always_continue
    nodes["should_continue"] = route_always_continue
    nodes["is_valid_plan"] = route_always_continue

    graph = create_research_graph(nodes).compile()

    state = initialize_state()
    state["current_phase"] = "editing"
    state["editor_repair_count"] = 1
    state["editor_repair_findings"] = "Fix references"
    state["supervisor_feedback"] = {"results": 9.0, "originality": 9.0}
    state["current_results"] = "Results v1"
    result = graph.invoke(state)
    assert "writing_results" in executed


def test_graph_routes_editing_to_end_when_no_repair():
    from core.workflow import create_research_graph
    from core.state import initialize_state

    executed = []

    def record_node(name):
        def _node(state):
            executed.append(name)
            return state
        return _node

    def route_always_continue(state):
        return "continue"

    nodes = {name: record_node(name) for name in [
        "topic_discovery", "hypothesis_debate", "planning",
        "terminal_planning_failure", "data_validation",
        "writing_narrative", "engineering", "independent_validation",
        "writing_results", "supervision", "meta_evaluation",
        "editing", "reset", "write_plan",
    ]}
    nodes["should_reset"] = route_always_continue
    nodes["should_continue"] = route_always_continue
    nodes["is_valid_plan"] = route_always_continue

    graph = create_research_graph(nodes).compile()

    state = initialize_state()
    state["current_phase"] = "editing"
    state["editor_repair_count"] = 0
    state["supervisor_feedback"] = {"results": 9.0, "originality": 9.0}
    state["current_results"] = "Results v1"
    result = graph.invoke(state)
    editing_idx = executed.index("editing")
    results_after_editing = [
        i for i, n in enumerate(executed) if n == "writing_results" and i > editing_idx
    ]
    assert not results_after_editing, "writing_results ran after editing on non-repair path"


# ---------------------------------------------------------------------------
# 11. Generic section revision feedback reach
# ---------------------------------------------------------------------------

def test_generic_section_receives_revision_feedback(monkeypatch):
    from agents.writer import WriterAgent

    captured = {}

    def fake_call(prompt, **_kwargs):
        captured["prompt"] = prompt
        return "x " * 200  # enough to pass min-char check

    monkeypatch.setattr("agents.writer.call_llm", fake_call)
    writer = WriterAgent.__new__(WriterAgent)
    writer._active_revision_feedback = "Fix vague claims"
    writer._section_memory = {}

    content = writer._draft_generic_section(
        "Future Work",
        {"title": "Test", "description": "Test topic"},
        {"title": "Plan", "experiments": []},
        {},
    )
    assert content
    assert "Fix vague claims" in captured["prompt"]
