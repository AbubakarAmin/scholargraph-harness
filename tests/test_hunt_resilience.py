"""Regression tests for topic-hunter yield resilience.

Covers the fixes for the Sep-30 run where one seed retrieved 278 papers but
kept 0 gaps from 12 generated: generation burned the whole seed time budget
and the gate chain then discarded everything unexamined on its first deadline
check (plus abstract coverage helpers that raise evidence density).
"""

from unittest.mock import patch

from agents import topic_hunter
from tests.test_topic_hunter_features_8_15 import _make_agent


def _test_agent(**overrides):
    agent = _make_agent()
    agent.runtime_config.research_domain = "ml"
    # Hermetic I/O boundary (mirrors test_topic_hunter_features_8_15.py).
    agent.runtime_config.capability_first_dataset_scoping_enabled = False
    agent.runtime_config.structural_gap_mining_enabled = False
    agent.runtime_config.sparsity_matrix_enabled = False
    agent.runtime_config.contradiction_mining_enabled = False
    agent.runtime_config.replication_target_mining_enabled = False
    agent.runtime_config.negative_result_seeding_enabled = False
    agent.runtime_config.persona_ensemble_enabled = False
    agent.runtime_config.seed_strategy_elo_enabled = False
    agent.runtime_config.hyde_enabled = False
    agent.runtime_config.multi_hop_retrieval_enabled = False
    agent.runtime_config.topic_fallback_enabled = False
    for key, value in overrides.items():
        setattr(agent.runtime_config, key, value)
    return agent


def _gap(title, desc="a sufficiently long description for quality gates"):
    return {"title": title, "description": desc}


def test_iter_candidate_gaps_is_lazy():
    """The second persona's LLM call must not fire until batch 1 is consumed."""
    agent = _test_agent()
    calls = []

    def fake_llm(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return "p1"
        return "p2"

    def fake_parse(raw):
        if raw == "p1":
            return {"gaps": [_gap("Gap One")]}
        return {"gaps": [_gap("Gap Two")]}

    used = {"count": 0, "generation": 0, "gate": 0}

    def check_budget(kind="gate"):
        used["count"] += 1
        return True

    sink = []
    with patch.object(topic_hunter, "call_llm", side_effect=fake_llm), patch.object(
        topic_hunter, "parse_json_from_llm", side_effect=fake_parse
    ):
        stream = agent._iter_candidate_gaps(
            base_prompt="BASE",
            seed_hint="seed",
            use_persona_ensemble=True,
            persona_count=2,
            check_budget=check_budget,
            used_counter=used,
            deadline_reached=lambda stage: False,
            gaps_sink=sink,
        )
        first = next(stream)
        assert first["title"] == "Gap One"
        assert len(calls) == 1, "persona 2 must not be generated before batch 1 is evaluated"
        rest = list(stream)
        assert [g["title"] for g in rest] == ["Gap Two"]
        assert len(calls) == 2
        assert [g["title"] for g in sink] == ["Gap One", "Gap Two"]


def test_iter_candidate_gaps_first_persona_wins_dedup():
    """Same title from two personas is yielded once (first occurrence wins)."""
    agent = _test_agent()
    # Fresh objects per call: persona tagging mutates the gap dicts, so a
    # shared return_value would alias both personas onto one object.
    with patch.object(
        topic_hunter, "call_llm", side_effect=["x", "y"]
    ), patch.object(
        topic_hunter,
        "parse_json_from_llm",
        side_effect=lambda _raw: {"gaps": [_gap("Same Title")]},
    ):
        out = list(
            agent._iter_candidate_gaps(
                base_prompt="B",
                seed_hint="s",
                use_persona_ensemble=True,
                persona_count=2,
                check_budget=lambda kind="gate": True,
                used_counter={"count": 0},
                deadline_reached=lambda stage: False,
                gaps_sink=[],
            )
        )
    assert len(out) == 1
    assert out[0]["persona"] == "skeptic"


def test_backfill_missing_abstracts_joins_sibling_rows():
    """Abstracts are recovered across DOI/arXiv-titled duplicates, then ordered first."""
    agent = _test_agent()
    papers = [
        {"title": "Score centering in off policy reinforcement learning", "abstract": "", "doi": "10.1/abc"},
        {"title": "score centering in off policy reinforcement learning", "abstract": "We study score centering.", "arxiv_id": "2601.00001"},
        {"title": "   ", "abstract": "   ", "doi": "10.2/xyz"},
        {"title": "Unrelated work without any abstract anywhere", "abstract": ""},
    ]
    filled = agent._backfill_missing_abstracts(papers)
    assert filled == 1
    assert papers[0]["abstract"] == "We study score centering."
    # Abstract-bearing rows come first (stable), whitespace-only counts as missing.
    assert papers[0]["abstract"].strip() != ""
    assert papers[1]["abstract"].strip() != ""
    assert all(isinstance(p, dict) for p in papers)


class _Clock:
    """Scripted monotonic clock: frozen until released."""

    def __init__(self):
        self.now = 0.0
        self.released = False

    def monotonic(self):
        return 1000.0 if self.released else self.now


def _mocked_hunt(agent, mock_llm, mock_parse, clock):
    """Run _hunt_once with every I/O boundary mocked (mirrors existing tests)."""
    fake_paper = {
        "title": "machine learning validation",
        "abstract": "We validate ml models",
        "doi": "",
        "arxiv_id": "",
        "cited_by_count": 0,
    }
    formalized = {
        "hypothesis": "X improves Y",
        "falsification_condition": "No improvement",
        "dependent_variables": ["accuracy"],
        "research_question": "Does X improve Y?",
        "minimum_viable_experiment": {
            "dataset": "bundled_synthetic",
            "models": ["logistic_regression"],
            "conditions": ["control", "treatment"],
            "metrics": ["accuracy"],
            "seeds": 3,
        },
    }
    screen_calls = []

    def fake_screen(gap, evidence, citation_gap_signal=0.0):
        screen_calls.append(gap.get("title"))
        if len(screen_calls) >= 1:
            clock.released = True  # time expires right after the first evaluation
        return {"status": "PASS", "gap_type": "supported"}

    with patch("agents.topic_hunter.CrossRunMemory") as MockCRM, patch.object(
        topic_hunter, "call_llm", mock_llm
    ), patch.object(topic_hunter, "parse_json_from_llm", mock_parse), patch.object(
        topic_hunter, "generate_embedding", return_value=[0.1] * 10
    ), patch.object(
        topic_hunter, "build_cross_paper_evidence_map", return_value={"bridges": []}
    ), patch.object(
        topic_hunter,
        "validate_candidate_bridge_claim",
        return_value={"valid": True, "bridge_ids": []},
    ), patch.object(
        topic_hunter, "validate_topic_admission", return_value={"admitted": True, "errors": []}
    ), patch.object(
        agent, "_generate_hyde_abstract", return_value=None
    ), patch.object(
        agent, "_build_arxiv_queries", return_value=["q"]
    ), patch.object(
        agent, "_build_openalex_queries", return_value=[("q", {})]
    ), patch.object(
        agent, "_preflight_dedup", side_effect=lambda q, _: q
    ), patch.object(
        agent, "search_openalex", return_value=[fake_paper]
    ), patch.object(
        agent, "search_arxiv", return_value=[]
    ), patch.object(
        agent, "fetch_citation_graph", return_value=None
    ), patch.object(
        agent, "screen_research_gap", side_effect=fake_screen
    ), patch.object(
        agent,
        "evaluate_layered_novelty",
        return_value={"reject": False, "verdict": "NOVEL", "max_similarity": 0.0},
    ), patch.object(
        agent, "feasibility_filter", return_value={"ok": True, "reasons": []}
    ), patch.object(
        agent, "formalize_hypothesis", return_value=formalized
    ), patch(
        "core.sources_s2_bulk.search_s2_bulk", return_value=[]
    ), patch(
        "agents.topic_hunter.time", clock
    ):
        crm_inst = MockCRM.return_value
        crm_inst.get_prompt_context.return_value = []
        crm_inst.excluded_topic_titles.return_value = []
        crm_inst.get_negative_result_lessons.return_value = []
        crm_inst.record_rejection.return_value = None
        return agent._hunt_once(domain="ml", seed_hint="test seed", seed_strategy="kind_bias")


def test_hunt_once_keeps_first_batch_when_time_expires_mid_generation():
    """Interleaving: batch 1 is judged before persona 2 is even generated.

    Old behavior: both personas generate first, then the gate chain breaks on
    its first deadline check and everything is discarded (0 kept, 2 LLM calls
    wasted). New behavior: persona 2 is never generated and batch 1 survives.
    """
    from unittest.mock import MagicMock

    agent = _test_agent(persona_ensemble_enabled=True, persona_count=2)
    # Pin the seed deadline the _Clock script depends on (M0 L21 unifies the
    # code fallback to 1200; this test's expiry script needs deadline=180).
    agent.runtime_config.topic_seed_timeout_sec = 180
    mock_llm = MagicMock(
        side_effect=[
            "p1",
            "p2-never",  # must never be consumed
        ]
    )
    # Gap text must share vocabulary with the fake retrieved paper
    # ("machine learning validation / we validate ml models") to clear the
    # grounding pre-filter and reach the screener.
    mock_parse = MagicMock(
        side_effect=[
            {"gaps": [_gap("Machine learning validation gap", "validating ml models thoroughly")]},
            {"gaps": [_gap("Machine learning validation followup", "validating ml models further")]},
        ]
    )
    clock = _Clock()
    result = _mocked_hunt(agent, mock_llm, mock_parse, clock)
    assert len(result) == 1, f"expected the judged batch-1 gap to survive, got {result}"
    assert result[0]["title"] == "Machine learning validation gap"
    assert mock_llm.call_count == 1, "expired budget must not start persona 2's call"
