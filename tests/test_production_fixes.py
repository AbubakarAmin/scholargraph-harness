"""Production fixes from the b099dcaa/c22341c4 incident review.

Locks in (offline, mocked):
- P0-1: proposer LLM infra (truncation/transport/404) yields an infra
  DebateResult instead of terminal-killing the run; no Elo/memory poison.
- P0-3: retired model ids fail fast in validate_config.
- P1-2: per-seed seal/load roundtrip for killed-run resume.
- P0-2: deterministic formalize pre-gate + delta-contract merge.
- P0-1b: infra topic rejections are excluded from the resurface filter.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents.hypothesis_debate import (
    HypothesisDebateSystem,
    ProposerAgent,
    _classify_infra_error,
)
from core.llm import LLMUnavailable


def _make_proposer():
    agent = ProposerAgent.__new__(ProposerAgent)
    agent.context = None
    agent.client = None
    agent.vector_memory = type("Mem", (), {"get_prompt_context": lambda *a, **k: []})()
    agent._proposer_invalid = False
    agent._proposer_error_code = ""
    return agent


def test_classify_infra_error_codes():
    assert _classify_infra_error(LLMUnavailable("LLM truncated with no content (finish_reason=length)")) == "truncation_empty"
    assert _classify_infra_error(ConnectionError("getaddrinfo failed")) == "transport"
    assert _classify_infra_error(Exception("404 NOT_FOUND model no longer available")) == "model_404"
    assert _classify_infra_error(Exception("LLM blocked (finish_reason=['SAFETY'])")) == "blocked"


def test_proposer_truncation_sets_infra_not_raise(monkeypatch):
    agent = _make_proposer()

    def _raise(*_a, **_k):
        raise LLMUnavailable("LLM truncated with no content (finish_reason=length)")

    monkeypatch.setattr("agents.hypothesis_debate.call_llm", _raise)
    out = agent.build_argument({"title": "t", "description": "d"})
    assert out == ""
    assert agent._proposer_invalid is True
    assert agent._proposer_error_code == "truncation_empty"


def test_conduct_debate_proposer_infra_skips_moderator(monkeypatch):
    system = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    system.context = None
    system.runtime_config = SimpleNamespace(
        debate_min_rounds=2, debate_max_rounds=2,
        get_ensemble_models=lambda: [], resolve_model=lambda _t="default": "m",
    )
    system.elo = MagicMock()
    proposer = _make_proposer()
    system.proposer = proposer
    system.challenger = MagicMock()
    moderator = MagicMock()
    moderator.evaluate_debate.side_effect = AssertionError("moderator must not run on infra")
    system.moderator = moderator

    def _raise(*_a, **_k):
        raise LLMUnavailable("LLM truncated with no content (finish_reason=length)")

    monkeypatch.setattr("agents.hypothesis_debate.call_llm", _raise)
    result = system.conduct_debate({"title": "infra topic"})
    assert result.infra_failure is True
    assert result.passed is False
    assert result.moderator_decision == "INFRA_FAIL"
    moderator.evaluate_debate.assert_not_called()
    system.elo.update.assert_not_called()


def test_conduct_debate_mock_proposer_not_misread_as_infra():
    """MagicMock doubles auto-create _proposer_invalid as truthy — the
    conduct_debate guard must use `is True` so mocked proposers still debate."""
    system = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    mock_cfg = MagicMock()
    mock_cfg.debate_min_rounds = 2
    mock_cfg.debate_max_rounds = 2
    mock_cfg.get_ensemble_models.return_value = []
    mock_cfg.resolve_model.return_value = "m"
    system.context = MagicMock()
    system.context.config = mock_cfg
    system.runtime_config = mock_cfg
    mock_elo = MagicMock()
    mock_elo.update.return_value = 0.0
    system.elo = mock_elo
    mock_proposer = MagicMock()
    mock_proposer.build_argument.return_value = "argument"
    mock_proposer.respond_to_objections.return_value = "response"
    mock_challenger = MagicMock()
    mock_challenger.build_rebuttal.return_value = "rebuttal"
    mock_challenger._last_objections = []
    mock_challenger._challenger_invalid = False
    mock_challenger.followup_objections.return_value = []
    mock_moderator = MagicMock()
    mock_moderator.evaluate_debate.return_value = {
        "score": 8.0, "passed": True, "decision": "PASS",
        "needs_longer_debate": False, "reasoning": "r",
        "ensemble_scores": [8.0], "disagreement": 0.0,
        "valid_judge_responses": 1, "hard_gates": [],
        "unresolved_objections": [],
    }
    system.proposer = mock_proposer
    system.challenger = mock_challenger
    system.moderator = mock_moderator
    result = system.conduct_debate({"title": "t"})
    assert result.infra_failure is False
    assert result.passed is True


def test_revise_skips_infra_result():
    system = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    system.runtime_config = SimpleNamespace()
    system.challenger = SimpleNamespace(_challenger_invalid=False, _challenger_fallback_used=False)
    system.proposer = SimpleNamespace(_proposer_invalid=False)
    from agents.hypothesis_debate import DebateResult
    infra = DebateResult(topic="t", proposer_argument="", challenger_argument="",
                         moderator_decision="INFRA_FAIL", score=0.0, passed=False,
                         reasoning="infra", infra_failure=True)
    assert system.revise_topic_from_objections({"title": "t"}, infra) is None


def test_retired_model_rejected(monkeypatch):
    from core import config as config_mod
    monkeypatch.setattr(config_mod.config, "gemini_model", "gemini-2.0-flash")
    monkeypatch.setattr(config_mod.config, "openai_model", "gpt-4o-mini")
    monkeypatch.setattr(config_mod.config, "openalex_email", "real@example.org")
    try:
        config_mod.validate_config()
    except ValueError as exc:
        assert "retired" in str(exc).lower()
    else:
        raise AssertionError("retired model id must fail validate_config")


def test_excluded_titles_skips_infra(tmp_path):
    from core.run_log import CrossRunMemory
    mem = CrossRunMemory.__new__(CrossRunMemory)
    rows = [
        {"kind": "topic", "item": "Infra Topic", "reason": "infra_llm_unavailable", "meta": {}},
        {"kind": "topic", "item": "Science Fail", "reason": "failed_hypothesis_debate", "meta": {}},
    ]
    mem.load = lambda _kind, limit=100: rows
    excluded = CrossRunMemory.excluded_topic_titles(mem)
    assert "Science Fail" in excluded
    assert "Infra Topic" not in excluded


def test_seed_seal_roundtrip(tmp_path):
    from agents.topic_hunter import TopicHunterAgent
    agent = TopicHunterAgent.__new__(TopicHunterAgent)
    agent.runtime_config = SimpleNamespace(output_dir=str(tmp_path))
    seed = {"seed": "quantized inference latency", "strategy": "llm_diverse"}
    topics = [{"title": "T1", "description": "d"}]
    agent._seal_seed("run123", seed, topics)
    sealed, preloaded = agent._load_sealed_seeds("run123")
    assert len(sealed) == 1
    assert preloaded and preloaded[0]["title"] == "T1"


def test_formalize_pregate():
    from agents.topic_hunter import TopicHunterAgent
    agent = TopicHunterAgent.__new__(TopicHunterAgent)
    ok, _ = agent._formalize_pregate({
        "title": "Latency benchmarks are non-reproducible across hardware",
        "description": "We measure memory bandwidth and compare against baselines",
        "rationale": "r",
    })
    assert ok is True
    ok2, reason = agent._formalize_pregate({"title": "", "description": "", "rationale": ""})
    assert ok2 is False
    assert reason == "candidate_text_too_short"


def test_formalize_delta_merge(monkeypatch):
    """Phase-2 delta (no core re-emission) merges with the validated core."""
    from agents import topic_hunter as th
    agent = th.TopicHunterAgent.__new__(th.TopicHunterAgent)
    agent.runtime_config = SimpleNamespace()
    candidate = {"title": "T " * 20, "description": "D " * 20, "rationale": "measure and compare baselines",
                 "literature_evidence": []}
    phase1 = ('{"research_question": "Q", "hypothesis": "H", '
              '"falsification_condition": "F", "dependent_variables": ["m"]}')
    delta = ('{"baselines": ["b"], "metrics": ["m"], '
             '"minimum_viable_experiment": {"dataset": "bundled_synthetic"}}')
    with patch.object(th, "call_llm", side_effect=[phase1, delta]):
        result = agent.formalize_hypothesis(candidate, {"g": 1}, {"n": 1})
    assert result is not None
    assert result["hypothesis"] == "H"  # core preserved without re-emission
    assert result["baselines"] == ["b"]  # delta merged
    assert result["gap_report"] == {"g": 1}


# ── Run 21a145ad: S2 429 storm + persona truncation ──

def test_semantic_scholar_bulk_shares_semantic_scholar_bucket(monkeypatch):
    """semantic_scholar and semantic_scholar_bulk hit one upstream host: throttle one, both pause."""
    from core.api_gateway import APIGateway, RateLimitError
    monkeypatch.setattr("core.api_gateway.time.sleep", lambda s: None)
    gw = APIGateway()
    gw._THROTTLE_OPEN_THRESHOLD = 3

    def always_429():
        raise RateLimitError("semantic_scholar", 1.0)

    for _ in range(3):
        try:
            gw.request("semantic_scholar", always_429, retries=0)
        except RateLimitError:
            pass
    assert gw.is_available("semantic_scholar") is False
    # Same host → shared gate: bulk callers skip too instead of doubling load.
    assert gw.is_available("semantic_scholar_bulk") is False


def test_windowed_gate_trips_on_interleaved_storm(monkeypatch):
    """Alternating 429/success must still trip the gate (streak alone can't:
    one success resets it, so run 21a145ad's 8-minute storm never engaged)."""
    from core.api_gateway import APIGateway, RateLimitError
    monkeypatch.setattr("core.api_gateway.time.sleep", lambda s: None)
    gw = APIGateway()
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] % 2 == 1:
            raise RateLimitError("semantic_scholar", 1.0)
        return "ok"

    # 11 signals ending on a throttle: window holds 5 throttles, tail is a
    # throttle → gated (consecutive streak is only 1 — streak logic is blind).
    for _ in range(11):
        try:
            gw.request("semantic_scholar", flaky, retries=0)
        except RateLimitError:
            pass
    assert gw.is_available("semantic_scholar") is False


def test_successful_probe_reopens_after_storm(monkeypatch):
    """A successful probe must re-open traffic even with a loaded window
    (window tail shows success → no gate)."""
    from core.api_gateway import APIGateway, RateLimitError
    monkeypatch.setattr("core.api_gateway.time.sleep", lambda s: None)
    gw = APIGateway()
    for _ in range(10):
        try:
            gw.request("semantic_scholar", lambda: (_ for _ in ()).throw(RateLimitError("semantic_scholar", 1.0)), retries=0)
        except RateLimitError:
            pass
    assert gw.is_available("semantic_scholar") is False
    # Direct request bypasses the is_available gate; success decays the window.
    assert gw.request("semantic_scholar", lambda: "ok", retries=0) == "ok"
    assert gw.is_available("semantic_scholar") is True


def test_coupling_loop_fails_fast_on_s2_storm(monkeypatch):
    """Every-paper-429 must stop the loop after 3 misses, not burn 20 fetches."""
    from core import structural_gaps as sg
    # Hermetic S2 flag (v5 fix): the gateway is stubbed, so the
    # provider-enabled gate must not depend on the machine's .env/keys.json.
    from core import config as _config_mod
    monkeypatch.setattr(_config_mod.config, "semantic_scholar_enabled", True)
    calls = {"n": 0}

    class StubGW:
        def is_available(self, provider):
            return True

        def request(self, provider, fn, **kwargs):
            calls["n"] += 1
            raise sg.RateLimitError("semantic_scholar", 1.0)

    monkeypatch.setattr("core.api_gateway.get_gateway", lambda: StubGW())
    papers = [{"title": f"Paper {i} on ml validation", "doi": f"10.0/{i}",
               "arxiv_id": "", "abstract": "x"} for i in range(10)]
    out = sg.find_coupling_gaps(papers, s2_headers={})
    assert out == []
    assert calls["n"] == 3


def _persona_agent():
    from tests.test_topic_hunter_features_8_15 import _make_agent
    agent = _make_agent()
    agent.runtime_config.persona_ensemble_enabled = True
    agent.runtime_config.persona_count = 2
    return agent


def test_persona_truncation_retries_slimmed_once(monkeypatch):
    """Truncation_empty gets ONE slimmed retry; the persona survives."""
    from agents import topic_hunter as th
    agent = _persona_agent()
    calls = []

    def fake_llm(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            raise LLMUnavailable("LLM truncated with no content (finish_reason=length)")
        return '{"gaps": [{"title": "Recovered gap", "description": "measuring test accuracy with baselines"}]}'

    monkeypatch.setattr(th, "call_llm", fake_llm)
    used = {"count": 0, "generation": 0, "gate": 0}

    def check_budget(kind="gate"):
        used["count"] += 1
        return True

    gaps_sink: list = []
    gen = agent._iter_candidate_gaps(
        base_prompt="Seed angle: x. " + ("evidence " * 3000) + ' JSON: {"gaps": []}',
        seed_hint="test seed", use_persona_ensemble=True, persona_count=2,
        check_budget=check_budget, used_counter=used,
        deadline_reached=lambda stage: False, gaps_sink=gaps_sink,
    )
    gaps = list(gen)
    assert len(calls) == 3  # skeptic initial + exactly one slimmed retry, practitioner clean
    assert any(g.get("title") == "Recovered gap" for g in gaps)
    # Retry prompt is slimmed (evidence elided), schema preserved.
    assert len(calls[1]) < len(calls[0])
    assert "gaps" in calls[1]


def test_persona_transport_error_no_retry(monkeypatch):
    """Transport errors had gateway retries already — no extra persona retry."""
    from agents import topic_hunter as th
    agent = _persona_agent()
    calls = []

    def fake_llm(prompt, **kwargs):
        calls.append(prompt)
        raise ConnectionError("getaddrinfo failed")

    monkeypatch.setattr(th, "call_llm", fake_llm)
    used = {"count": 0, "generation": 0, "gate": 0}

    def check_budget(kind="gate"):
        used["count"] += 1
        return True

    gaps_sink: list = []
    gen = agent._iter_candidate_gaps(
        base_prompt="Seed angle: x.",
        seed_hint="test seed", use_persona_ensemble=True, persona_count=2,
        check_budget=check_budget, used_counter=used,
        deadline_reached=lambda stage: False, gaps_sink=gaps_sink,
    )
    assert list(gen) == []
    assert len(calls) == 2  # one per persona, zero retries


def test_revise_repair_truncation_returns_none_not_raise(monkeypatch):
    """21a145ad: repair LLM truncation_empty crashed the tournament.

    revise_topic_from_objections must fail closed (None) so the tournament
    continues to the next candidate.
    """
    system = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    system.runtime_config = SimpleNamespace()
    system.challenger = SimpleNamespace(_challenger_invalid=False, _challenger_fallback_used=False)
    system.proposer = SimpleNamespace(_proposer_invalid=False)
    from agents.hypothesis_debate import DebateResult
    result = DebateResult(
        topic="t", proposer_argument="arg", challenger_argument="rebut",
        moderator_decision="FAIL", score=6.5, passed=False, reasoning="r",
        objections=[{"criterion": "baseline", "objection": "missing baseline",
                     "severity": 3, "status": "unresolved"}],
    )
    topic = {"title": "t", "structured_hypothesis": {"research_question": "q"}}

    def _raise(*_a, **_k):
        raise LLMUnavailable("LLM truncated with no content (finish_reason=length)")

    monkeypatch.setattr("agents.hypothesis_debate.call_llm", _raise)
    assert system.revise_topic_from_objections(topic, result) is None


def test_conduct_with_repair_survives_repair_infra(monkeypatch):
    """Repair infra must yield [first], never raise."""
    system = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    system.runtime_config = SimpleNamespace()
    from agents.hypothesis_debate import DebateResult
    first = DebateResult(
        topic="t", proposer_argument="a", challenger_argument="c",
        moderator_decision="FAIL", score=6.5, passed=False, reasoning="r",
        objections=[{"criterion": "baseline", "objection": "x",
                     "severity": 3, "status": "unresolved"}],
    )
    system._debate_once = lambda _t, _m="standard": first

    def _raise(_t, _r):
        raise LLMUnavailable("LLM truncated with no content (finish_reason=length)")

    system.revise_topic_from_objections = _raise
    out = system.conduct_with_repair({"title": "t"})
    assert out == [first]


def test_tournament_continues_after_matchup_crash():
    """One candidate raising must become infra, not abort the tournament."""
    system = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    system.runtime_config = SimpleNamespace()
    from agents.hypothesis_debate import DebateResult
    good = DebateResult(
        topic="good", proposer_argument="a", challenger_argument="c",
        moderator_decision="PASS", score=8.0, passed=True, reasoning="r",
    )
    calls = []

    def _fake(topic, mode="standard"):
        calls.append(topic.get("title"))
        if topic.get("title") == "bad":
            raise LLMUnavailable("LLM truncated with no content (finish_reason=length)")
        return [good]

    system.conduct_with_repair = _fake
    results = system.conduct_tournament([{"title": "bad"}, {"title": "good"}])
    assert [c for c in calls] == ["bad", "good"]
    assert any(getattr(r, "infra_failure", False) for r in results)
    assert any(getattr(r, "passed", False) for r in results)
