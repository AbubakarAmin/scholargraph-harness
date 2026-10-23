"""Regression tests for the 2026-09-19 run failures.

Log findings fixed here:
1. ``Hypothesis debate subsystem crashed: 'list' object has no attribute 'get'``
   — ChallengerAgent.build_rebuttal called .get on parse_json_from_llm output,
   which is a list when the model returns a bare JSON objections array.
2. ``malformed_rebuttal_json`` x2 -> ``rebuttal_invalid_after_retry`` —
   build_rebuttal retried blindly; now self-corrects via call_llm_json.
3. ``degenerate_followup`` x2 — followup_objections rejected valid bare-array
   payloads and failed closed.
"""

from agents.hypothesis_debate import ChallengerAgent
from core.utils import parse_json_from_llm


def _make_challenger():
    agent = ChallengerAgent.__new__(ChallengerAgent)
    agent.context = None
    agent.client = None
    agent.vector_memory = type("Mem", (), {"get_prompt_context": lambda *a, **k: []})()
    return agent


def _patch_challenger_env(monkeypatch, responses):
    calls = {"n": 0}

    def fake_llm(*_args, **_kwargs):
        calls["n"] += 1
        return responses.pop(0) if responses else responses[-1]

    monkeypatch.setattr("agents.hypothesis_debate.call_llm", fake_llm)
    monkeypatch.setattr("agents.hypothesis_debate.check_plan_feasibility", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "agents.hypothesis_debate.EloStore",
        lambda **_kwargs: type("E", (), {"get": lambda self, *_a, **_k: 1500})(),
    )
    return calls


def test_build_rebuttal_survives_bare_array_json(monkeypatch):
    """The exact crash from the run log: model returns a bare JSON array of
    objections; build_rebuttal must coerce, not crash with
    'list' object has no attribute 'get'."""
    agent = _make_challenger()
    array_payload = (
        '[{"criterion": "baseline", "objection": "no baseline comparison", '
        '"severity": 4}]'
    )
    _patch_challenger_env(monkeypatch, [array_payload])

    summary = agent.build_rebuttal({"title": "t", "description": "d"}, "argument")

    assert agent._challenger_invalid is False
    assert summary != "CHALLENGER_OUTPUT_INVALID"
    assert any(
        o["criterion"] == "baseline" and o["severity"] == 4
        for o in agent._last_objections
    )


def test_build_rebuttal_coerces_single_objection_dict(monkeypatch):
    agent = _make_challenger()
    payload = (
        '{"objections": {"criterion": "falsifiability", "objection": "vague", '
        '"severity": 3}, "summary_rebuttal": "weak"}'
    )
    _patch_challenger_env(monkeypatch, [payload])

    summary = agent.build_rebuttal({"title": "t", "description": "d"}, "argument")

    assert agent._challenger_invalid is False
    assert len(agent._last_objections) == 1
    assert agent._last_objections[0]["criterion"] == "falsifiability"
    assert agent._last_objections[0]["source"] == "challenger_audit"


def test_build_rebuttal_reasks_with_parse_error_feedback(monkeypatch):
    """Second attempt must carry parse-error feedback, not an identical re-roll."""
    agent = _make_challenger()
    prompts = []
    responses = ["this is prose, not json", '{"objections": [{"criterion": "baseline", "objection": "x", "severity": 2}]}']

    def fake_llm(prompt, **_kwargs):
        prompts.append(prompt)
        return responses.pop(0)

    monkeypatch.setattr("agents.hypothesis_debate.call_llm", fake_llm)
    monkeypatch.setattr("agents.hypothesis_debate.check_plan_feasibility", lambda *_a, **_k: [])

    agent.build_rebuttal({"title": "t", "description": "d"}, "argument")

    assert len(prompts) == 2
    assert "could not be parsed as JSON" in prompts[1]


def test_followup_objections_accepts_bare_array(monkeypatch):
    agent = _make_challenger()
    monkeypatch.setattr(
        "agents.hypothesis_debate.call_llm",
        lambda *_a, **_k: '[{"criterion": "baseline", "objection": "resolved now", "severity": 3, "status": "resolved"}]',
    )
    prior = [{"criterion": "baseline", "objection": "no baseline", "severity": 4, "status": "unresolved"}]

    updated = agent.followup_objections({"title": "t"}, "responses", prior)

    assert updated[0]["status"] == "resolved"
    # Prior objection text was rephrased by the LLM, so it is preserved unresolved.
    assert any(
        o["criterion"] == "baseline" and o["status"] == "unresolved" for o in updated[1:]
    )


def test_followup_objections_fail_closed_preserves_priors(monkeypatch):
    agent = _make_challenger()
    monkeypatch.setattr("agents.hypothesis_debate.call_llm", lambda *_a, **_k: "garbage")
    prior = [{"criterion": "soundness", "objection": "unfalsifiable", "severity": 4, "status": "unresolved"}]

    updated = agent.followup_objections({"title": "t"}, "responses", prior)

    assert updated[0]["status"] == "unresolved"
    assert updated[0]["objection"] == "unfalsifiable"


def test_parse_json_from_llm_is_type_safe():
    assert parse_json_from_llm(None) is None
    assert parse_json_from_llm(12345) is None
    assert parse_json_from_llm("[not closed") is None
    # Dual-type contract: bare arrays are returned as lists.
    assert parse_json_from_llm('before [{"a": 1}] after') == [{"a": 1}]
    assert parse_json_from_llm('{"k": 2}') == {"k": 2}


def test_normalize_objection_payload_shapes():
    from agents.hypothesis_debate import _normalize_objection_payload

    assert _normalize_objection_payload([{"criterion": "x"}]) == {"objections": [{"criterion": "x"}]}
    assert _normalize_objection_payload({"objections": {"criterion": "x"}}) == {"objections": [{"criterion": "x"}]}
    assert _normalize_objection_payload(None) == {}
    assert _normalize_objection_payload("junk") == {}
