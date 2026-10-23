"""Offline checks for the opt-in Gemma 4 generation profile."""

from __future__ import annotations

from types import SimpleNamespace


def test_gemma_4_is_the_default_and_all_tiers_inherit_it_when_unset():
    from core.config import Config

    cfg = Config(
        llm_provider="gemini",
        gemini_model="gemma-4-26b-a4b-it",
        llm_model_cheap="",
        llm_model_strong="",
        llm_model_judge="",
    )
    assert cfg.is_gemma_4_model()
    assert cfg.is_gemma_4_model("models/gemma-4-31b-it")
    assert not cfg.is_gemma_4_model("gemini-2.5-flash")
    assert {cfg.resolve_model(tier) for tier in ("cheap", "strong", "judge", "default")} == {
        "gemma-4-26b-a4b-it"
    }


def test_gemma_4_requests_enable_high_thinking_only_for_gemma(monkeypatch):
    import core.llm as llm_mod

    class FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            part = SimpleNamespace(text="ok")
            candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason="STOP")
            return SimpleNamespace(candidates=[candidate])

    fake = FakeModels()
    client = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
    client._gemini = SimpleNamespace(models=fake)
    monkeypatch.setattr(llm_mod.config, "gemini_model", "gemma-4-26b-a4b-it")
    monkeypatch.setattr(llm_mod.config, "gemma_thinking_level", "high", raising=False)

    assert client._chat_gemini("test", 0.1, "gemma-4-26b-a4b-it", 32, None) == "ok"
    assert fake.calls[-1]["config"]["thinking_config"] == {"thinking_level": "high"}

    assert client._chat_gemini("test", 0.1, "gemini-2.5-flash", 32, None) == "ok"
    assert "thinking_config" not in fake.calls[-1]["config"]
