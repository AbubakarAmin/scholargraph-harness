"""Tests for thinking-tag stripping in LLM output processing."""

import re
import json

_THINKING_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_thinking_tags(text: str) -> str:
    if not text:
        return text
    return _THINKING_TAG_RE.sub("", text).strip()


class TestThinkingTagStrip:
    def test_single_block_with_content_after(self):
        raw = (
            "<think>We need to generate a compelling realistic argument. "
            "The content is about Cross-Domain Feature Stability Analysis.</think>\n"
            "Here is my argument for the hypothesis."
        )
        clean = strip_thinking_tags(raw)
        assert "<thinking>" not in clean
        assert "Here is my argument" in clean

    def test_thinking_with_json_inside(self):
        raw = (
            "<think>Let me analyze... the key problem is feasibility.</think>\n"
            '{"objections": [{"criterion": "feasibility", "severity": 3}], '
            '"summary_rebuttal": "The hypothesis has feasibility issues."}'
        )
        clean = strip_thinking_tags(raw)
        assert "<thinking>" not in clean
        assert clean.startswith("{")
        assert clean.endswith("}")
        parsed = json.loads(clean)
        assert "objections" in parsed

    def test_thinking_with_nested_curly_braces(self):
        raw = (
            "<think>The function returns {\"key\": \"value\"} and outputs {confidence: 0.9}.</think>\n"
            '{"objections": []}'
        )
        clean = strip_thinking_tags(raw)
        assert "<thinking>" not in clean
        assert clean == '{"objections": []}'

    def test_multiple_thinking_blocks(self):
        raw = (
            "<think>First thought...</think>\n"
            "Some text in between\n"
            "<think>Second thought...</think>\n"
            "Final content."
        )
        clean = strip_thinking_tags(raw)
        assert "<thinking>" not in clean
        assert "Some text in between" in clean
        assert "Final content." in clean
        assert "<thinking>" not in clean

    def test_no_thinking_tags(self):
        raw = "This is a normal response without any thinking tags."
        clean = strip_thinking_tags(raw)
        assert clean == raw

    def test_empty_string(self):
        assert strip_thinking_tags("") == ""

    def test_none_passthrough(self):
        assert strip_thinking_tags(None) is None

    def test_only_thinking_tags(self):
        raw = "<think>Just thinking, no real content.</think>"
        clean = strip_thinking_tags(raw)
        assert clean == ""
        assert len(clean) == 0

    def test_case_insensitive(self):
        raw = "<think>Case test</think>Real content"
        clean = strip_thinking_tags(raw)
        assert clean == "Real content"

    def test_multiline_thinking_block(self):
        raw = (
            "<think>\n"
            "Line 1 of thinking\n"
            "Line 2 of thinking\n"
            "Line 3 of thinking\n"
            "</think>\n"
            "Actual output here."
        )
        clean = strip_thinking_tags(raw)
        assert "<thinking>" not in clean
        assert clean == "Actual output here."

    def test_parse_json_after_strip(self):
        raw = (
            "<think>Analyzing objection types...</think>\n"
            '{"objections": [{"criterion": "soundness", "objection": "test", '
            '"severity": 4, "status": "unresolved"}], "summary_rebuttal": "Found issues."}'
        )
        clean = strip_thinking_tags(raw)
        parsed = json.loads(clean)
        assert len(parsed["objections"]) == 1
        assert parsed["objections"][0]["criterion"] == "soundness"

    def test_proposer_pattern_from_logs(self):
        raw = (
            "<think>We need to generate a compelling realistic argument for the research proposal. "
            "The user wants a research proposer building a compelling, realistic argument. "
            "The content: \\\"Cross-Domain Feature Stability Analysis: Mapping Representation "
            "Drift Under Synthetic Domain Shifts\\\". "
            "The user has provided a structured hypothesis contract. "
            "We need to ground novelty claims in retrieved evidence.</think>\n"
            "## Hypothesis\n\n"
            "Cross-domain feature representations exhibit systematic drift under synthetic "
            "domain shifts, with predictable failure thresholds for linear correction methods."
        )
        clean = strip_thinking_tags(raw)
        assert "<thinking>" not in clean
        assert "## Hypothesis" in clean
        assert "Cross-domain feature" in clean

    def test_challenger_json_pattern_from_logs(self):
        raw = (
            "<think>The proposer's argument has several methodological issues. "
            "First, the falsification condition is overly strict. "
            "Second, we identify critical confounders.</think>\n"
            '{"objections": ['
            '{"criterion": "falsifiability", "objection": "Overly strict falsification", '
            '"severity": 4, "status": "unresolved", "source": "challenger_audit"},'
            '{"criterion": "confounder", "objection": "Uncontrolled capacity", '
            '"severity": 3, "status": "unresolved", "source": "challenger_audit"}'
            '], "summary_rebuttal": "Multiple methodological issues found."}'
        )
        clean = strip_thinking_tags(raw)
        assert "<thinking>" not in clean
        parsed = json.loads(clean)
        assert len(parsed["objections"]) == 2
        assert parsed["objections"][0]["criterion"] == "falsifiability"
        assert parsed["objections"][1]["criterion"] == "confounder"
