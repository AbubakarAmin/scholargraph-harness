"""
SupervisorAgent — hard deterministic checks first, LLM soft review last.
Citation grounding + statistical validity gate soft scores.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import config
from core.utils import log_agent_action, validate_math_expression, verify_math_derivation, parse_json_from_llm, call_llm_json
from core.llm import call_llm
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.memory import memory
from core.verification import hard_verify_section, verify_citations, classify_claim
from core.run_log import get_tracker
from core.research_db import research_db
from core.contracts import ExperimentOutput, VerificationReport

logger = logging.getLogger(__name__)


def _supervisor_system(prompt_name: str, **slots: Any) -> Dict[str, Any]:
    """System kwarg from the v2 registry (stub-safe: {} when unavailable)."""
    try:
        from core.prompts import render_system

        system = render_system(prompt_name, **slots)
        return {"system": system} if system else {}
    except Exception:
        return {}


REVIEW_CHECKLIST = (
    "every quantitative claim has a traceable artifact and uncertainty",
    "baselines and datasets match the committed experiment contract",
    "negative or inconclusive outcomes are reported",
    "the falsifiable prediction is explicitly evaluated and its verdict "
    "(supported / falsified / inconclusive) is stated with controls and robustness evidence",
    "limitations and reproducibility details are disclosed",
    "literature claims are supported by retrieved source text",
)


class SupervisorAgent:
    """Quality gate: hard checks dominate; LLM review is last filter only."""

    # R4 (§6.1): per-type score weights. The legacy single vector was
    # hard=0.35 / math=0.15 / code=0.15 / soft=0.20 / hall=0.15. The
    # experimental entry is byte-identical to that vector (no regression);
    # every other entry starts at legacy parity with its reason recorded in
    # SCORE_WEIGHT_REASONS (tune only with calibration evidence).
    SCORE_WEIGHTS_DEFAULT = {"hard": 0.35, "math": 0.15, "code": 0.15, "soft": 0.20, "hall": 0.15}
    # Lab calibration v1 (offline harness + gate semantics): experiment-grounded
    # types weight hard/code higher; synthesis types weight hard/hall higher and
    # down-weight code (no executable artifact). All rows sum to 1.0.
    PER_TYPE_WEIGHTS = {
        "experimental": {"hard": 0.35, "math": 0.15, "code": 0.15, "soft": 0.20, "hall": 0.15},
        "comparative": {"hard": 0.35, "math": 0.15, "code": 0.20, "soft": 0.15, "hall": 0.15},
        "empirical": {"hard": 0.40, "math": 0.15, "code": 0.15, "soft": 0.15, "hall": 0.15},
        "cause_effect": {"hard": 0.40, "math": 0.15, "code": 0.15, "soft": 0.15, "hall": 0.15},
        "review": {"hard": 0.45, "math": 0.05, "code": 0.05, "soft": 0.20, "hall": 0.25},
        "analytical": {"hard": 0.40, "math": 0.10, "code": 0.05, "soft": 0.20, "hall": 0.25},
        "argumentative": {"hard": 0.40, "math": 0.05, "code": 0.05, "soft": 0.25, "hall": 0.25},
        "descriptive": {"hard": 0.40, "math": 0.10, "code": 0.05, "soft": 0.20, "hall": 0.25},
        "survey": {"hard": 0.45, "math": 0.10, "code": 0.05, "soft": 0.15, "hall": 0.25},
        "case_study": {"hard": 0.45, "math": 0.05, "code": 0.05, "soft": 0.20, "hall": 0.25},
    }
    SCORE_WEIGHT_REASONS = {
        "experimental": "legacy parity: the calibrated single vector, unchanged",
        "comparative": "lab v1: code up (baselines must run), soft down",
        "empirical": "lab v1: hard up (outcomes + artifacts must trace)",
        "cause_effect": "lab v1: hard up (identification + verb governor dominate)",
        "review": "lab v1: hard/hall up (PRISMA + citation grounding), math/code down (no execution)",
        "analytical": "lab v1: hard/hall up (thesis triad + bridges), code down",
        "argumentative": "lab v1: soft/hall up (steelman quality), code down",
        "descriptive": "lab v1: hard up (verb-lint + CI gate), code down",
        "survey": "lab v1: hard up (instrument/nonresponse/synthetic gates), code down",
        "case_study": "lab v1: hard up (triangulation/rival gates), code down",
    }

    # R4 (§6.1): per-type pass thresholds. None means "fall back to the
    # runtime config supervisor_threshold" (default 8.5) — i.e. legacy
    # behavior. The experimental entry is None by construction (no regression).
    PER_TYPE_THRESHOLDS = {
        "experimental": None,
        "comparative": None,
        "empirical": None,
        "cause_effect": None,
        "review": None,
        "analytical": None,
        "argumentative": None,
        "descriptive": None,
        "survey": None,
        "case_study": None,
    }

    @classmethod
    def _normalize_paper_type(cls, paper_type: Any) -> str:
        ptype = str(paper_type or "experimental").strip().lower() or "experimental"
        if ptype == "case":
            ptype = "case_study"
        return ptype

    @classmethod
    def weights_for(cls, paper_type: Any) -> Dict[str, float]:
        """Per-type score weights; unknown types fall back to legacy default."""
        return dict(cls.PER_TYPE_WEIGHTS.get(cls._normalize_paper_type(paper_type), cls.SCORE_WEIGHTS_DEFAULT))

    def threshold_for(self, paper_type: Any) -> float:
        """Per-type pass threshold; None entries fall back to runtime config."""
        override = self.PER_TYPE_THRESHOLDS.get(self._normalize_paper_type(paper_type), None)
        return float(override) if override is not None else float(self.runtime_config.supervisor_threshold)

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        self.ledger = self.context.research_db if self.context else research_db
        self.feedback_memory = self.context.memory if self.context else memory
        self.math_checker = MathChecker()
        self.code_checker = CodeChecker()
        self.reviewer_bot = ReviewerBot()

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    # Paper types whose release rule centers on the thesis, not experiments (§7.1).
    THESIS_DISPATCH_TYPES = ("analytical", "argumentative", "descriptive", "review")

    def evaluate_section(
        self,
        section_name: str,
        content: str,
        engineer_outputs: Optional[Dict[str, ExperimentOutput]] = None,
        content_requirements: Optional[str] = None,
        paper_type: str = "experimental",
        thesis: Optional[Dict[str, Any]] = None,
        quarantine_singles: Optional[bool] = None,
        instrument: Optional[Dict[str, Any]] = None,
        survey_analysis: Optional[Dict[str, Any]] = None,
        case_protocol: Optional[Dict[str, Any]] = None,
        causal_claim: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        logger.info("Evaluating section: %s (paper_type=%s)", section_name, paper_type)
        log_agent_action("SupervisorAgent", "start_evaluation", {"section": section_name, "paper_type": paper_type})
        # R5 slice: thesis types quarantine single citation failures to the
        # repair pass (run continues); experimental types stay fail-closed
        # unless the caller explicitly opts in.
        if quarantine_singles is None:
            quarantine_singles = paper_type in self.THESIS_DISPATCH_TYPES

        # --- HARD CHECKS FIRST ---
        hard = hard_verify_section(
            content,
            engineer_outputs=engineer_outputs,
            section_name=section_name,
            content_requirements=content_requirements,
            quarantine_singles=bool(quarantine_singles),
        )
        # --- PER-TYPE DISPATCH (§7.1, M1): thesis/verb checks join the hard bundle ---
        type_passed, type_cap, type_feedback = self._per_type_section_checks(
            section_name, content, paper_type, thesis,
            instrument=instrument, survey_analysis=survey_analysis,
            case_protocol=case_protocol, causal_claim=causal_claim,
        )
        self._record_claim_evidence(section_name, content, hard, engineer_outputs)
        math_score, math_fb = self.math_checker.evaluate(content, section_name)
        code_score, code_fb = self.code_checker.evaluate(content, section_name)

        hard_bundle_score = min(hard["score"], math_score, code_score, type_cap)
        feedbacks = [
            f"HARD_CITATION_STATS: {hard['feedback']}",
            f"math_checker: {math_fb}",
            f"code_checker: {code_fb}",
            f"type_dispatch[{paper_type}]: {type_feedback}",
        ]
        hard_passed = bool(hard["passed"]) and bool(type_passed)
        logger.info("Hard checks passed=%s for section: %s (citation=%.1f, math=%.1f, code=%.1f, type=%s)", hard_passed, section_name, hard["score"], math_score, code_score, type_feedback)

        if not hard_passed:
            tracker = get_tracker()
            if tracker:
                tracker.bump("hard_check_fails")
            overall = min(hard_bundle_score, 4.0)
            overall_feedback = "HARD CHECK FAILED — deterministic checks block release.\n" + "\n".join(feedbacks)
            logger.info("Hard check FAILED for section %s, blocking release (score=%.1f)", section_name, overall)
            self.feedback_memory.add_feedback_entry(
                "SupervisorAgent",
                section_name,
                overall,
                overall_feedback,
                1,
                structured_signal={
                    "verdict": "block",
                    "score": float(overall),
                    "section": section_name,
                    "blocking": True,
                    "category": "hard_check_failed",
                },
                run_id=tracker.run_id if tracker else None,
                outcome_status="failed",
            )
            return overall, overall_feedback

        # --- SOFT LLM REVIEW LAST ---
        soft_score, soft_fb = self.reviewer_bot.evaluate(content, section_name)
        feedbacks.append(f"reviewer_bot: {soft_fb}")

        # Also run citation-aware hallucination soft check only after hard pass
        hall_score, hall_fb = self._soft_hallucination_check(content, section_name, hard)
        feedbacks.append(f"hallucination_checker: {hall_fb}")

        weights = self.weights_for(paper_type)
        overall = (
            weights["hard"] * hard["score"]
            + weights["math"] * math_score
            + weights["code"] * code_score
            + weights["soft"] * soft_score
            + weights["hall"] * hall_score
        )
        # d5b5a1db: honest-but-empty Results/Discussion (experiments: [] with no
        # engineer_outputs) scored 8.23/8.53 and hid the failed run behind high
        # soft marks. Cap experiment-requiring types at 6.0 when there is no
        # artifact to report — honesty must not outscore real work.
        try:
            _needs_data = str(paper_type or "").lower() in (
                "experimental", "comparative", "cause_effect", "empirical")
            if _needs_data and not (engineer_outputs or {}):
                if overall > 6.0:
                    feedbacks.append(
                        "empty_artifact_cap: no engineer_outputs for an "
                        "experiment-requiring type; capping 8.x soft score at 6.0 "
                        "(honest absence of data, not publishable evidence).")
                    overall = 6.0
        except Exception:
            pass
        overall_feedback = "\n".join(feedbacks)
        threshold = self.threshold_for(paper_type)
        verdict = "pass" if overall >= threshold else "revise"
        logger.info("Final score for %s: %.2f (threshold=%.1f, verdict=%s)", section_name, overall, threshold, verdict)
        logger.info("Feedback summary for %s: %s", section_name, overall_feedback[:500])

        tracker = get_tracker()
        self.feedback_memory.add_feedback_entry(
            "SupervisorAgent",
            section_name,
            overall,
            overall_feedback,
            1,
            structured_signal={
                "verdict": "pass" if overall >= threshold else "revise",
                "score": float(overall),
                "section": section_name,
                "blocking": False,
                "category": "supervisor_review",
            },
            run_id=tracker.run_id if tracker else None,
            outcome_status="released" if overall >= threshold else "revised",
        )
        log_agent_action("SupervisorAgent", "evaluation_complete", {
            "section": section_name,
            "overall_score": overall,
            "hard_passed": hard_passed,
            "paper_type": paper_type,
        })
        return overall, overall_feedback

    def _per_type_section_checks(
        self,
        section_name: str,
        content: str,
        paper_type: str,
        thesis: Optional[Dict[str, Any]],
        instrument: Optional[Dict[str, Any]] = None,
        survey_analysis: Optional[Dict[str, Any]] = None,
        case_protocol: Optional[Dict[str, Any]] = None,
        causal_claim: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, str]:
        """Per-type hard dispatch for one section (§7.1, M1 + M3 + M4).

        Returns (passed, score_cap, feedback). The cap joins the hard bundle
        (section-cap: a type failure caps the section, never the whole paper
        directly — aggregation is the supervision node's job, L12).
        """
        from agents.writer import lint_causal_verbs as _lint_verbs
        from core.verification import FAKE_FIELD_PHRASES as _fake_phrases

        def _fake_field_hits(text: str) -> List[str]:
            # R9 Methods allowlist: instrument/methods prose may name the
            # guard without tripping it.
            if "method" in str(section_name or "").lower():
                return []
            lowered = (text or "").lower()
            return [phrase for phrase in _fake_phrases if phrase in lowered]

        if paper_type == "descriptive":
            # Descriptive: causal-verb lint is a hard section gate (§7.3).
            hits = _lint_verbs(content)
            if hits:
                return False, 4.0, f"causal-verb lint failed: {', '.join(hits)}"
            ci_present = bool(re.search(r"confidence interval|\bCI\b|ci\s*[=:]|±", content, re.I))
            quant = bool(re.search(r"\d+(?:\.\d+)?\s*%|\b(?:mean|std)\b", content, re.I))
            if quant and not ci_present:
                return False, 4.0, "quantitative descriptive claims require confidence intervals"
            return True, 10.0, "descriptive dispatch ok (verb lint clean, CIs present or no quantities)"
        if paper_type in ("analytical", "argumentative"):
            # Thesis triad must exist before any section can pass (§7.1).
            # Bridge coverage itself is manuscript-level (release referee).
            if not isinstance(thesis, dict) or not str(thesis.get("claim") or "").strip():
                return False, 4.0, "thesis triad missing: no thesis claim for analytical/argumentative section"
            if paper_type == "argumentative" and section_name.lower() == "counterargument":
                # The counterargument must substantively address the steelman:
                # at least half of the steelman's content tokens appear here.
                steelman = str((thesis or {}).get("steelman") or "")
                tokens = [w for w in re.findall(r"[a-z0-9]+", steelman.lower()) if len(w) > 3]
                lowered = content.lower()
                hits = sum(1 for t in set(tokens) if t in lowered)
                if not tokens or hits / len(set(tokens)) < 0.5:
                    return False, 4.0, "counterargument section does not address the steelman"
            return True, 10.0, "thesis dispatch ok (triad present)"
        if paper_type == "review":
            # Review section gate (§7.1 M2): manuscript-level PRISMA /
            # completeness / recall / synthesis live in verify_review
            # (release referee + supervision_node gate). Section-level stays
            # permissive so narrative why-not-meta wording ("Meta-analysis
            # not performed ...") never trips a meta-claim heuristic — the
            # meta gate (≥3 commensurate effects + I²) is enforced on the
            # corpus synthesis, not on prose substrings.
            return True, 10.0, "review dispatch ok (manuscript gate owns PRISMA/completeness/recall/synthesis)"
        if paper_type == "survey":
            # Survey section gate (§7.1 M3): fake-field lint is hard;
            # quantitative claims need confidence intervals; instrument /
            # nonresponse / fishing / synthetic live in verify_survey
            # (manuscript gate + supervision_node gate).
            fake = _fake_field_hits(content)
            if fake:
                return False, 4.0, f"fake-field lint failed: {', '.join(fake)}"
            quant = bool(re.search(r"\d+(?:\.\d+)?\s*%|\b(?:mean|std)\b", content, re.I))
            ci_present = bool(re.search(r"confidence interval|\bCI\b|ci\s*[=:]|±", content, re.I))
            if quant and not ci_present:
                return False, 4.0, "quantitative survey claims require confidence intervals"
            return True, 10.0, "survey dispatch ok (section lint clean; manuscript gate owns instrument/nonresponse/synthetic)"
        if paper_type == "case_study":
            # Case section gate (§7.1/§4.9 M3): fake-field lint is hard;
            # triangulation + rival live in verify_case (manuscript gate).
            fake = _fake_field_hits(content)
            if fake:
                return False, 4.0, f"fake-field lint failed: {', '.join(fake)}"
            return True, 10.0, "case dispatch ok (manuscript gate owns triangulation/rival)"
        if paper_type == "empirical":
            # Empirical-secondary section gate (M3): fake-field lint is hard;
            # stats/dossier/code-claim checks stay in the shared hard bundle.
            fake = _fake_field_hits(content)
            if fake:
                return False, 4.0, f"fake-field lint failed: {', '.join(fake)}"
            return True, 10.0, "empirical dispatch ok (secondary-data gate owns protocol/hash)"
        if paper_type == "comparative":
            # Comparative section gate (§7.1 M4): fake-field lint is hard;
            # the baseline matrix + compulsory ablation live in
            # verify_comparative (manuscript gate + release referee).
            fake = _fake_field_hits(content)
            if fake:
                return False, 4.0, f"fake-field lint failed: {', '.join(fake)}"
            return True, 10.0, "comparative dispatch ok (manuscript gate owns baseline-matrix/ablation)"
        if paper_type == "cause_effect":
            # Cause-effect section gate (§7.1 M4): identification + verb
            # governance live in verify_causal (manuscript gate). At section
            # level only a downgraded (associational) claim carrying strong
            # causal verbs in Results prose fails fast.
            level = str((causal_claim or {}).get("verb_level") or "causal")
            if level != "causal" and section_name.lower() == "results":
                from agents.writer import CAUSAL_VERBS as _causal_verbs

                lowered = (content or "").lower()
                hits = [verb for verb in _causal_verbs if verb in lowered]
                if hits:
                    return False, 4.0, f"causal-verb governor failed (associational level): {', '.join(hits)}"
            return True, 10.0, "cause-effect dispatch ok (manuscript gate owns audit/governor)"
        return True, 10.0, "no per-type section gate for this paper type"

    def _record_claim_evidence(self, section_name: str, content: str, hard: VerificationReport, engineer_outputs: Optional[Dict[str, ExperimentOutput]]):
        """Persist an auditable claim ledger instead of hiding evidence in reviewer prose."""
        tracker = get_tracker()
        if not tracker:
            return
        citation = hard.get("citation", {})
        for item in citation.get("resolved", []):
            identifier = item.get("doi") or item.get("arxiv_id") or "source"
            self.ledger.record_claim(tracker.run_id, section_name, identifier, "citation", "verified", item)
        for item in citation.get("failed", []):
            identifier = item.get("doi") or item.get("arxiv_id") or "source"
            self.ledger.record_claim(tracker.run_id, section_name, identifier, "citation", "failed", item)
        # Quantitative sentences are evidence claims; keep their deterministic check beside them.
        stat_status = "verified" if hard.get("passed") else "needs_revision"
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            if re.search(r"\d+(?:\.\d+)?\s*%|\b(?:mean|std|p\s*[<=>]|accuracy|f1)\b", sentence, re.I):
                claim_type = classify_claim(sentence, [item.get("experiment") for item in hard.get("statistics", [])])
                eligible = claim_type == "empirical_result" and bool(hard.get("statistics"))
                self.ledger.record_claim(tracker.run_id, section_name, sentence[:500], claim_type, stat_status if eligible else "unverifiable_by_construction", {"statistics": hard.get("statistics", [])})

    def _soft_hallucination_check(self, content: str, section_name: str, hard: Dict) -> Tuple[float, str]:
        prompt = f"""
After HARD citation resolution (passed={hard['passed']}), review remaining soft issues:
Section: {section_name}
Content: {content[:4000]}
Resolved citations: {json.dumps(hard.get('citation', {}).get('resolved', [])[:5], default=str)}

Score 1-10 for unsupported generalizations (not already caught by DOI checks).
JSON: {{"score": 8, "issues": [], "recommendations": []}}
"""
        try:
            result = call_llm_json(prompt, temperature=0.2, tier="judge", attempts=2, call_fn=call_llm,
                                   **_supervisor_system(
                                       "supervisor.soft_check",
                                       hard_passed=hard["passed"])) or {}
            return float(result.get("score", 5)), json.dumps(result)[:500]
        except Exception as e:
            return 5.0, str(e)


class MathChecker:
    def evaluate(self, content: str, section_name: str) -> Tuple[float, str]:
        expressions = self._extract_math_expressions(content)
        if not expressions:
            return 10.0, "No mathematical expressions found"

        valid = 0
        invalid = []
        derivation_notes = []
        for expr in expressions:
            if validate_math_expression(expr):
                valid += 1
            else:
                invalid.append(expr)

        # Attempt derivation pairs: "A = B" style
        for m in re.finditer(r"\$([^$]+)=([^$]+)\$", content):
            left, right = m.group(1).strip(), m.group(2).strip()
            if len(left) < 40 and len(right) < 40:
                check = verify_math_derivation([], left, right)
                derivation_notes.append(f"{left} == {right}? {check.get('passed')}")

        total = len(expressions)
        score = (valid / total) * 10 if total else 10.0
        fb = f"Math validation: {valid}/{total} valid"
        if invalid:
            fb += f" | invalid: {', '.join(invalid[:3])}"
        if derivation_notes:
            fb += " | derivations: " + "; ".join(derivation_notes[:3])
        return score, fb

    def _extract_math_expressions(self, content: str) -> List[str]:
        patterns = [
            r"\$([^$]+)\$",
            r"\\\[([^\]]+)\\\]",
            r"\\begin\{equation\}(.*?)\\end\{equation\}",
        ]
        expressions = []
        for pattern in patterns:
            expressions.extend(re.findall(pattern, content, re.DOTALL))
        return list(set(expressions))


class CodeChecker:
    def evaluate(self, content: str, section_name: str) -> Tuple[float, str]:
        snippets = self._extract_code_snippets(content)
        if not snippets:
            return 10.0, "No code snippets found"
        valid = 0
        invalid = []
        for snippet in snippets:
            if self._validate_code_snippet(snippet):
                valid += 1
            else:
                invalid.append(snippet[:50] + "...")
        total = len(snippets)
        score = (valid / total) * 10 if total else 10.0
        fb = f"Code validation: {valid}/{total} snippets valid"
        if invalid:
            fb += f" | invalid: {', '.join(invalid[:3])}"
        return score, fb

    def _extract_code_snippets(self, content: str) -> List[str]:
        patterns = [
            r"```[\w]*\n(.*?)\n```",
            r"`([^`]+)`",
            r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}",
        ]
        snippets = []
        for pattern in patterns:
            snippets.extend(re.findall(pattern, content, re.DOTALL))
        return list(set(snippets))

    def _validate_code_snippet(self, snippet: str) -> bool:
        try:
            if any(k in snippet for k in ("def ", "import ", "class ")):
                compile(snippet, "<string>", "exec")
                return True
            return self._balanced(snippet)
        except Exception:
            return False

    def _balanced(self, text: str) -> bool:
        stack = []
        pairs = {")": "(", "}": "{", "]": "["}
        for char in text:
            if char in "({[":
                stack.append(char)
            elif char in ")}]":
                if not stack or stack.pop() != pairs[char]:
                    return False
        return len(stack) == 0


class ReviewerBot:
    """LLM soft peer review — only meaningful AFTER hard checks."""

    def evaluate(self, content: str, section_name: str) -> Tuple[float, str]:
        prompt = f"""
You are a peer reviewer. Soft qualitative review ONLY (hard citation/stats already checked).
Section: {section_name}
Content: {content[:5000]}

Checklist:
{json.dumps(REVIEW_CHECKLIST)}

Criteria 1-10: clarity, accuracy, flow, completeness, writing.
JSON:
{{"scores": {{"clarity": 8, "accuracy": 7, "flow": 8, "completeness": 6, "writing": 7}},
    "overall_score": 7.2, "checklist": {{"item": true}}, "strengths": [], "weaknesses": [], "suggestions": []}}
"""
        try:
            result = call_llm_json(prompt, temperature=0.3, tier="judge", attempts=2, call_fn=call_llm,
                                   **_supervisor_system("supervisor.review",
                                                        section=section_name)) or {}
            score = float(result.get("overall_score", 5.0))
            fb = f"Peer review score: {score}/10"
            if result.get("weaknesses"):
                fb += f" | weaknesses: {', '.join(result['weaknesses'][:2])}"
            return score, fb
        except Exception as e:
            return 5.0, f"Peer review error: {e}"
