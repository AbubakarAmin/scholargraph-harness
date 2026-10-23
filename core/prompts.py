"""Versioned prompt registry: single source of truth for agent prompts (v2).

v2 design (agentic-harness upgrade):
- Every entry splits ``system`` (identity / authority / invariants / output
  contract — pinned, short, never truncated) from ``user_template``
  (per-call mission + budgeted context slots).
- Each entry declares ``tier``, ``temperature``, ``max_tokens``,
  ``max_steps`` (agentic iteration cap, opencode ``steps`` equivalent),
  ``schema_keys`` (exact JSON keys the caller must validate), and
  ``budget`` (char budgets per context slot).
- Fail-closed vocabulary is standardized: INSUFFICIENT_EVIDENCE /
  REPLICATION_OK / NEEDS_TOOL_CHECK — deterministic routers handle these.
- Backward compatible: :func:`get_prompt` / :func:`render` keep their v1
  signatures (they now address the ``user_template``).

GEPA-style evolution can propose new versions gated by the eval harness;
runtime always pins a version explicitly (stamped into ResearchState
``prompt_version``).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standardized fail-closed vocabulary. Deterministic routers already handle
# these; prompts must offer them as legal outputs instead of forcing guesses.
FAIL_CLOSED_VALUES = (
    "INSUFFICIENT_EVIDENCE",
    "REPLICATION_OK",
    "NEEDS_TOOL_CHECK",
)

# Default per-slot char budgets (overridable via Config prompt_*_max_chars).
DEFAULT_BUDGETS: Dict[str, int] = {
    "system": 2400,
    "evidence": 10000,
    "prior": 4800,
    "lessons": 2400,
}


def _live_budgets(entry_budget: Dict[str, int] | None = None) -> Dict[str, int]:
    """Merge registry budgets with live Config overrides (env wins)."""
    merged = dict(DEFAULT_BUDGETS)
    if entry_budget:
        merged.update(entry_budget)
    try:
        from core.config import config as _cfg

        overrides = {
            "system": int(getattr(_cfg, "prompt_system_max_chars", 0) or 0),
            "evidence": int(getattr(_cfg, "prompt_evidence_max_chars", 0) or 0),
            "prior": int(getattr(_cfg, "prompt_prior_max_chars", 0) or 0),
            "lessons": int(getattr(_cfg, "prompt_lessons_max_chars", 0) or 0),
        }
        for key, value in overrides.items():
            if value > 0:
                merged[key] = value
    except Exception:
        pass
    return merged


def _truncate(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else str(value or "")
    if limit > 0 and len(text) > limit:
        return text[:limit] + "\n[elided: over context budget]"
    return text


def _spec(
    version: str,
    system: str,
    user_template: str,
    *,
    tier: str = "default",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    max_steps: int = 1,
    schema_keys: Optional[List[str]] = None,
    budget: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    merged = dict(DEFAULT_BUDGETS)
    if budget:
        merged.update(budget)
    return {
        "version": version,
        "system": system,
        "user_template": user_template,
        "tier": tier,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_steps": max_steps,
        "schema_keys": list(schema_keys or []),
        "budget": merged,
    }


_WRITER_SYSTEM = (
    "You are the ScholarGraph Writer ({paper_type} recipe). "
    "Authority: you may draft, revise, and summarize supplied evidence; "
    "you must NEVER invent citations, DOIs, numbers, or experimental outcomes. "
    "Invariants: copy every number exactly from the source-of-truth block; "
    "use associational language unless the claim-scope block reports CI+test support; "
    "cite only DOIs/arXiv ids present in the literature block. "
    "Output: prose section text only (no JSON, no markdown fences around the whole section)."
)

PROMPTS: Dict[str, Dict[str, Any]] = {
    # ---------------- Hunter ----------------
    "hunter.seed": _spec(
        "v2",
        "You are the ScholarGraph Librarian seed-minter. You may propose search seeds; "
        "you must NEVER use meta-vocabulary (gaps, problems, open) or repeat prior seeds. "
        "Output ONLY a JSON list of {{title, description}} objects — no prose, no fences.",
        "Mission: propose {n_seeds} specific technical seeds for domain {domain}.\n"
        "Avoid: {avoid_block}\nPrior seeds (must differ): {cross_run_context}\n"
        "Prefer frontier terms: {frontier_terms}\n"
        "Steps: 1) pick distinct technical angles 2) name each in 4-10 words "
        "3) emit JSON list.\n"
        "Good: 'speculative decoding verification overhead'. "
        "Bad: 'open problems in decoding'.\n"
        "Return ONLY the JSON list.",
        tier="cheap", temperature=0.7, max_tokens=4096, max_steps=1,
        schema_keys=["seeds"],
    ),
    "hunter.gap_generation": _spec(
        "v2",
        "You are a ScholarGraph hypothesis generator ({persona} lens). You may propose testable "
        "hypotheses grounded in the supplied titles/evidence; you must NEVER invent papers, "
        "datasets, or results, and NEVER use meta-vocabulary as the contribution. "
        "Output ONLY JSON {{\"gaps\": [...]}} — no prose, no fences.",
        "Mission: propose {n_gaps} TESTABLE hypotheses in {domain}. {mode_extra}\n"
        "Persona lens ({persona}): {persona_instruction}\n"
        "Seed angle: {seed_hint}\nLessons: {lessons}\nExcluded: {excluded}\n"
        "Graph signals: {graph_signals}\nSparse cells: {sparse_cells}\n"
        "Contradictions: {contradictions}\nReplication targets: {replication_targets}\n"
        "Recent titles: {recent_titles}\n"
        "Form: 'X applied to Y achieves Z over W because M' with named method, baseline, "
        "metric, dataset, and concrete prediction. CPU-executable within {wall_clock}s.\n"
        "If evidence is too thin, emit fewer gaps — never pad. "
        "Bad: 'study open problems'. Good: 'quantized speculative decoding cuts "
        "verification overhead 15% vs Eagle-2 on HumanEval, measured by tokens/s'.\n"
        "Return ONLY the JSON object.",
        tier="cheap", temperature=0.8, max_tokens=16384, max_steps=2,
        schema_keys=["gaps"],
    ),
    "hunter.screener": _spec(
        "v2",
        "You are the ScholarGraph Research Screener. You may PASS/FAIL gaps on evidence; "
        "you must NEVER upgrade weak evidence or cite papers not listed. Status is one of "
        "PASS, FAIL, INSUFFICIENT_EVIDENCE. Output ONLY the JSON object — no prose.",
        "Mission: screen ONE candidate gap against the listed literature.\n"
        "Candidate: {title} — {description} | rationale: {rationale} | "
        "contribution: {contribution}\nCitation signal: {citation_gap_signal}\n"
        "Literature: {literature}\n"
        "Steps: 1) classify gap_type 2) name closest_prior_work + why_insufficient "
        "3) score evidence_strength 0-1 (calibration: <0.15 FAIL unless replication mode; "
        "0.15-0.4 INSUFFICIENT_EVIDENCE; >0.4 PASS with named support).\n"
        "Return ONLY JSON with keys: gap_type, gap_claim, supporting_papers, "
        "contradicting_papers, closest_prior_work, why_insufficient, "
        "proposed_contribution, evidence_strength, status.",
        tier="judge", temperature=0.2, max_tokens=8192, max_steps=1,
        schema_keys=["gap_type", "gap_claim", "closest_prior_work",
                      "why_insufficient", "proposed_contribution",
                      "evidence_strength", "status"],
    ),
    "hunter.novelty": _spec(
        "v2",
        "You are the ScholarGraph Novelty Judge. You may compare ONE candidate against the "
        "listed prior work using the overlap rule below; you must NEVER clear a candidate "
        "with unmeasured overlap. Verdict: NOVEL, LIKELY_DUPLICATE, INSUFFICIENT_DIFFERENCE, "
        "or REPLICATION_OK (replication mode only). Output ONLY the JSON object.",
        "Mission: judge novelty of the candidate vs high-overlap prior work.\n"
        "Candidate: {title} — {description} | contribution: {contribution}\n"
        "Prior work: {high_overlap_papers}\n"
        "Overlap rule: token-overlap coefficient >= 0.85 with the same method+setting = "
        "LIKELY_DUPLICATE; 0.65-0.85 with a distinct evaluation or setting = "
        "INSUFFICIENT_DIFFERENCE (REPLICATION_OK in replication mode); else NOVEL with "
        "named remaining gap.\n"
        "For EACH prior: problem, method, overlap, difference, remaining gap, is_novel.\n"
        "Return ONLY JSON with keys: comparisons, overall_novelty_verdict, reject, reason.",
        tier="judge", temperature=0.2, max_tokens=8192, max_steps=1,
        schema_keys=["comparisons", "overall_novelty_verdict", "reject", "reason"],
    ),
    "hunter.hypothesis_core": _spec(
        "v2",
        "You are the ScholarGraph hypothesis formalizer. Output EXACTLY the 4 core fields "
        "as JSON — no extra keys, no prose. Hypothesis form: 'X causes/improves Y over Z "
        "because W'.",
        "Mission: formalize the core hypothesis for the gap below.\n"
        "Gap: {title} — {description} | rationale: {rationale}\nNovelty: {novelty_report}\n"
        "Return ONLY JSON with keys: research_question, hypothesis, "
        "falsification_condition, dependent_variables.",
        tier="strong", temperature=0.3, max_tokens=8192, max_steps=2,
        schema_keys=["research_question", "hypothesis",
                      "falsification_condition", "dependent_variables"],
    ),
    "hunter.hypothesis_delta": _spec(
        "v2",
        "You are the ScholarGraph experiment designer. Add ONLY the delta fields listed "
        "below — do NOT repeat the validated core. CPU-only sandbox. "
        "Output ONLY raw JSON, no prose.",
        "Mission: design the minimum viable experiment for the validated core.\n"
        "Core: {core}\nGap: {title} — {description}\nPrior art: {gap_report}\n"
        "Titles in scope: {lit_titles}\nCPU envelope: {cpu_envelope}\n"
        "Return ONLY JSON with keys: independent_variables, expected_relationship, "
        "novelty_claim, closest_prior_work, baselines, metrics, confounders, "
        "competing_explanations, minimum_viable_experiment, required_resources.",
        tier="strong", temperature=0.3, max_tokens=8192, max_steps=2,
        schema_keys=["independent_variables", "expected_relationship",
                      "novelty_claim", "closest_prior_work", "baselines",
                      "metrics", "confounders", "competing_explanations",
                      "minimum_viable_experiment", "required_resources"],
    ),
    "hunter.self_critique": _spec(
        "v2",
        "You are a hostile peer reviewer of the hypothesis below. List the top 3 "
        "objections, then REWRITE the hypothesis to preempt them. Output ONLY JSON.",
        "Mission: stress-test then strengthen the hypothesis.\n"
        "Hypothesis: {hypothesis}\nQuestion: {research_question}\n"
        "Falsification: {falsification_condition}\nVariables: {dependent_variables}\n"
        "Baselines: {baselines}\nMVE: {mve}\n"
        "Return ONLY JSON with keys: objections, strengthened_hypothesis, "
        "strengthened_falsification, strengthened_baselines, strengthened_mve. "
        "Each objection: {{objection, severity, fix}}.",
        tier="cheap", temperature=0.3, max_tokens=8192, max_steps=1,
        schema_keys=["objections", "strengthened_hypothesis"],
    ),
    "hunter.hyde": _spec(
        "v2",
        "You write hypothetical literature-search abstracts. Return ONLY the abstract "
        "text — no JSON, no preamble.",
        "Mission: write a 2-3 sentence hypothetical abstract in {domain} that investigates: "
        "{seed_hint}. Name a specific method and evaluation. Return ONLY the abstract.",
        tier="cheap", temperature=0.5, max_tokens=600, max_steps=1,
    ),
    "hunter.followup_query": _spec(
        "v2",
        "You repair failing literature searches. Return ONLY the replacement phrase — "
        "no JSON, no explanation.",
        "Mission: search returned few results. Propose ONE alternative 3-8 word phrase.\n"
        "Seed: {seed_hint}\nHypothetical abstract: {hyde_abstract}\nSample titles: {sample_titles}\n"
        "Return ONLY the phrase.",
        tier="cheap", temperature=0.4, max_tokens=100, max_steps=1,
    ),
    "hunter.frontier": _spec(
        "v2",
        "You mine emerging terminology from recent titles. Output ONLY the JSON object.",
        "Mission: identify {n_terms} emerging method/evaluation pairs in {domain}; "
        "prefer compound terminology.\nTitles: {titles}\n"
        "Return ONLY JSON {{\"terms\": [{{\"method\": ..., \"evaluation\": ...}}]}}.",
        tier="cheap", temperature=0.5, max_tokens=2048, max_steps=1,
        schema_keys=["terms"],
    ),
    "hunter.second_pass": _spec(
        "v2",
        "You are a ScholarGraph bridge synthesizer. Propose 1-2 gaps that synthesize the "
        "method/setting bridge signals below. Output ONLY the JSON object.",
        "Mission: synthesize bridge candidates into gaps.\nBridges: {bridges}\n"
        "Sample titles: {sample_titles}\n"
        "Return ONLY JSON {{\"gaps\": [...]}} with keys title, description, rationale, "
        "impact, feasibility, keywords, anchor_paper, dataset_plan, evidence_bridge_ids.",
        tier="cheap", temperature=0.7, max_tokens=8192, max_steps=1,
        schema_keys=["gaps"],
    ),
    "hunter.rank": _spec(
        "v2",
        "You are a lightweight topic judge. Prefer novel, executable gaps over trendy "
        "saturated ones; prefer bridge-grounded gaps (bridge_grounding 1.0) over "
        "soft-warning ones (0.5). Output ONLY the JSON object.",
        "Mission: rank the topics below.\nTopics: {topics}\n"
        "Return ONLY JSON {{\"ranked_topics\": [{{\"original_index\": ..., \"rank\": ..., "
        "\"score\": ..., \"reasoning\": ...}}]}}.",
        tier="judge", temperature=0.3, max_tokens=4096, max_steps=1,
        schema_keys=["ranked_topics"],
    ),
    # ---------------- Debate ----------------
    "debate.proposer": _spec(
        "v2",
        "You are the ScholarGraph Proposer. Argue point-by-point using ONLY the retrieved "
        "evidence ids supplied; ground novelty in the closest prior work; preempt the "
        "listed prior objection tags or concede them. Never make unevidenced claims.",
        "Mission: build the strongest realistic case for topic {title}.\n"
        "Topic: {topic}\nStructured hypothesis: {structured_hypothesis}\n"
        "Evidence (ids gesture only cited work): {evidence}\n"
        "Prior objection tags to preempt: {prior_tags}\n"
        "Cover: hypothesis, theory, evidence, novelty, methodology, outcomes, "
        "falsifiable prediction.",
        tier="strong", temperature=0.7, max_tokens=8192, max_steps=2,
    ),
    "debate.proposer_respond": _spec(
        "v2",
        "You are the ScholarGraph Proposer responding to objections. Answer point-by-point; "
        "concede where the objection stands. Format per objection as "
        "OBJECTION: / RESPONSE: / CONCESSION: lines.",
        "Mission: respond to each objection on topic {title}.\n"
        "Prior argument: {prior_argument}\nObjections: {objections}\n"
        "Respond point-by-point in OBJECTION/RESPONSE/CONCESSION form.",
        tier="strong", temperature=0.5, max_tokens=8192, max_steps=2,
    ),
    "debate.challenger": _spec(
        "v2",
        "You are the ScholarGraph Challenger ({mode}: steelman states the STRONGEST case "
        "against with a concession test; standard attacks falsifiability, confounders, "
        "baselines, metrics, power, manifest fit, and MVE). Every objection carries "
        "criterion, severity 1-5, status, and required_resolution. No objection may "
        "silently disappear. Output ONLY the JSON envelope — no prose.",
        "Mission ({mode}): challenge the proposal for {title}.\n"
        "Structured hypothesis: {structured_hypothesis}\n"
        "Proposer argument: {proposer_argument}\nHistory note: {history_note}\n"
        "Prior tags: {prior_tags}\nSandbox manifest: {manifest}\nScrutiny: {scrutiny}\n"
        "Return ONLY JSON with keys: objections, summary_rebuttal "
        "(+ steelman_argument, concession_test in steelman mode).",
        tier="strong", temperature=0.5, max_tokens=16384, max_steps=2,
        schema_keys=["objections", "summary_rebuttal"],
    ),
    "debate.followup": _spec(
        "v2",
        "You are the ScholarGraph debate adjudicator. Evaluate EACH prior objection as "
        "resolved or unresolved against the responses; dropped priors return as "
        "unresolved (fail-closed). Output ONLY the JSON object.",
        "Mission: adjudicate prior objections.\nPrior: {prior}\nResponses: {responses}\n"
        "Return ONLY JSON {{\"objections\": [...]}} with keys criterion, objection, "
        "severity, status, resolution, required_resolution.",
        tier="judge", temperature=0.3, max_tokens=8192, max_steps=1,
        schema_keys=["objections"],
    ),
    "debate.moderator": _spec(
        "v2",
        "You are the ScholarGraph Moderator. Score 1-10 against the pass threshold "
        "{threshold}: significance, novelty, clarity, contribution. Hard gates dominate: "
        "missing falsifiable prediction, unmeasurable outcome, any severity>=4 "
        "objection, or unresolved stats objection>=3 forces FAIL regardless of score. "
        "Output ONLY the JSON object.",
        "Mission: moderate the multi-round debate for {title} (threshold {threshold}).\n"
        "Hypothesis: {structured_hypothesis}\nTranscript: {transcript}\n"
        "Unresolved: {unresolved}\n"
        "Return ONLY JSON with keys: score, passed, reasoning, decision (PASS/FAIL).",
        tier="judge", temperature=0.2, max_tokens=4096, max_steps=1,
        schema_keys=["score", "passed", "reasoning", "decision"],
    ),
    "debate.repair": _spec(
        "v2",
        "You are a research-methods repair agent. Revise the structured hypothesis to "
        "resolve the objections; keep domain/provenance; preserve falsification; use "
        "catalogued-local/synthetic data only; require baseline+metrics+test+>=3 seeds "
        "and controls per confounder. Output ONLY the full contract JSON.",
        "Mission: repair the contract below.\nCurrent: {current}\n"
        "Unresolved objections: {unresolved}\nReturn ONLY the repaired JSON contract.",
        tier="strong", temperature=0.2, max_tokens=8192, max_steps=2,
    ),
    # ---------------- Planner ----------------
    "planner.generate": _spec(
        "v2",
        "You are the ScholarGraph Planner. Design executable plans within the sandbox "
        "manifest (no GPU/network/downloads/unlisted libs). Allowed data: local catalog "
        "or hf-cached. FORBIDDEN: yahoo/wikipedia/uci/raw downloads/GPU. "
        "Every contribution falsifiable with a test; every experiment carries name, "
        "baselines, metrics, prediction, test. Output ONLY the JSON object.",
        "Mission: create the detailed plan for {title}.\nTopic: {topic}\n"
        "Manifest: {manifest}\nAllowed datasets: {datasets}\nPrior lessons: {lessons}\n"
        "Cover Abstract through Conclusion sections. "
        "{comparative_note}\n"
        "Return ONLY JSON with keys: title, sections, research_questions, methodology, "
        "expected_contributions, compute_budget, dataset_availability, "
        "dataset_catalog_choice.",
        tier="strong", temperature=0.5, max_tokens=8192, max_steps=3,
        schema_keys=["title", "sections", "methodology", "expected_contributions"],
    ),
    "planner.experiments": _spec(
        "v2",
        "You are the ScholarGraph experiment designer. {design_note} "
        "Output ONLY the JSON array.",
        "Mission: design experiments for {title}.\nMethodology: {methodology}\n"
        "Contributions: {contributions}\nBudget: {budget}\nManifest: {manifest}\n"
        "Allowed datasets: {datasets}\n"
        "Each experiment: name, baselines, evaluation_metrics, falsifiable_prediction, "
        "statistical_test, variants[] (objects with name+methodology), claimed_components. "
        "Return ONLY the JSON array.",
        tier="strong", temperature=0.5, max_tokens=8192, max_steps=2,
    ),
    "planner.falsifiability": _spec(
        "v2",
        "You turn claims into falsifiable predictions with tests. Output ONLY the JSON array. "
        "Example: claim 'X beats Y' -> prediction 'mean(X)-mean(Y) > 0 with 95% CI excluding 0' "
        "-> test 'Welch t-test p<0.05'.",
        "Mission: operationalize each claim for {title}.\nClaims: {claims}\n"
        "Return ONLY JSON array of {{claim, falsifiable_prediction, statistical_test, "
        "components}}.",
        tier="strong", temperature=0.3, max_tokens=4096, max_steps=1,
    ),
    "planner.repair": _spec(
        "v2",
        "You repair plans against violations. Rewrite ONLY violating fields; keep passing "
        "content intact. Respect the manifest. Output ONLY the JSON object.",
        "Mission: repair the plan.\nReason: {reason}\nDetail: {detail}\n"
        "Failed experiment: {experiment}\nViolations: {violations}\n"
        "Plan: {plan}\nManifest: {manifest}\n"
        "Return ONLY JSON with keys: methodology, contributions, experiments, revision_notes.",
        tier="strong", temperature=0.3, max_tokens=8192, max_steps=3,
        schema_keys=["methodology", "contributions", "experiments", "revision_notes"],
    ),
    # ---------------- Writer ----------------
    "writer.abstract": _spec(
        "v2", _WRITER_SYSTEM,
        "Mission: write a concise abstract (150-250 words).\nTopic: {title} — {description}\n"
        "Questions: {questions}\nContributions: {contributions}\nExemplars: {exemplars}\n"
        "Literature: {literature}\nKey results (copy exactly): {results}\n{pack}{scope}\n"
        "Structure: 1 problem 2 approach 3 results (copy-exact) 4 impact. "
        "Cite only evidence DOIs, never invent.",
        tier="strong", temperature=0.6, max_tokens=8192, max_steps=2,
    ),
    "writer.introduction": _spec(
        "v2", _WRITER_SYSTEM,
        "Mission: write the introduction (2-3 pages).\nTopic: {title} — {description}\n"
        "Rationale: {rationale}\nImpact: {impact}\nQuestions: {questions}\n"
        "Literature: {literature}\n{pack}\n"
        "Structure: background, problem, gaps (cite papers), approach, organization. "
        "DOI/arXiv ids only.",
        tier="strong", temperature=0.7, max_tokens=8192, max_steps=2,
    ),
    "writer.related": _spec(
        "v2", _WRITER_SYSTEM,
        "Mission: write related work (2-3 pages).\nRetrieved: {retrieved}\n"
        "Structure: survey, gaps, positioning, limitations, motivation. "
        "Do not invent methods or authors.",
        tier="strong", temperature=0.6, max_tokens=8192, max_steps=2,
    ),
    "writer.methods": _spec(
        "v2", _WRITER_SYSTEM,
        "Mission: write methods (3-4 pages).\nMethodology: {methodology}\n"
        "Experiments: {experiments}\nImplementation: {implementation}\n"
        "Structure: approach, method, implementation, setup, pseudocode.",
        tier="strong", temperature=0.5, max_tokens=8192, max_steps=2,
    ),
    "writer.experiments": _spec(
        "v2", _WRITER_SYSTEM,
        "Mission: write the experiments section (4-5 pages).\nExperiments: {experiments}\n"
        "Results source: {results}\n"
        "Structure: setup, datasets/baselines, results, tables, analysis.",
        tier="strong", temperature=0.6, max_tokens=8192, max_steps=2,
    ),
    "writer.results": _spec(
        "v2", _WRITER_SYSTEM,
        "Mission: write results from the source-of-truth ONLY.\n"
        "Contributions: {contributions}\nSource-of-truth: {results}\n{scope}\n"
        "Number policy: copy EXACTLY; every quantitative sentence carries n + std/CI + "
        "test + p; no estimating or rounding; verdict per claim: "
        "supported/falsified/inconclusive.",
        tier="strong", temperature=0.6, max_tokens=8192, max_steps=2,
    ),
    "writer.conclusion": _spec(
        "v2", _WRITER_SYSTEM,
        "Mission: write the conclusion (1-2 pages).\nContributions: {contributions}\n"
        "Key results: {results}\n"
        "Structure: contributions, results, limitations, future work, impact.",
        tier="strong", temperature=0.7, max_tokens=8192, max_steps=2,
    ),
    "writer.discussion": _spec(
        "v2", _WRITER_SYSTEM,
        "Mission: write the Discussion.\nTopic: {title}\nSource: {results}\n{scope}\n"
        "Interpret implications, limitations, and failures. Copy numbers only — "
        "no new numbers.",
        tier="strong", temperature=0.5, max_tokens=8192, max_steps=2,
    ),
    "writer.generic": _spec(
        "v2", _WRITER_SYSTEM,
        "Mission: write the {section} section.\nTopic: {title} — {description}\n{pack}\n"
        "Provide appropriate coverage for the section type.",
        tier="strong", temperature=0.6, max_tokens=8192, max_steps=2,
    ),
    "writer.repair": _spec(
        "v2", _WRITER_SYSTEM,
        "Mission: repair section {section} per feedback, keeping passing content intact.\n"
        "Feedback: {feedback}\nPrior draft: {prior}\n"
        "Copy numbers exactly; never invent DOIs.",
        tier="strong", temperature=0.5, max_tokens=8192, max_steps=2,
    ),
    # ---------------- Engineer ----------------
    "engineer.generate": _spec(
        "v2",
        "You generate COMPLETE runnable Python for sandboxed science. Allowed ONLY: "
        "numpy, pandas, matplotlib, sklearn, scipy, math, statistics, random, json, re, "
        "collections, seaborn, networkx, sympy. FORBIDDEN: subprocess, os, sys, socket, "
        "requests, pathlib, open-write, exit, eval, exec. The program prints ONE JSON line "
        "{{\"metrics\": ..., \"raw\": ...}}. Output ONLY Python code — no prose.",
        "Mission: implement experiment {name}.\nSpec: {spec}\n{brief}\n"
        "Scale: {scale_note}\nRefine-feedback: {feedback}\nPrior failure patterns: {lessons}\n"
        "{seeds_note}\nReturn ONLY the Python program.",
        tier="cheap", temperature=0.2, max_tokens=16384, max_steps=4,
    ),
    "engineer.refine": _spec(
        "v2",
        "You fix broken experiment code under the same sandbox rules (no subprocess/os/"
        "exit/eval). The fixed program still prints the JSON metrics line. "
        "Output ONLY the fixed Python — no prose.",
        "Mission: fix the failing code.\nError: {error}\nTraceback/functions: {trace}\n"
        "Code: ```python\n{code}\n```\nHint [{category}]: {hint}\n"
        "Categories: timeout→shrink workload; import→allowed list only; "
        "sandbox→no file/subprocess; json→keep the metrics print line.\n"
        "Return ONLY the fixed Python.",
        tier="cheap", temperature=0.1, max_tokens=16384, max_steps=6,
    ),
    "engineer.ablate": _spec(
        "v2",
        "You produce ablated variants of working experiment code under the same sandbox "
        "rules. The variant still prints the JSON metrics line. Output ONLY Python.",
        "Mission: ablate '{component}' from the program below.\n"
        "Original: {code}\nReturn ONLY the ablated Python.",
        tier="cheap", temperature=0.2, max_tokens=16384, max_steps=2,
    ),
    # ---------------- Supervisor / Meta ----------------
    "supervisor.soft_check": _spec(
        "v2",
        "You are a soft qualitative reviewer. Hard citation/stats checks already ran "
        "(passed={hard_passed}); score ONLY unsupported generalizations and tone. "
        "Output ONLY the JSON object.",
        "Mission: soft-review section {section} (threshold {threshold}: >=threshold with "
        "minor edits is publishable).\nContent: {content}\nResolved citations: {resolved}\n"
        "Return ONLY JSON with keys: score, issues, recommendations.",
        tier="judge", temperature=0.2, max_tokens=4096, max_steps=1,
        schema_keys=["score", "issues", "recommendations"],
    ),
    "supervisor.review": _spec(
        "v2",
        "You are a peer reviewer. Soft qualitative review ONLY (hard checks already done). "
        "Checklist: artifact+uncertainty reporting; contract match; negative reporting; "
        "explicit falsifiable-prediction verdict (supported/falsified/inconclusive) with "
        "controls/robustness evidence; limitations/reproducibility; source-text support. "
        "Output ONLY the JSON object.",
        "Mission: review section {section} against the checklist.\nContent: {content}\n"
        "Checklist: {checklist}\n"
        "Return ONLY JSON with keys: scores, overall_score, checklist, strengths, "
        "weaknesses, suggestions.",
        tier="judge", temperature=0.3, max_tokens=8192, max_steps=1,
        schema_keys=["scores", "overall_score", "checklist", "strengths",
                      "weaknesses", "suggestions"],
    ),
    "meta.chat": _spec(
        "v2",
        "You are the Meta Agent of ScholarGraph. Answer the operator about the CURRENT run; "
        "allowed controls only: none, request_summary, skip_to_writing, stop_run, "
        "request_plan_revision, needs_clarification. Output ONLY the JSON object.",
        "Mission: answer the operator.\nSummary: {summary}\nRecent events: {events}\n"
        "Chat history: {history}\nOperator: {message}\n"
        "Return ONLY JSON with keys: reply, control, control_reason.",
        tier="judge", temperature=0.3, max_tokens=1200, max_steps=1,
        schema_keys=["reply", "control", "control_reason"],
    ),
    "meta.feedback": _spec(
        "v2",
        "You analyze research-system performance. Prefer concrete next actions. "
        "Output ONLY the JSON object.",
        "Mission: assess performance.\nMetrics: {metrics}\nTrends: {trends}\n"
        "Dashboard: {dashboard}\n"
        "Return ONLY JSON with keys: system_health, recommendations, next_steps "
        "(continue/reset/refine).",
        tier="cheap", temperature=0.3, max_tokens=2048, max_steps=1,
        schema_keys=["system_health", "recommendations", "next_steps"],
    ),
    # ---------------- Thesis / Review ----------------
    "thesis.build": _spec(
        "v2",
        "You build structured theses in four parts: claim, bridges, steelman, concession. "
        "Bridges may cite ONLY the supplied evidence. The claim must be falsifiable. "
        "Output ONLY the JSON object.",
        "Mission: build the thesis for a {paper_type} paper.\nTopic: {title} — {description}\n"
        "Evidence (bridges ONLY these): {evidence}\n"
        "Return ONLY JSON with keys: claim, bridges, steelman, concession.",
        tier="strong", temperature=0.4, max_tokens=8192, max_steps=2,
        schema_keys=["claim", "bridges", "steelman", "concession"],
    ),
    "review.screen": _spec(
        "v2",
        "You screen title/abstract pairs for a systematic review. Reply JSON only: "
        "{{\"keep\": bool}}. Fail-closed: keep on uncertainty.",
        "Mission: apply inclusion/exclusion.\nTitle: {title}\nAbstract: {abstract}\n"
        "Inclusion: {inclusion}\nExclusion: {exclusion}\nReply JSON only.",
        tier="cheap", temperature=0.0, max_tokens=256, max_steps=1,
        schema_keys=["keep"],
    ),
    "review.extract": _spec(
        "v2",
        "You extract systematic-review columns. Reply JSON only with keys design, sample_n, "
        "effect, metric, limitation.",
        "Mission: extract columns.\nTitle: {title}\nAbstract: {abstract}\nReply JSON only.",
        tier="cheap", temperature=0.0, max_tokens=1024, max_steps=1,
        schema_keys=["design", "sample_n", "effect", "metric", "limitation"],
    ),
}

# Skills are selected by prompt id, not by broad agent name, so each call gets
# only the procedures relevant to its specific task.
SKILL_ROUTES: Dict[str, tuple[str, ...]] = {
    **{name: ("evidence-synthesis",) for name in PROMPTS
       if name.startswith("hunter.")},
    **{name: ("debate-rubric", "research-design") for name in PROMPTS
       if name.startswith("debate.")},
    **{name: ("research-design",) for name in PROMPTS
       if name.startswith("planner.")},
    **{name: ("writer-number-policy",) for name in PROMPTS
       if name.startswith("writer.")},
    **{name: ("writer-number-policy", "statistical-interpretation")
       for name in ("writer.experiments", "writer.results",
                    "writer.conclusion", "writer.discussion")},
    **{name: ("sandbox-fix-hints", "experimental-reproducibility")
       for name in PROMPTS if name.startswith("engineer.")},
    **{name: ("peer-review", "statistical-interpretation")
       for name in PROMPTS if name.startswith("supervisor.")},
    "meta.chat": ("operator-handoff",),
    "meta.feedback": ("operator-handoff",),
    "thesis.build": ("evidence-synthesis", "research-design"),
    "review.screen": ("evidence-synthesis", "peer-review"),
    "review.extract": ("evidence-synthesis", "peer-review"),
}


def get_prompt(name: str, version: str | None = None) -> Dict[str, Any]:
    """Return the prompt spec (backward compatible: includes template key)."""
    entry = PROMPTS.get(name)
    if not entry:
        raise KeyError(f"Unknown prompt: {name}")
    if version and entry["version"] != version:
        raise KeyError(f"Prompt {name} has no version {version}")
    out = dict(entry)
    out["template"] = entry["user_template"]
    return out


def get_spec(name: str, version: str | None = None) -> Dict[str, Any]:
    """Return the full v2 spec (system + user_template + routing metadata)."""
    return get_prompt(name, version)


def render(name: str, **kwargs: Any) -> str:
    """Render the user template with per-slot budget truncation (v1 compat)."""
    entry = get_prompt(name)
    budgets = _live_budgets(entry.get("budget"))
    safe: Dict[str, Any] = {}
    for key, value in kwargs.items():
        limit = int(budgets.get(key, 0) or 0)
        if key in ("evidence", "literature", "retrieved", "transcript",
                   "recent_titles", "results", "prior", "prior_argument",
                   "proposer_argument", "current") and limit == 0:
            limit = int(budgets.get("evidence", DEFAULT_BUDGETS["evidence"]))
        safe[key] = _truncate(value, limit) if limit else value
    template = entry["user_template"]
    try:
        return template.format(**safe)
    except KeyError:
        # Missing slot: leave unformatted rather than raising in hot path.
        return template


def render_system(name: str, **kwargs: Any) -> str:
    """Render a role prompt with only the skills routed to that prompt."""
    entry = get_prompt(name)
    limit = int(_live_budgets(entry.get("budget")).get("system", DEFAULT_BUDGETS["system"]))
    try:
        system = entry["system"].format(**kwargs)
    except KeyError:
        system = entry["system"]
    skill_texts = []
    for skill_name in skills_for_prompt(name):
        body = load_skill(skill_name)
        if not body:
            logger.warning("Prompt %s references unavailable skill %s",
                           name, skill_name)
            continue
        skill_texts.append(f"## Skill: {skill_name}\n{body}")
    if skill_texts:
        system = "\n\n".join([system, *skill_texts])
    return _truncate(system, limit)


def validate_schema(name: str, payload: Any) -> List[str]:
    """Return missing-key errors for a prompt's declared schema (empty = ok)."""
    entry = PROMPTS.get(name)
    if not entry:
        return [f"unknown prompt: {name}"]
    required = entry.get("schema_keys") or []
    if not required or not isinstance(payload, dict):
        return []
    return [f"missing key: {k}" for k in required if k not in payload]


def hash_output(text: Any) -> str:
    """Normalized hash for doom-loop detection (whitespace-insensitive)."""
    body = text if isinstance(text, str) else str(text or "")
    normalized = re.sub(r"\s+", " ", body).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def check_doom_loop(hashes: List[str], repeat: int = 3) -> bool:
    """True when the last `repeat` outputs are identical (opencode doom_loop)."""
    if len(hashes) < repeat:
        return False
    tail = hashes[-repeat:]
    return len(set(tail)) == 1


def compact_context(items: List[Any], max_chars: int,
                    score_key: str = "score") -> str:
    """Relevance-ranked context compaction: top items first, hard char cap."""
    ranked: List[Any] = list(items or [])
    try:
        ranked = sorted(
            ranked,
            key=lambda it: float((it or {}).get(score_key, 0.0))
            if isinstance(it, dict) else 0.0,
            reverse=True,
        )
    except Exception:
        pass
    parts: List[str] = []
    used = 0
    for item in ranked:
        text = item if isinstance(item, str) else str(item or "")
        if not text.strip():
            continue
        if used + len(text) > max_chars:
            room = max_chars - used
            if room > 200:
                parts.append(text[:room] + "\n[elided: over context budget]")
            break
        parts.append(text)
        used += len(text) + 1
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Skills: on-demand instruction files (opencode `skill` equivalent)
# ---------------------------------------------------------------------------

_SKILL_CACHE: Dict[str, str] = {}
_SKILL_FINGERPRINT_CACHE: Dict[str, str] = {}


def clear_skill_cache() -> None:
    """Reload skill files for an explicitly requested fresh run or test."""
    _SKILL_CACHE.clear()
    _SKILL_FINGERPRINT_CACHE.clear()


def skills_for_prompt(prompt_name: str) -> tuple[str, ...]:
    """Return the skills explicitly routed to a registered prompt."""
    return SKILL_ROUTES.get(str(prompt_name or ""), ())


def _skills_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "skills"
        if candidate.is_dir():
            return candidate
    return here.parent / "skills"


def _skill_path(name: str) -> Optional[Path]:
    key = str(name or "").strip()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", key):
        return None
    root = _skills_dir()
    directory_skill = root / key / "SKILL.md"
    if directory_skill.is_file():
        return directory_skill
    legacy_skill = root / f"{key}.md"
    return legacy_skill if legacy_skill.is_file() else None


def _parse_skill_document(text: str) -> tuple[Dict[str, str], str]:
    """Read the portable name/description frontmatter used by SKILL.md."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}, text
    metadata: Dict[str, str] = {}
    for line in lines[1:end]:
        if line.startswith(("name:", "description:")):
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, "\n".join(lines[end + 1:])


def skill_fingerprint(name: str) -> str:
    """Return the SHA-256 of the exact skill file used by the loader."""
    load_skill(name)
    return _SKILL_FINGERPRINT_CACHE.get(str(name or "").strip(), "")


def validate_skill_registry() -> List[str]:
    """Report missing, malformed, or unroutable skill definitions."""
    errors: List[str] = []
    for prompt_name, skill_names in SKILL_ROUTES.items():
        if prompt_name not in PROMPTS:
            errors.append(f"{prompt_name}: unknown prompt")
        if len(skill_names) != len(set(skill_names)):
            errors.append(f"{prompt_name}: duplicate skill route")
        for skill_name in skill_names:
            path = _skill_path(skill_name)
            if path is None or not path.is_file():
                errors.append(f"{prompt_name}: missing skill {skill_name}")
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                errors.append(f"{skill_name}: unreadable skill ({exc})")
                continue
            metadata, body = _parse_skill_document(text)
            if metadata.get("name") != skill_name:
                errors.append(f"{skill_name}: frontmatter name must match folder")
            if path.name == "SKILL.md" and path.parent.name != skill_name:
                errors.append(f"{skill_name}: folder name must match frontmatter name")
            if not metadata.get("description"):
                errors.append(f"{skill_name}: missing description")
            elif len(metadata["description"]) > 1024:
                errors.append(f"{skill_name}: description exceeds 1024 characters")
            if not body.strip():
                errors.append(f"{skill_name}: empty skill body")
    return errors


def load_skill(name: str) -> str:
    """Load a skill body on demand from SKILL.md or the legacy flat format.

    Skills are cached after first use so a run cannot observe a mid-run edit.
    """
    key = str(name or "").strip()
    if not key:
        return ""
    if key in _SKILL_CACHE:
        return _SKILL_CACHE[key]
    path = _skill_path(key)
    if path is None or not path.is_file():
        _SKILL_CACHE[key] = ""
        _SKILL_FINGERPRINT_CACHE[key] = ""
        return ""
    try:
        raw_bytes = path.read_bytes()
        raw_text = raw_bytes.decode("utf-8")
    except (OSError, UnicodeError):
        logger.exception("Unable to load skill %s", key)
        _SKILL_CACHE[key] = ""
        _SKILL_FINGERPRINT_CACHE[key] = ""
        return ""
    _, text = _parse_skill_document(raw_text)
    _SKILL_CACHE[key] = text
    _SKILL_FINGERPRINT_CACHE[key] = hashlib.sha256(raw_bytes).hexdigest()
    return text


def skill_section(name: str, section: str) -> str:
    """Return one ``## section`` body from a skill file ("" when missing)."""
    body = load_skill(name)
    if not body:
        return ""
    target = str(section or "").strip().lower()
    current: Optional[str] = None
    collected: List[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current == target and collected:
                break
            current = line[3:].strip().lower()
            collected = []
            continue
        if current == target:
            collected.append(line)
    text = "\n".join(collected).strip()
    return text if current == target else ""
