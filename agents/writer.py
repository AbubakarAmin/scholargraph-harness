"""
WriterAgent - Drafts paper sections and integrates citations and code outputs.
Handles abstract, introduction, methods, results, and conclusion sections.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.config import config
from core.utils import log_agent_action, extract_citations, is_degenerate_llm_output, strip_markdown_headers
from core.llm import call_llm, generate_embedding, get_llm_client
from core.context import RunContext, get_active_context
from core.contracts import ExperimentOutput, Plan, PlanSection, Topic
from core.memory import memory

logger = logging.getLogger(__name__)

# Section name -> v2 registry prompt key (central system/threading seam).
_WRITER_SECTION_PROMPTS = {
    "abstract": "writer.abstract",
    "introduction": "writer.introduction",
    "related work": "writer.related",
    "related_work": "writer.related",
    "methods": "writer.methods",
    "method": "writer.methods",
    "experiments": "writer.experiments",
    "results": "writer.results",
    "conclusion": "writer.conclusion",
    "discussion": "writer.discussion",
}


def _writer_system(section_name: str, paper_type: str = "experimental") -> Dict[str, Any]:
    """System kwarg from the v2 registry (stub-safe: {} when unavailable)."""
    try:
        from core.prompts import render_system

        key = _WRITER_SECTION_PROMPTS.get((section_name or "").lower(), "writer.generic")
        # All writer systems share the paper_type slot — always fill it so no
        # literal {paper_type} ever reaches the model (generic sections too).
        system = render_system(key, section=section_name,
                               paper_type=paper_type or "experimental")
        return {"system": system} if system else {}
    except Exception:
        return {}


_MIN_SECTION_BODY_CHARS = {
    "abstract": 120,
    "introduction": 400,
    "related work": 400,
    "methods": 300,
    "experiments": 250,
    "results": 200,
    "discussion": 250,
    "conclusion": 150,
    "limitations": 120,
}

# Writer prompt packs per no-experiment paper type (proposal §7.3, M1).
# Each pack fixes the manuscript headers, the evidence policy, and the
# causal-language governor for that type.
WRITER_PACKS = {
    "analytical": {
        "headers": ["Introduction", "Background", "Analysis", "Discussion", "Limitations", "Conclusion"],
        "policy": (
            "Analytical synthesis pack: use synthesis headers (Background, Analysis). "
            "Every bridge claim must cite its source by DOI or arXiv ID from the retrieved "
            "literature evidence. Use no causal verbs (causes, drives, produces, ATE) beyond "
            "what the cited evidence states."
        ),
    },
    "descriptive": {
        "headers": ["Introduction", "Background", "Findings", "Discussion", "Limitations", "Conclusion"],
        "policy": (
            "Descriptive profile pack: use profile headers (Background, Findings). "
            "Report confidence intervals for every quantitative profile statement. "
            "Descriptive findings are associational: causal verbs (causes, caused, causing, "
            "leads to, results in, produces, drives, causal, treatment effect, ATE) are "
            "forbidden and trigger a redraft."
        ),
    },
    "argumentative": {
        "headers": ["Introduction", "Thesis", "Evidence", "Counterargument", "Discussion", "Limitations", "Conclusion"],
        "policy": (
            "Argumentative pack: use Thesis / Evidence / Counterargument headers. "
            "The Thesis section states the thesis claim verbatim. The Counterargument section "
            "must present the steelman counter-reading and the concession test. "
            "A strength governor applies: never overstate — qualify claims the evidence does not settle."
        ),
    },
    "review": {
        "headers": ["Introduction", "Methods", "Results", "Discussion", "Limitations", "Conclusion"],
        "policy": (
            "Review pack: use protocol headers (Introduction, Methods-protocol, Results-corpus, "
            "Discussion, Limitations-coverage, Conclusion). Cite the PRISMA flow in Methods "
            "(identified / screened / eligible / included with exact counts). "
            "No meta-analytic claims unless the synthesis reports ≥3 commensurate effects "
            "with heterogeneity I²; otherwise state the explicit why-not-meta sentence. "
            "Every material claim cites an extracted DOI; closed sources are marked "
            "full_text_unavailable and synthesized abstract-only."
        ),
    },
    # Field packs (proposal §7.3, M3): instrument/protocol headers with the
    # prereg NUMBER-guard (numbers trace to the pack, never invented).
    "survey": {
        "headers": ["Introduction", "Related Work", "Methods", "Results", "Discussion", "Limitations", "Conclusion"],
        "policy": (
            "Survey pack: use instrument headers (Introduction, Related Work, "
            "Methods-instrument+sample, Results, Discussion, "
            "Limitations-nonresponse, Conclusion). Report the response-table "
            "hash, the codebook hash, item nonresponse rates and the unit "
            "nonresponse rate. Every quantitative claim carries n with a 95% "
            "confidence interval. Report only preregistered subgroups "
            "(prereg NUMBER-guard); non-preregistered cuts stay unreported. "
            "Never invent respondents: fabricating response rows is a "
            "release-blocker."
        ),
    },
    "case_study": {
        "headers": ["Introduction", "Case", "Methods", "Findings", "Discussion", "Limitations", "Conclusion"],
        "policy": (
            "Case pack: use protocol headers (Introduction, Case&Context, "
            "Methods-protocol, Findings, Discussion, Limitations, Conclusion). "
            "Every material claim triangulates to ≥2 independent source types "
            "from the case pack; single-source claims carry an explicit "
            "single_source_flag plus a Limitations entry. Address ≥1 rival "
            "explanation or record the gap explicitly (prereg NUMBER-guard: "
            "counts trace to pack entries, never invented)."
        ),
    },
    "empirical": {
        "headers": ["Introduction", "Related Work", "Methods", "Experiments", "Results", "Discussion", "Limitations", "Conclusion"],
        "policy": (
            "Empirical-secondary pack: use protocol headers with the dataset "
            "protocol (source, access date, license, hash, row cap). "
            "Secondary-data claims trace to the pinned dataset hash "
            "(prereg NUMBER-guard); no primary-data collection is claimed."
        ),
    },
    # Experiment packs (proposal §7.3, M4): instrument/protocol/DAG headers
    # with the prereg NUMBER-guard (numbers trace to recorded outputs).
    "comparative": {
        "headers": ["Introduction", "Related Work", "Methods", "Experiments", "Results", "Discussion", "Limitations", "Conclusion"],
        "policy": (
            "Comparative pack: report the full baseline matrix (every declared "
            "baseline on the SAME split) with per-method n/std/CI, the "
            "proposed-vs-best-baseline delta, and the compulsory ablation "
            "(leave-one-component-out with degradation). "
            "Prereg NUMBER-guard: every number copies exactly from the "
            "recorded experiment outputs."
        ),
    },
    "cause_effect": {
        "headers": ["Introduction", "Related Work", "Methods", "Experiments", "Results", "Discussion", "Limitations", "Conclusion"],
        "policy": (
            "Cause-effect pack: use DAG headers (Methods states the DAG — "
            "treatment, outcome, adjustment set — plus the identification "
            "strategy; Results reports the estimate with CI; Discussion "
            "carries the sensitivity note and the placebo result or its "
            "not_feasible limitation). Causal verbs (causes, ATE, treatment "
            "effect) are allowed ONLY at verb_level causal; otherwise the "
            "governor restates claims associationally. Prereg NUMBER-guard: "
            "every number copies exactly from the recorded outputs."
        ),
    },
}

# Causal verbs forbidden in descriptive (and evidence-bounded analytical) prose.
# The causal-audit governor owns these terms for cause_effect (M4); here they
# are a lint signal that forces a redraft, not a terminal failure.
CAUSAL_VERBS = (
    "causes", "caused", "causing", "leads to", "lead to", "result in", "results in",
    "resulted in", "produces", "produced", "drives", "driven by", "causal",
    "treatment effect", "average treatment effect",
)


def lint_causal_verbs(text: str) -> List[str]:
    """Return the forbidden causal-verb hits in descriptive prose (M1 verb lint)."""
    lowered = (text or "").lower()
    return [verb for verb in CAUSAL_VERBS if verb in lowered]

class DraftChecker:
    """Deterministic co-author: lint + verify a draft without LLM calls."""

    @staticmethod
    def check(section_name: str, content: str, paper_type: str = "experimental") -> Dict[str, Any]:
        from core.verification import extract_citation_ids, filler_prose_hits

        findings: List[str] = []
        body = strip_markdown_headers(content or "")
        if is_degenerate_llm_output(content, min_chars=_MIN_SECTION_BODY_CHARS.get(section_name.lower(), 200)):
            findings.append("degenerate: too short or stub")
        if "DRAFT_FAILED" in (content or ""):
            findings.append("draft_failed: generation failed, blocking release")
        for hit in filler_prose_hits(content or ""):
            findings.append(f"template filler: '{hit[:60]}...'")
            break
        if paper_type == "descriptive":
            hits = lint_causal_verbs(content or "")
            if hits:
                findings.append(f"causal verbs forbidden in descriptive: {hits}")
        ids = extract_citation_ids(content or "")
        return {
            "passed": not findings,
            "findings": findings,
            "n_dois": len(ids.get("dois", [])),
            "n_arxiv": len(ids.get("arxiv_ids", [])),
        }


class WriterAgent:
    """Agent for drafting research paper sections (Drafter + Checker lab pair)."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        self._active_revision_feedback: Optional[str] = None

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    @property
    def vector_memory(self):
        return self.context.memory if self.context else memory

    def draft_section(self, section_name: str, topic: Topic,
                      plan: Plan, engineer_outputs: Dict[str, ExperimentOutput],
                      revision_feedback: Optional[str] = None,
                      paper_type: Optional[str] = None,
                      thesis: Optional[Dict[str, Any]] = None) -> str:
        """Draft a specific section of the research paper.

        revision_feedback carries deterministic check failures / reviewer
        feedback from a prior draft so revision passes repair specific
        defects instead of re-rolling the same prompt blind.

        paper_type + thesis select a §7.3 writer pack (analytical /
        descriptive / argumentative). Descriptive drafts run the causal-verb
        lint and redraft once (bounded) when forbidden verbs appear.
        """
        paper_type = paper_type or (plan or {}).get("paper_type") or "experimental"
        log_agent_action("WriterAgent", "start_drafting", {"section": section_name, "paper_type": paper_type})
        logger.info("Drafting section: %s for topic: %s", section_name, topic.get("title", "unknown"))
        if revision_feedback:
            logger.info("Revision feedback present for section '%s': %.200s...", section_name, revision_feedback)
        self._active_revision_feedback = revision_feedback
        # Carried into _draft_with_retry for registry system rendering.
        self._active_paper_type = paper_type

        # Released exemplars only — cold-start returns empty until runs clear the release gate.
        exemplars = self._released_exemplars(section_name)

        # Get section requirements from plan
        section_plan = self._get_section_plan(section_name, plan)

        # Generate content based on section type
        if section_name.lower() == 'abstract':
            content = self._draft_abstract(topic, plan, engineer_outputs, exemplars=exemplars, paper_type=paper_type, thesis=thesis)
        elif section_name.lower() == 'introduction':
            content = self._draft_introduction(topic, plan, exemplars=exemplars, paper_type=paper_type, thesis=thesis)
        elif section_name.lower() == 'related work':
            content = self._draft_related_work(topic, plan)
        elif section_name.lower() == 'methods':
            content = self._draft_methods(topic, plan, engineer_outputs)
        elif section_name.lower() == 'experiments':
            content = self._draft_experiments(topic, plan, engineer_outputs)
        elif section_name.lower() == 'results':
            content = self._draft_results(topic, plan, engineer_outputs)
        elif section_name.lower() == 'discussion':
            content = self._draft_discussion(topic, plan, engineer_outputs)
        elif section_name.lower() == 'conclusion':
            content = self._draft_conclusion(topic, plan, engineer_outputs)
        else:
            content = self._draft_generic_section(section_name, topic, plan, engineer_outputs, paper_type=paper_type, thesis=thesis)

        # Descriptive verb lint (§7.3): causal verbs trigger exactly one
        # bounded redraft with the hits as revision feedback — never a loop.
        if paper_type == "descriptive" and not revision_feedback:
            hits = lint_causal_verbs(content)
            if hits:
                logger.info("Descriptive verb lint hit %s in '%s' — bounded redraft", hits, section_name)
                log_agent_action("WriterAgent", "verb_lint_redraft", {
                    "section": section_name,
                    "hits": hits,
                })
                lint_feedback = (
                    "Descriptive verb lint: remove these causal verbs entirely and restate "
                    "the findings associationally (use 'associated with', 'observed', "
                    "'confidence interval'): " + ", ".join(hits)
                )
                self._active_revision_feedback = lint_feedback
                content = self._redraft_without_verbs(section_name, topic, plan, engineer_outputs, paper_type, thesis, content)

        # Store in memory
        self._store_section(section_name, content, topic)
        self._active_revision_feedback = None

        logger.info("Section '%s' complete: %d characters", section_name, len(content))

        log_agent_action("WriterAgent", "section_complete", {
            "section": section_name,
            "content_length": len(content),
            "revised": bool(revision_feedback),
            "paper_type": paper_type,
        })

        return content

    def _pack_block(self, paper_type: str, thesis: Optional[Dict[str, Any]] = None) -> str:
        """Render the §7.3 writer pack policy + thesis for prompt grounding."""
        pack = WRITER_PACKS.get(paper_type or "")
        if not pack:
            return ""
        block = "\n\nPAPER-TYPE PACK (" + paper_type + "): " + pack["policy"]
        block += "\nRequired headers: " + ", ".join(pack["headers"]) + "."
        if thesis:
            block += (
                "\nStructured thesis:\nClaim: " + str(thesis.get("claim") or "")[:800]
                + "\nSteelman: " + str(thesis.get("steelman") or "")[:800]
                + "\nConcession: " + str(thesis.get("concession") or "")[:800]
            )
        return block

    def _redraft_without_verbs(self, section_name: str, topic: Topic, plan: Plan,
                               engineer_outputs: Dict[str, ExperimentOutput],
                               paper_type: str, thesis: Optional[Dict[str, Any]],
                               prior: str) -> str:
        """One bounded verb-lint redraft (never recursive)."""
        prompt = f"""
        Rewrite the {section_name} section below for a descriptive research paper.

        Topic: {topic['title']}
        Description: {topic['description']}

        Prior draft (contains forbidden causal verbs — remove every one):
        {prior[:6000]}
        {self._pack_block(paper_type, thesis)}

        Rewrite associationally with confidence intervals. Keep all numbers
        exactly as stated. Write professional academic content.
        """
        return self._draft_with_retry(section_name, prompt, temperature=0.5, fallback=prior)

    def _revision_block(self) -> str:
        """Prompt block injecting prior check failures / reviewer feedback."""
        feedback = getattr(self, "_active_revision_feedback", None)
        if not feedback:
            return ""
        return (
            "\n\nREVISION REQUIRED — a previous draft of this section failed deterministic checks "
            "or review. Repair every issue below in the new draft; keep what already passed intact:\n"
            + str(feedback)[:2400]
            + "\n"
        )

    def _literature_block(self, topic: Topic, max_papers: int = 6, snippet_chars: int = 400) -> str:
        """Render retrieved literature evidence for prompt grounding."""
        papers = topic.get("literature_evidence") or []
        slim = []
        for paper in papers[:max_papers]:
            if not isinstance(paper, dict):
                continue
            slim.append({
                "title": paper.get("title", ""),
                "abstract": (paper.get("abstract") or "")[:snippet_chars],
                "doi": paper.get("doi"),
                "arxiv_id": paper.get("arxiv_id"),
                "year": paper.get("year"),
            })
        slim = [p for p in slim if p["title"] or p["abstract"]]
        if not slim:
            return "No retrieved literature evidence is available. Do not cite any paper by DOI, arXiv ID, or author-year."
        return json.dumps(slim, default=str)[:8000]

    def _released_exemplars(self, section_name: str) -> List[Dict[str, Any]]:
        """Only released structured exemplars may enter Writer prompts."""
        return self.vector_memory.get_prompt_context(
            namespace="writer_exemplars",
            outcome_status="released",
            k=3,
            purpose=section_name.lower(),
        )
    
    def _get_section_plan(self, section_name: str, plan: Plan) -> PlanSection:
        """Get the plan for a specific section."""
        for section in plan.get('sections', []):
            if section['name'].lower() == section_name.lower():
                return section
        return {}

    def _min_body_chars(self, section_name: str) -> int:
        return _MIN_SECTION_BODY_CHARS.get(section_name.lower(), 200)

    def _claim_scope_block(
        self, plan: Plan, topic: Topic, engineer_outputs: Dict[str, Any]
    ) -> str:
        """v5: carry falsifiable prediction + scope ban into Results/Abstract prompts."""
        plan = plan or {}
        topic = topic or {}
        outputs = engineer_outputs or {}
        has_real_ci = False
        for output in outputs.values():
            if not isinstance(output, dict):
                continue
            blob = json.dumps(output, sort_keys=True, default=str).lower()
            if "ci_low" in blob or "confidence_interval_95" in blob:
                has_real_ci = True
        pred = ""
        for exp in (plan.get("experiments") or []):
            if isinstance(exp, dict) and exp.get("falsifiable_prediction"):
                pred = str(exp["falsifiable_prediction"])[:500]
                break
        if not pred and isinstance(topic.get("structured_hypothesis"), dict):
            pred = str(topic["structured_hypothesis"].get("falsification_condition") or "")[:500]
        exploratory = bool(
            plan.get("exploratory_no_release")
            or plan.get("comparative_language_forbidden")
            or (topic.get("fallback_provenance") or topic.get("exploratory_no_release"))
            or any(str((o or {}).get("outcome") or "") == "fallback_synthetic" for o in outputs.values() if isinstance(o, dict))
        )
        block = ""
        if pred:
            block += f"\nFalsifiable prediction (evaluate explicitly as supported/falsified/inconclusive): {pred}"
        if exploratory or not has_real_ci:
            block += (
                "\nSCOPE BAN: no computed difference CI is available, so comparative adjectives "
                "are FORBIDDEN (never write 'wider than reported', 'often include zero', "
                "'significantly better', 'outperform'). Describe the plumbing outcome only, "
                "state the verdict as inconclusive, and disclose that no interval was computed."
            )
        return block

    def _draft_with_retry(
        self,
        section_name: str,
        prompt: str,
        *,
        temperature: float = 0.6,
        fallback: Optional[str] = None,
        paper_type: Optional[str] = None,
    ) -> str:
        """Call the LLM once, retry on empty/garbled stubs, then fail loudly (v5)."""
        logger.info("LLM call for section '%s' (max 2 attempts, min %d chars)", section_name, self._min_body_chars(section_name))
        last_content = ""
        min_chars = self._min_body_chars(section_name)
        full_prompt = prompt + self._revision_block()
        for attempt in range(2):
            try:
                _ptype = paper_type or getattr(self, "_active_paper_type", None) or "experimental"
                raw = call_llm(full_prompt, temperature=temperature, tier="strong",
                               **_writer_system(section_name, _ptype))
                formatted = self._format_section_content(raw, section_name)
                body = strip_markdown_headers(formatted)
                if not is_degenerate_llm_output(formatted, min_chars=min_chars) and len(body) >= min_chars:
                    return formatted
                last_content = formatted
                logger.warning("Degenerate output for section '%s' on attempt %d", section_name, attempt + 1)
                log_agent_action("WriterAgent", "degenerate_section", {
                    "section": section_name,
                    "attempt": attempt + 1,
                    "content_length": len(formatted),
                    "preview": formatted[:120],
                })
            except Exception as exc:
                # d5b5a1db: a single transient LLM error (429 / breaker /
                # truncation_empty) hit `break` here, so the documented
                # second attempt never ran and Abstract went DRAFT_FAILED.
                # Log and CONTINUE so the second attempt actually runs.
                log_agent_action("WriterAgent", f"{section_name.lower().replace(' ', '_')}_error", {"error": str(exc)[:300], "attempt": attempt + 1})
                last_content = last_content  # keep any prior degenerate output for fallback
                continue
        # v5 (f537f638: 547-char template filler shipped as Methods): never
        # return template filler as a completed section. Fallbacks are explicit
        # failure markers so hard_verify_section blocks release instead.
        if fallback and "DRAFT_FAILED" not in fallback:
            from core.verification import filler_prose_hits as _filler_hits

            if not _filler_hits(fallback):
                return fallback
        if last_content and not is_degenerate_llm_output(last_content, min_chars=20):
            return last_content
        return self._format_section_content(
            f"DRAFT_FAILED: section draft unavailable after degenerate model output for {section_name}. "
            "No publishable prose generated; blocking release pending repair.",
            section_name,
        )

    def draft_with_lab_repair(
        self,
        section_name: str,
        prompt: str,
        *,
        paper_type: str = "experimental",
        temperature: float = 0.6,
        max_repairs: int = 3,
    ) -> Dict[str, Any]:
        """Drafter+Checker lab loop: draft, deterministically check, repair.

        Returns {content, passed, attempts, findings}. The Checker never calls
        the LLM; repairs re-enter _draft_with_retry with findings as feedback.
        """
        feedback = self._revision_block()
        content = self._draft_with_retry(section_name, prompt + feedback, temperature=temperature)
        findings: List[str] = []
        attempts = 1
        for _ in range(max_repairs):
            report = DraftChecker.check(section_name, content, paper_type)
            findings = report["findings"]
            if report["passed"]:
                return {"content": content, "passed": True, "attempts": attempts, "findings": findings}
            self._active_revision_feedback = (feedback + "\nChecker findings: " + "; ".join(findings))[:2400]
            content = self._draft_with_retry(section_name, prompt, temperature=temperature, fallback=content)
            attempts += 1
        self._active_revision_feedback = None
        return {"content": content, "passed": False, "attempts": attempts, "findings": findings}
    
    def _draft_abstract(self, topic: Topic, plan: Plan,
                       engineer_outputs: Dict[str, ExperimentOutput],
                       exemplars: Optional[List[Dict[str, Any]]] = None,
                       paper_type: str = "experimental",
                       thesis: Optional[Dict[str, Any]] = None) -> str:
        """Draft the abstract section."""
        prompt = f"""
        Write a concise abstract for the following research paper:

        Topic: {topic['title']}
        Description: {topic['description']}
        Research Questions: {plan.get('research_questions', [])}
        Expected Contributions: {plan.get('expected_contributions', [])}
        Released exemplar signals (metadata only): {json.dumps(exemplars or [])[:800]}

        Retrieved literature evidence (grounding context):
        {self._literature_block(topic, max_papers=4, snippet_chars=250)}

        Key Results (if available):
        {self._format_engineer_outputs(engineer_outputs)}
        {self._pack_block(paper_type, thesis)}
        {self._claim_scope_block(plan, topic, engineer_outputs)}

        The abstract should:
        1. State the problem clearly
        2. Describe the approach/methodology
        3. Summarize key results — only numbers copied exactly from Key Results above
        4. Highlight contributions and impact
        5. Be 150-250 words

        Citation policy: only cite sources that appear in the retrieved literature evidence
        above (by DOI or arXiv ID). If the evidence does not support a claim, omit the claim.
        Never invent DOIs, arXiv IDs, author-year pairs, or paper titles.

        Write a professional, academic abstract suitable for a research paper.
        """
        
        return self._draft_with_retry(
            "Abstract",
            prompt,
            temperature=0.6,
            fallback=self._create_fallback_abstract(topic, plan, engineer_outputs),
        )
    
    def _draft_introduction(self, topic: Topic, plan: Plan,
                            exemplars: Optional[List[Dict[str, Any]]] = None,
                            paper_type: str = "experimental",
                            thesis: Optional[Dict[str, Any]] = None) -> str:
        """Draft the introduction section."""
        prompt = f"""
        Write an introduction section for the following research paper:

        Topic: {topic['title']}
        Description: {topic['description']}
        Rationale: {topic.get('rationale', 'N/A')}
        Impact: {topic.get('impact', 'N/A')}
        Research Questions: {plan.get('research_questions', [])}
        Expected Contributions: {plan.get('expected_contributions', [])}
        Released exemplar signals (metadata only): {json.dumps(exemplars or [])[:800]}

        Retrieved literature evidence (use these sources for background and gap claims):
        {self._literature_block(topic)}
        {self._pack_block(paper_type, thesis)}

        The introduction should include:
        1. Background and motivation, grounded in the retrieved literature evidence
        2. Problem statement
        3. Challenges and limitations of existing work (reference the specific papers above)
        4. Our approach and contributions
        5. Paper organization

        Citation policy: every citation must come from the retrieved literature evidence
        above, cited by DOI (doi:10.XXXX/...) or arXiv ID (arXiv:XXXX.XXXXX). Never invent
        DOIs, arXiv IDs, author-year pairs, or paper titles. If the evidence is insufficient
        for a claim, state that explicitly instead of citing.

        Write 2-3 pages of professional academic content.
        """
        
        return self._draft_with_retry(
            "Introduction",
            prompt,
            temperature=0.7,
            fallback=self._create_fallback_introduction(topic, plan),
        )
    
    def _draft_related_work(self, topic: Topic, plan: Plan) -> str:
        """Draft the related work section."""
        retrieved = topic.get("literature_evidence") or plan.get("literature_evidence") or []
        prompt = f"""
        Write a related work section for the following research topic:
        
        Topic: {topic['title']}
        Description: {topic['description']}
        Research Questions: {plan.get('research_questions', [])}

        Retrieved literature evidence (use only these abstracts for paper-specific claims):
        {json.dumps(retrieved, indent=2, default=str)[:12000]}
        
        The related work should:
        1. Survey relevant literature
        2. Identify gaps in existing work
        3. Position our contribution
        4. Discuss limitations of current approaches
        5. Build motivation for our work
        
        Write 2-3 pages of comprehensive literature review.
        Do not invent paper methods, authors, titles, or citations. If the retrieved
        evidence is insufficient, state that the claim is not established.
        """
        
        return self._draft_with_retry(
            "Related Work",
            prompt,
            temperature=0.6,
            fallback=self._create_fallback_related_work(topic, plan),
        )
    
    def _draft_methods(self, topic: Topic, plan: Plan,
                      engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Draft the methods section."""
        prompt = f"""
        Write a methods section for the following research:
        
        Topic: {topic['title']}
        Methodology: {plan.get('methodology', 'N/A')}
        Experiments: {json.dumps(plan.get('experiments', []), indent=2)[:6000]}
        
        Implementation Details:
        {self._format_engineer_outputs(engineer_outputs)}
        
        The methods section should:
        1. Describe the overall approach
        2. Detail the methodology
        3. Explain implementation details
        4. Describe experimental setup
        5. Include algorithms and pseudocode where appropriate
        
        Write 3-4 pages of detailed methodology description.
        """
        
        return self._draft_with_retry(
            "Methods",
            prompt,
            temperature=0.5,
            fallback=self._create_fallback_methods(topic, plan),
        )
    
    def _draft_experiments(self, topic: Topic, plan: Plan,
                          engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Draft the experiments section."""
        prompt = f"""
        Write an experiments section for the following research:
        
        Topic: {topic['title']}
        Experiments: {json.dumps(plan.get('experiments', []), indent=2)[:6000]}
        
        Experimental Results:
        {self._format_engineer_outputs(engineer_outputs)}
        
        The experiments section should:
        1. Describe experimental setup
        2. Detail datasets and baselines
        3. Present experimental results
        4. Include tables and figures
        5. Analyze performance metrics
        
        Write 4-5 pages of comprehensive experimental evaluation.
        """
        
        return self._draft_with_retry(
            "Experiments",
            prompt,
            temperature=0.6,
            fallback=self._create_fallback_experiments(topic, plan),
        )
    
    def _draft_results(self, topic: Topic, plan: Plan,
                      engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Draft the results section."""
        prompt = f"""
        Write a results section for the following research:

        Topic: {topic['title']}
        Expected Contributions: {plan.get('expected_contributions', [])}

        Experimental Results (single source of truth):
        {self._format_engineer_outputs(engineer_outputs)}

        The results section should:
        1. Analyze experimental results
        2. Compare against baselines
        3. Conduct ablation studies
        4. Provide insights and analysis
        5. Discuss implications

        Number policy (deterministic checks enforce this):
        - Copy every number EXACTLY from the Experimental Results source of truth above.
        - Every quantitative sentence must include n= (seed/sample count) and the
          standard deviation or a confidence interval for the reported metric.
        - Report the statistical test outcome (test name and p-value) for each comparison
          where the source of truth provides one.
        - Do not estimate, round, extrapolate, or invent any number. If a measurement is
          absent from the source of truth, describe it qualitatively instead.
        - For each comparison, state whether it supports or falsifies the hypothesis's
          falsifiable prediction; report negative or inconclusive outcomes honestly.
        {self._claim_scope_block(plan, topic, engineer_outputs)}
        """
        
        return self._draft_with_retry(
            "Results",
            prompt,
            temperature=0.6,
            fallback=self._create_fallback_results(topic, plan),
        )
    
    def _draft_conclusion(self, topic: Topic, plan: Plan,
                         engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Draft the conclusion section."""
        prompt = f"""
        Write a conclusion section for the following research:
        
        Topic: {topic['title']}
        Expected Contributions: {plan.get('expected_contributions', [])}
        
        Key Results:
        {self._format_engineer_outputs(engineer_outputs)}
        
        The conclusion should:
        1. Summarize key contributions
        2. Highlight main results
        3. Discuss limitations
        4. Suggest future work
        5. End with impact statement
        
        Write 1-2 pages of conclusion.
        """
        
        return self._draft_with_retry(
            "Conclusion",
            prompt,
            temperature=0.7,
            fallback=self._create_fallback_conclusion(topic, plan),
        )

    def _draft_discussion(self, topic: Topic, plan: Plan,
                          engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Interpret results after engineering without introducing new measurements."""
        prompt = f"""
Write the Discussion section for this paper.
Topic: {topic['title']}
Experimental source of truth:
{self._format_engineer_outputs(engineer_outputs)}
{self._claim_scope_block(plan, topic, engineer_outputs)}

Interpret implications, limitations, and failures. Any quantitative statement must
be copied from the experimental source of truth above; do not estimate, fabricate,
or introduce a new number. If a measurement is absent, describe it qualitatively.
"""
        return self._draft_with_retry(
            "Discussion",
            prompt,
            temperature=0.5,
            fallback="# Discussion\n\nDRAFT_FAILED: discussion unavailable after degenerate model output. Blocking release pending repair.",
        )
    
    def _draft_generic_section(self, section_name: str, topic: Topic,
                               plan: Plan, engineer_outputs: Dict[str, ExperimentOutput],
                               paper_type: str = "experimental",
                               thesis: Optional[Dict[str, Any]] = None) -> str:
        """Draft a generic section via _draft_with_retry for revision feedback and min-char guards."""
        prompt = f"""
        Write a {section_name} section for the following research:

        Topic: {topic['title']}
        Description: {topic['description']}

        The {section_name} should be appropriate for a research paper and cover relevant content for this section.
        Write professional academic content suitable for publication.
        {self._pack_block(paper_type, thesis)}
        """

        try:
            content = self._draft_with_retry(section_name, prompt, temperature=0.6)
            return self._format_section_content(content, section_name)
        except Exception as e:
            log_agent_action("WriterAgent", "generic_section_error", {"error": str(e)})
            return f"Error drafting {section_name} section: {str(e)}"
    
    def _format_engineer_outputs(self, engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Serialize structured evidence without exposing raw Engineer diagnostics."""
        if not engineer_outputs:
            return json.dumps({"experiments": []}, sort_keys=True)

        evidence = []
        failures = []
        for exp_name, output in engineer_outputs.items():
            if not isinstance(output, dict):
                continue
            if not output.get("success"):
                failures.append(self._summarize_engineer_failure(exp_name, output))
                continue
            evidence.append({
                "experiment": exp_name,
                "outcome": output.get("outcome", "inconclusive"),
                "aggregate_metrics": output.get("aggregate_metrics") or {},
                "results": output.get("results") or {},
                "contract_hash": output.get("contract_hash"),
            })
        return json.dumps(
            {"experiments": evidence, "failure_summaries": failures},
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def _summarize_engineer_failure(exp_name: str, output: Dict[str, Any]) -> Dict[str, Any]:
        """Map failure evidence to safe categories; never copy exception text."""
        raw = str(output.get("error") or "").lower()
        if any(token in raw for token in ("timeout", "timed out", "resource")):
            category = "timeout_or_resource"
            impact = "The experiment did not complete within the execution budget."
        elif any(token in raw for token in ("sandbox", "forbidden", "blocked")):
            category = "sandbox_validation"
            impact = "The generated implementation was rejected by execution policy."
        elif any(token in raw for token in ("syntax", "nameerror", "typeerror", "import", "dependency", "api")):
            category = "dependency_or_api"
            impact = "The implementation could not be executed with the available runtime dependencies."
        elif "claim" in raw or output.get("code_claim_consistency"):
            category = "code_claim_mismatch"
            impact = "The implementation did not satisfy the committed methodological claims."
        elif "feasible" in raw or "dataset" in raw:
            category = "plan_infeasible"
            impact = "The committed experiment requirements were not executable in the available environment."
        else:
            category = "missing_metrics"
            impact = "The experiment did not produce a verified result artifact."
        return {
            "experiment": exp_name,
            "status": "failed",
            "category": category,
            "impact": impact,
            "attempt_count": len(output.get("decision_log") or output.get("attempts") or []),
        }
    
    def _format_section_content(self, content: str, section_name: str) -> str:
        """Format section content with proper structure."""
        # Clean up content
        content = content.strip()
        
        # Add section header if not present
        if not content.startswith(f"# {section_name}") and not content.startswith(f"## {section_name}"):
            content = f"# {section_name}\n\n{content}"
        
        return content
    
    def _store_section(self, section_name: str, content: str, topic: Topic):
        """Store section content in memory as generated_narrative (prompt-ineligible)."""
        try:
            from core.run_log import get_tracker

            citations = extract_citations(content)
            tracker = get_tracker()
            self.vector_memory.add_embedding(
                generate_embedding(content),
                {
                    "type": "paper_section",
                    "namespace": "writer_sections",
                    "content_class": "generated_narrative",
                    "retrieval_eligible": False,
                    "agent": "WriterAgent",
                    "run_id": tracker.run_id if tracker else "unknown",
                    "outcome_status": "unknown",
                    "section": section_name,
                    "topic": topic["title"],
                    "content_length": len(content),
                    "citations": citations,
                    "content": content[:2000],
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            log_agent_action("WriterAgent", "section_storage_error", {"error": str(e)})

    def store_released_exemplar(self, section_name: str, topic: Topic, quality_score: float):
        """Publish a prompt-eligible exemplar only after a section clears the release gate."""
        try:
            from core.run_log import get_tracker

            tracker = get_tracker()
            # Structured signal only — no draft body — so exemplars cannot poison prose.
            self.vector_memory.add_embedding(
                generate_embedding(f"{section_name} {topic.get('title', '')}"),
                {
                    "namespace": "writer_exemplars",
                    "content_class": "structured_signal",
                    "retrieval_eligible": True,
                    "agent": "WriterAgent",
                    "run_id": tracker.run_id if tracker else "unknown",
                    "outcome_status": "released",
                    "signal": {
                        "purpose": section_name.lower(),
                        "section": section_name,
                        "topic_title": topic.get("title"),
                        "quality_score": float(quality_score),
                    },
                },
            )
        except Exception as e:
            log_agent_action("WriterAgent", "exemplar_storage_error", {"error": str(e)})
    
    # Fallback methods for error handling (v5: explicit failure markers, never filler prose).
    def _create_fallback_abstract(self, topic: Dict[str, Any], plan: Dict[str, Any],
                                  engineer_outputs: Optional[Dict[str, Any]] = None) -> str:
        # d5b5a1db: when experiments==[] the abstract prompt is contradictory
        # (summarize key results from an empty record) and a transient LLM
        # error produced DRAFT_FAILED -> hard-blocked release. Return an
        # honest protocol-template abstract instead: no numbers, no
        # comparative adjectives, no invented citations — passes hard checks
        # and states the verdict as inconclusive.
        try:
            _exps = (plan or {}).get("experiments") or []
            _has_data = bool(_exps) and bool(engineer_outputs)
        except Exception:
            _has_data = False
        if not _has_data:
            _title = str((topic or {}).get("title") or "this study")
            _fp = str((topic or {}).get("falsifiable_prediction") or (plan or {}).get("claim_scope") or "preregistered prediction")
            return (
                "# Abstract\n\n"
                f"This manuscript describes an evaluation protocol template for {_title}. "
                "No experiments were executed in this run and no empirical results are reported. "
                f"The preregistered expectation ({_fp}) was not evaluated because the experiment record is empty; "
                "the verdict is inconclusive pending a bounded CPU execution on bundled_synthetic with controlled seeds. "
                "No intervals, baseline comparisons, or statistical tests are claimed. "
                "This template is intended to guide a future empirical run rather than to support inferential conclusions."
            )
        return (
            "# Abstract\n\nDRAFT_FAILED: abstract unavailable after degenerate model output. "
            "No publishable prose generated; blocking release pending repair."
        )
    
    def _create_fallback_introduction(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return "# Introduction\n\nDRAFT_FAILED: introduction unavailable after degenerate model output. Blocking release pending repair."

    def _create_fallback_related_work(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return "# Related Work\n\nDRAFT_FAILED: related work unavailable after degenerate model output. Blocking release pending repair."

    def _create_fallback_methods(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return "# Methods\n\nDRAFT_FAILED: methods unavailable after degenerate model output. Blocking release pending repair."

    def _create_fallback_experiments(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return "# Experiments\n\nDRAFT_FAILED: experiments unavailable after degenerate model output. Blocking release pending repair."

    def _create_fallback_results(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return "# Results\n\nDRAFT_FAILED: results unavailable after degenerate model output. Blocking release pending repair."
    
    def _create_fallback_conclusion(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return "# Conclusion\n\nDRAFT_FAILED: conclusion unavailable after degenerate model output. Blocking release pending repair."

# Example usage
if __name__ == "__main__":
    writer = WriterAgent()
    
    example_topic = {
        'title': 'Novel Attention Mechanisms for Transformer Models',
        'description': 'Developing new attention mechanisms that improve efficiency and interpretability',
        'rationale': 'Current attention mechanisms have limitations in scalability',
        'impact': 'Could enable larger, more efficient language models',
        'feasibility': 8
    }
    
    example_plan = {
        'research_questions': ['How can we improve attention efficiency?'],
        'expected_contributions': ['Novel attention mechanism', 'Improved efficiency'],
        'methodology': 'Experimental evaluation with quantitative analysis'
    }
    
    abstract = writer.draft_section('Abstract', example_topic, example_plan, {})
    print(f"Generated abstract ({len(abstract)} characters)") 
