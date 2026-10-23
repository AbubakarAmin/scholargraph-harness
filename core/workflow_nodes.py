"""Phase node implementations for the ScholarGraph research workflow."""

import json
import inspect
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config
from core.context import get_active_context
from core.evidence_gate import build_contract, gate_engineering_outputs, validate_dataset_identity, validate_experiments
from core.llm import call_llm
from core.run_log import CrossRunMemory, get_tracker
from core.research_db import research_db
from core.sandbox_dispatch import execute_multi_seed
from core.state import ResearchState, downgrade_provenance, initialize_state
from core.utils import log_agent_action, parse_json_from_llm
from core.verification import NON_EMPIRICAL_TYPES, FIELD_WAIVER_TYPES, reproducibility_dossier, validate_empirical_claims, verify_citations

from agents.editor import EditorAgent
from agents.data import DataAgent
from agents.analysis import AnalysisAgent
from agents.engineer import EngineerAgent
from agents.execution import ExecutionAgent
from agents.hypothesis_debate import HypothesisDebateSystem
from agents.meta_agent import MetaAgent
from agents.planner import PlannerAgent
from agents.supervisor import SupervisorAgent
from agents.topic_hunter import ResearchSourceUnavailable, TopicHunterAgent
from agents.verification import VerificationAgent
from agents.writer import WriterAgent

logger = logging.getLogger(__name__)


# Phase -> registry prompts pinned for this run (opencode todowrite +
# reproducible-prompt equivalent). _record_work_plan stamps both the work
# plan and the prompt versions into state on phase entry.
_PHASE_PROMPTS: Dict[str, List[str]] = {
    "topic_discovery": ["hunter.seed", "hunter.gap_generation",
                        "hunter.screener", "hunter.novelty",
                        "hunter.hypothesis_core", "hunter.hypothesis_delta",
                        "hunter.rank"],
    "hypothesis_debate": ["debate.proposer", "debate.proposer_respond",
                          "debate.challenger", "debate.followup",
                          "debate.moderator", "debate.repair"],
    "planning": ["planner.generate", "planner.experiments",
                 "planner.falsifiability", "planner.repair"],
    "engineering": ["engineer.generate", "engineer.refine", "engineer.ablate"],
    "writing_narrative": ["writer.abstract", "writer.introduction",
                          "writer.related", "writer.methods"],
    "writing_results": ["writer.experiments", "writer.results",
                        "writer.conclusion", "writer.discussion",
                        "writer.repair"],
    "supervision": ["supervisor.soft_check", "supervisor.review"],
    "editing": ["writer.repair"],
}


def _record_work_plan(state: ResearchState, phase: str, steps: List[str]) -> None:
    """Record the phase work plan + pin prompt versions (best-effort)."""
    try:
        plans = list(state.get("work_plans") or [])
        plans.append({"phase": phase, "steps": list(steps),
                      "at": datetime.now().isoformat()})
        state["work_plans"] = plans
    except Exception:
        logger.exception("Unable to append work plan for phase %s",
                         phase)
    try:
        from core.prompts import (PROMPTS as _REGISTRY, skill_fingerprint,
                                  skills_for_prompt)

        versions = dict(state.get("prompt_version") or {})
        prompt_names = _PHASE_PROMPTS.get(phase, [])
        for name in prompt_names:
            spec = _REGISTRY.get(name)
            if spec:
                versions[name] = str(spec.get("version", "v1"))
        skill_names = {
            skill
            for name in prompt_names
            for skill in skills_for_prompt(name)
        }
        for skill_name in sorted(skill_names):
            fingerprint = skill_fingerprint(skill_name)
            if fingerprint:
                versions[f"skill:{skill_name}"] = fingerprint
        state["prompt_version"] = versions
    except Exception:
        logger.exception("Unable to record prompt and skill versions for phase %s",
                         phase)


def _create_agent(agent_class):
    context = get_active_context()
    if context is None:
        return agent_class()
    # Keep the composition root compatible with simple test/demo doubles and
    # older third-party agents that still expose a no-argument constructor.
    # Inspect the signature rather than catching arbitrary TypeErrors raised
    # from inside an agent constructor.
    try:
        parameters = list(inspect.signature(agent_class).parameters.values())
        accepts_context = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters) or any(
            p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for p in parameters
        )
    except (TypeError, ValueError):
        accepts_context = True
    return agent_class(context) if accepts_context else agent_class()


def topic_discovery_node(state: ResearchState) -> ResearchState:
    logger.info(f"Topic discovery starting at iteration {state['iteration']}")
    log_agent_action("Orchestrator", "start_topic_discovery", {"iteration": state["iteration"]})
    _record_work_plan(state, "topic_discovery", [
        "generate diverse seeds, then gaps per seed (cheap drafter)",
        "screen + novelty-judge every gap (judge tier), keep winners",
    ])
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("topic_discovery")
    try:
        hunter = _create_agent(TopicHunterAgent)
        try:
            hunt_mode = str(state.get("hunt_mode") or "try_and_report")
            topics = hunter.discover_topics(config.research_domain, hunt_mode=hunt_mode)
        except TypeError:
            topics = hunter.discover_topics(config.research_domain)
            hunt_mode = str(state.get("hunt_mode") or "try_and_report")
        if topics:
            # P1e: keep the full ranked list (multi-paper from one hunt) and
            # tag provenance; downstream picks winner but losers stay for
            # exploratory artifacts instead of being discarded.
            try:
                for t in topics:
                    if isinstance(t, dict):
                        t.setdefault("hunt_mode", hunt_mode)
                state["hunt_mode"] = hunt_mode
            except Exception:
                pass
            # Lab upgrade: enrich with citation-graph + evidence bridges +
            # contradictions (best-effort), then fan-out top-k for parallel
            # debate instead of silently discarding losers.
            try:
                from agents.lab_society import enrich_topics_with_lab_graph, hunter_fan_out, post_handoff

                topics = enrich_topics_with_lab_graph(topics, state)
                shortlist = hunter_fan_out(topics, k=3)
                try:
                    state["lab_topic_shortlist"] = shortlist  # type: ignore[typeddict-unknown-key]
                except Exception:
                    pass
                post_handoff(state, "Hunter", "Debate", {
                    "n_topics": len(topics),
                    "shortlist": [t.get("title", "") for t in shortlist],
                })
            except Exception:
                pass
            state["topics"] = topics
            # B4: debate-skipping recipes (review/empirical/descriptive/case/
            # survey) never run hypothesis_debate, which is the only node
            # that sets selected_topic. Promote the top-ranked topic so
            # thesis_argument/systematic_review/planning have a topic.
            # Debate overwrites this when it runs and picks a winner.
            if not state.get("selected_topic"):
                try:
                    state["selected_topic"] = topics[0]
                except Exception:
                    pass
            state["current_phase"] = _next_phase(state, "topic_discovery", "hypothesis_debate")
            logger.info(f"Topic discovery found {len(topics)} topic(s): {[topic['title'] for topic in topics[:3]]}")
            log_agent_action("Orchestrator", "topics_discovered", {"count": len(topics), "iteration": state["iteration"], "topics": [topic["title"] for topic in topics[:3]]})
        elif state["iteration"] + 1 >= config.max_iterations:
            # Exhausting discovery is a failed research run, not a successful
            # completion.  Preserve that distinction so the artifact layer
            # writes a failure dossier rather than an empty normal summary.
            # (Was hardcoded to 5 while MAX_ITERATIONS=10 — debate-fail loops
            # ran twice as long as discovery-fail loops for no reason.)
            logger.warning("No viable topic found after multiple attempts")
            message = "No viable, evidence-supported, sandbox-executable topic was discovered after multiple attempts"
            state["current_phase"] = "complete"
            state["should_continue"] = False
            state["should_reset"] = False
            state["terminal_error"] = message
            state["technical_failures"] = {
                "topic_discovery": {
                    "success": False,
                    "failure_kind": "research_exhausted",
                    "reason_code": "no_viable_topic_after_retries",
                    "message": message,
                }
            }
            state["meta_feedback"].append(message)
            log_agent_action("Orchestrator", "no_topics_found_after_retries", {
                "iteration": state["iteration"],
                "reason_code": "no_viable_topic_after_retries",
            })
        else:
            state["should_reset"] = True
            state["meta_feedback"].append("No topics discovered - resetting")
            logger.warning(f"No topics discovered at iteration {state['iteration']}")
            log_agent_action("Orchestrator", "no_topics_found", {"iteration": state["iteration"]})
        return state
    except ResearchSourceUnavailable as exc:
        message = str(exc)
        logger.warning(f"Research sources unavailable: {message}")
        state["terminal_error"] = message
        state["meta_feedback"].append(message)
        state["current_phase"] = "complete"
        state["should_continue"] = False
        log_agent_action("Orchestrator", "research_sources_unavailable", {"message": message})
        return state
    except Exception as exc:
        logger.error(f"Topic discovery failed: {exc}", exc_info=True)
        # L8: config-driven cap (was hardcoded 5 while MAX_ITERATIONS=10).
        if state["iteration"] >= config.max_iterations:
            state["current_phase"] = "complete"
            state["meta_feedback"].append(f"Topic discovery failed after multiple attempts: {exc}")
        else:
            state["should_reset"] = True
            state["meta_feedback"].append(f"Topic discovery error: {exc}")
        return state


def hypothesis_debate_node(state: ResearchState) -> ResearchState:
    logger.info(f"Hypothesis debate starting with {len(state['topics'])} topic(s)")
    log_agent_action("Orchestrator", "start_hypothesis_debate", {"topics_remaining": len(state["topics"])})
    _record_work_plan(state, "hypothesis_debate", [
        "proposer argues from evidence ids; challenger attacks with severity",
        "moderator ensemble judges vs threshold; repair or exploratory handoff",
    ])
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("hypothesis_debate")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    # B5 (§4.4 mini): recipes without hypothesis_debate (empirical, review,
    # descriptive, case_study, survey, analytical) get exactly the 1-call
    # feasibility+ethics screen — never a full tournament. Blocks only on
    # severity-5 resource/ethics findings; otherwise promotes the top topic.
    # (The superset graph normally skips this node for those recipes; this
    # guard covers the flag-off legacy path and direct calls.)
    _recipe = state.get("pipeline_recipe") or []
    if _recipe and "hypothesis_debate" not in _recipe:
        try:
            if not state.get("selected_topic") and state.get("topics"):
                state["selected_topic"] = state["topics"][0]
            _report = dict(state.get("intake_report") or {})
            _screen = _mini_feasibility_screen(
                str(_report.get("goal") or state.get("user_query") or ""),
                str(_report.get("domain") or getattr(config, "research_domain", "")),
                str(state.get("paper_type") or "experimental"),
                _report.get("pack_manifest"),
            )
            if _screen.get("blocked"):
                state["terminal_error"] = (
                    "needs_primary_data: mini feasibility screen blocked "
                    f"({str(_screen.get('findings'))[:300]})"
                )
                state["current_phase"] = "complete"
                state["should_continue"] = False
                log_agent_action("Orchestrator", "debate_mini_blocked", {"findings": _screen.get("findings")})
                return state
            state["hypothesis_passed"] = True
            log_agent_action("Orchestrator", "debate_mini_passed", {
                "paper_type": state.get("paper_type"),
                "degraded": bool(_screen.get("degraded")),
            })
        except Exception as _mini_exc:
            logger.warning("Debate mini-screen failed open (%s)", str(_mini_exc)[:160], exc_info=True)
            if not state.get("selected_topic") and state.get("topics"):
                state["selected_topic"] = state["topics"][0]
            state["hypothesis_passed"] = True
        state["current_phase"] = _next_phase(state, "hypothesis_debate", "planning")
        return state
    try:
        # Filter out topics that already failed or were rejected (including within-iteration failures)
        excluded = set(t.strip().lower() for t in CrossRunMemory().excluded_topic_titles() if t)
        state["topics"] = [
            t for t in state["topics"]
            if (t.get("title") or "").strip().lower() not in excluded
        ]
        if not state["topics"]:
            state["should_reset"] = True
            return state
        # Pre-debate self-critique: strengthen hypotheses before adversarial debate
        try:
            _th_agent = _create_agent(TopicHunterAgent)
            state["topics"] = [_th_agent.pre_debate_self_critique(t) for t in state["topics"]]
        except Exception as critique_err:
            log_agent_action("Orchestrator", "pre_debate_critique_error", {"error": str(critique_err)})
        topics_tried = 0
        debater = _create_agent(HypothesisDebateSystem)
        # Steelman mode (§4.4): argumentative runs debate the thesis with a
        # steelman challenger. The standard path keeps its exact call shape
        # (test doubles patch single-argument methods).
        debate_mode = "steelman" if str(state.get("paper_type") or "") == "argumentative" else "standard"
        mode_kw = {"mode": debate_mode} if debate_mode != "standard" else {}
        if debate_mode != "standard":
            log_agent_action("Orchestrator", "debate_steelman_mode", {"paper_type": state.get("paper_type")})
        # (topic, result) pairs for the terminal best-effort promotion below.
        attempts_log: list = []
        if len(state["topics"]) >= 3 and hasattr(debater, "conduct_tournament"):
            logger.info(f"Running tournament debate with {len(state['topics'])} topics")
            tournament_results = debater.conduct_tournament(state["topics"], rounds=1, **mode_kw)
            state["debate_results"].extend(tournament_results)
            logger.info(f"Tournament completed: {len(tournament_results)} result(s)")
            winner = next((result for result in tournament_results if result.passed), None)
            if winner:
                state["selected_topic"] = next(
                    topic for topic in state["topics"] if topic.get("title") == winner.topic
                )
                state["hypothesis_passed"] = True
                state["current_phase"] = _next_phase(state, "hypothesis_debate", "planning")
                return state
            for result in tournament_results:
                CrossRunMemory().record_rejection(
                    "topic",
                    getattr(result, "topic", "?"),
                    "infra_llm_unavailable" if getattr(result, "infra_failure", False) else "failed_hypothesis_debate",
                    {
                        "score": getattr(result, "score", None),
                        "decision": getattr(result, "moderator_decision", None),
                        "unresolved": list(getattr(result, "unresolved_objections", []) or [])[:5],
                        "infra_failure": bool(getattr(result, "infra_failure", False)),
                    },
                )
            # P0-1: all-infra tournament → terminal infra (resumable), not reset.
            if tournament_results and all(getattr(r, "infra_failure", False) for r in tournament_results):
                message = (
                    f"All {len(tournament_results)} tournament debate(s) failed on LLM "
                    "infrastructure; no science verdict reached"
                )
                state["terminal_error"] = f"infra_llm_unavailable: {message}"
                state["technical_failures"] = {
                    "hypothesis_debate": {
                        "success": False,
                        "error": message,
                        "failure_kind": "infra",
                        "subsystem": "hypothesis_debate",
                    }
                }
                state["current_phase"] = "complete"
                state["should_continue"] = False
                state["should_reset"] = False
                state["meta_feedback"].append(message)
                log_agent_action("Orchestrator", "hypothesis_debate_infra_unavailable",
                                 {"topics_tried": len(tournament_results)})
                return state
            # Terminal best-effort (same policy as the serial path below):
            # on the last iteration, hand the best near-miss to bounded
            # exploratory planning instead of ending with nothing.
            if state.get("iteration", 0) + 1 >= config.max_iterations and tournament_results:
                threshold = float(getattr(
                    getattr(debater, "runtime_config", None),
                    "debate_pass_threshold", config.debate_pass_threshold,
                ) or config.debate_pass_threshold)
                viable = [r for r in tournament_results if not getattr(r, "infra_failure", False)]
                if viable:
                    best = max(viable, key=lambda r: float(getattr(r, "score", 0) or 0))
                    best_score = float(getattr(best, "score", 0) or 0)
                    if best_score >= threshold - 1.5:
                        promoted = next(
                            (t for t in state["topics"] if t.get("title") == getattr(best, "topic", None)),
                            None,
                        )
                        if promoted is not None:
                            promoted = dict(promoted)
                            promoted["fallback_provenance"] = "best_effort_terminal"
                            state["selected_topic"] = promoted
                            state["debate_results"].append(_fallback_debate_result(promoted, best))
                            state["hypothesis_passed"] = True
                            state["provenance"] = downgrade_provenance(
                                str(state.get("provenance") or "clean"), "exploratory")
                            state["current_phase"] = _next_phase(state, "hypothesis_debate", "planning")
                            state["meta_feedback"].append(
                                f"Last iteration: tournament produced no pass; promoting "
                                f"best-scoring topic ({best_score:.2f} vs bar {threshold}) "
                                f"as bounded exploratory work."
                            )
                            log_agent_action("Orchestrator", "best_effort_topic_promoted", {
                                "topic": promoted.get("title"),
                                "score": best_score,
                                "threshold": threshold,
                            })
                            return state
            state["topics"] = []
            state["should_reset"] = True
            return state
        while state["topics"]:
            current_topic = state["topics"][0]
            state["selected_topic"] = current_topic
            topics_tried += 1
            logger.info(f"Trying topic {topics_tried}: {current_topic['title']} ({len(state['topics'])} remaining)")
            log_agent_action("Orchestrator", "trying_topic", {"topic": current_topic["title"], "attempt": topics_tried, "topics_remaining": len(state["topics"])})
            # Both tournament and serial discovery use the same bounded
            # repair protocol.  Without this branch, a run with one or two
            # candidates silently lost the contract-revision capability.
            attempts = (
                debater.conduct_with_repair(current_topic, **mode_kw)
                if hasattr(debater, "conduct_with_repair")
                else [debater.conduct_debate(current_topic, **mode_kw)]
            )
            state["debate_results"].extend(attempts)
            result = attempts[-1]
            logger.info(f"Debate result for '{current_topic['title']}': passed={result.passed}, score={getattr(result, 'score', None)}")
            if result.passed:
                state["hypothesis_passed"] = True
                state["current_phase"] = _next_phase(state, "hypothesis_debate", "planning")
                log_agent_action("Orchestrator", "hypothesis_passed", {"topic": current_topic["title"], "attempts": topics_tried})
                return state
            # A deterministic discovery fallback is deliberately allowed to
            # enter planning after an inconclusive debate. It is not a claim
            # that the topic is good; it is a labelled exploratory hand-off so
            # the harness can run a minimum experiment and generate evidence.
            # Infra-failed debates (garbled challenger) are excluded: there was
            # no substantive debate to be "inconclusive" about.
            if _exploratory_acceptable(current_topic, result):
                fallback_result = _fallback_debate_result(current_topic, result)
                state["debate_results"].append(fallback_result)
                # v5: label the handoff exploratory so planning/writing narrow
                # claims (no inferential adjectives on inconclusive debates).
                try:
                    marked = dict(current_topic)
                    marked["fallback_provenance"] = "exploratory_handoff_after_inconclusive_debate"
                    marked["exploratory_no_release"] = True
                    state["selected_topic"] = marked
                    state["provenance"] = downgrade_provenance(
                        str(state.get("provenance") or "clean"), "exploratory")
                except Exception:
                    pass
                state["hypothesis_passed"] = True
                state["current_phase"] = _next_phase(state, "hypothesis_debate", "planning")
                state["meta_feedback"].append(
                    "Debate was inconclusive; accepted bounded exploratory topic for evidence generation."
                )
                log_agent_action("Orchestrator", "exploratory_topic_accepted", {
                    "topic": current_topic.get("title"),
                    "reason": "bounded_fallback_after_debate_failure",
                })
                return state
            if current_topic.get("fallback_provenance") and getattr(result, "infra_failure", False):
                log_agent_action("Orchestrator", "exploratory_topic_refused", {
                    "topic": current_topic.get("title"),
                    "reason": "debate_failed_on_infrastructure_no_substantive_rebuttal",
                })
            CrossRunMemory().record_rejection(
                "topic",
                current_topic.get("title", "?"),
                # P0-1: infra rejections must not masquerade as science.
                # Distinct reason keeps Elo/memory clean and lets the exclusion
                # filter retry these topics once the provider recovers.
                "infra_llm_unavailable" if getattr(result, "infra_failure", False) else "failed_hypothesis_debate",
                {
                    "score": getattr(result, "score", None),
                    "decision": getattr(result, "moderator_decision", None),
                    "unresolved": list(getattr(result, "unresolved_objections", []) or [])[:5],
                    "structured_hypothesis": current_topic.get("structured_hypothesis"),
                    "infra_failure": bool(getattr(result, "infra_failure", False)),
                },
            )
            state["topics"] = state["topics"][1:]
            attempts_log.append((current_topic, result))
            log_agent_action("Orchestrator", "topic_failed", {"topic": current_topic["title"], "topics_remaining": len(state["topics"])})
        # Terminal best-effort: on the LAST allowed iteration, promote the
        # highest-scoring debated topic as a bounded exploratory handoff
        # instead of ending a multi-hour run with no artifact at all (run
        # 25ed95d4: 3 iterations × ~2h, scores 7.5–7.75 vs a 7.5 bar, zero
        # output). This is explicitly NOT a scientific pass — the fallback
        # debate record + bounded fallback plan are labelled exploratory, and
        # planning still runs its own feasibility gates. Bar: within 1.5 of
        # the pass threshold, and no infrastructure failure (a garbled debate
        # says nothing about the hypothesis).
        if state.get("iteration", 0) + 1 >= config.max_iterations and attempts_log:
            threshold = float(getattr(
                getattr(debater, "runtime_config", None),
                "debate_pass_threshold", config.debate_pass_threshold,
            ) or config.debate_pass_threshold)
            scored = [
                (t, r) for t, r in attempts_log
                if not getattr(r, "infra_failure", False)
            ]
            if scored:
                best_topic, best_result = max(
                    scored, key=lambda tr: float(getattr(tr[1], "score", 0) or 0))
                best_score = float(getattr(best_result, "score", 0) or 0)
                if best_score >= threshold - 1.5:
                    promoted = dict(best_topic)
                    promoted["fallback_provenance"] = "best_effort_terminal"
                    state["selected_topic"] = promoted
                    state["debate_results"].append(_fallback_debate_result(promoted, best_result))
                    state["hypothesis_passed"] = True
                    state["current_phase"] = _next_phase(state, "hypothesis_debate", "planning")
                    state["meta_feedback"].append(
                        f"Last iteration: no debate pass; promoting best-scoring "
                        f"topic ({best_score:.2f} vs bar {threshold}) as bounded "
                        f"exploratory work, not a scientific endorsement."
                    )
                    log_agent_action("Orchestrator", "best_effort_topic_promoted", {
                        "topic": promoted.get("title"),
                        "score": best_score,
                        "threshold": threshold,
                        "unresolved": len(getattr(best_result, "unresolved_objections", []) or []),
                    })
                    return state
        state["should_reset"] = True
        # P0-1: all-infra means the provider was down, not that the science
        # failed. Terminal with an infra code (resumable) instead of burning
        # the next iteration's discovery budget on more doomed LLM calls.
        if attempts_log and all(getattr(r, "infra_failure", False) for _, r in attempts_log):
            message = (
                f"All {len(attempts_log)} debate(s) failed on LLM infrastructure "
                "(proposer/challenger unavailable); no science verdict reached"
            )
            state["terminal_error"] = f"infra_llm_unavailable: {message}"
            state["technical_failures"] = {
                "hypothesis_debate": {
                    "success": False,
                    "error": message,
                    "failure_kind": "infra",
                    "subsystem": "hypothesis_debate",
                }
            }
            state["current_phase"] = "complete"
            state["should_continue"] = False
            state["should_reset"] = False
            state["meta_feedback"].append(message)
            log_agent_action("Orchestrator", "hypothesis_debate_infra_unavailable",
                             {"topics_tried": len(attempts_log)})
            return state
        state["meta_feedback"].append(f"All {topics_tried} topics failed hypothesis debate")
        logger.warning(f"All {topics_tried} topics failed hypothesis debate")
        log_agent_action("Orchestrator", "all_topics_failed", {"topics_tried": topics_tried})
        return state
    except Exception as exc:
        message = f"Hypothesis debate subsystem crashed: {exc}"
        logger.error(message, exc_info=True)
        state["meta_feedback"].append(message)
        state["terminal_error"] = message
        state["technical_failures"] = {
            "hypothesis_debate": {
                "success": False,
                "error": str(exc),
                "failure_kind": "technical",
                "subsystem": "hypothesis_debate",
            }
        }
        state["current_phase"] = "complete"
        state["should_continue"] = False
        state["should_reset"] = False
        log_agent_action(
            "Orchestrator",
            "hypothesis_debate_technical_failure",
            {"error": str(exc)},
        )
        return state


def planning_node(state: ResearchState) -> ResearchState:
    logger.info("Planning starting")
    log_agent_action("Orchestrator", "start_planning", {})
    _record_work_plan(state, "planning", [
        "draft plan + experiments within sandbox manifest (strong tier)",
        "validate schema + feasibility; repair violating fields only",
    ])
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("planning")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    # P1d: bind deferred runs now that topics/literature exist.
    try:
        state = resolve_deferred_routing(state)
    except Exception:
        pass
    try:
        if not state.get("selected_topic"):
            state["should_reset"] = True
            return state
        planner = _create_agent(PlannerAgent)
        plan = None
        # M4 (§4.5): comparative / cause_effect plans carry per-type
        # schemas. The paper_type travels as an explicit kwarg for those
        # two types only (TypeError fallback keeps narrow test doubles
        # working); every other type keeps the exact legacy call shape.
        _plan_type = str(state.get("paper_type") or "experimental")
        if state.get("plan") and state.get("plan_revision_requests"):
            plan = state["plan"]
            if hasattr(planner, "revise_plan"):
                # e0ec9e91: 7 identical llm_unavailable replays for the same
                # experiment spun 14:16-14:23 then killed the run. Dedupe by
                # (reason, experiment) and cap at 2 replays per key.
                _seen: dict = {}
                _replayed: list = []
                for request in state["plan_revision_requests"] or []:
                    _key = (str((request or {}).get("reason") or ""),
                            str((request or {}).get("experiment") or ""))
                    _seen[_key] = _seen.get(_key, 0) + 1
                    if _seen[_key] > 2:
                        log_agent_action("Orchestrator", "planning_revision_deduped", {
                            "reason": _key[0], "experiment": _key[1],
                            "occurrences": _seen[_key],
                        })
                        continue
                    _replayed.append(request)
                for request in _replayed:
                    try:
                        plan = planner.revise_plan(plan, request, state["selected_topic"])
                    except Exception as _rev_exc:
                        # e0ec9e91 terminal: revise_plan raised LLMUnavailable
                        # (truncation) which escaped as planning_exception_
                        # terminal_failure. Infra during revision means "keep
                        # current plan + record feedback", never a run kill.
                        from core.llm import LLMUnavailable as _LLMU
                        if isinstance(_rev_exc, _LLMU) or "truncat" in str(_rev_exc).lower() \
                                or "llm" in str(_rev_exc).lower():
                            log_agent_action("Orchestrator", "planning_revision_infra_skipped", {
                                "error": str(_rev_exc)[:200],
                                "reason": str((request or {}).get("reason") or ""),
                            })
                            state.setdefault("meta_feedback", []).append(
                                "Plan revision skipped (LLM infra): " + str(_rev_exc)[:200])
                            continue
                        raise
            else:
                # Minimal test/demo planners often only implement create_plan;
                # don't turn that interface difference into a terminal run.
                plan = _minimal_fallback_plan(
                    state["selected_topic"],
                    "planner does not implement revise_plan",
                    _plan_type,
                )
                state["provenance"] = downgrade_provenance(
                    str(state.get("provenance") or "clean"), "fallback")
            state["plan_revision_requests"] = []
        elif _plan_type in ("comparative", "cause_effect"):
            try:
                plan = planner.create_plan(state["selected_topic"], paper_type=_plan_type)
            except TypeError:
                plan = planner.create_plan(state["selected_topic"])
        else:
            plan = planner.create_plan(state["selected_topic"])

        experiments = (plan or {}).get("experiments", [])
        logger.info(f"Plan created with {len(plan.get('sections', []))} sections, {len(experiments)} experiment(s)")
        plan_errors = validate_experiments(experiments, paper_type=_plan_type, plan=plan)
        if plan_errors:
            message = "; ".join(plan_errors)
            attempts = int((plan or {}).get("schema_revision_attempts") or 0)
            if attempts < 2:
                plan = dict(plan or {})
                plan["schema_revision_attempts"] = attempts + 1
                state["plan"] = plan
                revision = {
                    "reason": "invalid_experiment_plan",
                    "experiment": None,
                    "detail": message,
                    "timestamp": datetime.now().isoformat(),
                }
                state.setdefault("plan_revision_requests", []).append(revision)
                CrossRunMemory().record_plan_revision("invalid_experiment_plan", meta=revision)
                state["meta_feedback"].append(f"Plan revision requested for schema errors: {message}")
                state["current_phase"] = "planning"
                state["should_continue"] = True
                log_agent_action("Orchestrator", "planning_schema_error_request_revision", {
                    "message": message,
                    "attempts": attempts + 1,
                })
                return state
            else:
                # L6 single 2-revision policy: exactly 2 repairs, then bounded
                # fallback (preserving the counter so the bound is visible
                # downstream). Dead attempts>=3 branch removed.
                plan = _minimal_fallback_plan(state["selected_topic"], message, _plan_type)
                plan["schema_revision_attempts"] = attempts
                state["plan"] = plan
                state["provenance"] = downgrade_provenance(
                    str(state.get("provenance") or "clean"), "fallback")
                state["plan_revision_requests"] = []
                state["meta_feedback"].append(
                    "Planner contract remained invalid after repairs; using bounded synthetic fallback: " + message
                )
                log_agent_action("Orchestrator", "planning_bounded_fallback", {"message": message})

        state["plan"] = plan
        # M4: recipe-aware successor (comparative planning → data_validation
        # per recipe; legacy runs without a recipe keep the historical label).
        state["current_phase"] = _next_phase(state, "planning", "writing_narrative")
        log_agent_action("Orchestrator", "plan_created", {"sections": len(plan.get("sections", []))})
        return state
    except Exception as exc:
        logger.error(f"Planning failed: {exc}", exc_info=True)
        err_msg = f"Planning failed: {exc}"
        msg_lower = str(exc).lower()
        # Feasibility rejections (sandbox capability manifest) must NOT strand
        # the run the way infra exceptions do. Planner.create_plan already ran
        # rescope + deterministic auto-fix; if banned tokens still survive, fall
        # back to the bounded synthetic plan and let the evidence gate label
        # the result exploratory instead of writing a failure dossier.
        if "sandbox capability manifest" in msg_lower or "feasibility" in msg_lower or "planner_manifest_violation" in msg_lower:
            try:
                fallback = _minimal_fallback_plan(
                    state.get("selected_topic") or {}, err_msg,
                    str(state.get("paper_type") or "experimental"))
            except Exception:
                fallback = _minimal_fallback_plan({}, err_msg)
            state["plan"] = fallback
            state["provenance"] = downgrade_provenance(
                str(state.get("provenance") or "clean"), "fallback")
            state["plan_revision_requests"] = []
            state["feasibility_errors"] = [str(exc)]
            state["meta_feedback"].append(
                "Planner feasibility rejection after rescope+auto-fix; using bounded synthetic fallback: " + str(exc)
            )
            state["current_phase"] = _next_phase(state, "planning", "writing_narrative")
            state["should_continue"] = True
            log_agent_action("Orchestrator", "planning_feasibility_fallback", {"error": str(exc)})
            return state
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["meta_feedback"].append(err_msg)
        state["current_phase"] = "complete"
        state["should_continue"] = False
        log_agent_action("Orchestrator", "planning_exception_terminal_failure", {"error": str(exc)})
        return state


def data_validation_node(state: ResearchState) -> ResearchState:
    """Validate an explicitly requested dataset before experiments begin."""
    logger.info("Data validation starting")
    log_agent_action("Orchestrator", "start_data_validation", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("data_validation")
    try:
        from core.run_log import emit_event as _emit_dv
        _emit_dv("phase_start", {"phase": "data_validation", "agent": "DataAgent",
                                 "status": "ok", "message_human": "data validation start",
                                 "cause_class": "science"}, agent="DataAgent")
    except Exception:
        pass
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    # M3: survey/descriptive/empirical validate the pack response_table here
    # (DataAgent handoff for survey_analysis) even without a plan dataset.
    # Survey/case recipes have no planning phase, so a missing plan must not
    # strand them — validate the pack and continue down the recipe.
    paper_type = str(state.get("paper_type") or "experimental")
    # P1d backstop: deferred runs reaching here without a type get routed now.
    if paper_type in ("auto-deferred", "auto", ""):
        try:
            state = resolve_deferred_routing(state)
            paper_type = str(state.get("paper_type") or "experimental")
        except Exception:
            pass
    if not state.get("plan") and paper_type not in ("survey", "case_study", "descriptive", "empirical"):
        # P3: missing plan is downgradeable, not terminal (except true
        # missing-user-data refusals which stay terminal below). Synthesize
        # the bounded fallback so engineering still has executable work.
        try:
            state["plan"] = _minimal_fallback_plan(
                state.get("selected_topic") or {}, "data_validation missing plan — bounded fallback",
                paper_type,
            )
            state["provenance"] = downgrade_provenance(
                str(state.get("provenance") or "clean"), "fallback")
            state["meta_feedback"].append("Data validation: missing plan downgraded to bounded fallback")
            log_agent_action("Orchestrator", "data_validation_plan_downgraded", {"paper_type": paper_type})
        except Exception as exc:
            err_msg = f"Data validation aborted: No valid plan available ({exc})"
            state["terminal_error"] = state.get("terminal_error") or err_msg
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state
    try:
        plan = state.get("plan") or {}
        if paper_type == "survey":
            from core.packs import resolve_pack_reference

            intake = state.get("intake_report") or {}
            pack = resolve_pack_reference(dict(intake.get("pack_manifest") or {}))
            for key in ("response_table", "codebook"):
                if key in state and state[key] and key not in pack:  # type: ignore[index]
                    pack[key] = state[key]  # type: ignore[index]
            response_ref = pack.get("response_table")
            response_path = None
            target = None
            if isinstance(response_ref, dict):
                response_path = response_ref.get("path") or response_ref.get("location")
                target = response_ref.get("target")
                manifest = response_ref.get("manifest")
                if not response_path and isinstance(manifest, dict):
                    response_path = manifest.get("path") or manifest.get("location")
            elif isinstance(response_ref, str):
                response_path = response_ref
            codebook_ref = pack.get("codebook")
            codebook_path = codebook_ref.get("path") if isinstance(codebook_ref, dict) else (codebook_ref if isinstance(codebook_ref, str) else None)
            if not response_path:
                state["data_validation"] = {"passed": False, "score": 0.0,
                                            "note": "needs_response_table: no response table supplied",
                                            "refusal": {"reason": "needs_response_table",
                                                        "provided": "no response table supplied",
                                                        "alternative": "pick empirical-secondary with public data",
                                                        "how_to_proceed": "upload response_table.csv/json + codebook.csv or pick another type"}}
                state["terminal_error"] = "needs_response_table"
                state["current_phase"] = "complete"
                state["should_continue"] = False
                return state
            codebook_targets: list = []
            if codebook_path:
                from core.packs import validate_codebook_file as _validate_cb
                codebook_result = _validate_cb(codebook_path)
                codebook_targets = list(codebook_result.get("targets") or [])
                if not codebook_result.get("ok"):
                    state["data_validation"] = {"passed": False, "score": 0.0,
                                                "note": "; ".join(codebook_result.get("reasons") or ["codebook invalid"]),
                                                "codebook": codebook_result}
                    state["terminal_error"] = "needs_pack_invalid"
                    state["current_phase"] = "complete"
                    state["should_continue"] = False
                    return state
            target = target or (codebook_targets[0] if codebook_targets else None)
            result = _create_agent(DataAgent).validate_response_table(response_path, target)
            state["data_validation"] = {"passed": bool(result.get("ok")), "score": 10.0 if result.get("ok") else 0.0,
                                        "note": "; ".join(result.get("reasons") or ["response_table validated"]),
                                        "truncated": result.get("truncated"), "encoding": result.get("encoding"),
                                        "skipped_files": result.get("skipped_files"),
                                        "response_manifest": result.get("manifest")}
            if result.get("manifest"):
                state.setdefault("data_artifacts", {})["response_table"] = result["manifest"]
            if not result.get("ok"):
                reason = str((result.get("reasons") or ["needs_pack_invalid"])[0])
                state["terminal_error"] = "needs_response_table" if reason.startswith("needs_response_table") else "needs_pack_invalid"
                state["current_phase"] = "complete"
                state["should_continue"] = False
                return state
            state["current_phase"] = _next_phase(state, "data_validation", "writing_narrative")
            try:
                from core.run_log import emit_event as _emit_dv2
                _emit_dv2("phase_end", {"phase": "data_validation", "agent": "DataAgent",
                                        "status": "ok", "message_human": "response_table validated",
                                        "cause_class": "science"}, agent="DataAgent")
            except Exception:
                pass
            return state
        dataset_path = plan.get("dataset_path") or plan.get("dataset_file")
        if not dataset_path:
            logger.info("No external dataset requested; using synthetic data")
            state["data_validation"] = {
                "passed": True,
                "score": 10.0,
                "note": "No external dataset requested; experiment may use declared synthetic data",
                "checks": {"dataset_requested": False},
            }
            state["current_phase"] = _next_phase(state, "data_validation", "writing_narrative")
            return state

        spec = plan.get("dataset_spec") or {
            "name": Path(dataset_path).stem,
            "access_policy": "user-provided",
        }
        artifact = _create_agent(DataAgent).validate_dataset(dataset_path, spec)
        key = artifact.get("spec", {}).get("name") or Path(dataset_path).name
        state["data_artifacts"][key] = artifact
        state["data_validation"] = artifact.get("validation", {})
        logger.info(f"Dataset validation for '{dataset_path}': passed={state['data_validation'].get('passed')}")
        if not state["data_validation"].get("passed"):
            state["terminal_error"] = "Dataset validation failed"
            state["meta_feedback"].append(json.dumps(state["data_validation"]))
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state
        state["current_phase"] = _next_phase(state, "data_validation", "writing_narrative")
        return state
    except Exception as exc:
        logger.error(f"Data validation failed: {exc}", exc_info=True)
        state["terminal_error"] = f"Data validation failed: {exc}"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
NARRATIVE_SECTION_NAMES = {"abstract", "introduction", "related work", "methods", "method", "background", "thesis", "evidence", "case"}
RESULTS_SECTION_NAMES = {"abstract", "results", "discussion", "conclusion", "experiments", "analysis", "findings", "counterargument"}

# Paper types whose writer calls carry the §7.3 pack (paper_type + thesis).
# P4: single unified pack — all non-experimental types carry paper_type, thesis
# types additionally carry thesis. Legacy empty-dict path preserved only for
# the experimental spine to keep old test doubles green.
THESIS_PACK_TYPES = ("analytical", "descriptive", "argumentative", "review")

# Field packs (M3): survey / case_study / empirical carry the §7.3
# instrument/protocol pack via paper_type only (no thesis slot).
FIELD_PACK_TYPES = ("survey", "case_study", "empirical")

# Experiment packs (M4): comparative / cause_effect carry the §7.3
# baseline-matrix / DAG pack via paper_type (ablation + governor).
CODE_PACK_TYPES = ("comparative", "cause_effect")

# P4 unified set: every typed run carries its paper_type into the writer.
TYPED_WRITER_TYPES = frozenset(
    THESIS_PACK_TYPES + FIELD_PACK_TYPES + CODE_PACK_TYPES + ("empirical", "cause_effect", "comparative")
)


def _writer_type_kwargs(state: ResearchState, paper_type: str) -> Dict[str, Any]:
    """Single passthrough: typed runs carry paper_type (+thesis when relevant)."""
    if paper_type in THESIS_PACK_TYPES:
        return {"paper_type": paper_type, "thesis": state.get("thesis")}
    if paper_type in TYPED_WRITER_TYPES:
        return {"paper_type": paper_type}
    return {}


def _non_empirical_plan(paper_type: str, thesis: Optional[Dict[str, Any]], topic: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Synthesize the light plan for no-experiment types (§4.5, M1).

    No-experiment recipes have no planning phase, so nothing else populates
    state["plan"]. The plan carries the thesis slot, source list, bridge
    claims, an empty experiment list, and the per-type manuscript sections.
    """
    from agents.editor import MANUSCRIPT_ORDERS

    order = MANUSCRIPT_ORDERS.get(paper_type) or MANUSCRIPT_ORDERS["analytical"]
    evidence = ((topic or {}).get("literature_evidence") or []) if isinstance(topic, dict) else []
    sources = [
        paper.get("doi") or paper.get("arxiv_id") or paper.get("title")
        for paper in evidence[:40]
        if isinstance(paper, dict)
    ]
    return {
        "paper_type": paper_type,
        "plan_mode": "non_empirical",
        "methodology": f"{paper_type} synthesis without experiments",
        "experiments": [],
        "thesis_slot": ((thesis or {}).get("claim") or "") if isinstance(thesis, dict) else "",
        "sources": sources,
        "claims": [
            bridge.get("statement")
            for bridge in ((thesis or {}).get("bridges") or [])
            if isinstance(bridge, dict) and bridge.get("statement")
        ],
        "sections": [{"name": "Abstract"}] + [{"name": name} for name in order],
    }


def _ensure_thesis_and_plan(state: ResearchState, paper_type: str) -> None:
    """Build thesis (+ non-empirical plan) when the recipe skips planning.

    Covers analytical/argumentative (thesis_argument ran but a resume or a
    hand-built state may lack them) and descriptive (no thesis_argument
    phase in its recipe at all). Never clobbers an existing plan or thesis.
    """
    from agents.thesis_agent import ThesisAgent

    topic = state.get("selected_topic") or {}
    if not state.get("thesis") and topic:
        evidence = topic.get("literature_evidence") or state.get("literature_context") or []
        state["thesis"] = _create_agent(ThesisAgent).build_thesis(topic, evidence, paper_type)
        log_agent_action("Orchestrator", "thesis_ensured", {"paper_type": paper_type})
    if not state.get("plan") and state.get("thesis"):
        state["plan"] = _non_empirical_plan(paper_type, state.get("thesis"), topic)
        log_agent_action("Orchestrator", "non_empirical_plan_synthesized", {
            "paper_type": paper_type,
            "sections": len(state["plan"]["sections"]),
        })


def _next_phase(state: ResearchState, current: str, default: str) -> str:
    """Recipe-aware phase label: next recipe phase when a recipe is set.

    Legacy runs (no pipeline_recipe) keep the exact historical label, so
    flag-off behavior is byte-identical. Typed recipes advance to their
    actual next phase instead of a hard-coded legacy successor.
    """
    recipe = state.get("pipeline_recipe") or []
    if current in recipe:
        index = recipe.index(current)
        if index + 1 < len(recipe):
            return recipe[index + 1]
    return default


def _plan_section_names(plan: Dict[str, Any]) -> List[str]:
    return [section.get("name", "Unknown") if isinstance(section, dict) else str(section) for section in plan.get("sections", [])]


def _numeric_values(value: Any) -> List[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numeric_values(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numeric_values(item)]
    return []


def _seed_values(value: Any) -> List[float]:
    """Collect recorded RNG seed values (numbers under any "seeds" key)."""
    found: List[float] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "seeds" and isinstance(item, (list, tuple)):
                found.extend(float(x) for x in item if isinstance(x, (int, float)) and not isinstance(x, bool))
            else:
                found.extend(_seed_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_seed_values(item))
    return found


def _is_allowlist_year(claim: float) -> bool:
    """L14 allowlist: integer-valued calendar years 1800–2100 are grounding-exempt."""
    return float(claim).is_integer() and 1800 <= float(claim) <= 2100


def verify_result_numbers(content: str, engineer_outputs: Dict[str, Any], rtol: float = 0.05, *, allowed_counts: Optional[List[float]] = None) -> Dict[str, Any]:
    """Check manuscript numbers against recorded experiment values.

    L14: every number must be grounded, except the allowlist — calendar
    years (1800–2100), recorded RNG seeds, and explicitly allowed counts.
    ``allowed_counts`` is keyword-only so the ``(0.05,)`` positional-default
    contract asserted by the M0 gate tests is preserved.
    """
    allowed = _numeric_values(engineer_outputs) + _seed_values(engineer_outputs)
    if allowed_counts:
        allowed = allowed + [float(v) for v in allowed_counts if isinstance(v, (int, float)) and not isinstance(v, bool)]
    claims = [float(match.group(1)) / (100 if match.group(2) else 1) for match in re.finditer(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(%)?", content)]
    # L14 (pinned behavior, see test_results_redraft_carries_check_failures
    # + test_verify_result_numbers_year_allowlist): bare integers without a
    # literal % sign are exempt from grounding (covers n=3 / seed counts in
    # redrafted prose); percent values, floats, years-outside-allowlist and
    # mismatched magnitudes are still checked. Fabricated sample sizes remain
    # caught downstream by supervisor stats-recompute + cross-section checks.
    claims = [claim for claim in claims if not float(claim).is_integer() or f"{int(claim)}%" in content]
    mismatches = [
        claim for claim in claims
        if not _is_allowlist_year(claim)
        and not any(abs(claim - raw) <= max(abs(raw) * rtol, 1e-3) for raw in allowed)
    ]
    return {"passed": bool(allowed) and not mismatches, "claims": claims, "allowed_values": allowed, "mismatches": mismatches}


def section_revision_feedback(
    section_name: str,
    check_result: Dict[str, Any],
    supervisor_feedback: Optional[Dict[str, str]] = None,
    editor_findings: Optional[str] = None,
) -> Optional[str]:
    """Build the revision prompt payload for one failing section (pure helper).

    Combines the section's own numeric/empirical check failures with the
    supervisor's review feedback and, when an editor repair round is active,
    the release-referee findings. Returns None when nothing needs repair.
    """
    parts: List[str] = []
    mismatches = check_result.get("mismatches") or []
    if mismatches:
        parts.append(
            "Untraceable numeric claims — every number must be copied exactly from the "
            "experiment source of truth; rewrite or remove these: "
            + "; ".join(str(item) for item in mismatches[:12])
        )
    empirical = check_result.get("empirical_claims") or {}
    if empirical.get("prohibited_text"):
        parts.append(
            "Remove leaked harness diagnostics entirely: "
            + ", ".join(str(item) for item in empirical["prohibited_text"][:5])
        )
    elif not empirical.get("passed", True):
        parts.append(
            "Empirical claims must be grounded in completed experiment outputs; "
            "remove unsupported performance language."
        )
    prior = (supervisor_feedback or {}).get(section_name)
    if prior:
        parts.append("Prior supervisor feedback on this section: " + str(prior)[:1500])
    # Machine repair hints from hard_verify_section (check/expected/detail):
    # precise, actionable, and cheaper than re-describing raw failures.
    for hint in (check_result.get("repair_hints") or [])[:8]:
        if isinstance(hint, dict):
            parts.append(
                f"[{hint.get('check', 'gate')}] expected: {hint.get('expected', '')} "
                f"— detail: {hint.get('detail', '')}"[:400]
            )
    if editor_findings:
        parts.append("Editor release-referee findings to fix: " + str(editor_findings)[:1200])
    return "\n".join(parts) if parts else None


def editor_repair_route(error_message: str, repair_count: int, max_repairs: int = 2) -> str:
    """Decide whether an editor failure is repairable or terminal (pure helper).

    Only release-referee failures are repairable by re-drafting sections
    (proposal §5.3/R8: second repair allowed, i.e. bound 2 total);
    structural failures (e.g. failed experiments) remain terminal.
    """
    message = str(error_message or "")
    if "release referee failed" not in message:
        return "terminal"
    if int(repair_count) >= max_repairs:
        return "terminal"
    return "repair"


def write_narrative_sections(state: ResearchState) -> ResearchState:
    logger.info("Narrative writing starting")
    log_agent_action("Orchestrator", "start_writing_narrative", {})
    _record_work_plan(state, "writing_narrative", [
        "draft narrative sections from plan + literature (strong tier)",
        "checker loop repairs defects; copy-exact numbers where cited",
    ])
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("writing_narrative")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    if not state.get("plan"):
        err_msg = "Writing narrative aborted: No valid plan available"
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        writer = _create_agent(WriterAgent)
        paper_type = str(state.get("paper_type") or "experimental")
        # No-experiment recipes skip planning: ensure thesis + light plan exist
        # before narrative drafting (descriptive has no thesis_argument phase
        # at all; resumes may lack either artifact).
        if bool((state.get("evidence_requirements") or {}).get("needs_thesis")) and (
            not state.get("thesis") or not state.get("plan")
        ):
            try:
                _ensure_thesis_and_plan(state, paper_type)
            except Exception as thesis_exc:
                logger.error(f"Thesis ensure failed: {thesis_exc}", exc_info=True)
        supervisor_scores = state.get("supervisor_scores") or {}
        supervisor_feedback = state.get("supervisor_feedback") or {}
        revision_allowed = (
            state["iteration"] > 0
            and int(state.get("narrative_revision_count") or 0) < 3
            and bool(supervisor_scores)
        )
        for section_name in _plan_section_names(state["plan"]):
            if section_name.lower() not in NARRATIVE_SECTION_NAMES:
                continue
            if section_name not in state["draft_sections"]:
                logger.info(f"Drafting narrative section: {section_name}")
                state["draft_sections"][section_name] = writer.draft_section(
                    section_name, state["selected_topic"], state["plan"], {},
                    **_writer_type_kwargs(state, paper_type),
                )
                state["current_section"] = section_name
                log_agent_action("Orchestrator", "section_written", {"section": section_name})
                continue
            # Meta-continue pass: re-draft below-threshold narrative sections with
            # the supervisor's feedback injected into the prompt so the revision
            # repairs specific defects (reviewer-guided revision >> blind re-roll).
            if not revision_allowed:
                continue
            score = supervisor_scores.get(section_name)
            if score is None or score >= config.supervisor_threshold:
                continue
            revision_feedback = section_revision_feedback(
                section_name,
                {},
                supervisor_feedback=supervisor_feedback,
                editor_findings=state.get("editor_repair_findings"),
            )
            if not revision_feedback:
                continue
            logger.info(f"Revising narrative section '{section_name}' with feedback (prior score: {score})")
            state["draft_sections"][section_name] = writer.draft_section(
                section_name,
                state["selected_topic"],
                state["plan"],
                state.get("engineer_outputs") or {},
                revision_feedback,
                **_writer_type_kwargs(state, paper_type),
            )
            state["current_section"] = section_name
            log_agent_action("Orchestrator", "narrative_section_revised_with_feedback", {
                "section": section_name,
                "prior_score": score,
            })
        if revision_allowed:
            state["narrative_revision_count"] += 1
        state["current_phase"] = _next_phase(state, "writing_narrative", "engineering")
        state["error_count"] = 0
        return state
    except Exception as exc:
        logger.error(f"Writing failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Writing error: {exc}")
        state["error_count"] = int(state.get("error_count") or 0) + 1
        if state["error_count"] >= 3:
            state["should_reset"] = True
        else:
            state["current_phase"] = _next_phase(state, "writing_narrative", "engineering")
        return state


def write_results_sections(state: ResearchState) -> ResearchState:
    logger.info("Results writing starting")
    log_agent_action("Orchestrator", "start_writing_results", {})
    _record_work_plan(state, "writing_results", [
        "draft results from source-of-truth outputs only (copy-exact)",
        "verify numbers; redraft failing sections with repair hints",
    ])
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("writing_results")
    if state.get("terminal_error") or state.get("evidence_gate", {}).get("terminal") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    if not state.get("plan"):
        err_msg = "Writing results aborted: No valid plan available"
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        writer = _create_agent(WriterAgent)
        paper_type = str(state.get("paper_type") or "experimental")
        # R12 slice: a non-empirical plan (experiments: [] per §4.5) with no
        # engineer outputs skips experiment-bound checks instead of failing
        # them. An experimental plan with missing outputs still fails below.
        plan_experiments = [e for e in (state.get("plan") or {}).get("experiments", []) if isinstance(e, dict)]
        # B3: waiver keys on plan shape (waiver type / non_empirical mode),
        # not on output emptiness — stale leftover outputs must not
        # re-arm experiment-bound checks for a non-experimental plan.
        _waiver_type = paper_type in (NON_EMPIRICAL_TYPES | FIELD_WAIVER_TYPES) or (
            state.get("plan") or {}
        ).get("plan_mode") == "non_empirical"
        non_empirical_waiver = not plan_experiments and (
            not (state.get("engineer_outputs") or {}) or _waiver_type
        )
        if non_empirical_waiver:
            log_agent_action("Orchestrator", "results_checks_waived_non_empirical", {"paper_type": paper_type})
        names = _plan_section_names(state["plan"])
        # Discussion + Abstract are universal; a Results section is only
        # required when experiments exist. Thesis-pack types without
        # experiments must not gain a misleading Results section.
        required_sections = ["Discussion", "Abstract"]
        if plan_experiments or paper_type not in THESIS_PACK_TYPES:
            required_sections.insert(0, "Results")
        for required in required_sections:
            if not any(name.lower() == required.lower() for name in names):
                names.append(required)
        for section_name in names:
            if section_name.lower() in RESULTS_SECTION_NAMES:
                logger.info(f"Drafting results section: {section_name}")
                state["draft_sections"][section_name] = writer.draft_section(section_name, state["selected_topic"], state["plan"], state["engineer_outputs"], **_writer_type_kwargs(state, paper_type))
                state["current_section"] = section_name
        checked = {}
        for name in names:
            if name.lower() not in RESULTS_SECTION_NAMES or name not in state["draft_sections"]:
                continue
            section = state["draft_sections"][name]
            if non_empirical_waiver:
                checked[name] = {
                    "passed": True,
                    "waived": True,
                    "reason": "non_empirical_no_outputs",
                    "claims": [],
                    "allowed_values": [],
                    "mismatches": [],
                    "empirical_claims": {"passed": True, "prohibited_text": [], "waived": True},
                }
                continue
            numeric_check = verify_result_numbers(section, state["engineer_outputs"])
            empirical_check = validate_empirical_claims(section, state["engineer_outputs"])
            checked[name] = {
                **numeric_check,
                "empirical_claims": empirical_check,
                "passed": numeric_check["passed"] and empirical_check["passed"],
            }
            tracker = get_tracker()
            if tracker:
                context = get_active_context()
                ledger = context.research_db if context and context.research_db else research_db
                ledger.record_claim(
                    tracker.run_id,
                    name,
                    section[:1000],
                    "empirical_section",
                    "verified" if checked[name]["passed"] else "rejected",
                    {
                        "numeric": numeric_check,
                        "empirical": empirical_check,
                        "contract_hashes": {
                            exp_name: output.get("contract_hash")
                            for exp_name, output in state["engineer_outputs"].items()
                            if isinstance(output, dict)
                        },
                    },
                )
        failures = {name: result for name, result in checked.items() if not result["passed"]}
        logger.info(f"Numeric verification: {len(checked)} section(s) checked, {len(failures)} failure(s)")
        state["results_verification"] = checked
        hard_failures = {
            name: result for name, result in failures.items()
            if result.get("empirical_claims", {}).get("prohibited_text")
        }
        if hard_failures:
            message = "Manuscript contains leaked execution diagnostics"
            state["terminal_error"] = message
            state["evidence_gate"] = {
                "allowed": False,
                "terminal": True,
                "reason_code": "prohibited_manuscript_text",
                "message": message,
                "sections": list(hard_failures),
            }
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state
        if failures and state["results_redraft_count"] < 2:
            logger.warning(f"Results redraft needed for section(s): {list(failures.keys())}")
            state["results_redraft_count"] += 1
            # Targeted artifact repair: re-draft only the failing sections and
            # feed their specific check failures back into the writer prompt.
            state["meta_feedback"].append(f"Results numeric grounding failed: {failures}")
            for name, result in failures.items():
                feedback = section_revision_feedback(
                    name,
                    result,
                    supervisor_feedback=state.get("supervisor_feedback") or {},
                    editor_findings=state.get("editor_repair_findings"),
                )
                state["draft_sections"][name] = writer.draft_section(
                    name,
                    state["selected_topic"],
                    state["plan"],
                    state["engineer_outputs"],
                    feedback,
                    **_writer_type_kwargs(state, paper_type),
                )
                log_agent_action("Orchestrator", "results_section_revised_with_feedback", {
                    "section": name,
                    "had_feedback": bool(feedback),
                })
            state["current_phase"] = "writing_results"
            log_agent_action("Orchestrator", "results_numeric_grounding_failed", {"sections": list(failures)})
            return state
        state["current_phase"] = "supervision"
        return state
    except Exception as exc:
        logger.error(f"Results writing failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Results writing error: {exc}")
        state["current_phase"] = "supervision"
        return state


def _engineering_wall_clock_guard(
    state: ResearchState, eng_deadline: float, label: str
) -> Optional[ResearchState]:
    """Terminal budget_exhausted state when the engineering phase cap is spent.

    Returns the terminal state, or None when budget remains. Covers every
    point that would otherwise start new work (branching search, each
    experiment) so a spent cap never launches another LLM/sandbox batch.
    """
    if time.monotonic() < eng_deadline:
        return None
    state["terminal_error"] = (
        "budget_exhausted: engineering wall clock exceeded "
        f"({getattr(config, 'engineering_wall_clock_sec', 1800)}s cap): "
        f"{label} not started"
    )
    state["evidence_gate"] = {
        "allowed": False,
        "terminal": True,
        "reason_code": "budget_exhausted",
        "message": state["terminal_error"],
    }
    state["current_phase"] = "complete"
    state["should_continue"] = False
    log_agent_action("Orchestrator", "engineering_wall_clock_exceeded", {
        "label": label,
    })
    return state


def engineering_node(state: ResearchState) -> ResearchState:
    logger.info("Engineering starting")
    log_agent_action("Orchestrator", "start_engineering", {})
    _record_work_plan(state, "engineering", [
        "generate code cheap, escalate to strong after cascade threshold",
        "refine per failure category; record trial outcomes to bank",
    ])
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("engineering")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    if not state.get("plan"):
        err_msg = "Engineering aborted: No valid plan available"
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    # R12-style waiver (B2): non-experimental plans carry experiments: [] and
    # produce no code. Skip engineering instead of failing the contract gate;
    # the superset router advances to the next recipe phase. Never terminal.
    _eng_type = str(state.get("paper_type") or "experimental")
    _eng_experiments = (state.get("plan") or {}).get("experiments", [])
    if (
        isinstance(_eng_experiments, list)
        and not _eng_experiments
        and (
            _eng_type in (NON_EMPIRICAL_TYPES | FIELD_WAIVER_TYPES)
            or (state.get("plan") or {}).get("plan_mode") == "non_empirical"
        )
    ):
        reason = (
            "engineering skipped for non-experimental type "
            f"'{_eng_type}': plan carries experiments: [], nothing to execute"
        )
        logger.info(reason)
        log_agent_action("Orchestrator", "skip_engineering", {
            "paper_type": _eng_type,
            "cause_class": "science",
            "reason": reason,
        })
        state.setdefault("analysis_reports", {})["engineering"] = {
            "skipped": True,
            "reason": reason,
            "paper_type": _eng_type,
        }
        state["current_phase"] = _next_phase(state, "engineering", "writing_results")
        state["error_count"] = 0
        return state
    try:
        experiments = state["plan"].get("experiments", [])
        _eng_type = str(state.get("paper_type") or "experimental")
        if (
            isinstance(experiments, list)
            and experiments
            and all(isinstance(item, dict) for item in experiments)
        ):
            # Defensive contract alignment: LLM aliases like experiment_name → name
            normalizer = getattr(PlannerAgent, "_normalize_experiments", None)
            if callable(normalizer):
                experiments = normalizer(experiments)
            state["plan"]["experiments"] = experiments
            plan_errors = validate_experiments(experiments, paper_type=_eng_type, plan=state.get("plan"))
            plan_errors.extend(validate_dataset_identity(experiments, state.get("data_artifacts")))
            if plan_errors:
                logger.warning(f"Engineering found plan schema errors: {'; '.join(plan_errors)}")
                message = "; ".join(plan_errors)
                attempts = int((state.get("plan") or {}).get("schema_revision_attempts") or 0)
                if attempts >= 2:
                    state["terminal_error"] = message
                    state["evidence_gate"] = {
                        "allowed": False,
                        "terminal": True,
                        "reason_code": "invalid_experiment_plan",
                        "message": message,
                    }
                    state["current_phase"] = "complete"
                    state["should_continue"] = False
                    log_agent_action("Orchestrator", "engineering_blocked_invalid_plan", {
                        "message": message,
                        "schema_revision_attempts": attempts,
                    })
                    return state
                plan = dict(state["plan"] or {})
                plan["schema_revision_attempts"] = attempts + 1
                state["plan"] = plan
                revision = {
                    "reason": "invalid_experiment_plan",
                    "experiment": None,
                    "detail": message,
                    "timestamp": datetime.now().isoformat(),
                }
                state.setdefault("plan_revision_requests", []).append(revision)
                CrossRunMemory().record_plan_revision("invalid_experiment_plan", meta=revision)
                state["meta_feedback"].append(f"Plan revision requested for schema errors: {message}")
                state["current_phase"] = "planning"
                state["should_continue"] = True
                state["terminal_error"] = None
                log_agent_action("Orchestrator", "engineering_blocked_invalid_plan", {
                    "message": message,
                    "action": "request_plan_revision",
                    "schema_revision_attempts": plan["schema_revision_attempts"],
                })
                return state
        non_dict = [i for i, e in enumerate(experiments) if not isinstance(e, dict)]
        if non_dict:
            message = f"Malformed experiments at indices: {non_dict}"
            logger.warning(message)
            attempts = int((state.get("plan") or {}).get("schema_revision_attempts") or 0)
            if attempts >= 2:
                state["terminal_error"] = message
                state["evidence_gate"] = {
                    "allowed": False,
                    "terminal": True,
                    "reason_code": "invalid_experiment_plan",
                    "message": message,
                }
                state["current_phase"] = "complete"
                state["should_continue"] = False
                log_agent_action("Orchestrator", "engineering_blocked_invalid_plan", {
                    "message": message,
                    "schema_revision_attempts": attempts,
                })
                return state
            plan = dict(state["plan"] or {})
            plan["schema_revision_attempts"] = attempts + 1
            state["plan"] = plan
            revision = {
                "reason": "invalid_experiment_plan",
                "experiment": None,
                "detail": message,
                "timestamp": datetime.now().isoformat(),
            }
            state.setdefault("plan_revision_requests", []).append(revision)
            CrossRunMemory().record_plan_revision("invalid_experiment_plan", meta=revision)
            state["meta_feedback"].append(f"Plan revision requested for malformed experiments: {message}")
            state["current_phase"] = "planning"
            state["should_continue"] = True
            state["terminal_error"] = None
            log_agent_action("Orchestrator", "engineering_blocked_invalid_plan", {
                "message": message,
                "action": "request_plan_revision",
                "schema_revision_attempts": plan["schema_revision_attempts"],
            })
            return state
        for experiment in experiments:
            name = experiment["name"]
            logger.info(f"Processing experiment: {name}")
            contract = build_contract(experiment)
            prior_contract = state["experiment_contracts"].get(name)
            if prior_contract and prior_contract.get("contract_hash") != contract.get("contract_hash"):
                message = f"Committed experiment contract changed: {name}"
                state["terminal_error"] = message
                state["evidence_gate"] = {
                    "allowed": False,
                    "terminal": True,
                    "reason_code": "experiment_contract_drift",
                    "message": message,
                }
                state["current_phase"] = "complete"
                state["should_continue"] = False
                log_agent_action("Orchestrator", "engineering_blocked_contract_drift", {"experiment": name})
                return state
            state["experiment_contracts"][name] = prior_contract or contract
        engineer = _create_agent(EngineerAgent)
        method_text = "\n".join(state["draft_sections"].get(section, "") for section in ("Methods", "Method", "Experiments"))
        # M4: total engineering wall-time cap (§11-M4). The per-seed sandbox
        # timeouts bound single runs; this bounds their sum.
        eng_deadline = time.monotonic() + float(
            getattr(config, "engineering_wall_clock_sec", 1800) or 1800
        )
        wall_clock_state = _engineering_wall_clock_guard(state, eng_deadline, "branching_search")
        if wall_clock_state is not None:
            return wall_clock_state
        branch_winner_name = None
        branching_search = getattr(engineer, "run_branching_search", None)
        if (
            callable(branching_search)
            and experiments
            and any(isinstance(item, dict) and (item.get("variants") or item.get("alternatives")) for item in experiments)
        ):
            logger.info("Running branching search for variant experiments")
            try:
                if tracker:
                    tracker.message("Engineering: branching cheap probes…")
                # M4 (R13): branch candidates inherit the run's paper type
                # (copies — the committed plan experiments are never mutated).
                branched = branching_search(
                    [{**item, "paper_type": item.get("paper_type", _eng_type)}
                     if isinstance(item, dict) else item
                     for item in experiments if isinstance(item, dict)],
                    method_description=method_text)
                branch_winner_name = branched.get("experiment_name") or branched.get("approach") or "branch_winner"
                if branch_winner_name in state["experiment_contracts"]:
                    branched["contract_hash"] = state["experiment_contracts"][branch_winner_name]["contract_hash"]
                state["engineer_outputs"][branch_winner_name] = branched
                winner = (branched.get("branch_search") or {}).get("winner")
                if winner and winner not in state["engineer_outputs"]:
                    state["engineer_outputs"][winner] = branched
            except Exception as exc:
                logger.error(f"Branching search failed: {exc}", exc_info=True)
                if tracker:
                    tracker.message(f"Branching search failed: {exc}", level="error")
        for experiment in experiments:
            exp_name = experiment.get("name", "unknown_experiment") if isinstance(experiment, dict) else str(experiment)
            exp_config = dict(experiment) if isinstance(experiment, dict) else {"name": exp_name}
            if isinstance(experiment, dict) and exp_name in state["experiment_contracts"]:
                exp_config["contract_hash"] = state["experiment_contracts"][exp_name]["contract_hash"]
            # M4 (R13): the experiment carries its paper type so the
            # Engineer applies the per-type code policy (baseline matrix /
            # DAG brief, early pivot, compulsory ablation). Contract-excluded
            # key: build_contract never reads it, so provenance is stable.
            exp_config.setdefault("paper_type", _eng_type)
            if exp_name in state["engineer_outputs"] or (branch_winner_name and exp_name == branch_winner_name):
                continue
            # M4: total engineering wall-time cap — stop starting new
            # experiments once the phase budget is spent. Exhaustion keeps
            # the locked budget_exhausted taxonomy (§1.5/§5.3); the partial
            # bundle is retained in state.
            wall_clock_state = _engineering_wall_clock_guard(state, eng_deadline, exp_name)
            if wall_clock_state is not None:
                return wall_clock_state
            try:
                if tracker:
                    tracker.message(f"Engineering: running {exp_name}")
                alternatives = list(exp_config.get("variants") or exp_config.get("alternatives") or [])
                runner = engineer.run_experiment
                try:
                    state["engineer_outputs"][exp_name] = runner(
                        exp_config, alternatives=alternatives, method_description=method_text
                    )
                except TypeError as exc:
                    # Compatibility with the original narrow EngineerAgent
                    # interface used by demos and lightweight integrations.
                    if "unexpected keyword argument" not in str(exc):
                        raise
                    state["engineer_outputs"][exp_name] = runner(exp_config)
                output = state["engineer_outputs"][exp_name]
                # L2: a pivoted run re-stamped its contract mid-flight —
                # commit the new contract instead of overwriting it with
                # the stale one.
                if isinstance(output, dict) and output.get("contract_restamped") and output.get("contract"):
                    state["experiment_contracts"][exp_name] = output["contract"]
                    log_agent_action("Orchestrator", "experiment_contract_restamped", {
                        "experiment": exp_name,
                        "contract_hash": output["contract"].get("contract_hash"),
                    })
                else:
                    output["contract_hash"] = state["experiment_contracts"][exp_name]["contract_hash"]
                log_agent_action("Orchestrator", "experiment_run", {"experiment": exp_name})
                # Experience-bank trial (unfreezes wins/losses): failures teach,
                # successes-after-refine promote the recovery.
                try:
                    from core.experience_bank import record_trial as _record_trial

                    _ok = bool(isinstance(output, dict) and output.get("success"))
                    if _ok:
                        decision_log = output.get("decision_log") or []
                        if any(
                            isinstance(decision, dict) and decision.get("decision") == "REFINE"
                            for decision in decision_log
                        ):
                            _record_trial("engineering_retry",
                                          f"{exp_name}: retry succeeded", exp_name, True)
                    else:
                        _record_trial(
                            str((output or {}).get("failure_kind") or "technical"),
                            str((output or {}).get("error") or "experiment failed")[:800],
                            exp_name, False)
                except Exception:
                    logger.exception("Failed to record engineering experience trial for %s", exp_name)
            except Exception as exc:
                logger.error(f"Experiment {exp_name} failed: {exc}", exc_info=True)
                try:
                    from core.experience_bank import record_trial as _record_trial2

                    _record_trial2("technical", str(exc)[:800], exp_name, False)
                except Exception:
                    logger.exception("Failed to record engineering failure trial for %s", exp_name)
                state["engineer_outputs"][exp_name] = {
                    "success": False,
                    "error": str(exc),
                    "experiment_name": exp_name,
                    "contract_hash": state["experiment_contracts"][exp_name]["contract_hash"],
                    "failure_kind": "technical",
                }
        gate = gate_engineering_outputs(
            state.get("plan"),
            state.get("engineer_outputs"),
            state.get("experiment_contracts"),
        )
        if not gate.get("allowed") and _synthetic_fallback_allowed(state):
            recovered = _run_synthetic_fallback_outputs(state)
            if recovered:
                state["engineer_outputs"].update(recovered)
                gate = gate_engineering_outputs(
                    state.get("plan"),
                    state.get("engineer_outputs"),
                    state.get("experiment_contracts"),
                )
                state["meta_feedback"].append(
                    "Engineering used a deterministic bundled-synthetic fallback; results are exploratory."
                )
                log_agent_action("Orchestrator", "engineering_synthetic_fallback", {
                    "experiments": list(recovered),
                    "reason": "local_experiment_execution_failed",
                })
        state["evidence_gate"] = gate
        logger.info(f"Engineering gate result: allowed={gate.get('allowed')}, reason={gate.get('reason_code')}")
        request_consumer = getattr(engineer, "consume_plan_revision_requests", None)
        requests = request_consumer() if callable(request_consumer) else []
        if not gate.get("allowed"):
            state["technical_failures"] = {
                name: output
                for name, output in state.get("engineer_outputs", {}).items()
                if isinstance(output, dict) and not output.get("success")
            }
            revision_attempts = int((state.get("plan") or {}).get("engineer_revision_attempts") or 0)
            if requests and revision_attempts < 2:
                plan = dict(state.get("plan") or {})
                plan["engineer_revision_attempts"] = revision_attempts + 1
                state["plan"] = plan
                state.setdefault("plan_revision_requests", []).extend(requests)
                # Revised plan gets a fresh contract identity; clear prior commit.
                state["engineer_outputs"] = {}
                state["experiment_contracts"] = {}
                state["terminal_error"] = None
                state["evidence_gate"] = {
                    "allowed": False,
                    "terminal": False,
                    "reason_code": "plan_revision_requested",
                    "message": "Engineering exhausted attempts; requesting Planner revision before terminal failure",
                }
                state["meta_feedback"].append(
                    "Plan revision requested after engineering failures: "
                    + json.dumps(requests, default=str)
                )
                state["current_phase"] = "planning"
                state["should_continue"] = True
                log_agent_action("Orchestrator", "engineering_request_plan_revision", {
                    "gate": gate,
                    "requests": requests,
                    "engineer_revision_attempts": plan["engineer_revision_attempts"],
                })
                return state
            state["terminal_error"] = gate.get("message") or gate.get("reason_code")
            state["current_phase"] = "complete"
            state["should_continue"] = False
            if requests:
                state["meta_feedback"].append(
                    "Plan revision requests discarded at terminal engineering failure: "
                    + json.dumps(requests, default=str)
                )
            log_agent_action("Orchestrator", "engineering_terminal_failure", gate)
            return state
        if requests:
            state["meta_feedback"].append(
                "Plan revision requested after a successful run; contract remains immutable: "
                + json.dumps(requests, default=str)
            )
        state["current_phase"] = _next_phase(state, "engineering", "writing_results")
        state["error_count"] = 0
        return state
    except Exception as exc:
        logger.error(f"Engineering failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Engineering error: {exc}")
        state["error_count"] = int(state.get("error_count") or 0) + 1
        if state["error_count"] >= 3:
            state["should_reset"] = True
        else:
            state["current_phase"] = _next_phase(state, "engineering", "writing_results")
        return state


def independent_validation_node(state: ResearchState) -> ResearchState:
    """Replay engineer code, analyze outputs, and record independent findings."""
    logger.info("Independent validation starting")
    log_agent_action("Orchestrator", "start_independent_validation", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("independent_validation")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    if not state.get("plan"):
        err_msg = "Independent validation aborted: No valid plan available"
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    # R12 (M6): non-experimental recipes produce no engineer outputs, so
    # there is nothing to replay. Skip execution replay; verify
    # proofs/citations instead (advisory, never terminal). Experimental
    # types (experimental/comparative/cause_effect) fall through to the
    # legacy no-bypass path unchanged.
    engineer_outputs = state.get("engineer_outputs") or {}
    paper_type = str(state.get("paper_type") or "experimental")
    if not engineer_outputs and paper_type in (NON_EMPIRICAL_TYPES | FIELD_WAIVER_TYPES):
        reason = (
            "independent validation skipped for non-experimental type "
            f"'{paper_type}': engineer_outputs is empty, nothing to replay"
        )
        logger.info(reason)
        log_agent_action("Orchestrator", "skip_independent_validation", {
            "paper_type": paper_type,
            "cause_class": "science",
            "reason": reason,
        })
        citation_notes: List[Any] = []
        draft_text = "\n".join(
            value for value in (state.get("draft_sections") or {}).values()
            if isinstance(value, str)
        )
        if draft_text.strip():
            try:
                citation_check = verify_citations(draft_text)
                if not citation_check.get("passed"):
                    citation_notes = list(citation_check.get("failed", []))
            except Exception as exc:
                logger.warning("R12 skip-path citation check failed: %s", exc, exc_info=True)
        state.setdefault("analysis_reports", {})["independent"] = {
            "skipped": True,
            "reason": reason,
            "paper_type": paper_type,
            "citation_notes": citation_notes,
        }
        findings = state.get("verification_findings")
        if findings is None:
            findings = []
            state["verification_findings"] = findings
        findings.append({
            "finding_id": "independent_validation:skipped_non_experimental",
            "severity": "info",
            "check": "independent_validation",
            "message": reason,
            "blocking": False,
            "status": "skipped",
        })
        state["current_phase"] = _next_phase(state, "independent_validation", "writing_results")
        return state
    try:
        code_artifacts = {
            name: {
                "experiment_name": name,
                "source_code": output.get("code", ""),
                "execution_backend": output.get("backend", ""),
                "language": "python",
            }
            for name, output in state.get("engineer_outputs", {}).items()
            if output.get("success") and output.get("code")
        }
        logger.info(f"Validating {len(code_artifacts)} code artifact(s): {list(code_artifacts.keys())}")
        if not code_artifacts:
            message = "Independent validation failed: no executable code artifacts"
            state["terminal_error"] = state.get("terminal_error") or message
            state["evidence_gate"] = {
                "allowed": False,
                "terminal": True,
                "reason_code": "missing_code_artifact",
                "message": message,
            }
            state["meta_feedback"].append(message)
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state

        execution_agent = _create_agent(ExecutionAgent)
        for name, code_artifact in code_artifacts.items():
            experiment = next(
                (item for item in (state.get("plan") or {}).get("experiments", [])
                 if isinstance(item, dict) and item.get("name") == name),
                {},
            )
            seeds = [42 + index * 1009 for index in range(config.experiment_seeds)]
            state["execution_artifacts"][name] = execution_agent.execute(
                code_artifact,
                {
                    "experiment_name": name,
                    "seeds": seeds,
                    "resource_limits": {"timeout_seconds": config.sandbox_timeout_sec},
                },
            )

        analysis_plan = (state.get("plan") or {}).get("analysis_plan") or {
            "primary_metric": ((state.get("plan") or {}).get("experiments") or [{}])[0].get(
                "evaluation_metrics", ["accuracy"]
            )[0],
            "statistical_test": "Welch t-test",
            "confidence_level": 0.95,
        }
        report = _create_agent(AnalysisAgent).analyze(state["execution_artifacts"], analysis_plan)
        state["analysis_reports"]["independent"] = report
        reports_by_experiment = {
            name: {"metrics": report.get("metrics", {}).get(name, {})}
            for name in state["execution_artifacts"]
        }
        state["verification_findings"] = _create_agent(VerificationAgent).verify(
            state["execution_artifacts"],
            reports_by_experiment,
        )
        blocking_count = sum(1 for f in state["verification_findings"] if f.get("blocking"))
        logger.info(f"Verification complete: {len(state['verification_findings'])} finding(s), {blocking_count} blocking")
        if any(finding.get("blocking") for finding in state["verification_findings"]):
            message = "Independent validation produced blocking findings"
            state["terminal_error"] = state.get("terminal_error") or message
            state["evidence_gate"] = {
                "allowed": False,
                "terminal": True,
                "reason_code": "blocking_verification_finding",
                "message": message,
            }
            state["meta_feedback"].append(message)
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state
        state["current_phase"] = _next_phase(state, "independent_validation", "writing_results")
        return state
    except Exception as exc:
        logger.error(f"Independent validation failed: {exc}", exc_info=True)
        state["verification_findings"].append({
            "finding_id": "independent_validation:error",
            "severity": "error",
            "check": "independent_validation",
            "message": str(exc),
            "blocking": True,
            "status": "failed",
        })
        state["meta_feedback"].append(f"Independent validation error: {exc}")
        state["terminal_error"] = state.get("terminal_error") or f"Independent validation failed: {exc}"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
def supervision_node(state: ResearchState) -> ResearchState:
    logger.info("Supervision starting")
    log_agent_action("Orchestrator", "start_supervision", {})
    _record_work_plan(state, "supervision", [
        "hard verify every section (citations/stats/substance) first",
        "soft judge review; record section verdicts to experience bank",
    ])
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("supervision")
    try:
        supervisor = _create_agent(SupervisorAgent)
        paper_type = str(state.get("paper_type") or "experimental")
        section_requirements = {
            str(section.get("name") or "").lower(): section.get("content_requirements")
            for section in ((state.get("plan") or {}).get("sections") or [])
            if isinstance(section, dict)
        }
        for section_name, content in state["draft_sections"].items():
            score, feedback = supervisor.evaluate_section(
                section_name,
                content,
                engineer_outputs=state.get("engineer_outputs"),
                content_requirements=section_requirements.get(section_name.lower()),
                paper_type=paper_type,
                thesis=state.get("thesis"),
                instrument=state.get("instrument"),
                survey_analysis=(state.get("analysis_reports") or {}).get("survey"),
                case_protocol=(state.get("analysis_reports") or {}).get("case_protocol"),
                causal_claim=state.get("causal_claim"),
            )
            logger.info(f"Section '{section_name}' score: {score}")
            state["supervisor_scores"][section_name] = score
            state["supervisor_feedback"][section_name] = feedback
            if score < config.supervisor_threshold and get_tracker():
                get_tracker().bump("sections_bounced")
            # Experience-bank trial: section verdicts train future revisions.
            try:
                from core.experience_bank import record_trial as _sup_trial

                _passed = float(score or 0.0) >= float(config.supervisor_threshold)
                _sup_trial("supervision_section",
                           str(feedback or "")[:800] if not _passed else f"{section_name}: passed {score}",
                           f"{paper_type}/{section_name}", _passed)
            except Exception:
                pass
        state["reproducibility"] = reproducibility_dossier(
            state.get("plan"), state.get("engineer_outputs"), paper_type=paper_type,
            thesis=state.get("thesis"),
            survey_analysis=(state.get("analysis_reports") or {}).get("survey"),
            case_protocol=(state.get("analysis_reports") or {}).get("case_protocol"),
            instrument=state.get("instrument"),
        )
        state["outcome_calibration"] = research_db.outcome_calibration()
        if not state["reproducibility"]["passed"]:
            state["meta_feedback"].append("Reproducibility dossier incomplete: " + json.dumps(state["reproducibility"]["checks"]))
        # M2 (§7.1): review manuscript gate — PRISMA + extraction + recall +
        # synthesis spot-grounding. A failing verify_review blocks release to
        # meta_evaluation (recall demotion itself is a banner, not a block).
        if paper_type == "review" and state.get("corpus"):
            try:
                from core.verification import verify_review as _verify_review

                review_gate = _verify_review(state.get("corpus"), state.get("draft_sections"))
                state["verification_findings"] = list(state.get("verification_findings") or []) + [
                    {"check": f"review_{f.get('check', 'gate')}", "blocking": True,
                     "details": f.get("details", "")} for f in review_gate.get("findings", [])]
                try:
                    from core.run_log import emit_event as _emit_review

                    for row in (("prisma_reconciled", review_gate.get("prisma", {}).get("passed", False)),
                                ("extraction_completeness", (review_gate.get("extraction", {}) or {}).get("completeness")),
                                ("seed_recall", review_gate.get("recall")),
                                ("synthesis_grounded", review_gate.get("synthesis_ok", False))):
                        _emit_review("verification_item", {
                            "phase": "supervision", "agent": "SupervisorAgent",
                            "status": "ok" if row[1] else "fail",
                            "message_human": f"review {row[0]}={row[1]} mode={review_gate.get('mode')}",
                            "check": row[0], "actual": row[1], "mode": review_gate.get("mode"),
                            "cause_class": "science"},
                            agent="SupervisorAgent")
                except Exception:
                    pass
                if review_gate.get("banner"):
                    state.setdefault("meta_feedback", []).append(str(review_gate["banner"]))
                    log_agent_action("Orchestrator", "review_recall_banner", {
                        "banner": review_gate["banner"], "mode": review_gate.get("mode")})
                if not review_gate["passed"]:
                    state["current_phase"] = "meta_evaluation"
                    log_agent_action("Orchestrator", "release_blocked_by_review_gate", {
                        "findings": review_gate.get("findings", [])})
                    return state
            except Exception as review_exc:
                logger.error(f"Review gate failed: {review_exc}", exc_info=True)
        # M3 (§7.1): survey manuscript gate — instrument + nonresponse +
        # no-fishing + synthetic screen. A failing verify_survey blocks
        # release to meta_evaluation (synthetic trips are blocking items).
        if paper_type == "survey" and (state.get("instrument") or (state.get("analysis_reports") or {}).get("survey")):
            try:
                from core.verification import verify_survey as _verify_survey

                survey_gate = _verify_survey((state.get("analysis_reports") or {}).get("survey"),
                                             state.get("draft_sections"), state.get("instrument"))
                state["verification_findings"] = list(state.get("verification_findings") or []) + [
                    {"check": f"survey_{f.get('check', 'gate')}", "blocking": bool(f.get("blocking", True)),
                     "details": f.get("details", "")} for f in survey_gate.get("findings", [])]
                try:
                    from core.run_log import emit_event as _emit_survey

                    for row in (("instrument", survey_gate.get("instrument_ok", False)),
                                ("nonresponse", survey_gate.get("nonresponse_ok", False)),
                                ("synthetic", survey_gate.get("synthetic_ok", False)),
                                ("fishing", survey_gate.get("fishing_ok", False))):
                        _emit_survey("verification_item", {
                            "phase": "supervision", "agent": "SupervisorAgent",
                            "status": "ok" if row[1] else "fail",
                            "message_human": f"survey {row[0]}={row[1]}",
                            "check": row[0], "actual": row[1],
                            "cause_class": "science"},
                            agent="SupervisorAgent")
                except Exception:
                    pass
                if not survey_gate["passed"]:
                    state["current_phase"] = "meta_evaluation"
                    log_agent_action("Orchestrator", "release_blocked_by_survey_gate", {
                        "findings": survey_gate.get("findings", [])})
                    return state
            except Exception as survey_exc:
                logger.error(f"Survey gate failed: {survey_exc}", exc_info=True)
        # M3 (§7.1/§4.9): case manuscript gate — triangulation + rival.
        # Single-source flags are warnings with limitation entries (run
        # continues); pack/rival failures block release to meta_evaluation.
        if paper_type == "case_study" and (state.get("analysis_reports") or {}).get("case_protocol"):
            try:
                from core.verification import verify_case as _verify_case

                case_gate = _verify_case((state.get("analysis_reports") or {}).get("case_protocol"),
                                         state.get("draft_sections"))
                state["verification_findings"] = list(state.get("verification_findings") or []) + [
                    {"check": f"case_{f.get('check', 'gate')}", "blocking": bool(f.get("blocking", True)),
                     "details": f.get("details", "")} for f in case_gate.get("findings", [])]
                try:
                    from core.run_log import emit_event as _emit_case

                    for row in (("pack", case_gate.get("pack_ok", False)),
                                ("triangulation", case_gate.get("triangulation_ok", False)),
                                ("rival", case_gate.get("rival_ok", False)),
                                ("single_source_flags", case_gate.get("n_flagged", 0))):
                        _emit_case("verification_item", {
                            "phase": "supervision", "agent": "SupervisorAgent",
                            "status": "ok" if row[1] or row[0] == "single_source_flags" else "fail",
                            "message_human": f"case {row[0]}={row[1]}",
                            "check": row[0], "actual": row[1],
                            "cause_class": "science"},
                            agent="SupervisorAgent")
                except Exception:
                    pass
                if not case_gate["passed"]:
                    state["current_phase"] = "meta_evaluation"
                    log_agent_action("Orchestrator", "release_blocked_by_case_gate", {
                        "findings": case_gate.get("findings", [])})
                    return state
            except Exception as case_exc:
                logger.error(f"Case gate failed: {case_exc}", exc_info=True)
        # M4 (§7.1/§4.7): cause-effect manuscript gate — audit pass +
        # assumption/sensitivity section + verb governor. A failing
        # verify_causal blocks release to meta_evaluation (placebo
        # infeasibility itself is a limitation entry, never a block).
        if paper_type == "cause_effect" and state.get("causal_claim"):
            try:
                from core.verification import verify_causal as _verify_causal

                indep_reports = state.get("analysis_reports") or {}
                indep_ok = bool(indep_reports.get("independent"))
                causal_gate = _verify_causal(state.get("causal_claim"),
                                             state.get("draft_sections"), indep_pass=indep_ok)
                state["verification_findings"] = list(state.get("verification_findings") or []) + [
                    {"check": f"causal_{f.get('check', 'gate')}", "blocking": bool(f.get("blocking", True)),
                     "details": f.get("details", "")} for f in causal_gate.get("findings", [])]
                try:
                    from core.run_log import emit_event as _emit_causal

                    for row in (("identification", causal_gate.get("strategy")),
                                ("verb_level", causal_gate.get("verb_level")),
                                ("sensitivity", causal_gate.get("sensitivity_ok")),
                                ("governor", causal_gate.get("governor_ok"))):
                        _emit_causal("verification_item", {
                            "phase": "supervision", "agent": "SupervisorAgent",
                            "status": "ok" if row[1] else "fail",
                            "message_human": f"causal {row[0]}={row[1]}",
                            "check": row[0], "actual": row[1],
                            "cause_class": "science"},
                            agent="SupervisorAgent")
                except Exception:
                    pass
                if not causal_gate["passed"]:
                    state["current_phase"] = "meta_evaluation"
                    log_agent_action("Orchestrator", "release_blocked_by_causal_gate", {
                        "findings": causal_gate.get("findings", [])})
                    return state
            except Exception as causal_exc:
                logger.error(f"Causal gate failed: {causal_exc}", exc_info=True)
        # M4 (§7.1): comparative manuscript gate — baseline matrix +
        # compulsory ablation. A failing verify_comparative blocks release.
        if paper_type == "comparative" and state.get("plan"):
            try:
                from core.verification import verify_comparative as _verify_comparative

                comparative_gate = _verify_comparative(state.get("plan"), state.get("engineer_outputs"))
                state["verification_findings"] = list(state.get("verification_findings") or []) + [
                    {"check": f"comparative_{f.get('check', 'gate')}", "blocking": bool(f.get("blocking", True)),
                     "details": f.get("details", "")} for f in comparative_gate.get("findings", [])]
                try:
                    from core.run_log import emit_event as _emit_comparative

                    for row in (("baseline_matrix", comparative_gate.get("baseline_ok")),
                                ("ablation", comparative_gate.get("ablation_ok"))):
                        _emit_comparative("verification_item", {
                            "phase": "supervision", "agent": "SupervisorAgent",
                            "status": "ok" if row[1] else "fail",
                            "message_human": f"comparative {row[0]}={row[1]}",
                            "check": row[0], "actual": row[1],
                            "cause_class": "science"},
                            agent="SupervisorAgent")
                except Exception:
                    pass
                if not comparative_gate["passed"]:
                    state["current_phase"] = "meta_evaluation"
                    log_agent_action("Orchestrator", "release_blocked_by_comparative_gate", {
                        "findings": comparative_gate.get("findings", [])})
                    return state
            except Exception as comparative_exc:
                logger.error(f"Comparative gate failed: {comparative_exc}", exc_info=True)
        has_blocking_findings = any(
            finding.get("blocking") for finding in state.get("verification_findings", [])
        )
        if has_blocking_findings:
            state["current_phase"] = "meta_evaluation"
            log_agent_action("Orchestrator", "release_blocked_by_verification", {
                "findings": len(state.get("verification_findings", [])),
            })
        elif state["supervisor_scores"]:
            # v5 (f537f638: mean 2.19 hid five 0.0 sections behind one 9.14):
            # all types gate on the weakest section (min) — one failing
            # section must never hide behind strong ones.
            overall_score = min(state["supervisor_scores"].values())
            aggregation = "min"
            logger.info(f"Overall supervision score: {overall_score:.2f} (threshold: {config.supervisor_threshold}, aggregation: {aggregation}, paper_type: {paper_type})")
            state["current_phase"] = "editing" if overall_score >= config.supervisor_threshold else "meta_evaluation"
            log_agent_action("Orchestrator", "quality_threshold_met" if overall_score >= config.supervisor_threshold else "quality_below_threshold", {"score": overall_score, "aggregation": aggregation, "paper_type": paper_type})
        else:
            state["current_phase"] = "meta_evaluation"
        return state
    except Exception as exc:
        logger.error(f"Supervision failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Supervision error: {exc}")
        # Do not silently send a failed release review back to engineering;
        # that creates an unbounded engineering→supervision loop.  A run with
        # no trustworthy supervisor result is a bounded terminal failure and
        # its dossier remains inspectable.
        state["terminal_error"] = state.get("terminal_error") or f"Supervision failed: {exc}"
        state["should_continue"] = False
        state["current_phase"] = "complete"
        return state


def _synthetic_fallback_allowed(state: ResearchState) -> bool:
    """Only recover when the fallback does not change the requested study."""
    plan = state.get("plan") or {}
    if plan.get("plan_mode") == "bounded_fallback":
        return True
    experiments = plan.get("experiments") or []
    if not experiments:
        return False
    for experiment in experiments:
        dataset = experiment.get("dataset") if isinstance(experiment, dict) else {}
        name = dataset.get("name") if isinstance(dataset, dict) else dataset
        if str(name or "").strip().lower() not in {"bundled_synthetic", "synthetic"}:
            return False
    return True


def _run_synthetic_fallback_outputs(state: ResearchState) -> Dict[str, Dict[str, Any]]:
    """Run a tiny deterministic local experiment for bounded recovery."""
    raw_dir = Path(config.raw_results_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    # v5: no fixed random.seed() — the sandbox preamble seeds each run from
    # its per-seed value, so runs must differ. A fixed seed here collapsed all
    # n_seeds to identical values (f537f638: std=0.0) and faked a zero-width CI.
    code = (
        "import json, random\n"
        "print(json.dumps({'metrics': {'accuracy': 0.6 + random.random() * 0.2, "
        "'seed_variance': random.random() * 0.05}}))\n"
    )
    try:
        aggregate = execute_multi_seed(code, n_seeds=3)
    except Exception as exc:
        logger.warning("Synthetic fallback execution failed: %s", exc)
        return {}
    if not aggregate.get("success"):
        return {}
    recovered: Dict[str, Dict[str, Any]] = {}
    for experiment in (state.get("plan") or {}).get("experiments", []):
        if not isinstance(experiment, dict):
            continue
        name = str(experiment.get("name") or "minimum_viable_experiment")
        raw_path = raw_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}_fallback.json"
        raw_path.write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
        recovered[name] = {
            "success": True,
            "outcome": "fallback_synthetic",
            "failure_recovery": "deterministic local fallback after engineer failure",
            "limitations": ["synthetic data", "exploratory result", "not evidence of novelty"],
            "experiment_name": name,
            "aggregate_metrics": aggregate.get("aggregate_metrics", {}),
            "raw_results_path": str(raw_path),
            "raw_results": aggregate.get("raw_results", []),
            "code": code,
            "contract_hash": (state.get("experiment_contracts", {}).get(name) or {}).get("contract_hash"),
        }
    if recovered:
        # Provenance honesty latch: synthetic plumbing is never clean evidence.
        state["provenance"] = downgrade_provenance(
            str(state.get("provenance") or "clean"), "synthetic")
    return recovered


def meta_evaluation_node(state: ResearchState) -> ResearchState:
    logger.info("Meta evaluation starting")
    log_agent_action("Orchestrator", "start_meta_evaluation", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("meta_evaluation")
    if state.get("terminal_error") or state.get("evidence_gate", {}).get("terminal"):
        state["should_continue"] = False
        state["current_phase"] = "complete"
        log_agent_action("Orchestrator", "meta_blocked_terminal_failure", {})
        return state
    try:
        agent = _create_agent(MetaAgent)
        state["meta_feedback"].append(agent.evaluate_system_performance(state))
        # Funnel feedback (measurement now feeds meta, not just dashboards):
        # record the deepest stage reached; pre-engineering terminals bump a
        # tracker counter so repeated early death is visible run-over-run.
        try:
            from core.funnel import funnel_stage

            _stage = funnel_stage(state)
            state.setdefault("meta_feedback", []).append(f"funnel_stage={_stage}")
            if _stage in ("intake", "topic_discovery", "hypothesis_debate",
                          "planning", "data_validation") and tracker:
                try:
                    tracker.bump("funnel_pre_engineering_death")
                except Exception:
                    pass
        except Exception:
            pass
        if agent.should_reset(state):
            state["should_reset"] = True
            logger.info(f"Meta decision: reset at iteration {state['iteration']}")
            log_agent_action("Orchestrator", "meta_reset_triggered", {"iteration": state["iteration"]})
        elif agent.should_continue(state):
            state["should_continue"] = True
            state["iteration"] += 1
            # Recipe-aware re-entry: survey has no writing_narrative, so fall
            # back to the first drafting phase present in the recipe (B2).
            _order = list(state.get("pipeline_recipe") or [])
            if _order:
                if "writing_narrative" in _order:
                    state["current_phase"] = "writing_narrative"
                elif "writing_results" in _order:
                    state["current_phase"] = "writing_results"
                else:
                    state["current_phase"] = _order[0]
            else:
                state["current_phase"] = "writing_narrative"
            logger.info(f"Meta decision: continue, new iteration {state['iteration']}")
            log_agent_action("Orchestrator", "meta_continue_triggered", {"iteration": state["iteration"]})
        else:
            state["should_continue"] = False
            logger.info(f"Meta decision: stop at iteration {state['iteration']}")
            log_agent_action("Orchestrator", "meta_stop_triggered", {"iteration": state["iteration"]})
        return state
    except Exception as exc:
        logger.error(f"Meta evaluation failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Meta evaluation error: {exc}")
        state["terminal_error"] = f"Meta evaluation failed: {exc}"
        state["should_continue"] = False
        state["current_phase"] = "complete"
        log_agent_action("Orchestrator", "meta_error_stop", {})
        return state


def editing_node(state: ResearchState) -> ResearchState:
    logger.info("Editing starting")
    log_agent_action("Orchestrator", "start_editing", {})
    _record_work_plan(state, "editing", [
        "release-referee gate; route machine patch hints to repair",
        "assemble manuscript; stamp provenance banner when non-clean",
    ])
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("editing")
    if state.get("terminal_error") or state.get("evidence_gate", {}).get("terminal"):
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        editor = _create_agent(EditorAgent)
        try:
            _run_stats = dict(tracker.stats) if tracker is not None and getattr(tracker, "stats", None) else None
        except Exception:
            _run_stats = None
        final_paper = editor.create_final_paper(state["selected_topic"], state["draft_sections"], state["plan"], state["engineer_outputs"], debate_results=state.get("debate_results"), paper_type=str(state.get("paper_type") or "experimental"), thesis=state.get("thesis"), corpus=state.get("corpus"), instrument=state.get("instrument"), survey_analysis=(state.get("analysis_reports") or {}).get("survey"), case_protocol=(state.get("analysis_reports") or {}).get("case_protocol"), causal_claim=state.get("causal_claim"), indep_pass=bool((state.get("analysis_reports") or {}).get("independent")), run_stats=_run_stats, provenance=str(state.get("provenance") or "clean"))
        state["final_paper"] = final_paper
        logger.info(f"Final paper created: {len(final_paper)} chars")
        # Provenance honesty latch: non-clean evidence is never publishable —
        # stamp the banner instead of silently shipping exploratory output.
        if str(state.get("provenance") or "clean") != "clean":
            banner = (
                f"PROVENANCE={state.get('provenance')}: results are exploratory "
                "and NOT publishable evidence; see Limitations."
            )
            try:
                final_paper["provenance_banner"] = banner  # type: ignore[typeddict-unknown-key]
            except Exception:
                pass
            state.setdefault("meta_feedback", []).append(banner)
            log_agent_action("Orchestrator", "provenance_banner_stamped", {
                "provenance": state.get("provenance")})
        state["latex_output"] = editor.generate_latex(final_paper)
        logger.info(f"LaTeX generated: {len(state['latex_output'])} chars")
        state["current_phase"] = "complete"
        log_agent_action("Orchestrator", "editing_complete", {})
        return state
    except RuntimeError as exc:
        # WARA-style artifact repair: a release-referee failure routes the
        # affected sections back for bounded feedback-aware repair rounds
        # (§5.3/R8: bound 2 total) instead of discarding the whole run.
        route = editor_repair_route(str(exc), state.get("editor_repair_count", 0))
        if route == "repair":
            state["editor_repair_count"] = int(state.get("editor_repair_count", 0)) + 1
            state["editor_repair_findings"] = str(exc)[:4000]
            # Machine patch hints ride the exception (EditorAgent.build_repair_hints):
            # route them into supervisor feedback so the redraft is targeted.
            try:
                _hints = list(getattr(exc, "patch_hints", None) or [])
            except Exception:
                _hints = []
            if _hints:
                try:
                    state.setdefault("tool_calls", []).append(
                        {"agent": "EditorAgent", "tool": "release_referee",
                         "capability": "verification.claims",
                         "status": f"{len(_hints)} patch hints"})
                except Exception:
                    pass
                state.setdefault("supervisor_feedback", {})["__editor_patch_hints__"] = "; ".join(
                    f"[{h.get('check')}] {h.get('section')}: {h.get('expected')} ({h.get('detail')})"
                    for h in _hints[:8] if isinstance(h, dict))[:4000]
            state["results_redraft_count"] = min(state.get("results_redraft_count", 0), 1)
            state.setdefault("supervisor_feedback", {})["__editor_findings__"] = str(exc)[:4000]
            state["current_phase"] = "writing_results"
            state["meta_feedback"].append(
                "Editor release referee failed; routing sections for repair pass "
                f"({state['editor_repair_count']}/2): "
                + str(exc)[:1500]
            )
            log_agent_action("Orchestrator", "editor_referee_repair_routed", {
                "repair_count": state["editor_repair_count"],
            })
            return state
        message = f"Editing failed terminally: {exc}"
        logger.error(message, exc_info=True)
        state["meta_feedback"].append(message)
        state["terminal_error"] = state.get("terminal_error") or message
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    except Exception as exc:
        logger.error(f"Editing failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Editing error: {exc}")
        state["terminal_error"] = state.get("terminal_error") or f"Editing failed: {exc}"
        state["should_continue"] = False
        state["current_phase"] = "complete"
        return state


def _exploratory_acceptable(topic: Dict[str, Any], result: Any) -> bool:
    """Whether a failed debate may still hand off as a bounded exploratory topic.

    P3 relaxed: any non-infra FAIL with a near-miss score (within 2.5 of the
    bar) OR any try_and_report topic with a falsifiable hypothesis may proceed
    bounded — not just fallback_provenance topics on the last iteration.
    Infra failures (no substantive debate) still refuse.
    """
    if getattr(result, "infra_failure", False):
        return False
    try:
        from core.config import config as _cfg

        threshold = float(getattr(_cfg, "debate_pass_threshold", 7.0) or 7.0)
    except Exception:
        threshold = 7.0
    try:
        score = float(getattr(result, "score", 0) or 0)
    except Exception:
        score = 0.0
    if (topic or {}).get("fallback_provenance"):
        return True
    if str((topic or {}).get("hunt_mode") or "") in ("try_and_report", "replication"):
        # Try tracks need only a falsifiable core to earn bounded work.
        hyp = (topic or {}).get("structured_hypothesis") or {}
        if isinstance(hyp, dict) and hyp.get("falsification_condition"):
            return True
    if score >= threshold - 2.5:
        return True
    return False


def _fallback_debate_result(topic: Dict[str, Any], prior_result: Any) -> Any:
    """Create an explicit, non-scientific-pass debate record for fallback work."""
    from agents.hypothesis_debate import DebateResult

    return DebateResult(
        topic=str(topic.get("title") or "exploratory topic"),
        proposer_argument=str(topic.get("structured_hypothesis", {}).get("hypothesis") or topic.get("description") or ""),
        challenger_argument="The strict debate did not establish novelty or scientific strength.",
        moderator_decision="EXPLORATORY_HANDOFF",
        score=float(getattr(prior_result, "score", 0.0) or 0.0),
        passed=True,
        reasoning="Accepted only to generate a bounded empirical artifact; not a novelty or quality endorsement.",
        rounds=[{"kind": "fallback", "prior_decision": getattr(prior_result, "moderator_decision", "FAIL")}],
        unresolved_objections=list(getattr(prior_result, "unresolved_objections", []) or [])[:10],
    )


def _minimal_fallback_plan(
    topic: Dict[str, Any],
    validation_error: str,
    paper_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the smallest schema-valid, locally executable research plan.

    ``paper_type`` carries the §4.5 per-type schema into the fallback (M4):
    comparative fallbacks name ≥2 baselines; cause-effect fallbacks carry a
    naive ``causal_spec`` stub so the audit downgrades honestly instead of
    failing on a missing schema. Non-experimental fallbacks carry
    ``experiments: []`` + ``plan_mode: non_empirical`` (owner finish-ready
    directive 2026-10-04: never force engineering on a theory/field plan).
    ``None`` preserves the legacy shape exactly.
    """
    title = str((topic or {}).get("title") or "exploratory study")
    hypothesis = str((topic or {}).get("structured_hypothesis", {}).get("hypothesis") or title)
    plan: Dict[str, Any] = {
        "title": f"Bounded exploratory plumbing check (no inferential claims): {title}",
        "methodology": "Deterministic bundled-synthetic plumbing check with controlled seeds. No inferential claims permitted.",
        "plan_mode": "bounded_fallback",
        "fallback_reason": validation_error,
        # v5: fallback plans are exploratory plumbing — the evidence gate
        # terminal-blocks them (reason exploratory_only) so they can never
        # release as papers. Comparative adjectives are forbidden downstream.
        "exploratory_no_release": True,
        "claim_scope": "plumbing only: execution works, no accuracy/CI conclusions",
        "comparative_language_forbidden": True,
        "sections": [
            {"name": "Abstract"}, {"name": "Introduction"}, {"name": "Related Work"},
            {"name": "Methods"}, {"name": "Results"}, {"name": "Discussion"},
        ],
        "experiments": [{
            "name": "minimum_viable_experiment",
            "type": "comparative_evaluation",
            "description": hypothesis,
            "falsifiable_prediction": hypothesis,
            "dataset": {"name": "bundled_synthetic", "access_policy": "local-only"},
            "evaluation_metrics": ["accuracy", "seed_variance"],
            "variants": [
                {"name": "logistic_regression", "method": "logistic_regression"},
                {"name": "majority_class", "method": "majority_class"},
            ],
            "baselines": ["majority_class"],
            "seeds": [0, 1, 2],
            "statistical_test": "Welch t-test or inconclusive if underpowered",
            "analysis_plan": {"report": ["mean", "std", "effect_size"]},
        }],
    }
    if paper_type == "comparative":
        plan["experiments"][0]["baselines"] = ["majority_class", "random_guess"]
        plan["experiments"][0]["ablation_required"] = True
    if paper_type in (NON_EMPIRICAL_TYPES | FIELD_WAIVER_TYPES):
        # Waiver types only: survey coverage tables / case triangulation
        # matrices are the artifacts. Empirical/experimental/comparative/
        # cause_effect ALWAYS keep the bounded CPU MVE below (d5b5a1db:
        # emptying empirical produced experiments:[] -> skipped engineering
        # -> unpublishable empty paper).
        plan["experiments"] = []
        plan["plan_mode"] = "non_empirical"
    if paper_type == "cause_effect" and plan["experiments"]:
        plan["causal_spec"] = {
            "dag": {"treatment": "intervention", "outcome": title, "confounders": []},
            "id_strategy": "naive",
        }
    if paper_type:
        plan["paper_type"] = paper_type
    return plan


def reset_node(state: ResearchState) -> ResearchState:
    current_iteration = state["iteration"]
    log_agent_action("Orchestrator", "system_reset", {"iteration": current_iteration})
    if current_iteration >= config.max_iterations:
        state["current_phase"] = "complete"
        best = max(
            (float(getattr(r, "score", 0) or 0) for r in state.get("debate_results", []) or []),
            default=None,
        )
        summary = (
            f"Maximum iterations ({config.max_iterations}) reached - stopping"
            + (f"; best debate score this run: {best:.2f}" if best is not None else "; no debates completed")
        )
        state["meta_feedback"].append(summary)
        return state
    # Preserve type routing across resets: a mid-run reset must retry the
    # same paper type, not silently fall back to the experimental spine (B3).
    preserved = {
        key: state.get(key)
        for key in (
            "paper_type", "pipeline_recipe", "evidence_requirements",
            "intake_report", "user_query", "literature_context", "run_id",
        )
    }
    state = initialize_state()
    for key, value in preserved.items():
        if value is not None and value != [] and value != {}:
            try:
                state[key] = value
            except Exception:
                pass
    state["iteration"] = current_iteration + 1
    state["current_phase"] = "topic_discovery"
    log_agent_action("Orchestrator", "reset_complete", {"new_iteration": state["iteration"]})
    return state


def classify_plan_issues(state: ResearchState) -> Dict[str, List[str]]:
    """Split plan problems into hard (must repair/fallback) vs soft (auto-fixable).

    Hard: missing plan, missing experiment name/metrics, empty experiments for
    experimental types. Soft: variant shape issues (missing variant name, non-dict
    variant) — auto-fixable by dropping/naming the variant without killing the run.
    P3: routers use this so soft issues never route to the downgrade node.
    """
    plan = state.get("plan")
    hard: List[str] = []
    soft: List[str] = []
    if not plan or not isinstance(plan, dict):
        return {"hard": ["missing plan"], "soft": []}
    experiments = plan.get("experiments")
    if isinstance(experiments, list) and not experiments:
        effective_type = str(plan.get("paper_type") or state.get("paper_type") or "experimental")
        if effective_type in (NON_EMPIRICAL_TYPES | FIELD_WAIVER_TYPES) or plan.get("plan_mode") == "non_empirical":
            return {"hard": [], "soft": []}
        # P3: exploratory/try tracks may carry a single-MVE plan synthesized
        # downstream — empty here is hard but downgradeable, not terminal.
        return {"hard": ["plan.experiments empty for empirical type"], "soft": []}
    if not isinstance(experiments, list) or not experiments:
        return {"hard": ["plan.experiments must be a non-empty list"], "soft": []}
    for exp in experiments:
        if not isinstance(exp, dict):
            hard.append("experiment is not an object")
            continue
        name = exp.get("name")
        if not isinstance(name, str) or not name.strip():
            hard.append("experiment missing name")
        if not exp.get("evaluation_metrics"):
            hard.append(f"{name or 'unnamed'} missing evaluation_metrics")
        for idx, variant in enumerate(exp.get("variants") or exp.get("alternatives") or []):
            if not isinstance(variant, dict):
                soft.append(f"{name or 'unnamed'} variant[{idx}] not an object (droppable)")
            elif not (isinstance(variant.get("name"), str) and variant["name"].strip()):
                soft.append(f"{name or 'unnamed'} variant[{idx}] missing name (auto-fixable)")
    return {"hard": hard, "soft": soft}


def autofix_soft_plan_issues(state: ResearchState) -> bool:
    """Drop/name malformed variants in place. Returns True when a fix applied."""
    plan = state.get("plan")
    if not isinstance(plan, dict):
        return False
    fixed = False
    for exp in plan.get("experiments") or []:
        if not isinstance(exp, dict):
            continue
        for key in ("variants", "alternatives"):
            variants = exp.get(key)
            if not isinstance(variants, list):
                continue
            kept = []
            for i, v in enumerate(variants):
                if not isinstance(v, dict):
                    fixed = True
                    continue
                if not (isinstance(v.get("name"), str) and v["name"].strip()):
                    v = dict(v)
                    v["name"] = f"variant_{i}"
                    fixed = True
                kept.append(v)
            exp[key] = kept
    return fixed


def is_valid_plan(state: ResearchState) -> bool:
    """Guard: return True only when the plan is present and experiment specs
    satisfy the contract schema required by downstream nodes.

    This catches malformed variants (e.g. dicts missing ``name``) that
    ``validate_experiments`` in evidence_gate passes because it only checks
    ``isinstance(variant, dict)``.

    R3 waiver (mirrors ``validate_experiments``): non-empirical plans
    (``experiments: []`` + waiver paper_type or ``plan_mode ==
    "non_empirical"``) are valid without experiment specs.
    P3: soft variant-shape issues are auto-fixed in place and count as valid
    so they never terminal-route; only hard issues return False.
    """
    issues = classify_plan_issues(state)
    if issues["soft"] and not issues["hard"]:
        try:
            autofix_soft_plan_issues(state)
        except Exception:
            pass
        # Re-evaluate hard issues only after the soft autofix.
        remaining = classify_plan_issues(state)
        if not remaining["hard"] and not remaining["soft"]:
            return True
        if not remaining["hard"]:
            return True
    plan = state.get("plan")
    if not plan or not isinstance(plan, dict):
        return False
    experiments = plan.get("experiments")
    if isinstance(experiments, list) and not experiments:
        effective_type = str(
            plan.get("paper_type") or state.get("paper_type") or "experimental"
        )
        if (
            effective_type in (NON_EMPIRICAL_TYPES | FIELD_WAIVER_TYPES)
            or plan.get("plan_mode") == "non_empirical"
        ):
            return True
        # P3: bounded/exploratory fallback plans are valid carriers even when
        # the type label is still experimental — the downgrade node labels them.
        if plan.get("plan_mode") in ("bounded_fallback", "exploratory"):
            return True
        return False
    if not isinstance(experiments, list) or not experiments:
        return False
    for exp in experiments:
        if not isinstance(exp, dict):
            return False
        name = exp.get("name")
        if not isinstance(name, str) or not name.strip():
            return False
        if not exp.get("evaluation_metrics"):
            return False
        for variant in exp.get("variants") or exp.get("alternatives") or []:
            if not isinstance(variant, dict):
                return False
            if not (isinstance(variant.get("name"), str) and variant["name"].strip()):
                return False
    return True


def terminal_planning_failure(state: ResearchState) -> ResearchState:
    """Downgrade node (P3): schema-invalid plans fall back to bounded work.

    Legacy behavior terminal-killed the run here (END with zero artifact).
    Now: build the deterministic bounded fallback plan and continue down the
    recipe with exploratory provenance. True terminals (infra/config/budget)
    are set elsewhere; this node never ends a run for a fixable schema error.
    Kept under the legacy name so graph wiring + tests keep passing.
    """
    errors = []
    plan = state.get("plan") or {}
    experiments = plan.get("experiments") or []
    for exp in experiments:
        if not isinstance(exp, dict):
            errors.append(f"experiment is not an object: {type(exp).__name__}")
            continue
        name = exp.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append("experiment missing required 'name' field")
        for idx, variant in enumerate(exp.get("variants") or exp.get("alternatives") or []):
            if not isinstance(variant, dict):
                errors.append(f"{name or 'unnamed'} variant[{idx}] is not an object")
            elif not (isinstance(variant.get("name"), str) and variant["name"].strip()):
                errors.append(f"{name or 'unnamed'} variant[{idx}] missing required 'name' field")
    message = "PlannerAgent produced invalid plan: " + "; ".join(errors) if errors else "PlannerAgent produced invalid plan"
    try:
        fallback = _minimal_fallback_plan(
            state.get("selected_topic") or {}, message,
            str(state.get("paper_type") or "experimental"),
        )
    except Exception:
        fallback = _minimal_fallback_plan({}, message)
    state["plan"] = fallback
    state["plan_revision_requests"] = []
    state["evidence_gate"] = {
        "allowed": True,
        "terminal": False,
        "reason_code": "planning_downgraded_exploratory",
        "message": message + " — continuing with bounded fallback",
    }
    state["meta_feedback"].append(message + " — downgraded to bounded exploratory plan")
    try:
        state["current_phase"] = _next_phase(state, "planning", "data_validation")
    except Exception:
        state["current_phase"] = "data_validation"
    state["should_continue"] = True
    state["should_reset"] = False
    # Clear any stale terminal so downstream nodes run.
    state["terminal_error"] = None
    log_agent_action("Orchestrator", "planning_downgraded_exploratory", {"message": message})
    return state


def should_reset(state: ResearchState) -> str:
    if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
        return "end"
    return "reset" if state.get("should_reset") else "continue"


def should_continue(state: ResearchState) -> str:
    if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
        return "end"
    return "continue" if state.get("should_continue") else "end"


# --- M0 intake + pure router (proposal §8.4, §3.2–§3.3) ---

INTAKE_MINI_SCREEN_PROMPT_HASH = "intake-mini-screen-v1"
INTAKE_MINI_SCREEN_PROMPT = (
    "Feasibility+ethics+resources screen (1 call, cheap tier). "
    "Given goal, domain, paper_type, pack_manifest: list blocking "
    "severity-5 resource/ethics findings only. Reply JSON "
    '{"blocking": [], "notes": ""}.'
)


def type_router(paper_type: str, pack: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Pure router: maps type + pack → recipe (proposal §8.4).

    P4: thin compat wrapper over phase_registry.resolve_recipe (single
    implementation). No LLM. For auto requests prefer deferred intake
    (paper_type=auto-deferred + infer_candidates); this wrapper preserves the
    legacy immediate-collapse shape for old callers/tests.
    """
    return route_with_tiebreak(paper_type, pack, llm_tiebreak_fn=None)


def route_with_tiebreak(paper_type: str, pack: Dict[str, Any] | None = None, llm_tiebreak_fn=None) -> Dict[str, Any]:
    """Router with optional 1-call cheap auto-tiebreak (logged).

    P4: single router implementation. resolve_recipe owns tiebreak + recipe.
    """
    from core.phase_registry import resolve_recipe

    recipe, skipped, evidence, manifest = resolve_recipe(paper_type, pack, llm_tiebreak_fn=llm_tiebreak_fn)
    # resolve_recipe already applied tiebreak; recover routed type.
    routed = "experimental"
    try:
        from core.phase_registry import PIPELINES

        for ptype, rec in PIPELINES.items():
            if rec == recipe:
                routed = ptype
                break
    except Exception:
        pass
    return {
        "paper_type": routed if paper_type == "auto" else paper_type,
        "pipeline_recipe": recipe,
        "skipped_phases": skipped,
        "evidence_requirements": evidence,
        "pack_manifest": manifest,
    }


def _mini_feasibility_screen(goal: str, domain: str, paper_type: str, pack_manifest: Any) -> Dict[str, Any]:
    """Exactly 1 cheap-tier call; fail-open degraded on error (§3.3/§4.4 mini).

    Blocks only on severity-5 resource/ethics findings. Any LLM/parse
    failure returns a degraded pass (never crash, never block).
    """
    try:
        from core.llm import call_llm as _call

        prompt = (
            f"{INTAKE_MINI_SCREEN_PROMPT}\n"
            f"goal={goal[:500]} domain={domain} type={paper_type} "
            f"pack={str(pack_manifest)[:500]}"
        )
        # NOTE: 256 tokens starves reasoning endpoints (CoT prefix eats the
        # budget → length-truncation with zero text, run 2026-10-05). 2048
        # covers the reasoning prefix plus the small JSON verdict; json_mode
        # returns partials for parse-salvage instead of a full ~300s re-run.
        raw = _call(prompt, tier="cheap", temperature=0.0, max_tokens=2048,
                    retry_on_truncation=False)
        data = parse_json_from_llm(raw) if raw else None
        if isinstance(data, dict):
            blocking = data.get("blocking") or []
            sev5 = [b for b in blocking if isinstance(b, dict) and int(b.get("severity", 0)) >= 5]
            if sev5:
                return {"blocked": True, "findings": sev5}
            return {"blocked": False, "findings": []}
    except Exception as e:
        # Concise one-liner, never exc_info: a mini-screen miss is routine
        # degraded operation (the run continues), not a crash.
        logger.warning("Intake mini screen degraded open (%s)", str(e)[:160])
    # Fail-open degraded: signal the screen didn't run without crashing intake.
    return {"blocked": False, "findings": [], "degraded": True}


def intake_node(state: ResearchState) -> ResearchState:
    """Validate pack → manifest → 1-call mini screen → intake_report (§8.4)."""
    from core.phase_registry import resolve_recipe, validate_pack

    tracker = get_tracker()
    if tracker:
        tracker.set_phase("intake")
    try:
        from core.run_log import emit_event as _emit

        _emit("phase_start", {"phase": "intake", "agent": "DataAgent", "status": "ok", "message_human": "intake start", "cause_class": "science"}, agent="DataAgent")
    except Exception:
        pass
    report = dict(state.get("intake_report") or {})
    goal = str(report.get("goal") or state.get("user_query") or "")
    domain = str(report.get("domain") or getattr(config, "research_domain", "computer_science"))
    requested = str(report.get("requested") or state.get("paper_type") or getattr(config, "paper_type_default", "experimental"))
    pack = dict(report.get("pack_manifest") or {})
    # M3: resolve pack_id carriers (POST /api/packs) into full pack dicts.
    try:
        from core.packs import resolve_pack_reference as _resolve_pack

        pack = _resolve_pack(pack)
    except Exception:
        pass
    # CLI/web may stash pack under state keys; merge when present.
    for key in ("response_table", "case_pack", "corpus_override", "codebook", "protocol", "dag"):
        if key in state and state[key] and key not in pack:
            try:
                pack[key] = state[key]  # type: ignore[index]
            except Exception:
                pass
    depth = str(report.get("depth") or "Standard")
    budgets = dict(report.get("budgets") or {})

    # 1. Pack validation → named refusal (never traceback).
    ok, refusal = validate_pack(requested if requested != "auto" else _infer_pack_type(pack), pack)
    if not ok and refusal:
        state["terminal_error"] = refusal["reason"]
        state["current_phase"] = "complete"
        state["should_continue"] = False
        state["intake_report"] = {
            "goal": goal, "domain": domain, "requested": requested,
            "pack_manifest": pack, "depth": depth, "budgets": budgets,
            "preflight": {"refusal": refusal},
        }
        if tracker:
            tracker.message(f"Intake refusal {refusal['reason']}", level="error")
        try:
            from core.run_log import emit_event as _emit2

            _emit2("refusal", {"phase": "intake", "agent": "DataAgent", "status": "blocked", "message_human": f"intake refusal {refusal['reason']}", "refusal": refusal, "cause_class": "config"}, agent="DataAgent")
            _emit2("phase_end", {"phase": "intake", "agent": "DataAgent", "status": "blocked", "message_human": "intake refused", "cause_class": "config"}, agent="DataAgent")
        except Exception:
            pass
        logger.error("Intake refusal %s", refusal["reason"])
        return state

    # 2. Route → recipe + skipped + evidence.
    # P1 deferred binding: requested==auto no longer collapses to a single
    # experimental recipe. Intake records ranked candidates and leaves
    # paper_type=auto-deferred + empty recipe; post-evidence routing
    # (resolve_deferred_routing) binds the final type after topics/literature/data.
    # Explicit requested types keep the exact legacy immediate binding.
    try:
        from core.phase_registry import DEFERRED_TYPE, infer_candidates

        if requested == "auto":
            candidates = infer_candidates(pack)
            # Mini screen runs against the top candidate label only for the
            # ethics/resources preflight; it does not bind the run.
            screen = _mini_feasibility_screen(goal, domain, str(candidates[0]["type"] if candidates else "experimental"), pack)
            if screen.get("blocked"):
                state["terminal_error"] = "needs_primary_data"
                state["current_phase"] = "complete"
                state["should_continue"] = False
                state["intake_report"] = {
                    "goal": goal, "domain": domain, "requested": requested,
                    "pack_manifest": pack, "depth": depth, "budgets": budgets,
                    "preflight": {"mini_screen": screen},
                }
                return state
            state["paper_type"] = DEFERRED_TYPE
            state["pipeline_recipe"] = []
            state["skipped_phases"] = []
            state["evidence_requirements"] = {}
            state["paper_candidates"] = candidates
            state["hunt_mode"] = str(report.get("hunt_mode") or state.get("hunt_mode") or "try_and_report")
            state["routing_provenance"] = {"stage": "intake-deferred", "candidates": [c["type"] for c in candidates]}
            state["intake_report"] = {
                "goal": goal, "domain": domain, "requested": requested,
                "pack_manifest": pack, "depth": depth, "budgets": budgets,
                "preflight": {"mini_screen": screen, "prompt_hash": INTAKE_MINI_SCREEN_PROMPT_HASH},
            }
            state["current_phase"] = "topic_discovery"
            try:
                from core.run_log import emit_event as _emit3

                _emit3("phase_end", {"phase": "intake", "agent": "DataAgent", "status": "ok", "message_human": "intake deferred auto", "recipe": [], "cause_class": "science"}, agent="DataAgent")
            except Exception:
                pass
            return state
        recipe, skipped, evidence, _manifest = resolve_recipe(requested, pack)
        routed = requested
    except Exception as e:
        logger.error("Intake routing failed: %s", e, exc_info=True)
        state["terminal_error"] = str(e)
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state

    # 3. 1-call mini screen (cheap, fail-closed).
    screen = _mini_feasibility_screen(goal, domain, routed, pack)
    if screen.get("blocked"):
        state["terminal_error"] = "needs_primary_data"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        state["intake_report"] = {
            "goal": goal, "domain": domain, "requested": requested,
            "pack_manifest": pack, "depth": depth, "budgets": budgets,
            "preflight": {"mini_screen": screen},
        }
        return state

    state["paper_type"] = routed
    state["pipeline_recipe"] = recipe
    state["skipped_phases"] = skipped
    state["evidence_requirements"] = evidence
    state["intake_report"] = {
        "goal": goal, "domain": domain, "requested": requested,
        "pack_manifest": pack, "depth": depth, "budgets": budgets,
        "preflight": {"mini_screen": screen, "prompt_hash": INTAKE_MINI_SCREEN_PROMPT_HASH},
    }
    state["current_phase"] = "topic_discovery"
    try:
        from core.run_log import emit_event as _emit3

        _emit3("phase_end", {"phase": "intake", "agent": "DataAgent", "status": "ok", "message_human": f"intake routed {routed}", "recipe": recipe, "cause_class": "science"}, agent="DataAgent")
    except Exception:
        pass
    return state


def _infer_pack_type(pack: Dict[str, Any]) -> str:
    # Backward-compat shim over phase_registry._auto_route (P4 fluff removal:
    # single inference implementation lives in phase_registry).
    try:
        from core.phase_registry import _auto_route as _reg_auto

        return _reg_auto(pack)
    except Exception:
        if pack.get("case_pack"):
            return "case_study"
        if pack.get("response_table"):
            return "survey"
        return "experimental"


def resolve_deferred_routing(state: ResearchState) -> ResearchState:
    """Bind auto-deferred runs to a final paper_type AFTER evidence exists.

    Called lazily at the start of planning (and data_validation as backstop)
    so topic_discovery + debate + literature_context inform the decision.
    Pure-deterministic: pack signals win when present; otherwise the topic's
    evidence strength / hunt_mode picks the honest type. Never terminal —
    falls back to experimental with provenance labelled.
    """
    from core.phase_registry import DEFERRED_TYPE, PIPELINES, EVIDENCE_REQUIREMENTS

    if str(state.get("paper_type") or "") != DEFERRED_TYPE:
        return state
    pack = dict((state.get("intake_report") or {}).get("pack_manifest") or {})
    for key in ("response_table", "case_pack", "corpus_override", "codebook", "protocol", "dag"):
        if key in state and state[key] and key not in pack:  # type: ignore[index]
            try:
                pack[key] = state[key]  # type: ignore[index]
            except Exception:
                pass
    topic = state.get("selected_topic") or {}
    hunt_mode = str(state.get("hunt_mode") or topic.get("hunt_mode") or "try_and_report")
    evidence = topic.get("literature_evidence") or state.get("literature_context") or []
    n_evidence = len(evidence) if isinstance(evidence, list) else 0
    routed = "experimental"
    why = "default executable fallback"
    if pack.get("case_pack"):
        routed, why = "case_study", "case_pack present at routing"
    elif pack.get("response_table"):
        routed, why = "survey", "response_table present at routing"
    elif pack.get("corpus_override") or pack.get("protocol"):
        routed, why = "review", "corpus/protocol present at routing"
    elif pack.get("causal_claim") or pack.get("dag"):
        routed, why = "cause_effect", "causal claim/dag present at routing"
    elif hunt_mode == "replication" or str(topic.get("replication_target") or ""):
        routed, why = "empirical", "replication target with evidence"
    elif n_evidence == 0 and hunt_mode == "try_and_report":
        # No literature but executable idea -> honest empirical try-and-report,
        # not a fake gap claim.
        routed, why = "empirical", "try_and_report without prior literature"
    elif n_evidence >= 3 and bool((topic.get("structured_hypothesis") or {}).get("falsification_condition")):
        routed, why = "experimental", "falsifiable hypothesis + literature support"
    try:
        recipe = list(PIPELINES[routed])
        skipped = [{"phase": name, "reason": f"not in {routed} recipe"} for name in (
            "systematic_review", "thesis_argument", "survey_analysis", "case_protocol", "causal_audit")
            if name not in set(recipe)]
        state["paper_type"] = routed
        state["pipeline_recipe"] = recipe
        state["skipped_phases"] = skipped
        state["evidence_requirements"] = dict(EVIDENCE_REQUIREMENTS[routed])
        # Ranked multi-paper candidates from this single hunt (P1e).
        cands = list(state.get("paper_candidates") or [])
        if routed not in [c.get("type") for c in cands]:
            cands.insert(0, {"type": routed, "confidence": 0.75, "why": why})
        state["paper_candidates"] = cands
        state["routing_provenance"] = {
            "stage": "post-evidence", "routed": routed, "why": why,
            "n_evidence": n_evidence, "hunt_mode": hunt_mode,
        }
        log_agent_action("Orchestrator", "deferred_routing_resolved", {
            "routed": routed, "why": why, "n_evidence": n_evidence,
        })
    except Exception as exc:
        logger.warning("Deferred routing failed open to experimental (%s)", str(exc)[:160])
        state["paper_type"] = "experimental"
        state["pipeline_recipe"] = list(PIPELINES["experimental"])
        state["evidence_requirements"] = dict(EVIDENCE_REQUIREMENTS["experimental"])
        state["routing_provenance"] = {"stage": "post-evidence-fallback", "why": str(exc)[:200]}
    return state


def routing_node(state: ResearchState) -> ResearchState:
    """Explicit graph node wrapper for deferred routing (superset graph)."""
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("routing")
    state = resolve_deferred_routing(state)
    state["current_phase"] = _next_phase(state, "planning", "planning")
    return state


def thesis_argument_node(state: ResearchState) -> ResearchState:
    """Build the structured thesis for no-experiment paper types (§3.3, M1).

    Runs ThesisAgent.build_thesis over the selected topic + retrieved
    literature evidence and stores state["thesis"], plus the light
    non-empirical plan (§4.5) when no plan exists yet — these recipes have
    no planning phase, so nothing else populates it. Downstream gates enforce
    the triad + 100% bridge coverage (§7.1); this node never terminal-fails
    on a thin thesis — it records the triad status for audit.
    """
    from agents.thesis_agent import ThesisAgent

    logger.info("Thesis argument starting")
    log_agent_action("Orchestrator", "start_thesis_argument", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("thesis_argument")
    try:
        from core.run_log import emit_event as _emit

        _emit("phase_start", {"phase": "thesis_argument", "agent": "ThesisAgent", "status": "ok", "message_human": "thesis build start", "cause_class": "science"}, agent="ThesisAgent")
    except Exception:
        pass
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    topic = state.get("selected_topic") or {}
    if not topic:
        state["meta_feedback"].append("Thesis build skipped: no selected topic")
        state["current_phase"] = _next_phase(state, "thesis_argument", "writing_narrative")
        return state
    try:
        paper_type = str(state.get("paper_type") or "experimental")
        if not state.get("thesis"):
            evidence = topic.get("literature_evidence") or state.get("literature_context") or []
            state["thesis"] = _create_agent(ThesisAgent).build_thesis(topic, evidence, paper_type)
        if not state.get("plan"):
            state["plan"] = _non_empirical_plan(paper_type, state.get("thesis"), topic)
            log_agent_action("Orchestrator", "non_empirical_plan_synthesized", {
                "paper_type": paper_type,
                "sections": len(state["plan"]["sections"]),
            })
        thesis = state["thesis"]
        triad_ok = all(ThesisAgent._triad_status(thesis).values())
        logger.info("Thesis built for '%s': triad_complete=%s bridges=%d", topic.get("title"), triad_ok, len(thesis.get("bridges") or []))
        log_agent_action("Orchestrator", "thesis_built", {
            "paper_type": paper_type,
            "triad_complete": triad_ok,
            "n_bridges": len(thesis.get("bridges") or []),
            "fallback_used": bool(thesis.get("fallback_used")),
        })
        # Recipe-aware successor: review/analytical → writing_narrative,
        # argumentative → hypothesis_debate (steelman); legacy runs without a
        # recipe keep the historical writing_narrative label.
        state["current_phase"] = _next_phase(state, "thesis_argument", "writing_narrative")
        state["error_count"] = 0
        try:
            from core.run_log import emit_event as _emit2

            _emit2("phase_end", {"phase": "thesis_argument", "agent": "ThesisAgent", "status": "ok" if triad_ok else "warn", "message_human": f"thesis built triad_complete={triad_ok}", "cause_class": "science"}, agent="ThesisAgent")
        except Exception:
            pass
        return state
    except Exception as exc:
        logger.error(f"Thesis build failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Thesis build error: {exc}")
        state["error_count"] = int(state.get("error_count") or 0) + 1
        if state["error_count"] >= 3:
            state["should_reset"] = True
        else:
            state["current_phase"] = _next_phase(state, "thesis_argument", "writing_narrative")
        return state


def systematic_review_node(state: ResearchState) -> ResearchState:
    """Run the systematic review corpus pass (§3.3 recipe, §4.6, M2).

    Screen → OA-mark → extract → synthesize → PRISMA reconcile → seed recall.
    Stores ``state["corpus"]`` (``{prisma, screen_log, extraction,
    seed_recall}`` per §3.1) and narrows the topic's ``literature_evidence``
    to the included, OA-marked papers so ``thesis_argument`` synthesizes from
    the corpus. Closed content stays abstract-only (``full_text_unavailable``)
    and continues — never terminal, never paywalled fetch.

    Seed recall below 4/5 demotes systematic → scoping + banner (not terminal).
    This node never terminal-fails on a thin corpus — downstream
    ``verify_review`` (§7.1) enforces the release gate.
    """
    from agents.review_agent import ReviewAgent

    logger.info("Systematic review starting")
    log_agent_action("Orchestrator", "start_systematic_review", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("systematic_review")
    try:
        from core.run_log import emit_event as _emit

        _emit("phase_start", {"phase": "systematic_review", "agent": "ReviewAgent",
                              "status": "ok", "message_human": "systematic review start",
                              "cause_class": "science"}, agent="ReviewAgent")
    except Exception:
        pass
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    topic = state.get("selected_topic") or {}
    if not topic:
        state["meta_feedback"].append("Systematic review skipped: no selected topic")
        state["current_phase"] = _next_phase(state, "systematic_review", "thesis_argument")
        return state
    try:
        candidates = list(topic.get("literature_evidence") or state.get("literature_context") or [])
        intake = state.get("intake_report") or {}
        pack = dict(intake.get("pack_manifest") or {})
        depth = str(intake.get("depth") or "Standard")
        from agents.review_agent import (
            build_prisma_counts,
            prisma_reconcile,
            review_caps,
            seed_recall_check,
        )

        # Review caps (§5.3): screen caps 40/120/200 by depth. Truncate with
        # a labelled warning instead of silently over-spending LLM budget.
        screen_cap, _fulltext_cap = review_caps(depth)
        if len(candidates) > screen_cap:
            log_agent_action("Orchestrator", "review_screen_cap_truncated", {
                "identified": len(candidates), "screen_cap": screen_cap, "depth": depth})
            candidates = candidates[:screen_cap]
            state.setdefault("meta_feedback", []).append(
                f"Review screen capped at {screen_cap} ({depth}); excess candidates deferred.")
        # Seed keys (§4.6 step 6 + §5.2 corpus_override): explicit pins win,
        # then corpus_override DOIs (≤50, truncated + logged), then topic pins,
        # then the first 5 candidate DOIs (retention probe, not a gold set).
        seed_keys = list(intake.get("seed_keys") or topic.get("seed_keys") or [])
        override_dois: list = []
        override = pack.get("corpus_override") or state.get("corpus_override") or {}
        if isinstance(override, dict):
            override_dois = [str(d).strip() for d in (override.get("dois") or []) if str(d or "").strip()]
        elif isinstance(override, list):
            override_dois = [str(d).strip() for d in override if str(d or "").strip()]
        if override_dois and not seed_keys:
            if len(override_dois) > 50:
                log_agent_action("Orchestrator", "corpus_override_truncated", {
                    "n_dois": len(override_dois), "cap": 50})
                state.setdefault("meta_feedback", []).append(
                    "corpus_override truncated to 50 entries (§5.2).")
                override_dois = override_dois[:50]
            seed_keys = override_dois
        if not seed_keys:
            seed_keys = [str(p.get("doi") or "").strip() for p in candidates[:5]
                         if isinstance(p, dict) and p.get("doi")]
        query = str(intake.get("goal") or topic.get("title") or "")
        criteria = intake.get("protocol") or topic.get("protocol") or {}

        agent = _create_agent(ReviewAgent)
        screened = agent.screen(candidates, criteria, query)
        marked, limitations = agent.mark_full_text(screened["included"])
        extracted = agent.extract(marked)
        synthesis = agent.synthesize(extracted["extraction"])

        prisma_counts = build_prisma_counts(candidates, screened["included"], screened["excluded"])
        prisma_counts["included"] = len(marked)
        prisma_counts["eligible"] = len(screened["included"])
        prisma_counts["excluded_fulltext"] = max(0, prisma_counts["eligible"] - len(marked))

        prisma = prisma_reconcile(prisma_counts)
        recall = seed_recall_check([str(p.get("doi") or "") for p in marked], seed_keys)

        try:
            from core.run_log import emit_event as _emit_gate

            _emit_gate("gate_decision", {
                "phase": "systematic_review", "agent": "ReviewAgent", "status": "ok" if prisma["passed"] else "fail",
                "message_human": f"PRISMA reconcile {'ok' if prisma['passed'] else 'FAIL'} {prisma_counts}",
                "gate": "prisma_reconcile", "threshold": "exact balance", "actual": prisma_counts,
                "cause_class": "science", "retryable": False,
                "next_action": "continue" if prisma["passed"] else "fix counts (blocking)"},
                agent="ReviewAgent")
            _emit_gate("gate_decision", {
                "phase": "systematic_review", "agent": "ReviewAgent",
                "status": "ok" if extracted["passed"] else "fail",
                "message_human": f"extraction completeness {extracted['completeness']:.2f} vs 0.90",
                "gate": "extraction_completeness", "threshold": 0.9, "actual": extracted["completeness"],
                "cause_class": "science", "retryable": False,
                "next_action": "continue" if extracted["passed"] else "re-extract (blocking)"},
                agent="ReviewAgent")
            _emit_gate("gate_decision", {
                "phase": "systematic_review", "agent": "ReviewAgent",
                "status": "ok" if recall["recall"] is None or recall["passed"] else "warn",
                "message_human": f"seed recall {recall.get('recall')} mode={recall.get('mode')}",
                "gate": "seed_recall", "threshold": 0.8, "actual": recall.get("recall"),
                "cause_class": "science", "retryable": False,
                "next_action": "continue systematic" if recall.get("passed", True) else "demote to scoping (banner)"},
                agent="ReviewAgent")
            _emit_gate("source_fetch", {
                "phase": "systematic_review", "agent": "ReviewAgent", "status": "ok",
                "message_human": f"OA resolution: {len(marked) - len(limitations)} OA, {len(limitations)} abstract-only",
                "provider": "oa_resolution", "counts": {"oa": len(marked) - len(limitations), "closed": len(limitations)},
                "cause_class": "science"}, agent="ReviewAgent")
        except Exception:
            pass

        corpus = {
            "prisma": prisma_counts,
            "prisma_reconciled": prisma["passed"],
            "screen_log": screened["screen_log"],
            "extraction": extracted["extraction"],
            "extraction_completeness": extracted["completeness"],
            "synthesis": synthesis,
            "seed_recall": recall,
            "full_text_limitations": limitations,
            "mode": recall.get("mode", "systematic"),
            "banner": recall.get("banner"),
            "n_identified": len(candidates),
            "n_included": len(marked),
        }
        state["corpus"] = corpus
        # Narrow downstream evidence to the included corpus (OA-marked).
        if isinstance(state.get("selected_topic"), dict):
            state["selected_topic"] = dict(state["selected_topic"])
            state["selected_topic"]["literature_evidence"] = marked
        if limitations:
            state.setdefault("meta_feedback", []).append(
                "Full-text limits (abstract-only): " + "; ".join(limitations[:5]))
        if recall.get("demote_to_scoping"):
            state.setdefault("meta_feedback", []).append(str(recall.get("banner") or ""))
            log_agent_action("Orchestrator", "review_demoted_to_scoping", {
                "recall": recall.get("recall"), "banner": recall.get("banner")})
        logger.info("Systematic review complete: identified=%d included=%d prisma=%s recall=%s mode=%s",
                    len(candidates), len(marked), prisma["passed"], recall.get("recall"), corpus["mode"])
        log_agent_action("Orchestrator", "systematic_review_complete", {
            "identified": len(candidates), "included": len(marked),
            "prisma_reconciled": prisma["passed"],
            "extraction_completeness": extracted["completeness"],
            "recall": recall.get("recall"), "mode": corpus["mode"],
            "full_text_unavailable": len(limitations),
        })
        state["current_phase"] = _next_phase(state, "systematic_review", "thesis_argument")
        state["error_count"] = 0
        try:
            from core.run_log import emit_event as _emit2

            _emit2("phase_end", {"phase": "systematic_review", "agent": "ReviewAgent",
                                "status": "ok" if prisma["passed"] else "warn",
                                "message_human": f"review corpus included={len(marked)} mode={corpus['mode']}",
                                "cause_class": "science"}, agent="ReviewAgent")
        except Exception:
            pass
        return state
    except Exception as exc:
        logger.error(f"Systematic review failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Systematic review error: {exc}")
        state["error_count"] = int(state.get("error_count") or 0) + 1
        if state["error_count"] >= 3:
            state["should_reset"] = True
        else:
            state["current_phase"] = _next_phase(state, "systematic_review", "thesis_argument")
        return state


def survey_analysis_node(state: ResearchState) -> ResearchState:
    """Run the survey response-table analysis (§3.3 recipe, §4.8, M3).

    DataAgent validates the pack response_table (handoff); SurveyAgent runs
    descriptives → α → nonresponse → fishing guard → synthetic screen.
    Stores ``state["instrument"]`` (``{questionnaire, codebook_hash,
    response_hash, nonresponse}``) and
    ``state["analysis_reports"]["survey"]``. A tripped synthetic screen is
    recorded as a blocking finding for ``verify_survey`` — this node never
    fabricates rows to pass the gate.
    """
    from agents.survey_agent import SurveyAgent

    logger.info("Survey analysis starting")
    log_agent_action("Orchestrator", "start_survey_analysis", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("survey_analysis")
    try:
        from core.run_log import emit_event as _emit

        _emit("phase_start", {"phase": "survey_analysis", "agent": "SurveyAgent",
                              "status": "ok", "message_human": "survey analysis start",
                              "cause_class": "science"}, agent="SurveyAgent")
    except Exception:
        pass
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        from core.packs import resolve_pack_reference, validate_codebook_file

        intake = state.get("intake_report") or {}
        pack = resolve_pack_reference(dict(intake.get("pack_manifest") or {}))
        for key in ("response_table", "codebook"):
            if key in state and state[key] and key not in pack:  # type: ignore[index]
                pack[key] = state[key]  # type: ignore[index]
        response_ref = pack.get("response_table") or {}
        response_path = response_ref.get("path") if isinstance(response_ref, dict) else (response_ref if isinstance(response_ref, str) else None)
        if isinstance(response_ref, dict) and not response_path:
            manifest = response_ref.get("manifest") if isinstance(response_ref.get("manifest"), dict) else {}
            response_path = manifest.get("path") or manifest.get("location")
        target = response_ref.get("target") if isinstance(response_ref, dict) else None
        codebook_ref = pack.get("codebook")
        codebook_path = codebook_ref.get("path") if isinstance(codebook_ref, dict) else (codebook_ref if isinstance(codebook_ref, str) else None)
        codebook: Dict[str, Any] = {}
        if codebook_path:
            codebook = validate_codebook_file(codebook_path)
        if not target and codebook.get("targets"):
            target = list(codebook["targets"])[0]
        subgroups = list(codebook.get("preregistered_subgroups") or []) if codebook else []
        # DataAgent handoff: the frame comes from the validated pack file.
        agent_data = _create_agent(DataAgent)
        info = agent_data._read_frame_info(Path(response_path)) if response_path else None
        if info is None:
            state["terminal_error"] = "needs_response_table"
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state
        frame = info["frame"]
        requested = pack.get("subgroups") or (state.get("survey_subgroups") if isinstance(state.get("survey_subgroups"), list) else []) or []
        analysis = _create_agent(SurveyAgent).run_analysis(
            frame, target, None, None, list(requested or []), subgroups,
            {"items": codebook.get("items", []), "manifest": codebook.get("manifest", {}),
             "preregistered_subgroups": subgroups} if codebook else None)
        state["instrument"] = analysis["instrument"]
        reports = dict(state.get("analysis_reports") or {})
        reports["survey"] = analysis
        state["analysis_reports"] = reports
        # Surface pack-level flags on the instrument for the release bundle.
        state["instrument"] = dict(state["instrument"] or {})
        state["instrument"]["response_truncated"] = bool(info.get("truncated"))
        state["instrument"]["response_encoding"] = info.get("encoding")
        try:
            from core.run_log import emit_event as _emit_gate

            _emit_gate("gate_decision", {
                "phase": "survey_analysis", "agent": "SurveyAgent",
                "status": "ok" if analysis["passed"] else "fail",
                "message_human": f"survey descriptives rows={analysis['validation'].get('rows')} alpha={analysis['alpha'].get('alpha')} synthetic={analysis['synthetic'].get('is_synthetic')}",
                "gate": "survey_analysis", "threshold": "no synthetic + valid table",
                "actual": {"passed": analysis["passed"], "synthetic": analysis["synthetic"].get("is_synthetic"),
                           "alpha": analysis["alpha"].get("alpha"),
                           "unit_nonresponse": analysis["nonresponse"].get("unit_rate")},
                "cause_class": "science", "retryable": False,
                "next_action": "continue" if analysis["passed"] else "blocked: fix pack or record refusal"},
                agent="SurveyAgent")
            _emit_gate("artifact_write", {
                "phase": "survey_analysis", "agent": "SurveyAgent", "status": "ok",
                "message_human": f"survey instrument response_hash={analysis['instrument'].get('response_hash', '')[:12]}",
                "artifact_path": str(response_path), "sha256": str(analysis["instrument"].get("response_hash") or ""),
                "rows": int(analysis["validation"].get("rows") or 0), "cause_class": "science"},
                agent="SurveyAgent")
        except Exception:
            pass
        if analysis["synthetic"].get("blocking"):
            state.setdefault("meta_feedback", []).append(
                "Synthetic screen tripped: " + "; ".join(analysis["synthetic"].get("reasons") or []))
        log_agent_action("Orchestrator", "survey_analysis_complete", {
            "rows": analysis["validation"].get("rows"), "alpha": analysis["alpha"].get("alpha"),
            "synthetic": analysis["synthetic"].get("is_synthetic"), "passed": analysis["passed"]})
        # Field recipes have no planning phase: synthesize the light plan
        # (§4.5, experiments: []) when missing so writing_results can run.
        # Never clobbers an existing plan.
        if not state.get("plan"):
            topic = state.get("selected_topic") or {}
            state["plan"] = _non_empirical_plan("survey", None, topic if isinstance(topic, dict) else {})
            log_agent_action("Orchestrator", "non_empirical_plan_synthesized", {
                "paper_type": "survey", "sections": len(state["plan"]["sections"])})
        state["current_phase"] = _next_phase(state, "survey_analysis", "writing_results")
        state["error_count"] = 0
        try:
            from core.run_log import emit_event as _emit2

            _emit2("phase_end", {"phase": "survey_analysis", "agent": "SurveyAgent",
                                 "status": "ok" if analysis["passed"] else "warn",
                                 "message_human": f"survey analysis passed={analysis['passed']}",
                                 "cause_class": "science"}, agent="SurveyAgent")
        except Exception:
            pass
        return state
    except Exception as exc:
        logger.error(f"Survey analysis failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Survey analysis error: {exc}")
        state["current_phase"] = _next_phase(state, "survey_analysis", "writing_results")
        return state


def case_protocol_node(state: ResearchState) -> ResearchState:
    """Run the case-study protocol pass (§3.3 recipe, §4.9, M3).

    CaseAgent validates the pack (≥2 source types), builds the
    triangulation matrix (every material claim → source list), and runs the
    rival check (≥1 rival or explicit gap). Stores
    ``state["analysis_reports"]["case_protocol"]``. Single-source flags are
    warnings with limitation entries — the run continues.
    """
    from agents.case_agent import CaseAgent

    logger.info("Case protocol starting")
    log_agent_action("Orchestrator", "start_case_protocol", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("case_protocol")
    try:
        from core.run_log import emit_event as _emit

        _emit("phase_start", {"phase": "case_protocol", "agent": "CaseAgent",
                              "status": "ok", "message_human": "case protocol start",
                              "cause_class": "science"}, agent="CaseAgent")
    except Exception:
        pass
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        from core.packs import resolve_pack_reference, validate_case_pack_file

        intake = state.get("intake_report") or {}
        pack = resolve_pack_reference(dict(intake.get("pack_manifest") or {}))
        if "case_pack" in state and state["case_pack"] and "case_pack" not in pack:  # type: ignore[index]
            pack["case_pack"] = state["case_pack"]  # type: ignore[index]
        manifest = pack.get("case_pack") or {}
        # File carriers ({path: ...}) validate through the §5.2 zip reader
        # first; entry manifests pass straight to the agent.
        if isinstance(manifest, dict) and manifest.get("path") and not manifest.get("entries"):
            file_result = validate_case_pack_file(manifest["path"])
            if file_result.get("ok"):
                manifest = {"manifest": file_result.get("manifest"),
                            "entries": file_result.get("entries"),
                            "distinct_types": file_result.get("distinct_types")}
        topic = state.get("selected_topic") or {}
        claims = pack.get("claims") or state.get("case_claims") or topic.get("case_claims") or []
        evidence_map = pack.get("evidence_map") or state.get("case_evidence_map") or topic.get("case_evidence_map") or {}
        rivals = pack.get("rivals") or state.get("case_rivals") or topic.get("case_rivals") or []
        gap_note = pack.get("rival_gap") or state.get("case_rival_gap") or topic.get("case_rival_gap") or ""
        protocol = _create_agent(CaseAgent).run_protocol(manifest, list(claims or []),
                                                         dict(evidence_map or {}),
                                                         list(rivals or []), str(gap_note or ""))
        reports = dict(state.get("analysis_reports") or {})
        reports["case_protocol"] = protocol
        state["analysis_reports"] = reports
        if protocol["triangulation"].get("limitations"):
            state.setdefault("meta_feedback", []).append(
                "Single-source limitations: " + "; ".join(protocol["triangulation"]["limitations"][:5]))
        try:
            from core.run_log import emit_event as _emit_gate

            _emit_gate("gate_decision", {
                "phase": "case_protocol", "agent": "CaseAgent",
                "status": "ok" if protocol["passed"] else "fail",
                "message_human": f"case protocol claims={protocol['triangulation'].get('n_claims')} flagged={protocol['triangulation'].get('n_flagged')} rival_ok={protocol['rival'].get('passed')}",
                "gate": "case_protocol", "threshold": "pack valid + rival addressed-or-gap",
                "actual": {"passed": protocol["passed"],
                           "n_flagged": protocol["triangulation"].get("n_flagged"),
                           "rival_passed": protocol["rival"].get("passed")},
                "cause_class": "science", "retryable": False,
                "next_action": "continue" if protocol["passed"] else "blocked: fix pack or record refusal"},
                agent="CaseAgent")
        except Exception:
            pass
        log_agent_action("Orchestrator", "case_protocol_complete", {
            "n_claims": protocol["triangulation"].get("n_claims"),
            "n_flagged": protocol["triangulation"].get("n_flagged"),
            "rival_passed": protocol["rival"].get("passed"), "passed": protocol["passed"]})
        # Field recipes have no planning phase: synthesize the light plan
        # (§4.5, experiments: []) when missing so narrative/results can run.
        # Never clobbers an existing plan.
        if not state.get("plan"):
            topic = state.get("selected_topic") or {}
            state["plan"] = _non_empirical_plan("case_study", None, topic if isinstance(topic, dict) else {})
            log_agent_action("Orchestrator", "non_empirical_plan_synthesized", {
                "paper_type": "case_study", "sections": len(state["plan"]["sections"])})
        state["current_phase"] = _next_phase(state, "case_protocol", "writing_narrative")
        state["error_count"] = 0
        try:
            from core.run_log import emit_event as _emit2

            _emit2("phase_end", {"phase": "case_protocol", "agent": "CaseAgent",
                                 "status": "ok" if protocol["passed"] else "warn",
                                 "message_human": f"case protocol passed={protocol['passed']}",
                                 "cause_class": "science"}, agent="CaseAgent")
        except Exception:
            pass
        return state
    except Exception as exc:
        logger.error(f"Case protocol failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Case protocol error: {exc}")
        state["current_phase"] = _next_phase(state, "case_protocol", "writing_narrative")
        return state


def causal_audit_node(state: ResearchState) -> ResearchState:
    """Run the cause-effect identification audit (§3.3 recipe, §4.7, M4).

    CausalAgent classifies the identification strategy (rct / quasi /
    synthetic_benchmark / naive), lints the adjustment set, issues the
    placebo hint, and sets the verb level. Stores
    ``state["causal_claim"]`` (``{dag, id_strategy, sensitivity,
    verb_level}``). A naive design downgrades the run to analytical
    (locked §0 decision 3: downgrade, never terminal) — the recipe,
    paper type, and evidence requirements reroute, and the banner is
    recorded in meta_feedback. Placebo infeasibility is a limitation
    entry, never a block.
    """
    from agents.causal_agent import CausalAgent

    logger.info("Causal audit starting")
    log_agent_action("Orchestrator", "start_causal_audit", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("causal_audit")
    try:
        from core.run_log import emit_event as _emit

        _emit("phase_start", {"phase": "causal_audit", "agent": "CausalAgent",
                              "status": "ok", "message_human": "causal audit start",
                              "cause_class": "science"}, agent="CausalAgent")
    except Exception:
        pass
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        plan = state.get("plan") or {}
        outputs = state.get("engineer_outputs") or {}
        code_text = ""
        for output in outputs.values():
            if isinstance(output, dict) and isinstance(output.get("code"), str):
                code_text += "\n" + output["code"]
        audit = _create_agent(CausalAgent).run_audit(plan, outputs, code_text=code_text or None)
        state["causal_claim"] = audit
        try:
            from core.run_log import emit_event as _emit_gate

            _emit_gate("gate_decision", {
                "phase": "causal_audit", "agent": "CausalAgent",
                "status": "ok" if audit["passed"] else "fail",
                "message_human": f"causal audit strategy={audit['id_strategy']} verb={audit['verb_level']}",
                "gate": "causal_audit", "threshold": "identified strategy (non-naive)",
                "actual": {"strategy": audit["id_strategy"], "verb_level": audit["verb_level"],
                           "placebo_feasible": audit["placebo"].get("feasible"),
                           "n_limitations": len(audit.get("limitations") or [])},
                "cause_class": "science", "retryable": False,
                "next_action": "continue" if audit["passed"] else "downgrade to analytical"},
                agent="CausalAgent")
            for limitation in audit.get("limitations") or []:
                _emit_gate("artifact_write", {
                    "phase": "causal_audit", "agent": "CausalAgent", "status": "ok",
                    "message_human": f"causal limitation: {str(limitation)[:160]}",
                    "cause_class": "science"}, agent="CausalAgent")
        except Exception:
            pass
        for limitation in audit.get("limitations") or []:
            state.setdefault("meta_feedback", []).append("Causal limitation: " + str(limitation)[:500])
        downgrade = audit.get("downgrade")
        if downgrade:
            # Locked downgrade path (§0 decision 3): naive causal work
            # reroutes to the analytical recipe instead of terminating.
            from core.phase_registry import EVIDENCE_REQUIREMENTS, PIPELINES

            state["paper_type"] = downgrade["to"]
            state["pipeline_recipe"] = list(PIPELINES[downgrade["to"]])
            state["evidence_requirements"] = dict(EVIDENCE_REQUIREMENTS[downgrade["to"]])
            recipe_set = set(state["pipeline_recipe"])
            from core.phase_registry import PHASES as _PHASES

            state["skipped_phases"] = [
                {"phase": name, "reason": f"downgraded cause_effect → {downgrade['to']}: {downgrade['reason']}"}
                for name in _PHASES
                if name not in recipe_set and name not in ("reset", "terminal_planning_failure")
            ]
            banner = str(downgrade.get("banner") or "")
            state.setdefault("meta_feedback", []).append(banner)
            # The analytical recipe's release gate requires a verified
            # thesis — synthesize one from the topic + retrieved evidence
            # (deterministic fallback offline; never clobbers an existing
            # thesis). The experimental plan stays: its outputs become the
            # associational evidence base.
            if not state.get("thesis"):
                try:
                    from agents.thesis_agent import ThesisAgent

                    topic = state.get("selected_topic") or {}
                    evidence = (topic.get("literature_evidence")
                                if isinstance(topic, dict) else None)
                    state["thesis"] = _create_agent(ThesisAgent).build_thesis(
                        topic if isinstance(topic, dict) else {}, evidence, "analytical")
                    log_agent_action("Orchestrator", "downgrade_thesis_synthesized", {
                        "paper_type": "analytical"})
                except Exception as thesis_exc:
                    logger.error(f"Downgrade thesis failed: {thesis_exc}", exc_info=True)
            try:
                from core.run_log import emit_event as _emit_downgrade

                _emit_downgrade("downgrade", {
                    "phase": "causal_audit", "agent": "CausalAgent",
                    "status": "warn", "message_human": banner,
                    "cause_class": "science"}, agent="CausalAgent")
            except Exception:
                pass
            log_agent_action("Orchestrator", "causal_downgrade_to_analytical", {
                "reason": downgrade.get("reason")})
        log_agent_action("Orchestrator", "causal_audit_complete", {
            "strategy": audit["id_strategy"], "verb_level": audit["verb_level"],
            "downgraded": bool(downgrade)})
        if downgrade:
            # Downgraded runs proceed as associational syntheses over the
            # recorded outputs (the analytical recipe has no causal_audit or
            # independent_validation phase to advance into).
            state["current_phase"] = "writing_results"
        else:
            state["current_phase"] = _next_phase(state, "causal_audit", "independent_validation")
        state["error_count"] = 0
        try:
            from core.run_log import emit_event as _emit2

            _emit2("phase_end", {"phase": "causal_audit", "agent": "CausalAgent",
                                 "status": "ok" if audit["passed"] else "warn",
                                 "message_human": f"causal audit strategy={audit['id_strategy']}",
                                 "cause_class": "science"}, agent="CausalAgent")
        except Exception:
            pass
        return state
    except Exception as exc:
        logger.error(f"Causal audit failed: {exc}", exc_info=True)
        state["meta_feedback"].append(f"Causal audit error: {exc}")
        state["current_phase"] = _next_phase(state, "causal_audit", "independent_validation")
        return state
