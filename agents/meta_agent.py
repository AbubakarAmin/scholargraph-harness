"""
MetaAgent — observability, operator chat, light mid-run control, give-up guards.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action, parse_json_from_llm, call_llm_json
from core.llm import call_llm
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.memory import memory
from core.run_log import CrossRunMemory, get_tracker, build_run_summary, read_events, emit_event
from core.contracts import MetaChatResult, MetaDashboard
from core.state import ResearchState

logger = logging.getLogger(__name__)


def _meta_system(prompt_name: str, **slots: Any) -> Dict[str, Any]:
    """System kwarg from the v2 registry (stub-safe: {} when unavailable)."""
    try:
        from core.prompts import render_system

        system = render_system(prompt_name, **slots)
        return {"system": system} if system else {}
    except Exception:
        return {}


class MetaAgent:
    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.runtime_config = self.context.config if self.context else config
        self.feedback_memory = self.context.memory if self.context else memory
        self.client = get_llm_client()

    def evaluate_system_performance(self, state: ResearchState) -> str:
        log_agent_action("MetaAgent", "start_evaluation", {})
        try:
            metrics = self._gather_performance_metrics(state)
            trends = self._analyze_trends(state)
            dashboard = self.get_run_dashboard(state)
            feedback = self._generate_performance_feedback(metrics, trends, dashboard)
            tracker = get_tracker()
            if tracker:
                tracker.scratch("MetaAgent", "evaluation", {"metrics": metrics, "dashboard": dashboard})
            return feedback
        except Exception as e:
            return f"Meta evaluation error: {e}"

    def chat(
        self,
        message: str,
        state: Optional[ResearchState] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> MetaChatResult:
        """
        Operator ↔ Meta conversation. Answers 'what are we doing?', failures, next steps.
        May return control directives the UI/orchestrator can apply.
        """
        state = state or {}
        summary = build_run_summary(state, error=state.get("terminal_error"))
        tracker = get_tracker()
        events = read_events(limit=20, run_id=summary.get("run_id"))
        hist = history or []
        hist_txt = "\n".join(f"{h.get('role','user')}: {h.get('content','')}" for h in hist[-8:])
        prompt = f"""You are the Meta Agent of ScholarGraph, a multi-agent research harness.
Answer the human operator clearly and concretely about the CURRENT run.
If they ask what we are doing, explain the active phase and agent in plain language.
If the run failed or stalled, summarize what was tried, how, and where we stopped.
You MAY propose control actions, but only from this allowed set:
- none
- request_summary
- skip_to_writing (use only if experiments already have some results)
- request_plan_revision
- stop_run

Return JSON:
{{
  "reply": "markdown-friendly answer for the operator",
  "control": "none|request_summary|skip_to_writing|request_plan_revision|stop_run|needs_clarification",
  "control_reason": "why, or empty"
}}

CURRENT SUMMARY:
{json.dumps(summary, default=str)[:6000]}

RECENT EVENTS:
{json.dumps(events[-12:], default=str)[:3000]}

CHAT HISTORY:
{hist_txt}

OPERATOR MESSAGE:
{message}
"""
        raw = call_llm(prompt, temperature=0.3, tier="judge", max_tokens=1200,
                       **_meta_system("meta.chat"))
        parsed = parse_json_from_llm(raw) or {}
        reply = parsed.get("reply") or raw or summary.get("narrative") or "No status available yet."
        control = (parsed.get("control") or "none").strip().lower()
        if control not in {
            "none",
            "request_summary",
            "skip_to_writing",
            "request_plan_revision",
            "stop_run",
            "needs_clarification",
        }:
            control = "none"
        result = {
            "reply": reply,
            "control": control,
            "control_reason": parsed.get("control_reason") or "",
            "summary": summary,
        }
        if tracker:
            tracker.scratch("MetaAgent", "chat", {"message": message, "reply": reply, "control": control})
            emit_event(
                "meta_chat",
                {"message": message[:500], "control": control},
                run_id=tracker.run_id,
                agent="MetaAgent",
            )
        log_agent_action("MetaAgent", "chat", {"control": control})
        return result

    def apply_control(self, control: str, state: Optional[ResearchState] = None) -> ResearchState:
        """Apply a light Meta control directive to state (mutates and returns)."""
        state = state or {}
        control = (control or "none").lower()
        if control == "request_summary":
            state.setdefault("meta_feedback", []).append(build_run_summary(state).get("narrative"))
        elif control == "skip_to_writing":
            if state.get("engineer_outputs"):
                state["current_phase"] = "writing_results"
                state.setdefault("meta_feedback", []).append("Meta control: skip to writing_results")
        elif control == "request_plan_revision":
            state["current_phase"] = "planning"
            state.setdefault("plan_revision_requests", []).append({
                "reason": "operator_meta_control",
                "detail": "Operator/Meta requested plan revision",
            })
        elif control == "stop_run":
            state["should_continue"] = False
            state["current_phase"] = "complete"
            state["terminal_error"] = state.get("terminal_error") or "Stopped by Meta/operator"
        elif control == "needs_clarification":
            # Question-tool equivalent: park the run with an explicit operator
            # question instead of guessing on ambiguous evidence. try_and_report
            # runs degrade to bounded fallback rather than blocking forever.
            state.setdefault("meta_feedback", []).append(
                "Meta needs operator clarification before proceeding.")
            if str(state.get("hunt_mode") or "") in ("try_and_report", "replication"):
                state["current_phase"] = "planning"
                state["should_continue"] = True
            else:
                state["should_continue"] = False
        return state

    def get_run_dashboard(self, state: ResearchState) -> MetaDashboard:
        tracker = get_tracker()
        stats = tracker.stats if tracker else {}
        return {
            "iteration": state.get("iteration", 0),
            "phase": state.get("current_phase"),
            "rejected_topics": stats.get("rejected_topics", 0),
            "debate_rounds": stats.get("debate_rounds", 0),
            "sections_bounced": stats.get("sections_bounced", 0),
            "pivots": stats.get("pivots", 0),
            "refines": stats.get("refines", 0),
            "plan_revisions": stats.get("plan_revisions", 0),
            "hard_check_fails": stats.get("hard_check_fails", 0),
            "avg_supervisor": (
                sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
                if state.get("supervisor_scores")
                else None
            ),
            "lessons_available": len(CrossRunMemory().load(limit=100)),
        }

    def should_reset(self, state: ResearchState) -> bool:
        if self._detect_stuck_loop(state):
            return True
        if self._detect_low_quality_research(state):
            return True
        if state["iteration"] >= self.runtime_config.max_iterations:
            return True
        if self._detect_no_progress(state):
            return True
        return False

    def should_continue(self, state: ResearchState) -> bool:
        if not state["selected_topic"] or not state["plan"] or not state["draft_sections"]:
            return False
        # d5b5a1db: Abstract DRAFT_FAILED (transient 429/truncation) scored 3.0
        # -> avg 6.6, meta stopped at iteration 0 and the repairable draft never
        # got its revision pass. A DRAFT_FAILED marker is infra, not science:
        # force one more iteration while budget remains.
        try:
            _drafts = state.get("draft_sections") or {}
            if any(isinstance(v, str) and "DRAFT_FAILED" in v for v in _drafts.values()):
                if int(state.get("iteration") or 0) < int(self.runtime_config.max_iterations or 10):
                    return True
        except Exception:
            pass
        if self._scores_improving(state) or self._close_to_threshold(state):
            return True
        return False

    def validate_failure_claim(self, agent: str, claim: str, artifact: Any) -> bool:
        """
        Give-up early guard (8.5): bare 'not feasible' without artifact is suspicious.
        """
        if not artifact:
            log_agent_action("MetaAgent", "suspicious_give_up", {"agent": agent, "claim": claim})
            return False
        if isinstance(artifact, dict) and artifact.get("suspicious_bare_claim"):
            return False
        # Require concrete evidence keys
        if isinstance(artifact, dict):
            if artifact.get("error") or artifact.get("traceback") or artifact.get("decision_log"):
                return True
        if isinstance(artifact, str) and len(artifact) > 40:
            return True
        return False

    def _gather_performance_metrics(self, state) -> Dict[str, Any]:
        metrics = {
            "iteration": state["iteration"],
            "topics_discovered": len(state["topics"]),
            "debates_completed": len(state["debate_results"]),
            "sections_written": len(state["draft_sections"]),
            "experiments_run": len(state["engineer_outputs"]),
            "average_score": 0.0,
            "best_score": 0.0,
            "has_topic": state["selected_topic"] is not None,
            "has_plan": state["plan"] is not None,
            "has_content": len(state["draft_sections"]) > 0,
            "has_experiments": len(state["engineer_outputs"]) > 0,
        }
        if state["supervisor_scores"]:
            scores = list(state["supervisor_scores"].values())
            metrics["average_score"] = sum(scores) / len(scores)
            metrics["best_score"] = max(scores)
        return metrics

    def _analyze_trends(self, state) -> Dict[str, Any]:
        trends = {
            "score_trend": "stable",
            "stuck_pattern": False,
            "content_growth": "stable",
        }
        recent_feedback = self.feedback_memory.get_feedback_signals(limit=10)
        if len(recent_feedback) >= 3:
            scores = [e["score"] for e in recent_feedback if e.get("score") is not None]
            recent_avg = sum(scores[-3:]) / 3
            older = scores[:-3]
            older_avg = sum(older) / len(older) if older else scores[0]
            if recent_avg > older_avg + 0.5:
                trends["score_trend"] = "improving"
            elif recent_avg < older_avg - 0.5:
                trends["score_trend"] = "declining"
            if len(set(scores[-3:])) <= 1:
                trends["stuck_pattern"] = True
        return trends

    def _generate_performance_feedback(self, metrics, trends, dashboard) -> str:
        prompt = f"""
Analyze research system performance. Prefer concrete next actions.
Metrics: {json.dumps(metrics)}
Trends: {json.dumps(trends)}
Dashboard: {json.dumps(dashboard)}
JSON: {{"system_health": "good|medium|poor", "recommendations": [], "next_steps": "continue|reset|refine"}}
"""
        parsed = call_llm_json(prompt, temperature=0.3, tier="cheap", attempts=2, call_fn=call_llm,
                                **_meta_system("meta.feedback")) or {}
        recs = parsed.get("recommendations") or []
        return (
            f"Health: {parsed.get('system_health', 'unknown')}\n"
            f"Next: {parsed.get('next_steps', 'continue')}\n"
            + "\n".join(f"- {r}" for r in recs)
            + f"\nDashboard: pivots={dashboard.get('pivots')} hard_fails={dashboard.get('hard_check_fails')}"
        )

    def _detect_stuck_loop(self, state) -> bool:
        recent = self.feedback_memory.get_feedback_signals(limit=5)
        if len(recent) >= 5:
            scores = [e["score"] for e in recent if e.get("score") is not None]
            if len(scores) >= 5 and all(s < 3.0 for s in scores):
                return True
        return False

    def _detect_low_quality_research(self, state) -> bool:
        if not state["selected_topic"]:
            return True
        if state["selected_topic"].get("feasibility", 5) < 3:
            return True
        if state["engineer_outputs"] and not any(
            o.get("success") for o in state["engineer_outputs"].values() if isinstance(o, dict)
        ):
            outcomes = [o.get("outcome") for o in state["engineer_outputs"].values() if isinstance(o, dict)]
            if outcomes and all(outcome in {"negative", "inconclusive"} for outcome in outcomes):
                # A contract-bound, honestly reported negative result is not a system failure.
                return False
            # Check give-up artifacts
            for o in state["engineer_outputs"].values():
                if isinstance(o, dict) and not self.validate_failure_claim(
                    "Engineer", o.get("error", "failed"), o.get("failure_artifact") or o
                ):
                    return True
            return True
        return False

    def _detect_no_progress(self, state) -> bool:
        if not state["draft_sections"]:
            return True
        total = sum(len(c) for c in state["draft_sections"].values())
        if total < 200:
            return True
        if state["supervisor_scores"]:
            avg = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
            if avg < 2.0:
                return True
        return False

    def _scores_improving(self, state) -> bool:
        recent = self.feedback_memory.get_feedback_signals(limit=5)
        if len(recent) >= 3:
            scores = [e["score"] for e in recent if e.get("score") is not None]
            if len(scores) < 3:
                return False
            return sum(scores[-3:]) / 3 > sum(scores[:-3]) / max(len(scores) - 3, 1)
        return False

    def _close_to_threshold(self, state) -> bool:
        if not state["supervisor_scores"]:
            return False
        avg = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
        return avg >= (self.runtime_config.supervisor_threshold - 1.0)

    def get_system_recommendations(self, state: ResearchState) -> List[str]:
        recs = []
        if not state["selected_topic"]:
            recs.append("Discover topics")
        if not state["plan"]:
            recs.append("Create plan with falsifiable predictions")
        if state.get("supervisor_scores"):
            avg = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
            if avg < self.runtime_config.supervisor_threshold:
                recs.append(f"Quality {avg:.1f} < {self.runtime_config.supervisor_threshold}")
        return recs
