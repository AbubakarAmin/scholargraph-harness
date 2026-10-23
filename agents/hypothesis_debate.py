"""
Hypothesis Debate — multi-round adversarial debate with ensemble judging + Elo.
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action, parse_json_from_llm, is_degenerate_llm_output, call_llm_json
from core.llm import call_llm
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.memory import memory
from core.run_log import get_tracker
from core.capabilities import SANDBOX_CAPABILITY_MANIFEST, check_plan_feasibility
from core.evidence_synthesis import validate_topic_admission

logger = logging.getLogger(__name__)


def _debate_system(prompt_name: str, **slots: Any) -> Dict[str, Any]:
    """System kwarg from the v2 registry (stub-safe: {} when unavailable)."""
    try:
        from core.prompts import render_system

        system = render_system(prompt_name, **slots)
        return {"system": system} if system else {}
    except Exception:
        return {}

REVIEWER_CHECKLIST = [
    "soundness",
    "significance",
    "reproducibility",
    "ethics",
    "novelty",
    "feasibility",
]


def _coerce_int(value: Any, default: int = 0) -> int:
    """int() that treats missing/None as default (dict.get default does not)."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class DebateResult:
    topic: str
    proposer_argument: str
    challenger_argument: str
    moderator_decision: str
    score: float
    passed: bool
    reasoning: str
    # Steelman mode (§4.4, M1): strongest counter-reading + concession test.
    # Empty unless the debate ran with mode="steelman".
    steelman: str = ""
    concession_test: str = ""
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    objections: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_objections: List[str] = field(default_factory=list)
    ensemble_scores: List[float] = field(default_factory=list)
    elo_delta: float = 0.0
    # True when the debate never substantively happened: challenger output
    # was garbled AND no template-fallback objections were available, so the
    # FAIL reflects infrastructure, not hypothesis quality. Orchestration
    # must not promote such topics (not even via the exploratory path).
    infra_failure: bool = False


class EloStore:
    """Backward-compatible Elo store with observation counts and shrinkage.

    Legacy files that store `{kind: float}` or partial dicts (missing/null
    observations) are upgraded on read and rewritten once so new fields are
    never read as None. Selection uses the shrunk rating; raw_rating is
    preserved for audit.
    """

    def __init__(self, path: Optional[str] = None, context: Optional[RunContext] = None):
        runtime_config = (context or get_active_context()).config if (context or get_active_context()) else config
        self.path = Path(path or runtime_config.elo_ratings_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prior = float(getattr(runtime_config, "elo_prior_rating", 1500.0))
        self.shrinkage_k = float(getattr(runtime_config, "elo_shrinkage_k", 8.0))
        self.min_observations = _coerce_int(getattr(runtime_config, "elo_min_observations", 5), 5)
        self.records: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.records = self._upgrade(raw)
                if self._needs_migration(raw):
                    self._persist()
            except Exception as e:
                logger.warning("EloStore corrupted, resetting: %s", e)
                self.records = {}

    @staticmethod
    def _needs_migration(raw: Dict[str, Any]) -> bool:
        """True when on-disk shape is legacy flat or missing/null observation fields."""
        for value in (raw or {}).values():
            if not isinstance(value, dict):
                return True
            if "rating" not in value:
                return True
            if "observations" not in value and "observation_count" not in value:
                return True
            if value.get("observations") is None and value.get("observation_count") is None:
                return True
            if "raw_rating" not in value or "shrinkage_k" not in value:
                return True
        return False

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")

    def _upgrade(self, raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        upgraded: Dict[str, Dict[str, Any]] = {}
        for key, value in (raw or {}).items():
            if isinstance(value, dict) and ("rating" in value or "raw_rating" in value):
                rating = _coerce_float(value.get("rating", value.get("raw_rating")), self.prior)
                # Accept both observations and legacy observation_count aliases.
                observations = _coerce_int(
                    value.get("observations", value.get("observation_count")),
                    0,
                )
                record = {
                    "rating": rating,
                    "raw_rating": _coerce_float(value.get("raw_rating", rating), rating),
                    "observations": observations,
                    "prior": _coerce_float(value.get("prior"), self.prior),
                    "shrinkage_k": _coerce_float(value.get("shrinkage_k"), self.shrinkage_k),
                    "updated_at": value.get("updated_at"),
                }
            else:
                # Legacy flat float
                rating = _coerce_float(value, self.prior)
                record = {
                    "rating": rating,
                    "raw_rating": rating,
                    "observations": 0,
                    "prior": self.prior,
                    "shrinkage_k": self.shrinkage_k,
                    "updated_at": None,
                }
            upgraded[key] = record
        return upgraded

    @property
    def ratings(self) -> Dict[str, float]:
        """Shrunk ratings used for selection (legacy-compatible mapping)."""
        return {key: float(rec["rating"]) for key, rec in self.records.items()}

    def _shrink(self, raw_rating: float, observations: int, prior: float, k: float) -> float:
        # Pull toward prior until enough observations accumulate.
        weight = observations / (observations + k) if (observations + k) else 0.0
        return prior + (raw_rating - prior) * weight

    def get_record(self, key: str) -> Dict[str, Any]:
        if key in self.records:
            return dict(self.records[key])
        kind = hypothesis_kind(key)
        if kind in self.records:
            return dict(self.records[kind])
        return {
            "rating": self.prior,
            "raw_rating": self.prior,
            "observations": 0,
            "prior": self.prior,
            "shrinkage_k": self.shrinkage_k,
            "updated_at": None,
        }

    def get(self, key: str, default: Optional[float] = None) -> float:
        if key in self.records:
            return float(self.records[key]["rating"])
        kind = hypothesis_kind(key)
        if kind in self.records:
            return float(self.records[kind]["rating"])
        return float(self.prior if default is None else default)

    def observations(self, key: str) -> int:
        return _coerce_int(self.get_record(key).get("observations"), 0)

    def under_observed_kinds(self, kinds: List[str]) -> List[str]:
        return [k for k in kinds if self.observations(k) < self.min_observations]

    def update(self, hypothesis_key: str, score: float, passed: bool) -> float:
        """Elo vs a fixed reviewer bar; persist raw and shrunk ratings separately."""
        kind = hypothesis_kind(hypothesis_key)
        record = self.get_record(kind)
        ra = _coerce_float(record.get("raw_rating"), self.prior)
        rb = self.prior
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        outcome = 1.0 if passed else (0.5 if score >= 6 else 0.0)
        k_factor = 32
        delta = k_factor * (outcome - ea)
        new_raw = ra + delta
        observations = _coerce_int(record.get("observations"), 0) + 1
        prior = _coerce_float(record.get("prior"), self.prior)
        shrink_k = _coerce_float(record.get("shrinkage_k"), self.shrinkage_k)
        # Until min observations, force stronger shrinkage toward prior.
        effective_k = (
            max(shrink_k, self.min_observations - observations + 1)
            if observations < self.min_observations
            else shrink_k
        )
        shrunk = self._shrink(new_raw, observations, prior, effective_k)
        self.records[kind] = {
            "rating": shrunk,
            "raw_rating": new_raw,
            "observations": observations,
            "prior": prior,
            "shrinkage_k": shrink_k,
            "updated_at": datetime.now().isoformat(),
        }
        self._persist()
        logger.info("Elo updated: kind=%s score=%.1f passed=%s raw_rating=%.1f->%.1f shrunk=%.1f delta=%.1f observations=%d",
                     kind, score, passed, ra, new_raw, shrunk, delta, observations)
        return delta

    def record_strategy_outcome(self, strategy: str, outcome_status: str) -> float:
        """Feature 8: record seed-strategy outcome under a strategy:<name> namespace key,
        reusing the existing rating/shrinkage record shape. Returns Elo delta.

        outcome_status uses debate-derived labels: "debate_pass", "debate_weak",
        "debate_fail". These are NOT experiment-level outcomes — kind-Elo uses
        the same semantics (there is no post-experiment Elo update in the system).
        """
        key = f"strategy:{strategy}"
        record = self.get_record(key)
        ra = _coerce_float(record.get("raw_rating"), self.prior)
        rb = self.prior
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        outcome_map = {"debate_pass": 1.0, "debate_weak": 0.5, "debate_fail": 0.0}
        outcome = outcome_map.get(outcome_status, 0.5)
        k_factor = 32
        delta = k_factor * (outcome - ea)
        new_raw = ra + delta
        observations = _coerce_int(record.get("observations"), 0) + 1
        prior = _coerce_float(record.get("prior"), self.prior)
        shrink_k = _coerce_float(record.get("shrinkage_k"), self.shrinkage_k)
        effective_k = (
            max(shrink_k, self.min_observations - observations + 1)
            if observations < self.min_observations
            else shrink_k
        )
        shrunk = self._shrink(new_raw, observations, prior, effective_k)
        self.records[key] = {
            "rating": shrunk,
            "raw_rating": new_raw,
            "observations": observations,
            "prior": prior,
            "shrinkage_k": shrink_k,
            "updated_at": datetime.now().isoformat(),
        }
        self._persist()
        logger.info("Elo strategy outcome: strategy=%s outcome=%s delta=%.1f new_rating=%.1f", strategy, outcome_status, delta, shrunk)
        return delta


def hypothesis_kind(title: str) -> str:
    """Coarse bucket for Elo (kinds of hypotheses)."""
    t = title.lower()
    for kind in ("attention", "graph", "diffusion", "reinforcement", "federated", "llm", "vision", "nlp"):
        if kind in t:
            return kind
    return "general"


def _as_dict(value: Any) -> Dict[str, Any]:
    """Return value when it is a dict, else {} — guards .get against lists."""
    return value if isinstance(value, dict) else {}


def _normalize_objection_payload(parsed: Any) -> Dict[str, Any]:
    """Coerce LLM objection payloads into the canonical envelope shape.

    Models frequently return a bare array of objections or a single objection
    object instead of the requested {"objections": [...]}. Both are valid
    content in the wrong shape — coerce instead of crashing on ``.get``.
    """
    if isinstance(parsed, list):
        parsed = {"objections": parsed}
    if not isinstance(parsed, dict):
        return {}
    raw_objections = parsed.get("objections")
    if isinstance(raw_objections, dict):
        parsed["objections"] = [raw_objections]
    return parsed


def _ensure_objections_are_dicts(objections: List[Any]) -> List[Dict[str, Any]]:
    """Defensive normalizer: ensure every item in an objections list is a dict.
    LLMs occasionally return nested lists or bare strings inside objection lists.
    This prevents 'list object has no attribute get' crashes downstream."""
    out: List[Dict[str, Any]] = []
    for item in objections:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, list):
            # Nested list — flatten: take first dict found, or wrap
            for sub in item:
                if isinstance(sub, dict):
                    out.append(sub)
                    break
            else:
                out.append({"objection": str(item), "severity": 3, "status": "unresolved"})
        elif isinstance(item, str):
            out.append({"objection": item, "severity": 3, "status": "unresolved"})
    return out


def _classify_infra_error(exc: Exception) -> str:
    """Stable error_code for LLM infra failures (P0-1).

    Lets orchestration distinguish truncation/transport/model-404 without
    parsing free-text messages, and keeps run_events queryable.
    """
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "truncated with no content" in msg or "finish_reason=length" in msg or "truncated" in msg:
        return "truncation_empty"
    if "blocked" in msg or "content_filter" in msg or "safety" in msg:
        return "blocked"
    if "404" in msg or "not_found" in msg or "no longer available" in msg or "invalid model" in msg:
        return "model_404"
    if "getaddrinfo" in msg or "connection" in msg or "timeout" in msg or "timed out" in msg or "readerror" in msg or "connecterror" in msg:
        return "transport"
    if "429" in msg or "rate" in msg or "overloaded" in msg or "unavailable" in msg:
        return "throttled"
    return "unknown_infra"


class ProposerAgent:
    # Lab identity (plan-mode subagent): broker-only tools, never code/data.
    manifest_name: str = "ProposerAgent"
    allowed_tools: List[str] = ["search_literature", "citation_graph",
                                "evidence_synthesize", "memory_recall"]

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        self.vector_memory = self.context.memory if self.context else memory
        # P0-1: proposer infra state, reset per debate. When True the debate
        # never substantively happened (round-1 raised) — orchestration must
        # treat the result as infra, never as science.
        self._proposer_invalid = False
        self._proposer_error_code = ""

    def _prior_objection_tags(self) -> List[Dict[str, Any]]:
        """Structured objection tags from prior debates so round-1 arguments
        can preempt historically recurring failure modes."""
        rows = self.vector_memory.get_prompt_context(namespace="debate_transcripts", k=8)
        tags = []
        for row in rows:
            signal = row.get("signal") or {}
            tags.append({
                "objection_type": signal.get("objection_type") or signal.get("objection_types"),
                "severity": signal.get("severity"),
                "resolution_status": signal.get("resolution_status"),
            })
        return tags

    def _evidence_context(self, topic: Dict[str, Any], max_papers: int = 5, snippet_chars: int = 350) -> str:
        """Render the topic's retrieved literature evidence for grounded arguing."""
        papers = topic.get("literature_evidence") or []
        slim = []
        for paper in papers[:max_papers]:
            if not isinstance(paper, dict):
                continue
            slim.append({
                "title": paper.get("title", ""),
                "abstract": (paper.get("abstract") or "")[:snippet_chars],
            })
        slim = [p for p in slim if p["title"] or p["abstract"]]
        return json.dumps(slim, default=str)[:4000] if slim else "none retrieved"

    def build_argument(self, topic: Dict[str, Any]) -> str:
        logger.info("Proposer generating argument for: %s", topic.get("title", "<untitled>"))
        # Reset per-debate infra state (same agent instance debates many topics).
        self._proposer_invalid = False
        self._proposer_error_code = ""
        prior_tags = self._prior_objection_tags()
        structured_hyp = _as_dict(topic.get("structured_hypothesis"))
        prompt = f"""
You are a research proposer. Build a compelling, realistic argument.
Topic: {topic.get('title')}
Description: {topic.get('description')}
Rationale: {topic.get('rationale', 'N/A')}
Impact: {topic.get('impact', 'N/A')}
Feasibility: {topic.get('feasibility', 5)}/10

Structured Hypothesis Contract (authoritative):
{json.dumps(topic.get('structured_hypothesis') or {}, indent=2, default=str)[:3000]}

Retrieved literature evidence (ground every novelty/gap claim in this):
{self._evidence_context(topic)}

Historically recurring objection tags from prior debates (preempt these proactively):
{json.dumps(self._prior_objection_tags(), default=str)[:1200]}

Rules:
- Ground every novelty claim in the retrieved evidence above; explicitly name the
  closest prior work and state what remains untested.
- Address the recurring objection types listed above before they are raised.
- Do not claim evidence you do not have; describe what will be measured and how.
Include: hypothesis, theory, evidence, novelty, methodology, expected outcomes, falsifiable prediction.
"""
        arg = ""
        for attempt in range(2):
            try:
                raw = call_llm(prompt, temperature=0.7, tier="strong",
                               **_debate_system("debate.proposer",
                                                title=str(topic.get("title", ""))))
            except Exception as exc:
                # P0-1 (c22341c4): LLMUnavailable(truncated with no content)
                # used to propagate and terminal-kill the run. Classify as
                # infra and fail soft — the debate never happened, so the
                # result must carry infra_failure, never a science FAIL.
                code = _classify_infra_error(exc)
                self._proposer_invalid = True
                self._proposer_error_code = code
                log_agent_action("ProposerAgent", "proposer_infra_failure", {
                    "topic": topic.get("title"),
                    "attempt": attempt + 1,
                    "error_code": code,
                    "error": str(exc)[:300],
                })
                logger.warning("Proposer infra failure for %r (code=%s): %s",
                               topic.get("title", "")[:160], code, str(exc)[:160])
                return ""
            if not is_degenerate_llm_output(raw, min_chars=30):
                arg = raw
                break
            log_agent_action("ProposerAgent", "degenerate_argument_retry", {"attempt": attempt + 1})

        if not arg:
            arg = "PROPOSER_ARGUMENT_INVALID"
            log_agent_action("ProposerAgent", "argument_failed_closed", {"topic": topic.get("title")})
        else:
            log_agent_action("ProposerAgent", "built_argument", {"len": len(arg)})
        return arg

    def respond_to_objections(
        self,
        topic: Dict[str, Any],
        prior_argument: str,
        objections: List[Dict[str, Any]],
    ) -> str:
        """Respond to EACH specific objection (not a global rebuttal)."""
        logger.info("Proposer responding to %d objections for: %s", len(objections), topic.get("title", "<untitled>"))
        obj_text = "\n".join(
            f"- [{o.get('criterion')}] {o.get('objection')}" for o in objections
        )
        prompt = f"""
Respond point-by-point to each Challenger objection. Do not give a vague global rebuttal.
Topic: {topic.get('title')}
Your prior argument: {prior_argument[:3000]}
Objections:
{obj_text}

Format:
For each objection: OBJECTION: ... RESPONSE: ... CONCESSION (if any): ...
"""
        response = ""
        for attempt in range(2):
            try:
                raw = call_llm(prompt, temperature=0.5, tier="strong",
                               **_debate_system("debate.proposer_respond",
                                                title=str(topic.get("title", ""))))
            except Exception as exc:
                # P0-1: round-2 proposer failure is not fatal — round-1 already
                # substantively happened. Return a stub so the moderator can
                # still judge round-1; do NOT set _proposer_invalid.
                code = _classify_infra_error(exc)
                log_agent_action("ProposerAgent", "proposer_response_infra", {
                    "topic": topic.get("title"),
                    "attempt": attempt + 1,
                    "error_code": code,
                    "error": str(exc)[:300],
                })
                logger.warning("Proposer round-2 infra for %r (code=%s)",
                               topic.get("title", "")[:160], code)
                return "PROPOSER_RESPONSE_INVALID"
            if not is_degenerate_llm_output(raw, min_chars=20):
                response = raw
                break
            log_agent_action("ProposerAgent", "degenerate_response_retry", {"attempt": attempt + 1})

        return response or "PROPOSER_RESPONSE_INVALID"


class ChallengerAgent:
    # Lab identity (plan-mode subagent): broker-only tools, never code/data.
    manifest_name: str = "ChallengerAgent"
    allowed_tools: List[str] = ["search_literature", "citation_graph",
                                "contradiction_scan", "memory_recall"]

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        self.vector_memory = self.context.memory if self.context else memory

    def _prior_objection_tags(self) -> List[Dict[str, Any]]:
        """Retrieve only structured objection tags — never prior argument prose."""
        mem = getattr(self, "vector_memory", None) or (
            self.context.memory if getattr(self, "context", None) else memory
        )
        rows = mem.get_prompt_context(namespace="debate_transcripts", k=8)
        tags = []
        for row in rows:
            signal = row.get("signal") or {}
            tags.append({
                "objection_type": signal.get("objection_type") or signal.get("objection_types"),
                "severity": signal.get("severity"),
                "resolution_status": signal.get("resolution_status"),
                "run_id": row.get("run_id"),
                "outcome_status": row.get("outcome_status"),
            })
        return tags

    def build_rebuttal(self, topic: Dict[str, Any], proposer_argument: str, mode: str = "standard") -> str:
        """Checklist-grounded critique — attacks scientific validity, confounders, baselines, and feasibility.

        mode="steelman" (§4.4, argumentative): the challenger must produce the
        strongest counterargument plus a concession test instead of a plain
        checklist audit. Unresolved severity >= 4 still blocks downstream.
        """
        logger.info("Challenger generating rebuttal for: %s (mode=%s)", topic.get("title", "<untitled>"), mode)
        rating = float(topic.get("elo_rating", EloStore(context=self.context).get(hypothesis_kind(topic.get("title", "")))))
        kind = topic.get("hypothesis_kind") or hypothesis_kind(topic.get("title", ""))
        history_note = (
            f"This is a {kind} hypothesis with a low historical Elo ({rating:.0f}). "
            "Scrutinize the failure modes that have historically made this kind fail."
            if rating < 1450 else
            f"This is a {kind} hypothesis with historical Elo {rating:.0f}; still independently verify its claims."
        )
        prior_tags = self._prior_objection_tags()
        structured_hyp = _as_dict(topic.get("structured_hypothesis"))
        if mode == "steelman":
            role_block = """You are a steelman reviewer. Your goal is to state the STRONGEST possible case
AGAINST this position — stronger and fairer than its critics have made it —
then give a concession test: the single observation that would force the
position to concede."""
            scrutiny_block = """Scrutinize as a steelman:
1. Strongest counterargument (the best reasons the position is wrong, stated at full strength)
2. Concession test (what observation would force concession, stated concretely)
3. Supporting objections in the standard criteria (confounder|baseline|falsifiability|soundness|feasibility|evaluation|statistical|novelty)"""
            return_block = """Return JSON:
{
  "steelman_argument": "strongest case against the position",
  "concession_test": "concrete observation that would force concession",
  "objections": [
    {
      "criterion": "confounder|baseline|falsifiability|soundness|feasibility|evaluation|statistical|novelty",
      "objection": "...",
      "severity": 1-5,
      "status": "unresolved",
      "required_resolution": "...",
      "source": "steelman_audit"
    }
  ],
  "summary_rebuttal": "..."
}"""
        else:
            role_block = """You are an adversarial scientific reviewer. Your goal is to find the strongest legitimate reasons this hypothesis may be flawed, unfalsifiable, confounded, or unexecutable."""
            scrutiny_block = """Scrutinize:
1. Hypothesis clarity & falsifiability (is the falsification condition concrete and measurable?)
2. Confounders & competing explanations (could an alternative mechanism produce the same result?)
3. Baseline adequacy (are meaningful comparison baselines included?)
4. Evaluation & metric adequacy (can the chosen metrics detect the proposed effect?)
5. Statistical adequacy & power (sample size, seed count, statistical test)
6. Feasibility vs Sandbox Manifest and Local Dataset Catalog (local datasets only, no downloads, CPU limits)
7. Minimum Viable Experiment sufficiency (does the experiment distinguish the hypothesis from competing explanations?)"""
            return_block = """Return JSON:
{
  "objections": [
    {
      "criterion": "confounder|baseline|falsifiability|soundness|feasibility|evaluation|statistical|novelty",
      "objection": "...",
      "severity": 1-5,
      "status": "unresolved",
      "required_resolution": "...",
      "source": "challenger_audit"
    }
  ],
  "summary_rebuttal": "..."
}"""
        prompt = f"""
{role_block}

Topic: {topic.get('title')}
Structured Hypothesis Contract:
{json.dumps(structured_hyp, indent=2, default=str)[:3000]}

Proposer Argument:
{proposer_argument[:3000]}

Historical signal: {history_note}
Prior objection tags only (never reuse prior prose): {json.dumps(prior_tags)[:1500]}
Sandbox capability manifest: {json.dumps(SANDBOX_CAPABILITY_MANIFEST.as_dict(), sort_keys=True)}

{scrutiny_block}

{return_block}
"""
        self._challenger_invalid = False
        self._challenger_fallback_used = False
        self._last_steelman = ""
        self._last_concession_test = ""
        objections: List[Dict[str, Any]] = []
        summary = ""
        raw = ""
        attempt_state = {"n": 0}

        def _rebuttal_call(current_prompt: str, **kwargs: Any) -> str:
            nonlocal raw
            try:
                raw = call_llm(current_prompt, **kwargs,
                               **_debate_system("debate.challenger", mode=mode,
                                                title=str(topic.get("title", ""))))
            except TypeError:
                raw = call_llm(current_prompt, **kwargs)
            attempt_state["n"] += 1
            if is_degenerate_llm_output(raw, min_chars=20):
                log_agent_action("ChallengerAgent", "degenerate_rebuttal", {
                    "attempt": attempt_state["n"],
                    "preview": str(raw)[:120],
                })
                return ""
            return raw

        # Self-correcting parse: on malformed output the model is re-asked with
        # the parse error and its own offending excerpt (same protocol as the
        # screener/followup paths) instead of a parse-blind identical re-roll.
        parsed = call_llm_json(
            prompt,
            temperature=0.5,
            tier="strong",
            attempts=2,
            max_tokens=16384,
            call_fn=_rebuttal_call,
        )
        # Fallback: if structured parse failed, try raw JSON extraction on the
        # last LLM response (reasoning models often emit valid JSON after
        # chain-of-thought that call_llm_json's stricter path misses).
        if parsed is None and raw:
            parsed = parse_json_from_llm(raw)
        parsed = _normalize_objection_payload(parsed)
        raw_objections = parsed.get("objections")
        if isinstance(raw_objections, list) and len(raw_objections) > 0:
            for obj in raw_objections:
                if isinstance(obj, dict):
                    obj.setdefault("status", "unresolved")
                    obj.setdefault("source", "steelman_audit" if mode == "steelman" else "challenger_audit")
                    objections.append(obj)
            steelman_text = str(parsed.get("steelman_argument") or "")
            concession_text = str(parsed.get("concession_test") or "")
            self._last_steelman = steelman_text
            self._last_concession_test = concession_text
            summary = str(parsed.get("summary_rebuttal") or raw)
            if mode == "steelman":
                headed = []
                if steelman_text:
                    headed.append("STEELMAN: " + steelman_text)
                if concession_text:
                    headed.append("CONCESSION TEST: " + concession_text)
                if headed:
                    summary = "\n".join(headed) + "\n" + summary
        else:
            # Fail-soft (v5, run 60271359): never hand downstream a 25-char
            # guard stub. Emit deterministic template objections grounded in
            # the topic so debate rounds + moderator still have substance.
            # _challenger_invalid stays True for audit; _challenger_fallback_used
            # tells conduct_debate NOT to cap score / force FAIL / skip rounds.
            self._challenger_invalid = True
            self._challenger_fallback_used = True
            t_title = str(topic.get("title", "the hypothesis"))[:160]
            objections = [
                {
                    "criterion": "falsifiability",
                    "objection": (
                        f"Template fallback critique for '{t_title}': the falsification condition "
                        "is not yet tied to a preregistered numeric threshold (effect size, metric delta, "
                        "p-value) with controlled seeds, so a null result cannot cleanly falsify the claim."
                    ),
                    "severity": 4,
                    "status": "unresolved",
                    "required_resolution": "State a numeric falsification threshold, metric, seeds>=3, and statistical test.",
                    "source": "challenger_template_fallback",
                },
                {
                    "criterion": "baseline",
                    "objection": (
                        f"Template fallback critique for '{t_title}': baseline adequacy is unverified — "
                        "the proposal must name concrete comparison baselines (e.g. logistic_regression, "
                        "majority_class) with identical evaluation metrics, otherwise the effect is confounded."
                    ),
                    "severity": 3,
                    "status": "unresolved",
                    "required_resolution": "Add named baselines with identical metrics and seeds.",
                    "source": "challenger_template_fallback",
                },
            ]
            summary = (
                f"Template fallback rebuttal for '{t_title}': LLM challenger output was empty/garbled after "
                "self-correcting retry, so deterministic checklist objections (falsifiability, baseline adequacy) "
                "are raised instead of a guard stub. Proposer should address numeric falsification thresholds, "
                "named baselines, seed count, and statistical test in the next round."
            )
            if mode == "steelman":
                self._last_steelman = (
                    f"Steelman fallback for '{t_title}': the strongest counter-reading is that the "
                    "position overstates what its cited evidence establishes, because none of the "
                    "cited sources was designed to test the claim as stated."
                )
                self._last_concession_test = (
                    "Concession test: produce one cited source whose stated result directly entails "
                    "the claim, or concede the scope limit."
                )
                summary = (
                    "STEELMAN: " + self._last_steelman + "\n"
                    "CONCESSION TEST: " + self._last_concession_test + "\n" + summary
                )
            log_agent_action("ChallengerAgent", "rebuttal_invalid_after_retry", {
                "raw_preview": str(raw)[:200],
                "parsed_shape": type(parsed).__name__,
            })
            log_agent_action("ChallengerAgent", "rebuttal_template_fallback", {
                "n_objections": len(objections),
                "topic": t_title[:80],
            })

        feasibility_errors = check_plan_feasibility(
            {"methodology": topic.get("description", ""), "experiments": [{"dataset": {"name": topic.get("dataset_plan", "")}}]},
            SANDBOX_CAPABILITY_MANIFEST,
        )
        objections.extend({
            "criterion": "feasibility",
            "objection": reason,
            "severity": 5,
            "status": "unresolved",
            "required_resolution": "Rescope experiment within sandbox capabilities",
            "source": "capability_manifest",
        } for reason in feasibility_errors)

        # Minimum viable experiment validation
        mve_check = self.validate_minimum_experiment(topic)
        if not mve_check.get("is_discriminative", True) or not mve_check.get("is_executable", True):
            for note in mve_check.get("validation_notes", []):
                objections.append({
                    "criterion": "soundness",
                    "objection": f"Minimum viable experiment flaw: {note}",
                    "severity": 4,
                    "status": "unresolved",
                    "required_resolution": "Refine MVE to be executable and discriminative",
                    "source": "minimum_experiment_check",
                })

        self._last_objections = objections
        log_agent_action("ChallengerAgent", "built_rebuttal", {
            "n_objections": len(objections),
            "invalid": bool(self._challenger_invalid),
            "mode": mode,
            "steelman_present": bool(getattr(self, "_last_steelman", "")),
        })
        return summary if isinstance(summary, str) else json.dumps(parsed)

    def validate_minimum_experiment(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that the minimum viable experiment is both executable and scientifically discriminative."""
        structured_hyp = _as_dict(topic.get("structured_hypothesis"))
        mve = structured_hyp.get("minimum_viable_experiment") or {}
        notes: List[str] = []
        is_executable = True
        is_discriminative = True

        if not mve:
            return {
                "is_executable": True,
                "is_discriminative": True,
                "validation_notes": [],
            }

        # Executability check against local catalog
        from core.datasets import list_datasets
        catalog_names = {d["name"] for d in list_datasets()}
        dataset_name = mve.get("dataset", "bundled_synthetic")
        if dataset_name not in catalog_names and "synthetic" not in dataset_name.lower():
            is_executable = False
            notes.append(f"Dataset '{dataset_name}' not found in local catalog and not synthetic")

        # Discriminative power check
        competing = structured_hyp.get("competing_explanations") or []
        falsification = structured_hyp.get("falsification_condition") or mve.get("falsification_test")
        if not falsification:
            is_discriminative = False
            notes.append("No defined falsification test in minimum viable experiment")
        if competing and not mve.get("baseline") and not mve.get("models"):
            is_discriminative = False
            notes.append("MVE lacks baselines to distinguish hypothesis from competing explanations")

        return {
            "is_executable": is_executable,
            "is_discriminative": is_discriminative,
            "validation_notes": notes,
        }

    def followup_objections(
        self,
        topic: Dict[str, Any],
        proposer_response: str,
        prior_objections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Track objections across rounds with invariant: no objection may silently disappear."""
        logger.info("Challenger evaluating followup objections for: %s (%d prior objections)",
                     topic.get("title", "<untitled>"), len(prior_objections))
        prompt = f"""
Given the proposer's point-by-point responses, evaluate each prior objection.
Prior objections:
{json.dumps(prior_objections, indent=2)[:3000]}

Proposer responses:
{proposer_response[:3000]}

For EACH prior objection, decide if it is resolved or remains unresolved.
Return JSON:
{{
  "objections": [
    {{
      "criterion": "...",
      "objection": "...",
      "severity": 3,
      "status": "resolved|unresolved",
      "resolution": "...",
      "required_resolution": "..."
    }}
  ]
}}
"""
        for attempt in range(2):
            parsed = call_llm_json(prompt, temperature=0.3, tier="judge", attempts=2, call_fn=call_llm,
                                   **_debate_system("debate.followup"))
            # A bare array of objections is valid content in the wrong envelope.
            if isinstance(parsed, list):
                parsed = {"objections": parsed}
            if not isinstance(parsed, dict):
                log_agent_action("ChallengerAgent", "degenerate_followup", {"attempt": attempt + 1})
                continue
            raw_objs = parsed.get("objections")
            if isinstance(raw_objs, list) and len(raw_objs) > 0:
                # Merge status onto prior objections to ensure no objection is dropped
                updated = []
                seen_texts = set()
                for obj in raw_objs:
                    if isinstance(obj, dict):
                        text = str(obj.get("objection", ""))
                        seen_texts.add(text.lower().strip())
                        updated.append(obj)
                    elif isinstance(obj, str):
                        updated.append({"objection": obj, "severity": 3, "status": "unresolved"})
                        seen_texts.add(obj.lower().strip())
                    elif isinstance(obj, list) and obj:
                        # Nested list — take first dict or wrap
                        for sub in obj:
                            if isinstance(sub, dict):
                                text = str(sub.get("objection", ""))
                                seen_texts.add(text.lower().strip())
                                updated.append(sub)
                                break
                        else:
                            updated.append({"objection": str(obj), "severity": 3, "status": "unresolved"})
                for prior in prior_objections:
                    if not isinstance(prior, dict):
                        prior = {"objection": str(prior), "severity": 3}
                    p_text = str(prior.get("objection", "")).lower().strip()
                    if p_text not in seen_texts:
                        # Prior objection was dropped by LLM; preserve as unresolved
                        unresolved_copy = dict(prior)
                        unresolved_copy["status"] = "unresolved"
                        updated.append(unresolved_copy)
                return updated

        # Fail closed: preserve all prior objections as unresolved
        return [
            dict(obj, status="unresolved") if isinstance(obj, dict)
            else {"objection": str(obj), "severity": 3, "status": "unresolved"}
            for obj in prior_objections
        ]


class ModeratorAgent:
    # Lab identity (plan-mode judge): no tools at all — verdicts only.
    manifest_name: str = "ModeratorAgent"
    allowed_tools: List[str] = []

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.runtime_config = self.context.config if self.context else config
        self.client = get_llm_client()

    def evaluate_debate(
        self,
        topic: Dict[str, Any],
        rounds: List[Dict[str, Any]],
        objections: List[Dict[str, Any]],
        mode: str = "standard",
    ) -> Dict[str, Any]:
        """Ensemble judge: hard deterministic gates dominate; soft scores cannot rescue hard fails.

        P2 light mode (try_and_report): requires only falsifiable + measurable,
        uses threshold 5.0 instead of 7.0, and never fails on novelty/severity
        alone — the outcome is labelled exploratory downstream.
        """
        logger.info("Moderator evaluating debate for: %s (%d rounds, %d unresolved objections)",
                     topic.get("title", "<untitled>"), len(rounds),
                     len([o for o in objections if isinstance(o, dict) and o.get("status") != "resolved"]))
        models = self._judge_models()
        scores: List[float] = []
        reasonings: List[str] = []
        valid_judge_responses = 0
        transcript = json.dumps(rounds, default=str)[:6000]

        unresolved = [
            o for o in objections if isinstance(o, dict) and o.get("status") != "resolved"
        ]

        # Hard Gates evaluation
        _light = str(mode or "standard") in ("light", "try_and_report", "replication") or str((topic or {}).get("hunt_mode") or "") in ("try_and_report", "replication")
        # v5 (f537f638: 6.5 PASS on 4 unresolved objections for a bootstrap-CI
        # claim via the 5.0 light bar): statistical topics never use the light
        # track — interval claims need the full standard bar + full hard gates.
        _topic_blob = json.dumps({k: topic.get(k) for k in ("title", "description", "rationale") if k in topic}, default=str) if isinstance(topic, dict) else ""
        import re as _re

        # e0ec9e91: single "CI"/"bootstrap" mention forced every hunter topic
        # (which always promises CIs) onto the full bar, so 6.50 still FAIL.
        # Require >=2 distinct stats signals before treating as stats topic.
        _stats_hits = len(set(_re.findall(r"bootstrap|confidence interval|\bCI\b|hypothesis test|t-test|wilcoxon|paired test", _topic_blob, _re.IGNORECASE)))
        _is_stats_topic = _stats_hits >= 2
        if _is_stats_topic:
            _light = False
        hard_gates = {
            "hypothesis_falsifiable": bool(
                topic.get("falsifiable_prediction")
                or _as_dict(topic.get("structured_hypothesis")).get("falsification_condition")
            ),
            "measurable_outcome_exists": bool(
                _as_dict(topic.get("structured_hypothesis")).get("dependent_variables")
                or topic.get("rationale")
            ),
            "no_severe_unresolved_objections": not any(
                _coerce_int(u.get("severity"), 0) >= 4 for u in unresolved
            ),
            "resources_available": not any(
                u.get("criterion") == "feasibility" and _coerce_int(u.get("severity"), 0) >= 4 for u in unresolved
            ),
        }
        if _light:
            # Light track: novelty/severity objections become advisory; only
            # falsifiable + measurable + no sev-5 resource block are hard.
            hard_gates["no_severe_unresolved_objections"] = not any(
                _coerce_int(u.get("severity"), 0) >= 5 for u in unresolved
            )
        # v5: statistical topics with >=3 unresolved objections fail closed —
        # an interval claim surviving 3+ objections has not earned a PASS.
        if _is_stats_topic and len(unresolved) >= 3:
            hard_gates["unresolved_objections_for_stats_claim"] = False
        all_hard_passed = all(hard_gates.values())

        base_prompt = f"""
Moderate this multi-round research debate.
Topic: {topic.get('title')}
Structured Hypothesis: {json.dumps(topic.get('structured_hypothesis', {}), default=str)[:1500]}
Transcript: {transcript}
Unresolved objections: {json.dumps(unresolved, default=str)[:2000]}
Pass threshold: {self.runtime_config.debate_pass_threshold}

Score overall scientific merit 1-10 on soft dimensions (significance, novelty, clarity, expected contribution).
Hard gates dominate: missing falsifiable prediction, unmeasurable outcome, any severity>=4
objection, or 3+ unresolved stats objections forces FAIL regardless of score.
JSON: {{"score": 7.5, "passed": true, "reasoning": "...", "decision": "PASS|FAIL"}}
"""
        for model in models:
            raw = call_llm(base_prompt, temperature=0.2, tier="judge", model=model,
                           **_debate_system(
                               "debate.moderator",
                               threshold=self.runtime_config.debate_pass_threshold,
                               title=str(topic.get("title", ""))))
            if is_degenerate_llm_output(raw, min_chars=20):
                continue
            parsed = parse_json_from_llm(raw) or {}
            if isinstance(parsed, dict) and "score" in parsed:
                try:
                    score_val = float(parsed["score"])
                    scores.append(score_val)
                    reasonings.append(str(parsed.get("reasoning", raw[:300])))
                    valid_judge_responses += 1
                except (TypeError, ValueError):
                    pass

        if valid_judge_responses == 0:
            # Fail closed: missing judge output must never become a passing score.
            return {
                "score": 0.0,
                "passed": False,
                "ensemble_scores": [],
                "disagreement": 0.0,
                "needs_longer_debate": False,
                "reasoning": "No valid judge responses obtained; debate fails closed.",
                "decision": "FAIL",
                "valid_judge_responses": 0,
                "hard_gates": hard_gates,
            }

        mean = sum(scores) / len(scores)
        disagreement = max(scores) - min(scores) if len(scores) > 1 else 0.0
        agreed = disagreement <= 2.5  # Relaxed from 1.5 to allow more diversity in judge scores

        # Key rule: soft score cannot override hard FAIL
        # P2 light track uses threshold 5.0 (vs 7.0) — executable + falsifiable
        # try-and-report ideas pass as exploratory, not as gap endorsements.
        # e0ec9e91: a flat 7.0 for every family meant bounded replication
        # topics (variance-reporting replications, the system's bread and
        # butter) could never pass — 17 FAILs at 5.0-6.5. Family-aware bar:
        # replications are judged at 6.0, novel-method claims keep 7.0.
        _threshold = float(getattr(self.runtime_config, "debate_pass_threshold", 7.0) or 7.0)
        if _light:
            _threshold = min(_threshold, 5.0)
        else:
            try:
                _is_replication_family = bool((topic or {}).get("replication_target")) or str(
                    (topic or {}).get("title") or "").strip().lower().startswith("replication:")
            except Exception:
                _is_replication_family = False
            if _is_replication_family:
                _threshold = min(_threshold, 6.0)
        passed = all_hard_passed and agreed and (mean >= _threshold)

        logger.info("Moderator decision: score=%.2f passed=%s disagreement=%.2f hard_gates=%s judge_responses=%d threshold=%.2f",
                     mean, passed, disagreement, all_hard_passed, valid_judge_responses, _threshold)

        return {
            "score": mean,
            "passed": passed,
            "ensemble_scores": scores,
            "disagreement": disagreement,
            "needs_longer_debate": disagreement > 2.5 and len(rounds) < self.runtime_config.debate_max_rounds,
            "reasoning": " | ".join(reasonings[:3]),
            "decision": "PASS" if passed else "FAIL",
            "valid_judge_responses": valid_judge_responses,
            "hard_gates": hard_gates,
            "unresolved_objections": unresolved,
        }

    def _judge_models(self) -> List[str]:
        models = self.runtime_config.get_ensemble_models()
        if models:
            return models[:3]
        out = []
        for tier in ("judge", "strong", "cheap"):
            m = self.runtime_config.resolve_model(tier)
            if m and m not in out:
                out.append(m)
        return out[:3] or [self.runtime_config.resolve_model("default")]


class HypothesisDebateSystem:
    def __init__(self, context: Optional[RunContext] = None):
        context = context or get_active_context()
        self.context = context
        self.runtime_config = context.config if context else config
        self.proposer = ProposerAgent(context)
        self.challenger = ChallengerAgent(context)
        self.moderator = ModeratorAgent(context)
        self.elo = EloStore(context=context)

    def conduct_debate(self, topic: Dict[str, Any], mode: str = "standard") -> DebateResult:
        title = topic.get("title", "<untitled>")
        logger.info("=== Debate start: %s (mode=%s) ===", title, mode)
        log_agent_action("HypothesisDebate", "start", {"topic": title, "mode": mode})
        rounds: List[Dict[str, Any]] = []
        argument = self.proposer.build_argument(topic)
        # NOTE: `is True` (not truthiness) — test doubles are MagicMocks that
        # auto-create any attribute as truthy; only a real infra flag trips this.
        if getattr(self.proposer, "_proposer_invalid", False) is True:
            # P0-1: proposer infra (truncation/transport/model-404) — the
            # debate never substantively happened. Return an infra result
            # WITHOUT challenger/moderator/Elo/memory writes so orchestration
            # tries the next topic instead of terminal-killing the run, and
            # no fake science lesson is recorded.
            code = getattr(self.proposer, "_proposer_error_code", "unknown_infra") or "unknown_infra"
            logger.warning("Debate infra-fail for %r (proposer code=%s); skipping challenger/moderator",
                           title[:80], code)
            log_agent_action("HypothesisDebate", "debate_infra_failure", {
                "topic": title, "stage": "proposer_round1", "error_code": code,
            })
            return DebateResult(
                topic=title,
                proposer_argument="",
                challenger_argument="",
                moderator_decision="INFRA_FAIL",
                score=0.0,
                passed=False,
                reasoning=f"Proposer LLM infra failure ({code}); no substantive debate occurred.",
                rounds=[{"round": 1, "proposer": "", "challenger": "", "mode": mode,
                         "infra_failure": True, "error_code": code}],
                objections=[],
                unresolved_objections=[],
                ensemble_scores=[],
                elo_delta=0.0,
                infra_failure=True,
            )
        logger.info("Round 1 proposer argument generated (%d chars)", len(argument))
        # Standard mode keeps the exact legacy call shape (test doubles patch
        # single-argument build_rebuttal); only steelman passes the keyword.
        if mode == "standard":
            rebuttal = self.challenger.build_rebuttal(topic, argument)
        else:
            rebuttal = self.challenger.build_rebuttal(topic, argument, mode=mode)
        logger.info("Round 1 challenger rebuttal generated (%d chars)", len(rebuttal))
        objections = getattr(self.challenger, "_last_objections", []) or [
            {"criterion": "soundness", "objection": rebuttal[:500], "severity": 3, "status": "unresolved"}
        ]
        objections = _ensure_objections_are_dicts(objections)
        rounds.append({
            "round": 1,
            "proposer": argument,
            "challenger": rebuttal,
            "objections": objections,
            "mode": mode,
            "steelman": getattr(self.challenger, "_last_steelman", ""),
            "concession_test": getattr(self.challenger, "_last_concession_test", ""),
        })

        min_r = self.runtime_config.debate_min_rounds
        max_r = self.runtime_config.debate_max_rounds
        current_objections = objections
        final = None

        # Skip additional rounds when challenger output was invalid — the
        # round-2 proposer response and challenger followup would be against
        # a garbled round-1, producing uninformative results.
        # Exception: template fallback objections ARE substantive, so debate
        # continues normally when _challenger_fallback_used is set.
        if getattr(self.challenger, "_challenger_invalid", False) and not getattr(self.challenger, "_challenger_fallback_used", False):
            logger.info("Challenger invalid in round 1 — skipping additional rounds")
            max_r = 1

        # e0ec9e91: hopeless topics (10+ unresolved, 3+ severe) burned a full
        # round-2 + repair + re-debate to reach the same FAIL. Early-abort to
        # the final judge after round 1.
        try:
            _r1_unresolved = [o for o in (objections or []) if isinstance(o, dict) and o.get("status") != "resolved"]
            _r1_severe = sum(1 for o in _r1_unresolved if _coerce_int(o.get("severity"), 0) >= 4)
            if len(_r1_unresolved) >= 10 and _r1_severe >= 3:
                logger.info("Round 1 hopeless (%d unresolved, %d severe) — early-abort to final judge",
                            len(_r1_unresolved), _r1_severe)
                max_r = 1
        except Exception:
            pass

        for r in range(2, max_r + 1):
            logger.info("Round %d: proposer responding to objections", r)
            response = self.proposer.respond_to_objections(topic, argument, current_objections)
            logger.info("Round %d: challenger evaluating followup objections", r)
            current_objections = self.challenger.followup_objections(topic, response, current_objections)
            current_objections = _ensure_objections_are_dicts(current_objections)
            unresolved_now_r = [o for o in current_objections if isinstance(o, dict) and o.get("status") != "resolved"]
            logger.info("Round %d complete: %d unresolved objections remaining", r, len(unresolved_now_r))
            rounds.append({
                "round": r,
                "proposer": response,
                "objections": current_objections,
            })
            argument = response
            tracker = get_tracker()
            if tracker:
                tracker.bump("debate_rounds")

            unresolved_now = [
                o for o in current_objections
                if isinstance(o, dict) and o.get("status") != "resolved"
            ]
            # Early stop after min rounds if no severe unresolved
            if r >= min_r and not any(_coerce_int(u.get("severity"), 0) >= 3 for u in unresolved_now):
                logger.info("Round %d: early stop — no severe unresolved objections", r)
                break

            # Ensemble disagreement can force another round.
            # e0ec9e91: with a single judge model, mid-round judging always
            # returns disagreement=0.0 and wastes a call — skip it and let the
            # single final evaluation decide.
            if r >= min_r:
                try:
                    _n_models = len(self.moderator._judge_models())
                except Exception:
                    _n_models = 2
                if _n_models <= 1:
                    final = None
                    logger.info("Round %d: single judge — skipping mid-round evaluation", r)
                    break
                try:
                    mid = self.moderator.evaluate_debate(topic, rounds, current_objections, mode=mode)
                except TypeError:
                    mid = self.moderator.evaluate_debate(topic, rounds, current_objections)
                if not mid.get("needs_longer_debate"):
                    final = mid
                    logger.info("Round %d: moderator accepted result (no longer debate needed)", r)
                    break
                else:
                    logger.info("Round %d: moderator requests longer debate (disagreement=%.2f)", r, mid.get("disagreement", 0))
        else:
            final = None

        if not final:
            logger.info("Running final moderator evaluation after %d rounds", len(rounds))
            try:
                final = self.moderator.evaluate_debate(topic, rounds, current_objections, mode=mode)
            except TypeError:
                final = self.moderator.evaluate_debate(topic, rounds, current_objections)
            if final.get("needs_longer_debate") and len(rounds) < max_r:
                logger.info("Moderator still wants longer debate; running extra round")
                response = self.proposer.respond_to_objections(topic, argument, current_objections)
                current_objections = self.challenger.followup_objections(topic, response, current_objections)
                current_objections = _ensure_objections_are_dicts(current_objections)
                rounds.append({"round": len(rounds) + 1, "proposer": response, "objections": current_objections})
                try:
                    final = self.moderator.evaluate_debate(topic, rounds, current_objections, mode=mode)
                except TypeError:
                    final = self.moderator.evaluate_debate(topic, rounds, current_objections)

        if getattr(self.challenger, "_challenger_invalid", False) and not getattr(self.challenger, "_challenger_fallback_used", False):
            logger.warning("Challenger output was invalid; capping score and forcing FAIL")
            final = {
                **(final or {}),
                "score": min(float((final or {}).get("score", 5.0)), 4.0),
                "passed": False,
                "decision": "FAIL",
                "reasoning": "Challenger output invalid after retry; debate fails closed.",
            }
        elif getattr(self.challenger, "_challenger_fallback_used", False):
            logger.info("Challenger used template fallback; moderator score stands without cap")

        unresolved_now = [
            o for o in current_objections
            if isinstance(o, dict) and o.get("status") != "resolved"
        ]
        # Skip Elo updates when debate failed on infrastructure (garbled
        # challenger with no fallback) — the outcome is uninformative about
        # hypothesis quality. Template fallback objections ARE informative.
        if getattr(self.challenger, "_challenger_invalid", False) and not getattr(self.challenger, "_challenger_fallback_used", False):
            delta = 0.0
        else:
            delta = self.elo.update(topic.get("title", "general"), final["score"], final["passed"])
        # Feature 8: record seed-strategy outcome alongside kind-Elo
        # NOTE: outcome_status here is debate-derived (argument quality), NOT
        # experiment-derived. kind-Elo uses the same semantics — there is no
        # second Elo update at the post-experiment stage.
        seed_strategy = topic.get("seed_strategy")
        if seed_strategy:
            strategy_outcome = "debate_pass" if final["passed"] else (
                "debate_weak" if final["score"] >= 6 else "debate_fail"
            )
            self.elo.record_strategy_outcome(seed_strategy, strategy_outcome)
        unresolved_text = [f"[{u.get('criterion')}] {u.get('objection')}" for u in unresolved_now]

        result = DebateResult(
            topic=topic.get("title", ""),
            proposer_argument=rounds[0].get("proposer", ""),
            challenger_argument=rounds[0].get("challenger", ""),
            steelman=str(rounds[0].get("steelman") or ""),
            concession_test=str(rounds[0].get("concession_test") or ""),
            moderator_decision=final.get("decision", "FAIL"),
            score=float(final.get("score", 0)),
            passed=bool(final.get("passed")),
            reasoning=final.get("reasoning", ""),
            rounds=rounds,
            objections=current_objections,
            unresolved_objections=unresolved_text,
            ensemble_scores=final.get("ensemble_scores") or [],
            elo_delta=delta,
            infra_failure=bool(
                getattr(self.challenger, "_challenger_invalid", False)
                and not getattr(self.challenger, "_challenger_fallback_used", False)
            ),
        )
        logger.info("=== Debate complete: %s | score=%.2f | decision=%s | rounds=%d | elo_delta=%.1f | unresolved=%d ===",
                     result.topic, result.score, result.moderator_decision,
                     len(rounds), result.elo_delta, len(unresolved_now))
        (self.context.memory if self.context else memory).add_debate_entry(
            result.topic,
            result.proposer_argument[:2000],
            result.challenger_argument[:2000],
            result.moderator_decision,
            result.score,
            structured_signal={
                "objection_type": (unresolved_now[0].get("criterion") if unresolved_now else "none"),
                "objection_types": sorted({str(item.get("criterion")) for item in unresolved_now if isinstance(item, dict) and item.get("criterion")}),
                "severity": max((_coerce_int(item.get("severity"), 0) for item in unresolved_now), default=0),
                "resolution_status": "unresolved" if unresolved_now else "resolved",
            },
            run_id=get_tracker().run_id if get_tracker() else "unknown",
            outcome_status="released",
        )
        tracker = get_tracker()
        if tracker:
            tracker.scratch("HypothesisDebate", "result", {
                "passed": result.passed,
                "score": result.score,
                "rounds": len(rounds),
                "unresolved": unresolved_text,
            })
        return result

    def revise_topic_from_objections(
        self,
        topic: Dict[str, Any],
        result: DebateResult,
    ) -> Optional[Dict[str, Any]]:
        """Produce one bounded, auditable scientific repair before rejection.

        The old workflow let a proposer answer objections rhetorically but did
        not permit it to change the experimental contract.  This method is the
        narrow repair loop: it can repair a baseline, metric, falsification
        condition, or MVE, but it may not erase provenance or silently turn a
        failed study into a pass.
        """
        if not getattr(self, "runtime_config", None):
            return None
        # P0-1: never repair infra failures — there are no scientific
        # objections to address (proposer never argued, or challenger garbled
        # with no fallback). Repairing would launder infra into science.
        if getattr(result, "infra_failure", False):
            return None
        proposer = getattr(self, "proposer", None)
        if getattr(proposer, "_proposer_invalid", False) is True:
            return None
        challenger = getattr(self, "challenger", None)
        if getattr(challenger, "_challenger_invalid", False) and not getattr(challenger, "_challenger_fallback_used", False):
            return None
        all_objections = _ensure_objections_are_dicts(result.objections or [])
        unresolved = [
            item for item in all_objections
            if isinstance(item, dict) and item.get("status") != "resolved"
        ]
        if not unresolved:
            return None
        repairable = {"baseline", "confounder", "evaluation", "statistical", "feasibility", "falsifiability", "soundness"}
        if not any(str(item.get("criterion")) in repairable for item in unresolved):
            return None
        logger.info("Attempting contract repair for '%s' (%d repairable objections)", topic.get("title", "<untitled>"), len(unresolved))
        current = _as_dict(topic.get("structured_hypothesis"))
        if not current:
            return None
        # e0ec9e91: repair prompt was current[:6000]+unresolved[:3500] at
        # max_tokens=8192 -> every repair hit truncation_empty. Cap to top-3
        # severest objections + compact contract, request 16384, and use
        # json_mode (no full re-run on truncation) so brace-repair can save
        # partial payloads instead of wasting a second ~300s call.
        _ranked = sorted(unresolved or [], key=lambda o: float((o or {}).get("severity") or 0), reverse=True)[:3]
        prompt = f"""
You are a research-methods repair agent. Revise the structured hypothesis below
to resolve the listed reviewer objections. Keep its research domain and every
evidence/provenance field intact. Do not claim novelty, performance, or a gap
without existing evidence. Produce a smaller executable study when necessary.

Current contract:
{json.dumps(current, indent=2, default=str)[:2500]}

Top unresolved objections (severest first, max 3 of {len(unresolved or [])}):
{json.dumps(_ranked, indent=2, default=str)[:2500]}

Hard requirements:
- preserve or strengthen the falsification condition;
- use only a catalogued local dataset or named synthetic data;
- name a baseline, metrics, a falsification test, and >=3 seeds;
- specify controls that address each confounder/baseline objection;
- return JSON for the complete structured hypothesis only. Keep it compact (<4000 chars).
"""
        for attempt in range(2):
            try:
                raw = call_llm(prompt, temperature=0.2, tier="strong",
                               max_tokens=16384, retry_on_truncation=False,
                               **_debate_system("debate.repair"))
            except Exception as exc:
                # 21a145ad: repair LLM raised LLMUnavailable(truncation_empty),
                # escaping conduct_with_repair -> conduct_tournament ->
                # hypothesis_debate_node and terminal-killing the run with 6
                # topics still undebated. Repair is optional polish on a topic
                # that already FAILED — infra here means "no repair", never a
                # run crash. Fail closed so the tournament continues.
                code = _classify_infra_error(exc)
                log_agent_action("HypothesisDebate", "repair_infra_failure", {
                    "topic": topic.get("title"),
                    "attempt": attempt + 1,
                    "error_code": code,
                    "error": str(exc)[:300],
                })
                logger.warning("Contract repair infra failure for %r (code=%s): %s",
                               topic.get("title", "")[:160], code, str(exc)[:160])
                return None
            parsed = parse_json_from_llm(raw) or {}
            admission = validate_topic_admission(parsed) if isinstance(parsed, dict) else {"admitted": False}
            if not admission.get("admitted"):
                log_agent_action("HypothesisDebate", "repair_contract_rejected", {
                    "attempt": attempt + 1,
                    "errors": admission.get("errors", ["invalid response"]),
                })
                continue
            revised = deepcopy(topic)
            revised["structured_hypothesis"] = parsed
            revised["falsifiable_prediction"] = parsed.get("falsification_condition", "")
            revised["research_question"] = parsed.get("research_question", "")
            revised.setdefault("repair_history", []).append({
                "source": "hypothesis_debate",
                "attempt": attempt + 1,
                "addressed_objections": [str(item.get("criterion")) for item in unresolved],
                "admission": admission,
            })
            log_agent_action("HypothesisDebate", "repair_contract_accepted", {
                "topic": topic.get("title"),
                "attempt": attempt + 1,
                "objections": len(unresolved),
            })
            return revised
        return None

    def _debate_once(self, topic: Dict[str, Any], mode: str) -> DebateResult:
        """Call conduct_debate preserving the exact legacy call shape for standard mode.

        Test doubles patch conduct_debate with single-argument fakes, so the
        standard path must not gain a keyword argument.
        """
        if mode == "standard":
            return self.conduct_debate(topic)
        return self.conduct_debate(topic, mode=mode)

    def conduct_with_repair(self, topic: Dict[str, Any], mode: str = "standard") -> List[DebateResult]:
        """Debate once, then allow exactly one contract repair and re-debate.

        Bulkhead (21a145ad): no LLM raise inside debate/repair may escape and
        kill the tournament. Infra becomes an infra_failure DebateResult so
        orchestration can try the next candidate.
        """
        title = topic.get("title", "<untitled>")
        logger.info("conduct_with_repair: initial debate for '%s'", title)
        try:
            first = self._debate_once(topic, mode)
        except Exception as exc:
            code = _classify_infra_error(exc)
            logger.warning("conduct_with_repair: initial debate infra for %r (code=%s): %s",
                           title[:160], code, str(exc)[:160])
            log_agent_action("HypothesisDebate", "debate_infra_failure", {
                "topic": title, "stage": "initial_debate", "error_code": code,
            })
            return [self._infra_result(title, code, f"Initial debate LLM infra failure ({code}); no verdict reached.")]
        if first.passed:
            logger.info("conduct_with_repair: '%s' passed on first attempt", title)
            return [first]
        logger.info("conduct_with_repair: '%s' failed, attempting contract repair", title)
        try:
            revised = self.revise_topic_from_objections(topic, first)
        except Exception as exc:
            code = _classify_infra_error(exc)
            logger.warning("conduct_with_repair: repair infra for %r (code=%s): %s",
                           title[:160], code, str(exc)[:160])
            log_agent_action("HypothesisDebate", "repair_infra_failure", {
                "topic": title, "error_code": code,
            })
            return [first]
        if not revised:
            logger.info("conduct_with_repair: repair failed for '%s', returning original result", title)
            return [first]
        # Preserve the candidate object's identity: the tournament/workflow
        # selects from its original topic list after this method returns.
        topic.clear()
        topic.update(revised)
        logger.info("conduct_with_repair: re-debating revised topic '%s'", revised.get("title", "<untitled>"))
        try:
            second = self._debate_once(revised, mode)
        except Exception as exc:
            code = _classify_infra_error(exc)
            logger.warning("conduct_with_repair: re-debate infra for %r (code=%s): %s",
                           title[:160], code, str(exc)[:160])
            log_agent_action("HypothesisDebate", "debate_infra_failure", {
                "topic": title, "stage": "re_debate", "error_code": code,
            })
            return [first]
        return [first, second]

    def _infra_result(self, title: str, code: str, reasoning: str) -> DebateResult:
        """Synthesize an infra_failure result when debate never ran."""
        return DebateResult(
            topic=title,
            proposer_argument="",
            challenger_argument="",
            moderator_decision="INFRA_FAIL",
            score=0.0,
            passed=False,
            reasoning=reasoning,
            rounds=[{"round": 1, "proposer": "", "challenger": "",
                     "infra_failure": True, "error_code": code}],
            objections=[],
            unresolved_objections=[],
            ensemble_scores=[],
            elo_delta=0.0,
            infra_failure=True,
        )

    def conduct_tournament(self, topics: List[Dict[str, Any]], rounds: int = 1, mode: str = "standard") -> List[DebateResult]:
        """Debate candidates until one passes, then hand off immediately.

        Each candidate is debated at most once. Remaining candidates are not
        debated once a hypothesis clears the bar — Planning should receive the
        accepted result without wasting LLM calls or re-debating failed topics.
        """
        logger.info("=== Tournament start: %d candidates ===", len(topics))
        results: List[DebateResult] = []
        debated_titles = set()
        for i, topic in enumerate(topics, 1):
            title = (topic.get("title") or "").strip()
            if not title or title.lower() in debated_titles:
                continue
            debated_titles.add(title.lower())
            logger.info("Tournament matchup %d/%d: '%s'", i, len(topics), title)
            try:
                attempts = self.conduct_with_repair(topic, mode=mode) if mode != "standard" else self.conduct_with_repair(topic)
            except Exception as exc:
                # Last-resort bulkhead: a single candidate must never abort
                # the tournament (21a145ad lost 6/8 topics to one repair
                # truncation). Record infra and continue.
                code = _classify_infra_error(exc)
                logger.warning("Tournament matchup %d infra for %r (code=%s): %s",
                               i, title[:160], code, str(exc)[:160])
                log_agent_action("HypothesisDebate", "tournament_matchup_infra", {
                    "topic": title, "error_code": code,
                })
                attempts = [self._infra_result(
                    title, code,
                    f"Tournament matchup LLM infra failure ({code}); no verdict reached.")]
            results.extend(attempts)
            if attempts[-1].passed:
                logger.info("Tournament winner found: '%s' (score=%.2f)", title, attempts[-1].score)
                return sorted(results, key=lambda item: (item.passed, item.score), reverse=True)
        logger.info("Tournament complete: no candidate passed (%d total results)", len(results))
        return sorted(results, key=lambda item: (item.passed, item.score), reverse=True)
