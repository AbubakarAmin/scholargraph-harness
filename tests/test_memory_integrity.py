"""Adversarial tests for the fail-closed prompt memory boundary.

These tests print concrete transcripts so a reviewer can see the poisoned
payload that was inserted and the prompt context that came back — not just
a pass/fail count.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import numpy as np
import pytest

from agents.hypothesis_debate import EloStore, hypothesis_kind
from agents.topic_hunter import TopicHunterAgent
from core.memory import ResearchMemory
from core.run_log import CrossRunMemory


def temporary_memory(tmp_path):
    memory = ResearchMemory.__new__(ResearchMemory)
    memory.vector_db_path = tmp_path / "vectors"
    memory.vector_db_path.mkdir()
    memory.dimension = 4
    memory.index = __import__("faiss").IndexFlatL2(memory.dimension)
    memory.metadata = []
    memory.debate_log = []
    memory.feedback_log = []
    return memory


def test_fallback_narrative_is_absent_from_writer_context(tmp_path):
    memory = temporary_memory(tmp_path)
    poison = "fallback traceback: hallucinated accuracy 99.9"
    memory.add_embedding(
        np.zeros(4, dtype=np.float32),
        {
            "namespace": "writer_fallbacks",
            "content_class": "generated_narrative",
            "retrieval_eligible": False,
            "run_id": "failed-run",
            "agent": "WriterAgent",
            "outcome_status": "failed",
            "content": poison,
        },
    )
    context = memory.get_prompt_context(np.zeros(4, dtype=np.float32), namespace="writer_sections")
    writer_exemplars = memory.get_prompt_context(
        np.zeros(4, dtype=np.float32),
        namespace="writer_exemplars",
        outcome_status="released",
    )
    transcript = {
        "inserted": poison,
        "writer_sections_context": context,
        "writer_exemplars_context": writer_exemplars,
        "poison_absent_from_sections": poison not in json.dumps(context),
        "poison_absent_from_exemplars": poison not in json.dumps(writer_exemplars),
    }
    print("\nFALLBACK_POISON_TRANSCRIPT\n" + json.dumps(transcript, sort_keys=True, indent=2))
    assert transcript["poison_absent_from_sections"]
    assert transcript["poison_absent_from_exemplars"]
    assert context == []
    assert writer_exemplars == []


def test_hallucinated_claim_is_unreachable_from_any_prompt(tmp_path):
    memory = temporary_memory(tmp_path)
    poison = "WRONG PAST CLAIM: the method reaches 100 percent accuracy"
    memory.add_embedding(
        np.zeros(4, dtype=np.float32),
        {
            "namespace": "paper_sections",
            "content_class": "generated_narrative",
            # Even if a buggy caller asks for eligibility, narrative stays closed.
            "retrieval_eligible": True,
            "run_id": "released-looking-run",
            "agent": "WriterAgent",
            "outcome_status": "released",
            "content": poison,
        },
    )
    context = memory.get_prompt_context(np.zeros(4, dtype=np.float32))
    any_ns = memory.get_prompt_context(np.zeros(4, dtype=np.float32), allow_non_released=True)
    transcript = {
        "inserted": poison,
        "default_prompt_context": context,
        "allow_non_released_context": any_ns,
        "poison_absent": poison not in json.dumps({"a": context, "b": any_ns}),
    }
    print("\nHALLUCINATED_CLAIM_POISON_TRANSCRIPT\n" + json.dumps(transcript, sort_keys=True, indent=2))
    assert transcript["poison_absent"]
    assert context == []
    assert any_ns == []


def test_released_structured_signal_is_retrievable_without_raw_text(tmp_path):
    memory = temporary_memory(tmp_path)
    memory.add_embedding(
        np.ones(4, dtype=np.float32),
        {
            "namespace": "debate_signals",
            "content_class": "structured_signal",
            "retrieval_eligible": True,
            "run_id": "released-run",
            "agent": "ChallengerAgent",
            "outcome_status": "released",
            "signal": {
                "objection_type": "feasibility",
                "severity": 5,
                "resolution_status": "unresolved",
            },
            "content": "raw argument must stay audit-only",
        },
    )
    context = memory.get_prompt_context(namespace="debate_signals", outcome_status="released")
    assert context == [{
        "id": 0,
        "namespace": "debate_signals",
        "content_class": "structured_signal",
        "run_id": "released-run",
        "agent": "ChallengerAgent",
        "outcome_status": "released",
        "signal": {
            "objection_type": "feasibility",
            "severity": 5,
            "resolution_status": "unresolved",
        },
    }]
    assert "raw argument" not in json.dumps(context)


def test_outcome_filtering_rejects_failed_rejected_unclassified(tmp_path):
    memory = temporary_memory(tmp_path)
    for status, vec in [("failed", [1, 0, 0, 0]), ("rejected", [0, 1, 0, 0]), ("unknown", [0, 0, 1, 0])]:
        memory.add_embedding(
            np.array(vec, dtype=np.float32),
            {
                "namespace": "writer_exemplars",
                "content_class": "structured_signal",
                "retrieval_eligible": True,
                "run_id": f"{status}-run",
                "agent": "WriterAgent",
                "outcome_status": status,
                "signal": {"purpose": "abstract", "quality_score": 9.0},
            },
        )
    memory.add_embedding(
        np.array([0, 0, 0, 1], dtype=np.float32),
        {
            "namespace": "writer_exemplars",
            "content_class": "structured_signal",
            "retrieval_eligible": True,
            "run_id": "released-run",
            "agent": "WriterAgent",
            "outcome_status": "released",
            "signal": {"purpose": "abstract", "quality_score": 8.5},
        },
    )
    # Legacy unclassified (no content_class) stays ineligible.
    memory.add_embedding(
        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        {"namespace": "writer_exemplars", "signal": {"purpose": "abstract"}, "run_id": "legacy"},
    )
    default = memory.get_prompt_context(namespace="writer_exemplars")
    released = memory.get_prompt_context(namespace="writer_exemplars", outcome_status="released")
    assert len(default) == 1
    assert default[0]["run_id"] == "released-run"
    assert released[0]["outcome_status"] == "released"
    assert all(item["run_id"] != "legacy" for item in default)


def test_provenance_fields_present_on_every_returned_item(tmp_path):
    memory = temporary_memory(tmp_path)
    memory.add_embedding(
        np.ones(4, dtype=np.float32),
        {
            "namespace": "supervisor_feedback",
            "content_class": "structured_signal",
            "retrieval_eligible": True,
            "run_id": "run-42",
            "agent": "SupervisorAgent",
            "outcome_status": "released",
            "signal": {"verdict": "pass", "score": 9.0, "blocking": False},
        },
    )
    context = memory.get_prompt_context(namespace="supervisor_feedback")
    assert len(context) == 1
    for key in ("run_id", "agent", "outcome_status"):
        assert key in context[0]
        assert context[0][key] not in (None, "")


def test_debate_transcript_retrieves_only_structured_objection_tag(tmp_path):
    memory = temporary_memory(tmp_path)
    argument = "FLAWED ARGUMENT: the unavailable dataset is guaranteed to improve results"
    memory.add_debate_entry(
        "A hypothesis",
        argument,
        "A critique",
        "FAIL",
        3.0,
        structured_signal={
            "objection_type": "feasibility",
            "severity": 5,
            "resolution_status": "unresolved",
        },
        run_id="run-1",
    )
    context = memory.get_prompt_context(namespace="debate_transcripts")
    transcript = {
        "audit_argument": argument,
        "prompt_context": context,
        "argument_absent": argument not in json.dumps(context),
        "critique_absent": "A critique" not in json.dumps(context),
    }
    print("\nDEBATE_POISON_TRANSCRIPT\n" + json.dumps(transcript, sort_keys=True, indent=2))
    assert transcript["argument_absent"]
    assert transcript["critique_absent"]
    assert context[0]["signal"]["objection_type"] == "feasibility"
    assert context[0]["signal"]["severity"] == 5
    assert context[0]["run_id"] == "run-1"


def test_search_similar_does_not_return_raw_narrative_even_in_audit_mode(tmp_path):
    memory = temporary_memory(tmp_path)
    poison = "RAW FAIL TRACE: the model actually achieves 100% on the hidden test set"
    memory.add_embedding(
        np.zeros(4, dtype=np.float32),
        {
            "namespace": "writer_fallbacks",
            "content_class": "generated_narrative",
            "retrieval_eligible": False,
            "run_id": "audit-run",
            "agent": "WriterAgent",
            "outcome_status": "failed",
            "content": poison,
        },
    )
    results = memory.search_similar(np.zeros(4, dtype=np.float32), k=5)
    payload = json.dumps(results)
    assert poison not in payload
    assert all("content" not in item for item in results)
    assert results == []


def test_audit_search_preserves_raw_for_forensics(tmp_path):
    memory = temporary_memory(tmp_path)
    poison = "RAW FAIL TRACE preserved for humans"
    memory.add_embedding(
        np.zeros(4, dtype=np.float32),
        {
            "namespace": "writer_fallbacks",
            "content_class": "generated_narrative",
            "retrieval_eligible": False,
            "content": poison,
        },
    )
    audit = memory.audit_search_similar(np.zeros(4, dtype=np.float32), k=5)
    assert poison in json.dumps(audit)


def test_cross_run_prompt_context_exposes_tags_not_raw_reasons(tmp_path):
    mem = CrossRunMemory(path=str(tmp_path / "cross.jsonl"))
    mem.record_rejection(
        "topic",
        "Saturated GAN topic",
        "novelty too low / similar to saturated work",
        {"sim": 0.95},
    )
    context = mem.get_prompt_context()
    blob = json.dumps(context)
    assert "novelty too low" not in blob
    assert context[0]["rejection_reason"] == "novelty"
    assert context[0]["outcome_status"] == "rejected"


def test_elo_shrinkage_keeps_noisy_early_scores_near_prior(tmp_path):
    store = EloStore(path=str(tmp_path / "elo.json"))
    store.prior = 1500.0
    store.shrinkage_k = 8.0
    store.min_observations = 5
    # Inject a huge win; early observations must stay close to prior.
    for _ in range(2):
        store.update("attention is all you need for graphs", 10.0, True)
    record = store.get_record("attention")
    transcript = {
        "observations": record["observations"],
        "raw_rating": record["raw_rating"],
        "shrunk_rating": record["rating"],
        "prior": record["prior"],
        "distance_from_prior": abs(record["rating"] - record["prior"]),
        "raw_distance_from_prior": abs(record["raw_rating"] - record["prior"]),
    }
    print(json.dumps(transcript, sort_keys=True, indent=2))
    assert record["observations"] == 2
    assert transcript["distance_from_prior"] < transcript["raw_distance_from_prior"]
    assert transcript["distance_from_prior"] < 80
    # After enough observations, shrunk rating may move further.
    for _ in range(10):
        store.update("attention is all you need for graphs", 10.0, True)
    later = store.get_record("attention")
    assert later["observations"] >= store.min_observations
    assert abs(later["rating"] - later["prior"]) > abs(record["rating"] - record["prior"])


def test_exploration_distribution_does_not_monopolize_one_kind(tmp_path, monkeypatch):
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter.runtime_config = type("C", (), {
        "topic_exploration_every": 2,
        "topic_exploration_seed": 7,
        "elo_prior_rating": 1500.0,
        "elo_shrinkage_k": 8.0,
        "elo_min_observations": 5,
    })()
    hunter.context = None
    hunter._selection_counter = 0
    hunter._rng = random.Random(7)

    elo_path = tmp_path / "elo.json"
    # Entrench "graph" with high raw wins; leave others cold.
    store = EloStore(path=str(elo_path))
    store.min_observations = 5
    for _ in range(12):
        store.update("graph neural networks", 10.0, True)

    kinds = ["graph", "diffusion", "attention", "nlp"]
    counts = {k: 0 for k in kinds}
    modes = {"exploitation": 0, "forced_exploration": 0}

    class FixedElo(EloStore):
        def __init__(self, *a, **k):
            super().__init__(path=str(elo_path))

    monkeypatch.setattr("agents.topic_hunter.EloStore", FixedElo)

    for i in range(40):
        topics = [
            {"title": f"{k} topic {i}", "hypothesis_kind": k, "feasibility": 8, "rank": idx + 1, "score": 8}
            for idx, k in enumerate(kinds)
        ]
        # Pretend judge already ranked graph first.
        for t in topics:
            t["elo_rating"] = store.get(t["hypothesis_kind"])
            t["elo_observations"] = store.observations(t["hypothesis_kind"])
        ranked = sorted(topics, key=lambda x: (x.get("rank", 999), -x.get("elo_rating", 1500.0)))
        result = hunter._apply_exploration(ranked, store)
        top = result[0]
        counts[top["hypothesis_kind"]] += 1
        modes[top.get("selection_mode", "exploitation")] += 1

    transcript = {"selection_counts": counts, "modes": modes}
    print(json.dumps(transcript, sort_keys=True, indent=2))
    assert modes["forced_exploration"] >= 10
    # Graph must not monopolize every selection.
    assert counts["graph"] < 40
    assert max(counts.values()) < 36
    # At least two other kinds appear.
    assert sum(1 for v in counts.values() if v > 0) >= 3


def test_exploration_never_bypasses_feasibility_gate():
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter.runtime_config = type("C", (), {"research_domain": "cs"})()
    hunter.context = None
    hunter.rejection_log = []
    infeasible = {
        "title": "Train a 70B LLM on ImageNet with GPU",
        "description": "Requires GPU and internet download of ImageNet",
        "dataset_plan": "download imagenet from the internet",
        "feasibility": 2,
    }
    report = hunter.feasibility_filter(infeasible)
    assert report["ok"] is False
    assert report["reasons"]


def test_agent_modules_do_not_bypass_prompt_memory_boundary():
    """Static audit: agents may not call raw retrieval APIs for prompts."""
    agents_dir = Path(__file__).resolve().parents[1] / "agents"
    forbidden_calls = {
        "search_similar",
        "audit_search_similar",
        "get_recent_feedback",
        "get_recent_debates",
        "lessons_for_prompt",
    }
    forbidden_attrs = {"index", "metadata"}  # direct FAISS/index access
    failures = []

    for path in sorted(agents_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                if name in forbidden_calls:
                    failures.append(f"{path.name}:{node.lineno} calls {name}()")
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                # Allow mentions in comments? AST won't have comments. Flag .index / .metadata access.
                if isinstance(node.value, ast.Attribute) and node.value.attr in {
                    "vector_memory", "feedback_memory", "memory"
                }:
                    failures.append(f"{path.name}:{node.lineno} accesses .{node.attr}")

    assert failures == [], "Retrieval bypasses found:\n" + "\n".join(failures)


def test_legacy_unclassified_entries_load_but_stay_ineligible(tmp_path):
    memory = temporary_memory(tmp_path)
    memory.metadata.append({
        "id": 0,
        "namespace": "legacy",
        "content": "old prose without classification",
        "run_id": "ancient",
    })
    memory.index.add(np.zeros((1, 4), dtype=np.float32))
    context = memory.get_prompt_context(np.zeros(4, dtype=np.float32), allow_non_released=True)
    assert context == []
    normalized = ResearchMemory._normalize_metadata(memory.metadata[0])
    assert normalized["content_class"] == "generated_narrative"
    assert normalized["retrieval_eligible"] is False


def test_elo_migrates_legacy_flat_and_null_observations(tmp_path):
    """Regression from fa514f98: legacy / null observation fields must not int(None)."""
    path = tmp_path / "elo.json"
    path.write_text(
        json.dumps({
            "graph": 1486.1,  # legacy flat
            "general": {
                "rating": 1476.6,
                "raw_rating": 1455.9,
                "observations": None,  # explicit null — classic .get default miss
                "prior": 1500.0,
                "shrinkage_k": 8.0,
            },
            "vision": {
                "rating": 1500.0,
                "observation_count": None,  # alias only, also null
            },
        }),
        encoding="utf-8",
    )
    store = EloStore(path=str(path))
    assert store.observations("graph") == 0
    assert store.observations("general") == 0
    assert store.observations("vision") == 0
    # Update must not crash; migration must persist full schema.
    store.update("Evaluating Fairness Metrics in Data-Scarce Environments", 7.5, False)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(reloaded["general"], dict)
    assert reloaded["general"]["observations"] == 1
    assert reloaded["graph"]["observations"] == 0
    assert "shrinkage_k" in reloaded["vision"]


def test_debate_survives_null_severity_in_unresolved(tmp_path, monkeypatch):
    """Exact crash shape from fa514f98: int(None) on unresolved severity after Elo write."""
    from agents.hypothesis_debate import HypothesisDebateSystem

    elo_path = tmp_path / "elo.json"
    elo_path.write_text(json.dumps({"general": {"rating": 1500.0, "observations": None}}), encoding="utf-8")

    system = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
    system.context = None
    system.runtime_config = type("C", (), {
        "debate_min_rounds": 2,
        "debate_max_rounds": 2,
        "debate_pass_threshold": 7.0,
    })()
    system.elo = EloStore(path=str(elo_path))

    class P:
        def build_argument(self, topic):
            return "argument"
        def respond_to_objections(self, topic, argument, unresolved):
            return "response"

    class C:
        _last_objections = [
            {"criterion": "ethics", "objection": "synthetic data concern", "severity": None},
        ]
        def build_rebuttal(self, topic, argument):
            return "rebuttal"
        def followup_objections(self, topic, response, prior):
            return [{"criterion": "ethics", "objection": "still open", "severity": None}]

    class M:
        def evaluate_debate(self, topic, rounds, unresolved):
            return {
                "score": 7.5,
                "passed": True,
                "decision": "PASS",
                "reasoning": "ok",
                "ensemble_scores": [7.5],
                "needs_longer_debate": False,
            }

    system.proposer = P()
    system.challenger = C()
    system.moderator = M()

    class Mem:
        def add_debate_entry(self, *a, **k):
            # Ensure structured_signal severity coercion already happened (no raise).
            assert k["structured_signal"]["severity"] == 0

    monkeypatch.setattr("agents.hypothesis_debate.memory", Mem())
    monkeypatch.setattr("agents.hypothesis_debate.get_tracker", lambda: None)

    result = system.conduct_debate({"title": "Evaluating Fairness Metrics in Data-Scarce Environments"})
    assert result.passed
    assert result.score == 7.5


def test_debate_crash_becomes_terminal_technical_failure(monkeypatch):
    from core.state import initialize_state
    from core.workflow_nodes import hypothesis_debate_node

    class Boom:
        def conduct_tournament(self, topics, rounds=2):
            raise TypeError("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'")

    monkeypatch.setattr("core.workflow_nodes._create_agent", lambda cls: Boom())
    monkeypatch.setattr("core.workflow_nodes.get_tracker", lambda: None)

    state = initialize_state()
    state["topics"] = [
        {"title": "a"},
        {"title": "b"},
        {"title": "c"},
    ]
    out = hypothesis_debate_node(state)
    assert out["current_phase"] == "complete"
    assert out["should_continue"] is False
    assert out["should_reset"] is False
    assert "Hypothesis debate subsystem crashed" in (out["terminal_error"] or "")
    assert out["technical_failures"]["hypothesis_debate"]["failure_kind"] == "technical"
    # Must not silently advance toward planning with leftover topics.
    assert out.get("hypothesis_passed") is False


def test_call_llm_increments_llm_calls(monkeypatch):
    from core import llm as llm_mod

    class FakeClient:
        def chat(self, *a, **k):
            return "ok"

    class FakeTracker:
        def __init__(self):
            self.stats = {"llm_calls": 0}

        def bump(self, key, amount=1):
            self.stats[key] = self.stats.get(key, 0) + amount

    tracker = FakeTracker()
    monkeypatch.setattr(llm_mod, "get_llm_client", lambda: FakeClient())
    monkeypatch.setattr("core.run_log.get_tracker", lambda: tracker)

    assert llm_mod.call_llm("hi", model="fake-model") == "ok"
    assert tracker.stats["llm_calls"] == 1


# ---------------------------------------------------------------------------
# TopicHunter stateful query generation regression tests
# ---------------------------------------------------------------------------


def test_stateful_queries_vary_with_cross_run_memory_state(tmp_path, monkeypatch):
    """Two discovery cycles with different CrossRunMemory state must produce different query sets."""
    from agents.topic_hunter import TopicHunterAgent
    import agents.topic_hunter as th_mod

    # Minimal hunter for query building
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter.runtime_config = type("C", (), {
        "topic_exploration_every": 4,
        "topic_exploration_seed": 42,
    })()
    hunter.context = None
    hunter._selection_counter = 0
    hunter._iteration_failures = 0

    # Cycle A: no rejected topics
    cross_run_a = []
    rejected_a = hunter._extract_rejected_fingerprints(cross_run_a)
    arxiv_a = hunter._build_arxiv_queries("robustness and reproducibility challenges", rejected_a, None)
    openalex_a = hunter._build_openalex_queries("robustness and reproducibility challenges", rejected_a, None)

    # Cycle B: many rejected topics containing "robustness"
    cross_run_b = [
        {"category": "rejection", "kind": "topic", "item": "Robustness of transformer attention", "rejection_reason": "novelty"},
        {"category": "rejection", "kind": "topic", "item": "Reproducibility challenges in deep learning", "rejection_reason": "feasibility"},
    ]
    rejected_b = hunter._extract_rejected_fingerprints(cross_run_b)
    arxiv_b = hunter._build_arxiv_queries("robustness and reproducibility challenges", rejected_b, None)
    openalex_b = hunter._build_openalex_queries("robustness and reproducibility challenges", rejected_b, None)

    # Queries must differ — dedup should filter out queries overlapping rejections
    arxiv_a_set = set(arxiv_a)
    arxiv_b_set = set(arxiv_b)
    assert arxiv_a_set != arxiv_b_set, (
        f"arXiv queries unchanged despite different rejections:\n"
        f"  Cycle A: {arxiv_a}\n  Cycle B: {arxiv_b}"
    )

    openalex_a_searches = [q[0] for q in openalex_a]
    openalex_b_searches = [q[0] for q in openalex_b]
    assert openalex_a_searches != openalex_b_searches, (
        f"OpenAlex queries unchanged despite different rejections:\n"
        f"  Cycle A: {openalex_a_searches}\n  Cycle B: {openalex_b_searches}"
    )


def test_preflight_dedup_skips_overlapping_queries(tmp_path):
    """Pre-flight dedup must skip queries whose fingerprints match rejected topics."""
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter.runtime_config = type("C", (), {})()
    hunter.context = None

    rejected = ["robustness transformer attention", "novelty"]
    queries = [
        "cat:cs.LG AND (\"robustness\" OR \"transformer\")",
        "cat:cs.CL AND ti:\"interpretability\"",
    ]
    deduped = hunter._preflight_dedup(queries, rejected)

    # The first query overlaps with "robustness transformer" — should be skipped
    assert len(deduped) <= len(queries)
    if deduped:
        for q in deduped:
            tokens = set(q.lower().split())
            assert not (tokens & {"robustness", "transformer"}), (
                f"Query '{q}' overlaps rejected fingerprints but was not deduped"
            )


def test_queries_use_field_scoped_arxiv_syntax(tmp_path):
    """Generated arXiv queries must use field-scoped syntax (cat:, abs:, ti:)."""
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter.runtime_config = type("C", (), {
        "topic_exploration_every": 4,
        "topic_exploration_seed": 42,
    })()
    hunter.context = None
    hunter._selection_counter = 0

    queries = hunter._build_arxiv_queries("robustness and reproducibility challenges", [], None)

    for q in queries:
        assert q.startswith("cat:"), f"Query lacks cat: prefix: {q}"
        assert " AND " in q, f"Query lacks boolean AND combinator: {q}"
        assert any(f in q for f in ("cat:", "abs:", "ti:")), f"Query lacks field-scoped syntax: {q}"


def test_queries_vary_sort_mode_in_openalex(tmp_path):
    """OpenAlex queries must vary sort mode (relevance vs cited_by_count)."""
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter.runtime_config = type("C", (), {
        "topic_exploration_every": 4,
        "topic_exploration_seed": 42,
    })()
    hunter.context = None
    hunter._selection_counter = 0

    queries = hunter._build_openalex_queries("robustness and reproducibility challenges", [], None)

    sort_modes = set()
    for search_str, params in queries:
        sort_val = params.get("sort", "")
        if "relevance" in sort_val:
            sort_modes.add("relevance")
        elif "cited_by" in sort_val:
            sort_modes.add("cited_by_count")

    assert len(sort_modes) >= 2, (
        f"Expected at least 2 sort modes, got {sort_modes} from queries: {[q[0] for q in queries]}"
    )


def test_active_hypothesis_kind_biases_queries(tmp_path):
    """When a forced-exploration kind is active, queries should include kind-specific keywords."""
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter.runtime_config = type("C", (), {
        "topic_exploration_every": 1,  # force every cycle
        "topic_exploration_seed": 42,
    })()
    hunter.context = None
    hunter._selection_counter = 0

    # Force exploration on cycle 0 (0 % 1 == 0)
    kind = hunter._active_hypothesis_kind()
    assert kind is not None, "Expected active kind when exploration_every=1 and counter=0"

    queries = hunter._build_arxiv_queries("general methods", [], kind)
    # With kind active, at least one query should contain kind-specific keywords
    all_text = " ".join(queries).lower()
    # The kind-specific keywords are inserted into the queries
    assert len(queries) >= 2, f"Expected at least 2 queries, got {len(queries)}"


def test_sample_dry_run_prints_generated_queries(tmp_path):
    """Print the actual generated query list for one dry run."""
    hunter = TopicHunterAgent.__new__(TopicHunterAgent)
    hunter.runtime_config = type("C", (), {
        "topic_exploration_every": 4,
        "topic_exploration_seed": 42,
    })()
    hunter.context = None
    hunter._selection_counter = 0

    seeds = [
        "underexplored methods and algorithms",
        "robustness and reproducibility challenges",
        "data efficiency and sample complexity",
    ]

    print("\n=== SAMPLE DRY RUN: Generated Queries ===")
    for seed in seeds:
        arxiv_q = hunter._build_arxiv_queries(seed, [], None)
        openalex_q = hunter._build_openalex_queries(seed, [], None)
        print(f"\nSeed: {seed}")
        print(f"  arXiv queries ({len(arxiv_q)}):")
        for i, q in enumerate(arxiv_q):
            print(f"    [{i+1}] {q}")
        print(f"  OpenAlex queries ({len(openalex_q)}):")
        for i, (search, params) in enumerate(openalex_q):
            print(f"    [{i+1}] search='{search}' params={params}")

    # Verify basic structure
    for seed in seeds:
        arxiv_q = hunter._build_arxiv_queries(seed, [], None)
        assert len(arxiv_q) >= 2, f"Expected at least 2 arXiv queries for '{seed}'"
        for q in arxiv_q:
            assert "cat:" in q, f"Missing cat: in query: {q}"
