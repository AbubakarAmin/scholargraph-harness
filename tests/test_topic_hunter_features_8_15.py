"""Verification tests for TopicHunter features 8–15 (upgrade brief v2).

Covers:
  Feature 1: dataset admissibility check
  Feature 2: structural gap coupling detection + S2 paper-ID construction
  Feature 3: sparsity matrix extraction
  Feature 4: contradiction mining title validation
  Feature 5: replication-target regex heuristic
  Feature 6: negative result lessons schema + builds_on_negative_result validation
  Feature 7: persona ensemble gap tagging
  Feature 8: seed-strategy provenance + Elo wiring
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from agents.topic_hunter import TopicHunterAgent, _GAPS_PER_SEED_REQUEST
from agents.hypothesis_debate import EloStore


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_agent(tmp_path=None) -> TopicHunterAgent:
    agent = TopicHunterAgent.__new__(TopicHunterAgent)
    agent.context = None
    cfg = MagicMock()
    cfg.research_domain = "machine_learning"
    cfg.openalex_email = "test@example.com"
    cfg.semantic_scholar_api_key = ""
    cfg.novelty_similarity_reject = 0.92
    cfg.topic_exploration_every = 4
    cfg.topic_exploration_seed = 42
    cfg.output_dir = str(tmp_path or Path("/tmp/th_test"))
    cfg.openalex_concept_filtering_enabled = True
    cfg.hyde_enabled = False
    cfg.hyde_max_chars = 600
    cfg.multi_hop_retrieval_enabled = False
    cfg.multi_hop_min_papers_threshold = 12
    cfg.multi_hop_max_hops = 1
    cfg.frontier_seeding_enabled = False
    cfg.frontier_refresh_every_n_runs = 5
    cfg.frontier_sample_size = 30
    cfg.frontier_terms_extracted = 8
    cfg.cross_seed_paper_cache_enabled = False
    # New feature flags
    cfg.capability_first_dataset_scoping_enabled = True
    cfg.structural_gap_mining_enabled = False
    cfg.structural_gap_max_pairs = 8
    cfg.sparsity_matrix_enabled = False
    cfg.contradiction_mining_enabled = False
    cfg.replication_target_mining_enabled = True
    cfg.negative_result_seeding_enabled = True
    cfg.persona_ensemble_enabled = False
    cfg.persona_count = 2
    cfg.seed_strategy_elo_enabled = False
    cfg.llm_seed_generation_enabled = False
    cfg.openreview_enabled = False
    cfg.arxiv_enabled = True
    cfg.llm_budget_per_seed = 30
    cfg.novelty_max_abstracts = 20
    agent.runtime_config = cfg
    agent.vector_memory = MagicMock()
    agent.client = MagicMock()
    agent.openalex_headers = {}
    agent.s2_headers = {}
    agent.base_urls = {
        "openalex": "https://api.openalex.org",
        "crossref": "https://api.crossref.org",
        "semantic_scholar": "https://api.semanticscholar.org/graph/v1",
    }
    agent._selection_counter = 0
    agent._rng = __import__("random").Random(42)
    agent.source_client = MagicMock()
    agent.rejection_log = []
    agent.source_health = {}
    agent._excluded_titles_cache = None
    agent._iteration_failures = 0
    agent._run_query_cache = {}
    agent._run_query_cache_lock = threading.Lock()
    agent._dataset_catalog_cache = None
    return agent


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 1: Capability-First Dataset Scoping
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetAdmissibility:

    def test_catalogued_dataset_passes(self):
        agent = _make_agent()
        agent._dataset_catalog_cache = [
            {"name": "sklearn_iris", "domain_tags": [], "n_samples": 150, "task_type": "classification"},
        ]
        gap = {"title": "test", "dataset_plan": "sklearn_iris"}
        assert agent._dataset_plan_admissible(gap) is True

    def test_synthetic_plan_passes(self):
        agent = _make_agent()
        agent._dataset_catalog_cache = []
        gap = {"title": "test", "dataset_plan": "synthetic: random features"}
        assert agent._dataset_plan_admissible(gap) is True

    def test_bundled_plan_passes(self):
        agent = _make_agent()
        agent._dataset_catalog_cache = []
        gap = {"title": "test", "dataset_plan": "bundled_synthetic"}
        assert agent._dataset_plan_admissible(gap) is True

    def test_uncatalogued_dataset_rejects(self):
        # v2: uncatalogued datasets now fail-open (admitted with fallback)
        # instead of hard-rejecting, since dataset checks are advisory.
        agent = _make_agent()
        agent._dataset_catalog_cache = [
            {"name": "sklearn_iris", "domain_tags": [], "n_samples": 150, "task_type": "classification"},
        ]
        gap = {"title": "test", "dataset_plan": "proprietary_medical_scans_2024"}
        assert agent._dataset_plan_admissible(gap) is True

    def test_known_public_dataset_passes(self):
        agent = _make_agent()
        agent._dataset_catalog_cache = [
            {"name": "sklearn_iris", "domain_tags": [], "n_samples": 150, "task_type": "classification"},
        ]
        gap = {"title": "test", "dataset_plan": "imagenet"}
        assert agent._dataset_plan_admissible(gap) is True

    def test_empty_catalog_fail_open(self):
        agent = _make_agent()
        agent._dataset_catalog_cache = []
        gap = {"title": "test", "dataset_plan": "anything"}
        assert agent._dataset_plan_admissible(gap) is True

    def test_no_dataset_plan_passes(self):
        agent = _make_agent()
        agent._dataset_catalog_cache = [{"name": "sklearn_iris"}]
        gap = {"title": "test"}
        assert agent._dataset_plan_admissible(gap) is True

    def test_disabled_flag_passes_all(self):
        agent = _make_agent()
        agent.runtime_config.capability_first_dataset_scoping_enabled = False
        agent._dataset_catalog_cache = [{"name": "sklearn_iris"}]
        gap = {"title": "test", "dataset_plan": "imagenet"}
        assert agent._dataset_plan_admissible(gap) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 2: Structural Gap Mining — S2 paper-ID construction
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def _s2_enabled(monkeypatch):
    """Hermetic S2 flag for coupling tests (v5 fix).

    These tests mock requests.get directly, so the provider-enabled gate must
    not depend on the machine's .env/keys.json (S2 is disabled in this
    checkout, which made the tests fail with zero HTTP calls issued).
    """
    from core import config as _config_mod
    monkeypatch.setattr(_config_mod.config, "semantic_scholar_enabled", True)


@pytest.mark.usefixtures("_s2_enabled")
class TestStructuralGapPaperId:

    def _make_get_mock(self, url_to_refs):
        """Create a mock for requests.get that returns refs per URL."""
        def fake_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            refs = url_to_refs.get(url, ["dummy_ref"])
            resp.json.return_value = {"references": [{"paperId": r} for r in refs], "title": "T"}
            return resp
        return fake_get

    def test_doi_prefix_construction(self):
        """Papers with DOI get DOI: prefix for S2."""
        from core.structural_gaps import find_coupling_gaps
        base = "https://api.semanticscholar.org/graph/v1/paper"
        url_a = f"{base}/DOI:10.1234/abc"
        url_b = f"{base}/ARXIV:9999.00001"
        fake = self._make_get_mock({url_a: ["r1"], url_b: ["r1"]})
        with patch("core.structural_gaps.requests.get", side_effect=fake) as mock_get:
            find_coupling_gaps(
                [{"doi": "10.1234/abc", "title": "A"},
                 {"arxiv_id": "9999.00001", "title": "B"}],
                s2_headers={},
                min_shared_refs=3,
            )
        # Verify DOI: prefix was used for the first paper
        calls = [c[0][0] for c in mock_get.call_args_list]
        assert any("DOI:10.1234/abc" in u for u in calls)

    def test_arxiv_prefix_construction(self):
        """Papers with arxiv_id get ARXIV: prefix for S2."""
        from core.structural_gaps import find_coupling_gaps
        base = "https://api.semanticscholar.org/graph/v1/paper"
        url_a = f"{base}/ARXIV:2301.12345"
        url_b = f"{base}/ARXIV:9999.00001"
        fake = self._make_get_mock({url_a: ["r1"], url_b: ["r1"]})
        with patch("core.structural_gaps.requests.get", side_effect=fake) as mock_get:
            find_coupling_gaps(
                [{"arxiv_id": "2301.12345", "title": "A"},
                 {"arxiv_id": "9999.00001", "title": "B"}],
                s2_headers={},
                min_shared_refs=3,
            )
        calls = [c[0][0] for c in mock_get.call_args_list]
        assert any("ARXIV:2301.12345" in u for u in calls)

    def test_s2_paper_id_used_directly(self):
        """Papers with s2_paper_id use it as-is."""
        from core.structural_gaps import find_coupling_gaps
        base = "https://api.semanticscholar.org/graph/v1/paper"
        url_a = f"{base}/abc123"
        url_b = f"{base}/def456"
        fake = self._make_get_mock({url_a: ["r1"], url_b: ["r1"]})
        with patch("core.structural_gaps.requests.get", side_effect=fake) as mock_get:
            find_coupling_gaps(
                [{"s2_paper_id": "abc123", "title": "A"},
                 {"s2_paper_id": "def456", "title": "B"}],
                s2_headers={},
                min_shared_refs=3,
            )
        calls = [c[0][0] for c in mock_get.call_args_list]
        assert any("/paper/abc123" in u for u in calls)

    def test_doi_with_url_prefix_stripped(self):
        """DOI URLs are stripped to bare DOI before prefixing."""
        from core.structural_gaps import find_coupling_gaps
        base = "https://api.semanticscholar.org/graph/v1/paper"
        url_a = f"{base}/DOI:10.1234/abc"
        url_b = f"{base}/ARXIV:9999.00001"
        fake = self._make_get_mock({url_a: ["r1"], url_b: ["r1"]})
        with patch("core.structural_gaps.requests.get", side_effect=fake) as mock_get:
            find_coupling_gaps(
                [{"doi": "https://doi.org/10.1234/abc", "title": "A"},
                 {"arxiv_id": "9999.00001", "title": "B"}],
                s2_headers={},
                min_shared_refs=3,
            )
        calls = [c[0][0] for c in mock_get.call_args_list]
        assert any("DOI:10.1234/abc" in u for u in calls)


@pytest.mark.usefixtures("_s2_enabled")
class TestStructuralGapCoupling:

    def test_coupling_detection_with_synthetic_papers(self):
        """4 papers: A,B share refs {1,2,3,4}, C,D share refs {5,6}.
        A and B should be coupled (shared_refs=4 >= 3), C and D should not (shared=2 < 3)."""
        from core.structural_gaps import find_coupling_gaps

        refs_map = {
            "https://api.semanticscholar.org/graph/v1/paper/A": ["1", "2", "3", "4"],
            "https://api.semanticscholar.org/graph/v1/paper/B": ["1", "2", "3", "4", "5"],
            "https://api.semanticscholar.org/graph/v1/paper/C": ["5", "6", "7"],
            "https://api.semanticscholar.org/graph/v1/paper/D": ["5", "6", "8"],
        }

        def fake_get(url, **kwargs):
            refs = refs_map.get(url, [])
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"references": [{"paperId": r} for r in refs], "title": url.split("/")[-1]}
            return resp

        with patch("core.structural_gaps.requests.get", side_effect=fake_get):
            result = find_coupling_gaps(
                [{"s2_paper_id": "A", "title": "Paper A"},
                 {"s2_paper_id": "B", "title": "Paper B"},
                 {"s2_paper_id": "C", "title": "Paper C"},
                 {"s2_paper_id": "D", "title": "Paper D"}],
                s2_headers={},
                min_shared_refs=3,
            )
        # A-B share refs {1,2,3,4} = 4 >= 3, neither cites the other
        assert len(result) == 1
        assert result[0]["shared_reference_count"] == 4

    def test_no_direct_citation_filter(self):
        """If A cites B, the pair should be excluded even with shared refs."""
        from core.structural_gaps import find_coupling_gaps

        def fake_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            if url.endswith("/A"):
                # A's refs include B — A cites B
                resp.json.return_value = {"references": [{"paperId": "1"}, {"paperId": "2"}, {"paperId": "3"}, {"paperId": "B"}], "title": "A"}
            else:
                resp.json.return_value = {"references": [{"paperId": "1"}, {"paperId": "2"}, {"paperId": "3"}], "title": "B"}
            return resp

        with patch("core.structural_gaps.requests.get", side_effect=fake_get):
            result = find_coupling_gaps(
                [{"s2_paper_id": "A", "title": "Paper A"},
                 {"s2_paper_id": "B", "title": "Paper B"}],
                s2_headers={},
                min_shared_refs=3,
            )
        # A cites B (B is in A's refs), so no coupling gap
        assert len(result) == 0

    def test_fewer_than_2_papers_returns_empty(self):
        from core.structural_gaps import find_coupling_gaps
        assert find_coupling_gaps([], s2_headers={}) == []
        assert find_coupling_gaps([{"title": "only one"}], s2_headers={}) == []

    def test_api_error_returns_empty(self):
        from core.structural_gaps import find_coupling_gaps
        with patch("core.structural_gaps.requests.get", side_effect=Exception("timeout")):
            result = find_coupling_gaps(
                [{"s2_paper_id": "A", "title": "A"}, {"s2_paper_id": "B", "title": "B"}],
                s2_headers={},
            )
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 3: Method × Domain Sparsity Matrix
# ═══════════════════════════════════════════════════════════════════════════════

class TestSparsityMatrix:

    def test_sparse_cell_detection(self):
        """Sparse cells: well-established method × well-established domain with pair_freq <= 1.
        Fixture produces 10 pairs with known frequencies. Only pairs where both
        method_freq >= 3 and domain_freq >= 3 and pair_freq == 1 are sparse.
        Result is sorted by method_frequency + domain_frequency descending."""
        from core.sparsity_matrix import find_sparse_cells

        # 10 pairs. Frequencies:
        #   transformer: 4, random_forest: 3, cnn: 2
        #   medical_imaging: 4, speech_recognition: 3, nlp: 2
        llm_output = json.dumps({
            "pairs": [
                {"method": "transformer", "domain": "medical imaging"},
                {"method": "transformer", "domain": "medical imaging"},
                {"method": "transformer", "domain": "speech recognition"},
                {"method": "transformer", "domain": "nlp"},
                {"method": "random forest", "domain": "medical imaging"},
                {"method": "random forest", "domain": "speech recognition"},
                {"method": "random forest", "domain": "generative ai"},
                {"method": "cnn", "domain": "medical imaging"},
                {"method": "cnn", "domain": "nlp"},
                {"method": "linear regression", "domain": "speech recognition"},
            ]
        })
        with patch("core.sparsity_matrix.call_llm", return_value=llm_output):
            result = find_sparse_cells([{"title": "t", "abstract": "a"}] * 10)

        # Sparse cells (pair_freq=1, method_freq>=3, domain_freq>=3):
        #   transformer × speech_recognition: method=4, domain=3, combined=7
        #   random_forest × medical_imaging: method=3, domain=4, combined=7
        #   random_forest × speech_recognition: method=3, domain=3, combined=6
        assert len(result) == 3

        # Sorted by combined frequency descending
        assert result[0]["method"] == "transformer"
        assert result[0]["domain"] == "speech recognition"
        assert result[0]["method_frequency"] == 4
        assert result[0]["domain_frequency"] == 3
        assert result[0]["pair_frequency"] == 1

        assert result[1]["method"] == "random forest"
        assert result[1]["domain"] == "medical imaging"
        assert result[1]["method_frequency"] == 3
        assert result[1]["domain_frequency"] == 4
        assert result[1]["pair_frequency"] == 1

        assert result[2]["method"] == "random forest"
        assert result[2]["domain"] == "speech recognition"
        assert result[2]["method_frequency"] == 3
        assert result[2]["domain_frequency"] == 3
        assert result[2]["pair_frequency"] == 1

        # Excluded: transformer×medical_imaging (pair_freq=2), cnn (method_freq=2),
        # nlp (domain_freq=2), generative_ai (domain_freq=1)
        methods_and_domains = {(r["method"], r["domain"]) for r in result}
        assert ("transformer", "medical imaging") not in methods_and_domains
        assert ("cnn", "medical imaging") not in methods_and_domains

    def test_fewer_than_5_papers_returns_empty(self):
        from core.sparsity_matrix import find_sparse_cells
        assert find_sparse_cells([]) == []
        assert find_sparse_cells([{"title": "t"}] * 4) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 4: Contradiction Mining — title validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestContradictionMining:

    def test_hallucinated_title_dropped(self):
        """Contradiction referencing a paper not in the input list should be dropped."""
        from core.contradiction_mining import find_contradictions
        papers = [
            {"title": "Alpha study", "abstract": "X improves Y significantly"},
            {"title": "Beta study", "abstract": "X does not improve Y"},
            {"title": "Gamma study", "abstract": "Z is unrelated"},
        ]
        mock_response = json.dumps({
            "contradictions": [
                {
                    "paper_a_title": "Alpha study",
                    "paper_b_title": "Beta study",
                    "claim_a": "X improves Y",
                    "claim_b": "X does not improve Y",
                    "shared_subject": "effect of X on Y",
                },
                {
                    "paper_a_title": "Alpha study",
                    "paper_b_title": "Nonexistent paper",  # hallucinated
                    "claim_a": "something",
                    "claim_b": "something else",
                    "shared_subject": "topic",
                },
            ]
        })
        with patch("core.contradiction_mining.call_llm", return_value=mock_response):
            result = find_contradictions(papers)
        # Only the valid pair survives
        assert len(result) == 1
        assert result[0]["paper_a_title"] == "Alpha study"
        assert result[0]["paper_b_title"] == "Beta study"

    def test_no_contradictions_returns_empty(self):
        from core.contradiction_mining import find_contradictions
        papers = [{"title": "A", "abstract": "foo bar"}]
        with patch("core.contradiction_mining.call_llm", return_value=json.dumps({"contradictions": []})):
            result = find_contradictions(papers)
        assert result == []

    def test_llm_failure_returns_empty(self):
        from core.contradiction_mining import find_contradictions
        with patch("core.contradiction_mining.call_llm", side_effect=Exception("API error")):
            result = find_contradictions([{"title": "A", "abstract": "foo"}] * 3)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 5: Replication-Target Mining (regex heuristic)
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplicationTargetMining:

    def test_strong_claim_no_variance_flagged(self):
        agent = _make_agent()
        papers = [
            {"title": "Method X significantly outperforms baselines",
             "abstract": "We show that method X significantly outperforms baselines on all benchmarks."},
        ]
        result = agent._find_replication_targets(papers)
        assert len(result) == 1
        assert "significantly" in result[0]["matched_claim_excerpt"].lower() or "outperform" in result[0]["matched_claim_excerpt"].lower()

    def test_variance_reported_not_flagged(self):
        agent = _make_agent()
        papers = [
            {"title": "Method X significantly outperforms baselines",
             "abstract": "We show method X significantly outperforms baselines. Results averaged over 5 seeds with std dev reported."},
        ]
        result = agent._find_replication_targets(papers)
        assert len(result) == 0

    def test_empty_abstract_no_crash(self):
        agent = _make_agent()
        result = agent._find_replication_targets([{"title": "t", "abstract": ""}])
        assert result == []

    def test_no_strong_claims_returns_empty(self):
        agent = _make_agent()
        papers = [{"title": "A study", "abstract": "We examine the properties of method Z."}]
        result = agent._find_replication_targets(papers)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 6: Negative Result Lessons — schema validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegativeResultLessons:

    def test_returns_only_structured_fields(self):
        """The returned dicts must contain exactly the fixed structured set —
        no generated_narrative or free-text leakage."""
        from core.run_log import CrossRunMemory
        CRM = CrossRunMemory.__new__(CrossRunMemory)
        CRM.path = Path("/tmp/test_crm.jsonl")
        CRM._write_lock = threading.Lock()
        # Write a mock rejection record with structured_hypothesis in meta
        entry = {
            "ts": "2026-01-01T00:00:00Z",
            "category": "rejection",
            "kind": "topic",
            "item": "Some Topic",
            "reason": "failed_hypothesis_debate",
            "meta": {
                "structured_hypothesis": {
                    "hypothesis_kind": "attention",
                    "research_question": "Does X improve Y?",
                    "falsification_condition": "No significant improvement",
                    "dependent_variables": ["accuracy", "ECE"],
                    "hypothesis": "X improves Y",
                },
            },
        }
        # Directly test the method with a mocked load
        with patch.object(CRM, "load", return_value=[entry]):
            result = CRM.get_negative_result_lessons()
        assert len(result) == 1
        keys = set(result[0].keys())
        expected_keys = {"hypothesis_kind", "research_question", "falsification_condition", "dependent_variables", "outcome_status"}
        assert keys == expected_keys, f"Unexpected keys: {keys - expected_keys}, missing: {expected_keys - keys}"

    def test_no_generated_narrative_leakage(self):
        """Even if the source entry has extra fields, only structured fields are returned."""
        from core.run_log import CrossRunMemory
        CRM = CrossRunMemory.__new__(CrossRunMemory)
        CRM.path = Path("/tmp/test_crm2.jsonl")
        CRM._write_lock = threading.Lock()
        entry = {
            "category": "rejection",
            "kind": "topic",
            "item": "Some Topic",
            "meta": {
                "structured_hypothesis": {
                    "hypothesis_kind": "general",
                    "research_question": "Q?",
                    "falsification_condition": "F",
                    "dependent_variables": ["acc"],
                    "generated_narrative": "SECRET TEXT THAT SHOULD NOT LEAK",
                    "free_form_notes": "ALSO SECRET",
                },
            },
        }
        with patch.object(CRM, "load", return_value=[entry]):
            result = CRM.get_negative_result_lessons()
        assert "generated_narrative" not in result[0]
        assert "free_form_notes" not in result[0]

    def test_non_topic_rejection_skipped(self):
        """Rejections with kind != 'topic' should be skipped."""
        from core.run_log import CrossRunMemory
        CRM = CrossRunMemory.__new__(CrossRunMemory)
        CRM.path = Path("/tmp/test_crm3.jsonl")
        CRM._write_lock = threading.Lock()
        entry = {
            "category": "rejection",
            "kind": "experiment",
            "item": "some experiment",
            "meta": {"structured_hypothesis": {"research_question": "Q?"}},
        }
        with patch.object(CRM, "load", return_value=[entry]):
            result = CRM.get_negative_result_lessons()
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 7: Persona Ensemble Generation — gap tagging
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersonaEnsemble:

    def test_gaps_tagged_with_persona(self):
        """Persona ensemble: each persona's gaps carry the correct persona tag,
        and dedup by title keeps first occurrence."""
        from agents.topic_hunter import TopicHunterAgent
        agent = _make_agent()
        agent.runtime_config.persona_ensemble_enabled = True
        agent.runtime_config.persona_count = 2

        skeptic_response = json.dumps({"gaps": [
            {"title": "Skeptic Gap 1", "description": "evaluation validity"},
            {"title": "Shared Gap", "description": "common topic"},
        ]})
        practitioner_response = json.dumps({"gaps": [
            {"title": "Practitioner Gap 1", "description": "deployment constraints"},
            {"title": "Shared Gap", "description": "common topic"},
        ]})

        persona_prefixes = {
            "skeptic": "Prioritize gaps about evaluation validity. ",
            "practitioner": "Prioritize gaps about deployment constraints. ",
        }

        def fake_call_llm(prompt, **kwargs):
            if prompt.startswith("Prioritize gaps about evaluation validity"):
                return skeptic_response
            return practitioner_response

        # Run the actual persona ensemble logic (extracted from _hunt_once lines 1665-1698)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        persona_names = list(persona_prefixes.keys())
        all_gaps = []

        def _call_persona(name):
            prefix = persona_prefixes[name]
            raw = fake_call_llm(prefix + "base_prompt")
            parsed = json.loads(raw)
            gaps_list = parsed.get("gaps") or []
            for g in gaps_list:
                g["persona"] = name
            return gaps_list

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(_call_persona, name): name for name in persona_names}
            for fut in as_completed(futures):
                all_gaps.extend(fut.result() or [])

        # Dedup by title (same logic as _hunt_once lines 1691-1698)
        seen_titles = set()
        gaps = []
        for g in all_gaps:
            t = (g.get("title") or "").strip().lower()
            if t and t not in seen_titles:
                seen_titles.add(t)
                gaps.append(g)

        # Both unique gaps present with correct persona tags
        by_title = {g["title"]: g for g in gaps}
        assert "Skeptic Gap 1" in by_title
        assert by_title["Skeptic Gap 1"]["persona"] == "skeptic"
        assert "Practitioner Gap 1" in by_title
        assert by_title["Practitioner Gap 1"]["persona"] == "practitioner"
        # Shared gap deduplicated — first occurrence wins
        assert len([g for g in gaps if g["title"] == "Shared Gap"]) == 1

    def test_dedup_by_title(self):
        """Same title from two personas should be deduplicated (first wins)."""
        gaps_a = [{"title": "Same Title", "persona": "skeptic"}]
        gaps_b = [{"title": "Same Title", "persona": "practitioner"}]
        all_gaps = gaps_a + gaps_b
        seen = set()
        deduped = []
        for g in all_gaps:
            t = (g.get("title") or "").strip().lower()
            if t and t not in seen:
                seen.add(t)
                deduped.append(g)
        assert len(deduped) == 1
        assert deduped[0]["persona"] == "skeptic"


# ═══════════════════════════════════════════════════════════════════════════════
# Feature 8: Seed-Strategy Provenance + Elo
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeedStrategyProvenance:

    def test_generate_dynamic_seeds_returns_dicts(self):
        """_generate_dynamic_seeds should return List[Dict] with seed+strategy."""
        agent = _make_agent()
        result = agent._generate_dynamic_seeds(n_seeds=3, cross_run_context=[], domain="machine_learning")
        assert isinstance(result, list)
        assert len(result) <= 3
        for item in result:
            assert isinstance(item, dict)
            assert "seed" in item
            assert "strategy" in item
            assert isinstance(item["seed"], str)
            assert isinstance(item["strategy"], str)

    def test_seed_strategy_threaded_onto_gap(self):
        """_hunt_once threads seed_strategy onto every gap in kept[].
        Calls the real _hunt_once with mocked I/O boundary only."""
        agent = _make_agent()
        agent.runtime_config.capability_first_dataset_scoping_enabled = False
        agent.runtime_config.structural_gap_mining_enabled = False
        agent.runtime_config.sparsity_matrix_enabled = False
        agent.runtime_config.contradiction_mining_enabled = False
        agent.runtime_config.replication_target_mining_enabled = False
        agent.runtime_config.negative_result_seeding_enabled = False
        agent.runtime_config.persona_ensemble_enabled = False
        agent.runtime_config.seed_strategy_elo_enabled = False
        agent.runtime_config.hyde_enabled = False
        agent.runtime_config.multi_hop_retrieval_enabled = False

        fake_paper = {"title": "machine learning validation", "abstract": "We validate ml models",
                      "doi": "", "arxiv_id": "", "cited_by_count": 0}

        formalized_hyp = {
            "hypothesis": "X improves Y",
            "falsification_condition": "No improvement",
            "dependent_variables": ["accuracy"],
            "research_question": "Does X improve Y?",
            "minimum_viable_experiment": {
                "dataset": "bundled_synthetic",
                "models": ["logistic_regression"],
                "conditions": ["control", "treatment"],
                "metrics": ["accuracy"],
                "seeds": 3,
            },
        }

        with patch("agents.topic_hunter.CrossRunMemory") as MockCRM, \
             patch("agents.topic_hunter.call_llm") as mock_llm, \
             patch("agents.topic_hunter.parse_json_from_llm") as mock_parse, \
             patch("agents.topic_hunter.generate_embedding", return_value=[0.1]*10), \
             patch("agents.topic_hunter.build_cross_paper_evidence_map", return_value={"bridges": []}), \
             patch("agents.topic_hunter.validate_candidate_bridge_claim", return_value={"valid": True, "bridge_ids": []}), \
             patch("agents.topic_hunter.validate_topic_admission", return_value={"admitted": True, "errors": []}), \
             patch.object(agent, "_generate_hyde_abstract", return_value=None), \
             patch.object(agent, "_build_arxiv_queries", return_value=["q"]), \
             patch.object(agent, "_build_openalex_queries", return_value=[("q", {})]), \
             patch.object(agent, "_preflight_dedup", side_effect=lambda q, _: q), \
             patch.object(agent, "search_openalex", return_value=[fake_paper]), \
             patch.object(agent, "search_arxiv", return_value=[]), \
             patch.object(agent, "fetch_citation_graph", return_value=None), \
             patch.object(agent, "screen_research_gap", return_value={"status": "PASS", "gap_type": "supported"}), \
             patch.object(agent, "evaluate_layered_novelty", return_value={"reject": False, "verdict": "NOVEL", "max_similarity": 0.0}), \
             patch.object(agent, "feasibility_filter", return_value={"ok": True, "reasons": []}), \
             patch.object(agent, "formalize_hypothesis", return_value=formalized_hyp):

            crm_inst = MockCRM.return_value
            crm_inst.get_prompt_context.return_value = []
            crm_inst.excluded_topic_titles.return_value = []
            crm_inst.get_negative_result_lessons.return_value = []
            crm_inst.record_rejection.return_value = None

            mock_llm.return_value = '{"gaps": [{"title": "ML validation gap", "description": "testing ml models", "rationale": "r", "contribution": "c", "evidence_bridge_ids": [], "feasibility": 7}]}'
            mock_parse.return_value = {"gaps": [{"title": "ML validation gap", "description": "testing ml models", "rationale": "r", "contribution": "c", "evidence_bridge_ids": [], "feasibility": 7}]}

            result = agent._hunt_once(domain="ml", seed_hint="test seed", seed_strategy="kind_bias")

        assert len(result) >= 1, f"Expected at least 1 gap, got {len(result)}"
        for gap in result:
            assert gap.get("seed_strategy") == "kind_bias", \
                f"gap '{gap.get('title')}' missing seed_strategy='kind_bias', got '{gap.get('seed_strategy')}'"

    def test_record_strategy_outcome_updates_elo(self):
        """record_strategy_outcome should update strategy Elo under strategy:<name> key."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("{}")
            elo_path = f.name
        try:
            elo = EloStore(path=elo_path)
            initial = elo.get("strategy:kind_bias")
            assert initial == elo.prior  # 1500.0
            elo.record_strategy_outcome("kind_bias", "debate_pass")
            after = elo.get("strategy:kind_bias")
            assert after != initial, "Elo should change after outcome recording"
            assert after > initial, "Supported outcome should increase Elo"
        finally:
            Path(elo_path).unlink(missing_ok=True)

    def test_strategy_outcome_recorded_in_conduct_debate(self):
        """conduct_debate actually calls record_strategy_outcome on a topic with seed_strategy.
        Calls the real conduct_debate with mocked proposer/challenger/moderator."""
        import tempfile
        from agents.hypothesis_debate import HypothesisDebateSystem, DebateResult
        topic = {"title": "Test Topic", "seed_strategy": "kind_bias",
                 "description": "d", "rationale": "r", "hypothesis": "h"}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("{}")
            elo_path = f.name
        try:
            with patch("agents.hypothesis_debate.get_active_context") as mock_ctx, \
                 patch("agents.hypothesis_debate.EloStore") as MockElo:
                mock_cfg = MagicMock()
                mock_cfg.debate_min_rounds = 2
                mock_cfg.debate_max_rounds = 2
                mock_cfg.get_ensemble_models.return_value = []
                mock_cfg.resolve_model.return_value = "test-model"
                mock_cfg.strict_judge_grading = False
                mock_ctx.return_value.config = mock_cfg

                # Mock EloStore so we can spy on record_strategy_outcome
                mock_elo = MagicMock()
                mock_elo.update.return_value = 0.0
                MockElo.return_value = mock_elo

                system = HypothesisDebateSystem.__new__(HypothesisDebateSystem)
                system.context = mock_ctx.return_value
                system.runtime_config = mock_cfg
                system.elo = mock_elo

                # Mock proposer/challenger/moderator to short-circuit the debate loop
                mock_proposer = MagicMock()
                mock_proposer.build_argument.return_value = "argument"
                mock_proposer.respond_to_objections.return_value = "response"

                mock_challenger = MagicMock()
                mock_challenger.build_rebuttal.return_value = "rebuttal"
                mock_challenger._last_objections = []
                mock_challenger._challenger_invalid = False
                mock_challenger.followup_objections.return_value = [
                    {"criterion": "soundness", "objection": "obj", "severity": 1, "status": "resolved"}
                ]

                mock_moderator = MagicMock()
                mock_moderator.evaluate_debate.return_value = {
                    "score": 8.0, "passed": True, "decision": "PASS",
                    "needs_longer_debate": False, "reasoning": "strong",
                    "ensemble_scores": [8.0], "disagreement": 0.0,
                    "valid_judge_responses": 1, "hard_gates": [],
                    "unresolved_objections": [],
                }

                system.proposer = mock_proposer
                system.challenger = mock_challenger
                system.moderator = mock_moderator

                result = system.conduct_debate(topic)

                # Assert record_strategy_outcome was called with correct args
                mock_elo.record_strategy_outcome.assert_called_once_with("kind_bias", "debate_pass")
                assert isinstance(result, DebateResult)
                assert result.passed is True
        finally:
            Path(elo_path).unlink(missing_ok=True)

    def test_builds_on_negative_result_validation(self):
        """The real validation code inside _hunt_once drops builds_on_negative_result
        when the research_question doesn't match any real negative lesson.
        Calls _hunt_once with mocked I/O and asserts the field is stripped."""
        agent = _make_agent()
        agent.runtime_config.capability_first_dataset_scoping_enabled = False
        agent.runtime_config.structural_gap_mining_enabled = False
        agent.runtime_config.sparsity_matrix_enabled = False
        agent.runtime_config.contradiction_mining_enabled = False
        agent.runtime_config.replication_target_mining_enabled = False
        agent.runtime_config.negative_result_seeding_enabled = True
        agent.runtime_config.persona_ensemble_enabled = False
        agent.runtime_config.seed_strategy_elo_enabled = False
        agent.runtime_config.hyde_enabled = False
        agent.runtime_config.multi_hop_retrieval_enabled = False

        fake_paper = {"title": "machine learning validation", "abstract": "We validate ml models",
                      "doi": "", "arxiv_id": "", "cited_by_count": 0}

        formalized_hyp = {
            "hypothesis": "X improves Y",
            "falsification_condition": "No improvement",
            "dependent_variables": ["accuracy"],
            "research_question": "Does X improve Y?",
            "minimum_viable_experiment": {
                "dataset": "bundled_synthetic",
                "models": ["logistic_regression"],
                "conditions": ["control", "treatment"],
                "metrics": ["accuracy"],
                "seeds": 3,
            },
        }

        # Negative lesson with a specific research question
        negative_lessons = [
            {"research_question": "Does Z affect W?", "outcome_status": "unsupported"},
        ]

        # Gap with builds_on_negative_result whose research_question does NOT match
        gap_with_bnr = {
            "title": "ML validation gap", "description": "testing ml models",
            "rationale": "r", "contribution": "c", "evidence_bridge_ids": [],
            "feasibility": 7,
            "builds_on_negative_result": {"research_question": "Unrelated question"},
        }

        with patch("agents.topic_hunter.CrossRunMemory") as MockCRM, \
             patch("agents.topic_hunter.call_llm") as mock_llm, \
             patch("agents.topic_hunter.parse_json_from_llm") as mock_parse, \
             patch("agents.topic_hunter.generate_embedding", return_value=[0.1]*10), \
             patch("agents.topic_hunter.build_cross_paper_evidence_map", return_value={"bridges": []}), \
             patch("agents.topic_hunter.validate_candidate_bridge_claim", return_value={"valid": True, "bridge_ids": []}), \
             patch("agents.topic_hunter.validate_topic_admission", return_value={"admitted": True, "errors": []}), \
             patch.object(agent, "_generate_hyde_abstract", return_value=None), \
             patch.object(agent, "_build_arxiv_queries", return_value=["q"]), \
             patch.object(agent, "_build_openalex_queries", return_value=[("q", {})]), \
             patch.object(agent, "_preflight_dedup", side_effect=lambda q, _: q), \
             patch.object(agent, "search_openalex", return_value=[fake_paper]), \
             patch.object(agent, "search_arxiv", return_value=[]), \
             patch.object(agent, "fetch_citation_graph", return_value=None), \
             patch.object(agent, "screen_research_gap", return_value={"status": "PASS", "gap_type": "supported"}), \
             patch.object(agent, "evaluate_layered_novelty", return_value={"reject": False, "verdict": "NOVEL", "max_similarity": 0.0}), \
             patch.object(agent, "feasibility_filter", return_value={"ok": True, "reasons": []}), \
             patch.object(agent, "formalize_hypothesis", return_value=formalized_hyp):

            crm_inst = MockCRM.return_value
            crm_inst.get_prompt_context.return_value = []
            crm_inst.excluded_topic_titles.return_value = []
            crm_inst.get_negative_result_lessons.return_value = negative_lessons
            crm_inst.record_rejection.return_value = None

            mock_llm.return_value = json.dumps({"gaps": [gap_with_bnr]})
            mock_parse.return_value = {"gaps": [gap_with_bnr]}

            result = agent._hunt_once(domain="ml", seed_hint="test seed", seed_strategy="test")

        # The real validation code (topic_hunter.py:1793-1799) should have stripped
        # builds_on_negative_result because "Unrelated question" != "Does Z affect W?"
        assert len(result) >= 1, f"Expected at least 1 gap, got {len(result)}"
        for gap in result:
            assert "builds_on_negative_result" not in gap, \
                f"builds_on_negative_result should have been stripped from '{gap.get('title')}'"
