"""Finish-ready per-type behavior: R4/R5/R7/R14 legacy gates (owner directive).

Offline only (no API keys, no network). Proves:
- R4: per-type score weights + thresholds, experimental identical to legacy.
- R7: reproducibility_dossier type-conditioned; unknown/None = strict legacy.
- R5: DOI retry with backoff; fail-closed default (fake DOI still fails).
- R14: stale-cache degrade labelled; no cache still terminal.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEGACY_WEIGHTS = {"hard": 0.35, "math": 0.15, "code": 0.15, "soft": 0.20, "hall": 0.15}


def _legacy_overall(hard, math, code, soft, hall):
    return (
        0.35 * hard + 0.15 * math + 0.15 * code + 0.20 * soft + 0.15 * hall
    )


# ---------------------------------------------------------------------------
# R4: per-type weights + thresholds
# ---------------------------------------------------------------------------

class TestR4PerTypeWeights:
    def test_experimental_weights_identical_to_legacy(self):
        from agents.supervisor import SupervisorAgent

        assert SupervisorAgent.weights_for("experimental") == LEGACY_WEIGHTS
        assert SupervisorAgent.weights_for(None) == LEGACY_WEIGHTS
        assert SupervisorAgent.weights_for("nonsense-type") == LEGACY_WEIGHTS

    def test_all_type_weights_sum_to_one(self):
        from agents.supervisor import SupervisorAgent

        for ptype, weights in SupervisorAgent.PER_TYPE_WEIGHTS.items():
            assert abs(sum(weights.values()) - 1.0) < 1e-9, ptype
            assert set(weights) == {"hard", "math", "code", "soft", "hall"}, ptype

    def test_non_experimental_differs_or_equal_with_reason(self):
        """Each non-experimental entry either differs from legacy or carries
        an explicit parity reason — no silent drift, no undocumented values."""
        from agents.supervisor import SupervisorAgent

        experimental = SupervisorAgent.weights_for("experimental")
        assert len(SupervisorAgent.PER_TYPE_WEIGHTS) > 1
        for ptype, weights in SupervisorAgent.PER_TYPE_WEIGHTS.items():
            if ptype == "experimental":
                continue
            reason = SupervisorAgent.SCORE_WEIGHT_REASONS.get(ptype, "")
            assert isinstance(reason, str) and reason.strip(), ptype
            assert dict(weights) != dict(experimental) or "parity" in reason.lower(), ptype

    def test_experimental_scoring_arithmetic_unchanged(self):
        from agents.supervisor import SupervisorAgent

        weights = SupervisorAgent.weights_for("experimental")
        scores = {"hard": 9.0, "math": 10.0, "code": 8.0, "soft": 7.5, "hall": 6.0}
        per_type = sum(weights[k] * scores[k] for k in scores)
        assert per_type == pytest.approx(_legacy_overall(**scores))

    def test_threshold_falls_back_to_config(self):
        from agents.supervisor import SupervisorAgent
        from core.config import config

        agent = SupervisorAgent.__new__(SupervisorAgent)
        agent.context = None  # runtime_config property falls back to global config
        assert agent.threshold_for("experimental") == float(config.supervisor_threshold)
        assert agent.threshold_for("review") == float(config.supervisor_threshold)
        assert agent.threshold_for(None) == float(config.supervisor_threshold)
        assert agent.threshold_for("unknown-type") == float(config.supervisor_threshold)


# ---------------------------------------------------------------------------
# R7: type-conditioned reproducibility dossier
# ---------------------------------------------------------------------------

def _full_plan_and_outputs(tmp_path):
    plan = {"experiments": [{"falsifiable_prediction": "A > B", "baselines": ["B"], "statistical_test": "t-test"}]}
    raw = tmp_path / "raw.json"
    raw.write_text("{}", encoding="utf-8")
    complete = {"x": {"raw_results_path": str(raw), "code": "print(1)", "contract_hash": "abc"}}
    return plan, complete


class TestR7DossierConditioning:
    def test_experimental_keeps_full_dossier(self, tmp_path):
        from core.verification import reproducibility_dossier

        plan, complete = _full_plan_and_outputs(tmp_path)
        assert reproducibility_dossier(plan, complete, paper_type="experimental")["passed"]
        assert not reproducibility_dossier(plan, {}, paper_type="experimental")["passed"]

    def test_strict_default_unknown_and_none(self):
        from core.verification import reproducibility_dossier

        for ptype in (None, "unknown-type", ""):
            empty = reproducibility_dossier(None, None, paper_type=ptype)
            assert not empty["passed"]
            assert empty["checks"]["contract_provenance"] is False
            assert empty["checks"]["limitations_disclosed"] is False

    def test_thesis_types_proofs_satisfy_dossier(self):
        from core.verification import reproducibility_dossier

        thesis = {
            "claim": "Synthesis workflows improve cumulative insight",
            "bridges": [{"statement": "Prior coupling studies map research fronts through shared references"}],
            "steelman": "Evidence may be insufficient",
            "concession": "Scope limited to published work",
        }
        plan = {"paper_type": "analytical", "plan_mode": "non_empirical", "experiments": []}
        for ptype in ("analytical", "argumentative"):
            result = reproducibility_dossier(plan, {}, paper_type=ptype, thesis=thesis)
            assert result["passed"], result
            assert result["mode"] == "thesis"
        # Light-plan shape (thesis_slot + claims) also satisfies.
        light = {"plan_mode": "non_empirical", "experiments": [],
                 "thesis_slot": "Synthesis workflows improve insight",
                 "claims": ["Prior coupling studies map research fronts through shared references"]}
        result = reproducibility_dossier(light, {}, paper_type="analytical")
        assert result["passed"], result
        # Absent thesis stays fail-closed (same verdict as legacy strict).
        assert not reproducibility_dossier(plan, {}, paper_type="analytical")["passed"]
        assert not reproducibility_dossier(plan, {}, paper_type="descriptive")["passed"]

    def test_survey_coverage_tables(self):
        from core.verification import reproducibility_dossier

        plan = {"paper_type": "survey", "plan_mode": "non_empirical", "experiments": []}
        bundle = {
            "descriptives": {"columns": {"q1": {"mean": 3.2}}},
            "nonresponse": {"item_rates": {"q1": 0.02}, "unit_rate": 0.85},
        }
        result = reproducibility_dossier(plan, {}, paper_type="survey",
                                         survey_analysis=bundle,
                                         instrument={"questionnaire": "q", "codebook_hash": "h", "response_hash": "r"})
        assert result["passed"], result
        assert result["mode"] == "survey"
        assert not reproducibility_dossier(plan, {}, paper_type="survey")["passed"]
        # Coverage without nonresponse still fails (fail-closed, no silent pass).
        assert not reproducibility_dossier(
            plan, {}, paper_type="survey",
            survey_analysis={"descriptives": {"columns": {"q1": {}}}})["passed"]

    def test_case_triangulation(self):
        from core.verification import reproducibility_dossier

        plan = {"paper_type": "case_study", "plan_mode": "non_empirical", "experiments": []}
        protocol = {
            "pack": {"passed": True},
            "triangulation": {"single_source_flags": []},
            "rival": {"passed": True},
            "limitations": [],
        }
        result = reproducibility_dossier(plan, {}, paper_type="case_study", case_protocol=protocol)
        assert result["passed"], result
        assert result["mode"] == "case"
        assert not reproducibility_dossier(plan, {}, paper_type="case_study")["passed"]
        # Rival missing stays fail-closed.
        no_rival = dict(protocol)
        no_rival["rival"] = {"passed": False}
        assert not reproducibility_dossier(plan, {}, paper_type="case", case_protocol=no_rival)["passed"]


# ---------------------------------------------------------------------------
# R5: citation retry, fail-closed default
# ---------------------------------------------------------------------------

class TestR5CitationRetry:
    def test_flaky_resolver_passes_after_retry(self, monkeypatch):
        from core import verification

        calls = []

        def flaky(doi):
            calls.append(doi)
            if len(calls) == 1:
                return {"resolved": False, "doi": doi, "error": "HTTP 503 Service Unavailable"}
            return {"resolved": True, "doi": doi, "title": "Real Paper", "authors": []}

        monkeypatch.setattr(verification, "resolve_doi", flaky)
        monkeypatch.setattr(verification, "_DOI_RETRY_BACKOFF_SEC", 0)
        result = verification.verify_citations("See doi:10.1234/flaky01")
        assert result["passed"] is True
        assert len(calls) == 2

    def test_fake_doi_still_fails_no_retry(self, monkeypatch):
        from core import verification

        calls = []

        def fake_resolve(doi):
            calls.append(doi)
            return {"resolved": False, "doi": doi, "error": "not found"}

        monkeypatch.setattr(verification, "resolve_doi", fake_resolve)
        result = verification.verify_citations("Claim supported by doi:10.9999/fake.doi.123")
        assert result["passed"] is False
        assert result["score"] < 10
        assert len(calls) == 1  # definitive miss fails fast, no retry storm

    def test_persistent_transient_fails_after_bounded_attempts(self, monkeypatch):
        from core import verification

        calls = []

        def down(doi):
            calls.append(doi)
            return {"resolved": False, "doi": doi, "error": "connection reset by peer"}

        monkeypatch.setattr(verification, "resolve_doi", down)
        monkeypatch.setattr(verification, "_DOI_RETRY_BACKOFF_SEC", 0)
        result = verification.verify_citations("See doi:10.1234/down01")
        assert result["passed"] is False
        assert len(calls) == verification._DOI_MAX_ATTEMPTS

    def test_quarantine_default_unchanged(self, monkeypatch):
        from core import verification

        monkeypatch.setattr(
            verification, "resolve_doi",
            lambda doi: {"resolved": False, "doi": doi, "error": "not found"},
        )
        result = verification.verify_citations("See doi:10.9999/fake.doi.123")
        assert result["quarantined"] == []
        assert result["passed"] is False


# ---------------------------------------------------------------------------
# R14: stale-cache degrade
# ---------------------------------------------------------------------------

def _write_openalex_artifact(cache_dir: Path, name="abc123.json"):
    artifact = {
        "source": "openalex",
        "url": "https://api.openalex.org/works?search=test",
        "status": "verified",
        "content": {
            "results": [
                {"title": "Cached study on attention efficiency",
                 "abstract": "A cached abstract about efficient attention.",
                 "doi": "10.1234/cached01", "year": 2024},
                {"title": "Cached study on sparsity", "abstract": "Another cached abstract.",
                 "doi": "10.1234/cached02", "year": 2023},
            ]
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / name).write_text(json.dumps(artifact), encoding="utf-8")


def _make_hunter(tmp_path, cache_dir=None):
    from agents.topic_hunter import TopicHunterAgent
    from core.sources import SourceClient

    agent = TopicHunterAgent.__new__(TopicHunterAgent)
    agent.context = None
    cfg = MagicMock()
    cfg.research_domain = "machine_learning"
    cfg.openalex_email = "test@example.com"
    cfg.semantic_scholar_api_key = ""
    cfg.topic_exploration_seed = 42
    cfg.topic_exploration_every = 4
    cfg.output_dir = str(tmp_path)
    cfg.topic_fallback_enabled = False
    cfg.topic_fallback_max_candidates = 1
    cfg.openreview_enabled = False
    agent.runtime_config = cfg
    agent.vector_memory = MagicMock()
    agent.client = MagicMock()
    agent.openalex_headers = {}
    agent.s2_headers = {}
    agent.base_urls = {"openalex": "https://api.openalex.org",
                       "crossref": "https://api.crossref.org",
                       "semantic_scholar": "https://api.semanticscholar.org/graph/v1"}
    agent._selection_counter = 1
    import random as _random
    agent._rng = _random.Random(42)
    agent.source_client = SourceClient(str(cache_dir or (tmp_path / "source_cache")))
    agent.rejection_log = []
    agent.source_health = {}
    agent._excluded_titles_cache = None
    agent._iteration_failures = 0
    agent._run_query_cache = {}
    agent._run_query_cache_lock = threading.Lock()
    agent._dataset_catalog_cache = None
    agent._attempted_seeds = set()
    agent._arxiv_consecutive_failures = 0
    return agent


class TestR14StaleCacheDegrade:
    def test_loader_reads_prior_responses_labelled_stale(self, tmp_path):
        from agents.topic_hunter import TopicHunterAgent

        cache = tmp_path / "source_cache"
        _write_openalex_artifact(cache)
        papers = TopicHunterAgent._load_stale_cached_papers(cache)
        assert len(papers) == 2
        assert all(p["stale"] is True for p in papers)
        assert any("attention efficiency" in p["title"] for p in papers)

    def test_loader_skips_non_paper_cache_shapes(self, tmp_path):
        from agents.topic_hunter import TopicHunterAgent

        cache = tmp_path / "source_cache"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "doaj_journals.json").write_text(
            json.dumps({"fetched_at": 0, "titles": ["Nature"]}), encoding="utf-8")
        (cache / "oa_text.json").write_text(
            json.dumps({"content": {"url": "https://x", "text": "full text"}}), encoding="utf-8")
        assert TopicHunterAgent._load_stale_cached_papers(cache) == []
        assert TopicHunterAgent._load_stale_cached_papers(tmp_path / "missing") == []

    def test_degrade_uses_cache_labelled(self, tmp_path):
        from agents.topic_hunter import TopicHunterAgent

        cache = tmp_path / "source_cache"
        _write_openalex_artifact(cache)
        agent = _make_hunter(tmp_path, cache)
        topics = agent._stale_cache_degrade("machine_learning",
                                            [{"seed": "attention efficiency", "strategy": "kind_bias"}])
        assert topics, "expected stale-cache candidates"
        topic = topics[0]
        assert topic.get("stale_corpus") is True
        assert "STALE" in str(topic.get("banner", "")).upper()
        assert topic.get("limitations"), "limitation entry required"
        assert topic.get("fallback_provenance", {}).get("reason") == "stale_cache_degrade"
        evidence = topic.get("literature_evidence") or []
        assert evidence and all(p.get("stale") is True for p in evidence)
        # Never fabricated: evidence titles are verbatim cached responses.
        assert any("attention efficiency" in p["title"] for p in evidence)

    def test_degrade_no_cache_returns_none(self, tmp_path):
        agent = _make_hunter(tmp_path, tmp_path / "empty_cache")
        assert agent._stale_cache_degrade("machine_learning", [{"seed": "x"}]) is None

    def test_discover_topics_stale_cache_labelled(self, tmp_path, monkeypatch):
        from agents.topic_hunter import ResearchSourceUnavailable, TopicHunterAgent

        cache = tmp_path / "source_cache"
        _write_openalex_artifact(cache)
        agent = _make_hunter(tmp_path, cache)
        monkeypatch.setattr(TopicHunterAgent, "_generate_dynamic_seeds",
                            lambda self, *a, **k: [{"seed": "attention efficiency", "strategy": "kind_bias"}])
        monkeypatch.setattr(TopicHunterAgent, "_hunt_once", lambda self, *a, **k: [])
        agent.source_health = {"openalex": {"ok": False, "error": "down"},
                               "arxiv": {"ok": False, "error": "down"},
                               "semantic_scholar": {"ok": False, "error": "down"}}
        ranked = agent.discover_topics(domain="machine_learning", n_parallel=1)
        assert ranked, "expected stale-cache candidates instead of terminal"
        assert ranked[0].get("stale_corpus") is True
        assert ranked[0].get("limitations")

    def test_discover_topics_no_cache_still_terminal(self, tmp_path, monkeypatch):
        from agents.topic_hunter import ResearchSourceUnavailable, TopicHunterAgent

        agent = _make_hunter(tmp_path, tmp_path / "empty_cache")
        monkeypatch.setattr(TopicHunterAgent, "_generate_dynamic_seeds",
                            lambda self, *a, **k: [{"seed": "attention efficiency", "strategy": "kind_bias"}])
        monkeypatch.setattr(TopicHunterAgent, "_hunt_once", lambda self, *a, **k: [])
        agent.source_health = {"openalex": {"ok": False, "error": "down"},
                               "arxiv": {"ok": False, "error": "down"}}
        with pytest.raises(ResearchSourceUnavailable):
            agent.discover_topics(domain="machine_learning", n_parallel=1)
