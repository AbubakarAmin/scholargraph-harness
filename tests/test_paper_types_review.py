"""M2: review pipeline — systematic/scoping review (proposal §11-M2).

Tests (6, per tracker):
- 30-paper mock reconciles exactly.
- Off-by-one PRISMA → blocked.
- Closed-access marked full_text_unavailable, run continues abstract-only.
- Extraction completeness 0.85 → blocked.
- Recall 3/5 → demote to scoping + banner (not terminal).
- Overlap 0.80 review text passes (R11: review threshold 0.85+).

Plus wiring guards (Jaccard boundary, OA order, skeleton/graph) and one
mocked scoping-L2 E2E (engineering never runs, OA-only, within caps).
"""

from __future__ import annotations

import pytest

DOI = lambda i: f"10.1234/review{i:02d}"


def _paper(i, **over):
    base = {
        "title": f"Review study {i} on synthesis methods",
        "abstract": f"This investigation examines synthesis method {i} with controlled evaluation.",
        "doi": DOI(i),
        "design": "cohort",
        "sample_n": 100 + i,
        "effect": 0.2 + i * 0.01,
        "metric": "accuracy",
        "limitation": "single site",
    }
    base.update(over)
    return base


def _balanced_prisma(n_identified=30, n_dup=2, n_excl_ta=8, n_included=20):
    screened = n_identified - n_dup
    eligible = screened - n_excl_ta
    assert eligible == n_included, "test helper must balance eligible==included (excluded_fulltext=0)"
    return {
        "identified": n_identified,
        "screened": screened,
        "duplicates_excluded": n_dup,
        "eligible": eligible,
        "excluded_title_abstract": n_excl_ta,
        "included": n_included,
        "excluded_fulltext": 0,
    }


def _full_extraction(n=20):
    return [_paper(i) for i in range(n)]


def _sections_with_dois(dois):
    cite = " ".join(f"doi:{d}." for d in dois)
    grounding = " Retrieved literature evidence frames this review domain."
    return {
        "Abstract": "# Abstract\n\nScoping synthesis of prior investigations. " + cite + grounding,
        "Introduction": "# Introduction\n\nBackground per " + cite + grounding,
        "Methods": "# Methods\n\nProtocol with PRISMA flow identified / screened / eligible / included." + grounding,
        "Results": "# Results\n\nCorpus synthesis with extraction per " + cite + grounding,
        "Discussion": "# Discussion\n\nImplications per " + cite + grounding,
        "Limitations": "# Limitations\n\nCoverage limits stated with unusual candor for this review domain." + grounding,
        "Conclusion": "# Conclusion\n\nTraceable synthesis per " + cite + grounding,
    }


# --- M2 cases ---------------------------------------------------------------

def test_prisma_30_paper_mock_reconciles_exactly():
    """30-paper mock reconciles exactly (identified=screened+dup, etc.)."""
    from agents.review_agent import prisma_reconcile

    counts = _balanced_prisma()
    result = prisma_reconcile(counts)
    assert result["passed"] is True, result["findings"]
    assert result["reconciled"] is True


def test_off_by_one_prisma_blocked():
    """Off-by-one PRISMA → blocked (must balance exactly)."""
    from core.verification import verify_review

    counts = _balanced_prisma()
    counts["identified"] = counts["identified"] + 1  # off by one
    rows = _full_extraction(20)
    seeds = [DOI(i) for i in range(5)]
    corpus = {
        "prisma": counts,
        "extraction": rows,
        "seed_recall": {"recall": 1.0, "hits": 5, "total": 5, "demote_to_scoping": False,
                        "banner": None, "mode": "systematic"},
        "synthesis": {"mode": "narrative", "narrative": "Synthesis narrative.",
                      "why_not_meta": "Meta-analysis not performed: fewer than 3 commensurate effects."},
    }
    result = verify_review(corpus, _sections_with_dois(seeds))
    assert result["passed"] is False
    assert any(f.get("check") == "prisma" for f in result["findings"]), result["findings"]


def test_closed_access_marked_run_continues_abstract_only():
    """Closed-access marked full_text_unavailable, run continues abstract-only."""
    from agents.review_agent import ReviewAgent

    papers = [_paper(0), _paper(1)]

    def _resolver(doi, openalex_row=None, s2_row=None):
        if doi == DOI(0):
            return {"url": None, "source": None, "license": "closed",
                    "oa_status": "closed", "full_text_unavailable": True}
        return {"url": "https://example.org/oa.pdf", "source": "best_oa_location",
                "license": "cc-by", "oa_status": "oa", "full_text_unavailable": False}

    agent = ReviewAgent.__new__(ReviewAgent)
    marked, limitations = ReviewAgent.mark_full_text(agent, papers, oa_resolver=_resolver)
    assert marked[0]["full_text_status"] == "full_text_unavailable"
    assert marked[1]["full_text_status"] == "oa_available"
    assert len(limitations) == 1 and DOI(0) in limitations[0]
    # Run continues: both papers stay in the corpus (abstract-only, never dropped).
    assert len(marked) == 2


def test_extraction_completeness_085_blocked():
    """Extraction completeness 0.85 → blocked (bar is ≥0.9)."""
    from core.verification import verify_review

    # 20 rows x 6 cols = 120 cells; empty 18 cells → 102/120 = 0.85.
    rows = _full_extraction(20)
    emptied = 0
    for row in rows:
        for col in ("limitation", "design", "metric"):
            if emptied >= 18:
                break
            row[col] = ""
            emptied += 1
        if emptied >= 18:
            break
    assert emptied == 18
    counts = _balanced_prisma()
    seeds = [DOI(i) for i in range(5)]
    corpus = {
        "prisma": counts,
        "extraction": rows,
        "seed_recall": {"recall": 1.0, "hits": 5, "total": 5, "demote_to_scoping": False,
                        "banner": None, "mode": "systematic"},
        "synthesis": {"mode": "narrative", "narrative": "Synthesis narrative.",
                      "why_not_meta": "Meta-analysis not performed: fewer than 3 commensurate effects."},
    }
    result = verify_review(corpus, _sections_with_dois(seeds))
    assert result["passed"] is False
    assert result["extraction"]["completeness"] == pytest.approx(0.85, abs=0.01)
    assert any(f.get("check") == "extraction" for f in result["findings"])


def test_recall_3_of_5_demotes_to_scoping_not_terminal():
    """Recall 3/5 → demote systematic→scoping + banner (not terminal)."""
    from core.verification import verify_review
    from agents.review_agent import seed_recall_check

    counts = _balanced_prisma()
    rows = _full_extraction(20)
    seeds = [DOI(i) for i in range(5)]
    # Included carries only 3 of the 5 seeds.
    included_dois = [DOI(0), DOI(1), DOI(2)] + [DOI(10 + i) for i in range(17)]
    recall = seed_recall_check(included_dois, seeds)
    assert recall["recall"] == pytest.approx(0.6)
    assert recall["demote_to_scoping"] is True
    assert recall["mode"] == "scoping"
    assert recall["banner"]

    corpus = {
        "prisma": counts,
        "extraction": rows,
        "seed_recall": recall,
        "synthesis": {"mode": "narrative", "narrative": "Synthesis narrative.",
                      "why_not_meta": "Meta-analysis not performed: fewer than 3 commensurate effects."},
    }
    result = verify_review(corpus, _sections_with_dois(included_dois[:5]))
    # Demotion is a banner, never a blocking finding: gate still passes.
    assert result["passed"] is True, result["findings"]
    assert result["mode"] == "scoping"
    assert result["banner"]
    assert result["demote_to_scoping"] is True


def test_overlap_080_review_passes():
    """R11: overlap 0.80 passes for review (threshold 0.85+), fails experimental."""
    from core.verification import novelty_overlap_check

    prior = "alpha beta gamma delta epsilon zeta theta iota kappa lambda"
    draft = "alpha beta gamma delta epsilon zeta theta iota quasar pulsar"
    # Raw framing (no markdown headers or extra prose — headers would add
    # abstract/introduction tokens and shift the coefficient; the gate reads
    # the same framing text in production).
    sections = {"Abstract": draft, "Introduction": draft}
    topic = {"literature_evidence": [{"title": "Prior synthesis", "abstract": prior}]}
    review = novelty_overlap_check(sections, topic, paper_type="review")
    assert review["max_overlap"] == pytest.approx(0.8, abs=0.01)
    assert review["threshold"] >= 0.85
    assert review["passed"] is True, review["findings"]
    experimental = novelty_overlap_check(sections, topic, paper_type="experimental")
    assert experimental["passed"] is False, "0.80 must still fail the 0.65 experimental bar"


# --- Extra wiring guards ----------------------------------------------------

def test_title_jaccard_08_boundary():
    """L19/R5 (§8.8.2): title Jaccard ≥0.8 passes, distant titles fail."""
    from core.verification import title_jaccard

    assert title_jaccard("Meta Learning for Robustness", "Meta Learning for Robustness") == pytest.approx(1.0)
    # One-word difference in a 5-token title: 4/6 ≈ 0.67 → fail.
    assert title_jaccard("alpha beta gamma delta epsilon", "alpha beta gamma delta quasar") < 0.8
    # Near-identical with punctuation/case drift: still ≥0.8.
    assert title_jaccard("Wasserstein GANs with Gradient Penalty!",
                         "wasserstein gans with gradient penalty") >= 0.8


def test_oa_resolution_order_best_first_then_unpaywall(monkeypatch):
    """resolve_oa_url order: best_oa_location → Unpaywall → S2 → PMC."""
    from core import oa_sources as oa

    row = {"best_oa_location": {"pdf_url": "https://example.org/best.pdf", "license": "cc-by"}}
    out = oa.resolve_oa_url("10.1234/x", openalex_row=row)
    assert out["source"] == "best_oa_location" and out["url"] == "https://example.org/best.pdf"

    monkeypatch.setattr(oa, "_flag_enabled", lambda name, default=False: True)
    monkeypatch.setattr(oa, "_gateway_available", lambda provider: True)

    class _Client:
        def fetch_json(self, source, url, **kw):
            assert source == "unpaywall"
            return {"status": "verified",
                    "content": {"is_oa": True,
                                "best_oa_location": {"url_for_pdf": "https://example.org/u.pdf",
                                                     "license": "cc-by"}}}

    out2 = oa.resolve_oa_url("10.1234/y", openalex_row={}, client=_Client())
    assert out2["source"] == "unpaywall" and out2["url"] == "https://example.org/u.pdf"

    # Flags off → closed, abstract-only (no network).
    monkeypatch.setattr(oa, "_flag_enabled", lambda name, default=False: False)
    out3 = oa.resolve_oa_url("10.1234/z", openalex_row={}, s2_row={}, client=_Client())
    assert out3["full_text_unavailable"] is True and out3["url"] is None


def test_review_skeleton_and_graph_wiring():
    """Skeleton latex_review.tex exists with token contract; recipe wires systematic_review."""
    from pathlib import Path
    from core.phase_registry import resolve_recipe
    from core.workflow import create_typed_graph
    from main import research_nodes

    skel = Path(__file__).resolve().parents[1] / "templates" / "latex_review.tex"
    assert skel.is_file(), "templates/latex_review.tex missing"
    text = skel.read_text(encoding="utf-8")
    for token in ("%TITLE%", "%ABSTRACT%", "%SECTIONS%", "%BIBLIOGRAPHY%"):
        assert token in text, token

    nodes = research_nodes()
    assert "systematic_review" in nodes, "M2 must wire systematic_review into the node mapping"
    recipe, _, _, _ = resolve_recipe("review", {})
    assert recipe == ["intake", "topic_discovery", "systematic_review", "thesis_argument",
                      "writing_narrative", "writing_results", "supervision", "editing"]
    graph = create_typed_graph(nodes, recipe)
    assert "systematic_review" in graph.nodes
    assert "engineering" not in graph.nodes


# --- Scoping-L2 E2E (mocked, OA-only, within caps) ----------------------------

E2E_BRIDGE_1 = "Prior coupling studies map research fronts through shared references, bounding synthesis scope."
E2E_BRIDGE_2 = ("Calibration literature documents overconfident estimates under distribution shift, "
                "motivating cautious synthesis.")
E2E_CLAIM = "Integrative synthesis workflows improve cumulative insight across fragmented applied domains."
E2E_GROUND = " Retrieved literature evidence frames this review domain."


def _e2e_evidence(n=20):
    """Distinct-domain abstracts (framing-safe for the novelty screen)."""
    templates = [
        ("Citation velocity indicators for emerging topics",
         "This study examines citation velocity and keyword bursts across large "
         "bibliographic databases to flag emerging topics early for database curators."),
        ("Calibration of probabilistic forecasts",
         "We evaluate probabilistic forecasts from ensemble models, documenting "
         "miscalibration patterns and recalibration procedures for forecast consumers."),
    ]
    papers = []
    for i in range(n):
        title_tpl, abs_tpl = templates[i % 2]
        papers.append({
            "title": f"{title_tpl} ({i})",
            "abstract": abs_tpl,
            "doi": DOI(i),
            "design": "cohort",
            "sample_n": 100 + i,
            "effect": 0.2 + i * 0.01,
            "metric": "accuracy",
            "limitation": "single site",
        })
    return papers


def _e2e_prose(name, dois):
    cite = " ".join(f"doi:{d}." for d in dois[:4])
    if name == "Abstract":
        body = ("Scoping synthesis of prior investigations with traceable provenance. " + cite
                + " Our synthesis maps stages, states scope boundaries, and records limitations."
                + E2E_GROUND)
        return "# Abstract\n\n" + body * 3
    if name == "Introduction":
        body = ("This review investigates how synthesis methods organize fragmented findings. "
                f"Prior work motivates the protocol per {cite} " + E2E_GROUND)
        return "# Introduction\n\n" + body * 5
    if name == "Methods":
        body = ("Protocol with PRISMA flow: identified / screened / eligible / included with exact counts. "
                f"Dual screening applied per {cite} " + E2E_GROUND)
        return "# Methods\n\n" + body * 4
    if name == "Results":
        body = ("Corpus synthesis with extraction per " + cite + " " + E2E_BRIDGE_1 + " " + E2E_BRIDGE_2
                + " Meta-analysis not performed: fewer than 3 commensurate effects with a shared metric "
                  "were available, so heterogeneity could not be computed; synthesis is narrative."
                + E2E_GROUND)
        return "# Results\n\n" + body * 3
    if name == "Discussion":
        body = ("Implications bound the synthesis scope per " + cite + " " + E2E_BRIDGE_2 + E2E_GROUND)
        return "# Discussion\n\n" + body * 4
    if name == "Conclusion":
        body = ("Traceable synthesis organizes fragmented findings per " + cite + E2E_GROUND)
        return "# Conclusion\n\n" + body * 4
    return "# " + name + "\n\nFiller with traceable provenance per " + cite + E2E_GROUND


def test_review_e2e_scoping_l2_oa_only_no_engineering(monkeypatch, tmp_path):
    """Scoping-L2 E2E mocked: recall 3/5 demotes (banner, not terminal), engineering never runs, OA-only."""
    import re
    from core import verification
    from core.config import config
    import agents.writer as writer_mod
    import agents.editor as editor_mod
    import agents.supervisor as supervisor_mod
    import agents.thesis_agent as thesis_mod
    import agents.review_agent as review_mod
    from core import oa_sources as oa_mod
    import core.workflow_nodes as wn

    monkeypatch.setattr(verification, "resolve_doi",
                        lambda doi: {"resolved": True, "doi": doi, "title": f"Title for {doi}", "authors": ["Smith"]})
    monkeypatch.setattr(verification, "resolve_arxiv",
                        lambda aid: {"resolved": True, "arxiv_id": aid, "title": f"Title for {aid}", "authors": ["Smith"]})
    monkeypatch.setattr(verification, "call_llm_json", lambda *a, **k: {"findings": []})
    monkeypatch.setattr(thesis_mod, "call_llm_json", lambda *a, **k: {
        "claim": E2E_CLAIM,
        "bridges": [{"statement": E2E_BRIDGE_1, "source": DOI(0)},
                     {"statement": E2E_BRIDGE_2, "source": DOI(1)}],
        "steelman": "The strongest counter-reading notes evidence remains insufficient.",
        "concession": "This thesis concedes its scope limits where evidence is thin."})

    def _review_llm(prompt, *a, **k):
        text = str(prompt)
        if "title/abstract screen" in text:
            return {"keep": True}
        if "Extract systematic review columns" in text:
            return {"design": "cohort", "sample_n": 120, "effect": 0.25,
                    "metric": "accuracy", "limitation": "single site"}
        return {"keep": True}

    monkeypatch.setattr(review_mod, "call_llm_json", _review_llm)

    def _oa_resolve(doi, openalex_row=None, s2_row=None, client=None):
        try:
            idx = int(str(doi).rsplit("review", 1)[-1])
        except Exception:
            idx = 0
        if idx % 2 == 0:
            return {"url": f"https://example.org/{idx}.pdf", "source": "best_oa_location",
                    "license": "cc-by", "oa_status": "oa", "full_text_unavailable": False}
        return {"url": None, "source": None, "license": "closed",
                "oa_status": "closed", "full_text_unavailable": True}

    monkeypatch.setattr(oa_mod, "resolve_oa_url", _oa_resolve)

    def _soft_review(*a, **k):
        return {"overall_score": 9.0, "scores": {}, "weaknesses": [], "checklist": {}}

    def _soft_hall(*a, **k):
        return {"score": 9, "issues": [], "recommendations": []}

    def _review_router(prompt, *a, **k):
        if "peer reviewer" in prompt:
            return _soft_review()
        return _soft_hall()

    monkeypatch.setattr(supervisor_mod, "call_llm_json", _review_router)
    monkeypatch.setattr(writer_mod.WriterAgent, "_store_section", lambda self, *a, **k: None)
    monkeypatch.setattr(editor_mod, "generate_embedding", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(config, "output_dir", str(tmp_path / "output"))
    monkeypatch.setattr(config, "companion_repo_dir", str(tmp_path / "companion"))

    def _writer_llm(prompt, *a, **k):
        lowered = str(prompt).lower()
        m = re.search(r"write (?:a|an|the) (\w+) section", lowered)
        if m:
            return _e2e_prose(m.group(1).capitalize(), [DOI(0), DOI(1), DOI(2), DOI(3)])
        if "concise abstract" in lowered:
            return _e2e_prose("Abstract", [DOI(0), DOI(1)])
        return _e2e_prose("Results", [DOI(0), DOI(1)])

    monkeypatch.setattr(writer_mod, "call_llm", _writer_llm)
    monkeypatch.setattr(wn, "engineering_node",
                        lambda state: (_ for _ in ()).throw(AssertionError("engineering must not run")))

    from core.state import initialize_state
    from core.phase_registry import resolve_recipe

    recipe, skipped, evidence, _ = resolve_recipe("review", {})
    state = initialize_state()
    state["run_id"] = "test-review-e2e"
    state["paper_type"] = "review"
    state["pipeline_recipe"] = recipe
    state["skipped_phases"] = skipped
    state["evidence_requirements"] = evidence
    state["intake_report"] = {"goal": "effects of synthesis workflows on cumulative insight",
                              "depth": "Standard",
                              "seed_keys": [DOI(0), DOI(1), DOI(2), "10.1234/missing01", "10.1234/missing02"]}
    state["selected_topic"] = {"title": "Test review topic",
                               "description": "A scoping synthesis for offline harness verification.",
                               "literature_evidence": _e2e_evidence(20)}
    state["plan"] = None
    state["engineer_outputs"] = {}

    state = wn.systematic_review_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    corpus = state.get("corpus") or {}
    assert corpus["n_identified"] == 20 and corpus["n_included"] == 20
    assert corpus["prisma_reconciled"] is True
    assert corpus["extraction_completeness"] >= 0.9
    assert corpus["mode"] == "scoping" and corpus["banner"], "recall 3/5 must demote with banner"
    closed = [p for p in (state["selected_topic"] or {}).get("literature_evidence", [])
              if p.get("full_text_status") == "full_text_unavailable"]
    assert len(closed) == 10, "odd-DOI papers stay abstract-only (OA-only rule)"
    assert corpus["full_text_limitations"], "closed content needs limitation entries"

    state = wn.thesis_argument_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    state = wn.write_narrative_sections(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    state = wn.write_results_sections(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    state = wn.supervision_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    assert state.get("current_phase") == "editing", state.get("supervisor_scores")
    state = wn.editing_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")

    assert state.get("final_paper") and state.get("latex_output")
    assert state["final_paper"].get("paper_type") == "review"
    assert state["final_paper"].get("corpus", {}).get("mode") == "scoping"
    from pathlib import Path as _Path
    companion = state["final_paper"].get("companion_repo") or {}
    assert _Path(str(companion.get("prisma_mmd"))).is_file()
    assert _Path(str(companion.get("extraction_csv"))).is_file()
    referee = verification.final_manuscript_referee(
        state["final_paper"]["sections"], state["plan"], state.get("engineer_outputs") or {},
        topic=state["selected_topic"], paper_type="review", thesis=state.get("thesis"),
        corpus=state.get("corpus"))
    assert referee["passed"], referee["findings"]


def test_fetch_oa_text_license_gate_and_cache(tmp_path, monkeypatch):
    """fetch_oa_text: closed license never fetches; OA text caches by URL hash."""
    from core import oa_sources as oa
    from core.config import config

    monkeypatch.setattr(config, "output_dir", str(tmp_path))
    closed = oa.fetch_oa_text("https://example.org/x.pdf", "closed")
    assert closed["status"] == "full_text_unavailable"

    calls = {"n": 0}

    class _Gw:
        def request(self, provider, fn, **kw):
            calls["n"] += 1
            return "OA full text " * 100

    monkeypatch.setattr(oa_mod_gateway := __import__("core.api_gateway", fromlist=["get_gateway"]),
                        "get_gateway", lambda **k: _Gw())
    first = oa.fetch_oa_text("https://example.org/paper.html", "cc-by", max_chars=500)
    assert first["status"] == "ok" and calls["n"] == 1
    second = oa.fetch_oa_text("https://example.org/paper.html", "cc-by", max_chars=500)
    assert second["status"] == "cached" and calls["n"] == 1, "second fetch must hit the file cache"
