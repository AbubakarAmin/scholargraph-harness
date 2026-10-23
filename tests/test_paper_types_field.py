"""M3: field-scoped types — empirical-secondary / survey / case (proposal §11-M3).

Tests (6, per tracker):
- Invalid pack → named reason (not traceback).
- latin-1 + jsonl + oversize-cap paths succeed with flags.
- HF transient preserved (not demoted to not_found).
- Synthetic table → blocked with item.
- Single-source material claim → flagged + limitation, run continues.
- Valid packs → L2. Missing packs → named refusal (no fabrication).

Plus wiring guards (skeletons/recipes/nodes, writer packs + supervisor
dispatch, pack schemas).
"""

from __future__ import annotations

import zipfile

DOI = lambda i: f"10.1234/field{i:02d}"


def _survey_frame(n=40):
    """Deterministic non-synthetic response frame (high entropy, no dups)."""
    import pandas as pd

    rows = []
    for i in range(n):
        rows.append({
            "age": 20 + (i * 37) % 50,
            "item1": 1 + (i * 7) % 5,
            "item2": 1 + (i * 11) % 5,
            "item3": 1 + (i * 13) % 5,
            "target": (i * 5) % 2,
            "note": f"respondent-{i}",
        })
    return pd.DataFrame(rows)


def _write_response_csv(path, n=40, encoding="utf-8"):
    df = _survey_frame(n)
    # A non-ascii cell exercises the encoding path for latin-1 files.
    df.loc[0, "note"] = "café-respondent-0"
    df.to_csv(str(path), index=False, encoding=encoding)
    return df


def _write_jsonl(path, n=20):
    import json as _json

    rows = [{"age": 20 + i, "item1": 1 + i % 5, "target": i % 2} for i in range(n)]
    with open(str(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json.dumps(row) + "\n")
    return rows


def _write_codebook(path, target="target"):
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        f.write("item,target,scale,subgroup_preregistered\n")
        f.write(f"age,{target},numeric,\n")
        f.write(f"item1,{target},likert-5,preregistered\n")
        f.write(f"item2,{target},likert-5,\n")
    return str(path)


def _write_case_zip(path):
    with zipfile.ZipFile(str(path), "w") as archive:
        archive.writestr("interview_memo.txt", "Field memo from site visit alpha.")
        archive.writestr("counts_table.csv", "group,n\napp,120\nweb,140\n")
    return str(path)


def _single_type_zip(path):
    with zipfile.ZipFile(str(path), "w") as archive:
        archive.writestr("memo_a.txt", "First memo.")
        archive.writestr("memo_b.txt", "Second memo.")
    return str(path)


# --- M3 cases ---------------------------------------------------------------

def test_bad_pack_named_reasons_no_traceback(tmp_path):
    """Invalid pack → named reason (not traceback)."""
    from core.packs import validate_case_pack_file, validate_response_table_file
    from core.phase_registry import validate_pack

    missing = validate_response_table_file(tmp_path / "absent.csv")
    assert missing["ok"] is False
    assert any("needs_response_table" in r for r in missing["reasons"])

    bad_suffix = tmp_path / "table.txt"
    bad_suffix.write_text("not tabular", encoding="utf-8")
    suffix_result = validate_response_table_file(bad_suffix)
    assert suffix_result["ok"] is False
    assert any("needs_pack_invalid" in r for r in suffix_result["reasons"])

    single = _single_type_zip(tmp_path / "single.zip")
    single_result = validate_case_pack_file(single)
    assert single_result["ok"] is False
    assert any("needs_pack_invalid" in r for r in single_result["reasons"])

    ok, refusal = validate_pack("survey", {})
    assert not ok and refusal["reason"] == "needs_response_table"
    ok, refusal = validate_pack("case_study", {})
    assert not ok and refusal["reason"] == "needs_case_pack"
    ok, refusal = validate_pack("case_study", {"case_pack": {"entries": [{"type": "doc"}]}})
    assert not ok and refusal["reason"] == "needs_pack_invalid"


def test_encoding_jsonl_rowcap_succeed_with_flags(tmp_path, monkeypatch):
    """latin-1 + jsonl + oversize-cap paths succeed with flags (D17)."""
    from agents.data import DataAgent

    latin = tmp_path / "latin.csv"
    _write_response_csv(latin, n=10, encoding="latin-1")
    latin_result = DataAgent().validate_response_table(str(latin), "target")
    assert latin_result["ok"] is True, latin_result["reasons"]
    assert latin_result["encoding"] != "utf-8"

    jsonl = tmp_path / "resp.jsonl"
    _write_jsonl(jsonl, n=12)
    jsonl_result = DataAgent().validate_response_table(str(jsonl), "target")
    assert jsonl_result["ok"] is True, jsonl_result["reasons"]
    assert jsonl_result["rows"] == 12
    assert "lines=True" in str(jsonl_result["encoding"])

    import core.packs as packs

    monkeypatch.setattr(packs, "RESPONSE_ROW_CAP", 50)
    big = tmp_path / "big.csv"
    _write_response_csv(big, n=60)
    # DataAgent reads the patched cap at call time (import inside helper).
    big_result = DataAgent().validate_response_table(str(big), "target")
    assert big_result["ok"] is True, big_result["reasons"]
    assert big_result["rows"] == 50
    assert big_result["truncated"] is True
    assert big_result["skipped_files"], "row-cap skip must be surfaced"


def test_hf_transient_preserved_not_demoted(monkeypatch):
    """HF transient preserved (not demoted to not_found, D19)."""
    from core import datasets as ds

    assert ds._hf_error_is_transient("500 internal error") is True
    assert ds._hf_error_is_transient("overloaded, try again") is True
    assert ds._hf_error_is_transient("not_found") is False
    assert ds._hf_error_is_transient(None) is False

    monkeypatch.setattr(ds, "get_hf_dataset_info",
                        lambda _did: ds.HFResult(found=False, error="503 service unavailable"))
    monkeypatch.setattr(ds, "search_hf_datasets", lambda *a, **k: [])
    result = ds.dataset_exists_on_hf("some-org/some-dataset")
    assert result.found is False
    assert result.error != "not_found", "transient must not demote to not_found"
    assert "503" in str(result.error)


def test_synthetic_table_blocked_with_item():
    """Synthetic table → blocked with item (§4.8 synthetic screen)."""
    import pandas as pd
    from agents.survey_agent import SurveyAgent, detect_synthetic
    from core.verification import verify_survey

    frame = pd.DataFrame([{"age": 30, "item1": 3, "target": 1}] * 20)
    screen = detect_synthetic(frame)
    assert screen["is_synthetic"] is True
    assert screen["blocking"] is True

    analysis = SurveyAgent().run_analysis(frame, "target")
    assert analysis["passed"] is False
    assert analysis["synthetic"]["blocking"] is True
    assert any(f.get("check") == "synthetic_screen" and f.get("blocking")
               for f in analysis["findings"])

    gate = verify_survey(analysis, {})
    assert gate["passed"] is False
    assert any(f.get("check") == "synthetic_screen" for f in gate["findings"])


def test_single_source_flag_limitation_run_continues():
    """Single-source material claim → flagged + limitation, run continues (§4.9)."""
    from agents.case_agent import CaseAgent
    from core.verification import verify_case

    manifest = {"entries": [{"path": "memo.txt", "type": "doc"},
                            {"path": "table.csv", "type": "table"}]}
    claim = "Adoption was driven by on-site champions."
    protocol = CaseAgent().run_protocol(
        manifest,
        claims=[claim],
        evidence_map={claim: [{"type": "doc", "ref": "memo.txt"}]},
        rivals_addressed=["Price pressure explains the same pattern."],
    )
    assert protocol["triangulation"]["n_flagged"] == 1
    assert protocol["triangulation"]["limitations"], "flag needs a limitation entry"
    # Flags warn — pack + rival hold, so the run continues.
    assert protocol["passed"] is True

    gate = verify_case(protocol, {})
    assert gate["passed"] is True, gate["findings"]
    assert gate["n_flagged"] == 1
    assert any(f.get("check") == "single_source_flag" and not f.get("blocking")
               for f in gate["findings"])


# --- L2 / refusal E2E (mocked) ------------------------------------------------

E2E_GROUND = " Retrieved literature evidence frames this field domain."
E2E_CLAIM_S = "Preregistered survey cuts generalize within the sampled population."
E2E_CLAIM_C = "On-site champions explain adoption across the two observed sites."


def _e2e_evidence(n=6):
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
        papers.append({"title": f"{title_tpl} ({i})", "abstract": abs_tpl, "doi": DOI(i)})
    return papers


def _e2e_prose(paper_type, name, dois):
    cite = " ".join(f"doi:{d}." for d in dois[:3])
    quant = (" With n=120 respondents the mean profile score is 3.4 "
             "(std 0.8, 95% confidence interval 3.2 to 3.6).")
    if name == "Abstract":
        body = ("Field synthesis with traceable pack provenance. " + cite + quant + E2E_GROUND)
        return "# Abstract\n\n" + body * 3
    if name == "Introduction":
        body = ("This field study investigates bounded applied practices. "
                f"Prior work motivates the protocol per {cite} " + E2E_GROUND)
        return "# Introduction\n\n" + body * 5
    if name == "Methods":
        body = ("Instrument protocol with pinned pack hashes and preregistered cuts. "
                f"Response table validated per {cite} " + quant + E2E_GROUND)
        return "# Methods\n\n" + body * 4
    if name == "Case":
        body = ("The observed site organizes applied practice around two teams. "
                f"Context per {cite} " + E2E_GROUND)
        return "# Case\n\n" + body * 5
    if name == "Results":
        body = ("Field results with descriptives per " + cite + " " + quant + E2E_GROUND)
        return "# Results\n\n" + body * 3
    if name == "Findings":
        body = ("Triangulated findings across independent pack sources per " + cite
                + " " + E2E_CLAIM_C + E2E_GROUND)
        return "# Findings\n\n" + body * 3
    if name == "Discussion":
        body = ("Implications bound the field scope per " + cite + " " + quant + E2E_GROUND)
        return "# Discussion\n\n" + body * 4
    if name == "Conclusion":
        body = ("Traceable field synthesis organizes bounded findings per " + cite + E2E_GROUND)
        return "# Conclusion\n\n" + body * 4
    return "# " + name + "\n\nFiller with traceable provenance per " + cite + E2E_GROUND


def _mock_env(monkeypatch, tmp_path):
    import re
    from core import verification
    from core.config import config
    import agents.writer as writer_mod
    import agents.editor as editor_mod
    import agents.supervisor as supervisor_mod
    import core.workflow_nodes as wn

    monkeypatch.setattr(verification, "resolve_doi",
                        lambda doi: {"resolved": True, "doi": doi, "title": f"Title for {doi}", "authors": ["Smith"]})
    monkeypatch.setattr(verification, "resolve_arxiv",
                        lambda aid: {"resolved": True, "arxiv_id": aid, "title": f"Title for {aid}", "authors": ["Smith"]})
    monkeypatch.setattr(verification, "call_llm_json", lambda *a, **k: {"findings": []})

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
    monkeypatch.setattr(wn, "engineering_node",
                        lambda state: (_ for _ in ()).throw(AssertionError("engineering must not run")))
    return wn, writer_mod


def _writer_llm(paper_type):
    import re

    def _llm(prompt, *a, **k):
        lowered = str(prompt).lower()
        match = re.search(r"write (?:a|an|the) (\w+) section", lowered)
        if match:
            return _e2e_prose(paper_type, match.group(1).capitalize(), [DOI(0), DOI(1), DOI(2)])
        if "concise abstract" in lowered:
            return _e2e_prose(paper_type, "Abstract", [DOI(0), DOI(1)])
        return _e2e_prose(paper_type, "Results", [DOI(0), DOI(1)])
    return _llm


def _make_state(paper_type, pack):
    from core.state import initialize_state
    from core.phase_registry import resolve_recipe

    recipe, skipped, evidence, _ = resolve_recipe(paper_type, pack)
    state = initialize_state()
    state["run_id"] = f"test-{paper_type}"
    state["paper_type"] = paper_type
    state["pipeline_recipe"] = recipe
    state["skipped_phases"] = skipped
    state["evidence_requirements"] = evidence
    state["intake_report"] = {"goal": "field practices and bounded applied outcomes",
                              "depth": "Standard",
                              "pack_manifest": dict(pack)}
    state["selected_topic"] = {"title": f"Test {paper_type} topic",
                               "description": "A bounded field investigation for offline harness verification.",
                               "literature_evidence": _e2e_evidence()}
    state["plan"] = None
    state["engineer_outputs"] = {}
    return state


def _assert_l2(state, paper_type):
    from core.verification import final_manuscript_referee

    assert state.get("final_paper"), "no final paper assembled"
    assert state.get("latex_output"), "no latex output"
    sections = state["final_paper"]["sections"]
    assert sections.get("Limitations") and len(sections["Limitations"]) >= 80
    scores = state["supervisor_scores"]
    assert scores and min(scores.values()) >= 8.5, scores
    kwargs: dict = {"topic": state["selected_topic"], "paper_type": paper_type}
    if paper_type == "survey":
        kwargs["survey_analysis"] = (state.get("analysis_reports") or {}).get("survey")
        kwargs["instrument"] = state.get("instrument")
    if paper_type == "case_study":
        kwargs["case_protocol"] = (state.get("analysis_reports") or {}).get("case_protocol")
    referee = final_manuscript_referee(
        sections, state.get("plan"), state.get("engineer_outputs") or {}, **kwargs)
    assert referee["passed"], referee["findings"]
    assert state["final_paper"].get("paper_type") == paper_type


def test_valid_packs_l2_missing_refusal(monkeypatch, tmp_path):
    """Valid packs → L2. Missing packs → named refusal (no fabrication)."""
    import agents.writer as writer_mod
    import core.workflow_nodes as wn

    wn_mod, _ = _mock_env(monkeypatch, tmp_path)
    monkeypatch.setattr(writer_mod, "call_llm", _writer_llm("survey"))

    # -- survey L2 with a valid pack --
    response_csv = tmp_path / "resp.csv"
    _write_response_csv(response_csv, n=40)
    codebook_csv = tmp_path / "codebook.csv"
    _write_codebook(codebook_csv)
    survey_pack = {"response_table": {"path": str(response_csv)},
                   "codebook": {"path": str(codebook_csv)}}
    state = _make_state("survey", survey_pack)
    state["response_table"] = {"path": str(response_csv)}
    state["codebook"] = {"path": str(codebook_csv)}
    state = wn_mod.data_validation_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    state = wn_mod.survey_analysis_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    assert state.get("instrument", {}).get("response_hash"), "instrument must hash the table"
    state = wn_mod.write_results_sections(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    state = wn_mod.supervision_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    assert state.get("current_phase") == "editing", state.get("supervisor_scores")
    state = wn_mod.editing_node(state)
    assert not state.get("terminal_error"), state.get("terminal_error")
    _assert_l2(state, "survey")
    from pathlib import Path as _Path
    companion = state["final_paper"].get("companion_repo") or {}
    assert _Path(str(companion.get("codebook_csv"))).is_file()
    assert _Path(str(companion.get("response_hash"))).is_file()

    # -- case L2 with a valid pack --
    monkeypatch.setattr(writer_mod, "call_llm", _writer_llm("case_study"))
    case_zip = tmp_path / "case.zip"
    _write_case_zip(case_zip)
    claim = E2E_CLAIM_C
    case_pack = {"case_pack": {"path": str(case_zip)}}
    cstate = _make_state("case_study", case_pack)
    cstate["case_pack"] = {"path": str(case_zip)}
    cstate["case_claims"] = [claim]
    cstate["case_evidence_map"] = {claim: [{"type": "doc", "ref": "interview_memo.txt"},
                                           {"type": "table", "ref": "counts_table.csv"}]}
    cstate["case_rival_gap"] = "No rival observed in the two sites; gap recorded explicitly."
    cstate = wn_mod.case_protocol_node(cstate)
    assert not cstate.get("terminal_error"), cstate.get("terminal_error")
    cstate = wn_mod.write_narrative_sections(cstate)
    assert not cstate.get("terminal_error"), cstate.get("terminal_error")
    cstate = wn_mod.write_results_sections(cstate)
    assert not cstate.get("terminal_error"), cstate.get("terminal_error")
    cstate = wn_mod.supervision_node(cstate)
    assert not cstate.get("terminal_error"), cstate.get("terminal_error")
    assert cstate.get("current_phase") == "editing", cstate.get("supervisor_scores")
    cstate = wn_mod.editing_node(cstate)
    assert not cstate.get("terminal_error"), cstate.get("terminal_error")
    _assert_l2(cstate, "case_study")
    ccompanion = cstate["final_paper"].get("companion_repo") or {}
    assert _Path(str(ccompanion.get("triangulation_csv"))).is_file()

    # -- missing packs → named refusal, never fabricated rows --
    from core.workflow_nodes import intake_node
    from core.state import initialize_state as _init

    for paper_type, reason in (("survey", "needs_response_table"), ("case_study", "needs_case_pack")):
        rstate = _init()
        rstate["paper_type"] = paper_type
        rstate["intake_report"] = {"goal": "field practices and bounded applied outcomes",
                                   "requested": paper_type, "pack_manifest": {},
                                   "depth": "Standard", "budgets": {}}
        rstate = intake_node(rstate)
        assert rstate.get("terminal_error") == reason, rstate.get("terminal_error")
        assert not rstate.get("final_paper"), "refused runs must not assemble a paper"


# --- Wiring guards ----------------------------------------------------------

def test_field_skeletons_recipes_and_nodes():
    """Skeletons exist (4-token contract); recipes + node map wire survey/case."""
    from pathlib import Path
    from core.phase_registry import resolve_recipe
    from core.workflow import create_typed_graph
    from main import research_nodes

    for skeleton in ("latex_survey.tex", "latex_case_study.tex", "latex_empirical.tex"):
        path = Path(__file__).resolve().parents[1] / "templates" / skeleton
        assert path.is_file(), f"templates/{skeleton} missing"
        text = path.read_text(encoding="utf-8")
        for token in ("%TITLE%", "%ABSTRACT%", "%SECTIONS%", "%BIBLIOGRAPHY%"):
            assert token in text, (skeleton, token)

    nodes = research_nodes()
    assert "survey_analysis" in nodes and "case_protocol" in nodes

    survey_recipe, _, _, _ = resolve_recipe("survey", {"response_table": {"manifest": {}}})
    assert survey_recipe == ["intake", "topic_discovery", "data_validation", "survey_analysis",
                             "writing_results", "supervision", "editing"]
    case_recipe, _, _, _ = resolve_recipe("case_study", {"case_pack": {"entries": [{}, {}]}})
    assert case_recipe == ["intake", "topic_discovery", "case_protocol", "writing_narrative",
                           "writing_results", "supervision", "editing"]
    graph = create_typed_graph(nodes, survey_recipe)
    assert "survey_analysis" in graph.nodes
    assert "engineering" not in graph.nodes

    from core.capabilities import DEFAULT_MANIFESTS
    for agent in ("SurveyAnalysisAgent", "SurveyAgent", "CaseProtocolAgent", "CaseAgent"):
        assert agent in DEFAULT_MANIFESTS, f"{agent} manifest missing"


def test_writer_packs_and_supervisor_field_dispatch():
    """Writer field packs exist; supervisor survey/case gates behave."""
    from agents.writer import WRITER_PACKS
    from agents.supervisor import SupervisorAgent

    for paper_type in ("survey", "case_study", "empirical"):
        assert paper_type in WRITER_PACKS, f"writer pack missing: {paper_type}"
        assert WRITER_PACKS[paper_type]["headers"] and WRITER_PACKS[paper_type]["policy"]

    agent = SupervisorAgent.__new__(SupervisorAgent)
    # Fake-field phrases block survey/case sections (Methods allowlisted).
    bad = "Our synthetic respondents confirm the pattern."
    passed, cap, _ = SupervisorAgent._per_type_section_checks(
        agent, "Results", bad, "survey", None)
    assert passed is False and cap == 4.0
    passed, _, _ = SupervisorAgent._per_type_section_checks(
        agent, "Methods", bad, "survey", None)
    assert passed is True, "Methods carries the R9 allowlist"
    good = ("Profile estimates with n=120 carry a 95% confidence interval. "
            "Retrieved literature evidence frames this field domain. " * 10)
    passed, _, _ = SupervisorAgent._per_type_section_checks(
        agent, "Results", good, "survey", None)
    assert passed is True


def test_pack_schemas_and_prohibited_scope():
    """Pack schema caps hold; prohibited scope is Results/Abstract (Methods exempt)."""
    from core import packs
    from core.verification import _scoped_prohibited_hits, PROHIBITED_MANUSCRIPT_TEXT

    assert packs.RESPONSE_TABLE_MAX_BYTES == 50 * 1024 * 1024
    assert packs.CASE_PACK_MAX_BYTES == 200 * 1024 * 1024
    assert packs.RESPONSE_ROW_CAP == 200_000
    assert packs.CODEBOOK_COLUMNS == ("item", "target", "scale", "subgroup_preregistered")
    assert any("respondent" in phrase for phrase in PROHIBITED_MANUSCRIPT_TEXT)

    sections = {"Results": "Our synthetic respondents confirm the pattern.",
                "Methods": "Our synthetic respondents confirm the pattern."}
    hits = _scoped_prohibited_hits(sections)
    assert "synthetic respondents" in hits
    methods_only = {"Methods": "Our synthetic respondents confirm the pattern."}
    assert _scoped_prohibited_hits(methods_only) == []
