"""
EditorAgent — submission-grade LaTeX, DOI bibliography, Limitations, companion repo.
Provides create_final_paper / generate_latex expected by main.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import config
from core.utils import log_agent_action
from core.llm import call_llm, generate_embedding
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.contracts import ExperimentOutput, Paper, Plan, Topic
from core.memory import memory
from core.verification import extract_citation_ids, resolve_doi, resolve_arxiv, final_manuscript_referee
from core.verification import NON_EMPIRICAL_TYPES, FIELD_WAIVER_TYPES

logger = logging.getLogger(__name__)


def _is_non_experimental_plan(plan: Optional[Dict[str, Any]], paper_type: Optional[str] = None) -> bool:
    """True when the plan/recipe carries no experiment artifacts (R1 §6.1).

    Non-experimental means ``experiments: []`` plus a non-empirical signal:
    a waiver ``paper_type`` (explicit kwarg wins, else the plan's own) or a
    ``non_empirical`` plan_mode. Experimental / comparative / cause_effect
    plans — and typeless legacy plans — are never non-experimental here.
    """
    experiments = (plan or {}).get("experiments") or []
    if experiments:
        return False
    effective_type = paper_type or (plan or {}).get("paper_type")
    return (
        effective_type in (NON_EMPIRICAL_TYPES | FIELD_WAIVER_TYPES)
        or (plan or {}).get("plan_mode") == "non_empirical"
    )


# Manuscript section orders per paper type (proposal §3.4, exact).
MANUSCRIPT_ORDERS: Dict[str, List[str]] = {
    "experimental": ["Introduction", "Related Work", "Methods", "Experiments", "Results", "Discussion", "Limitations", "Conclusion"],
    "comparative": ["Introduction", "Related Work", "Methods", "Experiments", "Results", "Discussion", "Limitations", "Conclusion"],
    "empirical": ["Introduction", "Related Work", "Methods", "Experiments", "Results", "Discussion", "Limitations", "Conclusion"],
    "cause_effect": ["Introduction", "Related Work", "Methods", "Experiments", "Results", "Discussion", "Limitations", "Conclusion"],
    "review": ["Introduction", "Methods", "Results", "Discussion", "Limitations", "Conclusion"],
    "analytical": ["Introduction", "Background", "Analysis", "Discussion", "Limitations", "Conclusion"],
    "descriptive": ["Introduction", "Background", "Findings", "Discussion", "Limitations", "Conclusion"],
    "argumentative": ["Introduction", "Thesis", "Evidence", "Counterargument", "Discussion", "Limitations", "Conclusion"],
    "case_study": ["Introduction", "Case", "Methods", "Findings", "Discussion", "Limitations", "Conclusion"],
    "survey": ["Introduction", "Related Work", "Methods", "Results", "Discussion", "Limitations", "Conclusion"],
}

# Skeleton contract: templates/latex_{type}.tex may define %TITLE%,
# %ABSTRACT%, %SECTIONS%, %BIBLIOGRAPHY% tokens filled at assembly time.
SKELETON_TOKENS = ("%TITLE%", "%ABSTRACT%", "%SECTIONS%", "%BIBLIOGRAPHY%")


class EditorAgent:
    @staticmethod
    def build_repair_hints(referee: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Machine repair hints from release-referee findings.

        Each hint is {section, check, expected, detail} — routed by
        editing_node into supervisor_feedback so the repair redraft fixes
        precise defects instead of re-rolling blind.
        """
        hints: List[Dict[str, Any]] = []
        for finding in (referee or {}).get("findings") or []:
            if not isinstance(finding, dict):
                continue
            hints.append({
                "section": str(finding.get("section") or finding.get("where") or ""),
                "check": str(finding.get("check") or "release_referee"),
                "expected": str(finding.get("expected") or finding.get("fix") or
                                "revise the flagged content per the finding"),
                "detail": str(finding.get("details") or finding.get("detail")
                               or finding.get("message") or "")[:400],
            })
        return hints

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        runtime_config = self.context.config if self.context else config
        self.vector_memory = self.context.memory if self.context else memory
        self.output_dir = runtime_config.output_dir
        self.companion_dir = runtime_config.companion_repo_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.companion_dir, exist_ok=True)

    # --- API expected by main.py ---

    def create_final_paper(
        self,
        topic: Topic,
        sections: Dict[str, str],
        plan: Plan,
        engineer_outputs: Optional[Dict[str, ExperimentOutput]] = None,
        debate_results: Optional[List[Any]] = None,
        paper_type: str = "experimental",
        thesis: Optional[Dict[str, Any]] = None,
        corpus: Optional[Dict[str, Any]] = None,
        instrument: Optional[Dict[str, Any]] = None,
        survey_analysis: Optional[Dict[str, Any]] = None,
        case_protocol: Optional[Dict[str, Any]] = None,
        causal_claim: Optional[Dict[str, Any]] = None,
        indep_pass: bool = False,
        run_stats: Optional[Dict[str, Any]] = None,
        provenance: str = "clean",
    ) -> Paper:
        # R1 (§6.1) per-type skip: a non-experimental plan (experiments: []
        # + non-empirical type/mode) has no experiment artifacts, so only
        # outputs tied to a planned experiment can fail assembly — with no
        # planned experiments the empty handoff (engineer_outputs == {})
        # assembles instead of raising. The experimental path keeps the exact
        # legacy semantics (any failed output stays terminal).
        non_experimental = _is_non_experimental_plan(plan, paper_type)
        planned_names = {
            experiment.get("name")
            for experiment in ((plan or {}).get("experiments") or [])
            if isinstance(experiment, dict) and experiment.get("name")
        }
        if non_experimental:
            failed = [
                name for name, output in (engineer_outputs or {}).items()
                if isinstance(output, dict) and not output.get("success") and name in planned_names
            ]
        else:
            failed = [name for name, output in (engineer_outputs or {}).items() if isinstance(output, dict) and not output.get("success")]
        if failed:
            raise RuntimeError("Cannot assemble manuscript from failed experiments: " + ", ".join(failed))
        sections = dict(sections)
        # Honest Limitations from unresolved Challenger objections + field/causal/review
        if "Limitations" not in sections or len(sections.get("Limitations", "")) < 80:
            sections["Limitations"] = self._limitations_from_debate(
                debate_results, plan,
                survey_analysis=survey_analysis, case_protocol=case_protocol,
                causal_claim=causal_claim, corpus=corpus,
            )

        referee = final_manuscript_referee(sections, plan, engineer_outputs, topic=topic, paper_type=paper_type, thesis=thesis, corpus=corpus, instrument=instrument, survey_analysis=survey_analysis, case_protocol=case_protocol, causal_claim=causal_claim, indep_pass=indep_pass)
        if not referee["passed"]:
            err = RuntimeError("Cannot assemble manuscript: release referee failed: " + json.dumps(referee["findings"], default=str))
            # Machine hints ride on the exception (message shape preserved
            # for editor_repair_route string matching).
            try:
                err.patch_hints = self.build_repair_hints(referee)  # type: ignore[attr-defined]
            except Exception:
                pass
            raise err

        bib_entries, bib_map = self._bibliography_from_dois(sections)
        companion = self._write_companion_repo(topic, plan, engineer_outputs or {}, paper_type=paper_type, corpus=corpus, instrument=instrument, survey_analysis=survey_analysis, case_protocol=case_protocol, causal_claim=causal_claim, run_stats=run_stats)

        return {
            "topic": topic,
            "sections": sections,
            "plan": plan,
            "engineer_outputs": engineer_outputs or {},
            "bibliography": bib_entries,
            "bib_map": bib_map,
            "companion_repo": companion,
            "debate_results": debate_results or [],
            "paper_type": paper_type,
            "thesis": thesis,
            "corpus": corpus,
            "instrument": instrument,
            "survey_analysis": survey_analysis,
            "case_protocol": case_protocol,
            "causal_claim": causal_claim,
            "publishable": False,
            "approval_required": True,
            "provenance": provenance,
            "timestamp": datetime.now().isoformat(),
        }

    def generate_latex(self, final_paper: Paper) -> str:
        topic = final_paper["topic"]
        sections = final_paper["sections"]
        plan = final_paper.get("plan") or {}
        bib = final_paper.get("bibliography") or ""
        paper_type = final_paper.get("paper_type") or "experimental"
        latex = self._generate_latex_document(topic, sections, plan, paper_type=paper_type)
        # Attach resolved bib as comment + file write
        bib_path = os.path.join(self.output_dir, "references.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(bib or "% no resolved citations\n")
        latex_path = os.path.join(
            self.output_dir, f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        )
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        try:
            self.vector_memory.add_embedding(
                generate_embedding(topic.get("title", "")),
                {
                    "type": "final_paper",
                    "namespace": "editor",
                    "content_class": "generated_narrative",
                    "retrieval_eligible": False,
                    "agent": "EditorAgent",
                    "outcome_status": "released",
                    "topic": topic.get("title"),
                    "latex": latex_path,
                },
            )
        except Exception:
            pass
        log_agent_action("EditorAgent", "latex_generated", {"path": latex_path})
        return latex

    def assemble_paper(
        self,
        topic: Topic,
        sections: Dict[str, str],
        plan: Plan,
        engineer_outputs: Optional[Dict[str, ExperimentOutput]] = None,
        debate_results: Optional[List[Any]] = None,
        paper_type: str = "experimental",
        thesis: Optional[Dict[str, Any]] = None,
        corpus: Optional[Dict[str, Any]] = None,
    ) -> Paper:
        final = self.create_final_paper(
            topic, sections, plan, engineer_outputs, debate_results,
            paper_type=paper_type, thesis=thesis, corpus=corpus,
        )
        latex = self.generate_latex(final)
        return {
            "success": True,
            "final_paper": final,
            "latex": latex,
            "latex_file": os.path.join(self.output_dir, "paper_output.tex"),
            "bib_file": os.path.join(self.output_dir, "references.bib"),
            "companion_repo": final.get("companion_repo"),
            "timestamp": str(datetime.now()),
        }

    def _limitations_from_debate(
        self,
        debate_results: Optional[List[Any]],
        plan: Plan,
        survey_analysis: Optional[Dict[str, Any]] = None,
        case_protocol: Optional[Dict[str, Any]] = None,
        causal_claim: Optional[Dict[str, Any]] = None,
        corpus: Optional[Dict[str, Any]] = None,
    ) -> str:
        objections = []
        for d in debate_results or []:
            unresolved = getattr(d, "unresolved_objections", None)
            if unresolved is None and isinstance(d, dict):
                unresolved = d.get("unresolved_objections") or []
            objections.extend(unresolved or [])
        # Lab upgrade: carry field + causal + review limitations, not just debate.
        for bundle, key in (
            (survey_analysis, "survey"), (case_protocol, "case"), (causal_claim, "causal"),
        ):
            if isinstance(bundle, dict):
                for lim in (bundle.get("limitations") or []):
                    if lim and lim not in objections:
                        objections.append(f"[{key}] {lim}")
                for f in (bundle.get("findings") or []):
                    if isinstance(f, dict) and not f.get("blocking") and f.get("details"):
                        item = f"[{key}] {f['details']}"[:300]
                        if item not in objections:
                            objections.append(item)
        if isinstance(corpus, dict):
            for lim in (corpus.get("full_text_limitations") or []):
                if lim and lim not in objections:
                    objections.append(f"[review] {lim}")
        if not objections:
            # Fall back to plan flags
            objections = (plan or {}).get("unfalsifiable_flags") or [
                "Results are based on synthetic or limited public data.",
                "Compute budget constrained sandbox experiments; scale-up unverified.",
            ]
        bullets = "\n".join(f"- {o}" for o in objections[:16])
        return (
            "# Limitations\n\n"
            "The following limitations are carried forward from adversarial debate, "
            "field protocols, causal audit, and review coverage "
            "(unresolved objections and unverified assumptions):\n\n"
            f"{bullets}\n"
        )

    def _bibliography_from_dois(self, sections: Dict[str, str]) -> tuple:
        text = "\n".join(sections.values())
        ids = extract_citation_ids(text)
        entries = []
        bib_map = {}
        idx = 1
        for doi in ids["dois"]:
            info = resolve_doi(doi)
            if not info.get("resolved"):
                continue
            key = f"doi{idx:03d}"
            title = (info.get("title") or "Untitled").replace("{", "").replace("}", "")
            year = info.get("year") or "n.d."
            journal = (info.get("container") or "Journal").replace("{", "").replace("}", "")
            entries.append(
                f"@article{{{key},\n"
                f"  title={{{title}}},\n"
                f"  year={{{year}}},\n"
                f"  journal={{{journal}}},\n"
                f"  doi={{{doi}}}\n}}"
            )
            bib_map[doi] = key
            idx += 1
        for aid in ids["arxiv_ids"]:
            info = resolve_arxiv(aid)
            if not info.get("resolved"):
                continue
            key = f"arxiv{idx:03d}"
            title = (info.get("title") or aid).replace("{", "").replace("}", "")
            entries.append(
                f"@article{{{key},\n"
                f"  title={{{title}}},\n"
                f"  year={{20{aid[:2]}}},\n"
                f"  journal={{arXiv}},\n"
                f"  eprint={{{aid}}}\n}}"
            )
            bib_map[aid] = key
            idx += 1
        return "\n\n".join(entries), bib_map

    def _write_companion_repo(
        self,
        topic: Topic,
        plan: Plan,
        engineer_outputs: Dict[str, ExperimentOutput],
        paper_type: str = "experimental",
        corpus: Optional[Dict[str, Any]] = None,
        instrument: Optional[Dict[str, Any]] = None,
        survey_analysis: Optional[Dict[str, Any]] = None,
        case_protocol: Optional[Dict[str, Any]] = None,
        causal_claim: Optional[Dict[str, Any]] = None,
        run_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        root = Path(self.companion_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "experiments").mkdir(exist_ok=True)

        from importlib.metadata import PackageNotFoundError, version

        requirements = []
        for package in ("numpy", "pandas", "scikit-learn", "matplotlib", "scipy"):
            try:
                requirements.append(f"{package}=={version(package)}")
            except PackageNotFoundError:
                requirements.append(package)
        req = "\n".join(requirements) + "\n"
        (root / "requirements.txt").write_text(req, encoding="utf-8")
        from core.config import config as _runtime_config

        manifest = {
            "topic": topic.get("title"),
            "contract_hashes": {
                name: output.get("contract_hash")
                for name, output in engineer_outputs.items()
                if isinstance(output, dict) and output.get("contract_hash")
            },
            "seeds": {
                name: (output.get("multi_seed") or {}).get("seeds", [])
                for name, output in engineer_outputs.items()
                if isinstance(output, dict)
            },
            "models": {  # §7.4/§10.6: models that produced this bundle
                "llm_provider": _runtime_config.llm_provider,
                "chat_default": _runtime_config.resolve_model("default"),
                "chat_cheap": _runtime_config.resolve_model("cheap") or _runtime_config.resolve_model("default"),
                "chat_strong": _runtime_config.resolve_model("strong") or _runtime_config.resolve_model("default"),
                "chat_judge": _runtime_config.resolve_model("judge") or _runtime_config.resolve_model("default"),
                "embedding_model": _runtime_config.gemini_embedding_model,
                "embedding_dimension": _runtime_config.embedding_dimension,
            },
            "generated_at": datetime.now().isoformat(),
            "git_commit": self._git_commit(),
            "python_version": __import__("sys").version,
        }
        (root / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # §10.6 decision bundle: per-experiment engineering decision logs.
        decision_log = [
            {"experiment": name, "attempts": (output.get("decision_log") or output.get("attempts") or [])}
            for name, output in engineer_outputs.items()
            if isinstance(output, dict)
        ]
        (root / "decision_log.json").write_text(json.dumps(decision_log, indent=2, default=str), encoding="utf-8")
        # §10.6 cost bundle: real tracker stats when supplied, else an honest
        # labelled partial (never fabricated numbers).
        stats = dict(run_stats or {})
        cost_report = {
            "llm_calls": stats.get("llm_calls", 0),
            "llm_tokens_in": stats.get("llm_tokens_in", 0),
            "llm_tokens_out": stats.get("llm_tokens_out", 0),
            "cost_estimate_usd": stats.get("cost_estimate_usd", 0.0),
            "complete": bool(run_stats),
        }
        (root / "cost_report.json").write_text(json.dumps(cost_report, indent=2, default=str), encoding="utf-8")

        readme = f"""# Companion code — {topic.get('title', 'Research')}

Auto-generated by ScholarGraph Editor.

## Setup
```bash
pip install -r requirements.txt
python run_experiments.py
```

## Experiments
See `experiments/` for code snapshots from the Engineer agent.
Raw multi-seed results live alongside the paper outputs.
"""
        (root / "README.md").write_text(readme, encoding="utf-8")

        runner_lines = [
            '"""Entry point for companion experiments (real runner, not a listing stub)."""',
            "import json",
            "import subprocess",
            "import sys",
            "from pathlib import Path",
            "",
            "def main():",
            "    root = Path(__file__).resolve().parent",
            "    manifest_path = root / 'reproducibility_manifest.json'",
            "    try:",
            "        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))",
            "    except Exception as exc:",
            "        print(json.dumps({'passed': False, 'error': f'missing manifest: {exc}'}))",
            "        raise SystemExit(1)",
            "    seeds = manifest.get('seeds') or {}",
            "    results = {}",
            "    failed = []",
            "    for p in sorted((root / 'experiments').glob('*.py')):",
            "        exp_seeds = seeds.get(p.stem) or [0]",
            "        ok_all = True",
            "        for seed in exp_seeds:",
            "            try:",
            "                completed = subprocess.run(",
            "                    [sys.executable, str(p)], cwd=root, capture_output=True,",
            "                    text=True, timeout=120)",
            "                ok = completed.returncode == 0",
            "            except Exception as exc:",
            "                ok = False",
            "                completed = None",
            "            ok_all = ok_all and ok",
            "            results[f'{p.stem}/seed_{seed}'] = ok",
            "        if not ok_all:",
            "            failed.append(p.stem)",
            "    print(json.dumps({'passed': not failed, 'failed': failed, 'results': results}))",
            "    raise SystemExit(0 if not failed else 1)",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
        (root / "run_experiments.py").write_text("\n".join(runner_lines) + "\n", encoding="utf-8")

        paths = {
            "readme": str(root / "README.md"),
            "requirements": str(root / "requirements.txt"),
            "reproducibility_manifest": str(root / "reproducibility_manifest.json"),
            "decision_log": str(root / "decision_log.json"),
            "cost_report": str(root / "cost_report.json"),
        }
        for name, out in engineer_outputs.items():
            code = out.get("code") if isinstance(out, dict) else None
            if code:
                safe = re.sub(r"[^\w\-]+", "_", name)[:60]
                path = root / "experiments" / f"{safe}.py"
                path.write_text(code, encoding="utf-8")
                paths[safe] = str(path)
        # Post-write manifest seal (replay-grade): code sha + expected metrics
        # so replay_companion can fail on hash/metric drift, not just crashes.
        try:
            import hashlib as _hl

            _manifest_path = root / "reproducibility_manifest.json"
            _manifest = json.loads(_manifest_path.read_text(encoding="utf-8"))
            _code_sha = {
                name: _hl.sha256(
                    (root / "experiments" / f"{re.sub(r'[^\\w\\-]+', '_', name)[:60]}.py")
                    .read_bytes()).hexdigest()[:16]
                for name, out in (engineer_outputs or {}).items()
                if isinstance(out, dict) and out.get("code")
                and (root / "experiments" / f"{re.sub(r'[^\\w\\-]+', '_', name)[:60]}.py").is_file()
            }
            _expected_metrics = {
                name: (out.get("aggregate_metrics")
                       or (out.get("parsed") or {}).get("metrics") or {})
                for name, out in (engineer_outputs or {}).items()
                if isinstance(out, dict)
            }
            _manifest["code_sha256"] = _code_sha
            _manifest["expected_metrics"] = _expected_metrics
            _manifest["metric_tolerance"] = 0.05
            _manifest_path.write_text(json.dumps(_manifest, indent=2, default=str),
                                      encoding="utf-8")
        except Exception as exc:
            logger.debug("Manifest seal failed: %s", exc)
        # Per-type bundle (§7.4): review → prisma.mmd + extraction.csv.
        if paper_type == "review" and isinstance(corpus, dict):
            try:
                paths.update(self._write_review_bundle(root, corpus))
            except Exception as exc:
                logger.warning("Review bundle write failed: %s", exc, exc_info=True)
        # M3 (§7.4): survey → codebook + response hash; case → triangulation.csv.
        if paper_type == "survey" and isinstance(survey_analysis or instrument, dict):
            try:
                paths.update(self._write_survey_bundle(root, survey_analysis or {}, instrument or {}))
            except Exception as exc:
                logger.warning("Survey bundle write failed: %s", exc, exc_info=True)
        if paper_type == "case_study" and isinstance(case_protocol, dict):
            try:
                paths.update(self._write_case_bundle(root, case_protocol))
            except Exception as exc:
                logger.warning("Case bundle write failed: %s", exc, exc_info=True)
        # M4 (§7.4): cause_effect → dag.json + sensitivity.md (+ DAG.png best-effort).
        if paper_type == "cause_effect" and isinstance(causal_claim, dict):
            try:
                paths.update(self._write_causal_bundle(root, causal_claim))
            except Exception as exc:
                logger.warning("Causal bundle write failed: %s", exc, exc_info=True)
        return paths

    def _write_causal_bundle(self, root: Path, causal_claim: Dict[str, Any]) -> Dict[str, str]:
        """Write dag.json + sensitivity.md for cause-effect types (§7.4, M4).

        DAG.png is rendered best-effort with the pinned scientific stack
        (networkx + matplotlib Agg, seeded layout); a render failure never
        blocks assembly — dag.json remains the machine-readable source.
        """
        import hashlib as _hashlib

        dag = causal_claim.get("dag") if isinstance(causal_claim.get("dag"), dict) else {}
        dag_path = root / "dag.json"
        dag_text = json.dumps({"dag": dag, "id_strategy": causal_claim.get("id_strategy"),
                               "verb_level": causal_claim.get("verb_level")}, indent=2, sort_keys=True)
        dag_path.write_text(dag_text, encoding="utf-8")
        sensitivity = str(causal_claim.get("sensitivity") or "No sensitivity analysis reported.")
        placebo = causal_claim.get("placebo") if isinstance(causal_claim.get("placebo"), dict) else {}
        limitations = causal_claim.get("limitations") or []
        md_lines = [
            "# Sensitivity analysis",
            "",
            f"Identification strategy: {causal_claim.get('id_strategy') or 'unknown'}",
            f"Verb level: {causal_claim.get('verb_level') or 'associational'}",
            "",
            "## Sensitivity note",
            sensitivity,
            "",
            "## Placebo / refutation",
            ("Feasible: " + str(placebo.get("hint") or "")) if placebo.get("feasible")
            else ("Not feasible in sandbox: " + str(placebo.get("limitation") or "")),
            "",
            "## Limitations",
            *[f"- {item}" for item in limitations],
        ]
        md_path = root / "sensitivity.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        paths = {
            "dag_json": str(dag_path),
            "dag_sha256": _hashlib.sha256(dag_text.encode("utf-8")).hexdigest(),
            "sensitivity_md": str(md_path),
        }
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx

            treatment = str(dag.get("treatment") or "treatment")
            outcome = str(dag.get("outcome") or "outcome")
            confounders = [str(c) for c in (dag.get("confounders") or []) if str(c).strip()]
            graph = nx.DiGraph()
            graph.add_node(treatment)
            graph.add_node(outcome)
            graph.add_edge(treatment, outcome)
            for confounder in confounders[:8]:
                graph.add_node(confounder)
                graph.add_edge(confounder, treatment)
                graph.add_edge(confounder, outcome)
            pos = nx.spring_layout(graph, seed=42)
            plt.figure(figsize=(6, 4))
            nx.draw(graph, pos, with_labels=True, node_color="lightblue",
                    arrowsize=20, font_size=9)
            png_path = root / "DAG.png"
            plt.savefig(str(png_path), bbox_inches="tight")
            plt.close()
            paths["dag_png"] = str(png_path)
        except Exception as exc:
            logger.warning("DAG.png render failed (dag.json retained): %s", exc)
        return paths

    def _write_review_bundle(self, root: Path, corpus: Dict[str, Any]) -> Dict[str, str]:
        """Write prisma.mmd + extraction.csv for review types (§7.4, M2)."""
        import csv as _csv
        import hashlib as _hashlib

        prisma = corpus.get("prisma") if isinstance(corpus.get("prisma"), dict) else {}
        counts = {k: prisma.get(k, 0) for k in (
            "identified", "screened", "duplicates_excluded", "eligible",
            "excluded_title_abstract", "included", "excluded_fulltext")}
        mmd = (
            "flowchart TD\n"
            f"    identified[\"Identified: {counts['identified']}\"] --> screened[\"Screened: {counts['screened']}\"]\n"
            f"    identified --> dup[\"Duplicates excluded: {counts['duplicates_excluded']}\"]\n"
            f"    screened --> eligible[\"Eligible: {counts['eligible']}\"]\n"
            f"    screened --> excl_ta[\"Excluded title/abstract: {counts['excluded_title_abstract']}\"]\n"
            f"    eligible --> included[\"Included: {counts['included']}\"]\n"
            f"    eligible --> excl_ft[\"Excluded fulltext: {counts['excluded_fulltext']}\"]\n"
        )
        mmd_path = root / "prisma.mmd"
        mmd_path.write_text(mmd, encoding="utf-8")
        rows = corpus.get("extraction") if isinstance(corpus.get("extraction"), list) else []
        csv_path = root / "extraction.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=["design", "sample_n", "effect", "metric", "limitation", "doi"])
            writer.writeheader()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                writer.writerow({k: row.get(k, "") for k in ("design", "sample_n", "effect", "metric", "limitation", "doi")})
        log_agent_action("EditorAgent", "review_bundle_written", {
            "prisma": str(mmd_path), "extraction": str(csv_path), "n_rows": len(rows),
        })
        try:
            from core.run_log import emit_event as _emit

            for label, path in (("prisma.mmd", mmd_path), ("extraction.csv", csv_path)):
                digest = _hashlib.sha256(path.read_bytes()).hexdigest()
                _emit("artifact_write", {
                    "phase": "editing", "agent": "EditorAgent", "status": "ok",
                    "message_human": f"review bundle {label} ({len(rows) if label.endswith('.csv') else counts['included']} rows)",
                    "artifact_path": str(path), "sha256": digest,
                    "rows": len(rows), "cause_class": "science"},
                    agent="EditorAgent")
        except Exception:
            pass
        return {"prisma_mmd": str(mmd_path), "extraction_csv": str(csv_path)}

    def _write_survey_bundle(self, root: Path, analysis: Dict[str, Any], instrument: Dict[str, Any]) -> Dict[str, str]:
        """Write codebook.csv + response-hash record for survey types (§7.4, M3)."""
        import csv as _csv
        import hashlib as _hashlib

        instrument = dict(instrument or (analysis or {}).get("instrument") or {})
        nonresponse = dict(instrument.get("nonresponse") or (analysis.get("nonresponse") if isinstance(analysis, dict) else {}) or {})
        codebook_path = root / "codebook.csv"
        with open(codebook_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=["item", "target", "scale", "subgroup_preregistered"])
            writer.writeheader()
            wrote = False
            for key, value in (nonresponse.get("item_rates") or {}).items():
                writer.writerow({"item": key, "target": "", "scale": "",
                                 "subgroup_preregistered": ""})
                wrote = True
            if not wrote:
                writer.writerow({"item": "response", "target": "target",
                                 "scale": "scale", "subgroup_preregistered": ""})
        hash_path = root / "response_hash.txt"
        hash_path.write_text(str(instrument.get("response_hash") or "") + "\n", encoding="utf-8")
        log_agent_action("EditorAgent", "survey_bundle_written", {
            "codebook": str(codebook_path),
            "response_hash": str(hash_path),
        })
        try:
            from core.run_log import emit_event as _emit

            digest = _hashlib.sha256(codebook_path.read_bytes()).hexdigest()
            _emit("artifact_write", {
                "phase": "editing", "agent": "EditorAgent", "status": "ok",
                "message_human": "survey bundle codebook.csv",
                "artifact_path": str(codebook_path), "sha256": digest,
                "rows": len(nonresponse.get("item_rates") or {}), "cause_class": "science"},
                agent="EditorAgent")
        except Exception:
            pass
        return {"codebook_csv": str(codebook_path), "response_hash": str(hash_path)}

    def _write_case_bundle(self, root: Path, protocol: Dict[str, Any]) -> Dict[str, str]:
        """Write triangulation.csv for case types (§7.4, M3)."""
        import csv as _csv
        import hashlib as _hashlib

        triang = (protocol or {}).get("triangulation") if isinstance(protocol, dict) else {}
        matrix = (triang or {}).get("matrix") if isinstance(triang, dict) else {}
        csv_path = root / "triangulation.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=["claim", "sources", "independent_types", "flag"])
            writer.writeheader()
            flags = {str(fl.get("claim", "")): fl for fl in ((triang or {}).get("single_source_flags") or []) if isinstance(fl, dict)}
            for claim, sources in (matrix or {}).items():
                flag = flags.get(str(claim), {})
                types = sorted({str(s.get("type") or "") for s in (sources or []) if isinstance(s, dict)})
                writer.writerow({"claim": str(claim)[:500],
                                 "sources": "; ".join(str(s.get("ref") or "") for s in (sources or []) if isinstance(s, dict))[:5],
                                 "independent_types": "|".join(types),
                                 "flag": flag.get("flag", "")})
        log_agent_action("EditorAgent", "case_bundle_written", {
            "triangulation": str(csv_path), "n_claims": len(matrix or {})})
        try:
            from core.run_log import emit_event as _emit

            digest = _hashlib.sha256(csv_path.read_bytes()).hexdigest()
            _emit("artifact_write", {
                "phase": "editing", "agent": "EditorAgent", "status": "ok",
                "message_human": f"case bundle triangulation.csv ({len(matrix or {})} claims)",
                "artifact_path": str(csv_path), "sha256": digest,
                "rows": len(matrix or {}), "cause_class": "science"},
                agent="EditorAgent")
        except Exception:
            pass
        return {"triangulation_csv": str(csv_path)}

    @staticmethod
    def _git_commit() -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    def _skeleton_path(self, paper_type: str) -> Optional[str]:
        """Skeleton template for one paper type (§7.4), if it exists."""
        base = Path(__file__).resolve().parent.parent / "templates" / f"latex_{paper_type}.tex"
        if base.is_file():
            return str(base)
        # Proposal §7.4 names the case skeleton latex_case.tex while the
        # stored paper_type is case_study — accept the proposal literal too.
        if paper_type == "case_study":
            alias = Path(__file__).resolve().parent.parent / "templates" / "latex_case.tex"
            if alias.is_file():
                return str(alias)
        return None

    def _generate_latex_document(self, topic, sections, plan, paper_type: str = "experimental") -> str:
        abstract = self._extract_abstract(sections) if sections.get("Abstract") else topic.get("description", "")
        order = MANUSCRIPT_ORDERS.get(paper_type) or MANUSCRIPT_ORDERS["experimental"]
        body_parts = []
        seen = set()
        for name in order:
            if name in sections and name != "Abstract":
                body_parts.append(f"\\section{{{name}}}\n\n{self._process_section_content(sections[name], name)}\n")
                seen.add(name)
        for name, content in sections.items():
            if name not in seen and name != "Abstract":
                body_parts.append(f"\\section{{{name}}}\n\n{self._process_section_content(content, name)}\n")
        body = "\n\n".join(body_parts)
        skeleton = self._skeleton_path(paper_type)
        if skeleton:
            try:
                template = Path(skeleton).read_text(encoding="utf-8")
                if all(token in template for token in SKELETON_TOKENS):
                    filled = (
                        template.replace("%TITLE%", topic.get("title", "Research Paper"))
                        .replace("%ABSTRACT%", self._process_section_content(abstract, "Abstract"))
                        .replace("%SECTIONS%", body)
                        .replace("%BIBLIOGRAPHY%", "\\bibliographystyle{plain}\n\\bibliography{references}")
                    )
                    log_agent_action("EditorAgent", "skeleton_used", {"paper_type": paper_type, "skeleton": skeleton})
                    return filled
                logger.warning("Skeleton %s missing tokens; using default layout", skeleton)
            except Exception as exc:
                logger.warning("Skeleton %s unreadable (%s); using default layout", skeleton, exc)
        header = self._create_document_header(topic, plan, abstract)
        footer = self._create_document_footer()
        return header + "\n\n" + body + "\n\n" + footer

    def _create_document_header(self, topic, plan, abstract: str = "") -> str:
        abs_block = abstract.replace("&", "\\&") if abstract else "Abstract pending."
        return f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{graphicx,hyperref,geometry,natbib,booktabs}}
\\geometry{{margin=1in}}
\\title{{{topic.get('title', 'Research Paper')}}}
\\author{{ScholarGraph Research System}}
\\date{{\\today}}
\\begin{{document}}
\\maketitle
\\begin{{abstract}}
{abs_block}
\\end{{abstract}}
"""

    def _create_document_footer(self) -> str:
        return "\\bibliographystyle{plain}\n\\bibliography{references}\n\\end{document}\n"

    def _process_section_content(self, content: str, section_name: str) -> str:
        content = re.sub(r"^#+\s*.*$", "", content, flags=re.MULTILINE)
        content = content.replace("&", "\\&").replace("%", "\\%")
        return content.strip()

    def _extract_abstract(self, sections: Dict[str, str]) -> str:
        if "Abstract" in sections:
            return re.sub(r"^#+\s*Abstract\s*", "", sections["Abstract"]).strip()
        return "Abstract to be added."
