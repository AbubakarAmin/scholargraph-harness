"""Chat skill router — drives the harness through the same APIs as the dashboard.

The chat UI is a friendly control plane on top of the harness. Skills map
operator intent to concrete harness actions (runs, config, packs, research,
docs, approvals). No auth gate — open access matching the Control Deck.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Pipeline stages shown in the chat stage rail (experimental default recipe).
DEFAULT_STAGES = [
    "intake",
    "topic_discovery",
    "hypothesis_debate",
    "planning",
    "data_validation",
    "engineering",
    "independent_validation",
    "writing_narrative",
    "writing_results",
    "supervision",
    "editing",
]

SKILL_KINDS = (
    "start_run",
    "research",
    "update_config",
    "upload_pack",
    "generate_doc",
    "approve",
    "reject",
    "resume_run",
    "inspect_run",
    "list_runs",
    "status",
    "help",
    "chat",
)


def parse_domain_topic_pairs(text: str) -> List[Dict[str, str]]:
    """Extract domain/topic pairs from free text or CSV-like uploads."""
    pairs: List[Dict[str, str]] = []
    is_first_pair = True
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # CSV: domain,topic  |  TSV  |  "domain | topic"  |  "domain: topic"
        if "," in line:
            parts = [p.strip().strip('"') for p in line.split(",", 1)]
        elif "\t" in line:
            parts = [p.strip() for p in line.split("\t", 1)]
        elif "|" in line:
            parts = [p.strip() for p in line.split("|", 1)]
        elif ":" in line and not line.lower().startswith("http"):
            parts = [p.strip() for p in line.split(":", 1)]
        else:
            continue
        if len(parts) == 2 and parts[0] and parts[1]:
            if is_first_pair:
                headers = [part.strip().strip('"').lower() for part in parts]
                if headers[0] in {"domain", "research_domain"} and headers[1] in {
                    "topic",
                    "goal",
                    "research_question",
                }:
                    is_first_pair = False
                    continue
            is_first_pair = False
            pairs.append({"domain": parts[0], "topic": parts[1]})
    return pairs


def classify_intent(message: str, has_files: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Lightweight intent classifier (no LLM required for control verbs)."""
    text = (message or "").strip()
    lower = text.lower()

    if has_files and (
        "upload" in lower
        or "pack" in lower
        or "domain" in lower
        or "topic" in lower
        or "start" in lower
        or "run" in lower
        or not text
    ):
        return "upload_pack", {}

    if re.search(r"\b(approve|ship|publish|release)\b", lower):
        m = re.search(r"\b([a-f0-9]{8})\b", lower)
        return "approve", {"run_id": m.group(1) if m else None}

    if re.search(r"\b(reject|deny|block)\b", lower) and "approv" in lower:
        m = re.search(r"\b([a-f0-9]{8})\b", lower)
        return "reject", {"run_id": m.group(1) if m else None}

    if re.search(r"\b(resume|continue)\b", lower):
        m = re.search(r"\b([a-f0-9]{8})\b", lower)
        return "resume_run", {"run_id": m.group(1) if m else None}

    if re.search(r"\b(list runs|show runs|active runs|what.?s running)\b", lower):
        return "list_runs", {}

    if re.search(r"\b(status|where are we|what.?s happening|progress)\b", lower):
        m = re.search(r"\b([a-f0-9]{8})\b", lower)
        return "status", {"run_id": m.group(1) if m else None}

    if re.search(r"\b(open|inspect|overview|show)\b.*\brun\b", lower) or re.search(
        r"\brun\s+([a-f0-9]{8})\b", lower
    ):
        m = re.search(r"\b([a-f0-9]{8})\b", lower)
        return "inspect_run", {"run_id": m.group(1) if m else None}

    if re.search(r"\b(summarize|summary|generate (a )?(doc|document|report|pdf|markdown))\b", lower):
        m = re.search(r"\b([a-f0-9]{8})\b", lower)
        fmt = "markdown"
        if "pdf" in lower:
            fmt = "pdf"
        elif "html" in lower:
            fmt = "html"
        elif "latex" in lower or "tex" in lower:
            fmt = "latex"
        return "generate_doc", {"run_id": m.group(1) if m else None, "format": fmt}

    if re.search(r"\b(set|configure|update|change)\b.*(model|provider|domain|depth|threshold|key)", lower):
        return "update_config", {"raw": text}

    if re.search(r"\b(start|launch|kick off|run research|begin)\b", lower) or re.search(
        r"\bresearch\b.+\bon\b", lower
    ):
        domain = None
        topic = None
        goal = text
        dm = re.search(r"\bdomain\s*[:=]\s*([^\n,;]+)", text, re.I)
        tm = re.search(r"\btopic\s*[:=]\s*([^\n,;]+)", text, re.I)
        if dm:
            domain = dm.group(1).strip()
        if tm:
            topic = tm.group(1).strip()
            goal = topic
        # "research X in Y" / "research X on Y"
        rm = re.search(
            r"research\s+(.+?)\s+(?:in|on|for)\s+(.+?)(?:\.|$)",
            text,
            re.I,
        )
        if rm and not topic:
            topic = rm.group(1).strip()
            domain = domain or rm.group(2).strip()
            goal = topic
        depth = "Standard"
        if "quick" in lower:
            depth = "Quick"
        elif "thorough" in lower or "deep" in lower:
            depth = "Thorough"
        return "start_run", {
            "domain": domain,
            "goal": goal if goal else (topic or text),
            "depth": depth,
            "paper_type": "auto",
            "hunt_mode": "try_and_report",
        }

    if re.search(r"\b(help|what can you do|capabilities)\b", lower):
        return "help", {}

    return "chat", {}


def extract_config_patches(raw: str) -> Dict[str, str]:
    """Pull KEY=value or 'set model to X' style patches from operator text."""
    patches: Dict[str, str] = {}
    for m in re.finditer(
        r"\b([A-Z][A-Z0-9_]{2,})\s*[:=]\s*([^\n,;]+)",
        raw or "",
    ):
        patches[m.group(1)] = m.group(2).strip().strip('"\'')

    lower = (raw or "").lower()
    mm = re.search(r"model\s+(?:to|=)\s*([a-z0-9._\-]+)", lower)
    if mm:
        patches["GEMINI_MODEL"] = mm.group(1)
    pm = re.search(r"provider\s+(?:to|=)\s*(gemini|openai|openai_compatible)", lower)
    if pm:
        patches["LLM_PROVIDER"] = pm.group(1)
    dm = re.search(r"domain\s+(?:to|=)\s*([^\n,.]+)", lower)
    if dm:
        patches["RESEARCH_DOMAIN"] = dm.group(1).strip()
    return patches


def build_run_overview(dash: Dict[str, Any], status: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal scannable run overview for chat cards."""
    workspace = dash.get("workspace") if isinstance(dash.get("workspace"), dict) else {}
    pipeline = dash.get("pipeline") or []
    if not pipeline:
        phase = dash.get("phase") or "idle"
        stages = workspace.get("pipeline_recipe") or DEFAULT_STAGES
        pipeline = []
        hit = False
        for name in stages:
            name = name if isinstance(name, str) else str(name.get("phase") or name)
            if name == phase:
                pipeline.append({"phase": name, "state": "active", "reason": ""})
                hit = True
            elif not hit and phase not in ("idle", "completed", None):
                pipeline.append({"phase": name, "state": "done", "reason": ""})
            else:
                pipeline.append({"phase": name, "state": "pending", "reason": ""})

    topic = ""
    selected = workspace.get("selected_topic") or {}
    if isinstance(selected, dict):
        topic = selected.get("title") or selected.get("topic") or ""
    elif isinstance(selected, str):
        topic = selected

    plan = workspace.get("plan") or {}
    experiments: List[Dict[str, Any]] = []
    if isinstance(plan, dict):
        for item in plan.get("experiments") or plan.get("contributions") or []:
            if isinstance(item, str):
                experiments.append({"title": item, "status": "planned"})
            elif isinstance(item, dict):
                experiments.append({
                    "title": item.get("title") or item.get("claim") or "Experiment",
                    "status": item.get("status") or "planned",
                    "hypothesis": item.get("hypothesis") or item.get("falsifiable_prediction") or "",
                })

    active_agent = None
    events = dash.get("events_tail") or []
    if events:
        last = events[-1] if isinstance(events[-1], dict) else {}
        active_agent = last.get("agent")

    running = bool(status.get("running"))
    release = dash.get("release") or {}
    run_status = "active" if running else (dash.get("status") or "idle")
    if release.get("needs_approval"):
        run_status = "awaiting_approval"
    elif dash.get("status") == "completed" or (not running and dash.get("phase") in ("complete", "editing", "completed")):
        if not running and dash.get("status") in ("completed", "failed", "interrupted"):
            run_status = dash.get("status")

    return {
        "run_id": dash.get("run_id"),
        "status": run_status,
        "phase": dash.get("phase") or "idle",
        "topic": topic,
        "active_agent": active_agent,
        "active_task": (events[-1].get("type") if events and isinstance(events[-1], dict) else None),
        "pipeline": pipeline,
        "plan_experiments": experiments[:8],
        "release": {
            "needs_approval": bool(release.get("needs_approval")),
            "ready": bool(release.get("ready")),
            "status": release.get("status"),
            "missing": release.get("missing") or [],
            "reason": release.get("reason") or "",
        },
        "spend": dash.get("spend") or {},
        "readiness": dash.get("readiness") or release.get("readiness"),
        "paper_type": workspace.get("paper_type") or "auto",
        "goal": (workspace.get("intake_report") or {}).get("goal")
        if isinstance(workspace.get("intake_report"), dict)
        else workspace.get("goal"),
    }


def help_text() -> str:
    return (
        "I'm the ScholarGraph chat control plane — same harness as the dashboard, "
        "driven conversationally.\n\n"
        "**What I can do**\n"
        "- Start research runs (`research X on Y`, `start a thorough run on …`)\n"
        "- Upload domain/topic lists or field packs and kick off runs\n"
        "- Inspect any run (stage, active agent, plan/experiments)\n"
        "- Update harness config (model, provider, domain, thresholds)\n"
        "- Approve / reject release gates with inline buttons\n"
        "- Summarize a run into markdown / HTML / LaTeX\n"
        "- List active and recent runs\n\n"
        "Switch to the full **Control Deck** anytime with the dashboard button."
    )


def summarize_run_fallback(overview: Dict[str, Any]) -> str:
    """Offline-safe document body when LLM is unavailable."""
    lines = [
        f"# Run {overview.get('run_id') or '—'}",
        "",
        f"- **Status:** {overview.get('status')}",
        f"- **Phase:** {overview.get('phase')}",
        f"- **Topic:** {overview.get('topic') or '—'}",
        f"- **Paper type:** {overview.get('paper_type')}",
        f"- **Active agent:** {overview.get('active_agent') or '—'}",
        "",
        "## Pipeline",
    ]
    for step in overview.get("pipeline") or []:
        mark = {"done": "✓", "active": "●", "pending": "○", "skipped": "–"}.get(
            step.get("state"), "·"
        )
        lines.append(f"- {mark} `{step.get('phase')}` ({step.get('state')})")
    lines.append("")
    lines.append("## Planned experiments")
    exps = overview.get("plan_experiments") or []
    if not exps:
        lines.append("_None recorded yet._")
    else:
        for ex in exps:
            lines.append(f"- **{ex.get('title')}** — {ex.get('status')}")
            if ex.get("hypothesis"):
                lines.append(f"  - {ex['hypothesis']}")
    release = overview.get("release") or {}
    lines.extend([
        "",
        "## Release",
        f"- Status: {release.get('status')}",
        f"- Needs approval: {release.get('needs_approval')}",
        f"- Missing: {', '.join(release.get('missing') or []) or '—'}",
    ])
    return "\n".join(lines)
