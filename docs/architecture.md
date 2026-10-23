# Architecture

ScholarGraph is a Python application that runs a paper-type-specific research workflow. The CLI and web interface share the same pipeline and state model.

## Runtime at a glance

```text
CLI (main.py) ───────┐
                     ├── ResearchPipeline ── LangGraph workflow
Web UI (web/app.py) ─┘                            │
                                                  ├── agents/
                                                  ├── core services
                                                  └── run state and artifacts
```

`main.py` is the command-line composition root. `run_ui.py` launches the FastAPI application in `web/app.py`. Both use `core.pipeline.ResearchPipeline` to execute and stream workflow phases.

## Main responsibilities

| Area | Responsibility |
|---|---|
| `agents/` | Role-specific research work: discovery, debate, planning, writing, engineering, analysis, verification, supervision, and editing. |
| `core/state.py` | Typed shared state passed between workflow phases. |
| `core/phase_registry.py` | Supported paper types, phase metadata, pipeline recipes, and evidence requirements. |
| `core/workflow.py` and `core/workflow_nodes.py` | Graph construction, routing, and phase-node implementations. |
| `core/pipeline.py` | Shared graph execution, run context, checkpoint-aware streaming, and finalization. |
| `core/llm.py` and `core/api_gateway.py` | Language-model access and shared controls for external API traffic. |
| `core/evidence_gate.py` and `core/contracts.py` | Experiment contracts and typed evidence handoffs. |
| `core/verification.py` | Deterministic checks for evidence, statistics, citations, and manuscript consistency. |
| `core/sandbox.py` and `core/sandbox_dispatch.py` | Configurable code-execution backends and dispatch. |
| `core/research_db.py`, `core/run_log.py`, and `core/memory.py` | Durable run records, events, logs, and reusable memory. |
| `web/` | FastAPI endpoints and the static Control Deck interface. |

## Data flow and ownership

1. Intake selects a paper type and its recipe.
2. The graph executes that recipe and passes a shared `ResearchState` between phases.
3. Agents use core services for provider calls, retrieval, contracts, execution, and verification rather than defining a separate workflow.
4. `RunContext` carries run-scoped dependencies. Checkpoints and the research ledger preserve run state; generated papers, experiment outputs, and other deliverables are written under `output/`.

The full-research recipe commonly includes discovery, debate, planning, data validation, writing, engineering, independent validation, supervision, and editing. Other paper types omit or replace phases; see [Workflow and paper types](workflow.md).

## Important boundaries

- `core.workflow` assembles the state machine; `core.workflow_nodes` implements graph nodes.
- `core.phase_registry` is the source for paper-type recipes and evidence requirements.
- `core.evidence_gate` checks that experiment execution remains tied to its committed contract. Verification and review can reject work; an LLM score does not establish scientific correctness.
- `core.api_gateway` centralizes external API pacing and resilience. New external source calls should use the gateway.
- The AST sandbox is an application-level restriction, not an OS isolation boundary. Docker execution is also configurable and must be operated with appropriate host-level protections.
- Local state under `memory/`, `output/`, and `logs/` can contain credentials, research data, or run artifacts. Do not commit private or generated state.

## Repository map

| Path | Purpose |
|---|---|
| `agents/` | Research agent implementations |
| `core/` | Shared workflow, configuration, persistence, source, and verification services |
| `data/catalog/` | Local catalog data |
| `docs/` | Maintained user and architecture guides |
| `skills/` | Research and writing guidance used by the application |
| `templates/` | Paper-generation templates |
| `tests/` | Offline and mocked pytest suite |
| `web/` | API and static web interface |
| `output/`, `memory/`, `logs/` | Local generated artifacts and durable state; keep private and untracked |
