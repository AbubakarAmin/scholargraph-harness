# ScholarGraph

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://www.langchain.com/langgraph"><img src="https://img.shields.io/badge/LangGraph-workflows-1C3C3C?style=for-the-badge" alt="LangGraph workflows" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-local%20UI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI local UI" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-sandboxed%20runs-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker sandboxed runs" /></a>
  <a href="https://pytest.org/"><img src="https://img.shields.io/badge/pytest-offline%20tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest offline tests" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License" /></a>
</p>

<p align="center">
  <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/Gemini-LLM%20%26%20embeddings-8E75B2?style=flat-square" alt="Gemini" /></a>
  <a href="https://platform.openai.com/"><img src="https://img.shields.io/badge/OpenAI-compatible-LLM-412991?style=flat-square" alt="OpenAI-compatible LLMs" /></a>
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-literature-B31B1B?style=flat-square" alt="arXiv literature search" /></a>
  <a href="https://openalex.org/"><img src="https://img.shields.io/badge/OpenAlex-literature-1769AA?style=flat-square" alt="OpenAlex literature search" /></a>
  <a href="https://www.semanticscholar.org/"><img src="https://img.shields.io/badge/Semantic%20Scholar-citations-1857B6?style=flat-square" alt="Semantic Scholar" /></a>
  <a href="https://openreview.net/"><img src="https://img.shields.io/badge/OpenReview-venue%20signals-6A5ACD?style=flat-square" alt="OpenReview" /></a>
  <a href="https://huggingface.co/datasets"><img src="https://img.shields.io/badge/Hugging%20Face-datasets-FFD21E?style=flat-square" alt="Hugging Face datasets" /></a>
</p>

**An evidence-oriented research workflow, not just a paper generator.**

ScholarGraph coordinates specialist agents to discover literature, shape a
research question, validate plans and data, run or analyze work where
appropriate, review claims, and assemble a manuscript with its supporting
artifacts. A recipe-aware workflow adapts those steps to the kind of paper
being developed.

**Explore:** [How it works](#how-it-works) · [Quick start](#quick-start) ·
[Paper types](#paper-types) · [Integrations](#integrations) ·
[Documentation](#documentation)

## Integrations

ScholarGraph connects to external services for language-model calls, research
discovery, open-access evidence, and optional dataset or experiment workflows.
These are configurable integrations—not endorsements or guarantees that a
particular source will return results for every query.

| Service | How ScholarGraph uses it | Setup / notes |
|---|---|---|
| [Google Gemini](https://ai.google.dev/) | Language-model chat when `LLM_PROVIDER=gemini`; also powers the configured text-embedding model used by research memory and similarity features. | Set `GOOGLE_API_KEY`. Embeddings use Gemini regardless of the selected chat provider. |
| [OpenAI](https://platform.openai.com/) and OpenAI-compatible APIs | Optional language-model chat through OpenAI or compatible endpoints such as OpenRouter, Groq, and local servers. | Set `LLM_PROVIDER=openai` or `openai_compatible`; configure `OPENAI_API_KEY` and, for compatible endpoints, `OPENAI_BASE_URL`. |
| [arXiv](https://arxiv.org/) | Searches scholarly preprints for topic discovery and literature evidence. | Enabled by default; configure with `ARXIV_ENABLED`. |
| [OpenAlex](https://openalex.org/) | Searches scholarly works and metadata, and supplies related-work and citation signals for discovery. | Enabled by default; provide a real `OPENALEX_EMAIL`; configure with `OPENALEX_ENABLED`. |
| [Semantic Scholar](https://www.semanticscholar.org/) | Adds paper metadata and citation-graph signals when available. | Controlled by `SEMANTIC_SCHOLAR_ENABLED`; an API key can be set with `SEMANTIC_SCHOLAR_API_KEY`. |
| [OpenReview](https://openreview.net/) | Retrieves venue submissions and review-related signals to inform topic discovery. | Configurable with `OPENREVIEW_ENABLED`; credentials may be needed for restricted content. |
| [Crossref](https://www.crossref.org/) | Resolves and checks DOI and publication metadata used in citation and evidence workflows. | Used as a metadata source where relevant; no API key is configured by ScholarGraph. |
| [Unpaywall](https://unpaywall.org/), [PubMed Central](https://pmc.ncbi.nlm.nih.gov/), and [DOAJ](https://doaj.org/) | Help locate open-access article text or verify open-access journal status for evidence workflows. | Availability depends on identifiers, source coverage, and the relevant source settings. |
| [Hugging Face Hub](https://huggingface.co/datasets) | Searches for public datasets and supports bounded dataset downloads when a workflow requests them. | Public datasets can be accessed without a token; set `HUGGINGFACE_TOKEN` when needed. |
| [OpenCode](https://opencode.ai/) and Docker | Optionally drafts experiment code with OpenCode; ScholarGraph runs arena experiments in Docker and validates the resulting artifacts. | OpenCode is optional with `ENGINEERING_BACKEND=auto`; Docker must be available for arena execution. |

Source calls are routed through the shared API gateway for configured rate
limits, retries, and provider health checks. Enable or disable scholarly sources
in `.env`; see [Configuration](docs/configuration.md) and `env_example.txt` for
the current settings.

## How it works

The CLI and web interface use the same research pipeline. The selected paper
type determines which phases run; not every project requires an experiment.

```text
Research request
      │
      ▼
Intake and recipe selection
      │
      ▼
Literature discovery ──► debate and planning (when applicable)
      │
      ▼
Data validation / analysis / experiment (recipe-dependent)
      │
      ▼
Independent checks ──► supervision and editing
      │
      ▼
Paper and research artifacts
```

The system is designed to make sources, decisions, results, and limitations
inspectable. Passing a workflow check does not prove that a finding is correct,
novel, generalizable, or ready for publication.

## What it provides

- **Recipe-aware research workflows:** distinct phase sequences for ten paper
  types, including experimental, review, survey, case-study, and causal work.
- **Specialist agents:** literature discovery, hypothesis debate, planning,
  data validation, engineering, analysis, verification, supervision, and
  editing.
- **Evidence and experiment controls:** structured research contracts, bounded
  execution, recorded results, and independent validation where the recipe
  calls for them.
- **Two ways to run:** a local FastAPI Control Deck or the command-line
  interface, sharing the same pipeline and run state.
- **Inspectable outputs:** generated manuscripts, experiment results, logs,
  and durable run state for diagnosis or recovery.

## Quick start

### Requirements

- Python 3.10 or newer.
- Docker Desktop installed and running for Docker-backed experiment execution.
- A language-model provider API key and a real `OPENALEX_EMAIL`.
- OpenCode installed and available on `PATH` to use the engineering arena.
  OpenCode is optional with the default `ENGINEERING_BACKEND=auto`, which falls
  back to the legacy LLM engineering path when OpenCode is unavailable.

### Install

In PowerShell:

```powershell
git clone https://github.com/AbubakarAmin/scholargraph-harness.git
Set-Location scholargraph-harness
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item env_example.txt .env
```

On macOS or Linux, activate the virtual environment with
`source .venv/bin/activate` and create the environment file with
`cp env_example.txt .env`.

Edit `.env` and set `LLM_PROVIDER`, the matching provider credentials, and
`OPENALEX_EMAIL`. Keep `.env` and API keys private. See
[Configuration](docs/configuration.md) for supported providers, sources, and
runtime options.

### Run the web interface

```powershell
python run_ui.py
```

Open <http://127.0.0.1:8765>.

### Run from the command line

```powershell
python main.py
```

See available options with `python main.py --help`. To resume a run from its
durable checkpoint:

```powershell
python main.py --resume RUN_ID
```

## Paper types

The active phase registry defines these workflow recipes:

| Type | Typical focus |
|---|---|
| `experimental` | A planned, controlled experiment |
| `comparative` | Comparison of methods or conditions |
| `empirical` | Empirical analysis using declared data |
| `review` | Literature review and synthesis |
| `analytical` | Analysis built around a structured argument |
| `descriptive` | Description and analysis of a phenomenon or data |
| `argumentative` | Development and evaluation of a thesis |
| `cause_effect` | A causal question with a causal audit |
| `case_study` | Analysis grounded in a supplied case pack |
| `survey` | Analysis grounded in a supplied response table |

Recipes may require specific inputs, such as a dataset, case pack, or survey
response table. See [Workflow and paper types](docs/workflow.md) for the
current phase behavior and input requirements.

## Project map

| Path | Responsibility |
|---|---|
| `agents/` | Role-specific discovery, debate, planning, writing, engineering, analysis, review, and verification agents |
| `core/phase_registry.py` | Paper types, workflow recipes, and evidence requirements |
| `core/workflow.py`, `core/workflow_nodes.py` | Workflow graph, routing, and phase implementations |
| `core/pipeline.py`, `core/state.py` | Shared pipeline execution and research state |
| `core/contracts.py`, `core/evidence_gate.py` | Typed handoffs and experiment evidence gates |
| `core/verification.py`, `core/api_gateway.py` | Verification checks and resilient external API access |
| `web/` | FastAPI application and local Control Deck |
| `docs/` | Maintained setup, architecture, workflow, and operations guides |
| `tests/` | Offline and mocked pytest suite |
| `output/`, `memory/`, `logs/` | Local run artifacts, durable state, and logs; keep private and untracked |

## Testing

Run the offline test suite and the standalone smoke check:

```powershell
python -m pytest tests -q
python tests/smoke_offline.py
```

The tests are primarily mocked/offline and generally do not require provider
credentials. They do not validate the scientific quality of a live research
run.

## Important operating notes

- `SANDBOX_BACKEND` defaults to `docker`. Some legacy execution paths may fall
  back to the in-process AST backend if Docker is unavailable; arena experiments
  require Docker and fail closed rather than using that fallback.
- The AST sandbox is an application-level restriction, not an operating-system
  security boundary. Docker also requires appropriate host-level protections.
- OpenCode authors arena code but is a local process; it is not OS-isolated by
  ScholarGraph.
- Live runs call external language-model and scholarly APIs, may consume quota,
  and can incur costs. Review inputs and generated evidence independently.
- Restart `run_ui.py` after changing backend Python files; automatic reload is
  not enabled.
- `.env`, `memory/`, `output/`, and `logs/` may contain secrets, research data,
  or valuable run state. Back up what you need and do not commit private data.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Workflow and paper types](docs/workflow.md)
- [Configuration](docs/configuration.md)
- [Operations and recovery](docs/operations.md)
- [Testing](docs/testing.md)

## License

This project is licensed under the [MIT License](LICENSE).
