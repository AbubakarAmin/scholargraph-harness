# ScholarGraph

ScholarGraph is a local-first, multi-agent research workflow. It combines literature discovery, paper-type-specific analysis, experiment planning and execution, independent validation, review, and paper assembly. It is designed to make evidence and limitations inspectable—not to guarantee that a result is novel, correct, or publishable.

## Quick start

### Prerequisites

- Python 3.10 or newer.
- [Docker Desktop](https://docs.docker.com/desktop/) installed and running for sandboxed experiment execution.
- [OpenCode](https://opencode.ai/docs/) installed and available on `PATH` to use the engineering arena. With `ENGINEERING_BACKEND=auto` (the default), ScholarGraph falls back to the legacy LLM-based engineering path if OpenCode is unavailable.

Install the Python dependencies and create your local environment file:

```powershell
python -m pip install -r requirements.txt
Copy-Item env_example.txt .env
```

Edit `.env` and set credentials for the chosen language-model provider and a real `OPENALEX_EMAIL`. Keep `.env` and any API keys private.

Start the local web interface:

```powershell
python run_ui.py
```

Then open <http://127.0.0.1:8765>. To run from the command line instead:

```powershell
python main.py
```

Run the offline tests and quick smoke check with:

```powershell
python -m pytest tests -q
python tests/smoke_offline.py
```

The pytest suite is the main test suite. The smoke script is a smaller check; neither is a substitute for reviewing the generated evidence. Most tests use mocks and do not require live provider credentials.

## How it works

ScholarGraph selects a workflow recipe for the requested paper type and coordinates specialist agents through a LangGraph state machine. Depending on that recipe, it can discover literature, develop and review a question, plan an analysis or experiment, validate inputs, run code, independently analyze results, check claims, and assemble a manuscript with supporting artifacts.

The workflow records run state and evidence so that results can be inspected or resumed. See the [architecture](docs/architecture.md) and [workflow guide](docs/workflow.md) for how those pieces fit together.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Research workflow and paper types](docs/workflow.md)
- [Configuration](docs/configuration.md)
- [Testing](docs/testing.md)
- [Running, outputs, and recovery](docs/operations.md)

## Important limitations

- A completed run or passing gate is not proof of scientific validity, novelty, or publication readiness. Review sources, data, methods, and conclusions independently.
- Generated experiment code is subject to application-level checks and a configurable execution backend. The AST sandbox is not an operating-system security boundary; use an appropriately isolated environment for untrusted code.
- Runs may call external language-model and scholarly APIs, consume quota, and incur costs. Use offline tests before starting a live run.
- Restart `run_ui.py` after changing backend Python code; the application does not hot-reload.

## Interface assets

The local-first control deck vendors its UI dependencies under `web/static/` so it remains usable on unreliable venue Wi-Fi. The visual direction takes inspiration from the interaction and transition studies in [Codrops](https://tympanus.net/codrops/), the open-source glass-card patterns from [Uiverse](https://uiverse.io/), and the particle examples in [Three.js](https://threejs.org/examples/). It does not copy their layouts or assets. Vendored libraries: Three.js (MIT), GSAP (standard no-charge web license), Alpine.js (MIT), Lucide (ISC), Lottie-web (MIT), daisyUI (MIT), and Google Fonts' Playfair Display and Manrope (SIL Open Font License). The ambient particle field is original code and has a CSS-only fallback.
