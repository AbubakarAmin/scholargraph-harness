# Configuration

Configuration is read from environment variables and the project-root `.env` file. The project root is resolved from `core/config.py`, so `.env` does not need to be in the process's current working directory. Start from `env_example.txt`; the available settings and code defaults may evolve, so consult those files when adding or changing options.

## Initial setup

```powershell
Copy-Item env_example.txt .env
```

Set a real `OPENALEX_EMAIL` and credentials for the selected language-model provider. Never commit `.env`, API keys, or `memory/keys.json`.

## Language-model providers

Set `LLM_PROVIDER` to one of:

| Value | Required settings |
|---|---|
| `gemini` | `GOOGLE_API_KEY`; optionally set `GEMINI_MODEL`. |
| `openai` | `OPENAI_API_KEY`; optionally set `OPENAI_MODEL` and `OPENAI_BASE_URL`. |
| `openai_compatible` | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and optionally `OPENAI_MODEL`. Compatible endpoints may use a placeholder non-empty key if they do not authenticate. |

`OPENALEX_EMAIL` must be a real email address; example placeholders are rejected during configuration validation. `LLM_FALLBACK_PROVIDER` and model-routing settings are optional. Provider calls can incur cost and consume quota.

## Sources and datasets

Source toggles and optional credentials include `ARXIV_ENABLED`, `OPENALEX_ENABLED`, `SEMANTIC_SCHOLAR_ENABLED`, `SEMANTIC_SCHOLAR_API_KEY`, `OPENREVIEW_ENABLED`, `OPENREVIEW_USERNAME`, `OPENREVIEW_PASSWORD`, and `HUGGINGFACE_TOKEN`. Review `env_example.txt` for dataset download limits and additional source controls.

Some paper types require explicit inputs or evidence packs. Do not assume that changing a source toggle supplies a required dataset or makes an infeasible study executable.

## Execution and workflow

| Setting | Purpose |
|---|---|
| `SANDBOX_BACKEND` | Selects `docker` or `ast` execution dispatch. The AST backend is not an OS security boundary. |
| `ENGINEERING_BACKEND` | `auto` tries the opencode-backed arena and falls back to the legacy LLM path only if opencode is unavailable; `opencode` requires that backend; `legacy` uses the legacy path. Arena code runs only in Docker and fails closed if Docker is unavailable. |
| `PAPER_TYPE_DEFAULT` | Default paper type when one is not selected in the request. |
| `TYPED_GRAPH_ENABLED` | Enables recipe-aware production graph routing. |
| `EXPERIMENT_SEEDS` | Default experiment seed count. |
| `MAX_ITERATIONS` | Workflow iteration limit. |
| `RUN_MAX_LLM_CALLS`, `RUN_MAX_TOKENS`, `RUN_MAX_USD`, `RUN_WALL_CLOCK_SEC` | Run-level budget limits. |

The arena agent authors code but cannot execute shell commands. ScholarGraph runs every seed in the Docker sandbox, validates the exact seed list and contract hash, and writes trusted aggregate metrics. The arena does not fall back to AST execution, including during independent validation. This does not OS-isolate the local opencode process itself; use a trusted local account. The arena also has settings for its directory, session timeout, rounds, model, and retention. See the `ARENA_*` entries and comments in `env_example.txt`.

## Web and local storage

`WEB_HOST` and `WEB_PORT` configure the local Control Deck listener; defaults are `127.0.0.1` and `8765`. Path settings in `env_example.txt` control local databases, logs, memory, and generated output.

The UI can store runtime keys in `memory/keys.json`. Treat this file as secret and keep it out of version control.
