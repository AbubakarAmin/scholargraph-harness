# Operations

## Start the web interface

```powershell
python run_ui.py
```

The default URL is <http://127.0.0.1:8765>. Configure the listener with `WEB_HOST` and `WEB_PORT`. The launcher attempts to stop an existing process listening on the selected port before starting the UI. Restart the process after changing backend Python files; automatic reload is not enabled.

## Run from the command line

```powershell
python main.py
```

Resume a run from its durable checkpoint:

```powershell
python main.py --resume RUN_ID
```

Use `python main.py --help` to see the available CLI options.

## Execution backends

The default sandbox backend is `docker`; configure `SANDBOX_BACKEND=ast` to select the in-process AST backend explicitly. If Docker is unavailable, the dispatcher may fall back to AST according to the current configuration and logs. Neither application-level AST restrictions nor a local Docker setup should be treated as a substitute for a properly isolated environment for untrusted code.

`ENGINEERING_BACKEND=auto` tries the opencode-backed engineering arena and falls back to the legacy path when that backend is unavailable. Set `opencode` to require it or `legacy` to disable the arena. Arena task files and experiment outputs are stored below the configured arena directory.

## Local state and outputs

- `output/` contains generated papers, experiment results, and run artifacts.
- `memory/` contains local vector data, runtime keys, and reusable run memory.
- `logs/` contains application logs, including per-run logs.
- Durable checkpoints and the research ledger are controlled by settings in `env_example.txt`.

These directories may contain sensitive or valuable state. Back up data you need, keep credentials private, and do not commit generated runtime state.
