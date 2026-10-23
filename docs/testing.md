# Testing

The automated suite is under `tests/` and uses pytest. Tests are intended to run offline with providers and external services mocked where needed.

## Commands

Run the full suite:

```powershell
python -m pytest tests -q
```

Run a focused test module:

```powershell
python -m pytest tests/test_api_gateway.py -q
```

Run the standalone smoke check:

```powershell
python tests/smoke_offline.py
```

The smoke check is a quick sanity check, not a replacement for pytest. Live runs require provider credentials, network access, and may incur API costs.

## Coverage areas

The suite exercises workflow routing and paper types, agent behavior, source and gateway resilience, configuration, data and evidence contracts, sandbox dispatch, memory and persistence, verification, and the web dashboard. Tests that create durable run artifacts are isolated by `tests/conftest.py`.

Use pytest tests as the maintained test surface. The retired root-level scripts were ad hoc integration or compatibility checks and are not part of the current offline suite.
