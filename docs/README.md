# Documentation

This documentation describes the current application at a system level. For exact defaults and supported options, `core/config.py` and `env_example.txt` are authoritative.

## Start here

- [Architecture](architecture.md) — runtime components, ownership, data flow, and boundaries.
- [Workflow and paper types](workflow.md) — recipe selection, common gates, and recovery behavior.
- [Configuration](configuration.md) — providers, credentials, research sources, execution, and run settings.
- [Testing](testing.md) — offline validation commands and test-suite guidance.
- [Operations](operations.md) — launching the UI or CLI, outputs, logs, and run recovery.

## Documentation policy

Keep this guide focused on supported behavior and operational use. Do not add generated per-file summaries, historical upgrade plans, or run-specific postmortems here; those become stale quickly. Update the relevant guide when behavior changes, and verify details against the implementation.
