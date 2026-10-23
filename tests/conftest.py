"""Pytest isolation for durable run artifacts.

Without this, any test that constructs a RunTracker, CrossRunMemory, or the
shared ResearchDatabase singleton writes into the production logs
(output/run_events.jsonl, output/run_scratchpad.jsonl, memory/cross_run.jsonl,
memory/research_ledger.sqlite) — polluting forensics, cross-run lessons, and
the research ledger with synthetic test runs. (The global logs/scholargraph.log
stream still captures test log records; that is a separate, read-only concern
for forensics, not writable state.)

This autouse fixture redirects every writable artifact path into the test's
tmp dir. It uses monkeypatch so all paths are restored after each test.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_durable_artifacts(tmp_path, monkeypatch):
    from core.config import config
    from core.research_db import research_db

    monkeypatch.setattr(config, "run_events_path", str(tmp_path / "run_events.jsonl"))
    monkeypatch.setattr(config, "run_log_path", str(tmp_path / "run_scratchpad.jsonl"))
    monkeypatch.setattr(config, "cross_run_memory_path", str(tmp_path / "cross_run.jsonl"))
    monkeypatch.setattr(config, "research_db_path", str(tmp_path / "research_ledger.sqlite"))
    monkeypatch.setattr(config, "elo_ratings_path", str(tmp_path / "elo_ratings.json"))
    # ResearchMemory writes these eagerly (add_debate_entry writes on every
    # call); without redirection a test can truncate the production logs.
    monkeypatch.setattr(config, "debate_log_path", str(tmp_path / "debate_log.json"))
    monkeypatch.setattr(config, "feedback_log_path", str(tmp_path / "feedback_log.json"))
    # The ledger singleton caches its path at import; repoint the live object
    # and create its schema (ResearchDatabase.__init__ normally migrates).
    monkeypatch.setattr(research_db, "path", tmp_path / "research_ledger.sqlite")
    research_db._migrate()
    yield
