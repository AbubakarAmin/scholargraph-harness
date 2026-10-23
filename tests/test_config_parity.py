"""M0: config parity vs §9 truth (5 tests, source-level to avoid .env pollution)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SRC = (ROOT / "core" / "config.py").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / "env_example.txt").read_text(encoding="utf-8")

SECTION9_KEYS = [
    "LLM_PROVIDER", "GEMINI_MODEL", "GEMINI_EMBEDDING_MODEL", "EMBEDDING_DIMENSION",
    "SUPERVISOR_THRESHOLD", "DEBATE_PASS_THRESHOLD", "DEBATE_MIN_ROUNDS", "DEBATE_MAX_ROUNDS",
    "NOVELTY_SIMILARITY_REJECT", "EXPERIMENT_SEEDS", "EXPERIMENT_BRANCH_COUNT",
    "ENGINEER_MAX_ATTEMPTS", "SANDBOX_TIMEOUT_SEC", "SANDBOX_BACKEND",
    "HUGGINGFACE_MAX_DOWNLOAD_MB", "TYPED_GRAPH_ENABLED", "PAPER_TYPE_DEFAULT",
    "PARALLEL_PHASES_ENABLED", "RUN_MAX_LLM_CALLS", "RUN_MAX_TOKENS", "RUN_MAX_USD",
    "RUN_WALL_CLOCK_SEC", "UNPAYWALL_ENABLED", "PMC_ENABLED", "DOAJ_GUARD_ENABLED",
    "SEED_SEQUENTIAL", "PERSONA_SEQUENTIAL", "PERSONA_COUNT", "LLM_BUDGET_PER_SEED",
    "NOVELTY_MAX_ABSTRACTS", "LLM_SEED_GENERATION_ENABLED",
    "ALLOW_SYNTHETIC_RESPONDENTS", "LLM_TIMEOUT_SEC", "LLM_PRICE_TABLE_JSON", "MAX_ITERATIONS",
]


def test_every_section9_key_in_mapping_sync_example():
    for key in SECTION9_KEYS:
        assert f'"{key}"' in CONFIG_SRC, f"{key} missing from apply_runtime_keys/sync_env_file"
    for key in SECTION9_KEYS:
        assert key in ENV_EXAMPLE, f"{key} missing from env_example.txt"


def test_placeholder_email_rejected():
    from core import config as config_mod

    real_email = config_mod.config.openalex_email
    try:
        config_mod.config.openalex_email = "you@example.com"
        with pytest.raises(ValueError, match="placeholder"):
            config_mod.validate_config()
        config_mod.config.openalex_email = "researcher@example.com"
        with pytest.raises(ValueError, match="placeholder"):
            config_mod.validate_config()
    finally:
        config_mod.config.openalex_email = real_email


def test_embedding_dimension_locked_3072():
    # Owner decision (2026-10-03): gemini-embedding-2 native 3072 output.
    # Overrides proposal §0 lock 10 (768/text-embedding-004); the on-disk
    # FAISS index + memory are 3072-wide, and memory.py rebuilds (wiping
    # embeddings) on any dim mismatch — so the lock protects data.
    assert 'os.getenv("EMBEDDING_DIMENSION", "3072")' in CONFIG_SRC
    assert "EMBEDDING_DIMENSION=3072" in ENV_EXAMPLE
    assert 'os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")' in CONFIG_SRC
    assert "GEMINI_EMBEDDING_MODEL=gemini-embedding-2" in ENV_EXAMPLE


def test_budget_and_flag_defaults_match_section9():
    # OWNER DIRECTIVE (binding, overrides proposal §5.3/§9 locks): the owner
    # does not care about run time/cost/tokens; their LLM provider is slow
    # (300s+ per response). Caps are generous so a paper always finalises,
    # and TYPED_GRAPH_ENABLED defaults to true (full upgrade). NOTE to
    # owner: sync tests/test_m6_hardening.py::test_m6_config_parity_locked_defaults
    # to these same values (that file was intentionally left untouched here).
    for literal in [
        'os.getenv("RUN_MAX_LLM_CALLS", "2000")',
        'os.getenv("RUN_MAX_TOKENS", "8000000")',
        'os.getenv("RUN_MAX_USD", "500.0")',
        'os.getenv("RUN_WALL_CLOCK_SEC", "21600")',
        'os.getenv("ENGINEERING_WALL_CLOCK_SEC", "7200")',
        'os.getenv("SANDBOX_TIMEOUT_SEC", "600")',
        'os.getenv("LLM_TIMEOUT_SEC", "1800")',
        'os.getenv("TOPIC_SEED_TIMEOUT_SEC", "5400")',
        'os.getenv("TOPIC_DISCOVERY_TIMEOUT_SEC", "21600")',
        'os.getenv("ENGINEER_MAX_ATTEMPTS", "4")',
        'os.getenv("TYPED_GRAPH_ENABLED", "true")',
        'os.getenv("PAPER_TYPE_DEFAULT", "experimental")',
        'os.getenv("PARALLEL_PHASES_ENABLED", "false")',
        'os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")',
        'os.getenv("DEBATE_MAX_ROUNDS", "5")',
        'os.getenv("NOVELTY_SIMILARITY_REJECT", "0.92")',
        'os.getenv("HUGGINGFACE_MAX_DOWNLOAD_MB", "100")',
    ]:
        assert literal in CONFIG_SRC, literal


def test_sync_known_covers_runtime_mapping():
    """UI key saves must persist: every apply_runtime_keys key needs a sync_env_file known entry."""
    import ast

    tree = ast.parse(CONFIG_SRC)
    mapping_keys: set[str] = set()
    known_keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [c.value for c in node.keys if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            if "LLM_PROVIDER" in keys and "RESEARCH_DOMAIN" in keys:
                mapping_keys.update(keys)  # apply_runtime_keys mapping
        elif isinstance(node, ast.Set):
            keys = [c.value for c in node.elts if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            if "LLM_PROVIDER" in keys and "RESEARCH_DOMAIN" in keys:
                known_keys.update(keys)  # sync_env_file known set
    assert mapping_keys, "runtime mapping not found"
    assert known_keys, "sync known set not found"
    missing = sorted(mapping_keys - known_keys)
    assert not missing, f"sync_env_file known omits UI-savable keys: {missing}"
