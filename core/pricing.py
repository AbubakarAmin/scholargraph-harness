"""Token/USD accounting (proposal §8.8.3 + §5.3).

Implementer MUST NOT hardcode prices at call sites. All pricing flows
through this module's PRICES table, loaded from config
LLM_PRICE_TABLE_JSON when present.

Prices are USD per 1M tokens: {model: (input_per_mtok, output_per_mtok)}.
Default table covers gemini-2.5-flash + gpt-4o-mini and is marked
estimates (vendors rotate pricing; override via LLM_PRICE_TABLE_JSON).
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Default price table (USD per 1M tokens), marked estimates per §8.8.3.
# Sources: vendor pricing pages at time of writing; override via env.
DEFAULT_PRICES: dict[str, dict[str, Any]] = {
    "gemini-2.5-flash": {"input_per_mtok": 0.30, "output_per_mtok": 2.50, "estimated": True},
    "gpt-4o-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.60, "estimated": True},
}

PRICES: dict[str, dict[str, Any]] = dict(DEFAULT_PRICES)

# Budgets (owner-generous defaults, 2026-10-04: time/cost/token caps are a
# backstop, not a target — mirrors core/config.py; keep in sync).
RUN_MAX_LLM_CALLS = 2000
RUN_MAX_TOKENS = 8_000_000
RUN_MAX_USD = 500.00
RUN_WALL_CLOCK_SEC = 21600

DEPTH_BUDGETS: dict[str, dict[str, float]] = {
    "Quick": {"calls": 750, "usd": 50.0, "wall_min": 225, "review_screen": 40, "review_fulltext": 8, "sources": 20},
    "Standard": {"calls": 2000, "usd": 125.0, "wall_min": 450, "review_screen": 120, "review_fulltext": 25, "sources": 40},
    "Thorough": {"calls": 4000, "usd": 300.0, "wall_min": 900, "review_screen": 200, "review_fulltext": 40, "sources": 60},
}


def load_price_table(raw_json: str | None) -> dict[str, dict[str, Any]]:
    """Load/override PRICES from LLM_PRICE_TABLE_JSON. Never raises."""
    global PRICES
    if not raw_json or not str(raw_json).strip():
        PRICES = dict(DEFAULT_PRICES)
        return PRICES
    try:
        data = json.loads(raw_json)
        if isinstance(data, dict) and data:
            merged = dict(DEFAULT_PRICES)
            for model, entry in data.items():
                if isinstance(entry, dict):
                    merged[str(model)] = {
                        "input_per_mtok": float(entry.get("input_per_mtok", 0.0)),
                        "output_per_mtok": float(entry.get("output_per_mtok", 0.0)),
                        "estimated": bool(entry.get("estimated", False)),
                    }
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    merged[str(model)] = {
                        "input_per_mtok": float(entry[0]),
                        "output_per_mtok": float(entry[1]),
                        "estimated": True,
                    }
            PRICES = merged
    except Exception as e:
        logger.warning("Invalid LLM_PRICE_TABLE_JSON, using defaults: %s", e)
        PRICES = dict(DEFAULT_PRICES)
    return PRICES


def tokens_of(text: str | None) -> tuple[int, bool]:
    """Return (token_count, estimated).

    Uses provider tokenizer when available, else ceil(len/4) flagged
    estimated=True per §8.8.3.
    """
    if not text:
        return (0, True)
    try:
        # Best-effort: tiktoken when installed (OpenAI-compatible counting).
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return (len(enc.encode(str(text))), False)
    except Exception:
        pass
    return (int(math.ceil(len(str(text)) / 4)), True)


def usd_for(model: str | None, tokens_in: int, tokens_out: int) -> tuple[float, bool]:
    """Return (usd, estimated). Unknown models fall back to default entry."""
    entry = PRICES.get(str(model or "")) if model else None
    if entry is None:
        # Fall back to first default entry, flagged estimated.
        entry = next(iter(DEFAULT_PRICES.values()))
        estimated = True
    else:
        estimated = bool(entry.get("estimated", False))
    usd = (float(tokens_in) / 1_000_000.0) * float(entry["input_per_mtok"]) + (
        float(tokens_out) / 1_000_000.0
    ) * float(entry["output_per_mtok"])
    return (usd, estimated)


def budget_exhausted(
    *,
    llm_calls: int,
    tokens_total: int,
    usd_spent: float,
    wall_sec: float,
    max_calls: int = RUN_MAX_LLM_CALLS,
    max_tokens: int = RUN_MAX_TOKENS,
    max_usd: float = RUN_MAX_USD,
    max_wall_sec: float = RUN_WALL_CLOCK_SEC,
) -> tuple[bool, str | None]:
    """Trip check per §5.3. Returns (exhausted, reason_literal_or_None)."""
    if llm_calls >= max_calls:
        return (True, "budget_exhausted")
    if tokens_total >= max_tokens:
        return (True, "budget_exhausted")
    if usd_spent >= max_usd:
        return (True, "budget_exhausted")
    if wall_sec >= max_wall_sec:
        return (True, "budget_exhausted")
    return (False, None)
