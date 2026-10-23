"""
Centralized logging configuration with per-run log files.

Each run gets its own log file under logs/<run_id>.log so errors
can be traced back to the exact run that produced them.
"""

import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Optional

_LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"
_INITIALIZED = False
_root_logger: Optional[logging.Logger] = None


def _utc_formatter() -> logging.Formatter:
    # run_events.jsonl already uses UTC ISO-8601 with "Z". File/console logs
    # used local time, forcing mental arithmetic during forensics. Emit UTC
    # everywhere with an explicit "Z" marker.
    logging.Formatter.converter = time.gmtime
    return logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%SZ",
    )


def _ensure_logs_dir():
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(log_level: str = "INFO"):
    """Configure root logger with console + rotating file handler.

    Called once at startup. Subsequent calls are no-ops.
    """
    global _INITIALIZED, _root_logger
    if _INITIALIZED:
        return

    _ensure_logs_dir()
    _root_logger = logging.getLogger()
    _root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates from basicConfig
    _root_logger.handlers.clear()

    # Quiet the HTTP transport loggers that emit one INFO line PER request.
    # google-genai's httpx (Gemini batchEmbedContents) fans out ~20 requests per
    # seed -> ~100+ identical "HTTP Request: POST ...batchEmbedContents" lines
    # pollute every run log (run fd6f4910). The app's own debug logging is
    # unaffected; WARNING+ still surfaces real transport errors.
    for _quiet in ("httpx", "httpcore"):
        logging.getLogger(_quiet).setLevel(logging.WARNING)

    formatter = _utc_formatter()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console.setFormatter(formatter)
    _root_logger.addHandler(console)

    # Global rotating file handler (captures everything across all runs)
    _ensure_logs_dir()
    global_handler = logging.handlers.RotatingFileHandler(
        _LOGS_DIR / "scholargraph.log",
        maxBytes=10_000_000,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    global_handler.setLevel(logging.DEBUG)
    global_handler.setFormatter(formatter)
    _root_logger.addHandler(global_handler)

    _INITIALIZED = True


def attach_run_log(run_id: str) -> logging.Logger:
    """Attach a per-run file handler to the root logger.

    Returns the root logger for convenience. The handler is removed when
    ``detach_run_log`` is called (or the process exits).
    """
    global _root_logger
    if _root_logger is None:
        _root_logger = logging.getLogger()

    # e0ec9e91: --resume re-added a second FileHandler with the same name+file
    # so every line 13:59-14:25 appeared twice. Dedupe by handler name.
    _target = f"run_{run_id}"
    for h in list(_root_logger.handlers):
        if getattr(h, "name", None) == _target:
            return _root_logger
    _ensure_logs_dir()
    run_log_path = _LOGS_DIR / f"{run_id}.log"
    formatter = _utc_formatter()
    handler = logging.handlers.RotatingFileHandler(
        run_log_path,
        maxBytes=10_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    handler.set_name(f"run_{run_id}")
    _root_logger.addHandler(handler)
    return _root_logger


def detach_run_log(run_id: str):
    """Remove the per-run file handler for *run_id*."""
    global _root_logger
    if _root_logger is None:
        return
    target_name = f"run_{run_id}"
    _root_logger.handlers = [
        h for h in _root_logger.handlers
        if getattr(h, "name", None) != target_name
    ]
