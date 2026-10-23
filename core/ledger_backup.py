"""Ledger backup + rotation job (proposal §11-M6 step 3, §12 migration).

Creates timestamped copies ``<ledger>.bak.<ts>`` and keeps the newest N
(default 5). NEVER deletes the live ledger — only ``*.bak.*`` siblings
are pruned, and only the explicit Re-run path may delete checkpoints
(proposal §12 / U6).

Offline-safe: pure file copy, no LLM, no gateway.

Usage:
    python main.py --backup-ledger [--backup-keep 5]
    # or
    from core.ledger_backup import backup_ledger
    backup_ledger()  # uses config.research_db_path, sibling dir, keep=5
"""
from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_KEEP = 5

_TS_RE = re.compile(r"\.bak\.\d{8}T\d{6}Z$")


def _utc_ts(now: Optional[datetime] = None) -> str:
    dt = now or datetime.now(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def backup_file_path(db_path: Path, ts: Optional[str] = None) -> Path:
    """Sibling backup path for a ledger file: ``<name>.bak.<ts>``."""
    return db_path.with_name(f"{db_path.name}.bak.{ts or _utc_ts()}")


def _backup_siblings(db_path: Path, backup_dir: Path) -> List[Path]:
    """Main ``*.bak.<ts>`` files for this ledger stem, oldest-first.

    Re-audit (M6): WAL/shm sidecars (``<dest>-wal``) are attachments of their
    main backup, never siblings — they are filtered here via ``_TS_RE`` and
    pruned/rotated together with their main file.
    """
    stem = db_path.name
    candidates = [
        p for p in backup_dir.glob(f"{stem}.bak.*")
        if p.is_file() and _TS_RE.search(p.name)
    ]
    return sorted(candidates, key=lambda p: p.name)


def _sidecars_of(backup: Path) -> List[Path]:
    """Sidecar attachments written alongside a main backup file."""
    return [s for s in (backup.parent / f"{backup.name}-wal",
                        backup.parent / f"{backup.name}-shm") if s.is_file()]


def rotate_backups(db_path: Path | str, backup_dir: Path | str | None = None,
                   keep: int = DEFAULT_KEEP) -> List[Path]:
    """Delete oldest ``*.bak.*`` siblings beyond ``keep`` newest. Returns survivors."""
    db = Path(db_path)
    bdir = Path(backup_dir) if backup_dir else db.parent
    if not bdir.exists():
        return []
    siblings = _backup_siblings(db, bdir)
    keep = max(int(keep), 1)  # re-audit (M6): keep>=1 — a typo must never wipe all history
    for stale in siblings[: max(len(siblings) - keep, 0)]:
        # Safety: only delete our own timestamped backups, never the live file.
        if ".bak." not in stale.name or stale.resolve() == db.resolve():
            continue
        try:
            stale.unlink()
            for sidecar in _sidecars_of(stale):
                try:
                    sidecar.unlink()
                except OSError as e:
                    logger.warning("ledger backup rotation failed for %s: %s", sidecar, e)
            logger.info("ledger backup rotation removed %s", stale)
        except OSError as e:
            logger.warning("ledger backup rotation failed for %s: %s", stale, e)
    return _backup_siblings(db, bdir)


def backup_ledger(db_path: Path | str | None = None,
                  backup_dir: Path | str | None = None,
                  keep: int = DEFAULT_KEEP) -> Optional[Path]:
    """Copy the live ledger to ``<ledger>.bak.<ts>`` + rotate to newest ``keep``.

    Returns the created backup path, or None when no live ledger exists yet
    (first run — nothing to copy, not an error). Never touches the live file.
    """
    from .config import config as _config

    db = Path(db_path or _config.research_db_path)
    if not db.exists():
        logger.info("backup_ledger: no live ledger at %s, nothing to back up", db)
        return None
    bdir = Path(backup_dir) if backup_dir else db.parent
    bdir.mkdir(parents=True, exist_ok=True)
    # Re-audit (M6): best-effort WAL checkpoint before copying so the main
    # file holds the latest frames; without this a hot copy can split-brain
    # between the DB copy and the sidecar copies below. Never fatal.
    try:
        import sqlite3 as _sqlite3

        with _sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5) as con:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    except Exception as e:
        logger.warning("ledger backup checkpoint skipped for %s: %s", db, e)
    dest = backup_file_path(db)
    if dest.exists():
        # Re-audit (M6): second-resolution stamps collide within one second —
        # bump with a counter suffix instead of silently overwriting.
        counter = 1
        while True:
            candidate = db.with_name(f"{db.name}.bak.{_utc_ts()}.{counter}")
            counter += 1
            if not candidate.exists():
                dest = candidate
                break
    # SQLite WAL mode: copy sidecars too when present so the backup is consistent.
    shutil.copy2(db, dest)
    for sidecar in (f"{db.name}-wal", f"{db.name}-shm"):
        src = db.parent / sidecar
        if src.exists():
            try:
                shutil.copy2(src, bdir / f"{dest.name}-{sidecar.split('-')[-1]}")
            except OSError as e:
                logger.warning("backup_ledger sidecar copy failed for %s: %s", src, e)
    logger.info("ledger backup created: %s", dest)
    rotate_backups(db, bdir, keep)
    return dest
