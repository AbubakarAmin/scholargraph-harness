"""Pack readers/validators + schemas for field-scoped paper types (proposal §5.2, M3).

Exact schemas:
- **response_table:** ``.csv`` or ``.json``, ≤50 MB, UTF-8 (``errors="replace"``
  fallback), header required, row cap 200k (truncate + ``truncated: true``),
  target declared in codebook.
- **codebook.csv:** columns ``item, target, scale, subgroup_preregistered``.
- **case_pack.zip:** ≤200 MB → manifest entries ``{path, bytes, sha256, type}``
  where ``type ∈ {doc, transcript, table, audio_note}``; ≥2 distinct types
  required, else ``needs_pack_invalid``.
- **corpus_override.json:** ``{dois: []}`` with ≤50 entries (unresolvable DOIs
  quarantined, logged) — enforced at read time here; the systematic-review
  node keeps its own truncation guard as the production path.

All validators return named reasons (never tracebacks) so intake and the
``POST /api/packs`` endpoint can surface ``needs_*`` refusals directly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Exact caps (§5.2) -------------------------------------------------------
RESPONSE_TABLE_MAX_BYTES = 50 * 1024 * 1024
CASE_PACK_MAX_BYTES = 200 * 1024 * 1024
RESPONSE_ROW_CAP = 200_000
CORPUS_OVERRIDE_CAP = 50

RESPONSE_TABLE_SUFFIXES = (".csv", ".json", ".jsonl")
CODEBOOK_COLUMNS = ("item", "target", "scale", "subgroup_preregistered")
CASE_PACK_TYPES = ("doc", "transcript", "table", "audio_note")

# Packs uploaded via POST /api/packs land here (gitignored output/ tree).
PACKS_DIRNAME = "packs"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def packs_dir(output_dir: str | Path = "./output") -> Path:
    root = Path(output_dir) / PACKS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


# --- response_table ----------------------------------------------------------

def validate_response_table_file(
    path: str | Path,
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a response_table file on disk (pure check, named reasons).

    Returns ``{ok, reasons, manifest, truncated, encoding, rows}``. ``ok``
    is False only for hard refusals (missing file, bad suffix, oversize,
    missing header, empty); encoding fallback and row-cap truncation succeed
    with flags (D17) so latin-1 / jsonl / oversize paths stay usable.
    """
    reasons: List[str] = []
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "reasons": ["needs_response_table: file not found"],
                "manifest": None, "truncated": False, "encoding": None, "rows": 0}
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        suffix_ok = True
    else:
        suffix_ok = suffix in (".csv", ".json")
    if not suffix_ok:
        return {"ok": False, "reasons": [f"needs_pack_invalid: response_table suffix {p.suffix!r} not in (.csv, .json, .jsonl)"],
                "manifest": None, "truncated": False, "encoding": None, "rows": 0}
    try:
        size = p.stat().st_size
    except OSError as exc:
        return {"ok": False, "reasons": [f"needs_pack_invalid: cannot stat response_table ({exc})"],
                "manifest": None, "truncated": False, "encoding": None, "rows": 0}
    if size > RESPONSE_TABLE_MAX_BYTES:
        return {"ok": False, "reasons": [f"needs_pack_invalid: response_table {size} bytes exceeds 50 MB cap"],
                "manifest": None, "truncated": False, "encoding": None, "rows": 0}
    # Read with D17 semantics (encoding fallback + lines=True retry + row cap).
    try:
        frame_info = read_response_table(path)
    except ValueError as exc:
        return {"ok": False, "reasons": [f"needs_pack_invalid: {exc}"],
                "manifest": None, "truncated": False, "encoding": None, "rows": 0}
    rows = int(frame_info["rows"])
    if rows == 0:
        reasons.append("needs_pack_invalid: response_table has no rows")
        return {"ok": False, "reasons": reasons, "manifest": None,
                "truncated": bool(frame_info.get("truncated")), "encoding": frame_info.get("encoding"), "rows": 0}
    columns = list(frame_info.get("columns") or [])
    if not columns:
        reasons.append("needs_pack_invalid: response_table header required (no columns)")
        return {"ok": False, "reasons": reasons, "manifest": None,
                "truncated": bool(frame_info.get("truncated")), "encoding": frame_info.get("encoding"), "rows": rows}
    if target and target not in columns:
        reasons.append(f"needs_pack_invalid: target {target!r} not in response_table columns")
        return {"ok": False, "reasons": reasons, "manifest": None,
                "truncated": bool(frame_info.get("truncated")), "encoding": frame_info.get("encoding"), "rows": rows}
    manifest = {
        "name": p.name,
        "bytes": size,
        "sha256": sha256_file(p),
        "rows": rows,
        "columns": columns,
        "truncated": bool(frame_info.get("truncated")),
        "encoding": frame_info.get("encoding"),
    }
    return {"ok": True, "reasons": reasons, "manifest": manifest,
            "truncated": bool(frame_info.get("truncated")),
            "encoding": frame_info.get("encoding"), "rows": rows}


def read_response_table(path: str | Path) -> Dict[str, Any]:
    """Read a response table with D17 resilience (never raises on encoding/jsonl/cap).

    - UTF-8 strict first; on ``UnicodeDecodeError`` retry latin-1, else
      ``errors="replace"`` (flagged in ``encoding``).
    - ``.json`` tries plain JSON first, then ``lines=True`` (jsonl) retry.
    - Row cap 200k: truncate + ``truncated: true`` (flag, not failure).

    Returns ``{rows, columns, truncated, encoding, frame}`` where ``frame``
    is a pandas DataFrame. Raises ``ValueError`` only for missing header /
    unsupported suffix (named invalid-pack reasons upstream).
    """
    import pandas as pd

    p = Path(path)
    suffix = p.suffix.lower()
    encoding_used = "utf-8"
    # Detect encoding: try strict utf-8, fall back to latin-1 (flagged).
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read response_table ({exc})")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        encoding_used = "latin-1"
    truncated = False

    def _cap(frame: "pd.DataFrame") -> "pd.DataFrame":
        nonlocal truncated
        if len(frame) > RESPONSE_ROW_CAP:
            truncated = True
            return frame.iloc[:RESPONSE_ROW_CAP].reset_index(drop=True)
        return frame

    if suffix == ".csv":
        try:
            if encoding_used == "utf-8":
                frame = pd.read_csv(p, encoding="utf-8")
            else:
                frame = pd.read_csv(p, encoding="latin-1")
        except UnicodeDecodeError:
            frame = pd.read_csv(p, encoding="utf-8", encoding_errors="replace")
            encoding_used = "utf-8-replace"
        if list(frame.columns) and all(str(c).startswith("Unnamed") for c in frame.columns):
            raise ValueError("response_table header required (no columns)")
        frame = _cap(frame)
        return {"rows": int(len(frame)), "columns": list(frame.columns),
                "truncated": truncated, "encoding": encoding_used, "frame": frame}
    if suffix in (".json", ".jsonl"):
        # Plain JSON first (array of records or {rows: [...]}), then jsonl.
        frame = None
        try:
            if encoding_used == "utf-8":
                frame = pd.read_json(p, encoding="utf-8")
            else:
                frame = pd.read_json(p, encoding="latin-1")
            # pd.read_json on a jsonl file without lines=True yields a
            # degenerate single-column frame — treat as needing the retry.
            if frame is not None and len(frame.columns) == 1 and suffix == ".jsonl":
                raise ValueError("jsonl needs lines=True")
        except ValueError as exc:
            msg = str(exc)
            if "lines=True" in msg or suffix == ".jsonl" or "Expected object or value" in msg or "Trailing data" in msg:
                try:
                    if encoding_used == "utf-8":
                        frame = pd.read_json(p, lines=True, encoding="utf-8")
                    else:
                        frame = pd.read_json(p, lines=True, encoding="latin-1")
                    encoding_used = encoding_used + "+lines=True" if "+lines" not in encoding_used else encoding_used
                    if encoding_used == "utf-8":
                        encoding_used = "utf-8+lines=True"
                except Exception as exc2:
                    raise ValueError(f"response_table json parse failed ({exc2})")
            else:
                raise ValueError(f"response_table json parse failed ({exc})")
        except UnicodeDecodeError:
            try:
                frame = pd.read_json(p, encoding="utf-8", encoding_errors="replace")
                encoding_used = "utf-8-replace"
            except Exception:
                frame = pd.read_json(p, lines=True)
                encoding_used = "utf-8+lines=True"
        if frame is None:
            raise ValueError("response_table json parse failed (empty)")
        if list(frame.columns) and all(str(c).startswith("Unnamed") for c in frame.columns):
            raise ValueError("response_table header required (no columns)")
        frame = _cap(frame)
        return {"rows": int(len(frame)), "columns": list(frame.columns),
                "truncated": truncated, "encoding": encoding_used, "frame": frame}
    raise ValueError(f"response_table suffix {p.suffix!r} not in (.csv, .json, .jsonl)")


# --- codebook -----------------------------------------------------------------

def validate_codebook_file(path: str | Path) -> Dict[str, Any]:
    """Validate codebook.csv (exact columns ``item, target, scale, subgroup_preregistered``)."""
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "reasons": ["needs_pack_invalid: codebook.csv not found"],
                "manifest": None, "items": [], "targets": [], "preregistered_subgroups": []}
    try:
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8")
            encoding_used = "utf-8"
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
            encoding_used = "latin-1"
        import io as _io
        reader = csv.DictReader(_io.StringIO(text))
        columns = list(reader.fieldnames or [])
        missing = [c for c in CODEBOOK_COLUMNS if c not in columns]
        if missing:
            return {"ok": False, "reasons": [f"needs_pack_invalid: codebook missing columns {missing}"],
                    "manifest": None, "items": [], "targets": [], "preregistered_subgroups": []}
        items: List[Dict[str, Any]] = []
        targets: List[str] = []
        prereg: List[str] = []
        for row in reader:
            items.append({k: (row.get(k) or "").strip() for k in CODEBOOK_COLUMNS})
            tgt = (row.get("target") or "").strip()
            if tgt:
                targets.append(tgt)
            item_name = (row.get("item") or "").strip()
            flag = (row.get("subgroup_preregistered") or "").strip().lower()
            if item_name and flag in ("1", "true", "yes", "y", "preregistered"):
                prereg.append(item_name)
        manifest = {"name": p.name, "bytes": len(raw), "sha256": sha256_bytes(raw),
                    "rows": len(items), "encoding": encoding_used}
        return {"ok": True, "reasons": [], "manifest": manifest, "items": items,
                "targets": sorted(set(targets)), "preregistered_subgroups": sorted(set(prereg))}
    except Exception as exc:
        return {"ok": False, "reasons": [f"needs_pack_invalid: codebook parse failed ({exc})"],
                "manifest": None, "items": [], "targets": [], "preregistered_subgroups": []}


# --- case_pack ------------------------------------------------------------------

def _infer_case_file_type(name: str) -> str:
    lowered = name.lower()
    base = lowered.rsplit("/", 1)[-1]
    if any(k in lowered for k in ("transcript", ".vtt", ".srt")):
        return "transcript"
    if (any(k in lowered for k in ("audio", "interview_recording", ".mp3", ".wav", ".m4a", ".ogg", ".flac")) and "audio_note" in lowered) or lowered.endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac")):
        return "audio_note"
    if "audio_note" in lowered or "field_note" in lowered:
        return "audio_note"
    if base.endswith((".csv", ".json", ".jsonl", ".xlsx", ".xls", ".tsv", ".parquet")) or "/table" in lowered or "table" in base:
        return "table"
    return "doc"


def validate_case_pack_file(path: str | Path) -> Dict[str, Any]:
    """Validate case_pack.zip (≤200 MB, ≥2 distinct source types).

    Returns ``{ok, reasons, manifest, entries, distinct_types}`` where each
    entry is ``{path, bytes, sha256, type}`` per §5.2. ``ok`` False carries a
    named ``needs_case_pack`` / ``needs_pack_invalid`` reason (never traceback).
    """
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "reasons": ["needs_case_pack: case_pack.zip not found"],
                "manifest": None, "entries": [], "distinct_types": []}
    if p.suffix.lower() != ".zip":
        return {"ok": False, "reasons": [f"needs_pack_invalid: case_pack suffix {p.suffix!r} must be .zip"],
                "manifest": None, "entries": [], "distinct_types": []}
    try:
        size = p.stat().st_size
    except OSError as exc:
        return {"ok": False, "reasons": [f"needs_pack_invalid: cannot stat case_pack ({exc})"],
                "manifest": None, "entries": [], "distinct_types": []}
    if size > CASE_PACK_MAX_BYTES:
        return {"ok": False, "reasons": [f"needs_pack_invalid: case_pack {size} bytes exceeds 200 MB cap"],
                "manifest": None, "entries": [], "distinct_types": []}
    try:
        archive = zipfile.ZipFile(p, "r")
    except zipfile.BadZipFile as exc:
        return {"ok": False, "reasons": [f"needs_pack_invalid: case_pack is not a valid zip ({exc})"],
                "manifest": None, "entries": [], "distinct_types": []}
    entries: List[Dict[str, Any]] = []
    skipped: List[str] = []
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
                skipped.append(name)
                continue
            try:
                data = archive.read(name)
            except Exception as exc:
                skipped.append(f"{name} (unreadable: {exc})")
                continue
            entries.append({"path": name, "bytes": len(data),
                            "sha256": sha256_bytes(data),
                            "type": _infer_case_file_type(name)})
    if not entries:
        return {"ok": False, "reasons": ["needs_pack_invalid: case_pack contains no readable entries"],
                "manifest": None, "entries": [], "distinct_types": []}
    distinct = sorted({e["type"] for e in entries})
    if len(distinct) < 2:
        return {"ok": False,
                "reasons": [f"needs_pack_invalid: case pack has {len(distinct)} source type(s), need >=2 (found: {distinct})"],
                "manifest": {"name": p.name, "bytes": size, "sha256": sha256_file(p),
                             "entries": len(entries), "skipped": skipped},
                "entries": entries, "distinct_types": distinct}
    manifest = {"name": p.name, "bytes": size, "sha256": sha256_file(p),
                "entries": len(entries), "distinct_types": distinct, "skipped": skipped}
    return {"ok": True, "reasons": [], "manifest": manifest,
            "entries": entries, "distinct_types": distinct}


# --- corpus_override --------------------------------------------------------------

def validate_corpus_override(data: Any) -> Dict[str, Any]:
    """Validate corpus_override.json (``{dois: []}``, ≤50 entries).

    Over-cap DOIs truncate (flagged); unresolvable entries are quarantined
    downstream by the review node — here they are kept with a logged note.
    """
    dois: List[str] = []
    if isinstance(data, dict):
        raw = data.get("dois", data.get("DOIs", []))
        dois = [str(d).strip() for d in (raw or []) if str(d or "").strip()]
    elif isinstance(data, list):
        dois = [str(d).strip() for d in data if str(d or "").strip()]
    else:
        return {"ok": False, "reasons": ["needs_pack_invalid: corpus_override must be {dois: [...]}"],
                "dois": [], "truncated": False}
    truncated = False
    if len(dois) > CORPUS_OVERRIDE_CAP:
        truncated = True
        dois = dois[:CORPUS_OVERRIDE_CAP]
    return {"ok": True, "reasons": [], "dois": dois, "truncated": truncated}


def validate_corpus_override_file(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "reasons": ["needs_pack_invalid: corpus_override.json not found"],
                "dois": [], "truncated": False}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reasons": [f"needs_pack_invalid: corpus_override parse failed ({exc})"],
                "dois": [], "truncated": False}
    return validate_corpus_override(data)


# --- pack manifest + store ----------------------------------------------------------

def codebook_match(response_columns: List[str], codebook_targets: List[str]) -> bool:
    """True when a codebook declares at least one response-table column as target."""
    cols = {str(c or "").strip() for c in (response_columns or []) if str(c or "").strip()}
    tgts = {str(t or "").strip() for t in (codebook_targets or []) if str(t or "").strip()}
    return bool(cols & tgts)


def build_pack_manifest(
    response_table: Optional[Dict[str, Any]] = None,
    codebook: Optional[Dict[str, Any]] = None,
    case_pack: Optional[Dict[str, Any]] = None,
    corpus_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the intake ``pack_manifest`` (§8.4): files[] + codebook_match + pack keys.

    ``files`` entries are ``{name, bytes, sha256, rows|entries}``; the
    response_table / case_pack / codebook / corpus_override sub-dicts are
    kept alongside so ``validate_pack`` sees the shapes it expects.
    """
    files: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {}
    if response_table and response_table.get("manifest"):
        manifest = dict(response_table["manifest"])
        files.append({k: manifest.get(k) for k in ("name", "bytes", "sha256", "rows") if manifest.get(k) is not None})
        out["response_table"] = {"manifest": manifest, "rows": response_table.get("rows"),
                                 "truncated": response_table.get("truncated", False),
                                 "encoding": response_table.get("encoding")}
    if codebook and codebook.get("manifest"):
        manifest = dict(codebook["manifest"])
        files.append({k: manifest.get(k) for k in ("name", "bytes", "sha256", "rows") if manifest.get(k) is not None})
        out["codebook"] = {"manifest": manifest, "items": codebook.get("items", []),
                           "targets": codebook.get("targets", []),
                           "preregistered_subgroups": codebook.get("preregistered_subgroups", [])}
    if case_pack and (case_pack.get("manifest") or case_pack.get("entries")):
        manifest = dict(case_pack.get("manifest") or {})
        files.append({"name": manifest.get("name"), "bytes": manifest.get("bytes"),
                      "sha256": manifest.get("sha256"), "entries": manifest.get("entries")})
        out["case_pack"] = {"manifest": manifest, "entries": case_pack.get("entries", []),
                            "distinct_types": case_pack.get("distinct_types", [])}
    if corpus_override and corpus_override.get("dois") is not None:
        out["corpus_override"] = {"dois": list(corpus_override.get("dois") or []),
                                  "truncated": bool(corpus_override.get("truncated", False))}
    match = False
    try:
        resp_cols = list(((response_table or {}).get("manifest") or {}).get("columns") or [])
        tgts = list(codebook.get("targets") or []) if isinstance(codebook, dict) else []
        match = codebook_match(resp_cols, tgts)
    except Exception:
        match = False
    out["files"] = files
    out["codebook_match"] = match
    return out


def save_pack(pack_manifest: Dict[str, Any], output_dir: str | Path = "./output",
              pack_id: Optional[str] = None) -> Tuple[str, Path]:
    """Persist a pack manifest under ``output/packs/<pack_id>.json`` (gitignored)."""
    import uuid as _uuid

    root = packs_dir(output_dir)
    pid = str(pack_id or _uuid.uuid4().hex[:8])
    path = root / f"{pid}.json"
    payload = {"pack_id": pid, "manifest": pack_manifest}
    path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    return pid, path


def load_pack(pack_id: str, output_dir: str | Path = "./output") -> Optional[Dict[str, Any]]:
    """Load a stored pack manifest by id (None when unknown)."""
    path = Path(output_dir) / PACKS_DIRNAME / f"{pack_id}.json"
    if not path.is_file():
        # Fall back to a pack_id that is already a manifest dict carrier.
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Pack %s unreadable: %s", pack_id, exc)
        return None
    manifest = payload.get("manifest") if isinstance(payload, dict) else None
    return manifest if isinstance(manifest, dict) else None


def resolve_pack_reference(pack: Dict[str, Any], output_dir: str | Path = "./output") -> Dict[str, Any]:
    """Resolve ``{"pack_id": ...}`` carriers into full pack dicts (additive).

    Legacy callers already pass ``response_table`` / ``case_pack`` inline —
    those pass through untouched. Web intake passes a bare pack_id; the
    stored manifest is merged in so ``validate_pack`` sees pack keys.
    """
    pack = dict(pack or {})
    pack_id = pack.get("pack_id")
    if pack_id and not any(k in pack for k in ("response_table", "case_pack", "codebook", "corpus_override")):
        stored = load_pack(str(pack_id), output_dir)
        if isinstance(stored, dict):
            merged = dict(stored)
            merged["pack_id"] = pack_id
            return merged
    return pack
