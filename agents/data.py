"""Dataset stewardship agent: inspect and validate data without running experiments."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from core.capabilities import DEFAULT_MANIFESTS
from core.contracts import DatasetArtifact, DatasetSpec
from core.context import RunContext, get_active_context

logger = logging.getLogger(__name__)


class DataAgent:
    """Create immutable dataset metadata for downstream experiment workers."""

    capability_manifest = DEFAULT_MANIFESTS["DataAgent"]

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()

    def validate_dataset(self, path: str, spec: DatasetSpec) -> DatasetArtifact:
        dataset_path = Path(path).resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

        info = self._read_frame_info(dataset_path)
        frame = info["frame"]
        warnings = []
        target = spec.get("target")
        if target and target not in frame.columns:
            warnings.append(f"target column missing: {target}")
        feature_names = set(spec.get("features") or [])
        overlap = sorted(feature_names & {target} if target else set())
        if overlap:
            warnings.append(f"target appears in features: {overlap[0]}")
        if frame.empty:
            warnings.append("dataset has no rows")
        if frame.isna().all(axis=None):
            warnings.append("dataset contains only missing values")

        digest = self._sha256(dataset_path)
        now = datetime.now(timezone.utc).isoformat()
        passed = not warnings
        # D17: encoding fallback / lines=True retry / row-cap truncation are
        # advisory flags (needs_review), never hard failures on their own.
        needs_review = bool(info.get("needs_review"))
        skipped_files: List[str] = list(info.get("skipped_files") or [])
        limitations = list(warnings)
        if needs_review:
            limitations = limitations + [f"needs_review: encoding={info.get('encoding')} truncated={info.get('truncated')}"]
        return {
            "location": str(dataset_path),
            "content_hash": digest,
            "row_count": int(len(frame)),
            "schema": {
                "columns": list(frame.columns),
                "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
            },
            "spec": spec,
            "validation": {
                "passed": passed,
                "score": 10.0 if passed else 0.0,
                "note": "Dataset validation completed",
                "mismatches": warnings,
                "checks": {
                    "exists": True,
                    "non_empty": not frame.empty,
                    "target_present": not target or target in frame.columns,
                    "target_not_in_features": not overlap,
                },
                "truncated": bool(info.get("truncated")),
                "encoding": info.get("encoding"),
                "skipped_files": skipped_files,
                "needs_review": needs_review,
            },
            "provenance": {
                "artifact_type": "dataset",
                "source": str(dataset_path),
                "content_hash": digest,
                "created_at": now,
                "status": "validated" if (passed and not needs_review) else "needs_review",
                "limitations": limitations,
            },
        }

    def validate_response_table(
        self,
        path: str,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate a survey response_table (§4.8 step 1, M3).

        Checks: target present, target not in features, non-empty, encoding
        fallback (via ``_read_frame_info`` — never a hard failure). Returns a
        named-reason dict (never a traceback): ``{ok, reasons, manifest,
        truncated, encoding, rows, skipped_files}``. Missing/empty/unknown
        shapes map to ``needs_response_table`` / ``needs_pack_invalid``.
        """
        dataset_path = Path(path)
        if not dataset_path.is_file():
            return {"ok": False, "reasons": ["needs_response_table: file not found"],
                    "manifest": None, "truncated": False, "encoding": None,
                    "rows": 0, "skipped_files": []}
        if dataset_path.suffix.lower() not in (".csv", ".json", ".jsonl"):
            return {"ok": False,
                    "reasons": [f"needs_pack_invalid: response_table suffix {dataset_path.suffix!r} not in (.csv, .json, .jsonl)"],
                    "manifest": None, "truncated": False, "encoding": None,
                    "rows": 0, "skipped_files": []}
        try:
            info = self._read_frame_info(dataset_path)
        except ValueError as exc:
            return {"ok": False, "reasons": [f"needs_pack_invalid: {exc}"],
                    "manifest": None, "truncated": False, "encoding": None,
                    "rows": 0, "skipped_files": []}
        frame = info["frame"]
        reasons: List[str] = []
        if frame.empty:
            reasons.append("needs_pack_invalid: response_table has no rows")
        if target and target not in frame.columns:
            reasons.append(f"needs_pack_invalid: target {target!r} not in response_table columns")
        overlap = sorted(set(features or []) & ({target} if target else set()))
        if overlap:
            reasons.append(f"needs_pack_invalid: target appears in features: {overlap[0]}")
        if reasons:
            return {"ok": False, "reasons": reasons, "manifest": None,
                    "truncated": bool(info.get("truncated")), "encoding": info.get("encoding"),
                    "rows": int(len(frame)), "skipped_files": list(info.get("skipped_files") or [])}
        try:
            size = dataset_path.stat().st_size
        except OSError:
            size = 0
        manifest = {"name": dataset_path.name, "bytes": size,
                    "sha256": self._sha256(dataset_path), "rows": int(len(frame)),
                    "columns": list(frame.columns),
                    "truncated": bool(info.get("truncated")), "encoding": info.get("encoding")}
        return {"ok": True, "reasons": [], "manifest": manifest,
                "truncated": bool(info.get("truncated")), "encoding": info.get("encoding"),
                "rows": int(len(frame)), "skipped_files": list(info.get("skipped_files") or [])}

    @staticmethod
    def _read_frame_info(path: Path) -> Dict[str, Any]:
        """Read a tabular file with D17 resilience (M3, proposal §6.4).

        - UTF-8 strict first; on ``UnicodeDecodeError`` retry latin-1, else
          ``errors="replace"`` (flagged in ``encoding``, never a failure).
        - ``.json`` tries plain JSON first, then ``lines=True`` (jsonl) retry.
        - Row cap 200k (``RESPONSE_ROW_CAP``): truncate + ``truncated: true``
          with the skipped row count surfaced in ``skipped_files``.
        """
        from core.packs import RESPONSE_ROW_CAP

        suffix = path.suffix.lower()
        if suffix not in (".csv", ".json", ".jsonl"):
            raise ValueError("Supported dataset formats are .csv and .json")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Dataset cannot be read: {exc}")
        encoding_used = "utf-8"
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            encoding_used = "latin-1"
        needs_review = encoding_used != "utf-8"
        skipped_files: List[str] = []

        def _cap(frame: pd.DataFrame) -> pd.DataFrame:
            nonlocal needs_review
            if len(frame) > RESPONSE_ROW_CAP:
                skipped = int(len(frame) - RESPONSE_ROW_CAP)
                skipped_files.append(f"row-cap: {skipped} rows beyond {RESPONSE_ROW_CAP} skipped (truncated: true)")
                needs_review = True
                return frame.iloc[:RESPONSE_ROW_CAP].reset_index(drop=True)
            return frame

        if suffix == ".csv":
            try:
                if encoding_used == "utf-8":
                    frame = pd.read_csv(path, encoding="utf-8")
                else:
                    frame = pd.read_csv(path, encoding="latin-1")
            except UnicodeDecodeError:
                frame = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
                encoding_used = "utf-8-replace"
                needs_review = True
            frame = _cap(frame)
            return {"frame": frame, "encoding": encoding_used,
                    "truncated": len(skipped_files) > 0,
                    "skipped_files": skipped_files, "needs_review": needs_review}
        # JSON / JSONL: plain first, then lines=True retry (D17).
        frame = None
        try:
            if encoding_used == "utf-8":
                frame = pd.read_json(path, encoding="utf-8")
            else:
                frame = pd.read_json(path, encoding="latin-1")
            if frame is not None and len(frame.columns) == 1 and suffix == ".jsonl":
                raise ValueError("jsonl needs lines=True")
        except ValueError as exc:
            msg = str(exc)
            if "lines=True" in msg or suffix == ".jsonl" or "Expected object or value" in msg or "Trailing data" in msg:
                if encoding_used == "utf-8":
                    frame = pd.read_json(path, lines=True, encoding="utf-8")
                else:
                    frame = pd.read_json(path, lines=True, encoding="latin-1")
                encoding_used = "utf-8+lines=True" if encoding_used == "utf-8" else encoding_used + "+lines=True"
                needs_review = True
            else:
                raise
        except UnicodeDecodeError:
            frame = pd.read_json(path, encoding="utf-8", encoding_errors="replace")
            encoding_used = "utf-8-replace"
            needs_review = True
        if frame is None:
            raise ValueError("Dataset JSON parse failed (empty)")
        frame = _cap(frame)
        return {"frame": frame, "encoding": encoding_used,
                "truncated": len(skipped_files) > 0,
                "skipped_files": skipped_files, "needs_review": needs_review}

    @staticmethod
    def _read_frame(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix not in (".csv", ".json", ".jsonl"):
            raise ValueError("Supported dataset formats are .csv and .json")
        return DataAgent._read_frame_info(path)["frame"]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
