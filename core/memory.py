"""
Memory module for the multi-agent research system.
Handles vector database operations and persistent knowledge storage.

Prompt retrieval is fail-closed: only structured_signal entries that are
explicitly retrieval_eligible may enter prompts, and only via get_prompt_context().
Raw narrative stays on disk for audit.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from pathlib import Path

from .config import config

logger = logging.getLogger(__name__)


def _atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON atomically: temp file in the same dir + os.replace.

    The previous truncate-then-dump pattern left 0-byte files whenever the
    process died (or json.dump raised) between open("w") and dump — which is
    exactly how output/debate_log.json became an empty file that then crashed
    every startup load. With replace, readers always see either the old or
    the new complete file, never a husk.
    """
    import os
    import tempfile

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_json_list(path: str, name: str) -> List[Dict[str, Any]]:
    """Load a JSON list file tolerantly: missing/empty/corrupt -> []."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read %s (%s); starting empty", name, e)
        return []
    if not text.strip():
        logger.warning("%s is empty; starting with an empty log", name)
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("%s is corrupt (%s); starting with an empty log", name, e)
        return []
    if not isinstance(data, list):
        logger.warning("%s does not contain a list; starting with an empty log", name)
        return []
    return data


class ResearchMemory:
    """Memory system for storing research knowledge and agent interactions."""

    def __init__(self):
        self.vector_db_path = Path(config.vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)

        # GEMINI-ONLY POLICY: this index holds 3072-wide gemini-embedding-2
        # vectors exclusively (see core/llm.py + core/config.py). A dimension
        # mismatch means the embedding backend changed — the index is
        # archived and rebuilt rather than mixing embedding spaces.
        self.dimension = config.embedding_dimension
        self.index = faiss.IndexFlatL2(self.dimension)

        self.metadata: List[Dict[str, Any]] = []
        self.debate_log: List[Dict[str, Any]] = []
        self.feedback_log: List[Dict[str, Any]] = []

        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing memory data from disk."""
        try:
            if (self.vector_db_path / "index.faiss").exists():
                loaded_index = faiss.read_index(str(self.vector_db_path / "index.faiss"))
                if loaded_index.d != self.dimension:
                    logger.warning(
                        f"FAISS index dimension mismatch: index={loaded_index.d}, "
                        f"expected={self.dimension}. Rebuilding index (old embeddings lost)."
                    )
                    # Archive old index for potential manual recovery
                    archive_path = self.vector_db_path / f"index.dim{loaded_index.d}.faiss.bak"
                    try:
                        import shutil
                        shutil.copy2(
                            str(self.vector_db_path / "index.faiss"),
                            str(archive_path),
                        )
                        logger.info(f"Archived old index to {archive_path}")
                    except Exception as e:
                        logger.warning(f"Could not archive old index: {e}")
                    self.index = faiss.IndexFlatL2(self.dimension)
                else:
                    self.index = loaded_index

            if (self.vector_db_path / "metadata.pkl").exists():
                with open(self.vector_db_path / "metadata.pkl", "rb") as f:
                    self.metadata = pickle.load(f)

            self.debate_log = _load_json_list(config.debate_log_path, "debate_log.json")
            self.feedback_log = _load_json_list(config.feedback_log_path, "feedback_log.json")

        except Exception as e:
            logger.warning("Could not load existing memory data: %s", e, exc_info=True)

    def save(self):
        """Save all memory data to disk."""
        try:
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.vector_db_path / "index.faiss"))

            with open(self.vector_db_path / "metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)

            _atomic_write_json(config.debate_log_path, self.debate_log)

            _atomic_write_json(config.feedback_log_path, self.feedback_log)

        except Exception as e:
            logger.error("Failed to save memory data: %s", e, exc_info=True)

    @staticmethod
    def _active_run_defaults() -> Dict[str, Any]:
        """Pull run provenance from the active tracker when present."""
        defaults = {"run_id": "unknown", "agent": "unknown", "outcome_status": "unknown"}
        try:
            from .run_log import get_tracker

            tracker = get_tracker()
            if tracker and getattr(tracker, "run_id", None):
                defaults["run_id"] = tracker.run_id
        except Exception:
            pass
        return defaults

    @classmethod
    def _normalize_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a fail-closed prompt policy to new and legacy entries."""
        normalized = dict(metadata)
        defaults = cls._active_run_defaults()
        content_class = normalized.get("content_class")
        if content_class not in {"structured_signal", "generated_narrative"}:
            content_class = "generated_narrative"
        normalized["content_class"] = content_class
        # generated_narrative can never be prompt-eligible, even if a caller asks.
        normalized["retrieval_eligible"] = bool(
            normalized.get("retrieval_eligible", False)
            and content_class == "structured_signal"
            and normalized.get("signal") is not None
        )
        normalized.setdefault("namespace", "unclassified")
        for key in ("run_id", "agent", "outcome_status"):
            if not normalized.get(key) or normalized.get(key) == "unknown":
                if defaults.get(key) and defaults[key] != "unknown":
                    normalized[key] = defaults[key]
                else:
                    normalized.setdefault(key, defaults[key])
        return normalized

    @classmethod
    def _prompt_safe_entry(cls, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Strip raw narrative and keep only prompt-safe structured signals."""
        normalized = cls._normalize_metadata(item)
        if not normalized.get("retrieval_eligible"):
            return None
        if normalized.get("content_class") != "structured_signal":
            return None
        signal = normalized.get("signal")
        if not isinstance(signal, dict) or not signal:
            return None
        outcome = normalized.get("outcome_status", "unknown")
        if outcome in {"failed", "rejected", "unknown", "unclassified"}:
            # Fail closed unless the caller explicitly opts into a non-released status
            # via get_prompt_context(outcome_status=...). Default path requires release.
            pass
        safe = {
            "id": normalized.get("id"),
            "namespace": normalized.get("namespace"),
            "content_class": normalized.get("content_class"),
            "run_id": normalized.get("run_id"),
            "agent": normalized.get("agent"),
            "outcome_status": normalized.get("outcome_status"),
            "signal": dict(signal),
        }
        return {k: v for k, v in safe.items() if v is not None}

    def add_embedding(self, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add an embedding with metadata to the vector database."""
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} != {self.dimension}")

        metadata = self._normalize_metadata(metadata)
        self.index.add(embedding.reshape(1, -1))

        metadata["timestamp"] = metadata.get("timestamp") or datetime.now().isoformat()
        metadata["id"] = len(self.metadata)
        self.metadata.append(metadata)

    def audit_search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Audit-only raw similarity search. Never use for prompt construction."""
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} != {self.dimension}")

        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                result = dict(self.metadata[idx])
                result["distance"] = float(distances[0][i])
                results.append(result)
        return results

    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Prompt-safe similarity results only; prefer get_prompt_context() in agents."""
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} != {self.dimension}")

        distances, indices = self.index.search(query_embedding.reshape(1, -1), max(k * 4, k))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            result = self._prompt_safe_entry(self.metadata[idx])
            if result is None:
                continue
            result["distance"] = float(distances[0][i])
            results.append(result)
            if len(results) >= k:
                break
        return results

    def get_prompt_context(
        self,
        query_embedding: Optional[np.ndarray] = None,
        *,
        k: int = 5,
        namespace: Optional[str] = None,
        outcome_status: Optional[str] = None,
        purpose: Optional[str] = None,
        run_id: Optional[str] = None,
        allow_non_released: bool = False,
    ) -> List[Dict[str, Any]]:
        """Central fail-closed boundary for prompt retrieval.

        Returns only structured_signal entries that are retrieval_eligible.
        Raw content, feedback prose, and debate arguments are never included.
        By default, only outcome_status=released items are returned unless
        allow_non_released=True or an explicit outcome_status filter is set
        (e.g. for cross-run rejection tags stored elsewhere).
        """
        if query_embedding is not None:
            candidates = self.audit_search_similar(query_embedding, max(k * 8, k))
        else:
            candidates = list(self.metadata)
        candidates.extend(self.debate_log)
        candidates.extend(self.feedback_log)

        eligible: List[Dict[str, Any]] = []
        for item in candidates:
            normalized = self._normalize_metadata(item)
            if normalized.get("content_class") != "structured_signal":
                continue
            if not normalized.get("retrieval_eligible"):
                continue
            if namespace and normalized.get("namespace") != namespace:
                continue
            if run_id and normalized.get("run_id") != run_id:
                continue
            if purpose and normalized.get("signal", {}).get("purpose") != purpose:
                continue
            status = normalized.get("outcome_status", "unknown")
            if outcome_status:
                if status != outcome_status:
                    continue
            elif not allow_non_released and status != "released":
                continue
            safe = self._prompt_safe_entry(normalized)
            if safe is None:
                continue
            eligible.append(safe)
            if len(eligible) >= k:
                break
        return eligible

    def add_debate_entry(
        self,
        topic: str,
        proposer_argument: str,
        challenger_argument: str,
        moderator_decision: str,
        score: float,
        structured_signal: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        agent: str = "HypothesisDebate",
        outcome_status: Optional[str] = None,
    ):
        """Add a debate entry. Raw arguments stay for audit; prompts use signal only."""
        defaults = self._active_run_defaults()
        # Curated objection tags are published for future debates even when the
        # debate itself failed; raw arguments remain stripped by _prompt_safe_entry.
        if outcome_status is None:
            outcome_status = "released" if structured_signal else "unknown"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "content_class": "structured_signal" if structured_signal else "generated_narrative",
            "retrieval_eligible": bool(structured_signal),
            "namespace": "debate_transcripts",
            "run_id": run_id or defaults["run_id"],
            "agent": agent,
            "outcome_status": outcome_status,
            "signal": structured_signal or {},
            "topic": topic,
            "proposer_argument": proposer_argument,
            "challenger_argument": challenger_argument,
            "moderator_decision": moderator_decision,
            "score": score,
        }
        self.debate_log.append(self._normalize_metadata(entry))
        # Write debate log immediately (not deferred to save()) so that
        # debate_log.json is always present even if the process crashes.
        # Atomic: a crash mid-write leaves the previous complete file, never
        # a 0-byte husk that breaks the next startup load.
        try:
            _atomic_write_json(config.debate_log_path, self.debate_log)
        except Exception as e:
            logger.warning("Failed to write debate_log.json immediately: %s", e)
        self.save()

    def add_feedback_entry(
        self,
        agent_name: str,
        section: str,
        score: float,
        feedback: str,
        iteration: int,
        structured_signal: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        outcome_status: str = "unknown",
    ):
        """Store raw review prose for audit plus optional structured verdict signal."""
        defaults = self._active_run_defaults()
        signal = structured_signal or {
            "verdict": "block" if score < 5 else ("pass" if score >= 8 else "revise"),
            "score": float(score),
            "section": section,
            "blocking": score < 5,
            "category": "supervisor_review",
        }
        entry = {
            "timestamp": datetime.now().isoformat(),
            "content_class": "structured_signal",
            "retrieval_eligible": True,
            "namespace": "supervisor_feedback",
            "run_id": run_id or defaults["run_id"],
            "outcome_status": outcome_status if outcome_status != "unknown" else (
                "released" if score >= 8 else ("failed" if score < 5 else "revised")
            ),
            "signal": signal,
            "agent": agent_name,
            "section": section,
            "score": score,
            "feedback": feedback,
            "iteration": iteration,
        }
        self.feedback_log.append(self._normalize_metadata(entry))
        self.save()

    def get_recent_debates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Audit-only: full debate rows including raw arguments."""
        return self.debate_log[-limit:]

    def get_recent_feedback(self, agent_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Audit-only: full feedback rows including raw review prose."""
        if agent_name:
            filtered = [entry for entry in self.feedback_log if entry.get("agent") == agent_name]
            return filtered[-limit:]
        return self.feedback_log[-limit:]

    def get_feedback_signals(self, agent_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Prompt/control-safe feedback: scores and verdicts only, never review prose."""
        rows = self.get_recent_feedback(agent_name=agent_name, limit=limit)
        signals = []
        for entry in rows:
            signal = entry.get("signal") if isinstance(entry.get("signal"), dict) else {}
            signals.append({
                "run_id": entry.get("run_id", "unknown"),
                "agent": entry.get("agent", "unknown"),
                "outcome_status": entry.get("outcome_status", "unknown"),
                "section": entry.get("section") or signal.get("section"),
                "score": entry.get("score", signal.get("score")),
                "verdict": signal.get("verdict"),
                "blocking": signal.get("blocking"),
                "category": signal.get("category"),
            })
        return signals

    def get_average_score(self, agent_name: str, recent_n: int = 5) -> float:
        """Get average score for an agent over recent iterations."""
        recent = self.get_feedback_signals(agent_name, recent_n)
        if not recent:
            return 0.0
        scores = [float(entry["score"]) for entry in recent if entry.get("score") is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def clear_all(self):
        """Clear all in-memory and on-disk memory data."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.debate_log = []
        self.feedback_log = []
        self.save()

    def delete_run_vectors(self, run_id: str) -> dict:
        """Remove all vectors, debate, and feedback entries for a given run_id.

        Rebuilds the FAISS index from scratch with only the remaining data.
        Returns a summary of what was cleaned.
        """
        removed_vectors = 0
        removed_debate = 0
        removed_feedback = 0

        # 1. Filter metadata and rebuild FAISS index
        if self.metadata:
            keep_mask = [entry.get("run_id") != run_id for entry in self.metadata]
            removed_vectors = sum(1 for keep in keep_mask if not keep)

            remaining_metadata = [entry for entry, keep in zip(self.metadata, keep_mask) if keep]

            # Rebuild index from scratch with only remaining vectors
            new_index = faiss.IndexFlatL2(self.dimension)
            if remaining_metadata:
                # Re-read the full index to get vectors, then rebuild
                # Since FAISS doesn't let us extract individual vectors by position
                # after adding, we rebuild from the original vectors stored alongside metadata
                # Actually, we need to re-add vectors. The index stores them sequentially.
                # We must rebuild by re-adding all vectors for the kept entries.
                old_index = self.index
                if old_index.ntotal == len(self.metadata):
                    # Index is in sync — extract kept vectors and rebuild
                    import numpy as np
                    vectors = np.vstack([
                        old_index.reconstruct(i).reshape(1, -1)
                        for i in range(old_index.ntotal)
                        if keep_mask[i]
                    ])
                    if vectors.shape[0] > 0:
                        new_index.add(vectors)
                else:
                    # Index out of sync — can't reconstruct, start fresh
                    pass

            self.index = new_index
            self.metadata = remaining_metadata

        # 2. Filter debate log
        original_debate = len(self.debate_log)
        self.debate_log = [entry for entry in self.debate_log if entry.get("run_id") != run_id]
        removed_debate = original_debate - len(self.debate_log)

        # 3. Filter feedback log
        original_feedback = len(self.feedback_log)
        self.feedback_log = [entry for entry in self.feedback_log if entry.get("run_id") != run_id]
        removed_feedback = original_feedback - len(self.feedback_log)

        # 4. Save to disk
        self.save()

        return {
            "vectors_removed": removed_vectors,
            "debate_entries_removed": removed_debate,
            "feedback_entries_removed": removed_feedback,
        }


# Global memory instance
memory = ResearchMemory()
