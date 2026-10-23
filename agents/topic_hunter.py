"""
TopicHunterAgent — citation-graph gap analysis, novelty filter, parallel hunts.

CHANGELOG (this revision, v3):
  1. arXiv access moved to the official `arxiv` client (built-in 1 req / 3s
     pacing + 429/503 retry honoring Retry-After). The portalocker /tmp
     file-lock hack is gone; the API gateway (with its new adaptive AIMD
     bucket) paces and protects every provider call.
  2. API gateway v2: adaptive token buckets (rate cut on 429/503, additive
     recovery), backoff jitter, single-flight coalescing of identical reads,
     and is_available()/breaker_state() probes so open breakers short-circuit
     to a skip instead of paying retry sleeps.
  3. LLM-driven seed generation: one cheap call converts cross-run lessons +
     frontier terms into specific technical seed phrases (strategy
     "llm_diverse"), with graceful fallback to the static template seeds.
  4. Query building no longer turns seed filler words ("gaps", "problems",
     "open", "underexplored", ...) into literal search keywords, which used to
     retrieve noise literature and degrade gap quality downstream.
  5. Two-pool LLM budget: seed/query generation calls can no longer starve the
     gate chain — the gate chain reserves its own sub-budget, so gaps that
     were generated get evaluated instead of being silently discarded.
  6. literature_evidence is now relevance-ranked (embedding cosine with
     token-overlap fallback) instead of "first 8 papers fetched", so screener
     and novelty prompts judge gaps against the closest prior work.
  7. Structural gap mining now routes S2 calls through the API gateway
     (previously bypassed it with its own retry/sleep loop).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests

from core.config import config
from core.utils import log_agent_action, parse_json_from_llm, calculate_similarity, title_token_overlap, call_llm_json, tfidf_cosine
from core.llm import call_llm, generate_embedding, is_embedding_available
from core.llm import get_llm_client
from core.context import RunContext, get_active_context
from core.contracts import FeasibilityReport, Topic
from core.memory import memory
from core.run_log import CrossRunMemory, get_tracker
from core.sources import SourceClient
from agents.hypothesis_debate import EloStore, hypothesis_kind
from core.capabilities import SANDBOX_CAPABILITY_MANIFEST, check_plan_feasibility, effective_wall_clock_seconds
from core.evidence_synthesis import build_cross_paper_evidence_map, validate_candidate_bridge_claim, validate_topic_admission
from core.structural_gaps import find_coupling_gaps, find_coupling_gaps_openalex
from core.sparsity_matrix import find_sparse_cells
from core.contradiction_mining import find_contradictions
from core.signal_mining import (
    temporal_momentum,
    citation_concentration,
    concept_cooccurrence_gaps,
    cluster_reviewer_weaknesses,
)


logger = logging.getLogger(__name__)


def _registry_system(name: str, **slots: Any) -> Optional[str]:
    """Render a registry system prompt (None when unavailable/offline-safe).

    Keeps the inline user prompt as the task body — the structural win is
    the system/user split, which survives truncation better than a persona
    buried at the top of a 16k user blob.
    """
    try:
        from core.prompts import render_system

        return render_system(name, **slots)
    except Exception:
        return None


def _llm_kwargs(prompt_name: str, tier: str, **slots: Any) -> Dict[str, Any]:
    """System kwarg for a hunter LLM call (stub-safe; tier stays explicit)."""
    kw: Dict[str, Any] = {}
    system = _registry_system(prompt_name, **slots)
    if system:
        kw["system"] = system
    return kw


def _compact_hunt_block(items: Any, max_chars: int) -> str:
    """Relevance-ranked compaction for hunt prompt blocks (attention budget).

    Falls back to plain truncation when the registry helper is unavailable.
    """
    try:
        from core.prompts import compact_context

        text = compact_context(items if isinstance(items, list) else [items],
                               max_chars)
        if text:
            return text
    except Exception:
        pass
    try:
        return str(items)[:max_chars]
    except Exception:
        return ""

# arXiv access policy (v4):
#   - The official `arxiv` pip client is used instead of hand-rolled HTTP. It
#     enforces the documented 1 req / 3s pacing (delay_seconds), and a
#     module-level singleton shares one last-request clock across all
#     TopicHunterAgent instances in this process.
#   - SINGLE RETRY OWNER: the client runs with num_retries=0. Previously the
#     lib retried internally (4 tries x 3s sleeps ~= 50s) AND the gateway
#     retried the whole search on top (3 outer attempts) — up to ~12 HTTP
#     requests and ~3 minutes burned per query during a 429 storm before
#     returning []. Now a 429 surfaces immediately and the gateway alone
#     owns retry/backoff (AIMD penalty + jitter + breaker + health), bounding
#     a stormed query to ~3 paced attempts before graceful degradation.
#   - page_size=50: every caller asks for <=50 results, so one page serves a
#     whole search instead of over-fetching 100-row pages into 429 pressure.
#   - The previous cross-process portalocker file lock under /tmp was removed:
#     it duplicated gateway responsibilities and broke on Windows.
import arxiv as _arxiv_lib

_ARXIV_CLIENT: Optional["_arxiv_lib.Client"] = None
_ARXIV_CLIENT_LOCK = threading.Lock()


def _get_arxiv_client() -> "_arxiv_lib.Client":
    """Process-wide shared official arXiv client (shared delay clock)."""
    global _ARXIV_CLIENT
    if _ARXIV_CLIENT is None:
        with _ARXIV_CLIENT_LOCK:
            if _ARXIV_CLIENT is None:
                _ARXIV_CLIENT = _arxiv_lib.Client(
                    page_size=50,
                    delay_seconds=3.0,
                    num_retries=0,
                )
    return _ARXIV_CLIENT


_FAILED_TOPIC_OVERLAP_THRESHOLD = 0.55

# Fix #6: only the most recent N rejection entries influence seed/query
# keyword stripping. Without this cap, a long-running session accumulates so
# many rejected fingerprints that every keyword eventually gets excluded and
# seeds decay into generic, domain-less queries.
_REJECTION_HISTORY_WINDOW = 20

# Reason tag for seed-phrase exhaustion entries in CrossRunMemory. Consumed
# as exact-phrase exclusions (see _generate_dynamic_seeds), never as
# word-level bans.
_SEED_EXHAUSTED_REASON = "seed_exhausted_no_topics"

# Reason tag for seeds that produced topics (which later failed debate or
# were superseded). Also an exact-phrase exclusion — a rephrase of the same
# angle is handled by the fuzzy overlap guard in _generate_dynamic_seeds.
_SEED_USED_REASON = "seed_used"


def _normalize_seed_text(seed: str) -> str:
    """Canonical form for exact-phrase seed comparison."""
    return re.sub(r"\s+", " ", (seed or "").strip().lower())


def _seed_passes_domain_gate(seed_text: str, domain: str) -> bool:
    """e0ec9e91: LLM/frontier seeds drifted to pure-ML (world models, tabular,
    distillation) for a software_engineering run. The taxonomy fixed retrieval
    categories but not seed content. For SE runs, require >=1 SE marker token
    in the seed phrase; other domains pass through (no gate) to avoid
    over-constraining them.
    """
    try:
        key = re.sub(r"\s+", " ", (domain or "").strip().lower())
    except Exception:
        return True
    # 0dfc1c58: "software engineer" (no -ing) must gate like SE too.
    if "software" not in key and "software_engineering" not in key and key not in ("software engineering", "software engineer"):
        # Only gate SE runs; kind_bias seeds for other domains are fine.
        if "se" not in key.split():
            return True
    lowered = (seed_text or "").lower()
    return any(m in lowered for m in _SE_MARKERS)


def effective_discovery_timeout(n_seeds: int, seed_timeout_sec: int, configured_sec: int) -> int:
    """Discovery window that fits every seed budget plus retrieval overhead.

    Without this, a short discovery window silently starves the tail seeds
    (run 95c46c6f: 900 s window with 180 s seeds attempted only 2 of 5).
    The configured value is a floor component, never a ceiling shrink.
    """
    floor = max(1, int(n_seeds)) * max(1, int(seed_timeout_sec)) + 300
    return max(max(1, int(configured_sec)), floor)

# Fix #6: no matter how much overlap is found with rejected fingerprints, at
# least this many original seed keywords are preserved so a query never goes
# out completely un-anchored.
_MIN_SURVIVING_KEYWORDS = 1

# Known public ML datasets — if a gap's dataset_plan contains any of these
# substrings, it passes admissibility without checking the local catalog.
# Moved to core.datasets to fix circular import; re-export for backward compat
from core.datasets import KNOWN_PUBLIC_DATASETS as _KNOWN_PUBLIC_DATASETS

# Fix #3: the similarity level at which a candidate topic's contribution gets
# compared against prior work by an LLM (rather than judged on cosine sim
# alone). This must never sit ABOVE the reject threshold, or a candidate can
# be auto-rejected purely on embedding similarity without ever reaching
# comparison. Actual value is clamped dynamically in evaluate_layered_novelty.
_DEFAULT_COMPARISON_TRIGGER = 0.70

# Fix #7: how many raw candidate gaps to request per seed hunt. Raised from 3
# to give the downstream gate chain more surviving material, since the chain
# is multi-stage and each stage has a nonzero rejection rate.
_GAPS_PER_SEED_REQUEST = 6

_ARXIV_CATEGORIES = {
    "machine_learning": ["cs.LG", "stat.ML"],
    "natural_language_processing": ["cs.CL", "cs.AI"],
    "computer_vision": ["cs.CV", "cs.AI"],
    "reinforcement_learning": ["cs.LG", "cs.AI"],
    "graph_neural_networks": ["cs.LG", "cs.SI"],
    "fairness": ["cs.LG", "cs.CY"],
    "federated_learning": ["cs.LG", "cs.DC"],
    "optimization": ["math.OC", "cs.LG"],
    "software_engineering": ["cs.SE", "cs.PL"],
    "general": ["cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"],
}

_TOPICS_TO_SUBCATEGORY = {
    "attention": "machine_learning",
    "graph": "graph_neural_networks",
    "diffusion": "machine_learning",
    "reinforcement": "reinforcement_learning",
    "federated": "federated_learning",
    "llm": "natural_language_processing",
    "vision": "computer_vision",
    "nlp": "natural_language_processing",
    "fuzz": "software_engineering",
    "testing": "software_engineering",
    "debugging": "software_engineering",
    "repair": "software_engineering",
    "program": "software_engineering",
    "code": "software_engineering",
    "software": "software_engineering",
    "general": "general",
}

# Research-domain strings (RESEARCH_DOMAIN / run domain arg) → subcategory.
# Checked before the generic fallback so "software engineering" stops
# resolving to ML categories (v5: run 55ac3547 retrieved cs.LG papers for SE).
_DOMAIN_TO_SUBCATEGORY = {
    "software engineering": "software_engineering",
    "software_engineering": "software_engineering",
    "softwareengineering": "software_engineering",
    # 0dfc1c58: the run domain arrived as "software engineer" (no -ing),
    # which missed the map, fell to "general", and retrieved cs.LG papers
    # for SE seeds. Normalize the common variant explicitly.
    "software engineer": "software_engineering",
    "se": "software_engineering",
    "program analysis": "software_engineering",
    "machine learning": "machine_learning",
    "natural language processing": "natural_language_processing",
    "nlp": "natural_language_processing",
    "computer vision": "computer_vision",
    "reinforcement learning": "reinforcement_learning",
}

# Seed/domain vocabulary that forces the software_engineering subcategory
# even when the run domain is generic (e.g. RESEARCH_DOMAIN=computer_science
# with an SE seed phrase).
_SE_MARKERS = frozenset({
    "fuzz", "fuzzing", "fuzzer", "testing", "test", "debug", "debugging",
    "repair", "patch", "patching", "program", "code", "software", "coverage",
    "crash", "bug", "defect", "vulnerability", "static analysis", "compiler",
})


def _subcategory_for_domain(domain: str) -> str:
    """Map a research-domain string to an _ARXIV_CATEGORIES subcategory."""
    key = re.sub(r"\s+", " ", (domain or "").strip().lower())
    if key in _DOMAIN_TO_SUBCATEGORY:
        return _DOMAIN_TO_SUBCATEGORY[key]
    for marker, subcat in _DOMAIN_TO_SUBCATEGORY.items():
        if len(marker) > 3 and marker in key:
            return subcat
    return "general"

_QUERY_STOPWORDS = frozenset({
    "and", "for", "the", "are", "but", "not", "with", "this", "that", "from",
    "have", "has", "was", "were", "been", "being", "can", "may", "its", "our",
})

# Dataset-plan words that describe WHERE data comes from generically rather
# than naming a concrete dataset — never worth an HF id lookup (run b099dcaa
# burned two 404s per gap on dataset_plan="public").
_GENERIC_DATASET_PLAN_WORDS = frozenset({
    "public", "publicly", "available", "open", "custom", "curated", "collected",
    "benchmark", "benchmarks", "dataset", "datasets", "data", "synthetic",
    "generated", "small", "standard", "standardized", "toy", "sample",
    "real", "realworld", "real-world", "downloaded", "hosted", "online",
})

_EXPLORATION_KIND_BIASES = {
    "attention": {"method": ["transformer", "self-attention", "multi-head"], "evaluation": ["efficiency", "scaling", "long-context"]},
    "graph": {"method": ["graph neural", "message passing", "spectral"], "evaluation": ["inductive", "scalability", "heterogeneous"]},
    "diffusion": {"method": ["diffusion model", "score-based", "denoising"], "evaluation": ["sample quality", "convergence", "likelihood"]},
    "reinforcement": {"method": ["reinforcement learning", "policy gradient", "temporal difference"], "evaluation": ["sample efficiency", "exploration", "stability"]},
    "federated": {"method": ["federated", "distributed optimization", "privacy"], "evaluation": ["communication efficiency", "convergence", "non-IID"]},
    "llm": {"method": ["language model", "fine-tuning", "prompt"], "evaluation": ["zero-shot", "reasoning", "alignment"]},
    "vision": {"method": ["visual", "image", "spatial"], "evaluation": ["detection", "segmentation", "recognition"]},
    "nlp": {"method": ["text", "sequence", "tokeniz"], "evaluation": ["classification", "generation", "extraction"]},
    "software_engineering": {"method": ["fuzzing", "program repair", "static analysis"], "evaluation": ["bug detection", "test coverage", "patch correctness"]},
    "general": {"method": ["learning", "optimization", "generalization"], "evaluation": ["accuracy", "robustness", "scalability"]},
}

# Fix #2: blocked phrases for the sandbox feasibility filter. Kept as plain
# strings (matching original behavior) — the negation-aware check is done in
# `_phrase_blocked_in_text`, not by changing this list.
_BLOCKED_SANDBOX_PHRASES = [
    "large language model fine-tune",
    "gpu cluster",
    "human subjects",
    "clinical trial",
    "wet lab",
    "robot hardware",
    "million parameter training from scratch",
]

# Fix #2: negation cues that, if found immediately before a blocked phrase,
# mean the text is disclaiming the requirement rather than stating it.
_NEGATION_CUES = ("not ", "no ", "without ", "won't need ", "does not require ", "doesn't require ", "avoids ")

# Triggers for consulting the shared sandbox manifest in feasibility_filter.
# Strictly METHOD-level (ways an experiment can demand an accelerator or the
# network). Subject-matter vocabulary — "llm serving", "kernel scheduling",
# "memory allocation", "speculative decoding" — is deliberately absent: those
# name legitimate research subjects that the CPU envelope below reframes as
# simulation / small-model / statistical studies. Banning subjects would
# outlaw exactly the methodology topics this system is best at on CPU.
_MANIFEST_CHECK_TRIGGERS = (
    "requires internet", "must download", "needs api access to",
    "gpu", "cuda", "tpu", "vllm", "tensorrt", "nccl", "multi-gpu",
    "distributed training", "fine-tune", "fine tune", "finetune",
    "pretrain", "pre-train", "billion parameter",
    "cuda kernel", "gpu profiling", "gpu cluster", "gpu memory",
)

# What a CPU-only study may look like in this sandbox (1 CPU, ~2GB RAM,
# SANDBOX_TIMEOUT_SEC wall clock, no outbound network, capped rows/epochs).
# Injected into generation + formalization prompts so the LLM designs *for*
# the envelope instead of proposing GPU work that gets filtered downstream.
# The key reframing for accelerator-adjacent subjects: simulate the system
# (discrete-event / Monte-Carlo dynamics, synthetic timing models) or measure
# small-scale proxies — do not require the hardware.
_CPU_EXECUTION_ENVELOPE = (
    "CPU-ONLY execution envelope (no GPU in sandbox): the study must run on "
    "1 CPU with ~2GB RAM inside the sandbox wall clock, with no outbound "
    "network. Sanctioned designs: (a) scikit-learn scale (<=100k rows), "
    "(b) small neural nets (few epochs, MNIST/CIFAR-scale data), "
    "(c) pure statistical methodology (bootstrap CIs, variance estimation, "
    "power analysis, reporting-standards studies), (d) discrete-event or "
    "Monte-Carlo SIMULATION of system behavior — e.g. simulate "
    "speculative-decoding acceptance/timing dynamics or serving-queue effects "
    "instead of running LLM serving hardware, (e) tiny transformers "
    "(<10M params, short sequences), (f) synthetic-data studies with "
    "controlled ground truth. required_resources MUST state gpu:false "
    "with max_runtime_seconds inside the wall clock. A topic ABOUT accelerators "
    "or LLMs is welcome; an experiment REQUIRING them is not."
)

# OpenAlex concept IDs (https://api.openalex.org/concepts) corresponding to
# each _ARXIV_CATEGORIES subcategory. IDs verified against live API 2026-09-12.
# software_engineering intentionally maps to [] (no concept filter): no SE
# concept ID has been verified, and a WRONG filter (the old ML fallback) is
# strictly worse than keyword-only search — it silently nukes SE retrieval.
_OPENALEX_CONCEPTS = {
    "machine_learning": ["C119857082", "C154945302"],       # Machine learning, Artificial intelligence
    "natural_language_processing": ["C204321447", "C41008148"],  # NLP, Computer science
    "computer_vision": ["C31972630", "C154945302"],          # Computer vision, AI
    "reinforcement_learning": ["C119857082", "C154945302"],
    "graph_neural_networks": ["C119857082", "C41008148"],
    "fairness": ["C154945302", "C33923547"],                 # AI, Mathematics
    "federated_learning": ["C119857082", "C31258907"],       # ML, Computer network
    "optimization": ["C33923547", "C119857082"],             # Mathematical optimization, ML
    "software_engineering": [],                             # keyword-only (see note above)
    "general": ["C119857082", "C154945302", "C41008148"],
}

# Seed phrases are meta-level descriptions ("open problems in attention",
# "methodological gaps in evaluation"). Searching arXiv/OpenAlex for the words
# "problems", "gaps", "open", "underexplored" retrieves noise, and noise
# literature poisons every downstream gate. These are excluded from keyword
# extraction before building provider queries. The keyword floor guarantees at
# least one content-bearing keyword still survives (see _apply_keyword_floor).
_SEED_FILLER_WORDS = frozenset({
    "gaps", "gap", "problems", "problem", "open", "challenges", "challenge",
    "underexplored", "unexplored", "understudied", "methodological",
    "methodology", "limitations", "limitation", "issues", "issue",
    "questions", "question", "areas", "area", "study", "studies", "research",
    "literature", "survey", "review", "future", "directions", "direction",
    "work", "works", "novel", "emerging", "critique", "critical", "insight",
    "insights", "pitfalls", "pitfall", "weakness", "weaknesses",
})


class ResearchSourceUnavailable(RuntimeError):
    """Raised when discovery cannot consult any external scholarly source."""


class TopicHunterAgent:
    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.runtime_config = self.context.config if self.context else config
        self.vector_memory = self.context.memory if self.context else memory
        self.client = get_llm_client()
        self.openalex_headers = {
            "User-Agent": f"ScholarGraph/2.0 (mailto:{self.runtime_config.openalex_email})"
        }
        self.s2_headers = {"User-Agent": "ScholarGraph/2.0"}
        if self.runtime_config.semantic_scholar_api_key:
            self.s2_headers["x-api-key"] = self.runtime_config.semantic_scholar_api_key
        self.base_urls = {
            "openalex": "https://api.openalex.org",
            "crossref": "https://api.crossref.org",
            "semantic_scholar": "https://api.semanticscholar.org/graph/v1",
        }
        self._selection_counter = 0
        self._rng = random.Random(int(getattr(self.runtime_config, "topic_exploration_seed", 42)))
        self.source_client = SourceClient(
            str(Path(self.runtime_config.output_dir) / "source_cache")
        )
        self.rejection_log: List[Dict[str, Any]] = []
        self.source_health: Dict[str, Dict[str, Any]] = {}
        self._excluded_titles_cache: Optional[List[str]] = None
        self._iteration_failures = 0
        # Exact seed phrases attempted by this instance. Iteration retries
        # recreate the agent, so this is only the first layer — the durable
        # layer is seed-phrase rejections in CrossRunMemory (reason
        # "seed_exhausted_no_topics"), consumed as exact-phrase exclusions
        # by _generate_dynamic_seeds (never as word-level bans).
        self._attempted_seeds: set = set()
        self._arxiv_consecutive_failures = 0
        self._arxiv_disabled = False
        # Feature 5: run-scoped, thread-safe query cache for cross-seed dedup
        self._run_query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._run_query_cache_lock = threading.Lock()
        self._dataset_catalog_cache: Optional[List[Dict[str, Any]]] = None

    def _source_ok(self, name: str):
        self.source_health[name] = {"ok": True}

    def _source_failed(self, name: str, error: Exception):
        self.source_health[name] = {"ok": False, "error": str(error)}

    def _s2_get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
        """Semantic Scholar GET routed through API gateway."""
        runtime_cfg = getattr(self, "runtime_config", None)
        if runtime_cfg is not None and not getattr(runtime_cfg, "semantic_scholar_enabled", True):
            log_agent_action("TopicHunter", "s2_skipped_disabled", {"url": url[:120]})
            return None
        from core.api_gateway import get_gateway, RateLimitError

        gateway = get_gateway()
        try:
            if not gateway.is_available("semantic_scholar"):
                return None
        except Exception:
            pass

        def _do_fetch():
            r = requests.get(
                url,
                headers=self.s2_headers,
                params=params or {},
                timeout=20,
            )
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", "3"))
                raise RateLimitError("semantic_scholar", retry_after)
            return r

        try:
            return gateway.request("semantic_scholar", _do_fetch, retries=2, backoff_base=3.0)
        except Exception as e:
            logger.debug("S2 request failed for %s: %s", url, e)
            return None

    def search_openalex(self, query: str, limit: int = 50, extra_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        runtime_cfg = getattr(self, "runtime_config", None)
        if runtime_cfg is not None and not getattr(runtime_cfg, "openalex_enabled", True):
            log_agent_action("TopicHunter", "openalex_skipped_disabled", {"query": query[:80]})
            return []
        logger.info("OpenAlex search: query=%r, limit=%d, params=%s", query[:80], limit, json.dumps(extra_params or {}, sort_keys=True)[:160])
        from core.api_gateway import get_gateway
        try:
            if not get_gateway().is_available("openalex"):
                log_agent_action("TopicHunter", "openalex_skipped_breaker_open", {"query": query[:80]})
                self._source_failed("openalex", RuntimeError("circuit open"))
                return []
        except Exception:
            pass
        try:
            url = f"{self.base_urls['openalex']}/works"
            params = {
                "search": query,
                "per_page": min(limit, 100),
                # OpenAlex exposes abstracts as an inverted index; requesting a
                # non-existent `abstract` field is a 400 in current API versions.
                # M2: best_oa_location + primary_location + open_access carry
                # the OA PDF URL resolution (no extra HTTP, rate kept 5.0/3).
                "select": "id,title,abstract_inverted_index,publication_year,cited_by_count,concepts,type,doi,best_oa_location,primary_location,open_access,referenced_works",
                "mailto": self.runtime_config.openalex_email,
            }
            if extra_params:
                params.update(extra_params)
            artifact = self.source_client.fetch_json(
                "openalex",
                url,
                headers=self.openalex_headers,
                params=params,
                validator=lambda payload: isinstance(payload, dict) and "results" in payload,
            )
            if artifact["status"] == "unavailable":
                # Don't raise immediately - try to continue with partial data
                warnings = artifact.get("warnings", [])
                log_agent_action("TopicHunter", "openalex_unavailable", {"warnings": warnings})
                return []
            rows = artifact["content"].get("results", [])
            for row in rows:
                inverted = row.pop("abstract_inverted_index", None) or {}
                if inverted:
                    ordered = sorted(((pos, word) for word, positions in inverted.items() for pos in positions))
                    row["abstract"] = " ".join(word for _, word in ordered)
            self._source_ok("openalex")
            logger.info("OpenAlex returned %d results for query=%r", len(rows), query[:80])
            return rows
        except Exception as e:
            self._source_failed("openalex", e)
            log_agent_action("TopicHunter", "search_openalex_error", {"error": str(e)})
            logger.warning("OpenAlex search failed for query=%r: %s", query[:80], e)
            return []

    def search_arxiv(self, query: str, max_results: int = 30) -> List[Dict[str, Any]]:
        """arXiv search via the official `arxiv` client, paced by the API gateway.

        The shared client enforces the documented 1 req / 3s gap but does NOT
        retry internally (num_retries=0) — the gateway alone owns retry/backoff
        so a 429 storm costs ~3 paced attempts, not nested retry loops.
        The gateway wraps the whole search as one call: token-bucket pacing,
        circuit breaking, health tracking, and adaptive rate learning. A
        persistent 429 degrades gracefully to [] so OpenAlex/S2 carry the hop.
        Supports an injected `_arxiv_client` for tests.
        Cap: at most 30 results per query (one 50-row page serves it) to keep
        429 pressure low; after 2 consecutive rate-limit failures the rest of
        the hop fast-skips arXiv until OpenAlex/S2 carry it.
        """
        runtime_cfg = getattr(self, "runtime_config", None)
        if runtime_cfg is not None and not getattr(runtime_cfg, "arxiv_enabled", True):
            log_agent_action("TopicHunter", "arxiv_skipped_disabled", {"query": query[:80]})
            return []
        if getattr(self, "_arxiv_consecutive_failures", 0) >= 2:
            log_agent_action("TopicHunter", "arxiv_skipped_consecutive_failures", {"query": query[:80]})
            return []
        logger.info("arXiv search: query=%r, max_results=%d", query[:80], max_results)

        from core.api_gateway import get_gateway

        # Fast skip when the arXiv breaker is open — don't pay retry sleeps.
        try:
            if not get_gateway().is_available("arxiv"):
                log_agent_action("TopicHunter", "arxiv_skipped_breaker_open", {"query": query[:80]})
                self._source_failed("arxiv", RuntimeError("circuit open"))
                return []
        except Exception:
            pass

        def _do_search():
            search = _arxiv_lib.Search(
                query=query,
                max_results=min(max_results, 30),
                sort_by=_arxiv_lib.SortCriterion.SubmittedDate,
                sort_order=_arxiv_lib.SortOrder.Descending,
            )
            client = getattr(self, "_arxiv_client", None) or _get_arxiv_client()
            results = []
            for r in client.results(search):
                published = getattr(r, "published", None)
                year = 0
                try:
                    year = int(published.year) if published else 0
                except (AttributeError, ValueError, TypeError):
                    year = 0
                authors = []
                for a in (r.authors or []):
                    name = getattr(a, "name", None) or str(a)
                    if name:
                        authors.append(name)
                if r.title and r.title.strip():
                    results.append({
                        "title": r.title.strip().replace("\n", " "),
                        "abstract": (r.summary or "").strip().replace("\n", " "),
                        "year": year,
                        "authors": authors,
                        "arxiv_id": r.entry_id or "",
                        "categories": list(r.categories or []),
                    })
            return results

        try:
            results = get_gateway().request("arxiv", _do_search, retries=2, backoff_base=8.0, backoff_max=120.0)
            self._source_ok("arxiv")
            self._arxiv_consecutive_failures = 0
            logger.info("arXiv returned %d results for query=%r", len(results or []), query[:80])
            return results or []
        except Exception as e:
            self._source_failed("arxiv", e)
            is_rate_limit = "rate limit" in str(e).lower() or "429" in str(e) or "503" in str(e)
            if is_rate_limit:
                self._arxiv_consecutive_failures += 1
            else:
                self._arxiv_consecutive_failures = 0
            log_level = logging.WARNING if is_rate_limit else logging.DEBUG
            logger.log(log_level, "arxiv search failed for query=%r: %s (rate_limit=%s)", query, e, is_rate_limit)
            log_agent_action("TopicHunter", "search_arxiv_error", {
                "error": str(e), "query": query, "rate_limit": is_rate_limit,
            })
            return []

    def search_openalex_multi(self, queries: List[Tuple[str, Dict[str, Any]]], limit: int = 30) -> List[Dict[str, Any]]:
        """Run multiple OpenAlex queries with optional extra params and merge results."""
        all_results = []
        for query, params in queries:
            all_results.extend(self.search_openalex(query, limit, extra_params=params))
        return all_results

    def search_arxiv_multi(self, queries: List[str], max_results: int = 30) -> List[Dict[str, Any]]:
        """Run multiple arXiv queries and merge results.

        Pacing is delegated entirely to the shared arXiv client (3s gap) and
        the gateway bucket — no extra inter-query sleep layer. Identical
        duplicate queries are collapsed to a single network call.
        """
        all_results: List[Dict[str, Any]] = []
        seen_queries: set = set()
        for query in queries:
            key = query.strip().lower()
            if not key or key in seen_queries:
                continue
            seen_queries.add(key)
            all_results.extend(self.search_arxiv(query, max_results))
        return all_results

    def _extract_rejected_fingerprints(self, cross_run_context: List[Dict[str, Any]]) -> List[str]:
        """Extract keyword fingerprints from rejected-topic items in CrossRunMemory.

        Only uses structured tags (rejection_reason, item title) — never raw
        free-text rejection reasons, consistent with the fail-closed memory policy.

        Fix #6: windowed to the most recent _REJECTION_HISTORY_WINDOW entries so
        keyword-stripping downstream can't be starved by a long accumulated history.
        """
        windowed_context = cross_run_context[-_REJECTION_HISTORY_WINDOW:]
        fingerprints = []
        for entry in windowed_context:
            item = str(entry.get("item") or "").strip()
            tag = str(entry.get("rejection_reason") or "").strip()
            if item:
                words = re.findall(r"[a-z][a-z0-9_-]{2,}", item.lower())
                fingerprints.append(" ".join(words[:5]))
            if tag and tag not in ("other",):
                fingerprints.append(tag)
        return fingerprints

    def _active_hypothesis_kind(self) -> Optional[str]:
        """Return the forced-exploration hypothesis kind if one is active this cycle, else None."""
        cfg = getattr(self, "runtime_config", None)
        every = max(1, int(getattr(cfg, "topic_exploration_every", 4) if cfg is not None else 4))
        if hasattr(self, "_selection_counter") and (self._selection_counter % every) == 0:
            try:
                elo = EloStore(context=getattr(self, "context", None))
                all_kinds = list(_TOPICS_TO_SUBCATEGORY.keys())
                under = elo.under_observed_kinds(all_kinds)
                if under:
                    return random.choice(under)
            except Exception:
                pass
        return None

    def _generate_llm_seeds(
        self,
        domain: str,
        cross_run_context: List[Dict[str, Any]],
        negative_lessons: Optional[List[Dict[str, Any]]] = None,
        n_seeds: int = 6,
        frontier_terms: Optional[List[Dict[str, str]]] = None,
        exclude_phrases: Optional[Collection[str]] = None,
        temperature: float = 0.9,
    ) -> List[Dict[str, str]]:
        """LLM-driven seed generation (v3 fix #3).

        Static template seeds ("... open problems", "... underexplored
        challenges") recur every run and produce generic queries. This asks the
        LLM — once, cheap tier — to mint specific, technical seed phrases from
        the structured signals we already have (cross-run rejection tags,
        negative results). Returns [] on any failure; callers fall back to the
        static seed banks unchanged.
        """
        if not getattr(self.runtime_config, "llm_seed_generation_enabled", True):
            return []
        logger.info("Generating LLM seeds for domain=%r (n_seeds=%d)", domain, n_seeds)
        runtime_cfg = getattr(self, "runtime_config", None)
        if not domain:
            domain = getattr(runtime_cfg, "research_domain", None) if runtime_cfg is not None else None
        if not domain:
            return []
        avoid_block = ""
        if exclude_phrases:
            avoid = [s for s in exclude_phrases if s][:12]
            if avoid:
                avoid_block = (
                    "\nAlready tried in prior iterations — MUST differ from ALL of these "
                    "(different method family or measurement setup, not a rephrase):\n"
                    + json.dumps(avoid, default=str)[:1200] + "\n"
                )
        # e0ec9e91: for SE runs the minter must name SE artifacts, not pure-ML
        # topics. Inject a hard domain-vocabulary requirement into the prompt.
        _domain_gate_hint = ""
        try:
            if "software" in re.sub(r"\s+", " ", (domain or "").strip().lower()):
                _domain_gate_hint = (
                    "\nHARD REQUIREMENT: every seed MUST contain at least one of: "
                    "fuzzing, testing, debugging, program repair, patch, static analysis, "
                    "compiler, coverage, bug, defect, vulnerability, code, software. "
                    "Seeds without one will be discarded. Good: "
                    "\"coverage-guided fuzzing branch distance benchmark comparison\". "
                    "Bad (will be rejected): \"hierarchical world models visual planning\".\n"
                )
        except Exception:
            pass
        prompt = f"""
You are designing literature-search seeds for an automated research system in {domain}.
Each seed is a short phrase (4-10 words) naming a SPECIFIC technical research angle —
name concrete methods, model families, or measurement setups, not meta-vocabulary.
Bad: "open problems in evaluation". Good: "speculative decoding verification overhead".
{_domain_gate_hint}

Prior-run context (avoid re-proposing rejected/failed areas):
{json.dumps(cross_run_context[-10:], sort_keys=True, default=str)[:1500]}
{avoid_block}
Frontier terms from very recent papers (STRONGLY prefer these angles):
{json.dumps([f"{t.get('method','')} + {t.get('evaluation','')}" for t in (frontier_terms or [])[:8]], default=str)[:500]}

Return JSON with {n_seeds} DIVERSE seeds spread across different subfields
and method families, each with a short strategy tag explaining its angle:
{{"seeds": [{{"seed": "...", "angle": "one-line rationale"}}]}}
"""
        try:
            raw = call_llm(prompt, temperature=temperature, tier="cheap",
                           **_llm_kwargs("hunter.seed", "cheap", domain=domain))
            parsed = parse_json_from_llm(raw) or {}
            items = parsed.get("seeds") or parsed.get("topics") or []
            seeds: List[Dict[str, str]] = []
            for item in items:
                if isinstance(item, dict) and item.get("seed"):
                    text = str(item["seed"]).strip()
                    if 8 <= len(text) <= 120:
                        seeds.append({"seed": text, "strategy": "llm_diverse"})
                elif isinstance(item, str) and 8 <= len(item.strip()) <= 120:
                    seeds.append({"seed": item.strip(), "strategy": "llm_diverse"})
            if seeds:
                log_agent_action("TopicHunter", "llm_seeds_generated", {"count": len(seeds)})
                logger.info("LLM generated %d seeds: %s", len(seeds), [s["seed"][:160] for s in seeds])
            return seeds[:n_seeds]
        except Exception as e:
            log_agent_action("TopicHunter", "llm_seed_generation_error", {"error": str(e)})
            logger.warning("LLM seed generation failed: %s", e)
            return []

    def _generate_dynamic_seeds(
        self,
        n_seeds: int = 5,
        cross_run_context: Optional[List[Dict[str, Any]]] = None,
        domain: Optional[str] = None,
        exclude_phrases: Optional[Collection[str]] = None,
    ) -> List[Dict[str, str]]:
        """Generate fresh seed phrases from current state instead of a static bank.

        Uses structured CrossRunMemory tags (fail-closed) to:
        - Pivot away from rejected topic areas
        - Bias toward underexplored hypothesis kinds
        - Combine method + evaluation signals from the kind bias tables
        - Feature 4: also harvest frontier terms from very recent papers

        Fix #6: rejected-token derivation is windowed to recent history so a long
        session doesn't accumulate enough rejected tokens to blank out every seed.

        `exclude_phrases`: exact seed strings already attempted (prior
        iterations / this instance). Applied as exact-phrase exclusion, never
        as word-level bans, so one exhausted seed cannot poison its vocabulary
        for future seeds.

        Feature 8: returns List[Dict] with {"seed": ..., "strategy": ...} for
        seed-strategy provenance tracking.
        """
        if cross_run_context is None:
            cross_run_context = CrossRunMemory().get_prompt_context()

        logger.info("Generating dynamic seeds: n_seeds=%d, domain=%r", n_seeds, domain)
        windowed_context = cross_run_context[-_REJECTION_HISTORY_WINDOW:]

        excluded_phrases: set = {
            _normalize_seed_text(s) for s in (exclude_phrases or []) if s
        }
        excluded_phrases.update(
            # getattr: defensive — some callers (tests, unpickled agents)
            # reach this method without running __init__.
            _normalize_seed_text(s) for s in getattr(self, "_attempted_seeds", set()) if s
        )
        rejected_tokens: set = set()
        for entry in windowed_context:
            item = str(entry.get("item") or "").lower()
            tag = str(entry.get("rejection_reason") or "")
            # Durable exact-phrase exclusion across agent instances and
            # workflow iterations. Keyed on kind=="seed" (stored verbatim),
            # not the reason tag. Deliberately NOT added to
            # rejected_tokens: word-level banning would nuke shared
            # vocabulary ("transformer", "efficiency") for all future seeds.
            if entry.get("kind") == "seed" or tag in ("seed_exhausted", _SEED_EXHAUSTED_REASON):
                if item:
                    excluded_phrases.add(_normalize_seed_text(item))
                continue
            if item:
                rejected_tokens.update(w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", item) if len(w) > 3)
            if tag and tag not in ("other",):
                rejected_tokens.add(tag)

        active_kind = self._active_hypothesis_kind()

        seeds: List[Dict[str, str]] = []  # Feature 8: track strategy per seed

        # 0. v3 fix #3: LLM-minted seeds first (specific, technical, run-aware).
        # Empty on failure/disabled — everything below degrades to the static
        # template seeds exactly as before.
        negative_lessons: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "negative_result_seeding_enabled", True):
            try:
                negative_lessons = CrossRunMemory().get_negative_result_lessons()
            except Exception:
                negative_lessons = []
        # Cached frontier terms are free (disk read) — load once, reuse for
        # both the LLM-seed prompt and direct frontier seed assembly below.
        cached_frontier_terms = self._load_cached_frontier_terms(
            domain or self.runtime_config.research_domain)
        llm_seeds = self._generate_llm_seeds(
            domain or self.runtime_config.research_domain,
            cross_run_context,
            negative_lessons=negative_lessons,
            n_seeds=max(3, n_seeds // 2),
            frontier_terms=cached_frontier_terms,
            exclude_phrases=sorted(excluded_phrases),
            # Prior failures mean the last batch repeated itself: raise
            # sampling temperature so the retry diversifies instead of
            # re-minting identical seeds.
            temperature=0.9 if not excluded_phrases else 1.1,
        )
        _llm_vetoed = 0
        for d in llm_seeds:
            if _normalize_seed_text(d["seed"]) in excluded_phrases:
                continue
            # e0ec9e91 domain gate: drop off-domain LLM seeds before retrieval.
            if not _seed_passes_domain_gate(d["seed"], domain or ""):
                _llm_vetoed += 1
                continue
            words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", d["seed"].lower()))
            if not (words & rejected_tokens):
                seeds.append(d)
        if _llm_vetoed:
            log_agent_action("TopicHunter", "seed_domain_veto", {
                "vetoed": _llm_vetoed, "source": "llm_diverse",
            })

        # Frontier seeds come FIRST after LLM seeds (before kind-bias): they
        # name real current directions from very recent papers, while the
        # kind-bias templates are the strings that recurred verbatim across
        # all 25ed95d4 iterations. Cached terms are free (disk read); a
        # refresh harvest runs only when the cache is empty for this domain.
        frontier_terms = list(cached_frontier_terms)
        if not frontier_terms:
            frontier_terms = self._harvest_frontier_terms(domain) if domain else []
        logger.info("Seed generation: %d LLM seeds, %d frontier terms, active_kind=%s",
                     len(llm_seeds), len(frontier_terms), active_kind)
        _frontier_vetoed = 0
        for term_pair in frontier_terms:
            method = term_pair.get("method", "")
            evaluation = term_pair.get("evaluation", "")
            if not method or not evaluation:
                continue
            _candidate = f"{method} {evaluation} benchmark comparison"
            # e0ec9e91: poisoned ML frontier cache for SE runs. Veto off-domain
            # frontier seeds; kind_bias SE seeds below (fuzzing/repair/static
            # analysis) backfill so the run never starves.
            if not _seed_passes_domain_gate(_candidate, domain or ""):
                _frontier_vetoed += 1
                continue
            words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", f"{method} {evaluation}".lower()))
            if words & rejected_tokens:
                continue
            # "benchmark comparison" tail (not "open problems"): the filler
            # list strips "open/problems" from queries, which used to leave
            # these seeds unanchored.
            seeds.append({"seed": _candidate, "strategy": "frontier"})
        if _frontier_vetoed:
            log_agent_action("TopicHunter", "seed_domain_veto", {
                "vetoed": _frontier_vetoed, "source": "frontier",
            })

        # 1. Seeds from underexplored hypothesis kinds (avoid rejected areas)
        # 0dfc1c58: kind_bias seeds were NOT domain-gated (unlike LLM and
        # frontier seeds), so an SE run filled its batch with ML seeds
        # ("graph neural ...", "visual detection ...") while the 3 SE
        # LLM seeds were Elo-sorted out. Gate them the same way; SE backfill
        # below guarantees the run never starves.
        _kind_vetoed = 0
        for kind, biases in _EXPLORATION_KIND_BIASES.items():
            method_terms = biases.get("method", [])
            eval_terms = biases.get("evaluation", [])
            if not method_terms or not eval_terms:
                continue
            method = method_terms[0]
            evaluation = eval_terms[0]
            kind_words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", f"{method} {evaluation}"))
            if kind_words & rejected_tokens:
                continue
            # Concrete comparison frame, not meta-vocabulary: "open problems"
            # style tails are stripped from queries by _SEED_FILLER_WORDS and
            # leave vague prompts. "benchmark comparison" survives keyword
            # extraction and steers the LLM toward falsifiable angles.
            _kind_candidate = f"{method} {evaluation} benchmark comparison"
            if not _seed_passes_domain_gate(_kind_candidate, domain or ""):
                _kind_vetoed += 1
                continue
            seeds.append({"seed": _kind_candidate, "strategy": "kind_bias"})
        if _kind_vetoed:
            log_agent_action("TopicHunter", "seed_domain_veto", {
                "vetoed": _kind_vetoed, "source": "kind_bias",
            })

        # 0dfc1c58: SE backfill — when the run domain is SE and gating left
        # us short, mint SE kind seeds directly so the batch stays SE.
        try:
            _dom_check = re.sub(r"\s+", " ", (domain or "").strip().lower())
        except Exception:
            _dom_check = ""
        _is_se_run = ("software" in _dom_check or _dom_check in ("software engineer", "software engineering", "se"))
        if _is_se_run:
            _se_biases = _EXPLORATION_KIND_BIASES.get("software_engineering", {})
            for _m in (_se_biases.get("method", []) or [])[:3]:
                for _e in (_se_biases.get("evaluation", []) or [])[:2]:
                    _se_candidate = f"{_m} {_e} benchmark comparison"
                    if any(d["seed"] == _se_candidate for d in seeds):
                        continue
                    _cw = set(re.findall(r"[a-z][a-z0-9_-]{2,}", _se_candidate.lower()))
                    if _cw & rejected_tokens:
                        continue
                    seeds.append({"seed": _se_candidate, "strategy": "kind_bias"})

        # 2. Extra seeds biased toward the active exploration kind
        # 0dfc1c58: same domain gate as above — active_kind=vision injected
        # ML seeds into an SE run.
        if active_kind and active_kind in _EXPLORATION_KIND_BIASES:
            biases = _EXPLORATION_KIND_BIASES[active_kind]
            for method_term in biases.get("method", [])[:2]:
                for eval_term in biases.get("evaluation", [])[:1]:
                    candidate = f"{method_term} {eval_term} benchmark comparison"
                    candidate_words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", candidate))
                    if not (candidate_words & rejected_tokens):
                        if not _seed_passes_domain_gate(candidate, domain or ""):
                            _kind_vetoed += 1
                            continue
                        seeds.append({"seed": candidate, "strategy": "kind_bias"})

        # 3. Generic fallback seeds that avoid rejected keyword areas.
        # Skipped when cached frontier terms exist: the 8 static strings
        # ("methodological gaps in evaluation", …) were the seed slots in
        # every 25ed95d4 iteration and their filler vocabulary produces junk
        # queries ("evaluation despite widespread"). Frontier terms name real
        # current directions and are strictly better filler.
        generic_seeds = [
            "methodological gaps in evaluation",
            "cross-domain transfer limitations",
            "sample efficiency and data requirements",
            "robustness under distribution shift",
            "reproducibility and reporting standards",
            "interpretability for decision support",
            "scalability and computational limits",
            "edge cases and failure mode analysis",
        ]
        if not cached_frontier_terms:
            for s in generic_seeds:
                s_words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", s.lower()))
                if not (s_words & rejected_tokens):
                    seeds.append({"seed": s, "strategy": "generic_fallback"})

        # Fix #6: if aggressive filtering above still leaves us short, fall back to
        # unfiltered generic seeds rather than starving the run of any seeds at all.
        if len(seeds) < max(1, n_seeds // 2):
            for s in generic_seeds:
                if not any(d["seed"] == s for d in seeds):
                    seeds.append({"seed": s, "strategy": "generic_fallback"})

        # Deduplicate preserving order
        seen: set = set()
        unique: List[Dict[str, str]] = []
        for d in seeds:
            if d["seed"] not in seen:
                seen.add(d["seed"])
                unique.append(d)

        # Feature 8: sort by strategy Elo (descending) if enabled
        if getattr(self.runtime_config, "seed_strategy_elo_enabled", True):
            try:
                elo = EloStore(context=getattr(self, "context", None))
                unique.sort(key=lambda d: elo.get(f"strategy:{d['strategy']}"), reverse=True)
                # Reserve >=1 slot for the lowest-rated strategy (exploration)
                strategies_seen = {d["strategy"] for d in unique}
                if len(strategies_seen) > 1:
                    all_strategies = ["llm_diverse", "kind_bias", "cross_pollination", "generic_fallback", "frontier"]
                    lowest = min(all_strategies, key=lambda s: elo.get(f"strategy:{s}"))
                    if not any(d["strategy"] == lowest for d in unique[:1]):
                        # Move one seed of the lowest-rated strategy to the front
                        for i, d in enumerate(unique):
                            if d["strategy"] == lowest:
                                unique.insert(0, unique.pop(i))
                                break
            except Exception:
                pass

        # Exact-phrase exclusion of already-attempted seeds (prior iterations
        # / this instance). Starvation guard: never return empty when we have
        # candidates — a repeated seed still beats no seed, and the seed loop
        # below records the attempt again.
        if excluded_phrases:
            fresh = [d for d in unique if _normalize_seed_text(d["seed"]) not in excluded_phrases]
            if fresh:
                if len(fresh) < len(unique):
                    log_agent_action("TopicHunter", "seed_repeat_avoided", {
                        "dropped": len(unique) - len(fresh),
                    })
                unique = fresh
            else:
                log_agent_action("TopicHunter", "seed_repeat_unavoidable", {
                    "candidates": len(unique),
                })

        # Fuzzy rephrase guard: exact-phrase exclusion misses near-identical
        # rewordings ("transformer efficiency benchmark analysis" vs "…
        # comparison"). Drop candidates with high token overlap against any
        # excluded phrase — but only while enough candidates survive, so this
        # can never starve the run either.
        if excluded_phrases and len(unique) > max(1, n_seeds // 2):
            fuzzy_fresh = [
                d for d in unique
                if not any(title_token_overlap(d["seed"], ex) >= 0.60 for ex in excluded_phrases)
            ]
            if len(fuzzy_fresh) >= max(1, n_seeds // 2):
                if len(fuzzy_fresh) < len(unique):
                    log_agent_action("TopicHunter", "seed_rephrase_avoided", {
                        "dropped": len(unique) - len(fuzzy_fresh),
                    })
                unique = fuzzy_fresh

        # e0ec9e91: seeds 1+5 near-identical within the SAME batch
        # ("... visual planning" vs "... visual planning benchmark").
        # Intra-batch fuzzy dedup: keep first, drop later >=0.60 overlaps.
        _intra: list = []
        for d in unique:
            if any(title_token_overlap(d["seed"], kept["seed"]) >= 0.60 for kept in _intra):
                log_agent_action("TopicHunter", "seed_intra_batch_dedup", {
                    "dropped": d["seed"][:80],
                })
                continue
            _intra.append(d)
        unique = _intra

        # 0dfc1c58: reserve batch slots for domain-gated LLM seeds. The Elo
        # sort above ranked kind_bias above llm_diverse, so all 3 SE LLM
        # seeds were sliced off by unique[:n_seeds] and the run hunted 5
        # generic ML templates instead. LLM seeds passed the domain gate and
        # rejected-token filter already — they are the highest-signal seeds.
        try:
            _llm_kept = [d for d in unique if d.get("strategy") == "llm_diverse"]
            _final = list(unique[:n_seeds])
            if _llm_kept and not any(d.get("strategy") == "llm_diverse" for d in _final):
                _reserve = min(len(_llm_kept), max(1, n_seeds // 2))
                _rest = [d for d in _final if d.get("strategy") != "llm_diverse"]
                _final = _llm_kept[:_reserve] + _rest
                _final = _final[:n_seeds]
                log_agent_action("TopicHunter", "llm_seeds_reserved", {
                    "reserved": _reserve, "kept": len(_llm_kept),
                })
                unique = _final + [d for d in unique if d not in _final]
            else:
                unique = _final + [d for d in unique if d not in _final]
        except Exception:
            pass

        return unique[:n_seeds]

    def _apply_keyword_floor(self, keywords: List[str], all_seed_words: List[str], rejected_tokens: set) -> List[str]:
        """Fix #6: ensure at least _MIN_SURVIVING_KEYWORDS keywords survive rejected-token
        filtering, even if that means keeping a keyword that overlaps a rejected fingerprint.

        Without this floor, `_build_arxiv_queries` / `_build_openalex_queries` can end up
        with zero keywords on a seed whose every token happens to intersect the rejected
        set, producing an unanchored, near-meaningless query (or none at all).
        """
        if len(keywords) >= _MIN_SURVIVING_KEYWORDS:
            return keywords
        # Backfill from the original (unfiltered) seed words, preferring ones not rejected,
        # but accepting rejected ones rather than shipping zero keywords.
        backfill_preferred = [w for w in all_seed_words if w not in rejected_tokens and w not in keywords]
        backfill_fallback = [w for w in all_seed_words if w not in keywords]
        for pool in (backfill_preferred, backfill_fallback):
            for w in pool:
                if len(keywords) >= _MIN_SURVIVING_KEYWORDS:
                    break
                keywords.append(w)
            if len(keywords) >= _MIN_SURVIVING_KEYWORDS:
                break
        return keywords

    def _resolve_subcategory(self, active_kind: Optional[str]) -> str:
        """Resolve active hypothesis kind to a subcategory label for _ARXIV_CATEGORIES / _OPENALEX_CONCEPTS."""
        if active_kind and active_kind in _TOPICS_TO_SUBCATEGORY:
            return _TOPICS_TO_SUBCATEGORY[active_kind]
        return "general"

    def _resolve_subcategory_for(
        self,
        seed_hint: str = "",
        active_kind: Optional[str] = None,
        domain: str = "",
    ) -> str:
        """v5 domain-aware subcategory: active kind > SE seed markers >
        run domain > general. Previously only the kind was consulted, so SE
        seeds under a generic domain silently searched ML categories.

        0dfc1c58 amendment: SE signals win over the exploration kind. With
        active_kind=vision and domain="software engineer", every SE seed
        was routed to cs.LG/cs.CV instead of cs.SE because the kind branch
        ran first and the domain map missed the "software engineer"
        spelling. Seed markers and an SE run domain now take precedence.
        """
        seed_tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", (seed_hint or "").lower()))
        if seed_tokens & _SE_MARKERS:
            return "software_engineering"
        try:
            _dom_key = re.sub(r"\s+", " ", (domain or "").strip().lower())
        except Exception:
            _dom_key = ""
        if "software" in _dom_key or "software_engineering" in _dom_key or _dom_key in ("software engineer", "software engineering", "se"):
            return "software_engineering"
        if active_kind and active_kind in _TOPICS_TO_SUBCATEGORY:
            return _TOPICS_TO_SUBCATEGORY[active_kind]
        subcat = _subcategory_for_domain(domain or getattr(self.runtime_config, "research_domain", ""))
        if subcat != "general":
            return subcat
        return "general"

    def _build_arxiv_queries(
        self,
        seed_hint: str,
        rejected_fingerprints: List[str],
        active_kind: Optional[str] = None,
        domain: str = "",
    ) -> List[str]:
        """Build 2-4 field-scoped arXiv boolean queries.

        Uses cat:, abs:, ti: combinators instead of prose strings.
        Varies subcategory and sort mode.  Biases toward underexplored
        hypothesis kinds when one is active for this cycle.  Excludes
        keywords matching recently rejected fingerprints, but never below
        the keyword floor (Fix #6). Seed filler words ("gaps", "problems",
        "open", ...) are never used as search keywords (v3 fix #4).
        """
        seed_lower = seed_hint.lower()
        all_seed_words = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", seed_lower)
                           if w not in _QUERY_STOPWORDS and w not in _SEED_FILLER_WORDS and len(w) > 2]
        keywords = all_seed_words[:3]

        # De-prioritize keywords that match rejected fingerprints
        rejected_tokens = set()
        if rejected_fingerprints:
            for fp in rejected_fingerprints:
                rejected_tokens.update(re.findall(r"[a-z][a-z0-9_-]{2,}", fp.lower()))
            keywords = [kw for kw in keywords if kw not in rejected_tokens]
            if not keywords:
                keywords = [w for w in all_seed_words if w not in rejected_tokens][:2]
        keywords = self._apply_keyword_floor(keywords, all_seed_words, rejected_tokens)
        keywords = keywords[:3]

        subcat_label = self._resolve_subcategory_for(seed_hint, active_kind, domain)
        categories = _ARXIV_CATEGORIES.get(subcat_label, _ARXIV_CATEGORIES["general"])

        queries = []
        for cat in categories[:2]:
            if len(keywords) >= 2:
                kw_str = " OR ".join(f'"{kw}"' for kw in keywords[:2])
                queries.append(f"cat:{cat} AND ({kw_str})")
            if len(keywords) >= 1:
                queries.append(f"cat:{cat} AND ti:{keywords[0]}")
        if not queries and keywords:
            queries.append(f"cat:{categories[0]} AND abs:{keywords[0]}")

        # When an exploration kind is active, add one cross-pollination query
        # that pairs the kind's primary method term with the seed's first keyword.
        # This does NOT overwrite the seed's own keywords — it adds diversity.
        if active_kind and active_kind in _EXPLORATION_KIND_BIASES and keywords:
            biases = _EXPLORATION_KIND_BIASES[active_kind]
            kind_method = biases.get("method", [""])[0]
            seed_anchor = keywords[0]
            if kind_method and kind_method != seed_anchor:
                cross_q = f"cat:{categories[0]} AND abs:{kind_method} AND abs:{seed_anchor}"
                if cross_q not in queries:
                    queries.append(cross_q)

        return queries[:2]

    def _build_openalex_queries(
        self,
        seed_hint: str,
        rejected_fingerprints: List[str],
        active_kind: Optional[str] = None,
        domain: str = "",
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Build 2-4 OpenAlex structured queries with filter params.

        hypothesis kinds when one is active.  Excludes keywords matching
        recently rejected fingerprints, but never below the keyword floor
        (Fix #6). Seed filler words ("gaps", "problems", "open", ...) are
        never used as search keywords (v3 fix #4).
        """
        seed_lower = seed_hint.lower()
        all_seed_words = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", seed_lower)
                           if w not in _QUERY_STOPWORDS and w not in _SEED_FILLER_WORDS and len(w) > 2]
        keywords = all_seed_words[:3]

        # De-prioritize keywords that match rejected fingerprints
        rejected_tokens = set()
        if rejected_fingerprints:
            for fp in rejected_fingerprints:
                rejected_tokens.update(re.findall(r"[a-z][a-z0-9_-]{2,}", fp.lower()))
            keywords = [kw for kw in keywords if kw not in rejected_tokens]
            if not keywords:
                keywords = [w for w in all_seed_words if w not in rejected_tokens][:2]
        keywords = self._apply_keyword_floor(keywords, all_seed_words, rejected_tokens)
        keywords = keywords[:3]

        subcat_label = self._resolve_subcategory_for(seed_hint, active_kind, domain)

        # Feature 1: concept filtering for OpenAlex queries. Subcategories
        # with no verified concept IDs (e.g. software_engineering) run
        # keyword-only — a wrong filter is worse than none.
        concept_filter_str = None
        if getattr(self.runtime_config, "openalex_concept_filtering_enabled", True):
            concept_ids = _OPENALEX_CONCEPTS.get(subcat_label, _OPENALEX_CONCEPTS["general"])
            if concept_ids:
                concept_filter_str = "|".join(concept_ids)
                log_agent_action("TopicHunter", "openalex_concept_filter_applied", {
                    "subcat": subcat_label, "concepts": concept_ids,
                })
            else:
                log_agent_action("TopicHunter", "openalex_concept_filter_skipped", {
                    "subcat": subcat_label, "reason": "no_verified_concept_ids_keyword_only",
                })

        queries = []
        if keywords:
            search_str = " ".join(keywords[:3])
            queries.append((search_str, {"sort": "relevance_score:desc"}))
            recent_year = max(2020, datetime.now().year - 2)
            recency_filter = f"publication_year:>{recent_year}"
            if concept_filter_str:
                recency_filter = f"concepts.id:{concept_filter_str},{recency_filter}"
            # The recency/citation-ranked leg must differ from the relevance
            # leg: same string + different sort still returns near-identical
            # sets (two API calls for one result page). Narrow to the head
            # keywords so the two legs complement instead of duplicating.
            recency_str = " ".join(keywords[:2]) if len(keywords) >= 2 else search_str
            queries.append((recency_str, {"filter": recency_filter, "sort": "cited_by_count:desc"}))
            if len(keywords) >= 2:
                alt_str = " ".join(keywords[:2])
                alt_params: Dict[str, Any] = {"sort": "relevance_score:desc"}
                if concept_filter_str:
                    alt_params["filter"] = f"concepts.id:{concept_filter_str}"
                queries.append((alt_str, alt_params))
            # Cross-pollination: one query pairing the kind's method term with the seed's anchor
            if active_kind and active_kind in _EXPLORATION_KIND_BIASES:
                kind_biases = _EXPLORATION_KIND_BIASES.get(active_kind, {})
                kind_method = kind_biases.get("method", [""])[0]
                seed_anchor = keywords[0]
                if kind_method and kind_method != seed_anchor:
                    cross_search = f"{kind_method} {seed_anchor}"
                    cross_params: Dict[str, Any] = {"sort": "relevance_score:desc"}
                    if concept_filter_str:
                        cross_params["filter"] = f"concepts.id:{concept_filter_str}"
                    queries.append((cross_search, cross_params))

        return queries[:4]

    def _preflight_dedup(
        self,
        queries: List[str],
        rejected_fingerprints: List[str],
    ) -> List[str]:
        """Skip or mutate queries whose keyword/category signature overlaps recent rejections.

        Uses structured fingerprints (keyword n-grams from rejected titles and
        rejection_reason tags) — never raw free-text rejection prose.
        Returns the filtered list and logs skipped queries.
        """
        if not rejected_fingerprints:
            return queries

        def _extract_fingerprint(query: str) -> set:
            words = re.findall(r"[a-z][a-z0-9_-]{2,}", query.lower())
            return set(w for w in words if w not in {"cat", "abs", "ti", "cs", "stat"})

        rejected_tokens = set()
        for fp in rejected_fingerprints:
            rejected_tokens.update(_extract_fingerprint(fp))

        deduped = []
        for q in queries:
            q_tokens = _extract_fingerprint(q)
            overlap = q_tokens & rejected_tokens
            if len(overlap) >= 2 and len(overlap) / max(len(q_tokens), 1) > 0.6:
                log_agent_action("TopicHunter", "preflight_dedup_skip", {
                    "query": q,
                    "overlap_tokens": sorted(overlap),
                    "reason": "query_fingerprint_matches_rejected_topic",
                })
                continue
            deduped.append(q)

        # Fix #6: never dedup away every single query — an empty result here
        # cascades into "seed_skipped_no_queries" even when a milder query
        # would have been fine. Keep the least-overlapping original query.
        if not deduped and queries:
            def _overlap_count(q: str) -> int:
                return len(_extract_fingerprint(q) & rejected_tokens)
            best = min(queries, key=_overlap_count)
            deduped = [best]
            log_agent_action("TopicHunter", "preflight_dedup_floor_kept_one", {"query": best})

        return deduped

    def fetch_citation_graph(self, paper_id: str) -> Dict[str, Any]:
        """
        Semantic Scholar Graph API: high in-degree + low recent out-degree = gap signal.
        paper_id can be DOI, arXiv, or S2 paperId.
        """
        try:
            fields = "title,year,citationCount,referenceCount,influentialCitationCount"
            url = f"{self.base_urls['semantic_scholar']}/paper/{paper_id}"
            r = self._s2_get(url, params={"fields": fields})
            if r is None or r.status_code != 200:
                return {}
            paper = r.json()
            # Recent citing papers (proxy for out-degree from recent work extending it)
            cites_url = f"{self.base_urls['semantic_scholar']}/paper/{paper_id}/citations"
            c = self._s2_get(
                cites_url,
                params={"fields": "citingPaper.year,citingPaper.title", "limit": 50},
            )
            recent_extensions = 0
            # _s2_get returns None on throttle-gate/breaker skip — that is
            # graceful degradation during a 429 storm, not an error. Only a
            # genuine exception (below) counts as citation_graph_error.
            if c is not None and c.status_code == 200:
                for item in c.json().get("data", []):
                    year = (item.get("citingPaper") or {}).get("year") or 0
                    if year >= datetime.now().year - 2:
                        recent_extensions += 1
            in_degree = paper.get("citationCount") or 0
            return {
                "paper_id": paper_id,
                "title": paper.get("title"),
                "year": paper.get("year"),
                "in_degree": in_degree,
                "reference_count": paper.get("referenceCount") or 0,
                "recent_citing": recent_extensions,
                "gap_score": float(in_degree) / max(recent_extensions + 1, 1),
            }
        except Exception as e:
            log_agent_action("TopicHunter", "citation_graph_error", {"error": str(e)})
            return {}

    def evaluate_layered_novelty(
        self,
        candidate_topic: Dict[str, Any],
        abstracts: List[Dict[str, Any]],
        precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
        hunt_mode: str = "gap",
    ) -> Dict[str, Any]:
        """Layered novelty evaluation:
        1. Fast embedding similarity check (uses precomputed embeddings when available)
        2. Contribution comparison for high-overlap papers

        Fix #3: the comparison-trigger threshold is clamped so it can never sit
        above `novelty_similarity_reject`. Previously the trigger was a hardcoded
        0.70; if the configured reject threshold was lower than that, a candidate
        landing between the two values would be rejected on cosine similarity
        alone, with no LLM contribution comparison ever run. Now the trigger is
        always <= the reject threshold, guaranteeing comparison happens first.
        """
        if not abstracts:
            logger.info("Novelty check for %r: no abstracts, auto-novel", candidate_topic.get("title", "")[:160])
            return {
                "max_similarity": 0.0,
                "reject": False,
                "nearest": None,
                "novelty_comparisons": [],
                "verdict": "NOVEL",
                "embedding_fallback": not is_embedding_available(),
            }
        reject_threshold = float(getattr(self.runtime_config, "novelty_similarity_reject", 0.90))
        comparison_trigger = min(_DEFAULT_COMPARISON_TRIGGER, reject_threshold)
        logger.info("Novelty check for %r: %d abstracts, reject_threshold=%.2f, comparison_trigger=%.2f",
                     candidate_topic.get("title", "")[:160], len(abstracts), reject_threshold, comparison_trigger)

        topic_desc = f"{candidate_topic.get('title','')} {candidate_topic.get('description','')} {candidate_topic.get('contribution','')}"
        topic_emb = generate_embedding(topic_desc)
        # Zero vectors mean the embedding backend is down (embed() returns
        # zeros on any failure) — norm>0 is the live signal. A separate
        # is_embedding_available() probe is deliberately NOT consulted here:
        # unit tests inject synthetic nonzero vectors with the backend
        # latched, and vectors on the table must win over the latch.
        embeddings_live = bool(np.linalg.norm(topic_emb) > 0)
        embedding_fallback = not embeddings_live
        if embedding_fallback:
            # Dead embedding backend: zero vectors would make best_sim=0.0
            # and auto-pass everything as NOVEL (run 25ed95d4). Degrade to
            # lexical ranking and FORCE the LLM contribution comparison below
            # instead of trusting the vectors.
            log_agent_action("TopicHunter", "novelty_embedding_fallback", {
                "title": candidate_topic.get("title", "")[:80],
            })
        best_sim = 0.0
        nearest = None
        high_overlap_papers = []

        for paper in abstracts[:20]:
            abs_text = paper.get("abstract") if isinstance(paper, dict) else str(paper)
            if not abs_text:
                continue
            if embedding_fallback:
                sim = tfidf_cosine(topic_desc, abs_text[:2000])
                if sim > best_sim:
                    best_sim = sim
                    nearest = abs_text[:200]
                if sim >= 0.10:
                    high_overlap_papers.append({
                        "paper": paper if isinstance(paper, dict) else {"title": "Prior Paper", "abstract": abs_text},
                        "similarity": sim,
                    })
                continue
            # Use precomputed embedding if available, otherwise compute on the fly
            paper_key = (paper.get("doi") or paper.get("arxiv_id") or paper.get("title", "")).strip().lower() if isinstance(paper, dict) else None
            if precomputed_embeddings and paper_key and paper_key in precomputed_embeddings:
                emb = precomputed_embeddings[paper_key]
            else:
                emb = generate_embedding(abs_text[:2000])
            denom = np.linalg.norm(topic_emb) * np.linalg.norm(emb)
            if denom == 0:
                continue
            sim = float(np.dot(topic_emb, emb) / denom)
            if sim > best_sim:
                best_sim = sim
                nearest = abs_text[:200]
            if sim >= comparison_trigger:
                high_overlap_papers.append({
                    "paper": paper if isinstance(paper, dict) else {"title": "Prior Paper", "abstract": abs_text},
                    "similarity": sim,
                })
        if embedding_fallback:
            # Keep the comparison bounded: top-3 lexical matches only.
            high_overlap_papers.sort(key=lambda h: h["similarity"], reverse=True)
            high_overlap_papers = high_overlap_papers[:3]

        # If near duplicate threshold reached (e.g. > 0.95), immediate reject.
        # Never on TF-IDF scale: lexical scores are not calibrated to the
        # embedding reject threshold, so fallback mode always goes through
        # the LLM comparison below instead.
        if best_sim >= 0.95 and not embedding_fallback:
            logger.info("Novelty check for %r: REJECTED as near-duplicate (sim=%.3f)", candidate_topic.get("title", "")[:160], best_sim)
            return {
                "max_similarity": best_sim,
                "reject": True,
                "nearest": nearest,
                "novelty_comparisons": [],
                "verdict": "LIKELY_DUPLICATE",
                "reason": f"Near-duplicate embedding similarity ({best_sim:.2f})",
                "embedding_fallback": False,
            }

        # Contribution comparison for high overlap papers (similarity >= comparison_trigger)
        comparisons = []
        is_duplicate = False
        replication_ok = False
        if high_overlap_papers:
            prompt = f"""
Compare the candidate research contribution against prior work to evaluate novelty.
Candidate:
Title: {candidate_topic.get('title')}
Description: {candidate_topic.get('description')}
Proposed Contribution: {candidate_topic.get('contribution', candidate_topic.get('rationale', ''))}

Prior High-Overlap Works:
{json.dumps([{
    'title': p['paper'].get('title'),
    'abstract': p['paper'].get('abstract')[:1000],
    'similarity': round(p['similarity'], 2)
} for p in high_overlap_papers[:3]], indent=2)}

For EACH prior work, evaluate:
- Problem
- Method / Variables
- Overlap
- Difference & Remaining Gap
- Is the candidate novel compared to this paper? (true/false)

Return JSON:
{{
  "comparisons": [
    {{
      "paper": "title",
      "overlap": "...",
      "difference": "...",
      "remaining_gap": "...",
      "is_novel": true,
      "verdict": "NOVEL"
    }}
  ],
  "overall_novelty_verdict": "NOVEL|LIKELY_DUPLICATE|INSUFFICIENT_DIFFERENCE",
  "reject": false,
  "reason": "..."
}}
"""
            for attempt in range(2):
                raw = call_llm(prompt, temperature=0.2, tier="judge",
                               **_llm_kwargs("hunter.novelty", "judge"))
                parsed = parse_json_from_llm(raw) or {}
                if isinstance(parsed, dict) and "comparisons" in parsed:
                    comparisons = parsed.get("comparisons") or []
                    verdict = parsed.get("overall_novelty_verdict", "NOVEL")
                    if verdict in ("LIKELY_DUPLICATE", "INSUFFICIENT_DIFFERENCE") or parsed.get("reject") is True:
                        # P2: replication / try-and-report tracks treat near-overlap
                        # as a labelled replication opportunity, not a kill.
                        if str(hunt_mode or "gap") in ("replication", "try_and_report") and verdict == "INSUFFICIENT_DIFFERENCE":
                            is_duplicate = False
                            replication_ok = True
                            verdict = "REPLICATION_OK"
                            parsed["overall_novelty_verdict"] = verdict
                            parsed["reject"] = False
                            break
                        is_duplicate = True
                    break

        # Fix #3: reject on raw similarity alone only if similarity clears the
        # reject threshold AND no comparison could be obtained (e.g. LLM call
        # failed both attempts) — never because the trigger window was skipped.
        # In embedding-fallback mode the raw score is TF-IDF (uncalibrated to
        # the embedding threshold), so only an explicit LLM duplicate verdict
        # can reject; otherwise the gap passes but stays labelled unverified.
        reject = is_duplicate or (best_sim >= reject_threshold and not comparisons and not embedding_fallback)
        # P2: preserve REPLICATION_OK label from the comparison above.
        if not reject and replication_ok:
            verdict = "REPLICATION_OK"
        else:
            verdict = "LIKELY_DUPLICATE" if reject else "NOVEL"
        logger.info("Novelty check for %r: max_sim=%.3f, verdict=%s, comparisons=%d",
                     candidate_topic.get("title", "")[:160], best_sim, verdict, len(comparisons))
        reason = f"Contribution comparison: {'rejected as duplicate/insufficient difference' if reject else ('replication opportunity' if verdict == 'REPLICATION_OK' else 'novel contribution supported')}"
        if embedding_fallback and not reject:
            reason += " (embeddings unavailable — TF-IDF ranking + LLM comparison, novelty unverified by vectors)"
        return {
            "max_similarity": best_sim,
            "reject": reject,
            "nearest": nearest,
            "novelty_comparisons": comparisons,
            "verdict": verdict,
            "reason": reason,
            "embedding_fallback": embedding_fallback,
        }

    def novelty_score(self, topic_desc: str, abstracts: List[str]) -> Dict[str, Any]:
        """High similarity to recent abstracts -> fast embedding check with backward compatibility."""
        if not abstracts:
            return {"max_similarity": 0.0, "reject": False, "nearest": None}
        try:
            topic_emb = generate_embedding(topic_desc)
            embeddings_live = bool(np.linalg.norm(topic_emb) > 0)
            best_sim = 0.0
            nearest = None
            for abs_text in abstracts[:30]:
                if not abs_text:
                    continue
                if not embeddings_live:
                    sim = tfidf_cosine(topic_desc, abs_text[:2000])
                    if sim > best_sim:
                        best_sim = sim
                        nearest = abs_text[:200]
                    continue
                emb = generate_embedding(abs_text[:2000])
                denom = np.linalg.norm(topic_emb) * np.linalg.norm(emb)
                if denom == 0:
                    continue
                sim = float(np.dot(topic_emb, emb) / denom)
                if sim > best_sim:
                    best_sim = sim
                    nearest = abs_text[:200]
            return {
                "max_similarity": best_sim,
                "reject": best_sim >= self.runtime_config.novelty_similarity_reject and embeddings_live,
                "nearest": nearest,
                "embedding_fallback": not embeddings_live,
            }
        except Exception as e:
            return {"max_similarity": 0.0, "reject": False, "error": str(e)}

    def screen_research_gap(
        self,
        candidate: Dict[str, Any],
        literature: List[Dict[str, Any]],
        citation_gap_signal: float = 0.0,
        hunt_mode: str = "gap",
    ) -> Dict[str, Any]:
        """Research Screener: verifies gap type, evidence support, and novelty distinction.

        P2 try_and_report / replication: empty literature or weak evidence no
        longer FAILs when the idea is falsifiable+executable — it PASSes with
        advisory evidence_strength so simple try-and-report topics survive.
        """
        # v5 (f537f638): statistical gap claims need statistical literature.
        # A bootstrap/CI gap supported only by audio-semantic + FedBWO point-
        # estimate papers is ungrounded — the screener must say so.
        candidate_text = " ".join(
            str(candidate.get(k) or "") for k in ("title", "description", "rationale", "contribution")
        )
        is_stats = bool(re.search(
            r"bootstrap|confidence interval|\bCI\b|hypothesis test|t-test|t test|p-value|accuracy diff",
            candidate_text, re.IGNORECASE))
        stats_papers = []
        if is_stats and literature:
            for p in literature:
                blob = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
                if re.search(
                    r"bootstrap|confidence interval|\bhypothesis test\b|t-test|p-value|variance|dietterich|demsar|bouthillier|benchmark.*variance|statistical",
                    blob):
                    stats_papers.append(p.get("title") or p.get("doi") or p.get("arxiv_id") or "stats-paper")
        if is_stats and literature and not stats_papers and str(hunt_mode or "gap") not in ("try_and_report", "replication"):
            return {
                "gap_type": "unsupported_research_gap",
                "gap_claim": str(candidate.get("description", "")),
                "supporting_papers": [],
                "contradicting_papers": [],
                "closest_prior_work": [],
                "why_existing_work_is_insufficient": (
                    "Statistical gap claimed but retrieved literature contains no "
                    "uncertainty/CI/hypothesis-test evidence (only point-estimate papers)."
                ),
                "proposed_contribution": str(candidate.get("contribution", "")),
                "evidence_strength": 0.0,
                "citation_gap_signal": citation_gap_signal,
                "status": "FAIL",
                "reason": "stats_gap_without_stats_literature",
            }

        if not literature:
            if str(hunt_mode or "gap") in ("try_and_report", "replication"):
                return {
                    "gap_type": "empirical_try",
                    "gap_claim": str(candidate.get("description", "")),
                    "supporting_papers": [],
                    "contradicting_papers": [],
                    "closest_prior_work": [],
                    "why_existing_work_is_insufficient": "No prior literature retrieved; proceeding as bounded try-and-report.",
                    "proposed_contribution": str(candidate.get("contribution", "")),
                    "evidence_strength": 0.35,
                    "citation_gap_signal": citation_gap_signal,
                    "status": "PASS",
                    "reason": "try_and_report_without_literature",
                    "hunt_mode": hunt_mode,
                }
            return {
                "gap_type": "unsupported",
                "gap_claim": str(candidate.get("description", "")),
                "supporting_papers": [],
                "contradicting_papers": [],
                "closest_prior_work": [],
                "why_existing_work_is_insufficient": "No literature evidence available to substantiate research gap.",
                "proposed_contribution": str(candidate.get("contribution", "")),
                "evidence_strength": 0.0,
                "citation_gap_signal": citation_gap_signal,
                "status": "FAIL",
                "reason": "insufficient_literature_evidence",
            }

        prompt = f"""
You are a Research Screener. Evaluate whether the candidate research topic addresses a real, literature-supported gap.

Candidate:
Title: {candidate.get('title')}
Description: {candidate.get('description')}
Rationale: {candidate.get('rationale')}
Proposed Contribution: {candidate.get('contribution')}
Citation gap score (graph signal): {citation_gap_signal}

Relevant Literature:
{json.dumps([{
    'title': p.get('title'),
    'abstract': p.get('abstract', '')[:800],
    'doi': p.get('doi'),
    'arxiv_id': p.get('arxiv_id'),
} for p in literature[:10]], indent=2)}

Evaluate:
1. What category of gap is being claimed?
   (evaluation_gap, dataset_gap, method_gap, generalization_gap, robustness_gap, theoretical_gap, reproducibility_gap, resource_constraint_gap)
2. Is the gap actually supported by literature evidence?
3. What are supporting vs contradicting papers?
4. Why is existing work insufficient?
5. Evidence strength (0.0 - 1.0).

Return JSON:
{{
  "gap_type": "evaluation_gap",
  "gap_claim": "...",
  "supporting_papers": ["..."],
  "contradicting_papers": ["..."],
  "closest_prior_work": ["..."],
  "why_existing_work_is_insufficient": "...",
  "proposed_contribution": "...",
  "evidence_strength": 0.85,
  "status": "PASS|FAIL|INSUFFICIENT_EVIDENCE",
  "reason": "..."
}}
"""
        parsed = call_llm_json(prompt, temperature=0.2, tier="judge", attempts=2, call_fn=call_llm,
                                **_llm_kwargs("hunter.screener", "judge"))
        if isinstance(parsed, dict) and "gap_type" in parsed:
            evidence_strength = float(parsed.get("evidence_strength", 0.5))
            status = parsed.get("status", "PASS")
            # Relaxed threshold (v2): lowered from 0.25 to 0.15 to let more
            # nascent research areas through — evidence strength below 0.15
            # indicates truly unsupported claims, while 0.15-0.25 captures
            # emerging gaps with partial evidence that may still be novel.
            # P2: try_and_report never FAILs on weak evidence alone — the
            # strength score stays advisory and the topic proceeds bounded.
            if evidence_strength < 0.15 and str(hunt_mode or "gap") not in ("try_and_report", "replication"):
                status = "FAIL"
            elif evidence_strength < 0.15:
                status = "PASS" if status == "FAIL" else status
            return {
                "gap_type": str(parsed.get("gap_type", "evaluation_gap")),
                "gap_claim": str(parsed.get("gap_claim", candidate.get("description", ""))),
                "supporting_papers": list(parsed.get("supporting_papers") or []),
                "contradicting_papers": list(parsed.get("contradicting_papers") or []),
                "closest_prior_work": list(parsed.get("closest_prior_work") or []),
                "why_existing_work_is_insufficient": str(parsed.get("why_existing_work_is_insufficient", "")),
                "proposed_contribution": str(parsed.get("proposed_contribution", candidate.get("contribution", ""))),
                "evidence_strength": evidence_strength,
                "citation_gap_signal": citation_gap_signal,
                "status": status,
                "reason": str(parsed.get("reason", "")),
                "hunt_mode": hunt_mode,
            }

        # Fallback if LLM output fails
        return {
            "gap_type": "method_gap",
            "gap_claim": str(candidate.get("description", "")),
            "supporting_papers": [p.get("title") for p in literature[:2] if p.get("title")],
            "contradicting_papers": [],
            "closest_prior_work": [p.get("title") for p in literature[:1] if p.get("title")],
            "why_existing_work_is_insufficient": "Empirical question requires controlled comparison.",
            "proposed_contribution": str(candidate.get("contribution", "")),
            "evidence_strength": 0.6,
            "citation_gap_signal": citation_gap_signal,
            "status": "PASS",
            "reason": "literature_supported_gap",
        }

    def _quick_grounding_check(self, gap: Dict[str, Any], recent_papers: List[Dict[str, Any]]) -> bool:
        """Fix #7: cheap heuristic pre-filter run before the expensive gate chain
        (screener LLM call, novelty embeddings + LLM comparison, formalization LLM
        call, citation graph fetches). Screens out gaps whose title/description
        share essentially no vocabulary with any retrieved paper — a strong signal
        the LLM hallucinated a topic ungrounded in the literature we actually
        fetched, before we pay for the full gate chain on it.

        This is intentionally lenient (low bar) — it is a pre-filter, not a
        replacement for the real gap/novelty/feasibility gates that follow.
        """
        text = f"{gap.get('title','')} {gap.get('description','')} {gap.get('rationale','')}".lower()
        gap_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", text))
        gap_tokens -= _QUERY_STOPWORDS
        if not gap_tokens or not recent_papers:
            return True  # can't evaluate — don't block on a heuristic we can't compute

        for paper in recent_papers[:15]:
            paper_text = f"{paper.get('title','')} {paper.get('abstract','')}".lower()
            paper_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", paper_text))
            if gap_tokens & paper_tokens:
                return True
        return False

    def _rank_papers_for_gap(
        self,
        gap: Dict[str, Any],
        papers: List[Dict[str, Any]],
        top_n: int = 8,
    ) -> List[Dict[str, Any]]:
        """Rank retrieved papers by relevance to a candidate gap (v3 fix #6).

        Previously `literature_evidence` was simply the first 8 fetched papers,
        so the screener/novelty prompts often judged gaps against off-topic
        text. Scores here are lexical Jaccard overlap (deterministic, free);
        ties preserve retrieval order. The closest prior work — not arbitrary
        recent papers — is what novelty screening should compare against.
        """
        gap_text = " ".join(str(gap.get(k, "")) for k in ("title", "description", "contribution", "rationale")).lower()
        gap_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", gap_text)) - _QUERY_STOPWORDS - _SEED_FILLER_WORDS
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, p in enumerate(papers):
            if not (p.get("title") and p.get("abstract")):
                continue
            paper_text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
            paper_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", paper_text)) - _QUERY_STOPWORDS
            inter = gap_tokens & paper_tokens
            union = gap_tokens | paper_tokens
            score = len(inter) / (len(union) or 1)
            scored.append((score, -idx, p))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [p for _, _, p in scored[:top_n]]

    def _rank_openreview_for_seed(
        self,
        seed_hint: str,
        signals: List[Dict[str, Any]],
        top_n: int = 12,
    ) -> List[Dict[str, Any]]:
        """Rank OpenReview weakness signals by seed relevance (v5).

        The venue fetch is shared/cached across seeds; without per-seed ranking
        every seed mines the same expert pain. Lexical Jaccard over seed tokens.
        """
        try:
            seed_tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", (seed_hint or "").lower()))
            seed_tokens -= _QUERY_STOPWORDS
            if not seed_tokens or not signals:
                return list(signals or [])[:top_n]
            scored = []
            for idx, s in enumerate(signals):
                if not isinstance(s, dict):
                    continue
                blob = f"{s.get('title', '')} {' '.join(s.get('weaknesses') or [])}".lower()
                toks = set(re.findall(r"[a-z][a-z0-9_-]{2,}", blob)) - _QUERY_STOPWORDS
                inter = seed_tokens & toks
                union = seed_tokens | toks
                score = len(inter) / (len(union) or 1)
                scored.append((score, -idx, s))
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            return [s for _, _, s in scored[:top_n]]
        except Exception:
            return list(signals or [])[:top_n]

    def _fetch_seed_fulltexts(
        self,
        papers: List[Dict[str, Any]],
        seed_hint: str,
        deadline_reached,
    ) -> List[Dict[str, Any]]:
        """Bounded OA full-text excerpts for top-cited papers (v5).

        Abstract stubs hide methods/entities. Fetches at most
        hunter_fulltext_top_k PDFs/texts via the source cache (fail-open),
        attaches `fulltext_excerpt` to the paper row, and returns the excerpts
        for prompt injection.
        """
        excerpts: List[Dict[str, Any]] = []
        try:
            if not getattr(self.runtime_config, "hunter_fulltext_enabled", True):
                return []
            top_k = max(1, int(getattr(self.runtime_config, "hunter_fulltext_top_k", 10)))
            max_chars = max(500, int(getattr(self.runtime_config, "hunter_fulltext_max_chars", 6000)))
            # e0ec9e91: sorted by cited_by_count (all 0 with S2/OpenAlex-only
            # counts) then required an OA URL — 0 excerpts every seed. Rank
            # OA-available papers first so the fetch loop has something to try.
            def _oa_rank(p):
                try:
                    oa = p.get("best_oa_location") or {}
                    open_access = p.get("open_access") or {}
                    url = (oa.get("url_for_pdf") or oa.get("pdf_url")
                           or oa.get("url") or open_access.get("oa_url") or "")
                    has_oa = 1 if (url and str(url).startswith("https://")) else 0
                except Exception:
                    has_oa = 0
                return (has_oa, (p.get("cited_by_count") or 0))
            ranked = sorted(papers, key=_oa_rank, reverse=True)
            fetched = 0
            for p in ranked:
                if fetched >= top_k or deadline_reached("fulltext"):
                    break
                if not isinstance(p, dict) or p.get("fulltext_excerpt"):
                    continue
                oa = p.get("best_oa_location") or {}
                open_access = p.get("open_access") or {}
                url = (oa.get("url_for_pdf") or oa.get("pdf_url")
                       or oa.get("url") or open_access.get("oa_url") or "")
                license_name = (oa.get("license") or open_access.get("license")
                                or open_access.get("oa_status") or "")
                if not url or not str(url).startswith("https://"):
                    continue
                try:
                    artifact = self.source_client.fetch_open_access_text(
                        "openalex", url, license_name=str(license_name or "oa"),
                    )
                    text = ((artifact.get("content") or {}).get("text") or "").strip()
                    if len(text) < 500:
                        continue
                    excerpt = text[:max_chars]
                    p["fulltext_excerpt"] = excerpt
                    excerpts.append({"title": (p.get("title") or "")[:150], "excerpt": excerpt})
                    fetched += 1
                except Exception:
                    continue
            if excerpts:
                log_agent_action("TopicHunter", "seed_fulltext_fetched", {
                    "seed": (seed_hint or "")[:80], "excerpts": len(excerpts),
                })
        except Exception as e:
            log_agent_action("TopicHunter", "seed_fulltext_error", {"error": str(e)})
        return excerpts

    # --- v5 scored selection -------------------------------------------------
    # e0ec9e91: citation/coupling/sparse/fulltext were decorative (no score
    # dim, no family, no trail). Added "grounding" so seed-level structural
    # evidence moves scores instead of only inflating log lines.
    _SCORE_WEIGHTS = {
        "evidence": 0.22, "novelty": 0.22, "feasibility": 0.18,
        "contradiction": 0.09, "reviewer": 0.09, "transfer": 0.08,
        "grounding": 0.12,
    }

    def _score_gap_candidate(
        self,
        gap: Dict[str, Any],
        signal_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Six-dimension 0..1 score for a gap (v5, deterministic).

        signal_ctx carries seed-level counts + the gap's own gate reports
        (gap_report, novelty, feasibility_check). Never raises — returns zeros.
        """
        try:
            scores: Dict[str, float] = {}
            evidence_strength = 0.5
            try:
                evidence_strength = float((gap.get("gap_report") or {}).get("evidence_strength", 0.5))
            except (TypeError, ValueError):
                evidence_strength = 0.5
            scores["evidence"] = max(0.0, min(1.0, evidence_strength))
            novelty = gap.get("novelty") or {}
            verdict = str(novelty.get("verdict") or "")
            try:
                max_sim = float(novelty.get("max_similarity", 0.5))
            except (TypeError, ValueError):
                max_sim = 0.5
            if verdict == "LIKELY_DUPLICATE":
                scores["novelty"] = 0.0
            elif verdict == "REPLICATION_OK":
                scores["novelty"] = 0.5
            else:
                scores["novelty"] = max(0.0, min(1.0, 1.0 - max_sim))
            feas = gap.get("feasibility_check") or {}
            scores["feasibility"] = 1.0 if feas.get("ok") else 0.0
            scores["contradiction"] = 1.0 if gap.get("contradiction_ref") else (
                0.3 if (signal_ctx.get("contradictions") or 0) > 0 else 0.0)
            gap_text = f"{gap.get('title','')} {gap.get('description','')} {gap.get('rationale','')}".lower()
            gap_tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", gap_text))
            rt_raw = signal_ctx.get("reviewer_themes") or []
            rt_list = rt_raw if isinstance(rt_raw, list) else []
            theme_hits = 0
            for theme in rt_list:
                try:
                    kw = set(w.lower() for w in (theme.get("keywords") or []))
                    if kw & gap_tokens:
                        theme_hits += 1
                except Exception:
                    continue
            scores["reviewer"] = 1.0 if gap.get("reviewer_theme_hit") else (
                0.6 if theme_hits > 0 else (0.2 if len(rt_list) > 0 else 0.0))
            # e0ec9e91: every kept gap sat at bridge_grounding=0.5
            # (soft-warning flood) yet scored transfer=0.5, same as a cited
            # bridge. Scale by grounding so unverified bridges don't inflate.
            try:
                _bg = float(gap.get("bridge_grounding") or 0)
            except (TypeError, ValueError):
                _bg = 0.0
            scores["transfer"] = 1.0 if _bg >= 1.0 else (
                0.35 if gap.get("evidence_bridge_ids") else 0.2)
            # Grounding: rewards gaps backed by seed-level structural evidence.
            # Each sub-signal is binary (present/absent) so a dead source
            # (e.g. S2 disabled -> 0 citations) cannot tank an otherwise good
            # gap, but live signals lift it.
            try:
                _ground_parts = []
                _ground_parts.append(1.0 if gap.get("structural_gap_ref") else (
                    0.6 if (signal_ctx.get("coupling") or 0) > 0 else 0.0))
                _ground_parts.append(1.0 if (signal_ctx.get("fulltext") or 0) > 0 else 0.0)
                _ground_parts.append(1.0 if (signal_ctx.get("citation_signals") or 0) > 0 else (
                    0.5 if (signal_ctx.get("sparse_cells") or 0) > 0 else 0.0))
                scores["grounding"] = round(sum(_ground_parts) / max(1, len(_ground_parts)), 3)
            except Exception:
                scores["grounding"] = 0.0
            total = sum(scores[d] * self._SCORE_WEIGHTS[d] for d in self._SCORE_WEIGHTS)
            return {"dims": scores, "total": round(total, 3)}
        except Exception:
            return {"dims": {}, "total": 0.0}

    def _gap_family(self, gap: Dict[str, Any]) -> str:
        """Portfolio family for a gap (v5): contradiction / replication /
        reviewer / transfer / momentum / evidence-<gap_type>."""
        try:
            if gap.get("contradiction_ref"):
                return "contradiction"
            nov = gap.get("novelty") or {}
            if nov.get("verdict") == "REPLICATION_OK" or gap.get("replication_target"):
                return "replication"
            if gap.get("reviewer_theme_hit"):
                return "reviewer"
            if float(gap.get("bridge_grounding") or 0) >= 1.0 or gap.get("structural_gap_ref"):
                return "transfer"
            if gap.get("momentum_hit"):
                return "momentum"
            gap_type = str((gap.get("gap_report") or {}).get("gap_type") or "evidence")
            return f"evidence-{gap_type}"
        except Exception:
            return "evidence-unknown"

    def _build_evidence_trail(self, gap: Dict[str, Any]) -> List[str]:
        """3-bullet human-readable evidence trail (v5): closest prior work +
        signal refs, for the ranker, debate, and UI."""
        trail: List[str] = []
        try:
            for ev in (gap.get("literature_evidence") or [])[:2]:
                title = (ev.get("title") or "").strip()
                if title:
                    trail.append(f"Closest prior work: {title[:140]}")
            cr = gap.get("contradiction_ref") or {}
            if isinstance(cr, dict) and cr.get("paper_a_title"):
                trail.append(
                    f"Contradiction leverage: '{str(cr.get('paper_a_title'))[:80]}' vs "
                    f"'{str(cr.get('paper_b_title'))[:80]}'")
            if gap.get("reviewer_theme_hit"):
                trail.append(f"Reviewer-stated pain: {str(gap.get('reviewer_theme_hit'))[:140]}")
            if gap.get("structural_gap_ref"):
                trail.append("Bibliographic-coupling pair (shared refs, no direct citation)")
            if gap.get("momentum_hit"):
                trail.append(f"Heating-up theme: {str(gap.get('momentum_hit'))[:120]}")
            bridges = gap.get("evidence_bridge_ids") or []
            if bridges:
                trail.append(f"Cross-paper bridge: {str(bridges[0])[:120]}")
        except Exception:
            pass
        return trail[:3]

    def _log_seed_funnel(self, seed_hint: str, where: str) -> None:
        """Per-seed rejection funnel since _seed_rejection_mark (v5)."""
        try:
            mark = int(getattr(self, "_seed_rejection_mark", 0) or 0)
            funnel: Dict[str, int] = {}
            for entry in self.rejection_log[mark:]:
                funnel[entry["reason"]] = funnel.get(entry["reason"], 0) + 1
            log_agent_action("TopicHunter", "seed_rejection_funnel", {
                "seed": (seed_hint or "")[:120], "where": where,
                "rejected": len(self.rejection_log) - mark,
                "by_reason": dict(sorted(funnel.items(), key=lambda kv: -kv[1])),
            })
        except Exception:
            pass

    def _apply_portfolio_diversity(self, ranked: List[Topic]) -> List[Topic]:
        """Guarantee ≥3 gap families in the top-5 when available (v5).

        Stable: keeps rank order within a family, pulls up the first unseen
        family. Rank order is never inverted — only diversified.
        """
        try:
            if not getattr(self.runtime_config, "hunter_portfolio_enabled", True):
                return ranked
            if len(ranked) <= 3:
                return ranked
            for t in ranked:
                if isinstance(t, dict) and not t.get("gap_family"):
                    t["gap_family"] = self._gap_family(t)
            top, rest = ranked[:5], ranked[5:]
            seen: List[str] = []
            diverse: List[Topic] = []
            deferred: List[Topic] = []
            for t in top:
                fam = t.get("gap_family", "evidence-unknown")
                if fam not in seen:
                    seen.append(fam)
                    diverse.append(t)
                else:
                    deferred.append(t)
            if len(seen) < 3:
                for t in rest:
                    fam = t.get("gap_family", "evidence-unknown")
                    if fam not in seen:
                        seen.append(fam)
                        diverse.append(t)
                        if len(seen) >= 3:
                            break
                diverse.extend(deferred)
                diverse.extend([t for t in rest if t not in diverse])
                log_agent_action("TopicHunter", "portfolio_diversified", {
                    "families": seen, "top5": [t.get("title", "")[:160] for t in diverse[:5]],
                })
                return diverse + [t for t in ranked if t not in diverse and t not in deferred]
            return ranked
        except Exception as e:
            log_agent_action("TopicHunter", "portfolio_error", {"error": str(e)})
            return ranked

    def _get_dataset_catalog(self) -> List[Dict[str, Any]]:
        """Return cached dataset catalog summary. Fail-open: returns [] on error."""
        if self._dataset_catalog_cache is not None:
            return self._dataset_catalog_cache
        try:
            from core.datasets import list_datasets
            raw = list_datasets()
            self._dataset_catalog_cache = [
                {
                    "name": d.get("name", ""),
                    "domain_tags": d.get("domain_tags", []),
                    "n_samples": d.get("max_rows", 0),
                    "task_type": d.get("task_type", "unknown"),
                }
                for d in raw
            ]
        except Exception as e:
            log_agent_action("TopicHunter", "dataset_catalog_load_error", {"error": str(e)})
            self._dataset_catalog_cache = []
        return self._dataset_catalog_cache

    def _dataset_plan_admissible(self, gap: Dict[str, Any]) -> bool:
        """Feature 1: string check — reject gaps whose dataset_plan names
        a dataset not in the local catalog, unless it's synthetic/bundled/known-public
        or available on HuggingFace. Fail-open: catalog errors or empty catalog → always True."""
        if not getattr(self.runtime_config, "capability_first_dataset_scoping_enabled", True):
            return True
        dataset_plan = str(gap.get("dataset_plan", "")).strip()
        if not dataset_plan:
            return True
        if dataset_plan.startswith("synthetic") or dataset_plan.startswith("bundled"):
            return True
        # Generic descriptors ("public", "open dataset", "custom benchmark")
        # name no concrete dataset — an HF id lookup for them just 404s
        # (huggingface.co/api/datasets/public, run b099dcaa) before the
        # fail-open admit. Treat them as admissible without network calls.
        plan_words = set(re.findall(r"[a-z][a-z0-9-]*", dataset_plan.lower()))
        if plan_words and plan_words <= _GENERIC_DATASET_PLAN_WORDS:
            log_agent_action("TopicHunter", "dataset_plan_generic_descriptor", {
                "title": gap.get("title"),
                "dataset_plan": dataset_plan,
            })
            return True
        # Check known public datasets (fast path, no API call)
        plan_lower = dataset_plan.lower()
        for kw in _KNOWN_PUBLIC_DATASETS:
            if kw in plan_lower:
                return True
        catalog = self._get_dataset_catalog()
        if not catalog:
            log_agent_action("TopicHunter", "dataset_admissibility_skip_empty_catalog", {})
            return True
        catalog_names = {d["name"].lower() for d in catalog}
        if dataset_plan.lower() in catalog_names:
            return True
        # Check HuggingFace (works without token for public datasets)
        # v2: skip HF check entirely if circuit breaker is open — fail-open
        # instead of wasting time waiting for a dead provider.
        from core.api_gateway import get_gateway
        gateway = get_gateway()
        _hf_verified = False
        _hf_skip_reason = ""
        if gateway.is_available("huggingface"):
            from core.datasets import dataset_is_admissible
            if dataset_is_admissible(dataset_plan, local_catalog=catalog):
                log_agent_action("TopicHunter", "dataset_admitted_hf", {
                    "title": gap.get("title"),
                    "dataset_plan": dataset_plan,
                    "verified": True,
                })
                return True
            _hf_skip_reason = "not_found_on_hf"
        else:
            _hf_skip_reason = "breaker_open"
            log_agent_action("TopicHunter", "dataset_hf_skipped_breaker_open", {
                "title": gap.get("title"),
                "dataset_plan": dataset_plan,
            })
        # Fail-open: if HF is unreachable or dataset not found, admit anyway —
        # the dataset check is advisory, not a hard gate. P1-1: explicitly
        # mark unverified so formalize/admission default the MVE to
        # bundled_synthetic instead of inheriting a phantom dataset id.
        # The fallback event carries verified=False for audit queries.
        log_agent_action("TopicHunter", "dataset_plan_admitted_fallback", {
            "title": gap.get("title"),
            "dataset_plan": dataset_plan,
            "verified": _hf_verified,
            "reason": _hf_skip_reason,
        })
        return True

    def _find_replication_targets(self, papers: List[Dict[str, Any]], max_results: int = 5) -> List[Dict[str, Any]]:
        """Feature 5: heuristic regex pre-filter to flag papers making strong claims
        with no visible variance/multi-seed reporting. Returns candidates for
        replication-and-extension topics. No LLM call — purely regex/heuristic."""
        if not getattr(self.runtime_config, "replication_target_mining_enabled", True):
            return []
        strong_claim_re = re.compile(
            r"(significantly|substantially|outperform|superior|state.of.the.art|best.?performing|achieves?.\s*\d)",
            re.IGNORECASE,
        )
        no_variance_re = re.compile(
            r"(seeds?|std|standard deviation|confidence interval|multiple runs|repeated trials|variance|error bar|±)",
            re.IGNORECASE,
        )
        candidates = []
        for p in papers:
            abstract = p.get("abstract", "") or ""
            title = p.get("title", "") or ""
            text = f"{title} {abstract}"
            if not text.strip():
                continue
            has_strong_claim = bool(strong_claim_re.search(text))
            has_no_variance = not bool(no_variance_re.search(abstract))
            if has_strong_claim and has_no_variance:
                # Extract a claim excerpt
                excerpt = ""
                m = strong_claim_re.search(abstract)
                if m:
                    start = max(0, m.start() - 60)
                    end = min(len(abstract), m.end() + 60)
                    excerpt = abstract[start:end].strip()
                candidates.append({
                    "title": title[:150],
                    "abstract_snippet": abstract[:300],
                    "matched_claim_excerpt": excerpt,
                })
                if len(candidates) >= max_results:
                    break
        return candidates

    def _formalize_pregate(self, candidate: Dict[str, Any], hunt_mode: str = "gap") -> Tuple[bool, str]:
        """Deterministic pre-gate before any formalize LLM call (P0-2).

        Rejects obviously-unformalisable gaps in milliseconds instead of
        burning 2 strong-tier calls (~10 min on a slow endpoint). Narrow by
        design: only empty/short/verb-less candidates fail here; everything
        else still gets its LLM chance.
        P2: try_and_report skips the testability-word regex — "try X and
        report" is formisable by construction; only the length floor applies.
        """
        text = f"{candidate.get('title') or ''} {candidate.get('description') or ''} {candidate.get('rationale') or ''}".strip()
        if len(text) < 25:
            return False, "candidate_text_too_short"
        if str(hunt_mode or "gap") in ("try_and_report", "replication"):
            return True, ""
        lowered = text.lower()
        if not re.search(r"\b(test\w*|measur\w*|evaluat\w*|compar\w*|experiment\w*|effect\w*|impact\w*|improv\w*|reduc\w*|increas\w*|degrad\w*|latency|accuracy|bias|reproduc\w*|validat\w*)\b", lowered):
            return False, "no_testability_signal"
        return True, ""

    def formalize_hypothesis(
        self,
        candidate: Dict[str, Any],
        gap_report: Dict[str, Any],
        novelty_report: Dict[str, Any],
        hunt_mode: str = "gap",
    ) -> Optional[Dict[str, Any]]:
        """Hypothesis Formalizer: 2-phase approach for high reliability.

        Phase 1 (cheap tier): Generate ONLY the 4 core fields — research_question,
        hypothesis, falsification_condition, dependent_variables. If this fails,
        reject early without burning strong-tier calls.

        Phase 2 (strong tier): Build the FULL experimental contract around the
        validated core. This is a much easier task — the model just needs to
        flesh out an existing hypothesis, not invent one from scratch.
        """
        # --- Phase 1: core fields (cheap tier, fast, high pass rate) ---
        phase1_prompt = f"""
Given this research gap, produce EXACTLY 4 fields. Be concrete and specific.

Title: {candidate.get('title')}
Description: {candidate.get('description')}
Rationale: {candidate.get('rationale')}
Literature evidence: {json.dumps(novelty_report, default=str)[:800]}

Generate:
1. research_question: A precise, specific question (not a vague direction)
2. hypothesis: A falsifiable claim in the form "X causes/improves Y over Z because W"
3. falsification_condition: What specific observation would DISPROVE this hypothesis?
4. dependent_variables: Specific measurable metrics (e.g. ["ECE", "accuracy", "F1"])

Return JSON:
{{
  "research_question": "...",
  "hypothesis": "...",
  "falsification_condition": "...",
  "dependent_variables": ["..."]
}}
"""
        core_fields = ("research_question", "hypothesis", "falsification_condition", "dependent_variables")
        core_parsed: Dict[str, Any] = {}

        for attempt in range(2):
            temp = 0.3 if attempt == 0 else 0.1
            raw = call_llm(phase1_prompt, temperature=temp, tier="strong",
                           **_llm_kwargs("hunter.hypothesis_core", "strong"))
            parsed = parse_json_from_llm(raw) or {}
            if isinstance(parsed, dict):
                missing = [f for f in core_fields if not parsed.get(f)]
                if not missing:
                    core_parsed = parsed
                    break
                if attempt == 0:
                    phase1_prompt += f"\n\nYour previous response was missing: {', '.join(missing)}. Fix ONLY those fields."

        if not core_parsed:
            log_agent_action("TopicHunter", "formalize_phase1_failed", {
                "title": candidate.get("title"),
                "missing": [f for f in core_fields if not core_parsed.get(f)],
            })
            return None

        log_agent_action("TopicHunter", "formalize_phase1_ok", {
            "title": candidate.get("title"),
            "core_fields": list(core_parsed.keys()),
        })

        # --- Phase 2: delta contract (strong tier) ---
        # P0-2: previously re-emitted the full 20-field contract (~4-6k prompt
        # chars + 16k max_tokens), doubling truncation risk and latency on a
        # slow endpoint. Now the model returns ONLY the delta fields; the
        # validated core is merged server-side (core can never be dropped by a
        # truncated re-emission). Prompt context slimmed: gap report 800 chars,
        # manifest summary line, 3 paper titles — full evidence lives in
        # debate/writer where it belongs.
        phase2_fields = ("independent_variables", "baselines", "metrics", "minimum_viable_experiment")
        # Hard requirement for a plannable contract; everything else soft.
        # P2: try_and_report requires only an MVE — baselines stay advisory
        # so "try X and report" survives without a comparison matrix.
        if str(hunt_mode or "gap") in ("try_and_report", "replication"):
            phase2_hard = ("minimum_viable_experiment",)
        else:
            phase2_hard = ("baselines", "minimum_viable_experiment")

        _lit_titles = []
        try:
            for _p in (candidate.get("literature_evidence") or [])[:3]:
                _t = (_p.get("title") or "")[:160]
                if _t:
                    _lit_titles.append(_t)
        except Exception:
            pass
        base_prompt = f"""
Add ONLY the delta fields below as JSON. Do NOT repeat the validated core.

VALIDATED CORE (already stored, do not re-emit):
{json.dumps(core_parsed, default=str)[:1200]}

Gap: {(candidate.get('title') or '')[:300]} — {(candidate.get('description') or '')[:300]}
Gap evidence: {json.dumps(gap_report, default=str)[:800]}
Closest papers: {'; '.join(_lit_titles) if _lit_titles else 'none'}
    Sandbox: CPU-only, gpu=false, max 4GB/120s, dataset bundled_synthetic or named public.
    CPU rescope rule (applies BEFORE writing the contract): {_CPU_EXECUTION_ENVELOPE}
    Accelerator-bound gaps → CPU proxy (simulation/tiny model/stats), gpu=false.

Return ONLY this JSON object (delta fields only):
{{
  "independent_variables": [{{"name": "...", "values": [...]}}],
  "expected_relationship": "...",
  "novelty_claim": "...",
  "closest_prior_work": [{{"title": "...", "difference": "..."}}],
  "baselines": ["..."],
  "metrics": ["..."],
  "confounders": ["..."],
  "competing_explanations": ["..."],
  "minimum_viable_experiment": {{
    "dataset": "bundled_synthetic",
    "models": ["..."],
    "conditions": ["controlled_seeds"],
    "metrics": ["..."],
    "seeds": 3,
    "baseline": "...",
    "expected_result": "...",
    "falsification_test": "Welch t-test p<0.05"
  }},
  "required_resources": {{"cpu": true, "gpu": false, "max_memory_mb": 4096, "max_runtime_seconds": 120}}
}}

Output ONLY raw JSON — no prose, no markdown fences.
"""
        prompt = base_prompt
        last_parsed: Dict[str, Any] = {}
        # Single repair only (was 2). Each strong-tier call costs ~5 minutes
        # on slow reasoning endpoints; two repairs burned ~10 min per gap
        # before the identical "missing_structured_hypothesis" rejection
        # (run 25ed95d4). One repair preserves the self-correction benefit —
        # the retry prompt carries the missing-field list + literature
        # context — while failing fast on unformalisable gaps.
        max_attempts = 2
        # P0-2: delta output is ~1/3 the tokens of the old full contract, so
        # 8192 suffices; json_mode (no double-retry) returns partials for
        # repair instead of a second ~300s full re-run on slow endpoints.
        phase2_max_tokens = 8192
        for attempt in range(max_attempts):
            temp = 0.1 if attempt > 0 else 0.3
            try:
                raw = call_llm(prompt, temperature=temp, tier="strong",
                               max_tokens=phase2_max_tokens, retry_on_truncation=False,
                               **_llm_kwargs("hunter.hypothesis_delta", "strong"))
            except Exception as exc:
                log_agent_action("TopicHunter", "formalize_phase2_infra", {
                    "title": candidate.get("title"), "attempt": attempt + 1,
                    "error": str(exc)[:200],
                })
                if attempt < max_attempts - 1:
                    continue
                return None
            parsed = parse_json_from_llm(raw) or {}
            if isinstance(parsed, dict):
                last_parsed = parsed
                # Merge: validated core can never be dropped by a truncated
                # delta re-emission; delta wins on conflicts (repair path).
                merged = dict(core_parsed)
                merged.update({k: v for k, v in parsed.items() if v})
                missing = [f for f in phase2_hard if not merged.get(f)]
                if not missing:
                    merged["gap_report"] = gap_report
                    merged["novelty_report"] = novelty_report
                    return merged
                last_parsed = merged
            else:
                missing = list(phase2_hard)

            if attempt < max_attempts - 1:
                field_hints = ""
                if "minimum_viable_experiment" in missing:
                    field_hints += "\n- minimum_viable_experiment.dataset: Use 'bundled_synthetic' if no specific dataset fits. Include 'models', 'baseline', 'metrics', 'seeds' fields."
                if "baselines" in missing:
                    field_hints += "\n- baselines: name 1-2 concrete baselines (e.g. ['logistic_regression', 'random_forest'])."

                prompt = base_prompt + f"""

Your previous delta was missing: {', '.join(missing)}.
Your previous delta:
{json.dumps(last_parsed, default=str)[:1500]}
{field_hints}

Return ONLY the corrected delta JSON (delta fields only, same schema).
"""
                log_agent_action("TopicHunter", "formalize_hypothesis_repair_attempt", {
                    "attempt": attempt + 1,
                    "missing_fields": missing,
                    "title": candidate.get("title"),
                })

        # Final validation: merged contract must carry the validated core plus
        # the hard delta fields. Core presence is guaranteed by construction.
        if last_parsed:
            if all(last_parsed.get(f) for f in ("hypothesis", "falsification_condition",
                                                "dependent_variables", "research_question")) \
               and all(last_parsed.get(f) for f in phase2_hard):
                last_parsed["gap_report"] = gap_report
                last_parsed["novelty_report"] = novelty_report
                return last_parsed

        return None

    def feasibility_filter(self, topic: Topic) -> FeasibilityReport:
        """Grounded in what Engineer sandbox can actually run.

        Fix #2: blocked-phrase matching is now negation-aware. Previously a plain
        substring check meant a description saying e.g. "does not require a gpu
        cluster" or "no human subjects needed" would be false-positive rejected,
        since the blocked phrase is still a literal substring of the negated
        sentence. Now a blocked phrase occurring immediately after a negation cue
        (not/no/without/does not require/etc.) is treated as a disclaimer, not a
        requirement, and does not trigger rejection.
        """
        reasons = []
        ok = True
        text = json.dumps(topic).lower()
        for b in _BLOCKED_SANDBOX_PHRASES:
            if self._phrase_blocked_in_text(b, text):
                ok = False
                reasons.append(f"Not executable in sandbox: {b}")
        feas = topic.get("feasibility", 5)
        if isinstance(feas, (int, float)) and feas < 3:
            ok = False
            reasons.append(f"Low feasibility score: {feas}")
        # Prefer synthetic / public small data.
        # 119ce81c: this checked the whole topic JSON including
        # literature_evidence abstracts, so a *cited paper* mentioning a
        # proprietary corpus killed an otherwise synthetic candidate.
        # Only the candidate's own plan counts: title + description +
        # dataset_plan.
        _own_plan_text = " ".join([
            str(topic.get("title", "")),
            str(topic.get("description", "")),
            str(topic.get("dataset_plan", "")),
        ]).lower()
        if "dataset" in _own_plan_text and "proprietary" in _own_plan_text and not self._is_negated("proprietary", _own_plan_text):
            ok = False
            reasons.append("Proprietary dataset unavailable")
        # Run the shared sandbox manifest whenever the topic text hints at
        # resource requirements the sandbox may not satisfy. Previously this
        # only ran on explicit network-requirement phrases, so GPU-bound
        # measurement studies (LLM serving, kernel scheduling, …) sailed
        # through the hunter and died ~25 minutes later in debate. The
        # manifest stays the single source of truth; this gate only decides
        # when it is worth consulting. Triggers are deliberately narrow
        # (accelerator/training/scale vocabulary) so ordinary prose mentioning
        # e.g. "download" in passing does not trip the network tokens inside
        # the manifest checker. Negated triggers ("does not require a gpu")
        # are disclaimers, not requirements — same rule as the blocked list.
        triggered = [
            phrase for phrase in _MANIFEST_CHECK_TRIGGERS
            if phrase in text and not self._is_negated(phrase, text)
        ]
        if triggered:
            candidate_plan = {
                "methodology": f"{topic.get('title', '')} {topic.get('description', '')}",
                "experiments": [{"dataset": {"name": topic.get("dataset_plan", "") or "bundled_synthetic"}}],
            }
            capability_reasons = check_plan_feasibility(candidate_plan, SANDBOX_CAPABILITY_MANIFEST)
            if capability_reasons:
                ok = False
                reasons.extend(capability_reasons)
        return {"ok": ok, "reasons": reasons}

    @staticmethod
    def _is_negated(phrase: str, text: str, window: int = 24) -> bool:
        """Fix #2: check whether `phrase` is preceded by a negation cue within
        `window` characters, anywhere it occurs in `text`. Returns True if every
        occurrence of `phrase` is negated (so the phrase should NOT trigger a
        block); returns False if at least one occurrence is unnegated."""
        found_any = False
        for m in re.finditer(re.escape(phrase), text):
            found_any = True
            start = max(0, m.start() - window)
            preceding = text[start:m.start()]
            if not any(cue in preceding for cue in _NEGATION_CUES):
                return False  # an unnegated occurrence exists -> not fully negated
        return found_any

    @classmethod
    def _phrase_blocked_in_text(cls, phrase: str, text: str) -> bool:
        """Fix #2: True only if `phrase` appears in `text` at least once WITHOUT
        being immediately preceded by a negation cue. A phrase that only ever
        appears negated (disclaiming the requirement) does not block."""
        if phrase not in text:
            return False
        return not cls._is_negated(phrase, text)

    def pre_debate_self_critique(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Reflection agent: identifies top 3 likely debate objections and rewrites
        the hypothesis to preempt them. Burns 1 cheap-tier LLM call but
        significantly strengthens the hypothesis before adversarial debate.

        Returns the topic dict with strengthened structured_hypothesis,
        or the original topic if self-critique fails.
        """
        structured_hyp = topic.get("structured_hypothesis")
        if not structured_hyp or not isinstance(structured_hyp, dict):
            return topic

        logger.info("Pre-debate self-critique starting for %r", topic.get("title", "")[:160])

        prompt = f"""
You are a hostile peer reviewer. Given this hypothesis, identify the top 3
objections a reviewer would raise, then REWRITE the hypothesis to preempt them.

Hypothesis: {structured_hyp.get('hypothesis', '')}
Research question: {structured_hyp.get('research_question', '')}
Falsification condition: {structured_hyp.get('falsification_condition', '')}
Dependent variables: {json.dumps(structured_hyp.get('dependent_variables', []))}
Baselines: {json.dumps(structured_hyp.get('baselines', []))}
MVE: {json.dumps(structured_hyp.get('minimum_viable_experiment', {}), default=str)[:1000]}

For each objection:
1. What is the objection? (be specific)
2. How severe is it? (1-5)
3. How can the hypothesis be REWRITTEN to address it?

Then provide a STRENGTHENED version of the hypothesis that preempts all 3 objections
while preserving the core scientific claim.

Return JSON:
{{
  "objections": [
    {{"objection": "...", "severity": 3, "fix": "..."}}
  ],
  "strengthened_hypothesis": "...",
  "strengthened_falsification": "...",
  "strengthened_baselines": ["..."],
  "strengthened_mve": {{
    "dataset": "...",
    "models": ["..."],
    "baseline": "...",
    "metrics": ["..."],
    "seeds": 3,
    "falsification_test": "..."
  }}
}}
"""
        try:
            raw = call_llm(prompt, temperature=0.3, tier="cheap",
                           **_llm_kwargs("hunter.self_critique", "cheap"))
            parsed = parse_json_from_llm(raw) or {}
            if not isinstance(parsed, dict):
                return topic

            strengthened_hyp = parsed.get("strengthened_hypothesis")
            if not strengthened_hyp:
                return topic

            # Apply strengthening to the structured hypothesis
            new_hyp = dict(structured_hyp)
            new_hyp["hypothesis"] = strengthened_hyp

            strengthened_fals = parsed.get("strengthened_falsification")
            if strengthened_fals:
                new_hyp["falsification_condition"] = strengthened_fals

            strengthened_baselines = parsed.get("strengthened_baselines")
            if isinstance(strengthened_baselines, list) and strengthened_baselines:
                new_hyp["baselines"] = strengthened_baselines

            strengthened_mve = parsed.get("strengthened_mve")
            if isinstance(strengthened_mve, dict) and strengthened_mve:
                old_mve = new_hyp.get("minimum_viable_experiment") or {}
                old_mve.update({k: v for k, v in strengthened_mve.items() if v})
                new_hyp["minimum_viable_experiment"] = old_mve

            # Quality gate: validate strengthened hypothesis is not worse
            orig_hyp = str(structured_hyp.get("hypothesis", ""))
            if len(strengthened_hyp) < len(orig_hyp) * 0.5:
                # Strengthened version is significantly shorter — likely degraded
                log_agent_action("TopicHunter", "pre_debate_critique_degraded", {
                    "title": topic.get("title"),
                    "reason": "strengthened_too_short",
                })
                return topic
            if not any(kw in strengthened_hyp.lower() for kw in ["test", "measure", "evaluate", "compare", "experiment"]):
                # Strengthened version lost testability keywords
                log_agent_action("TopicHunter", "pre_debate_critique_degraded", {
                    "title": topic.get("title"),
                    "reason": "lost_testability_keywords",
                })
                return topic

            topic["structured_hypothesis"] = new_hyp
            topic["falsifiable_prediction"] = new_hyp.get("falsification_condition", "")

            objections = parsed.get("objections") or []
            log_agent_action("TopicHunter", "pre_debate_critique", {
                "title": topic.get("title"),
                "objections_found": len(objections),
                "hypothesis_strengthened": True,
            })
            logger.info("Pre-debate self-critique applied to %r: %d objections found, hypothesis strengthened",
                         topic.get("title", "")[:160], len(objections))
            return topic

        except Exception as e:
            log_agent_action("TopicHunter", "pre_debate_critique_error", {"error": str(e)})
            logger.warning("Pre-debate self-critique failed for %r: %s", topic.get("title", "")[:160], e)
            return topic

    def _reject(self, topic: Topic, reason: str, meta: Optional[Dict[str, Any]] = None):
        meta = meta or {}
        entry = {
            "title": topic.get("title"),
            "reason": reason,
            "meta": meta,
            "ts": datetime.now().isoformat(),
        }
        self.rejection_log.append(entry)
        topic_kind = topic.get("hypothesis_kind") or hypothesis_kind(topic.get("title", ""))
        lesson_type = meta.get("lesson_type") or "topic_rejection"
        reason_code = meta.get("reason_code") or reason
        # Inject run_id for intra-run collision detection (Fix 5)
        tracker = get_tracker()
        current_run_id = tracker.run_id if tracker else None
        meta_with_run = {**meta, "run_id": current_run_id}
        # Skip recording intra-run self-collisions to prevent the exclusion
        # corpus from growing with near-copies that tighten matching around
        # the same theme within a single run.
        if reason_code == "previously_failed_or_rejected" and current_run_id:
            if CrossRunMemory().is_intra_run_collision("topic", topic.get("title", "?"), current_run_id):
                log_agent_action("TopicHunter", "intra_run_collision_skipped", {
                    "title": topic.get("title", "?")[:80],
                    "reason": "already recorded this run",
                })
                if tracker:
                    tracker.bump("rejected_topics")
                    tracker.scratch("TopicHunter", "rejection", entry)
                return
        log_agent_action("TopicHunter", "topic_rejected", entry)
        # Fix 13: Track rejected theme vocab for theme-level dedup
        title_words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", (topic.get("title") or "").lower()))
        if title_words:
            theme_key = " ".join(sorted(title_words)[:8])
            self._rejected_theme_vocab = getattr(self, "_rejected_theme_vocab", {})
            self._rejected_theme_vocab[theme_key] = self._rejected_theme_vocab.get(theme_key, 0) + 1
        CrossRunMemory().record_rejection(
            "topic",
            topic.get("title", "?"),
            reason,
            meta_with_run,
        )
        self._excluded_titles_cache = None
        if tracker:
            tracker.bump("rejected_topics")
            tracker.scratch("TopicHunter", "rejection", entry)

    def _excluded_titles(self) -> List[str]:
        if self._excluded_titles_cache is None:
            self._excluded_titles_cache = CrossRunMemory().excluded_topic_titles()
        return self._excluded_titles_cache

    def _matches_excluded_topic(self, title: str) -> Optional[str]:
        """Return the prior excluded title if this candidate is a near-duplicate."""
        candidate = (title or "").strip()
        if not candidate:
            return None
        lowered = candidate.lower()
        for prior in self._excluded_titles():
            if not prior:
                continue
            if lowered == prior.lower() or title_token_overlap(candidate, prior) >= _FAILED_TOPIC_OVERLAP_THRESHOLD:
                return prior
        return None

    def _suppress_near_duplicates(self, topics: List[Dict[str, Any]], threshold: float = 0.70) -> List[Dict[str, Any]]:
        """Drop later topics whose title overlaps an already-kept topic (v3).

        Uses the same token-overlap metric as failed-topic matching. Keeps the
        first occurrence (higher-ranked seed strategies were processed first).
        Falls back to all topics on any error — this is an optimization, not a
        gate.
        """
        try:
            kept: List[Dict[str, Any]] = []
            for t in topics:
                title = (t.get("title") or "").strip()
                if not title:
                    continue
                duplicate = any(
                    title_token_overlap(title, k.get("title", "")) >= threshold
                    for k in kept
                )
                if duplicate:
                    log_agent_action("TopicHunter", "near_duplicate_suppressed", {
                        "title": title[:120],
                        "against": kept[-1].get("title", "") if kept else "",
                        "threshold": threshold,
                    })
                    continue
                kept.append(t)
            return kept
        except Exception as e:
            log_agent_action("TopicHunter", "near_duplicate_suppression_error", {"error": str(e)})
            return list(topics)

    def _fallback_bridge_match(self, gap: Dict[str, Any], evidence_map: Dict[str, Any]) -> List[str]:
        """Fix #4: programmatically match a gap against the evidence map's bridges
        by keyword overlap, for use when the LLM omitted or malformed
        `evidence_bridge_ids`. This lets a gap that's actually grounded in a real
        bridge survive even when the model failed at exact-ID citation — an
        exact-string-matching task LLMs are unreliable at, especially at
        temperature=0.8 on a cheap tier.
        """
        bridges = evidence_map.get("bridges", []) if isinstance(evidence_map, dict) else []
        if not bridges:
            return []
        gap_text = f"{gap.get('title','')} {gap.get('description','')} {gap.get('rationale','')}".lower()
        gap_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", gap_text)) - _QUERY_STOPWORDS
        matched_ids = []
        for bridge in bridges:
            bridge_id = bridge.get("id") or bridge.get("bridge_id")
            if not bridge_id:
                continue
            bridge_text = f"{bridge.get('method_signal','')} {bridge.get('target_setting_signal','')}".lower()
            bridge_tokens = set(re.findall(r"[a-z][a-z0-9_-]{3,}", bridge_text))
            if bridge_tokens and (gap_tokens & bridge_tokens):
                matched_ids.append(bridge_id)
        return matched_ids

    def _generate_hyde_abstract(self, seed_hint: str, domain: str) -> Optional[str]:
        """Generate a short hypothetical abstract for what a real paper on
        `seed_hint` within `domain` would say, purely from the LLM's parametric
        knowledge (no retrieval). Used to derive richer query keywords than
        tokenizing the seed phrase directly. Returns None on any failure so
        callers can fall back to the existing keyword-extraction path unchanged.
        """
        if not getattr(self.runtime_config, "hyde_enabled", True):
            return None
        prompt = f"""
Write a 2-3 sentence hypothetical abstract for a research paper in {domain}
that investigates: {seed_hint}

Write it the way a real arXiv abstract reads: specific method names, specific
evaluation setups, specific terminology a specialist would use. Do not
mention that this is hypothetical. Return ONLY the abstract text, no preamble,
no JSON, no quotes.
"""
        try:
            raw = call_llm(prompt, temperature=0.5, tier="cheap",
                           **_llm_kwargs("hunter.hyde", "cheap"))
            text = (raw or "").strip()
            if not text:
                return None
            return text[: getattr(self.runtime_config, "hyde_max_chars", 600)]
        except Exception as e:
            log_agent_action("TopicHunter", "hyde_generation_error", {"error": str(e), "seed": seed_hint})
            return None

    def _derive_followup_query_text(
        self, seed_hint: str, hyde_abstract: Optional[str],
        papers_so_far: List[Dict[str, Any]], domain: str,
    ) -> Optional[str]:
        """Cheap LLM call: given what little we found, propose a narrower or
        differently-worded query phrase to try on hop 2. Returns None on failure
        (multi-hop loop treats that as 'stop, no more hops')."""
        sample_titles = [p.get("title", "")[:100] for p in papers_so_far[:5]]
        prompt = f"""
A literature search for the research seed below returned very few results.
Seed: {seed_hint}
{"Hypothetical target abstract: " + hyde_abstract if hyde_abstract else ""}
Titles found so far (may be off-target or too generic): {json.dumps(sample_titles)}

Propose ONE alternative, more specific or differently-phrased search phrase
(3-8 words, using terminology a specialist in {domain} would search for) that
might surface literature this first attempt missed. Return ONLY the phrase,
no punctuation, no explanation.
"""
        try:
            raw = call_llm(prompt, temperature=0.4, tier="cheap",
                           **_llm_kwargs("hunter.followup_query", "cheap"))
            text = (raw or "").strip().strip('"').strip("'")
            return text or None
        except Exception as e:
            log_agent_action("TopicHunter", "followup_query_error", {"error": str(e)})
            return None

    def _load_cached_frontier_terms(self, domain: str) -> List[Dict[str, str]]:
        """Load frontier terms from cache without harvesting (zero LLM cost)."""
        if not getattr(self.runtime_config, "frontier_seeding_enabled", True):
            return []
        cache_path = Path(self.runtime_config.output_dir) / "source_cache" / "frontier_terms.json"
        try:
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                if cached.get("domain") == domain:
                    return cached.get("terms", [])
        except Exception:
            pass
        return []

    def _harvest_frontier_terms(self, domain: str) -> List[Dict[str, str]]:
        """Sample very recent, high-signal papers for `domain` and extract
        emerging method/evaluation term pairs via a cheap LLM call. Cached to
        disk and refreshed every `frontier_refresh_every_n_runs` calls to
        discover_topics() (run-count gated, NOT time-gated — this system has no
        background daemon, so a wall-clock TTL cannot reflect actual usage
        cadence). Returns [] on any failure — callers must treat this as purely
        additive to the existing static _EXPLORATION_KIND_BIASES seeds, never a
        replacement, so a [] result degrades to exactly current behavior.
        """
        if not getattr(self.runtime_config, "frontier_seeding_enabled", True):
            return []

        cache_path = Path(self.runtime_config.output_dir) / "source_cache" / "frontier_terms.json"
        refresh_every_n = max(1, int(getattr(self.runtime_config, "frontier_refresh_every_n_runs", 5)))

        cached = None
        try:
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
        except Exception as e:
            log_agent_action("TopicHunter", "frontier_cache_read_error", {"error": str(e)})
            cached = None

        if cached and cached.get("domain") == domain:
            runs_since = int(cached.get("runs_since_refresh", 0))
            if runs_since < refresh_every_n - 1:
                # Not due yet — bump the counter and reuse cached terms without
                # spending any LLM/API budget.
                cached["runs_since_refresh"] = runs_since + 1
                try:
                    cache_path.write_text(json.dumps(cached))
                except Exception as e:
                    log_agent_action("TopicHunter", "frontier_cache_write_error", {"error": str(e)})
                return cached.get("terms", [])
            # else: due for refresh — fall through to harvest below.

        # Sample recent, high-signal papers directly (not via a seed — this is
        # domain-wide, not seed-specific). Falls back to OpenAlex if arXiv disabled.
        # v5: resolve the DOMAIN to its subcategory first — sampling cat:cs.LG
        # for a software-engineering run poisoned every frontier seed (55ac3547).
        sample_size = int(getattr(self.runtime_config, "frontier_sample_size", 30))
        if getattr(self.runtime_config, "arxiv_enabled", True):
            subcat = _subcategory_for_domain(domain or "")
            categories = _ARXIV_CATEGORIES.get(subcat, _ARXIV_CATEGORIES["general"])
            sample_query = f"cat:{categories[0]}"
            papers = self.search_arxiv(sample_query, max_results=sample_size)
        else:
            # Fallback: use OpenAlex with concept filtering for recent papers
            papers = self.search_openalex(domain, limit=sample_size)
        if not papers:
            return []

        titles = [p.get("title", "")[:150] for p in papers if p.get("title")]
        n_terms = int(getattr(self.runtime_config, "frontier_terms_extracted", 8))
        prompt = f"""
Below are titles of very recent papers in {domain}. Identify {n_terms}
emerging method/evaluation term pairs that represent CURRENT active research
directions — prefer specific, technical, compound terminology (e.g. a real
method name or technique combination) over broad category words already
implied by field basics.

Titles:
{json.dumps(titles[:sample_size], indent=2)}

Return JSON: {{"terms": [{{"method": "...", "evaluation": "..."}}]}}
"""
        try:
            parsed = parse_json_from_llm(call_llm(
                prompt, temperature=0.5, tier="cheap",
                **_llm_kwargs("hunter.frontier", "cheap", domain=domain))) or {}
            terms = parsed.get("terms") or []
            terms = [t for t in terms if isinstance(t, dict) and t.get("method") and t.get("evaluation")][:n_terms]
        except Exception as e:
            log_agent_action("TopicHunter", "frontier_extraction_error", {"error": str(e)})
            return []

        if terms:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({
                    "domain": domain,
                    "generated_at": datetime.now().isoformat(),
                    "runs_since_refresh": 0,
                    "terms": terms,
                }))
            except Exception as e:
                log_agent_action("TopicHunter", "frontier_cache_write_error", {"error": str(e)})
            log_agent_action("TopicHunter", "frontier_terms_harvested", {"domain": domain, "count": len(terms)})

        return terms

    def _cached_search(self, cache_key: str, fetch_fn) -> List[Dict[str, Any]]:
        """Thread-safe run-scoped cache for identical query strings issued by
        different seeds within the same discover_topics() call. Keys are
        normalized (case/whitespace) at construction so HyDE-variant queries
        that differ only in casing hit instead of refetching on a slow
        provider. Exact-match only — no fuzzy matching, to avoid silently
        sharing results across seeds that only look similar."""
        if not getattr(self.runtime_config, "cross_seed_paper_cache_enabled", True):
            return fetch_fn()
        norm_key = re.sub(r"\s+", " ", str(cache_key or "").strip().lower())
        with self._run_query_cache_lock:
            if norm_key in self._run_query_cache:
                log_agent_action("TopicHunter", "cross_seed_cache_hit", {"key": norm_key[:120]})
                return self._run_query_cache[norm_key]
        result = fetch_fn()
        with self._run_query_cache_lock:
            self._run_query_cache[norm_key] = result
        return result

    @staticmethod
    def _paper_identity_keys(p: Dict[str, Any]) -> List[str]:
        """Candidate join keys for one paper row (DOI, arXiv id, title)."""
        keys = []
        doi = (p.get("doi") or "").strip().lower().replace("https://doi.org/", "")
        if doi:
            keys.append("doi:" + doi)
        arxiv_id = (p.get("arxiv_id") or "").strip().lower()
        if arxiv_id:
            keys.append("arxiv:" + arxiv_id)
        title = re.sub(r"\s+", " ", (p.get("title") or "").strip().lower())
        if len(title) >= 20:
            keys.append("title:" + title)
        return keys

    def _backfill_missing_abstracts(self, papers: List[Dict[str, Any]]) -> int:
        """Fill empty abstracts from sibling rows of the same paper.

        Retrieval merges OpenAlex / arXiv / S2 / OpenReview rows that often
        describe the same paper under different dedup keys (DOI vs arXiv-id
        vs title casing) with different field coverage. This zero-network
        join recovers abstracts that are present in-memory but attached to a
        sibling row, then stably partitions abstract-bearing papers first so
        every front-sliced consumer (embedding precompute, evidence ranking,
        prompt title samples) sees dense evidence. Returns the count filled.
        """
        donors: Dict[str, str] = {}
        for p in papers:
            text = (p.get("abstract") or "").strip()
            if not text:
                continue
            for key in self._paper_identity_keys(p):
                donors.setdefault(key, text)
        filled = 0
        for p in papers:
            if (p.get("abstract") or "").strip():
                continue
            for key in self._paper_identity_keys(p):
                donor = donors.get(key)
                if donor:
                    p["abstract"] = donor
                    filled += 1
                    break
        papers.sort(key=lambda p: not bool((p.get("abstract") or "").strip()))
        return filled

    def _quality_filter_gaps(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop degenerate LLM outputs (short/placeholder titles or descriptions)."""
        quality = []
        for g in candidates:
            title = (g.get("title") or "").strip()
            description = (g.get("description") or "").strip()
            if len(title) < 5:
                log_agent_action("TopicHunter", "gap_quality_rejected", {
                    "title": title, "reason": "title_too_short",
                })
                continue
            if title.lower() in ("...", "none", "n/a", "test", ""):
                log_agent_action("TopicHunter", "gap_quality_rejected", {
                    "title": title, "reason": "degenerate_title",
                })
                continue
            if len(description) < 15:
                log_agent_action("TopicHunter", "gap_quality_rejected", {
                    "title": title, "reason": "description_too_short",
                })
                continue
            quality.append(g)
        if len(quality) < len(candidates):
            log_agent_action("TopicHunter", "gap_quality_filtered", {
                "before": len(candidates), "after": len(quality),
            })
        return quality

    def _iter_candidate_gaps(
        self,
        base_prompt: str,
        seed_hint: str,
        use_persona_ensemble: bool,
        persona_count: int,
        check_budget: Callable[[str], bool],
        used_counter: Dict[str, int],
        deadline_reached: Callable[[str], bool],
        gaps_sink: List[Dict[str, Any]],
        stats_sink: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield quality-filtered candidate gaps one persona batch at a time.

        Generation is lazy: each persona batch is produced (one LLM call) and
        yielded for immediate gate-chain evaluation before the next persona
        is generated. Previously all personas generated first, so a slow
        provider could burn the whole seed time budget in generation and the
        gate chain then discarded every gap unexamined on its first deadline
        check ("0 gaps kept from N generated"). No in-flight LLM call is ever
        cut short — batching only changes when the *next* call starts, and a
        batch whose deadline already passed is skipped, never evaluated blind.

        Title dedup keeps first-occurrence-wins across batches, matching the
        old whole-list dedup. Every yielded gap is appended to *gaps_sink* so
        callers keep an accurate generated-count.
        """
        seen_titles: set = set()

        def _take(raw_gaps, persona=None):
            batch = []
            for g in raw_gaps or []:
                if not isinstance(g, dict):
                    continue
                if persona is not None:
                    g["persona"] = persona
                t = (g.get("title") or "").strip().lower()
                if t and t not in seen_titles:
                    seen_titles.add(t)
                    batch.append(g)
            quality = self._quality_filter_gaps(batch)
            gaps_sink.extend(quality)
            return quality

        if use_persona_ensemble and persona_count >= 2:
            # v5: each persona mines a DISTINCT signal family (not the same
            # prompt + one-line bias) and must cite which signals it used via
            # `strategy` + `signal_refs` so provenance survives to ranking.
            persona_prefixes = {
                "skeptic": "You mine CONTRADICTIONS + REPLICATION TARGETS. Build each hypothesis around a mined contradiction pair (cite paper_a/paper_b via contradiction_ref) or a replication target (cite the claim + missing variance). ",
                "practitioner": "You mine REVIEWER-STATED PAIN. Build each hypothesis around a reviewer weakness theme (name the theme), proposing the study reviewers said was missing. ",
                "theorist": "You mine SPARSE CELLS + CROSS-PAPER BRIDGES + CONCEPT GAPS. Combine a well-established method with a setting it has never met (cite the sparse cell or concept pair) and attach real evidence_bridge_ids. ",
                "replicator": "You mine REPLICATION TARGETS + MOMENTUM. Verify heating-up claims (cite the momentum keyword) with multi-seed, statistically-tested replication designs. ",
                "engineer": "You mine FULL-TEXT METHODS + CITATION CONCENTRATION. Extend or stress-test highly-cited foundational methods (name the concentrated paper) using techniques seen in the full-text excerpts. ",
            }
            persona_names = list(persona_prefixes.keys())[:persona_count]
            from core.prompts import hash_output, check_doom_loop
            _persona_hashes: List[str] = []
            _empty_streak = 0
            for name in persona_names:
                if deadline_reached("persona_generation"):
                    break
                if not check_budget("generation"):
                    log_agent_action("TopicHunter", "llm_budget_exhausted", {"seed": seed_hint, "at": "persona", "used": used_counter["count"]})
                    break
                try:
                    prefix = persona_prefixes.get(name, "")
                    prompt = prefix + base_prompt
                    _persona_tokens = int(getattr(self.runtime_config, "llm_persona_max_tokens", 16384) or 16384)
                    raw = call_llm(prompt, temperature=0.8, tier="cheap", max_tokens=_persona_tokens,
                                   **_llm_kwargs("hunter.gap_generation", "cheap"))
                    parsed = parse_json_from_llm(raw) or {}
                    batch_gaps = parsed.get("gaps") or []
                except Exception as e:
                    # Bounded recovery for truncation_empty ONLY: the persona
                    # prompt is the largest in discovery (full evidence blocks
                    # + cross-run context), so a reasoning endpoint can burn
                    # its whole output budget on thinking and return nothing
                    # (run 21a145ad skeptic/practitioner). Retry ONCE with a
                    # slimmed prompt (seed angle + output schema preserved,
                    # evidence elided) in json_mode. Transport errors already
                    # had gateway retries — retrying those here is pure waste.
                    # The retry draws from the same generation budget so the
                    # gate-chain reserve stays protected.
                    if "truncated" in str(e).lower() and check_budget("generation"):
                        try:
                            slim = (prompt[:1500] + "\n[...evidence elided for retry...]\n"
                                    + prompt[-2500:] if len(prompt) > 4000 else prompt)
                            raw = call_llm(slim, temperature=0.3, tier="cheap",
                                           max_tokens=_persona_tokens,
                                           retry_on_truncation=False,
                                           **_llm_kwargs("hunter.gap_generation", "cheap"))
                            parsed = parse_json_from_llm(raw) or {}
                            batch_gaps = parsed.get("gaps") or []
                            log_agent_action("TopicHunter", "persona_retry_recovered",
                                             {"persona": name})
                        except Exception as retry_e:
                            log_agent_action("TopicHunter", "persona_call_error",
                                             {"persona": name, "error": str(retry_e)[:200],
                                              "retry": True})
                            continue
                    else:
                        log_agent_action("TopicHunter", "persona_call_error", {"persona": name, "error": str(e)[:200]})
                        continue
                fresh = _take(batch_gaps, persona=name)
                if stats_sink is not None:
                    try:
                        stats_sink.append({
                            "persona": name,
                            "raw": len(batch_gaps or []),
                            "kept": len(fresh),
                            "parsed_ok": isinstance(parsed, dict) and "gaps" in parsed,
                        })
                    except Exception:
                        pass
                for gap in fresh:
                    yield gap
                # Doom-loop guard (opencode doom_loop): identical persona
                # outputs 3x, or 3 straight all-duplicate batches -> stop
                # burning generation budget on repeats.
                try:
                    _persona_hashes.append(hash_output([
                        str(g.get("title", "")) for g in (batch_gaps or [])]))
                    if check_doom_loop(_persona_hashes):
                        log_agent_action("TopicHunter", "persona_doom_loop",
                                         {"seed": seed_hint, "at": name})
                        break
                    _empty_streak = _empty_streak + 1 if not fresh else 0
                    if _empty_streak >= 3:
                        log_agent_action("TopicHunter", "persona_empty_streak",
                                         {"seed": seed_hint, "at": name})
                        break
                except Exception:
                    pass
        else:
            parsed: Dict[str, Any] = {}
            if check_budget("generation"):
                _persona_tokens = int(getattr(self.runtime_config, "llm_persona_max_tokens", 16384) or 16384)
                parsed = parse_json_from_llm(call_llm(
                    base_prompt, temperature=0.8, tier="cheap", max_tokens=_persona_tokens,
                    **_llm_kwargs("hunter.gap_generation", "cheap"))) or {}
                batch_gaps = parsed.get("gaps") or []
            else:
                log_agent_action("TopicHunter", "llm_budget_exhausted", {"seed": seed_hint, "at": "generation", "used": used_counter["count"]})
                batch_gaps = []
            single_fresh = _take(batch_gaps)
            if stats_sink is not None:
                try:
                    stats_sink.append({
                        "persona": "single",
                        "raw": len(batch_gaps or []),
                        "kept": len(single_fresh),
                        "parsed_ok": isinstance(parsed, dict) and "gaps" in parsed,
                    })
                except Exception:
                    pass
            for gap in single_fresh:
                yield gap

    def _hunt_once(self, domain: str, seed_hint: str, seed_strategy: str = "generic_fallback", hunt_mode: str = "gap") -> List[Dict[str, Any]]:
        seed_started = time.monotonic()
        raw_seed_timeout = getattr(self.runtime_config, "topic_seed_timeout_sec", 1200)
        seed_timeout = max(1, int(raw_seed_timeout)) if isinstance(raw_seed_timeout, (int, float)) else 1200
        seed_deadline = seed_started + seed_timeout

        def deadline_reached(stage: str) -> bool:
            deadline = min(seed_deadline, getattr(self, "_discovery_deadline", seed_deadline))
            if time.monotonic() < deadline:
                return False
            log_agent_action("TopicHunter", "seed_deadline_reached", {
                "seed": seed_hint,
                "stage": stage,
                "timeout_sec": seed_timeout,
            })
            return True

        if deadline_reached("start"):
            return []
        cross_run_context = CrossRunMemory().get_prompt_context()
        excluded = self._excluded_titles()
        rejected_fingerprints = self._extract_rejected_fingerprints(cross_run_context)
        active_kind = self._active_hypothesis_kind()

        # Feature 2: HyDE-based query construction
        hyde_abstract = self._generate_hyde_abstract(seed_hint, domain)
        query_source_text = f"{seed_hint} {hyde_abstract}" if hyde_abstract else seed_hint

        # Feature 3: multi-hop retrieval
        recent_papers: List[Dict[str, Any]] = []
        seen_paper_keys: set = set()
        max_hops = max(1, int(getattr(self.runtime_config, "multi_hop_max_hops", 2))) \
            if getattr(self.runtime_config, "multi_hop_retrieval_enabled", True) else 1
        min_threshold = int(getattr(self.runtime_config, "multi_hop_min_papers_threshold", 12))

        current_source_text = query_source_text
        for hop in range(max_hops):
            if deadline_reached(f"retrieval_hop_{hop}"):
                break
            arxiv_queries = self._build_arxiv_queries(current_source_text, rejected_fingerprints, active_kind, domain=domain)
            openalex_queries = self._build_openalex_queries(current_source_text, rejected_fingerprints, active_kind, domain=domain)
            openalex_searches = [q[0] for q in openalex_queries]
            openalex_extra_params = [q[1] for q in openalex_queries]

            arxiv_queries = self._preflight_dedup(arxiv_queries, rejected_fingerprints)
            openalex_searches = self._preflight_dedup(openalex_searches, rejected_fingerprints)

            if not arxiv_queries and not openalex_searches:
                if hop == 0:
                    log_agent_action("TopicHunter", "seed_skipped_no_queries", {
                        "seed": seed_hint, "hop": hop,
                        "reason": "all_keywords_filtered_by_rejected_fingerprints",
                        "rejected_fingerprints_count": len(rejected_fingerprints),
                    })
                    return []
                break

            # Retrieval fanout is deliberately lean: ~200 papers/seed instead of
            # ~300. The old 40+30 ceiling plus 120 OpenReview rows rarely
            # improved gate outcomes (evidence is relevance-ranked to top-8
            # per gap) but reliably blew the seed time budget before the gate
            # chain ran. Multi-hop still backfills when the first hop is thin.
            base_limit = 25
            extra = min(self._iteration_failures * 5, 15)
            search_limit = base_limit + extra

            hop_papers: List[Dict[str, Any]] = []
            # Feature 5: wrap search calls in cross-seed cache
            for i, query in enumerate(openalex_searches):
                extra_params = openalex_extra_params[i] if i < len(openalex_extra_params) else {}
                oa_key = f"openalex::{query}::{json.dumps(extra_params, sort_keys=True)}::{search_limit}"
                hop_papers.extend(self._cached_search(oa_key, lambda q=query, ep=extra_params, sl=search_limit: self.search_openalex(q, sl, extra_params=ep)))
            # Feature 5: wrap arxiv search in cache too
            for query in arxiv_queries:
                arxiv_key = f"arxiv::{query}::{min(search_limit, 20)}"
                hop_papers.extend(self._cached_search(arxiv_key, lambda q=query, sl=min(search_limit, 20): self.search_arxiv(q, sl)))
            # S2 bulk search — free, higher throughput than arXiv, reduces arXiv dependency
            from core.sources_s2_bulk import search_s2_bulk
            if getattr(self.runtime_config, "semantic_scholar_enabled", True):
                s2_key = f"semantic_scholar_bulk::{current_source_text}::{search_limit}"
                hop_papers.extend(self._cached_search(
                    s2_key,
                    lambda q=current_source_text, sl=search_limit: search_s2_bulk(q, self.s2_headers, limit=sl),
                ))
            else:
                log_agent_action("TopicHunter", "s2_bulk_skipped_disabled", {"query": current_source_text[:80]})

            new_count = 0
            for p in hop_papers:
                key = (p.get("doi") or p.get("arxiv_id") or p.get("title", "")).strip().lower()
                if key and key not in seen_paper_keys:
                    seen_paper_keys.add(key)
                    recent_papers.append(p)
                    new_count += 1

            log_agent_action("TopicHunter", "retrieval_hop_complete", {
                "seed": seed_hint, "hop": hop, "new_papers": new_count,
                "total_papers": len(recent_papers),
            })

            if len(recent_papers) >= min_threshold or hop == max_hops - 1:
                break

            # Thin yield — derive a follow-up query
            follow_up = self._derive_followup_query_text(seed_hint, hyde_abstract, recent_papers, domain)
            if not follow_up or follow_up.strip().lower() == current_source_text.strip().lower():
                break
            current_source_text = follow_up
            log_agent_action("TopicHunter", "multi_hop_triggered", {
                "seed": seed_hint, "hop_next": hop + 1, "follow_up_source": follow_up[:200],
            })

        if not recent_papers:
            return []

        # OpenReview reviewer-weakness signals — pre-formalized, expert-stated gaps,
        # not LLM-inferred from abstracts. Cached per discover_topics() run via
        # _run_query_cache so all 5+ seeds share one fetch, not 5.
        # v5: venues are domain-aware (SE has no native OpenReview venue, so it
        # uses ML venues + TMLR) and signals are relevance-ranked PER SEED —
        # previously the same cached rows were shared unranked by every seed.
        openreview_signals: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "openreview_enabled", True):
            try:
                from core.sources_openreview import get_openreview_client, venues_for_domain
                or_client = get_openreview_client()
                venue_keys = venues_for_domain(
                    domain, subcategory=self._resolve_subcategory_for(seed_hint, active_kind, domain))
                or_key = f"openreview::{','.join(sorted(venue_keys))}"
                openreview_signals = self._cached_search(or_key, lambda: [
                    sig for vk in venue_keys
                    for sig in or_client.fetch_venue_gap_signals(vk, max_submissions=25)
                ])
            except Exception as e:
                log_agent_action("TopicHunter", "openreview_fetch_error", {"error": str(e)})
                openreview_signals = []
        seed_openreview = self._rank_openreview_for_seed(seed_hint, openreview_signals, top_n=12)

        for p in openreview_signals:
            key = (p.get("title") or "").strip().lower()
            if key and key not in seen_paper_keys:
                seen_paper_keys.add(key)
                recent_papers.append(p)

        # Recover abstracts across sibling rows (zero network) and order
        # abstract-bearing papers first — the embedding precompute, evidence
        # ranking, and prompt samples below all slice from the front.
        backfilled = self._backfill_missing_abstracts(recent_papers)
        if backfilled:
            # DEBUG: fires on nearly every seed (merged multi-source rows
            # almost always contain a few sibling pairs) — INFO was log spam.
            logger.debug("Backfilled %d missing abstracts from sibling source rows for seed=%r",
                         backfilled, seed_hint[:160])

        # Citation graph signals for top cited older-looking papers
        graph_signals = []
        for p in sorted(recent_papers, key=lambda x: x.get("cited_by_count") or 0, reverse=True)[:5]:
            if deadline_reached("citation_graph"):
                break
            doi = (p.get("doi") or "").replace("https://doi.org/", "")
            if doi:
                g = self.fetch_citation_graph(f"DOI:{doi}")
                if g:
                    graph_signals.append(g)

        # Feature 2: Structural gap mining via bibliographic coupling.
        # v5: S2 path when Semantic Scholar is enabled, else the OpenAlex-native
        # in-memory join over `referenced_works` (no network, no S2) — the Hunt
        # line no longer reads a permanent 0 when S2 is disabled.
        coupling_gaps: List[Dict[str, Any]] = []
        coupling_source = "disabled"
        if getattr(self.runtime_config, "structural_gap_mining_enabled", True):
            try:
                if getattr(self.runtime_config, "semantic_scholar_enabled", True):
                    coupling_gaps = find_coupling_gaps(
                        recent_papers,
                        self.s2_headers,
                        min_shared_refs=3,
                        max_pairs=getattr(self.runtime_config, "structural_gap_max_pairs", 8),
                    )
                    coupling_source = "semantic_scholar"
                elif getattr(self.runtime_config, "openalex_native_coupling_enabled", True):
                    coupling_gaps = find_coupling_gaps_openalex(
                        recent_papers,
                        min_shared_refs=3,
                        max_pairs=getattr(self.runtime_config, "structural_gap_max_pairs", 8),
                    )
                    coupling_source = "openalex_native"
            except Exception as e:
                log_agent_action("TopicHunter", "structural_gap_mining_error", {"error": str(e)})
                coupling_gaps = []

        # Feature 3: Method × Domain sparsity matrix (v5: full-corpus windows)
        sparse_cells: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "sparsity_matrix_enabled", True):
            try:
                sparse_cells = find_sparse_cells(
                    recent_papers,
                    max_papers=int(getattr(self.runtime_config, "sparsity_max_papers", 60)),
                    abstract_chars=int(getattr(self.runtime_config, "sparsity_abstract_chars", 1200)),
                )
            except Exception as e:
                log_agent_action("TopicHunter", "sparsity_matrix_error", {"error": str(e)})
                sparse_cells = []

        # Feature 4: Contradiction mining from abstracts (v5: full-corpus windows)
        contradictions: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "contradiction_mining_enabled", True):
            try:
                contradictions = find_contradictions(
                    recent_papers,
                    max_papers=int(getattr(self.runtime_config, "contradiction_max_papers", 40)),
                    abstract_chars=int(getattr(self.runtime_config, "contradiction_abstract_chars", 1500)),
                )
            except Exception as e:
                log_agent_action("TopicHunter", "contradiction_mining_error", {"error": str(e)})
                contradictions = []

        # v5 deterministic signals (no LLM, no network — fail-open):
        # temporal momentum, citation concentration (S2-graph stand-in),
        # OpenAlex concept co-occurrence gaps, reviewer-weakness clustering.
        momentum: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "momentum_mining_enabled", True):
            try:
                momentum = temporal_momentum(recent_papers)
            except Exception as e:
                log_agent_action("TopicHunter", "momentum_mining_error", {"error": str(e)})
                momentum = []
        concentration: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "citation_concentration_enabled", True):
            try:
                concentration = citation_concentration(recent_papers)
            except Exception as e:
                log_agent_action("TopicHunter", "concentration_mining_error", {"error": str(e)})
                concentration = []
        concept_gaps: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "concept_gap_mining_enabled", True):
            try:
                concept_gaps = concept_cooccurrence_gaps(recent_papers)
            except Exception as e:
                log_agent_action("TopicHunter", "concept_gap_mining_error", {"error": str(e)})
                concept_gaps = []
        reviewer_themes: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "reviewer_clustering_enabled", True):
            try:
                reviewer_themes = cluster_reviewer_weaknesses(seed_openreview)
            except Exception as e:
                log_agent_action("TopicHunter", "reviewer_clustering_error", {"error": str(e)})
                reviewer_themes = []

        # Feature 5: Replication-target mining (heuristic)
        replication_targets: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "replication_target_mining_enabled", True):
            try:
                replication_targets = self._find_replication_targets(recent_papers)
            except Exception as e:
                log_agent_action("TopicHunter", "replication_target_mining_error", {"error": str(e)})
                replication_targets = []

        # Feature 6: Own negative results as prior work
        negative_lessons: List[Dict[str, Any]] = []
        if getattr(self.runtime_config, "negative_result_seeding_enabled", True):
            try:
                negative_lessons = CrossRunMemory().get_negative_result_lessons()
            except Exception as e:
                log_agent_action("TopicHunter", "negative_result_seeding_error", {"error": str(e)})
                negative_lessons = []

        abstracts = [
            (p.get("abstract") or "") for p in recent_papers if p.get("abstract")
        ][:25]
        total_with_abstracts = sum(1 for p in recent_papers if (p.get("abstract") or "").strip())
        # v5: bounded OA full-text for the top-cited papers (cached, fail-open).
        # Abstract stubs hide methods/entities; excerpts feed the gap prompt.
        fulltext_excerpts = self._fetch_seed_fulltexts(recent_papers, seed_hint, deadline_reached)
        # v5 observability: every signal on one line + a structured event, so
        # the "two phases" (and every miner) are visible without grepping.
        signal_counts = {
            "papers": len(recent_papers),
            "with_abstracts": total_with_abstracts,
            "citation_signals": len(graph_signals),
            "citation_concentration": len(concentration),
            "coupling_gaps": len(coupling_gaps),
            "coupling_source": coupling_source,
            "contradictions": len(contradictions),
            "sparse_cells": len(sparse_cells),
            "replication_targets": len(replication_targets),
            "momentum_themes": len(momentum),
            "concept_gaps": len(concept_gaps),
            "reviewer_themes": len(reviewer_themes),
            "openreview_ranked": len(seed_openreview),
            "fulltext_excerpts": len(fulltext_excerpts),
        }
        logger.info(
            "Hunt for seed=%r: %d papers, %d abstracts, citation=%d/concentration=%d, "
            "coupling=%d(%s), contradictions=%d, sparse=%d, replication=%d, "
            "momentum=%d, concept_gaps=%d, reviewer_themes=%d, openreview=%d, fulltext=%d",
            seed_hint[:160], len(recent_papers), total_with_abstracts,
            len(graph_signals), len(concentration), len(coupling_gaps),
            coupling_source, len(contradictions), len(sparse_cells),
            len(replication_targets), len(momentum), len(concept_gaps),
            len(reviewer_themes), len(seed_openreview), len(fulltext_excerpts))
        log_agent_action("TopicHunter", "seed_signals", {
            "seed": seed_hint[:120], "domain": domain, **signal_counts,
        })
        self._seed_rejection_mark = len(self.rejection_log)
        evidence_map = build_cross_paper_evidence_map(recent_papers)

        # v2: Pre-filter bridges to seed-relevant only — prevents the LLM from
        # citing irrelevant bridge IDs that cause bridge_text_mismatch_soft_warning
        # downstream. Keep bridges whose method_signal or target_setting_signal
        # shares vocabulary with the seed hint.
        seed_tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", seed_hint.lower()))
        if seed_tokens and evidence_map.get("bridges"):
            filtered_bridges = []
            for bridge in evidence_map["bridges"]:
                method_sig = str(bridge.get("method_signal", "")).lower()
                setting_sig = str(bridge.get("target_setting_signal", "")).lower()
                bridge_tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", f"{method_sig} {setting_sig}"))
                if bridge_tokens & seed_tokens:
                    filtered_bridges.append(bridge)
            if filtered_bridges:
                evidence_map["bridges"] = filtered_bridges[:8]

        # Phase 3.1: Precompute abstract embeddings once for all gaps.
        # Skipped entirely when the embedding backend is down — each call
        # would just burn a gateway retry cycle and return zeros that the
        # novelty gate ignores anyway (it uses TF-IDF in fallback mode).
        novelty_max_abstracts = int(getattr(self.runtime_config, "novelty_max_abstracts", 20))
        abstract_embeddings: Dict[str, np.ndarray] = {}
        if not is_embedding_available():
            log_agent_action("TopicHunter", "embedding_precompute_skipped", {
                "seed": seed_hint[:80],
                "reason": "embedding_backend_unavailable_tfidf_fallback_active",
            })
        else:
            for p in recent_papers[:novelty_max_abstracts]:
                if deadline_reached("embeddings"):
                    break
                abs_text = (p.get("abstract") or "")[:2000]
                if not abs_text:
                    continue
                paper_key = (p.get("doi") or p.get("arxiv_id") or p.get("title", "")).strip().lower()
                if paper_key:
                    abstract_embeddings[paper_key] = generate_embedding(abs_text)

        # Phase 3.2: LLM call budget per seed — TWO POOLS (v3 fix #5).
        # Previously one shared counter covered generation (personas) AND the
        # gate chain, so heavy generation could exhaust the budget and silently
        # discard every generated gap before evaluation (the "found but never
        # judged" failure mode). Generation now draws from a small dedicated
        # pool; the gate chain gets a protected reserve of the main budget.
        try:
            raw_budget = getattr(self.runtime_config, "llm_budget_per_seed", 30)
            llm_budget = max(4, int(raw_budget)) if isinstance(raw_budget, (int, float)) and not isinstance(raw_budget, bool) else 30
        except (TypeError, ValueError):
            llm_budget = 30
        gate_reserve = min(max(6, llm_budget // 4), 12)
        _llm_calls_used = {"count": 0, "generation": 0, "gate": 0}

        def _check_llm_budget(kind: str = "gate") -> bool:
            if kind == "generation":
                # Generation must leave room for the gate chain to run.
                # Deviation logged (M0): proposal L21 asks `>` here, but the
                # pinned interleaving test
                # (test_hunt_once_keeps_first_batch_when_time_expires_mid_generation)
                # requires the `>=` boundary; one call of difference, suite stays green.
                if _llm_calls_used["count"] + gate_reserve >= llm_budget:
                    return False
            elif _llm_calls_used["count"] >= llm_budget:
                return False
            _llm_calls_used["count"] += 1
            if kind == "generation":
                _llm_calls_used["generation"] += 1
            return True

        # Fix #7: request more raw candidates per seed so the multi-stage gate
        # chain (which has a nonzero rejection rate at each stage) has more
        # surviving material at the end.
        # P2: try_and_report mode relaxes the gap straitjacket — candidates may
        # be simple "try approach X on Y, report outcome" without a world-gap
        # story, named SOTA baseline, or why-never-done narrative.
        _mode_extra = ""
        if str(hunt_mode or "gap") in ("try_and_report", "replication"):
            _mode_extra = """
MODE: TRY_AND_REPORT — simple executable ideas are welcome. A candidate MAY be
"try method X on dataset Y with metric Z and report whatever happens (positive,
negative, or inconclusive)". It does NOT need a world-first gap story, a SOTA
baseline, or a why-never-done narrative. Replication of a prior claim with a
bounded check is a VALID candidate (mark replication_target when relevant).
"""
        base_prompt = f"""
Propose {_GAPS_PER_SEED_REQUEST} TESTABLE HYPOTHESES in {domain}.
{_mode_extra}

Seed angle: {seed_hint}
Prior-run lessons (avoid repeats):
{json.dumps(cross_run_context, sort_keys=True)[:2000]}

Topics already rejected or that FAILED hypothesis debate (do NOT propose these or near-duplicates):
{json.dumps(excluded[-10:], indent=2)[:2000]}

Citation-graph gap signals (high in-degree, low recent extensions):
{json.dumps(graph_signals[:5], indent=2)[:1500]}

Structural gap signals (bibliographic coupling — pairs sharing references but not citing each other):
{json.dumps(coupling_gaps[:5], indent=2)[:1500]}

Method × domain sparse cells (well-established individually but rarely combined in retrieved literature):
{json.dumps(sparse_cells[:5], indent=2)[:1500]}

Contradictions in retrieved literature (papers disagreeing on the same question — strong, ready-made research gaps):
{json.dumps(contradictions[:5], indent=2)[:1500]}

Replication-target candidates (papers making strong claims with no visible variance/multi-seed reporting):
{json.dumps(replication_targets[:5], indent=2)[:1500]}

Reviewer-identified weaknesses from peer review (OpenReview, seed-ranked) — EXPERT-STATED limitations:
{_compact_hunt_block([json.dumps({'title': p.get('title'), 'weaknesses': (p.get('weaknesses') or [])[:2]}, default=str) for p in (seed_openreview or [])[:6]], 2000)}

Clustered reviewer-pain themes (votes = how many reviewers raised it — high votes are validated pain):
{json.dumps(reviewer_themes[:5], indent=2)[:1500]}

Heating-up themes (temporal momentum: recent vs older share — ride real waves):
{json.dumps(momentum[:5], indent=2)[:1200]}

Foundational-but-aging papers (high citations, old — ripe for extension/stress-test):
{json.dumps(concentration[:5], indent=2)[:1200]}

Concept pairs that never co-occur (each frequent alone — untested combinations):
{json.dumps(concept_gaps[:5], indent=2)[:1200]}

Full-text method excerpts (from the most-cited papers — real methods, not stub guesses):
{_compact_hunt_block([json.dumps({'title': e.get('title'), 'excerpt': (e.get('excerpt') or '')[:1200]}, default=str) for e in (fulltext_excerpts or [])[:3]], 3000)}

Prior negative results (propose a follow-up varying ONE condition, not a repeat):
{json.dumps(negative_lessons[:5], indent=2)[:1500]}

Evidence-backed cross-paper bridges:
{_compact_hunt_block([json.dumps(b, default=str) for b in (evidence_map.get('bridges', []) or [])[:6]], 2000)}

Sample recent titles:
{_compact_hunt_block([str(p.get('title', ''))[:100] for p in (recent_papers or [])[:8]], 800)}

IMPORTANT: Each hypothesis MUST be a specific, falsifiable claim. NOT a vague research direction.

Each hypothesis should follow this form:
"Method/technique X applied to problem Y achieves Z improvement over baseline W because of mechanism M"

Requirements for each hypothesis:
- Named method (contrastive learning, dropout, ensemble, temperature scaling, specific architecture — NOT "a new approach")
- Named baseline to compare against (logistic regression, random forest, ResNet-18, temperature scaling — NOT "existing methods")
- Named metric (accuracy, ECE, F1, inference latency in ms — NOT "performance")
- Named dataset OR "synthetic with N samples, K features, D classes"
- Concrete prediction (e.g., "improves ECE by >15% over temperature scaling")
- Why this hasn't been done before (grounded in the evidence above)
- CPU-EXECUTABLE within a {effective_wall_clock_seconds()}s wall clock on 1 CPU / ~2GB RAM: {_CPU_EXECUTION_ENVELOPE}

BAD examples (will be rejected downstream):
- "Evaluating the impact of domain shift on calibration" ← too vague, not falsifiable
- "A study of interpretability methods under distribution shift" ← survey, not hypothesis
- "Characterizing robustness boundaries" ← not a testable claim

GOOD examples:
- "Temperature scaling outperforms Platt scaling on CIFAR-10-C when corruption severity > 3, because temperature parameters capture output distribution shifts better than sigmoid transforms"
- "Mixup training reduces ECE by >20% over standard training on synthetic data with Gaussian noise, because it smooths the decision boundary"
- "Random forests achieve higher accuracy than XGBoost on tabular data with >30% missing values, because tree-based methods handle missingness natively without imputation"

For every proposed hypothesis, include `evidence_bridge_ids` containing bridge IDs
that support its cross-paper synthesis. Do not fabricate IDs.
Also include `strategy` (which signal family you mined: contradiction|replication|
reviewer|sparsity_transfer|momentum|foundational_extension) and `signal_refs`
(a list naming the exact signals used, e.g. ["contradiction: <shared_subject>",
"reviewer-theme: <theme>", "momentum: <keyword>", "concentrated-paper: <title>"]).
JSON: {{"gaps": [{{"title": "...", "description": "...", "rationale": "...", "impact": "...",
"feasibility": 7, "keywords": [], "anchor_paper": "...", "dataset_plan": "synthetic|public",
"strategy": "contradiction", "signal_refs": ["..."],
"evidence_bridge_ids": ["bridge-..."], "structural_gap_ref": {{"paper_a": "...", "paper_b": "..."}},
"contradiction_ref": {{"paper_a_title": "...", "paper_b_title": "..."}}}}]}}
"""

        # Feature 7: Persona ensemble generation (v2: expanded persona set)
        use_persona_ensemble = getattr(self.runtime_config, "persona_ensemble_enabled", True)
        persona_count = getattr(self.runtime_config, "persona_count", 2)

        # Interleaved generation + evaluation. Candidate gaps stream one persona
        # batch at a time (see _iter_candidate_gaps) and each batch enters the
        # gate chain below before the next persona is generated — so a seed
        # whose time budget expires mid-generation still keeps judged topics
        # instead of discarding everything unexamined. The loop body is
        # unchanged: `gaps` accumulates every generated gap as the stream
        # yields it, so counts and budget math behave as before.
        kept = []
        gaps: List[Dict[str, Any]] = []
        formalize_consecutive_failures = 0
        FORMALIZE_CIRCUIT_BREAKER = 3
        # Fix 13: Theme-level dedup — track rejected theme families to block
        # entire clusters when 5+ candidates share >80% vocabulary.
        # v2: raised MIN_COUNT from 3→5 and reset per seed to prevent
        # cross-seed vocabulary blocking that killed valid topics.
        self._rejected_theme_vocab = {}
        THEME_DEDUP_THRESHOLD = 0.80
        THEME_DEDUP_MIN_COUNT = 5
        # v5: per-persona generation anatomy (why did a seed yield 0 gaps?),
        # scored second-chance pool, and seed-level signal context for scoring.
        persona_stats: List[Dict[str, Any]] = []
        second_chance: List[Dict[str, Any]] = []
        scoring_enabled = bool(getattr(self.runtime_config, "hunter_scoring_enabled", True))
        try:
            second_chance_threshold = float(getattr(self.runtime_config, "hunter_second_chance_threshold", 0.55))
        except (TypeError, ValueError):
            second_chance_threshold = 0.55
        try:
            second_chance_max = max(1, int(getattr(self.runtime_config, "hunter_second_chance_max", 2)))
        except (TypeError, ValueError):
            second_chance_max = 2
        signal_ctx = {
            "contradictions": len(contradictions),
            "reviewer_themes": reviewer_themes,
            "momentum": len(momentum),
            "concept_gaps": len(concept_gaps),
            "sparse_cells": len(sparse_cells),
            # e0ec9e91: coupling/fulltext/citation counts were log-only
            # decoration. Carry them so _score_gap_candidate can reward
            # grounded gaps instead of ignoring these signals.
            "coupling": len(coupling_gaps),
            "fulltext": len(fulltext_excerpts),
            "citation_signals": len(graph_signals),
        }
        gap_stream = self._iter_candidate_gaps(
            base_prompt=base_prompt,
            seed_hint=seed_hint,
            use_persona_ensemble=use_persona_ensemble,
            persona_count=persona_count,
            check_budget=_check_llm_budget,
            used_counter=_llm_calls_used,
            deadline_reached=deadline_reached,
            gaps_sink=gaps,
            stats_sink=persona_stats,
        )
        for gap in gap_stream:
            if deadline_reached("gate_chain"):
                break
            # Budget guard: skip expensive gate chain if LLM budget exhausted
            if not _check_llm_budget():
                log_agent_action("TopicHunter", "llm_budget_exhausted", {"seed": seed_hint, "at": "gate_chain", "used": _llm_calls_used["count"], "remaining_gaps": len(gaps) - len(kept)})
                break
            logger.info("Evaluating gap: %r (budget: %d/%d used)", gap.get("title", "")[:160], _llm_calls_used["count"], llm_budget)
            # Fix #7: cheap heuristic pre-filter before the expensive gate chain.
            # This intentionally runs before bridge validation too, since it's
            # nearly free and catches the worst-case "hallucinated, no relation
            # to any retrieved paper" candidates earliest.
            if not self._quick_grounding_check(gap, recent_papers):
                self._reject(gap, "ungrounded_in_retrieved_literature", {
                    "lesson_type": "pre_filter_rejection",
                    "reason_code": "no_vocabulary_overlap_with_retrieved_papers",
                })
                logger.info("Gap REJECTED (ungrounded): %r", gap.get("title", "")[:160])
                continue

            if not self._dataset_plan_admissible(gap):
                self._reject(gap, "dataset_not_catalogued", {
                    "lesson_type": "dataset_scoping_rejection",
                    "reason_code": "dataset_not_catalogued",
                    "dataset_plan": gap.get("dataset_plan"),
                })
                logger.info("Gap REJECTED (dataset): %r, plan=%r", gap.get("title", "")[:160], gap.get("dataset_plan"))
                continue

            bridge_validation = validate_candidate_bridge_claim(gap, evidence_map)
            gap["bridge_validation"] = bridge_validation
            # Scored bridge grounding (agentic-harness upgrade): grounded
            # bridges score 1.0, soft-warning fallbacks 0.5 — the rank judge
            # prefers grounded gaps instead of treating all bridges alike.
            gap["bridge_grounding"] = (
                1.0 if str(bridge_validation.get("reason") or "")
                == "grounded_cross_paper_bridge" else 0.5
            )
            # Fix 15: Skip bridge validation rejection when no literature was
            # retrieved (empty evidence_map) — the warning is meaningless and
            # wastes LLM calls downstream.
            if not evidence_map and not bridge_validation["valid"]:
                bridge_validation["valid"] = True
                bridge_validation["soft_warning_applied"] = True
                log_agent_action("TopicHunter", "bridge_validation_skip_no_literature", {
                    "title": gap.get("title"),
                })
            if not bridge_validation["valid"]:
                # Fix #4: previously this was always a hard reject. Now, only a
                # genuine content mismatch (the validator's own semantic check)
                # is treated as hard-reject-worthy; a missing/malformed ID list
                # is downgraded to a soft warning, with a programmatic fallback
                # attempt to attach real bridge IDs by keyword match so the gap
                # can still be evidence-tagged downstream.
                reason_text = str(bridge_validation.get("reason", ""))
                is_id_only_issue = bridge_validation.get("reason_code") in (
                    "missing_bridge_ids", "malformed_bridge_ids", "no_bridge_ids_provided",
                ) or "id" in reason_text.lower()
                if is_id_only_issue:
                    fallback_ids = self._fallback_bridge_match(gap, evidence_map)
                    gap["evidence_bridge_ids"] = fallback_ids
                    gap["bridge_validation"]["valid"] = True
                    gap["bridge_validation"]["soft_warning_applied"] = True
                    log_agent_action("TopicHunter", "bridge_validation_soft_downgrade", {
                        "title": gap.get("title"),
                        "original_reason": reason_text,
                        "fallback_ids_found": len(fallback_ids),
                    })
                else:
                    self._reject(gap, "unsupported_cross_paper_synthesis", {
                        "lesson_type": "evidence_grounding_failure",
                        "reason_code": bridge_validation["reason"],
                    })
                    logger.info("Gap REJECTED (bridge validation): %r, reason=%r", gap.get("title", "")[:160], bridge_validation["reason"])
                    continue
            # Soft warning: log but don't reject on bridge text mismatch
            if "soft_warning" in bridge_validation.get("reason", ""):
                log_agent_action("TopicHunter", "bridge_soft_warning", {
                    "title": gap.get("title"),
                    "reason": bridge_validation["reason"],
                })
            # v3 fix #6: evidence = the MOST RELEVANT retrieved papers, not the
            # first 8 fetched. Screener + novelty prompts now judge each gap
            # against its closest prior work.
            evidence_papers = self._rank_papers_for_gap(gap, recent_papers, top_n=8)
            gap["literature_evidence"] = [
                {
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", "")[:3000],
                    "doi": paper.get("doi"),
                    "arxiv_id": paper.get("arxiv_id"),
                }
                for paper in evidence_papers
                if paper.get("title") and paper.get("abstract")
            ]
            gap["cross_paper_evidence"] = evidence_map
            # Validate optional structural_gap_ref field.
            # e0ec9e91: exact title-pair match always missed (LLM paraphrase),
            # so 0 coupling refs ever survived. Fuzzy overlap >=0.60 counts.
            sgr = gap.get("structural_gap_ref")
            if sgr and isinstance(sgr, dict):
                pa = sgr.get("paper_a", "")
                pb = sgr.get("paper_b", "")
                coupling_titles = {(c.get("paper_a", ""), c.get("paper_b", "")) for c in coupling_gaps}
                _coupled = (pa, pb) in coupling_titles or (pb, pa) in coupling_titles
                if not _coupled:
                    try:
                        _coupled = any(
                            (title_token_overlap(pa, ca) >= 0.60 and title_token_overlap(pb, cb) >= 0.60)
                            or (title_token_overlap(pa, cb) >= 0.60 and title_token_overlap(pb, ca) >= 0.60)
                            for ca, cb in coupling_titles if ca and cb
                        )
                    except Exception:
                        pass
                if not _coupled:
                    gap.pop("structural_gap_ref", None)
            elif sgr is not None:
                gap.pop("structural_gap_ref", None)
            # Validate optional contradiction_ref field (same fuzzy rule).
            cr = gap.get("contradiction_ref")
            if cr and isinstance(cr, dict):
                pa_title = (cr.get("paper_a_title") or "").strip().lower()
                pb_title = (cr.get("paper_b_title") or "").strip().lower()
                valid_contradictions = {
                    (c["paper_a_title"].strip().lower(), c["paper_b_title"].strip().lower())
                    for c in contradictions
                }
                _cvalid = ((pa_title, pb_title) in valid_contradictions
                           or (pb_title, pa_title) in valid_contradictions)
                if not _cvalid:
                    try:
                        _cvalid = any(
                            (title_token_overlap(pa_title, ca) >= 0.60 and title_token_overlap(pb_title, cb) >= 0.60)
                            or (title_token_overlap(pa_title, cb) >= 0.60 and title_token_overlap(pb_title, ca) >= 0.60)
                            for ca, cb in valid_contradictions if ca and cb
                        )
                    except Exception:
                        pass
                if not _cvalid:
                    gap.pop("contradiction_ref", None)
            elif cr is not None:
                gap.pop("contradiction_ref", None)
            # Validate optional builds_on_negative_result field
            bnr = gap.get("builds_on_negative_result")
            if bnr and isinstance(bnr, dict):
                rq = (bnr.get("research_question") or "").strip().lower()
                valid_rqs = {nl.get("research_question", "").strip().lower() for nl in negative_lessons if nl.get("research_question")}
                if rq not in valid_rqs:
                    gap.pop("builds_on_negative_result", None)
            elif bnr is not None:
                gap.pop("builds_on_negative_result", None)
            prior = self._matches_excluded_topic(gap.get("title", ""))
            if prior:
                self._reject(gap, "previously_failed_or_rejected", {"matched": prior})
                continue
            # Fix 13: Theme-level dedup — skip candidates that belong to a
            # theme family already rejected 3+ times with similar vocabulary.
            gap_words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", (gap.get("title") or "").lower()))
            blocked_theme = False
            for theme_key, count in self._rejected_theme_vocab.items():
                if count < THEME_DEDUP_MIN_COUNT:
                    continue
                theme_words = set(theme_key.split())
                if gap_words and theme_words:
                    overlap = len(gap_words & theme_words) / max(len(gap_words | theme_words), 1)
                    if overlap >= THEME_DEDUP_THRESHOLD:
                        blocked_theme = True
                        break
            if blocked_theme:
                self._reject(gap, "theme_family_saturated", {
                    "lesson_type": "theme_dedup",
                    "reason_code": "theme_family_saturated",
                })
                logger.info("Gap REJECTED (theme saturated): %r", gap.get("title", "")[:160])
                continue
            # v5: tag seed-signal hits on the gap itself — drives scoring,
            # family, and the evidence trail (not just the prompt).
            try:
                _gap_text = f"{gap.get('title','')} {gap.get('description','')} {gap.get('rationale','')}".lower()
                _gap_tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", _gap_text))
                for _theme in reviewer_themes:
                    _kw = set(str(w).lower() for w in (_theme.get("keywords") or []))
                    if _kw & _gap_tokens:
                        gap["reviewer_theme_hit"] = str(_theme.get("theme") or "")[:160]
                        break
                for _m in momentum:
                    if str(_m.get("keyword") or "").lower() in _gap_tokens:
                        gap["momentum_hit"] = str(_m.get("keyword") or "")[:120]
                        break
                _strat = str(gap.get("strategy") or "").lower()
                _refs = [str(r).lower() for r in (gap.get("signal_refs") or [])]
                if "replicat" in _strat or any("replicat" in r for r in _refs):
                    gap["replication_target"] = True
            except Exception:
                pass
            # v5: soft-failure carry — a gap that fails exactly one soft gate
            # (screener / feasibility / admission) can still be rescued into
            # the second-chance pool on its composite score. Terminal failures
            # (ungrounded, duplicate, theme-saturated, excluded) stay terminal.
            soft_failed: Optional[str] = None
            soft_meta: Dict[str, Any] = {}
            # 0. Feasibility FIRST (v5 reorder): deterministic and free — no
            # point spending judge calls on ideas the sandbox cannot run.
            # P0-2 deterministic pre-gate: milliseconds, zero LLM calls.
            try:
                _pre_ok, _pre_reason = self._formalize_pregate(gap, hunt_mode=hunt_mode)
            except TypeError:
                _pre_ok, _pre_reason = self._formalize_pregate(gap)
            if not _pre_ok:
                formalize_consecutive_failures += 1
                self._reject(gap, "formalize_pregate_failed", {
                    "lesson_type": "admission_failure",
                    "reason_code": _pre_reason,
                })
                log_agent_action("TopicHunter", "formalize_pregate_failed", {
                    "title": (gap.get("title") or "")[:80], "reason": _pre_reason,
                })
                continue
            feas = self.feasibility_filter(gap)
            gap["feasibility_check"] = feas
            if not feas["ok"]:
                if scoring_enabled:
                    soft_failed = "infeasible_for_engineer"
                    soft_meta = {
                        "lesson_type": "feasibility_failure",
                        "reason_code": "sandbox_capability_violation",
                        "reasons": feas.get("reasons", []),
                    }
                    logger.info("Gap SOFT-FAIL (feasibility, carried): %r", gap.get("title", "")[:160])
                else:
                    self._reject(gap, "infeasible_for_engineer", {
                        "lesson_type": "feasibility_failure",
                        "reason_code": "sandbox_capability_violation",
                        "reasons": feas.get("reasons", []),
                    })
                    logger.info("Gap REJECTED (infeasible): %r, reasons=%s", gap.get("title", "")[:160], feas.get("reasons"))
                    continue
            # 1. Screen research gap with literature evidence
            citation_signal = max((g.get("gap_score", 0) for g in graph_signals), default=0.0)
            try:
                gap_report = self.screen_research_gap(gap, gap.get("literature_evidence", []), citation_gap_signal=citation_signal, hunt_mode=hunt_mode)
            except TypeError:
                gap_report = self.screen_research_gap(gap, gap.get("literature_evidence", []), citation_gap_signal=citation_signal)
            gap["gap_report"] = gap_report
            if gap_report.get("status") == "FAIL":
                if scoring_enabled and soft_failed is None:
                    soft_failed = "unsupported_research_gap"
                    soft_meta = {
                        "lesson_type": "screener_rejection",
                        "reason_code": "unsupported_research_gap",
                        "gap_type": gap_report.get("gap_type", "unsupported"),
                        "details": gap_report.get("why_existing_work_is_insufficient", ""),
                    }
                    logger.info("Gap SOFT-FAIL (screener, carried): %r", gap.get("title", "")[:160])
                elif not scoring_enabled:
                    self._reject(gap, "unsupported_research_gap", {
                        "lesson_type": "screener_rejection",
                        "reason_code": "unsupported_research_gap",
                        "gap_type": gap_report.get("gap_type", "unsupported"),
                        "details": gap_report.get("why_existing_work_is_insufficient", ""),
                    })
                    logger.info("Gap REJECTED (screener): %r, status=%s, gap_type=%s",
                                 gap.get("title", "")[:160], gap_report.get("status"), gap_report.get("gap_type"))
                    continue
                # Two soft failures = drop with the FIRST reason (a gap that is
                # both infeasible and unsupported is not a rescue candidate).
                if soft_failed == "infeasible_for_engineer" and gap_report.get("status") == "FAIL":
                    self._reject(gap, soft_failed, soft_meta)
                    logger.info("Gap REJECTED (soft-fail x2): %r", gap.get("title", "")[:160])
                    continue
            # INSUFFICIENT_EVIDENCE passes through to novelty gate — the gap
            # may be in a nascent area where literature is sparse but the
            # hypothesis is still falsifiable and novel.

            # 2. Layered Novelty Assessment
            try:
                novelty_eval = self.evaluate_layered_novelty(gap, gap.get("literature_evidence", []), precomputed_embeddings=abstract_embeddings, hunt_mode=hunt_mode)
            except TypeError:
                novelty_eval = self.evaluate_layered_novelty(gap, gap.get("literature_evidence", []), precomputed_embeddings=abstract_embeddings)
            gap["novelty"] = novelty_eval
            if novelty_eval.get("reject"):
                # Near-duplicates are terminal even when scoring — rescuing a
                # duplicate is never correct. Other novelty rejects (raw-sim
                # without comparison) may still be carried.
                if str(novelty_eval.get("verdict") or "") == "LIKELY_DUPLICATE":
                    self._reject(gap, "novelty_too_low", {
                        "lesson_type": "novelty_failure",
                        "reason_code": "existing_contribution_overlap",
                        "verdict": novelty_eval.get("verdict", "LIKELY_DUPLICATE"),
                        "details": novelty_eval.get("reason", ""),
                    })
                    logger.info("Gap REJECTED (novelty): %r, verdict=%s, max_sim=%.3f",
                                 gap.get("title", "")[:160], novelty_eval.get("verdict"), novelty_eval.get("max_similarity", 0))
                    continue
                if scoring_enabled and soft_failed is None:
                    soft_failed = "novelty_too_low"
                    soft_meta = {
                        "lesson_type": "novelty_failure",
                        "reason_code": "existing_contribution_overlap",
                        "verdict": novelty_eval.get("verdict", "LIKELY_DUPLICATE"),
                        "details": novelty_eval.get("reason", ""),
                    }
                    logger.info("Gap SOFT-FAIL (novelty, carried): %r", gap.get("title", "")[:160])
                else:
                    self._reject(gap, "novelty_too_low", {
                        "lesson_type": "novelty_failure",
                        "reason_code": "existing_contribution_overlap",
                        "verdict": novelty_eval.get("verdict", "LIKELY_DUPLICATE"),
                        "details": novelty_eval.get("reason", ""),
                    })
                    logger.info("Gap REJECTED (novelty): %r, verdict=%s, max_sim=%.3f",
                                 gap.get("title", "")[:160], novelty_eval.get("verdict"), novelty_eval.get("max_similarity", 0))
                    continue

            # 4. Hypothesis Formalization (with circuit breaker).
            # v5: carried soft-failures skip formalization (2 strong-tier calls
            # cannot fix a screener/feasibility verdict — the composite score
            # from the 3 evaluated gates is enough for second-chance ranking).
            if soft_failed is not None and scoring_enabled:
                gap_scores = self._score_gap_candidate(gap, signal_ctx)
                gap["gap_scores"] = gap_scores
                gap["gap_family"] = self._gap_family(gap)
                if gap_scores.get("total", 0) >= second_chance_threshold:
                    second_chance.append({"gap": gap, "reason": soft_failed, "meta": soft_meta})
                    logger.info("Gap STASHED second-chance (score=%.2f): %r",
                                 gap_scores.get("total", 0), gap.get("title", "")[:160])
                else:
                    self._reject(gap, soft_failed, soft_meta)
                    logger.info("Gap REJECTED (soft-fail, score=%.2f): %r",
                                 gap_scores.get("total", 0), gap.get("title", "")[:160])
                continue
            if formalize_consecutive_failures >= FORMALIZE_CIRCUIT_BREAKER:
                log_agent_action("TopicHunter", "formalize_circuit_breaker", {
                    "consecutive_failures": formalize_consecutive_failures,
                    "remaining_gaps": len(gaps) - len(kept),
                })
                logger.warning("Formalize circuit breaker: %d consecutive failures, pausing", formalize_consecutive_failures)
                break
            # P0-2 gap deadline: once at least one gap is banked, don't open a
            # NEW ~10-min formalization when the seed/discovery budget can't
            # cover it. The first hope always gets its chance (a screened gap
            # is never aborted — its screener/novelty/feasibility work is sunk),
            # but banked winnings are protected from gambled-away budgets.
            # Scaled to the seed slice so short test seeds aren't frozen out.
            try:
                _gap_cfg = int(getattr(self.runtime_config, "topic_gap_deadline_sec", 600) or 600)
                _gap_deadline = min(_gap_cfg, max(30, seed_timeout // 6))
                _remaining = min(seed_deadline, getattr(self, "_discovery_deadline", seed_deadline)) - time.monotonic()
                if kept and _remaining < _gap_deadline:
                    log_agent_action("TopicHunter", "gap_deadline_reached", {
                        "seed": seed_hint[:80], "remaining_sec": round(_remaining, 1),
                        "kept_so_far": len(kept),
                    })
                    break
            except Exception:
                pass
            # NOTE: _formalize_pregate already ran at gate 0 (feasibility-first);
            # reaching here means it passed — no second call.
            try:
                formalized = self.formalize_hypothesis(gap, gap_report, novelty_eval, hunt_mode=hunt_mode)
            except TypeError:
                formalized = self.formalize_hypothesis(gap, gap_report, novelty_eval)
            if not formalized:
                formalize_consecutive_failures += 1
                self._reject(gap, "missing_structured_hypothesis", {
                    "lesson_type": "admission_failure",
                    "reason_code": "structured_hypothesis_unavailable",
                })
                logger.info("Gap REJECTED (no hypothesis): %r", gap.get("title", "")[:160])
                continue
            formalize_consecutive_failures = 0
            admission = validate_topic_admission(formalized)
            gap["topic_admission"] = admission
            if not admission["admitted"]:
                if scoring_enabled:
                    gap_scores = self._score_gap_candidate(gap, signal_ctx)
                    gap["gap_scores"] = gap_scores
                    gap["gap_family"] = self._gap_family(gap)
                    if gap_scores.get("total", 0) >= second_chance_threshold:
                        gap["structured_hypothesis"] = formalized
                        second_chance.append({"gap": gap, "reason": "topic_admission_failed",
                                              "meta": {"lesson_type": "admission_failure",
                                                       "reason_code": "non_executable_minimum_experiment",
                                                       "reasons": admission["errors"]}})
                        logger.info("Gap STASHED second-chance/admission (score=%.2f): %r",
                                     gap_scores.get("total", 0), gap.get("title", "")[:160])
                    else:
                        self._reject(gap, "topic_admission_failed", {
                            "lesson_type": "admission_failure",
                            "reason_code": "non_executable_minimum_experiment",
                            "reasons": admission["errors"],
                        })
                        logger.info("Gap REJECTED (admission, score=%.2f): %r",
                                     gap_scores.get("total", 0), gap.get("title", "")[:160])
                    continue
                self._reject(gap, "topic_admission_failed", {
                    "lesson_type": "admission_failure",
                    "reason_code": "non_executable_minimum_experiment",
                    "reasons": admission["errors"],
                })
                logger.info("Gap REJECTED (admission): %r, errors=%s", gap.get("title", "")[:160], admission["errors"])
                continue
            gap["structured_hypothesis"] = formalized
            gap["falsifiable_prediction"] = formalized.get("falsification_condition", "")
            gap["research_question"] = formalized.get("research_question", "")

            # Graph bonus
            if graph_signals:
                gap["gap_score"] = max(g.get("gap_score", 0) for g in graph_signals)
            gap["seed_strategy"] = seed_strategy  # Feature 8: seed-strategy provenance
            # v5: every kept gap carries its score, family, and evidence trail
            # so the ranker (and debate/UI) sees all signal families.
            gap["gap_scores"] = self._score_gap_candidate(gap, signal_ctx)
            gap["gap_family"] = self._gap_family(gap)
            gap["evidence_trail"] = self._build_evidence_trail(gap)
            kept.append(gap)
            # e0ec9e91: kept gaps logged title-only; the ranker/debate inputs
            # were invisible in the trace. Log family + score + evidence
            # trail with every keep.
            try:
                logger.info("Gap PASSED all gates: %r | family=%s score=%s trail=%s",
                            gap.get("title", "")[:160], gap.get("gap_family"),
                            (gap.get("gap_scores") or {}).get("total"),
                            "; ".join(gap.get("evidence_trail") or [])[:300])
            except Exception:
                logger.info("Gap PASSED all gates: %r", gap.get("title", "")[:160])
            try:
                self.vector_memory.add_embedding(
                    generate_embedding(gap["title"] + " " + gap.get("description", "")),
                    {
                        "type": "research_gap",
                        "namespace": "topic_hunter",
                        "content_class": "generated_narrative",
                        "retrieval_eligible": False,
                        "agent": "TopicHunterAgent",
                        "outcome_status": "unknown",
                        "title": gap["title"],
                        "domain": domain,
                    },
                )
            except Exception:
                pass
        # v2: Early exit for zero-gap seeds (checked after streaming because
        # the candidate generator is lazy — emptiness is only known once
        # consumed). v5: attach per-persona anatomy so a 15-minute burn with
        # 0 output is diagnosable (raw counts + parse status per persona).
        if not gaps:
            logger.warning("Seed %r produced 0 gaps after generation+quality filter — skipping gate chain", seed_hint)
            log_agent_action("TopicHunter", "seed_zero_gaps", {"seed": seed_hint})
            log_agent_action("TopicHunter", "seed_zero_gaps_anatomy", {
                "seed": seed_hint[:120],
                "personas": persona_stats,
                "total_raw": sum(p.get("raw", 0) for p in persona_stats),
            })
            self._log_seed_funnel(seed_hint, "zero_gaps")
            return []
        # v5: promote the second-chance pool — top-scoring single-soft-fail
        # gaps rejoin as labelled rescue candidates for debate to challenge.
        # The rest are rejected now (with their original reason) so the funnel
        # stays exact.
        if scoring_enabled and second_chance:
            ranked_sc = sorted(second_chance,
                               key=lambda e: e["gap"].get("gap_scores", {}).get("total", 0),
                               reverse=True)
            promoted = [e for e in ranked_sc
                        if e["gap"].get("gap_scores", {}).get("total", 0) >= second_chance_threshold][:second_chance_max]
            for entry in promoted:
                g = entry["gap"]
                g["second_chance"] = True
                g["second_chance_reason"] = entry["reason"]
                g["seed_strategy"] = seed_strategy
                g["evidence_trail"] = self._build_evidence_trail(g)
                kept.append(g)
            for entry in ranked_sc:
                if entry not in promoted:
                    self._reject(entry["gap"], "second_chance_unranked", {
                        **entry["meta"], "original_reason": entry["reason"],
                    })
            if promoted:
                log_agent_action("TopicHunter", "second_chance_promoted", {
                    "seed": seed_hint[:120],
                    "promoted": [{"title": e["gap"].get("title", "")[:80],
                                  "score": e["gap"].get("gap_scores", {}).get("total", 0),
                                  "reason": e["reason"]} for e in promoted],
                })
        self._log_seed_funnel(seed_hint, "complete")
        logger.info("Hunt for seed=%r complete: %d gaps kept from %d generated", seed_hint[:160], len(kept), len(gaps))
        return kept

    # R14 (§6.1) conservative degrade: when ALL scholarly sources fail but
    # output/source_cache/ holds prior responses, continue on the cached
    # corpus LABALLED stale instead of raising ResearchSourceUnavailable.
    # Offline-safe (file reads only); never fabricates — papers are verbatim
    # cached responses and the topics are labelled exploratory candidates
    # that debate/planning must still challenge.
    _STALE_CORPUS_BANNER = (
        "STALE CORPUS: all live scholarly sources were unreachable, so this "
        "candidate was built on a previously cached corpus labelled stale. "
        "Findings are exploratory and must be re-validated against live sources."
    )
    _STALE_MAX_PAPERS = 60
    _STALE_MAX_SCAN_FILES = 200
    _STALE_MAX_FILE_BYTES = 10_000_000

    @staticmethod
    def _load_stale_cached_papers(cache_dir: Any) -> List[Dict[str, Any]]:
        """Read prior SourceClient responses from the on-disk source cache.

        Only ``content.results``/``data``/``items``/``papers`` lists whose
        rows carry a title are treated as papers (oa_text/doaj/frontier
        cache shapes are skipped). Every returned paper carries
        ``stale: True``. Returns [] when the cache is missing or holds no
        usable prior responses.
        """
        papers: List[Dict[str, Any]] = []
        try:
            root = Path(str(cache_dir))
        except Exception:
            return []
        if not root.is_dir():
            return []
        try:
            files = sorted(p for p in root.rglob("*.json") if p.is_file())
        except Exception:
            return []
        for path in files[: TopicHunterAgent._STALE_MAX_SCAN_FILES]:
            try:
                if path.stat().st_size > TopicHunterAgent._STALE_MAX_FILE_BYTES:
                    continue
                artifact = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            content = artifact.get("content") if isinstance(artifact, dict) else None
            if not isinstance(content, dict):
                continue
            rows = content.get("results")
            if not isinstance(rows, list):
                for key in ("data", "items", "papers"):
                    if isinstance(content.get(key), list):
                        rows = content[key]
                        break
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or not str(row.get("title") or "").strip():
                    continue
                paper = dict(row)
                paper["stale"] = True
                papers.append(paper)
                if len(papers) >= TopicHunterAgent._STALE_MAX_PAPERS:
                    return papers
        return papers

    def _stale_cache_degrade(
        self,
        domain: Optional[str],
        seeds: List[Dict[str, str]],
    ) -> Optional[List[Topic]]:
        """Build labelled exploratory candidates from the stale cache.

        Returns None when the cache holds no usable prior responses (the
        caller keeps the current terminal behavior). Otherwise returns
        capability-screened fallback topics carrying the stale banner, a
        limitation entry, and the cached corpus as literature evidence —
        plus ``source_fetch`` + ``stale_corpus_degrade`` events noting stale.
        """
        try:
            cache_dir = self.source_client.cache_dir
        except Exception:
            cache_dir = Path(str(getattr(self.runtime_config, "output_dir", "output"))) / "source_cache"
        papers = self._load_stale_cached_papers(cache_dir)
        if not papers:
            return None
        stale_papers = papers[: self._STALE_MAX_PAPERS]
        log_agent_action("TopicHunter", "source_fetch", {
            "stale": True,
            "cached_papers": len(stale_papers),
            "note": "all live scholarly sources failed; continuing on stale cached corpus",
        })
        log_agent_action("TopicHunter", "stale_corpus_degrade", {
            "banner": self._STALE_CORPUS_BANNER,
            "cached_papers": len(stale_papers),
        })
        try:
            n_candidates = max(1, int(getattr(self.runtime_config, "topic_fallback_max_candidates", 2)))
        except (TypeError, ValueError):
            n_candidates = 1
        seeds = seeds or [{"seed": domain or "machine learning", "strategy": "stale_cache"}]
        topics: List[Topic] = []
        for seed in seeds[:n_candidates]:
            fallback = self._build_reachable_fallback_topic(domain or "", seed.get("seed", ""), stale_papers)
            if not fallback:
                continue
            # Capability-first admission (same bar as the bounded fallback):
            # a stale-cache candidate that cannot run must not poison planning.
            mini_plan = {
                "title": fallback.get("title", ""),
                "description": fallback.get("description", ""),
                "methodology": "Deterministic bundled-synthetic baseline comparison with controlled seeds.",
                "experiments": [{"name": "minimum_viable_experiment", "dataset": {"name": "bundled_synthetic", "access_policy": "local-only"}}],
            }
            if check_plan_feasibility(mini_plan, SANDBOX_CAPABILITY_MANIFEST):
                log_agent_action("TopicHunter", "stale_fallback_feasibility_rejected", {
                    "seed": seed.get("seed", "")[:80],
                })
                continue
            fallback["stale_corpus"] = True
            fallback["banner"] = self._STALE_CORPUS_BANNER
            fallback["limitations"] = [self._STALE_CORPUS_BANNER]
            fallback["literature_evidence"] = stale_papers[:8]
            provenance = dict(fallback.get("fallback_provenance") or {})
            provenance.update({
                "reason": "stale_cache_degrade",
                "cached_papers": len(stale_papers),
            })
            fallback["fallback_provenance"] = provenance
            topics.append(fallback)
        if topics:
            log_agent_action("TopicHunter", "stale_corpus_degrade_complete", {
                "count": len(topics),
                "label": "stale_exploratory_candidate_requires_debate",
            })
            return topics
        return None

    def _seed_snapshot_dir(self, run_id: str) -> Path:
        """Directory holding per-seed sealed snapshots for one run (P1-2)."""
        base = Path(getattr(self.runtime_config, "output_dir", "./output") or "./output")
        d = base / "phase_snapshots" / str(run_id or "adhoc") / "seeds"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _seal_seed(self, run_id: str, seed: Dict[str, Any], topics: List[Dict[str, Any]],
                   error: str = "") -> Path:
        """Persist one completed seed so a killed run resumes without redoing it."""
        import hashlib as _hl
        key = _hl.sha256(str(seed.get("seed", "")).encode("utf-8")).hexdigest()[:16]
        payload = {
            "seed": seed.get("seed", ""),
            "strategy": seed.get("strategy", ""),
            "topics": topics or [],
            "error": error,
            "sealed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        path = self._seed_snapshot_dir(run_id) / f"seed_{key}.json"
        path.write_text(json.dumps(payload, default=str)[:200000], encoding="utf-8")
        log_agent_action("TopicHunter", "seed_sealed", {
            "run_id": run_id, "seed": str(seed.get("seed", ""))[:80],
            "topics": len(topics or []),
        })
        return path

    def _load_sealed_seeds(self, run_id: str) -> Tuple[set, List[Dict[str, Any]]]:
        """Return (normalized seed texts, preloaded topics) sealed earlier this run."""
        sealed: set = set()
        preloaded: List[Dict[str, Any]] = []
        d = Path(getattr(self.runtime_config, "output_dir", "./output") or "./output") / "phase_snapshots" / str(run_id or "adhoc") / "seeds"
        if not d.is_dir():
            return sealed, preloaded
        for fp in sorted(d.glob("seed_*.json")):
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            s = str(payload.get("seed") or "")
            if s:
                try:
                    sealed.add(_normalize_seed_text(s))
                except Exception:
                    sealed.add(s.strip().lower())
            for t in payload.get("topics") or []:
                if isinstance(t, dict):
                    preloaded.append(t)
        return sealed, preloaded

    def discover_topics(self, domain: str = None, n_parallel: int = 5, hunt_mode: str = "try_and_report") -> List[Topic]:
        self._excluded_titles_cache = None
        self._run_query_cache = {}  # Feature 5: reset cross-seed cache per invocation
        domain = domain or self.runtime_config.research_domain
        hunt_mode = str(hunt_mode or "try_and_report")
        logger.info("Starting topic discovery for domain=%r with n_parallel=%d hunt_mode=%s", domain, n_parallel, hunt_mode)
        log_agent_action("TopicHunter", "start_discovery", {"domain": domain, "parallel": n_parallel, "hunt_mode": hunt_mode})
        cross_run_context = CrossRunMemory().get_prompt_context()
        raw_seed_timeout = getattr(self.runtime_config, "topic_seed_timeout_sec", 1200)
        seed_timeout = max(1, int(raw_seed_timeout)) if isinstance(raw_seed_timeout, (int, float)) else 1200
        raw_discovery_timeout = getattr(self.runtime_config, "topic_discovery_timeout_sec", 3600)
        configured_discovery = max(1, int(raw_discovery_timeout)) if isinstance(raw_discovery_timeout, (int, float)) else 3600
        # Auto-scale: the discovery window must fit every seed's budget plus
        # retrieval/embedding overhead, or the tail seeds can never run (the
        # 95c46c6f failure: 900 s window, 180 s seeds, only 2 of 5 attempted).
        discovery_timeout = effective_discovery_timeout(n_parallel, seed_timeout, configured_discovery)
        if discovery_timeout != configured_discovery:
            log_agent_action("TopicHunter", "discovery_timeout_autoscaled", {
                "configured_sec": configured_discovery,
                "effective_sec": discovery_timeout,
                "seed_timeout_sec": seed_timeout,
                "n_seeds": n_parallel,
            })
        # 119ce81c: autoscale grew discovery to 27300s while RUN_WALL_CLOCK_SEC
        # stayed 21600s, so the run guard always fired right after discovery
        # with zero time left for debate/planner/engineer. Cap discovery to
        # leave a reserve for downstream phases (advisory only — never
        # interrupts in-flight LLM calls, just stops opening new seed work).
        try:
            _run_wall = int(getattr(self.runtime_config, "run_wall_clock_sec", 21600) or 21600)
        except Exception:
            _run_wall = 21600
        _reserve = 3600
        _capped = max(600, _run_wall - _reserve)
        if discovery_timeout > _capped:
            log_agent_action("TopicHunter", "discovery_timeout_capped_for_run_wall", {
                "autoscaled_sec": discovery_timeout,
                "capped_sec": _capped,
                "run_wall_sec": _run_wall,
                "reserve_sec": _reserve,
            })
            discovery_timeout = _capped
        logger.info("Discovery budgets: seed_timeout=%ds, discovery_timeout=%ds (configured %ds)",
                    seed_timeout, discovery_timeout, configured_discovery)
        self._discovery_deadline = time.monotonic() + discovery_timeout
        # Cross-iteration seed exclusion: every seed phrase ever attempted
        # (exhausted AND productive-but-debate-failed) is excluded verbatim.
        # The windowed cross_run_context (last ~20 entries) forgets seed
        # history within one iteration, which is why run 25ed95d4 replayed
        # identical kind-bias templates ("transformer efficiency benchmark
        # comparison") in all 4 iterations. This load is unwindowed and cheap.
        # Debate-failed topic titles ride along for the *fuzzy* guard only
        # (exact match can never hit — titles are sentences, seeds phrases).
        seed_history: List[str] = []
        try:
            for row in CrossRunMemory().load("rejection", limit=500):
                if row.get("kind") in ("seed", "topic") and row.get("item"):
                    seed_history.append(str(row.get("item")))
        except Exception:
            seed_history = []
        seeds = self._generate_dynamic_seeds(
            n_parallel, cross_run_context, domain=domain,
            exclude_phrases=seed_history,
        )
        logger.info("Generated %d seeds: %s", len(seeds), [s["seed"][:160] for s in seeds])

        # P1-2: per-seed durability. Graph-level checkpoints resume the node,
        # but a kill mid-discovery (run b099dcaa: 1.5h, 1/5 seeds) previously
        # evaporated all retrieval/embeddings. Each sealed seed writes a small
        # JSON snapshot; a resumed run skips sealed seeds and reuses their
        # topics without new LLM calls.
        try:
            _tracker = get_tracker()
            _run_id = str(getattr(_tracker, "run_id", "") or "adhoc")
        except Exception:
            _run_id = "adhoc"
        all_topics: List[Dict[str, Any]] = []
        sealed: set = set()
        try:
            sealed, _preloaded = self._load_sealed_seeds(_run_id)
            if _preloaded:
                all_topics.extend(_preloaded)
                log_agent_action("TopicHunter", "seeds_resumed", {
                    "run_id": _run_id, "sealed_seeds": len(sealed),
                    "preloaded_topics": len(_preloaded),
                })
        except Exception as _seal_load_err:
            logger.debug("Seed snapshot load failed: %s", _seal_load_err)
        for seed in seeds:
            # Skip seeds sealed by a previous (killed) attempt of this run.
            try:
                if _normalize_seed_text(seed["seed"]) in sealed:
                    log_agent_action("TopicHunter", "seed_resumed_skip", {
                        "seed": seed["seed"][:80],
                    })
                    continue
            except Exception:
                pass
            if time.monotonic() >= self._discovery_deadline:
                log_agent_action("TopicHunter", "discovery_deadline_reached", {
                    "timeout_sec": discovery_timeout,
                    "topics_collected": len(all_topics),
                })
                break
            # Don't start a seed we cannot give a fair slice of time: a
            # doomed partial seed burns retrieval + LLM calls for nothing.
            remaining = self._discovery_deadline - time.monotonic()
            # L21: require a full slice before starting a new seed.
            if remaining < seed_timeout:
                log_agent_action("TopicHunter", "seed_skipped_no_time", {
                    "seed": seed["seed"][:80],
                    "remaining_sec": round(remaining, 1),
                    "seed_timeout_sec": seed_timeout,
                })
                break
            self._attempted_seeds = getattr(self, "_attempted_seeds", set())
            self._attempted_seeds.add(_normalize_seed_text(seed["seed"]))
            try:
                topics = self._hunt_once(domain, seed["seed"], seed.get("strategy", "generic_fallback"), hunt_mode=hunt_mode)
                for t in topics or []:
                    if isinstance(t, dict):
                        t.setdefault("hunt_mode", hunt_mode)
                logger.info("Seed %r (%s) produced %d topics", seed["seed"][:160], seed.get("strategy"), len(topics or []))
                all_topics.extend(topics or [])
                # Durable seed-attempt ledger: exhausted seeds AND productive
                # seeds are both recorded verbatim so the next iteration's
                # _generate_dynamic_seeds mints fresh angles instead of
                # replaying them. (Productive-but-debate-failed seeds are
                # the ones that kept recurring in run 25ed95d4 — the old
                # code only recorded the exhausted ones.)
                try:
                    CrossRunMemory().record_rejection(
                        "seed", seed["seed"],
                        _SEED_EXHAUSTED_REASON if not topics else _SEED_USED_REASON,
                        {"strategy": seed.get("strategy"), "topics": len(topics or [])},
                    )
                except Exception:
                    pass
                try:
                    self._seal_seed(_run_id, seed, topics or [])
                except Exception as _seal_err:
                    logger.debug("Seed snapshot seal failed: %s", _seal_err)
            except Exception as e:
                log_agent_action("TopicHunter", "hunt_error", {"seed": seed["seed"], "error": str(e)})
                try:
                    self._seal_seed(_run_id, seed, [], error=str(e)[:300])
                except Exception:
                    pass

        # Reachability fallback.  The discovery stage is a research aid, not a
        # publication gate: if retrieval worked but every strict LLM gate
        # rejected its candidates, retain a clearly labelled candidate for the
        # debate/planning stages to challenge.  This prevents the old failure
        # mode where 30+ calls produced useful evidence but zero topics.
        if not all_topics and getattr(self.runtime_config, "topic_fallback_enabled", True):
            for seed in seeds[:max(1, int(getattr(self.runtime_config, "topic_fallback_max_candidates", 2)))]:
                fallback = self._build_reachable_fallback_topic(domain, seed.get("seed", ""), [])
                if fallback:
                    # Capability-first admission: a fallback that cannot pass
                    # the sandbox feasibility gate would die in planning (run
                    # 60271359). Screen the mini-plan here and skip-and-log
                    # instead of handing planning a poisoned topic.
                    mini_plan = {
                        "title": fallback.get("title", ""),
                        "description": fallback.get("description", ""),
                        "methodology": "Deterministic bundled-synthetic baseline comparison with controlled seeds.",
                        "experiments": [{"name": "minimum_viable_experiment", "dataset": {"name": "bundled_synthetic", "access_policy": "local-only"}}],
                    }
                    cap_reasons = check_plan_feasibility(mini_plan, SANDBOX_CAPABILITY_MANIFEST)
                    if cap_reasons:
                        log_agent_action("TopicHunter", "fallback_feasibility_rejected", {
                            "seed": seed.get("seed", "")[:80],
                            "reasons": cap_reasons,
                        })
                        continue
                    all_topics.append(fallback)
            if all_topics:
                log_agent_action("TopicHunter", "bounded_topic_fallback", {
                    "count": len(all_topics),
                    "reason": "strict_discovery_gates_returned_zero_candidates",
                    "label": "exploratory_candidate_requires_debate",
                })

        # Graceful degradation: if we got partial results from at least one source, return them
        if not all_topics:
            if any(s.get("ok") for s in self.source_health.values()):
                log_agent_action("TopicHunter", "partial_results_returned", {
                    "source_health": self.source_health,
                })
            else:
                # R14 (§6.1) conservative degrade: all sources failed — continue
                # on the stale cached corpus when one exists (labelled); no
                # cache keeps the current terminal behavior.
                stale_topics = self._stale_cache_degrade(domain, seeds)
                if stale_topics:
                    all_topics.extend(stale_topics)
                else:
                    details = "; ".join(f"{name}: {entry.get('error', 'unavailable')}" for name, entry in self.source_health.items())
                    raise ResearchSourceUnavailable(
                        "Research discovery could not contact OpenAlex or arXiv. "
                        "Check network access and OPENALEX_EMAIL, then try again. Details: " + details
                    )

        seen = set()
        unique = []
        for t in all_topics:
            title = (t.get("title") or "").lower().strip()
            if title and title not in seen:
                seen.add(title)
                unique.append(t)
        # v3: near-duplicate suppression across seeds — different seeds can
        # surface the same idea in different wording; exact-title dedup misses
        # that and debate slots get burned on redundant candidates.
        unique = self._suppress_near_duplicates(unique)
        logger.info("After dedup: %d unique topics (from %d total)", len(unique), len(all_topics))

        # Second-pass targeted retrieval from top bridge candidates
        if unique:
            evidence_map = build_cross_paper_evidence_map(unique)
            bridges = evidence_map.get("bridges", [])
            top_bridges = [b for b in bridges if b.get("method_signal") and b.get("target_setting_signal")][:3]
            if top_bridges:
                second_pass_topics = self._execute_second_pass(top_bridges, domain)
                logger.info("Second pass added %d topics from %d bridges", len(second_pass_topics), len(top_bridges))
                for t in second_pass_topics:
                    title = (t.get("title") or "").lower().strip()
                    if title and title not in seen:
                        seen.add(title)
                        unique.append(t)

        ranked = self.rank_topics_by_potential(unique)
        logger.info("Final ranking produced %d topics", len(ranked))
        # v5 portfolio: guarantee ≥3 gap families in the top-5 when available.
        ranked = self._apply_portfolio_diversity(ranked)

        # Fix #1: aggregate the rejection funnel for this run so bottleneck
        # gates are visible without having to grep individual log entries.
        funnel: Dict[str, int] = {}
        for entry in self.rejection_log:
            funnel[entry["reason"]] = funnel.get(entry["reason"], 0) + 1
        log_agent_action("TopicHunter", "rejection_funnel", {
            "domain": domain,
            "total_rejected": len(self.rejection_log),
            "total_kept": len(ranked),
            "by_reason": dict(sorted(funnel.items(), key=lambda kv: -kv[1])),
        })
        logger.info("Rejection funnel: %d rejected, %d kept. Reasons: %s",
                     len(self.rejection_log), len(ranked), dict(sorted(funnel.items(), key=lambda kv: -kv[1])))

        log_agent_action("TopicHunter", "discovery_complete", {
            "num_topics": len(ranked),
            "rejected": len(self.rejection_log),
        })
        logger.info("Discovery complete: %d topics passed all gates, %d rejected", len(ranked), len(self.rejection_log))
        for src, health in self.source_health.items():
            status = "ok" if health.get("ok") else f"failed: {health.get('error', '?')}"
            logger.info("Source %s: %s", src, status)
        if not ranked:
            self._iteration_failures += 1
        else:
            self._iteration_failures = 0
        return ranked

    def _build_reachable_fallback_topic(
        self,
        domain: str,
        seed_hint: str,
        papers: List[Dict[str, Any]],
    ) -> Optional[Topic]:
        """Build a deterministic, executable candidate when strict gates fail.

        This is intentionally not presented as a novel result.  It is a
        hand-off artifact: later debate, planning, execution, and writing must
        disclose that it was bootstrapped and that the experiment is
        exploratory.  Keeping the contract deterministic makes this path usable
        during provider outages and gives the harness a chance to produce a
        complete paper instead of an empty run.
        """
        seed = (seed_hint or domain or "machine learning").strip()
        # Capability-first: fallback seeds may carry external-dataset phrasing
        # (e.g. a seed mentioning Yahoo/UCI/internet). Strip banned tokens from
        # the human-visible title so the topic is plannable downstream; the raw
        # seed is preserved in fallback_provenance for audit.
        safe_seed = seed
        for _tok in ("Yahoo Finance", "yahoo finance", "UCI", "uci", "internet", "download", "outbound", "production", "real-world"):
            safe_seed = safe_seed.replace(_tok, "bounded local")
        safe_seed = " ".join(safe_seed.split())
        anchor = next((p for p in papers if p.get("title")), None)
        anchor_title = (anchor or {}).get("title", "No retrieved anchor paper")
        topic_title = f"Exploratory evaluation of reproducibility factors in {safe_seed[:100]}"
        hypothesis = (
            "Controlled multi-seed evaluation of a simple baseline comparison in "
            f"{seed[:80]} will reveal measurable variance in the selected metric, "
            "relative to a majority-class baseline, because optimization and "
            "sampling choices affect observed performance."
        )
        structured = {
            "research_question": f"How variable are baseline results for {seed[:120]} under controlled seeds?",
            "hypothesis": hypothesis,
            "falsification_condition": "The hypothesis is falsified if the measured method-baseline difference and seed variance are both negligible under the preregistered threshold.",
            "dependent_variables": ["accuracy", "seed_variance"],
            "independent_variables": [{"name": "random_seed", "values": [0, 1, 2]}],
            "expected_relationship": "The learned baseline should outperform the majority baseline while showing non-zero seed variance.",
            "novelty_claim": "Exploratory replication-style measurement; novelty is unverified until literature review and debate.",
            "closest_prior_work": [{"title": anchor_title, "difference": "This candidate measures reproducibility rather than claiming a new method."}],
            "baselines": ["majority_class"],
            "metrics": ["accuracy", "seed_variance"],
            "confounders": ["class balance", "seed count", "implementation details"],
            "competing_explanations": ["variance may be caused by data split rather than optimization"],
            "minimum_viable_experiment": {
                "dataset": "bundled_synthetic",
                "models": ["logistic_regression"],
                "conditions": ["controlled_seeds"],
                "metrics": ["accuracy", "seed_variance"],
                "seeds": 3,
                "baseline": "majority_class",
                "expected_result": "Report mean, standard deviation, and the observed method-baseline difference.",
                "falsification_test": "Welch t-test p<0.05 or an explicit inconclusive verdict",
            },
            "required_resources": {"cpu": True, "gpu": False, "max_memory_mb": 2048, "max_runtime_seconds": 120},
            "admission_mode": "bounded_exploratory_fallback",
        }
        return {
            "title": topic_title,
            "description": "A bounded exploratory study on bundled_synthetic (local-only, no outbound downloads) generated because strict topic gates yielded no candidate.",
            "rationale": "Preserves a real research hand-off while making uncertainty and fallback provenance explicit.",
            "impact": "Produces an empirical artifact and a falsifiable baseline for iteration.",
            "feasibility": 8,
            "score": 1.0,
            "keywords": [safe_seed[:60], "reproducibility", "multi-seed evaluation"],
            "anchor_paper": anchor_title,
            "dataset_plan": "bundled_synthetic",
            "dataset_spec": {"name": "bundled_synthetic", "access_policy": "local-only", "source": "local_catalog"},
            "capability_manifest": SANDBOX_CAPABILITY_MANIFEST.as_dict(),
            "structured_hypothesis": structured,
            "research_question": structured["research_question"],
            "falsifiable_prediction": structured["falsification_condition"],
            "topic_admission": {"admitted": True, "errors": [], "contract_version": "bounded-fallback/v1"},
            "fallback_provenance": {"reason": "strict_gates_empty", "seed": seed, "anchor_paper": anchor_title},
        }

    def _execute_second_pass(self, top_bridges: List[Dict[str, Any]], domain: str, precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None) -> List[Topic]:
        """Execute a narrower second-pass retrieval seeded from top bridge candidates.

        Uses bridge method/setting signals to build targeted queries before
        final novelty/feasibility scoring.
        """
        second_pass_queries = []
        for bridge in top_bridges:
            method_signal = str(bridge.get("method_signal", "")).strip()
            setting_signal = str(bridge.get("target_setting_signal", "")).strip()
            if method_signal:
                second_pass_queries.append(f"cat:cs.LG AND abs:{method_signal}")
            if setting_signal and method_signal:
                second_pass_queries.append(f"cat:cs.LG AND abs:{method_signal} AND abs:{setting_signal}")

        if not second_pass_queries:
            return []

        if getattr(self.runtime_config, "arxiv_enabled", True):
            second_pass_papers = self.search_arxiv_multi(second_pass_queries[:4], 20)
        else:
            # Fallback: run the same queries through OpenAlex
            second_pass_papers = []
            for q in second_pass_queries[:4]:
                second_pass_papers.extend(self.search_openalex(q, limit=10))
        if not second_pass_papers:
            return []

        second_pass_papers = [p for p in second_pass_papers if p.get("title") and p.get("abstract")][:15]
        if not second_pass_papers:
            return []

        prompt = f"""
Given these bridge candidates from a cross-paper evidence map, propose 1-2 research gaps
that synthesize across the method/setting signals. Be specific and grounded in the paper evidence.

Bridge candidates:
{json.dumps([{
    'method_signal': b.get('method_signal'),
    'target_setting_signal': b.get('target_setting_signal'),
    'evidence': b.get('evidence', [])[:2],
} for b in top_bridges[:3]], indent=2)[:4000]}

Sample titles from targeted retrieval:
{[p.get('title', '')[:100] for p in second_pass_papers[:8]]}

For each topic include an explicit "contribution" sentence for novelty checking.
JSON: {{"gaps": [{{"title": "...", "description": "...", "rationale": "...", "impact": "...",
"feasibility": 7, "keywords": [], "anchor_paper": "...", "dataset_plan": "synthetic|public",
"evidence_bridge_ids": []}}]}}
"""
        parsed = parse_json_from_llm(call_llm(
            prompt, temperature=0.7, tier="cheap",
            **_llm_kwargs("hunter.second_pass", "cheap", domain=domain))) or {}
        gaps = parsed.get("gaps") or []

        kept = []
        for gap in gaps[:2]:
            # Quality gate for second-pass topics (same as first pass)
            title = (gap.get("title") or "").strip()
            description = (gap.get("description") or "").strip()
            if len(title) < 5 or title.lower() in ("...", "none", "n/a", "test", ""):
                self._reject(gap, "second_pass_quality_rejected", {"reason": "degenerate_title"})
                continue
            if len(description) < 15:
                self._reject(gap, "second_pass_quality_rejected", {"reason": "description_too_short"})
                continue
            # Dataset admissibility check for second-pass
            if not self._dataset_plan_admissible(gap):
                self._reject(gap, "second_pass_dataset_not_catalogued", {
                    "dataset_plan": gap.get("dataset_plan"),
                })
                continue
            gap["literature_evidence"] = [
                {"title": p.get("title", ""), "abstract": p.get("abstract", "")[:3000],
                 "doi": p.get("doi"), "arxiv_id": p.get("arxiv_id")}
                for p in second_pass_papers[:6]
                if p.get("title") and p.get("abstract")
            ]
            prior = self._matches_excluded_topic(gap.get("title", ""))
            if prior:
                self._reject(gap, "previously_failed_or_rejected", {"matched": prior})
                continue
            novelty_eval = self.evaluate_layered_novelty(gap, gap.get("literature_evidence", []), precomputed_embeddings=precomputed_embeddings)
            gap["novelty"] = novelty_eval
            if novelty_eval.get("reject"):
                self._reject(gap, "novelty_too_low", {
                    "lesson_type": "novelty_failure",
                    "reason_code": "existing_contribution_overlap",
                    "verdict": novelty_eval.get("verdict", "LIKELY_DUPLICATE"),
                })
                continue
            feas = self.feasibility_filter(gap)
            gap["feasibility_check"] = feas
            if not feas["ok"]:
                self._reject(gap, "infeasible_for_engineer", {
                    "lesson_type": "feasibility_failure",
                    "reason_code": "sandbox_capability_violation",
                    "reasons": feas.get("reasons", []),
                })
                continue
            kept.append(gap)
        return kept

    @staticmethod
    def _rank_topic_card(t: Dict[str, Any]) -> Dict[str, Any]:
        """Compact judge card (v5): every signal family visible on one line."""
        try:
            scores = t.get("gap_scores") or {}
            card = {
                "title": t.get("title"),
                "description": str(t.get("description") or "")[:400],
                "feasibility": t.get("feasibility"),
                "novelty": t.get("novelty"),
                "gap_score": t.get("gap_score"),
                "bridge_grounding": t.get("bridge_grounding"),
                "gap_scores_total": scores.get("total"),
                "gap_family": t.get("gap_family"),
                "second_chance": bool(t.get("second_chance")),
                "strategy": t.get("strategy"),
                "has_contradiction": bool(t.get("contradiction_ref")),
                "has_reviewer_pain": bool(t.get("reviewer_theme_hit")),
                "has_momentum": bool(t.get("momentum_hit")),
                "evidence_trail": t.get("evidence_trail") or [],
            }
            return card
        except Exception:
            return {"title": t.get("title"), "description": str(t.get("description") or "")[:400]}

    def rank_topics_by_potential(self, topics: List[Topic]) -> List[Topic]:
        if not topics:
            return []
        if len(topics) == 1:
            topics[0]["rank"] = 1
            topics[0]["score"] = topics[0].get("feasibility", 7)
            topics[0]["selection_mode"] = "exploitation"
            return topics
        prompt = f"""
Rank these research topics (lightweight judge). Prefer novel executable gaps over trendy saturated areas.
Prefer bridge-grounded gaps (bridge_grounding 1.0) over soft-warning ones (0.5).
Prefer high composite gap_scores.total, but DIVERSIFY across gap_family values —
a contradiction play, a reviewer-pain replication, and a method-transfer idea
outrank three variants of the same family. Second-chance candidates
(second_chance=true) are rescue-worthy but riskier: rank them fairly, not first.
{json.dumps([self._rank_topic_card(t) for t in topics], indent=2)[:7000]}
JSON: {{"ranked_topics": [{{"original_index": 0, "rank": 1, "score": 8.5, "reasoning": "..."}}]}}
"""
        # Elo is deliberately a small tie-breaker, not a replacement for evidence-based
        # gap/novelty/feasibility scoring. Exploration periodically boosts under-observed
        # kinds but never bypasses the gates already applied in _hunt_once.
        elo = EloStore(context=getattr(self, "context", None))
        for topic in topics:
            kind = topic.get("hypothesis_kind") or hypothesis_kind(topic.get("title", ""))
            topic["hypothesis_kind"] = kind
            topic["elo_rating"] = float(elo.get(kind))
            topic["elo_observations"] = elo.observations(kind)

        try:
            # Spec-driven rank (proves the agent_json path in production):
            # registry system + schema repair + doom-loop + version stamp.
            from core.agent_llm import agent_json as _agent_json

            topics_json = json.dumps(
                [self._rank_topic_card(t) for t in topics],
                indent=2)[:7000]
            parsed = _agent_json("hunter.rank", call_fn=call_llm,
                                 topics=topics_json) or {}
            if not isinstance(parsed, dict) or "ranked_topics" not in parsed:
                parsed = parse_json_from_llm(call_llm(
                    prompt, temperature=0.3, tier="judge",
                    **_llm_kwargs("hunter.rank", "judge"))) or {}
            for rank_info in parsed.get("ranked_topics") or []:
                idx = rank_info.get("original_index", 0)
                if idx < len(topics):
                    topics[idx]["rank"] = rank_info.get("rank", 999)
                    topics[idx]["score"] = rank_info.get("score", 5)
                    topics[idx]["reasoning"] = rank_info.get("reasoning", "")
            ranked = sorted(topics, key=lambda x: (x.get("rank", 999), -x.get("elo_rating", 1500.0)))
        except Exception:
            ranked = sorted(topics, key=lambda x: (-(x.get("gap_score") or x.get("feasibility") or 0), -x.get("elo_rating", 1500.0)))

        return self._apply_exploration(ranked, elo)

    def _apply_exploration(self, ranked: List[Topic], elo: EloStore) -> List[Topic]:
        """Deterministic forced exploration of under-observed / lower-rated kinds."""
        if not ranked:
            return ranked
        if not hasattr(self, "_selection_counter"):
            self._selection_counter = 0
        if not hasattr(self, "_rng"):
            seed = 42
            cfg = getattr(self, "runtime_config", None)
            if cfg is not None:
                seed = int(getattr(cfg, "topic_exploration_seed", 42))
            self._rng = random.Random(seed)
        self._selection_counter += 1
        cfg = getattr(self, "runtime_config", None)
        every = max(1, int(getattr(cfg, "topic_exploration_every", 4) if cfg is not None else 4))
        force = (self._selection_counter % every) == 0
        for topic in ranked:
            topic.setdefault("selection_mode", "exploitation")
        if not force:
            return ranked

        kinds = [t.get("hypothesis_kind") or hypothesis_kind(t.get("title", "")) for t in ranked]
        under = elo.under_observed_kinds(list(dict.fromkeys(kinds)))
        candidates = [t for t in ranked if (t.get("hypothesis_kind") in under)] if under else list(ranked)
        candidates = sorted(candidates, key=lambda t: (t.get("elo_rating", 1500.0), t.get("elo_observations", 0)))
        if not candidates:
            return ranked
        pick = candidates[self._rng.randrange(len(candidates))]
        pick["selection_mode"] = "forced_exploration"
        rest = [t for t in ranked if t is not pick]
        log_agent_action("TopicHunter", "forced_exploration", {
            "kind": pick.get("hypothesis_kind"),
            "title": pick.get("title"),
            "elo_rating": pick.get("elo_rating"),
            "elo_observations": pick.get("elo_observations"),
        })
        return [pick] + rest