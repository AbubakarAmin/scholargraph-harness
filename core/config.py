"""
Configuration module for the multi-agent research system.
Handles environment variables, API keys, and system settings.
"""

import logging
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Load environment variables from project-root .env (cwd-independent)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

# Force matplotlib to use non-interactive Agg backend process-wide to prevent
# Tcl_AsyncDelete: async handler deleted by the wrong thread on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

# Point matplotlib's cache at a writable temp dir. Without this, sandboxed or
# service contexts without a valid HOME fall back to /root/.config and every
# import prints "mkdir -p failed ... Read-only file system" noise into logs.
import tempfile as _tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(_tempfile.gettempdir(), "scholargraph-mpl"))

class Config(BaseSettings):
    """Configuration class for the research system."""

    # --- LLM Provider ---
    # gemini | openai | openai_compatible
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")

    # Gemini
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    # Gemma 4 is hosted through the Gemini API.  Keep the provider name as
    # ``gemini``: it preserves the existing Gemini SDK, key, gateway, and
    # fallback paths while allowing every agent to use the selected Gemma
    # model through the normal tier resolver.
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")
    # Gemma 4 supports only ``minimal`` (off) and ``high`` (on).  High is the
    # default for ScholarGraph's research, coding, and referee workflows.
    # This setting is ignored for non-Gemma models, so switching providers or
    # selecting another Gemini model remains fully backward compatible.
    gemma_thinking_level: str = os.getenv("GEMMA_THINKING_LEVEL", "high")
    # --- Embeddings: GEMINI-ONLY POLICY (owner-locked, 2026-10-03) ---
    # Chat may use any provider below, but embeddings ALWAYS use
    # gemini-embedding-2 at native 3072 dims, no matter the chat provider.
    # memory/vector_db (FAISS) and every stored vector are 3072-wide Gemini
    # vectors; another provider/model would corrupt similarity search, and
    # memory.py rebuilds (wiping stored embeddings) on any dim mismatch.
    # Do NOT add an OpenAI embeddings route: there is no `embeddings.create`
    # call anywhere in this codebase by design (see core/llm.py policy note).
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))

    # OpenAI / OpenAI-compatible (OpenRouter, Groq, Together, local vLLM, Ollama, etc.)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    # NOTE: legacy display key only (shown in web /api/config). It is NEVER
    # used for embeddings — see GEMINI-ONLY POLICY above + core/llm.py.

    # Cost-aware routing: cheap model for lookups, strong for debate/writing/verify
    llm_model_cheap: str = os.getenv("LLM_MODEL_CHEAP", "")
    llm_model_strong: str = os.getenv("LLM_MODEL_STRONG", "")
    llm_model_judge: str = os.getenv("LLM_MODEL_JUDGE", "")
    llm_model_fallback: str = os.getenv("LLM_MODEL_FALLBACK", "")

    # Provider-level fallback: when primary provider has connection errors,
    # try the other provider. LLM_FALLBACK_PROVIDER=gemini or openai_compatible.
    llm_fallback_provider: str = os.getenv("LLM_FALLBACK_PROVIDER", "")
    llm_fallback_model: str = os.getenv("LLM_FALLBACK_MODEL", "")

    # Optional secondary judge models (comma-separated) for ensemble
    ensemble_judge_models: str = os.getenv("ENSEMBLE_JUDGE_MODELS", "")

    # External APIs
    semantic_scholar_api_key: Optional[str] = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    openalex_email: str = os.getenv("OPENALEX_EMAIL", "researcher@example.com")

    # OpenReview
    openreview_username: str = os.getenv("OPENREVIEW_USERNAME", "")
    openreview_password: str = os.getenv("OPENREVIEW_PASSWORD", "")
    openreview_enabled: bool = os.getenv("OPENREVIEW_ENABLED", "true").lower() == "true"

    # HuggingFace (for dataset access)
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")
    huggingface_max_download_mb: int = int(os.getenv("HUGGINGFACE_MAX_DOWNLOAD_MB", "100"))
    openreview_venues: list = ["iclr2025", "iclr2026", "neurips2025"]
    arxiv_enabled: bool = os.getenv("ARXIV_ENABLED", "true").lower() == "true"
    openalex_enabled: bool = os.getenv("OPENALEX_ENABLED", "true").lower() == "true"
    semantic_scholar_enabled: bool = os.getenv("SEMANTIC_SCHOLAR_ENABLED", "true").lower() == "true"

    # System
    research_domain: str = os.getenv("RESEARCH_DOMAIN", "computer_science")
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))
    supervisor_threshold: float = float(os.getenv("SUPERVISOR_THRESHOLD", "8.5"))
    debate_pass_threshold: float = float(os.getenv("DEBATE_PASS_THRESHOLD", "7.0"))
    debate_min_rounds: int = int(os.getenv("DEBATE_MIN_ROUNDS", "2"))
    debate_max_rounds: int = int(os.getenv("DEBATE_MAX_ROUNDS", "5"))
    novelty_similarity_reject: float = float(os.getenv("NOVELTY_SIMILARITY_REJECT", "0.92"))
    experiment_seeds: int = int(os.getenv("EXPERIMENT_SEEDS", "3"))
    experiment_branch_count: int = int(os.getenv("EXPERIMENT_BRANCH_COUNT", "3"))
    engineer_max_attempts: int = int(os.getenv("ENGINEER_MAX_ATTEMPTS", "4"))
    # M4: total engineering wall-time cap (seconds) across all experiments in
    # one engineering phase — bounds multi-seed batches when per-seed
    # timeouts alone would still sum past the run budget.
    engineering_wall_clock_sec: int = int(os.getenv("ENGINEERING_WALL_CLOCK_SEC", "7200"))
    # Engineering Arena (opencode-backed experiment authoring): per-run
    # isolated task dirs where a headless `opencode run` agent implements
    # each experiment; trusted execution happens in Docker. "auto" tries opencode first and
    # falls back to the legacy single-shot LLM path when opencode is
    # unavailable; "opencode" hard-fails instead of falling back; "legacy"
    # keeps the pre-arena behaviour byte-identical.
    arena_backend: str = os.getenv("ENGINEERING_BACKEND", "auto")  # "auto" | "opencode" | "legacy"
    arena_dir: str = os.getenv("ARENA_DIR", "./engineering_arena")
    arena_session_timeout_sec: int = int(os.getenv("ARENA_SESSION_TIMEOUT_SEC", "1800"))
    arena_max_rounds: int = int(os.getenv("ARENA_MAX_ROUNDS", "3"))
    arena_model: str = os.getenv("ARENA_MODEL", "")
    arena_agent: str = os.getenv("ARENA_AGENT", "build")
    arena_retention_days: int = int(os.getenv("ARENA_RETENTION_DAYS", "7"))
    arena_cleanup_stale: bool = os.getenv("ARENA_CLEANUP_STALE", "true").lower() == "true"
    # 211505fb: when the LLM gate is holding, empty codegen burns all
    # attempts in <1s. Wait (bounded) for breaker recovery instead.
    engineer_gate_wait_sec: int = int(os.getenv("ENGINEER_GATE_WAIT_SEC", "120"))
    # Multi-type harness (§3, §9): typed graph + paper type + parallelism flags.
    typed_graph_enabled: bool = os.getenv("TYPED_GRAPH_ENABLED", "true").lower() == "true"
    paper_type_default: str = os.getenv("PAPER_TYPE_DEFAULT", "experimental")
    parallel_phases_enabled: bool = os.getenv("PARALLEL_PHASES_ENABLED", "false").lower() == "true"
    # Budgets (locked §5.3).
    run_max_llm_calls: int = int(os.getenv("RUN_MAX_LLM_CALLS", "2000"))
    run_max_tokens: int = int(os.getenv("RUN_MAX_TOKENS", "8000000"))
    run_max_usd: float = float(os.getenv("RUN_MAX_USD", "500.0"))
    run_wall_clock_sec: int = int(os.getenv("RUN_WALL_CLOCK_SEC", "21600"))
    # Review sources (M2, §5.1/§9): default ON — all three are free,
    # license-respecting, gateway-paced (breaker + AIMD + cache), and the M2
    # review pipeline degrades without them (abstract-only). Set false to run
    # abstract-only deliberately; the flag-off path stays tested.
    unpaywall_enabled: bool = os.getenv("UNPAYWALL_ENABLED", "true").lower() == "true"
    pmc_enabled: bool = os.getenv("PMC_ENABLED", "true").lower() == "true"
    doaj_guard_enabled: bool = os.getenv("DOAJ_GUARD_ENABLED", "true").lower() == "true"
    # Field-type honesty latch (locked §1, §9).
    allow_synthetic_respondents: bool = os.getenv("ALLOW_SYNTHETIC_RESPONDENTS", "false").lower() == "true"
    # Token/USD accounting (§8.8.3): JSON override for core.pricing.PRICES.
    llm_price_table_json: str = os.getenv("LLM_PRICE_TABLE_JSON", "")
    debug_mode: bool = os.getenv("DEBUG_MODE", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Memory / paths
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./memory/vector_db")
    memory_size: int = int(os.getenv("MEMORY_SIZE", "10000"))
    cross_run_memory_path: str = os.getenv("CROSS_RUN_MEMORY_PATH", "./memory/cross_run.jsonl")
    run_log_path: str = os.getenv("RUN_LOG_PATH", "./output/run_scratchpad.jsonl")
    run_events_path: str = os.getenv("RUN_EVENTS_PATH", "./output/run_events.jsonl")
    elo_ratings_path: str = os.getenv("ELO_RATINGS_PATH", "./memory/elo_ratings.json")
    elo_prior_rating: float = float(os.getenv("ELO_PRIOR_RATING", "1500"))
    elo_shrinkage_k: float = float(os.getenv("ELO_SHRINKAGE_K", "8"))
    elo_min_observations: int = int(os.getenv("ELO_MIN_OBSERVATIONS", "5"))
    topic_exploration_every: int = int(os.getenv("TOPIC_EXPLORATION_EVERY", "4"))
    topic_exploration_seed: int = int(os.getenv("TOPIC_EXPLORATION_SEED", "42"))
    checkpoint_path: str = os.getenv("CHECKPOINT_PATH", "./memory/checkpoints.sqlite")
    research_db_path: str = os.getenv("RESEARCH_DB_PATH", "./memory/research_ledger.sqlite")

    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    draft_versions_dir: str = os.getenv("DRAFT_VERSIONS_DIR", "./output/draft_versions")
    debate_log_path: str = os.getenv("DEBATE_LOG_PATH", "./output/debate_log.json")
    feedback_log_path: str = os.getenv("FEEDBACK_LOG_PATH", "./output/feedback_log.json")
    raw_results_dir: str = os.getenv("RAW_RESULTS_DIR", "./output/raw_results")
    companion_repo_dir: str = os.getenv("COMPANION_REPO_DIR", "./output/companion_repo")

    # Sandbox
    sandbox_timeout_sec: int = int(os.getenv("SANDBOX_TIMEOUT_SEC", "600"))
    sandbox_max_output_bytes: int = int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", "1048576"))
    sandbox_backend: str = os.getenv("SANDBOX_BACKEND", "docker")  # "ast" | "docker"
    sandbox_docker_memory: str = os.getenv("SANDBOX_DOCKER_MEMORY", "2g")
    sandbox_docker_cpus: str = os.getenv("SANDBOX_DOCKER_CPUS", "1.0")

    # --- TopicHunter v2 feature flags ---
    openalex_concept_filtering_enabled: bool = os.getenv("OPENALEX_CONCEPT_FILTERING_ENABLED", "true").lower() == "true"
    hyde_enabled: bool = os.getenv("HYDE_ENABLED", "true").lower() == "true"
    hyde_max_chars: int = int(os.getenv("HYDE_MAX_CHARS", "600"))
    multi_hop_retrieval_enabled: bool = os.getenv("MULTI_HOP_RETRIEVAL_ENABLED", "true").lower() == "true"
    multi_hop_min_papers_threshold: int = int(os.getenv("MULTI_HOP_MIN_PAPERS_THRESHOLD", "12"))
    multi_hop_max_hops: int = int(os.getenv("MULTI_HOP_MAX_HOPS", "2"))
    frontier_seeding_enabled: bool = os.getenv("FRONTIER_SEEDING_ENABLED", "true").lower() == "true"
    frontier_refresh_every_n_runs: int = int(os.getenv("FRONTIER_REFRESH_EVERY_N_RUNS", "5"))
    frontier_sample_size: int = int(os.getenv("FRONTIER_SAMPLE_SIZE", "30"))
    frontier_terms_extracted: int = int(os.getenv("FRONTIER_TERMS_EXTRACTED", "8"))
    cross_seed_paper_cache_enabled: bool = os.getenv("CROSS_SEED_PAPER_CACHE_ENABLED", "true").lower() == "true"
    capability_first_dataset_scoping_enabled: bool = os.getenv("CAPABILITY_FIRST_DATASET_SCOPING_ENABLED", "true").lower() == "true"
    structural_gap_mining_enabled: bool = os.getenv("STRUCTURAL_GAP_MINING_ENABLED", "true").lower() == "true"
    structural_gap_max_pairs: int = int(os.getenv("STRUCTURAL_GAP_MAX_PAIRS", "8"))
    sparsity_matrix_enabled: bool = os.getenv("SPARSITY_MATRIX_ENABLED", "true").lower() == "true"
    contradiction_mining_enabled: bool = os.getenv("CONTRADICTION_MINING_ENABLED", "true").lower() == "true"
    replication_target_mining_enabled: bool = os.getenv("REPLICATION_TARGET_MINING_ENABLED", "true").lower() == "true"
    negative_result_seeding_enabled: bool = os.getenv("NEGATIVE_RESULT_SEEDING_ENABLED", "true").lower() == "true"
    persona_ensemble_enabled: bool = os.getenv("PERSONA_ENSEMBLE_ENABLED", "true").lower() == "true"
    persona_count: int = int(os.getenv("PERSONA_COUNT", "2"))
    seed_strategy_elo_enabled: bool = os.getenv("SEED_STRATEGY_ELO_ENABLED", "true").lower() == "true"
    # --- TopicHunter v5: full-corpus mining windows (owner directive: quality
    # over cost — wider windows spend more LLM calls per seed deliberately) ---
    contradiction_max_papers: int = int(os.getenv("CONTRADICTION_MAX_PAPERS", "40"))
    contradiction_abstract_chars: int = int(os.getenv("CONTRADICTION_ABSTRACT_CHARS", "1500"))
    sparsity_max_papers: int = int(os.getenv("SPARSITY_MAX_PAPERS", "60"))
    sparsity_abstract_chars: int = int(os.getenv("SPARSITY_ABSTRACT_CHARS", "1200"))
    # --- TopicHunter v5: deterministic no-LLM signals (always fail-open) ---
    momentum_mining_enabled: bool = os.getenv("MOMENTUM_MINING_ENABLED", "true").lower() == "true"
    citation_concentration_enabled: bool = os.getenv("CITATION_CONCENTRATION_ENABLED", "true").lower() == "true"
    concept_gap_mining_enabled: bool = os.getenv("CONCEPT_GAP_MINING_ENABLED", "true").lower() == "true"
    reviewer_clustering_enabled: bool = os.getenv("REVIEWER_CLUSTERING_ENABLED", "true").lower() == "true"
    openalex_native_coupling_enabled: bool = os.getenv("OPENALEX_NATIVE_COUPLING_ENABLED", "true").lower() == "true"
    # --- TopicHunter v5: bounded OA full-text (top-K papers per seed) ---
    hunter_fulltext_enabled: bool = os.getenv("HUNTER_FULLTEXT_ENABLED", "true").lower() == "true"
    hunter_fulltext_top_k: int = int(os.getenv("HUNTER_FULLTEXT_TOP_K", "10"))
    hunter_fulltext_max_chars: int = int(os.getenv("HUNTER_FULLTEXT_MAX_CHARS", "6000"))
    # --- TopicHunter v5: scored selection (additive second-chance + portfolio) ---
    hunter_scoring_enabled: bool = os.getenv("HUNTER_SCORING_ENABLED", "true").lower() == "true"
    hunter_second_chance_threshold: float = float(os.getenv("HUNTER_SECOND_CHANCE_THRESHOLD", "0.55"))
    hunter_second_chance_max: int = int(os.getenv("HUNTER_SECOND_CHANCE_MAX", "2"))
    hunter_portfolio_enabled: bool = os.getenv("HUNTER_PORTFOLIO_ENABLED", "true").lower() == "true"

    # --- TopicHunter v3 ---
    llm_seed_generation_enabled: bool = os.getenv("LLM_SEED_GENERATION_ENABLED", "true").lower() == "true"

    # API gateway settings
    seed_sequential: bool = os.getenv("SEED_SEQUENTIAL", "true").lower() == "true"
    persona_sequential: bool = os.getenv("PERSONA_SEQUENTIAL", "true").lower() == "true"
    llm_budget_per_seed: int = int(os.getenv("LLM_BUDGET_PER_SEED", "30"))
    novelty_max_abstracts: int = int(os.getenv("NOVELTY_MAX_ABSTRACTS", "40"))
    # Reachability is a product requirement: a bounded, explicitly labelled
    # candidate is preferable to discarding an otherwise useful investigation
    # when an LLM gate or provider is unavailable.
    topic_fallback_enabled: bool = os.getenv("TOPIC_FALLBACK_ENABLED", "true").lower() == "true"
    topic_fallback_max_candidates: int = int(os.getenv("TOPIC_FALLBACK_MAX_CANDIDATES", "2"))
    # Budgets sized for slow reasoning endpoints (observed ~300 s per LLM
    # call on Minimax-M2.7-class models: 1 HyDE + 2-3 persona generations +
    # gate-chain screeners per seed). A seed needs ~6-10 LLM calls, so the
    # seed budget must cover ~20 min of blocking calls — advisory deadline
    # checks cannot interrupt in-flight requests, only skip starting new
    # work. Discovery auto-scales to fit n_seeds (see discover_topics).
    topic_seed_timeout_sec: int = int(os.getenv("TOPIC_SEED_TIMEOUT_SEC", "5400"))
    topic_discovery_timeout_sec: int = int(os.getenv("TOPIC_DISCOVERY_TIMEOUT_SEC", "21600"))
    # Gap-level deadline inside a seed: stop opening NEW gaps when less than
    # this remains in the seed slice. A partial seed with 1 topic beats a
    # killed seed with 0 (run b099dcaa). Advisory only — never interrupts
    # in-flight LLM calls.
    topic_gap_deadline_sec: int = int(os.getenv("TOPIC_GAP_DEADLINE_SEC", "600"))
    # HTTP-level timeout for a single LLM request (seconds). Must exceed the
    # endpoint's p99 latency or slow-but-healthy streams get cut off.
    llm_timeout_sec: float = float(os.getenv("LLM_TIMEOUT_SEC", "1800"))
    # Max output tokens ceiling for truncation-escalation (8192 -> ... -> ceiling).
    # Raised 32768 -> 65536: persona prompts (full evidence + cross-run context)
    # burn budget on reasoning and return finish_reason=length with no content.
    llm_max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "65536"))
    # First-attempt budget for persona gap generation (escalates to ceiling above).
    llm_persona_max_tokens: int = int(os.getenv("LLM_PERSONA_MAX_TOKENS", "16384"))

    # --- Prompt v2 budgets (agentic-harness upgrade) ---
    # Per-slot char budgets for registry rendering (system/user split).
    prompt_system_max_chars: int = int(os.getenv("PROMPT_SYSTEM_MAX_CHARS", "2400"))
    prompt_evidence_max_chars: int = int(os.getenv("PROMPT_EVIDENCE_MAX_CHARS", "10000"))
    prompt_prior_max_chars: int = int(os.getenv("PROMPT_PRIOR_MAX_CHARS", "4800"))
    prompt_lessons_max_chars: int = int(os.getenv("PROMPT_LESSONS_MAX_CHARS", "2400"))
    # Cheap-drafter -> strong-verifier cascade (FrugalGPT-style).
    llm_cascade_enabled: bool = os.getenv("LLM_CASCADE_ENABLED", "true").lower() == "true"
    engineer_cascade_attempt: int = int(os.getenv("ENGINEER_CASCADE_ATTEMPT", "3"))
    # Agentic iteration caps per phase (opencode `steps` equivalent).
    agent_max_steps_hunter: int = int(os.getenv("AGENT_MAX_STEPS_HUNTER", "5"))
    agent_max_steps_planner: int = int(os.getenv("AGENT_MAX_STEPS_PLANNER", "4"))
    agent_max_steps_engineer: int = int(os.getenv("AGENT_MAX_STEPS_ENGINEER", "6"))
    agent_max_steps_writer: int = int(os.getenv("AGENT_MAX_STEPS_WRITER", "2"))

    # Web UI
    web_host: str = os.getenv("WEB_HOST", "127.0.0.1")
    web_port: int = int(os.getenv("WEB_PORT", "8765"))
    keys_store_path: str = os.getenv("KEYS_STORE_PATH", "./memory/keys.json")

    def resolve_fallback_model(self, current_model: str = "") -> str:
        """e0ec9e91: LLM_FALLBACK_PROVIDER/MODEL empty meant auto-detect with
        no pinned route, so every logical call fanned out blindly during the
        novita outage. Explicit model wins; else cross-provider default
        (gemini <-> openai_model) so strong/judge have somewhere to go."""
        if (self.llm_model_fallback or "").strip():
            _fb = self.llm_model_fallback.strip()
            if _fb != (current_model or ""):
                return _fb
        if (self.llm_fallback_model or "").strip():
            _fb2 = self.llm_fallback_model.strip()
            if _fb2 != (current_model or ""):
                return _fb2
        provider = (self.llm_provider or "gemini").lower()
        if provider in ("openai", "openai_compatible"):
            if (self.gemini_model or "") and self.gemini_model != current_model and self.google_api_key:
                return self.gemini_model
            return ""
        if (self.openai_model or "") and self.openai_model != current_model and self.openai_api_key:
            return self.openai_model
        return ""

    def get_ensemble_models(self) -> list[str]:
        if not self.ensemble_judge_models.strip():
            return []
        return [m.strip() for m in self.ensemble_judge_models.split(",") if m.strip()]

    def resolve_model(self, tier: str = "default") -> str:
        """Resolve model id for a cost tier: cheap | strong | judge | default."""
        provider = (self.llm_provider or "gemini").lower()
        if tier == "cheap" and self.llm_model_cheap:
            return self.llm_model_cheap
        if tier == "strong" and self.llm_model_strong:
            return self.llm_model_strong
        if tier == "judge" and self.llm_model_judge:
            return self.llm_model_judge
        if provider in ("openai", "openai_compatible"):
            return self.openai_model
        return self.gemini_model

    def is_gemma_4_model(self, model: str = "") -> bool:
        """Whether a model id opts into the Gemma 4 generation profile."""
        model_id = (model or self.gemini_model or "").strip().lower()
        return model_id.removeprefix("models/").startswith("gemma-4-")

    class Config:
        env_file = ".env"
        extra = "ignore"


# Global configuration instance
config = Config()

def apply_runtime_keys(keys: dict) -> None:
    """Apply keys from the web UI store into the live config + process env."""
    mapping = {
        "GOOGLE_API_KEY": "google_api_key",
        "OPENAI_API_KEY": "openai_api_key",
        "OPENAI_BASE_URL": "openai_base_url",
        "LLM_PROVIDER": "llm_provider",
        "GEMINI_MODEL": "gemini_model",
        "GEMMA_THINKING_LEVEL": "gemma_thinking_level",
        "OPENAI_MODEL": "openai_model",
        "GEMINI_EMBEDDING_MODEL": "gemini_embedding_model",
        "OPENAI_EMBEDDING_MODEL": "openai_embedding_model",
        "EMBEDDING_DIMENSION": "embedding_dimension",
        "LLM_MODEL_CHEAP": "llm_model_cheap",
        "LLM_MODEL_STRONG": "llm_model_strong",
        "LLM_MODEL_JUDGE": "llm_model_judge",
        "LLM_MODEL_FALLBACK": "llm_model_fallback",
        "LLM_FALLBACK_PROVIDER": "llm_fallback_provider",
        "LLM_FALLBACK_MODEL": "llm_fallback_model",
        "LLM_TIMEOUT_SEC": "llm_timeout_sec",
        "ENSEMBLE_JUDGE_MODELS": "ensemble_judge_models",
        "SEMANTIC_SCHOLAR_API_KEY": "semantic_scholar_api_key",
        "OPENALEX_EMAIL": "openalex_email",
        "RESEARCH_DOMAIN": "research_domain",
        "SUPERVISOR_THRESHOLD": "supervisor_threshold",
        "DEBATE_PASS_THRESHOLD": "debate_pass_threshold",
        "DEBATE_MIN_ROUNDS": "debate_min_rounds",
        "DEBATE_MAX_ROUNDS": "debate_max_rounds",
        "NOVELTY_SIMILARITY_REJECT": "novelty_similarity_reject",
        "EXPERIMENT_SEEDS": "experiment_seeds",
        "EXPERIMENT_BRANCH_COUNT": "experiment_branch_count",
        "ENGINEER_MAX_ATTEMPTS": "engineer_max_attempts",
        "ENGINEERING_WALL_CLOCK_SEC": "engineering_wall_clock_sec",
        "ENGINEERING_BACKEND": "arena_backend",
        "ARENA_DIR": "arena_dir",
        "ARENA_SESSION_TIMEOUT_SEC": "arena_session_timeout_sec",
        "ARENA_MAX_ROUNDS": "arena_max_rounds",
        "ARENA_MODEL": "arena_model",
        "ARENA_AGENT": "arena_agent",
        "ARENA_RETENTION_DAYS": "arena_retention_days",
        "ARENA_CLEANUP_STALE": "arena_cleanup_stale",
        "ENGINEER_GATE_WAIT_SEC": "engineer_gate_wait_sec",
        "TYPED_GRAPH_ENABLED": "typed_graph_enabled",
        "PAPER_TYPE_DEFAULT": "paper_type_default",
        "PARALLEL_PHASES_ENABLED": "parallel_phases_enabled",
        "RUN_MAX_LLM_CALLS": "run_max_llm_calls",
        "RUN_MAX_TOKENS": "run_max_tokens",
        "RUN_MAX_USD": "run_max_usd",
        "RUN_WALL_CLOCK_SEC": "run_wall_clock_sec",
        "UNPAYWALL_ENABLED": "unpaywall_enabled",
        "PMC_ENABLED": "pmc_enabled",
        "DOAJ_GUARD_ENABLED": "doaj_guard_enabled",
        "ALLOW_SYNTHETIC_RESPONDENTS": "allow_synthetic_respondents",
        "LLM_PRICE_TABLE_JSON": "llm_price_table_json",
        "MAX_ITERATIONS": "max_iterations",
        "DEBUG_MODE": "debug_mode",
        "LOG_LEVEL": "log_level",
        "VECTOR_DB_PATH": "vector_db_path",
        "MEMORY_SIZE": "memory_size",
        "CROSS_RUN_MEMORY_PATH": "cross_run_memory_path",
        "RUN_LOG_PATH": "run_log_path",
        "RUN_EVENTS_PATH": "run_events_path",
        "ELO_RATINGS_PATH": "elo_ratings_path",
        "ELO_PRIOR_RATING": "elo_prior_rating",
        "ELO_SHRINKAGE_K": "elo_shrinkage_k",
        "ELO_MIN_OBSERVATIONS": "elo_min_observations",
        "TOPIC_EXPLORATION_EVERY": "topic_exploration_every",
        "TOPIC_EXPLORATION_SEED": "topic_exploration_seed",
        "CHECKPOINT_PATH": "checkpoint_path",
        "RESEARCH_DB_PATH": "research_db_path",
        "OUTPUT_DIR": "output_dir",
        "DRAFT_VERSIONS_DIR": "draft_versions_dir",
        "DEBATE_LOG_PATH": "debate_log_path",
        "FEEDBACK_LOG_PATH": "feedback_log_path",
        "RAW_RESULTS_DIR": "raw_results_dir",
        "COMPANION_REPO_DIR": "companion_repo_dir",
        "SANDBOX_TIMEOUT_SEC": "sandbox_timeout_sec",
        "SANDBOX_MAX_OUTPUT_BYTES": "sandbox_max_output_bytes",
        "SANDBOX_BACKEND": "sandbox_backend",
        "SANDBOX_DOCKER_MEMORY": "sandbox_docker_memory",
        "SANDBOX_DOCKER_CPUS": "sandbox_docker_cpus",
        "WEB_HOST": "web_host",
        "WEB_PORT": "web_port",
        "KEYS_STORE_PATH": "keys_store_path",
        "OPENALEX_CONCEPT_FILTERING_ENABLED": "openalex_concept_filtering_enabled",
        "HYDE_ENABLED": "hyde_enabled",
        "HYDE_MAX_CHARS": "hyde_max_chars",
        "MULTI_HOP_RETRIEVAL_ENABLED": "multi_hop_retrieval_enabled",
        "MULTI_HOP_MIN_PAPERS_THRESHOLD": "multi_hop_min_papers_threshold",
        "MULTI_HOP_MAX_HOPS": "multi_hop_max_hops",
        "FRONTIER_SEEDING_ENABLED": "frontier_seeding_enabled",
        "FRONTIER_REFRESH_EVERY_N_RUNS": "frontier_refresh_every_n_runs",
        "FRONTIER_SAMPLE_SIZE": "frontier_sample_size",
        "FRONTIER_TERMS_EXTRACTED": "frontier_terms_extracted",
        "CROSS_SEED_PAPER_CACHE_ENABLED": "cross_seed_paper_cache_enabled",
        "CAPABILITY_FIRST_DATASET_SCOPING_ENABLED": "capability_first_dataset_scoping_enabled",
        "STRUCTURAL_GAP_MINING_ENABLED": "structural_gap_mining_enabled",
        "STRUCTURAL_GAP_MAX_PAIRS": "structural_gap_max_pairs",
        "SPARSITY_MATRIX_ENABLED": "sparsity_matrix_enabled",
        "CONTRADICTION_MINING_ENABLED": "contradiction_mining_enabled",
        "REPLICATION_TARGET_MINING_ENABLED": "replication_target_mining_enabled",
        "NEGATIVE_RESULT_SEEDING_ENABLED": "negative_result_seeding_enabled",
        "PERSONA_ENSEMBLE_ENABLED": "persona_ensemble_enabled",
        "PERSONA_COUNT": "persona_count",
        "SEED_SEQUENTIAL": "seed_sequential",
        "PERSONA_SEQUENTIAL": "persona_sequential",
        "LLM_BUDGET_PER_SEED": "llm_budget_per_seed",
        "NOVELTY_MAX_ABSTRACTS": "novelty_max_abstracts",
        "CONTRADICTION_MAX_PAPERS": "contradiction_max_papers",
        "CONTRADICTION_ABSTRACT_CHARS": "contradiction_abstract_chars",
        "SPARSITY_MAX_PAPERS": "sparsity_max_papers",
        "SPARSITY_ABSTRACT_CHARS": "sparsity_abstract_chars",
        "MOMENTUM_MINING_ENABLED": "momentum_mining_enabled",
        "CITATION_CONCENTRATION_ENABLED": "citation_concentration_enabled",
        "CONCEPT_GAP_MINING_ENABLED": "concept_gap_mining_enabled",
        "REVIEWER_CLUSTERING_ENABLED": "reviewer_clustering_enabled",
        "OPENALEX_NATIVE_COUPLING_ENABLED": "openalex_native_coupling_enabled",
        "HUNTER_FULLTEXT_ENABLED": "hunter_fulltext_enabled",
        "HUNTER_FULLTEXT_TOP_K": "hunter_fulltext_top_k",
        "HUNTER_FULLTEXT_MAX_CHARS": "hunter_fulltext_max_chars",
        "HUNTER_SCORING_ENABLED": "hunter_scoring_enabled",
        "HUNTER_SECOND_CHANCE_THRESHOLD": "hunter_second_chance_threshold",
        "HUNTER_SECOND_CHANCE_MAX": "hunter_second_chance_max",
        "HUNTER_PORTFOLIO_ENABLED": "hunter_portfolio_enabled",
        "SEED_STRATEGY_ELO_ENABLED": "seed_strategy_elo_enabled",
        "TOPIC_SEED_TIMEOUT_SEC": "topic_seed_timeout_sec",
        "TOPIC_DISCOVERY_TIMEOUT_SEC": "topic_discovery_timeout_sec",
        "TOPIC_GAP_DEADLINE_SEC": "topic_gap_deadline_sec",
        "TOPIC_FALLBACK_ENABLED": "topic_fallback_enabled",
        "TOPIC_FALLBACK_MAX_CANDIDATES": "topic_fallback_max_candidates",
        "LLM_SEED_GENERATION_ENABLED": "llm_seed_generation_enabled",
        "OPENREVIEW_USERNAME": "openreview_username",
        "OPENREVIEW_PASSWORD": "openreview_password",
        "OPENREVIEW_ENABLED": "openreview_enabled",
        "ARXIV_ENABLED": "arxiv_enabled",
        "OPENALEX_ENABLED": "openalex_enabled",
        "SEMANTIC_SCHOLAR_ENABLED": "semantic_scholar_enabled",
        "HUGGINGFACE_TOKEN": "huggingface_token",
        "HUGGINGFACE_MAX_DOWNLOAD_MB": "huggingface_max_download_mb",
        "LLM_MAX_OUTPUT_TOKENS": "llm_max_output_tokens",
        "LLM_PERSONA_MAX_TOKENS": "llm_persona_max_tokens",
        "PROMPT_SYSTEM_MAX_CHARS": "prompt_system_max_chars",
        "PROMPT_EVIDENCE_MAX_CHARS": "prompt_evidence_max_chars",
        "PROMPT_PRIOR_MAX_CHARS": "prompt_prior_max_chars",
        "PROMPT_LESSONS_MAX_CHARS": "prompt_lessons_max_chars",
        "LLM_CASCADE_ENABLED": "llm_cascade_enabled",
        "ENGINEER_CASCADE_ATTEMPT": "engineer_cascade_attempt",
        "AGENT_MAX_STEPS_HUNTER": "agent_max_steps_hunter",
        "AGENT_MAX_STEPS_PLANNER": "agent_max_steps_planner",
        "AGENT_MAX_STEPS_ENGINEER": "agent_max_steps_engineer",
        "AGENT_MAX_STEPS_WRITER": "agent_max_steps_writer",
    }
    for env_key, attr in mapping.items():
        if env_key in keys and keys[env_key] not in (None, ""):
            value = keys[env_key]
            os.environ[env_key] = str(value)
            if hasattr(config, attr):
                current = getattr(config, attr)
                if isinstance(current, float):
                    setattr(config, attr, float(value))
                elif isinstance(current, int) and not isinstance(current, bool):
                    setattr(config, attr, int(value))
                elif isinstance(current, bool):
                    setattr(config, attr, str(value).lower() in {"1", "true", "yes", "on"})
                else:
                    setattr(config, attr, value)


def sync_env_file(keys: dict, env_path: Optional[Path] = None) -> Path:
    """Upsert UI-saved settings into project-root .env (preserves other lines/comments)."""
    path = Path(env_path) if env_path else (_PROJECT_ROOT / ".env")
    known = {
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LLM_PROVIDER",
        "GEMINI_MODEL",
        "GEMMA_THINKING_LEVEL",
        "OPENAI_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "EMBEDDING_DIMENSION",
        "LLM_MODEL_CHEAP",
        "LLM_MODEL_STRONG",
        "LLM_MODEL_JUDGE",
        "LLM_MODEL_FALLBACK",
        "LLM_FALLBACK_PROVIDER",
        "LLM_FALLBACK_MODEL",
        "LLM_TIMEOUT_SEC",
        "ENSEMBLE_JUDGE_MODELS",
        "SEMANTIC_SCHOLAR_API_KEY",
        "OPENALEX_EMAIL",
        "RESEARCH_DOMAIN",
        "SUPERVISOR_THRESHOLD",
        "DEBATE_PASS_THRESHOLD",
        "DEBATE_MIN_ROUNDS",
        "DEBATE_MAX_ROUNDS",
        "NOVELTY_SIMILARITY_REJECT",
        "EXPERIMENT_SEEDS",
        "EXPERIMENT_BRANCH_COUNT",
        "ENGINEER_MAX_ATTEMPTS",
        "ENGINEERING_WALL_CLOCK_SEC",
        "ENGINEERING_BACKEND",
        "ARENA_DIR",
        "ARENA_SESSION_TIMEOUT_SEC",
        "ARENA_MAX_ROUNDS",
        "ARENA_MODEL",
        "ARENA_AGENT",
        "ARENA_RETENTION_DAYS",
        "ARENA_CLEANUP_STALE",
        "ENGINEER_GATE_WAIT_SEC",
        "TYPED_GRAPH_ENABLED",
        "PAPER_TYPE_DEFAULT",
        "PARALLEL_PHASES_ENABLED",
        "RUN_MAX_LLM_CALLS",
        "RUN_MAX_TOKENS",
        "RUN_MAX_USD",
        "RUN_WALL_CLOCK_SEC",
        "UNPAYWALL_ENABLED",
        "PMC_ENABLED",
        "DOAJ_GUARD_ENABLED",
        "ALLOW_SYNTHETIC_RESPONDENTS",
        "LLM_PRICE_TABLE_JSON",
        "MAX_ITERATIONS",
        "DEBUG_MODE",
        "LOG_LEVEL",
        "VECTOR_DB_PATH",
        "MEMORY_SIZE",
        "CROSS_RUN_MEMORY_PATH",
        "RUN_LOG_PATH",
        "RUN_EVENTS_PATH",
        "ELO_RATINGS_PATH",
        "ELO_PRIOR_RATING",
        "ELO_SHRINKAGE_K",
        "ELO_MIN_OBSERVATIONS",
        "TOPIC_EXPLORATION_EVERY",
        "TOPIC_EXPLORATION_SEED",
        "CHECKPOINT_PATH",
        "RESEARCH_DB_PATH",
        "OUTPUT_DIR",
        "DRAFT_VERSIONS_DIR",
        "DEBATE_LOG_PATH",
        "FEEDBACK_LOG_PATH",
        "RAW_RESULTS_DIR",
        "COMPANION_REPO_DIR",
        "SANDBOX_TIMEOUT_SEC",
        "SANDBOX_MAX_OUTPUT_BYTES",
        "SANDBOX_BACKEND",
        "SANDBOX_DOCKER_MEMORY",
        "SANDBOX_DOCKER_CPUS",
        "WEB_HOST",
        "WEB_PORT",
        "KEYS_STORE_PATH",
        "OPENALEX_CONCEPT_FILTERING_ENABLED",
        "HYDE_ENABLED",
        "HYDE_MAX_CHARS",
        "MULTI_HOP_RETRIEVAL_ENABLED",
        "MULTI_HOP_MIN_PAPERS_THRESHOLD",
        "MULTI_HOP_MAX_HOPS",
        "FRONTIER_SEEDING_ENABLED",
        "FRONTIER_REFRESH_EVERY_N_RUNS",
        "FRONTIER_SAMPLE_SIZE",
        "FRONTIER_TERMS_EXTRACTED",
        "CROSS_SEED_PAPER_CACHE_ENABLED",
        "CAPABILITY_FIRST_DATASET_SCOPING_ENABLED",
        "STRUCTURAL_GAP_MINING_ENABLED",
        "STRUCTURAL_GAP_MAX_PAIRS",
        "SPARSITY_MATRIX_ENABLED",
        "CONTRADICTION_MINING_ENABLED",
        "REPLICATION_TARGET_MINING_ENABLED",
        "NEGATIVE_RESULT_SEEDING_ENABLED",
        "PERSONA_ENSEMBLE_ENABLED",
        "PERSONA_COUNT",
        "SEED_SEQUENTIAL",
        "PERSONA_SEQUENTIAL",
        "LLM_BUDGET_PER_SEED",
        "NOVELTY_MAX_ABSTRACTS",
        "CONTRADICTION_MAX_PAPERS",
        "CONTRADICTION_ABSTRACT_CHARS",
        "SPARSITY_MAX_PAPERS",
        "SPARSITY_ABSTRACT_CHARS",
        "MOMENTUM_MINING_ENABLED",
        "CITATION_CONCENTRATION_ENABLED",
        "CONCEPT_GAP_MINING_ENABLED",
        "REVIEWER_CLUSTERING_ENABLED",
        "OPENALEX_NATIVE_COUPLING_ENABLED",
        "HUNTER_FULLTEXT_ENABLED",
        "HUNTER_FULLTEXT_TOP_K",
        "HUNTER_FULLTEXT_MAX_CHARS",
        "HUNTER_SCORING_ENABLED",
        "HUNTER_SECOND_CHANCE_THRESHOLD",
        "HUNTER_SECOND_CHANCE_MAX",
        "HUNTER_PORTFOLIO_ENABLED",
        "SEED_STRATEGY_ELO_ENABLED",
        "TOPIC_SEED_TIMEOUT_SEC",
        "TOPIC_DISCOVERY_TIMEOUT_SEC",
        "TOPIC_GAP_DEADLINE_SEC",
        "TOPIC_FALLBACK_ENABLED",
        "TOPIC_FALLBACK_MAX_CANDIDATES",
        "LLM_SEED_GENERATION_ENABLED",
        "OPENREVIEW_USERNAME",
        "OPENREVIEW_PASSWORD",
        "OPENREVIEW_ENABLED",
        "ARXIV_ENABLED",
        "OPENALEX_ENABLED",
        "SEMANTIC_SCHOLAR_ENABLED",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_MAX_DOWNLOAD_MB",
        "LLM_MAX_OUTPUT_TOKENS",
        "LLM_PERSONA_MAX_TOKENS",
        "PROMPT_SYSTEM_MAX_CHARS",
        "PROMPT_EVIDENCE_MAX_CHARS",
        "PROMPT_PRIOR_MAX_CHARS",
        "PROMPT_LESSONS_MAX_CHARS",
        "LLM_CASCADE_ENABLED",
        "ENGINEER_CASCADE_ATTEMPT",
        "AGENT_MAX_STEPS_HUNTER",
        "AGENT_MAX_STEPS_PLANNER",
        "AGENT_MAX_STEPS_ENGINEER",
        "AGENT_MAX_STEPS_WRITER",
    }
    updates = {
        str(k): str(v)
        for k, v in keys.items()
        if k in known and v not in (None, "")
    }
    if not updates:
        return path

    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)

    missing = [k for k in updates if k not in seen]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append("# Synced from Control Deck UI")
        for key in missing:
            out.append(f"{key}={updates[key]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out)
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _resolve_keys_path() -> Path:
    """Resolve keys.json relative to project root (not process cwd)."""
    raw = Path(config.keys_store_path)
    if raw.is_absolute():
        return raw
    return (_PROJECT_ROOT / raw).resolve()


def _auto_load_keys():
    try:
        import json
        p = _resolve_keys_path()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                apply_runtime_keys(data)
    except Exception as e:
        logger.warning("Failed to auto-load keys from %s: %s", _resolve_keys_path(), e)


_auto_load_keys()


def validate_config():
    """Validate that required settings are present for the selected provider."""
    provider = (config.llm_provider or "gemini").lower()

    if provider == "gemini":
        if not config.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required when LLM_PROVIDER=gemini. "
                "Get it from https://aistudio.google.com/apikey"
            )
        if not config.gemini_embedding_model:
            raise ValueError(
                "GEMINI_EMBEDDING_MODEL is required when LLM_PROVIDER=gemini. "
                "Default is gemini-embedding-2."
            )
    elif provider in ("openai", "openai_compatible"):
        if not config.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is openai or "
                "openai_compatible (also used for OpenRouter/Groq/local endpoints). "
                "For local servers that ignore auth, set any non-empty value e.g. local."
            )
        if provider == "openai_compatible" and not (config.openai_base_url or "").strip():
            raise ValueError(
                "OPENAI_BASE_URL is required for openai_compatible "
                "(e.g. http://127.0.0.1:11434/v1 for Ollama)."
            )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider}. Use gemini, openai, or openai_compatible."
        )

    if not config.openalex_email:
        raise ValueError(
            "OPENALEX_EMAIL is required for rate limiting. Set it to your email."
        )

    # Semantic Scholar key visibility: the free-tier Graph API throttles
    # anonymous traffic far harder than authenticated traffic (run
    # 21a145ad: 429 storms at ~1 call/min even WITH the key; without it
    # expect worse). Surface the mode at startup so silent anonymous mode
    # is never mistaken for "key in use".
    if not getattr(config, "semantic_scholar_api_key", ""):
        logger.warning(
            "SEMANTIC_SCHOLAR_API_KEY not set — Semantic Scholar calls run "
            "ANONYMOUS (shared unauthenticated quota; much harsher 429s). "
            "Set SEMANTIC_SCHOLAR_API_KEY for the 1 RPS authenticated tier."
        )
    else:
        logger.info(
            "Semantic Scholar API key loaded — Graph API calls authenticated "
            "(1 RPS tier; free-tier shared quotas may still 429 under load)."
        )

    # M0 strict validate_config (§6.2 + §9): placeholder reject, model sanity,
    # fallback-key check, docker probe, writability, disk >= 500MB.
    email = (config.openalex_email or "").strip().lower()
    placeholders = {"", "you@example.com", "researcher@example.com", "test@example.com", "example@example.com"}
    if email in placeholders or "@" not in email or email.startswith("example@"):
        raise ValueError(
            f"OPENALEX_EMAIL placeholder rejected ({config.openalex_email!r}). Set a real email for the OpenAlex polite pool."
        )
    for model_attr in ("gemini_model", "openai_model"):
        model_val = str(getattr(config, model_attr, "") or "").strip()
        if not model_val or any(ch.isspace() for ch in model_val):
            raise ValueError(f"{model_attr.upper()} model sanity failed ({model_val!r}). Set a valid model id.")
    thinking_level = str(getattr(config, "gemma_thinking_level", "high") or "").strip().lower()
    if thinking_level not in {"minimal", "high"}:
        raise ValueError(
            "GEMMA_THINKING_LEVEL must be 'minimal' or 'high' "
            "(Gemma 4 supports an on/off thinking setting)."
        )
    # P0-3: retired-model denylist (run b099dcaa: gemini-2.0-flash 404'd mid-run
    # after fallback auto-detect used a stale hardcoded default). Fail fast at
    # startup instead of discovering retirement 2h into a run.
    _retired = {"gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gpt-3.5-turbo-0301"}
    for model_attr in ("gemini_model", "openai_model", "llm_model_cheap",
                       "llm_model_strong", "llm_model_judge", "llm_model_fallback",
                       "llm_fallback_model"):
        try:
            _v = str(getattr(config, model_attr, "") or "").strip()
        except Exception:
            continue
        if _v and _v.split("/")[-1] in _retired:
            raise ValueError(
                f"{model_attr.upper()}={_v!r} is retired (404 at runtime). "
                "Update to a current model id (e.g. gemini-2.5-flash)."
            )
    # Warn (not fail) when both provider keys exist but no explicit fallback is
    # pinned — auto-detect works but production should pin the chain in .env.
    try:
        if config.google_api_key and config.openai_api_key:
            if not str(getattr(config, "llm_fallback_provider", "") or "").strip():
                logger.warning(
                    "Both GOOGLE_API_KEY and OPENAI_API_KEY are set but "
                    "LLM_FALLBACK_PROVIDER is empty: auto-detect fallback will be used. "
                    "Pin LLM_FALLBACK_PROVIDER/MODEL explicitly in production."
                )
    except Exception:
        pass
    if config.embedding_dimension != 3072:
        raise ValueError(
            f"EMBEDDING_DIMENSION={config.embedding_dimension} rejected: locked to 3072 "
            "(gemini-embedding-2 native output; FAISS index + memory/vector_db are 3072-wide)."
        )
    # Fallback-key check: declared fallback provider must have its key.
    fallback_provider = str(getattr(config, "llm_fallback_provider", "") or "").strip().lower()
    if fallback_provider in ("gemini",):
        if not config.google_api_key:
            raise ValueError("LLM_FALLBACK_PROVIDER=gemini but GOOGLE_API_KEY is missing.")
    if fallback_provider in ("openai", "openai_compatible"):
        if not config.openai_api_key:
            raise ValueError(f"LLM_FALLBACK_PROVIDER={fallback_provider} but OPENAI_API_KEY is missing.")
    # Docker probe (warn only — dispatch falls back to AST automatically).
    if str(getattr(config, "sandbox_backend", "docker")).lower() == "docker":
        try:
            import socket as _sock

            _sock.create_connection(("localhost", 2375), timeout=1).close()
        except Exception:
            try:
                import subprocess as _sp

                _sp.run(["docker", "info"], capture_output=True, timeout=5)
            except Exception as e:
                logger.warning("Docker backend selected but daemon probe failed (%s); AST fallback will be used.", str(e)[:160])
    # Checkpoint/ledger writability + disk >= 500MB.
    import shutil as _shutil

    for label, raw_path in (("CHECKPOINT_PATH", config.checkpoint_path), ("RESEARCH_DB_PATH", config.research_db_path)):
        parent = Path(raw_path).parent if str(raw_path) else Path("./memory")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"{label} parent not writable ({parent}): {e}") from e
        probe = parent / ".scholargraph_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except Exception as e:
            raise ValueError(f"{label} not writable ({parent}): {e}") from e
        try:
            free = _shutil.disk_usage(str(parent)).free
            if free < 500 * 1024 * 1024:
                raise ValueError(f"Disk free <500MB at {parent} ({free // (1024*1024)}MB). Free space before running.")
        except ValueError:
            raise
        except Exception as e:
            logger.warning("Disk check skipped at %s: %s", parent, e)

    for path in (
        config.output_dir,
        config.draft_versions_dir,
        config.raw_results_dir,
        config.companion_repo_dir,
        os.path.dirname(config.vector_db_path) or "./memory",
        "./memory",
    ):
        os.makedirs(path, exist_ok=True)

    return True
 
