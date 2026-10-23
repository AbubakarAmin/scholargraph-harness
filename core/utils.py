"""
Shared utilities and common operations.
Use core.llm for LLM provider access and compatibility aliases.
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

from .config import config
from .llm import generate_embedding

logger = logging.getLogger(__name__)


def save_json(data: Dict[str, Any], filepath: str):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")


def load_json(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        return {}


def extract_citations(text: str) -> List[str]:
    patterns = [
        r"\[([^\]]+)\]",
        r"\(([^)]+)\)",
        r"Author et al\.\s+\d{4}",
    ]
    citations = []
    for pattern in patterns:
        citations.extend(re.findall(pattern, text))
    return list(set(citations))


def validate_math_expression(expression: str) -> bool:
    try:
        import sympy as sp

        sp.sympify(expression)
        return True
    except Exception:
        return False


def verify_math_derivation(steps: List[str], start: str, end: str) -> Dict[str, Any]:
    """SymPy check that symbolic simplification of start reaches end (when parseable)."""
    try:
        import sympy as sp

        start_e = sp.simplify(sp.sympify(start))
        end_e = sp.simplify(sp.sympify(end))
        equal = sp.simplify(start_e - end_e) == 0
        return {"passed": bool(equal), "start": str(start_e), "end": str(end_e)}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    return sanitized[:100]


def create_timestamped_filename(prefix: str, extension: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def format_section_text(section_name: str, content: str, level: int = 1) -> str:
    if level == 1:
        return f"\\section{{{section_name}}}\n\n{content}\n"
    if level == 2:
        return f"\\subsection{{{section_name}}}\n\n{content}\n"
    if level == 3:
        return f"\\subsubsection{{{section_name}}}\n\n{content}\n"
    return f"\\paragraph{{{section_name}}}\n\n{content}\n"


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    words = [w for w in words if w not in stop_words]
    return [w for w, _ in Counter(words).most_common(max_keywords)]


def calculate_similarity(text1: str, text2: str) -> float:
    try:
        embedding1 = generate_embedding(text1)
        embedding2 = generate_embedding(text2)
        denom = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        if denom == 0:
            # Embedding backend down (zeros) — TF-IDF fallback keeps the
            # novelty signal monotone instead of collapsing to 0.0.
            return tfidf_cosine(text1, text2)
        return float(np.dot(embedding1, embedding2) / denom)
    except Exception as e:
        logger.error(f"Failed to calculate similarity: {e}")
        try:
            return tfidf_cosine(text1, text2)
        except Exception:
            return 0.0


_TFIDF_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those",
    "from", "as", "it", "its", "our", "their", "his", "her", "which",
    "not", "no", "without", "using", "used", "use", "via", "based",
})


def _tf_vector(text: str) -> Counter:
    words = re.findall(r"[a-z][a-z0-9_-]{2,}", (text or "").lower())
    return Counter(w for w in words if w not in _TFIDF_STOPWORDS)


def tfidf_cosine(text1: str, text2: str) -> float:
    """Deterministic lexical similarity — offline fallback when embeddings die.

    Sublinear TF (1 + log tf) cosine over stopword-filtered tokens. No IDF
    term (only two documents); pure-python + Counter, no network, no model.
    Used by the novelty gate when is_embedding_available() is False so a dead
    embedding backend degrades to lexical ranking + forced LLM comparison
    instead of auto-NOVEL on zero vectors.
    """
    tf1, tf2 = _tf_vector(text1), _tf_vector(text2)
    if not tf1 or not tf2:
        return 0.0
    vocab = set(tf1) | set(tf2)
    dot = sum((1.0 + float(np.log1p(tf1[t]))) * (1.0 + float(np.log1p(tf2[t])))
              for t in vocab if t in tf1 and t in tf2)
    n1 = sum((1.0 + float(np.log1p(c))) ** 2 for c in tf1.values()) ** 0.5
    n2 = sum((1.0 + float(np.log1p(c))) ** 2 for c in tf2.values()) ** 0.5
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(max(0.0, min(1.0, dot / (n1 * n2))))


def _is_truncated_json(text: str) -> bool:
    """True when text ends with an unclosed JSON string or bracket stack."""
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return False
    stack: list = []
    in_string = False
    escaped = False
    for ch in text[start:]:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return False
            stack.pop()
    return bool(stack) or in_string


def _repair_truncated_json(text: str) -> Optional[Any]:
    """Best-effort recovery of a JSON object cut off mid-stream.

    When a reasoning model hits ``max_tokens`` the response ends mid-JSON —
    usually after the leading (most important) fields have been emitted.
    Closes an open string, then iteratively drops any dangling trailing
    entry (incomplete key or value) and closes the open bracket stack so
    ``json.loads`` can accept the prefix. Used only when normal parsing
    fails or the structure is unbalanced; a genuinely malformed body still
    yields None.
    """
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return None

    body = text[start:]
    # A truncated stream may also end with a complete-but-orphan inner block
    # while its parent was cut; iterative trailing-entry trim recovers the
    # leading fields (which is what downstream validation needs).
    for _ in range(8):
        # Re-scan per iteration: trimming can land inside or outside a
        # string, so the string-close quote must not be baked into a fixed
        # suffix.
        stack: list = []
        in_string = False
        escaped = False
        for ch in body:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    return None  # unbalanced beyond repair
                stack.pop()
        closing = ('"' if in_string else "") + "".join(
            "}" if opener == "{" else "]" for opener in reversed(stack)
        )
        try:
            return json.loads(body + closing)
        except Exception:
            pass
        cut = max(body.rfind(","), body.rfind(":"))
        if cut <= 0:
            return None
        body = body[:cut]
    return None


def parse_json_from_llm(response: str) -> Optional[Any]:
    """Extract JSON object/array from an LLM response.

    Returns dict OR list — callers must type-check the result before calling
    dict-only methods like ``.get`` (a bare JSON array is a valid model output).
    Strips `` blocks before extraction to prevent thinking-trace contamination.
    Prefers the LAST complete JSON block (reasoning models emit thinking first).
    Falls back to truncated-JSON repair when the response was cut off
    mid-object (finish_reason=length) so leading fields survive.
    """
    if not isinstance(response, str) or not response:
        return None
    response = strip_thinking_tags(response)
    # After stripping thinking tags, also strip any residual chain-of-thought
    # that may not be inside formal <think> tags (reasoning models sometimes
    # emit plain-text reasoning before the JSON).
    cleaned = response
    # Find all JSON blocks (object or array) in the response.
    # Track (start, end, parsed_value) so we can pick the outermost block.
    json_blocks: list = []
    # Try objects
    for m in re.finditer(r'\{', cleaned):
        start = m.start()
        # Find matching closing brace from the end
        end = cleaned.rfind('}', start)
        if end > start:
            candidate = cleaned[start:end + 1]
            try:
                parsed = json.loads(candidate)
                json_blocks.append((start, end, parsed))
            except Exception:
                pass
    # Try arrays
    for m in re.finditer(r'\[', cleaned):
        start = m.start()
        end = cleaned.rfind(']', start)
        if end > start:
            candidate = cleaned[start:end + 1]
            try:
                parsed = json.loads(candidate)
                json_blocks.append((start, end, parsed))
            except Exception:
                pass
    if json_blocks:
        # Truncated stream (finish_reason=length): the outer object is cut
        # but an inner nested block can still parse and masquerade as the
        # result (run b099dcaa: closest_prior_work entry surfaced instead of
        # the contract). Prefer brace-repair of the full prefix in that case.
        if _is_truncated_json(cleaned):
            repaired = _repair_truncated_json(cleaned)
            if repaired is not None:
                logger.warning("parse_json_from_llm: recovered truncated JSON via brace repair")
                return repaired
        # Prefer the outermost block (smallest start, largest end span).
        # When spans are equal (same block found as both dict and array),
        # prefer the one found later (array closing bracket tends to be
        # more reliable for arrays wrapping dicts).
        outermost = max(json_blocks, key=lambda b: (b[1] - b[0], -b[0]))
        return outermost[2]
    # Truncated stream with no parseable block: try to recover the prefix.
    repaired = _repair_truncated_json(cleaned)
    if repaired is not None:
        logger.warning("parse_json_from_llm: recovered truncated JSON via brace repair")
        return repaired
    return None


def call_llm_json(
    prompt: str,
    *,
    attempts: int = 3,
    temperature: float = 0.3,
    tier: str = "default",
    client: Any = None,
    model: Any = None,
    system: Optional[str] = None,
    max_tokens: int = 8192,
    call_fn: Optional[Any] = None,
    # e0ec9e91: blind full re-runs on truncation doubled slow-endpoint cost
    # (11:22:10 truncated -> 16384 retry -> 503, twice the burn, zero gain).
    # JSON callers self-repair via the re-ask loop below (parse error +
    # excerpt), so default to json_mode: partial payload now, guided repair
    # next — never a blind second full run.
    retry_on_truncation: bool = False,
) -> Optional[Any]:
    """Self-correcting structured-output call.

    Parses the LLM response as JSON; when parsing fails, re-asks with the
    parse error and the offending response excerpt appended so the model can
    repair its own malformed output (control-data flow separation: the parse
    protocol stays deterministic, only the content is re-generated).
    Returns None only after exhausting attempts.

    retry_on_truncation=False is json_mode for slow endpoints: return the
    partial payload for repair instead of a second full ~300s re-run.
    """
    if call_fn is None:
        from .llm import call_llm as call_fn

    last_response = ""
    last_error = "empty response"
    for attempt in range(max(1, attempts)):
        if attempt == 0:
            current_prompt = prompt
        else:
            current_prompt = (
                f"{prompt}\n\nYour previous response could not be parsed as JSON."
                f"\nParse problem: {last_error}"
                f"\nPrevious response (truncated): {str(last_response)[:400]}"
                "\nReturn ONLY the valid JSON object or array now — no prose, no markdown fences."
            )
        try:
            try:
                last_response = call_fn(
                    prompt=current_prompt,
                    client=client,
                    temperature=max(0.0, temperature - 0.1 * attempt),
                    model=model,
                    tier=tier,
                    system=system,
                    max_tokens=max_tokens,
                    retry_on_truncation=retry_on_truncation,
                )
            except TypeError:
                # Injectable stubs / legacy call_fns may not accept
                # retry_on_truncation — fall back to the legacy keyword set.
                last_response = call_fn(
                    prompt=current_prompt,
                    client=client,
                    temperature=max(0.0, temperature - 0.1 * attempt),
                    model=model,
                    tier=tier,
                    system=system,
                    max_tokens=max_tokens,
                )
        except TypeError:
            # Injectable stubs may not accept the full keyword set.
            try:
                last_response = call_fn(current_prompt)
            except Exception as exc:  # R10 re-audit (M6): infra errors retry too
                last_error = f"{type(exc).__name__}: {exc}"
                continue
        except Exception as exc:  # R10 re-audit (M6): attempts cover call errors, not just parse errors
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        try:
            parsed = parse_json_from_llm(last_response)
        except Exception as exc:  # defensive: parse must never raise
            parsed = None
            last_error = str(exc)
        else:
            if parsed is not None:
                return parsed
            last_error = "no JSON object or array found in response"
    # Re-audit (M6): call errors are retried, not raised, so the last error
    # stays diagnosable via logs even though the return is None.
    logger.warning("call_llm_json exhausted %d attempt(s); last problem: %s", max(1, attempts), last_error)
    return None


# Safety refusals / empty stubs that must never be treated as valid agent output.
# 0dfc1c58: the OpenAI-compatible proxy returned an HTTP-200 plain-English
# quota message ("Sorry, to prevent abuse of free resources ... topup")
# which passed min_chars and poisoned both the debate proposer argument and
# every engineer code generation (188 chars -> SyntaxError line 1 x48).
# Quota/billing payloads must be degenerate everywhere.
_PROVIDER_QUOTA_PATTERNS = (
    re.compile(r"prevent\s+abuse\s+of\s+free\s+resources", re.I),
    re.compile(r"accounts?\s+that\s+have\s+not\s+been\s+recharged", re.I),
    re.compile(r"increase\s+the\s+free\s+quota", re.I),
    re.compile(r"only\s+try\s+\d+\s+times", re.I),
    re.compile(r"aihubmix", re.I),
    re.compile(r"console\.\S*topup", re.I),
    re.compile(r"\btopup\b", re.I),
    re.compile(r"insufficient\s+(balance|credits?|quota|funds?)", re.I),
    re.compile(r"payment\s+required", re.I),
    re.compile(r"billing|out\s+of\s+credits?|quota\s+exceeded", re.I),
)

_DEGENERATE_LLM_PATTERNS = (
    re.compile(r"^\s*user\s+safety\s*:\s*safe\s*$", re.I),
    re.compile(r"^\s*safe\s*$", re.I),
    re.compile(r"i\s+can'?t\s+(help|assist)\s+with\s+that", re.I),
    re.compile(r"as\s+an\s+ai\s+(language\s+)?model", re.I),
    * _PROVIDER_QUOTA_PATTERNS,
)


def is_provider_quota_text(text: Any) -> bool:
    """True when `text` is a provider billing/quota notice, not model output.

    0dfc1c58 root cause: HTTP-200 quota prose treated as code/argument.
    Checked independently of length so short notices are also caught.
    """
    if text is None:
        return False
    try:
        compact = re.sub(r"\s+", " ", str(text)).strip()
    except Exception:
        return False
    if not compact:
        return False
    return any(p.search(compact) for p in _PROVIDER_QUOTA_PATTERNS)


def looks_like_python_code(text: Any) -> bool:
    """Heuristic: does `text` contain any Python code markers at all?"""
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    markers = (
        "import ", "from ", "def ", "class ", "for ", "while ", "return",
        "print(", "json.dumps", "np.", "numpy", "sklearn", "scipy", "pd.",
        "=", "(", ")",
    )
    return any(m in lowered for m in markers)


_THINKING_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_thinking_tags(text: str) -> str:
    """Remove LLM chain-of-thought `` blocks from output.

    Models like Qwen/DeepSeek wrap internal reasoning in `` tags.
    These leak into stored arguments and corrupt JSON extraction.
    """
    if not text:
        return text
    return _THINKING_TAG_RE.sub("", text).strip()


def strip_markdown_headers(text: str) -> str:
    """Remove leading markdown headings so length checks measure body prose."""
    lines = []
    for line in str(text or "").splitlines():
        if re.match(r"^\s*#{1,6}\s+", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def is_degenerate_llm_output(text: Any, *, min_chars: int = 40) -> bool:
    """True when an LLM response is empty, a safety stub, or otherwise non-substantive."""
    if text is None:
        return True
    if not isinstance(text, str):
        text = str(text)
    text = strip_thinking_tags(text)
    body = strip_markdown_headers(text)
    if len(body) < min_chars:
        return True
    compact = re.sub(r"\s+", " ", body).strip()
    for pattern in _DEGENERATE_LLM_PATTERNS:
        if pattern.search(compact):
            return True
    # Repeated header-only stubs like "# Related Work\\n\\n# Related Work\\n\\nUser Safety: safe"
    if re.search(r"user\s+safety\s*:\s*safe", compact, re.I) and len(compact) < 120:
        return True
    return False


def title_token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of title tokens; used to de-duplicate failed-debate topics."""
    def tokens(value: str) -> set:
        return {t for t in re.findall(r"[a-z0-9]+", (value or "").lower()) if len(t) > 2}

    left, right = tokens(a), tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def log_agent_action(agent_name: str, action: str, details: Dict[str, Any] = None):
    from .run_log import emit_event, get_tracker

    details = details or {}
    logger.info(f"Agent {agent_name}: {action}")
    if config.debug_mode:
        logger.debug(f"Details: {json.dumps(details, default=str)[:500]}")
    tracker = get_tracker()
    run_id = tracker.run_id if tracker else None
    emit_event(
        "agent_action",
        {"action": action, "details": details},
        run_id=run_id,
        agent=agent_name,
    )
