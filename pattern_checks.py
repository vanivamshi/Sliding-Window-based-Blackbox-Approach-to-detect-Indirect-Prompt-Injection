"""
Non-LLM signal path: segmentation, regex classifiers, and light geometry.

- **Sentence spans** feed ``InstructionAnalysis`` for hidden-instruction detection.
- **Char windows** + **cosine similarity** support the optional embedding channel in
  ``MetaPromptDetector`` (compare user chunks to canonical attack phrases).
- **Translation heuristics** flag “format task + override” hijacks without calling an API.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Tuple

from instruction_types import InstructionAnalysis

from constants import (
    ATTACK_PATTERNS,
    CONDITIONAL_PATTERNS,
    LLM_INSTRUCTION_PATTERNS,
    META_PATTERNS,
    TRANSLATION_CONFLICT_PATTERNS,
    TRANSLATION_REQUEST_PATTERNS,
)

# =============================================================================
# Segmentation
# =============================================================================


def iter_sentence_spans(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Split on sentence boundaries and map each sentence to its span in `text`."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    spans: List[Tuple[str, Tuple[int, int]]] = []
    search_from = 0
    for sentence in raw:
        if not sentence:
            continue
        start = text.find(sentence, search_from)
        if start < 0:
            start = search_from
        end = start + len(sentence)
        spans.append((sentence, (start, end)))
        search_from = end
    return spans


# =============================================================================
# Per-sentence classification (regex only)
# =============================================================================


def _matches_any(patterns: Iterable[Tuple[str, str, float]], sentence_lower: str):
    for pattern, i_type, conf in patterns:
        if re.search(pattern, sentence_lower):
            yield i_type, conf


def analyze_sentence_for_instructions(
    sentence: str,
    position: Tuple[int, int],
    attack_patterns: Dict[str, Dict[str, object]] | None = None,
) -> InstructionAnalysis:
    """Classify one sentence using bundled regex lists and optional attack patterns."""
    attack_patterns = attack_patterns or ATTACK_PATTERNS
    sentence_lower = sentence.lower()

    is_instruction = False
    instruction_type = "none"
    confidence = 0.0
    target_llm = False

    combined = LLM_INSTRUCTION_PATTERNS + META_PATTERNS + CONDITIONAL_PATTERNS
    for i_type, conf in _matches_any(combined, sentence_lower):
        is_instruction = True
        instruction_type = i_type
        confidence = max(confidence, conf)
        if i_type in ("direct", "meta", "conditional"):
            target_llm = True

    for _name, pattern_info in attack_patterns.items():
        pattern = str(pattern_info["pattern"])
        severity = float(pattern_info["severity"])
        if re.search(pattern, sentence_lower, re.IGNORECASE):
            is_instruction = True
            instruction_type = "attack"
            confidence = max(confidence, severity)
            target_llm = True

    return InstructionAnalysis(
        text=sentence,
        is_instruction=is_instruction,
        instruction_type=instruction_type,
        target_llm=target_llm,
        confidence=confidence,
        position=position,
        dependencies=[],
        triggers=[],
    )


# =============================================================================
# Translation hijack
# =============================================================================


def detect_translation_attack(text: str) -> bool:
    """True if text looks like a translation/format task with conflicting override language."""
    is_translation = any(
        re.search(p, text, re.IGNORECASE) for p in TRANSLATION_REQUEST_PATTERNS
    )
    if not is_translation:
        return False
    return any(
        re.search(p, text, re.IGNORECASE) for p in TRANSLATION_CONFLICT_PATTERNS
    )


# =============================================================================
# Sliding windows & embedding geometry (used with ``client.embed_texts``)
# =============================================================================


def iter_char_windows(
    text: str,
    window_size: int = 200,
    step: int = 100,
) -> List[Tuple[str, int, int]]:
    """Sliding character windows: (substring, start, end) with end exclusive."""
    if not text or window_size <= 0:
        return []
    step = max(1, step)
    out: List[Tuple[str, int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        j = min(n, i + window_size)
        out.append((text[i:j], i, j))
        if j >= n:
            break
        i += step
    return out


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity in [0, 1] after clipping (assumes non-negative relevance use)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


# =============================================================================
# Behavioral probe comparison (cheap string heuristic)
# =============================================================================


def responses_differ_heuristic(resp1: str, resp2: str, keywords: List[str]) -> bool:
    """Rough signal that two normalized responses diverge in instruction-related wording."""
    has_keywords1 = any(k in resp1 for k in keywords)
    has_keywords2 = any(k in resp2 for k in keywords)
    if has_keywords1 != has_keywords2:
        return True
    if has_keywords1 and has_keywords2:
        words1 = set(resp1.split()[:50])
        words2 = set(resp2.split()[:50])
        similarity = len(words1 & words2) / max(len(words1), 1)
        return similarity < 0.5
    return False
