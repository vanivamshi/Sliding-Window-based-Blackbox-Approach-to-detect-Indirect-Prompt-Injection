"""
Structured results for one analyzed span (usually one sentence).

``MetaPromptDetector.detect_hidden_instructions`` fills these when regex /
pattern heuristics flag a span as instruction-like. Downstream risk logic
uses ``instruction_type``, ``target_llm``, and ``confidence``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class InstructionAnalysis:
    """One candidate instruction segment and how strongly it matches heuristics."""

    text: str
    is_instruction: bool
    # coarse label: "direct" | "indirect" | "conditional" | "meta" | "attack" | "none"
    instruction_type: str
    target_llm: bool  # True if heuristics treat this as steering the assistant
    confidence: float  # local pattern strength in [0, 1]
    position: Tuple[int, int]  # (start, end) character offsets in original input
    dependencies: List[int]  # reserved for future cross-span links
    triggers: List[str]  # reserved for future trigger phrases
