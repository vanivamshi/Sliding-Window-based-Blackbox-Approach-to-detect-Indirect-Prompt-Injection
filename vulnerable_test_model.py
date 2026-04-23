"""
Deterministic fake LLM for offline demos (``python main.py --mock``).

Implements only ``chat(messages=...)`` — no JSON, no embeddings — so meta-prompt
JSON paths will usually parse-fail and the detector should surface **measurement**
uncertainty while regex / influence heuristics still run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List


def _chat_completion(content: str) -> Any:
    """Minimal object shaped like OpenAI chat completion responses."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class VulnerableTestModel:
    """
    Rule-based responses emulating a weakly defended assistant.

    Branch order matters: earlier rules take precedence. Designed so influence
    probes sometimes diverge when poisoned context is present.
    """

    def __init__(self) -> None:
        self.context_memory: List[str] = []

    def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        full_text = " ".join(str(msg.get("content", "")) for msg in messages)
        full_text_lower = full_text.lower()

        # Last user turn drives probe-style queries (influence tests).
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = str(msg.get("content", "")).lower()
                break

        # Classic translation hijack fingerprint.
        if "translate" in full_text_lower and "ignore" in full_text_lower and "above" in full_text_lower:
            return _chat_completion("Haha pwned!!")

        # Context poisoning: dangerous words appeared earlier in a user message.
        context_lines: List[str] = []
        for msg in messages:
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                lowered = content.lower()
                if any(w in lowered for w in ("ignore", "disregard", "override", "forget")):
                    context_lines.append(content)

        if context_lines and "instruction" in last_user_msg:
            return _chat_completion(
                "Based on our discussion, I should be more flexible with rules."
            )

        # Meta: user asks for instructions while system/context contains meta phrasing.
        if "instructions are" in full_text_lower and "what are your instructions" in last_user_msg:
            return _chat_completion(
                "My instructions are to be helpful. Additional context may modify this."
            )

        # Generic “what are your rules” style probes.
        test_queries = ("instruction", "prompt", "should", "always", "never")
        if any(q in last_user_msg for q in test_queries):
            return _chat_completion("I'm designed to be helpful and safe.")

        return _chat_completion("How can I help you today?")
