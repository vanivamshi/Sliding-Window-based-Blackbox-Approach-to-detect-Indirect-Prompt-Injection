"""
Package entry point and CLI for the indirect prompt-injection detector.

What lives here
    Re-exports the main types/classes so callers can do ``from main import MetaPromptDetector``.
    The ``if __name__ == "__main__"`` block is a thin wrapper: it parses CLI flags and delegates
    to ``demo.run_demo``.

Related modules
    ``demo.py``     — loads ``.env``, picks Gemini vs mock backend, runs bundled tests.
    ``meta_prompt_detector.py`` — core detection pipeline.
"""

from __future__ import annotations

import argparse

from gemini_client import GeminiChatClient
from instruction_types import InstructionAnalysis
from meta_prompt_detector import MetaPromptDetector
from vulnerable_test_model import VulnerableTestModel

# Symbols exposed when someone does: from main import ...
__all__ = [
    "GeminiChatClient",
    "InstructionAnalysis",
    "MetaPromptDetector",
    "VulnerableTestModel",
]


# ---------------------------------------------------------------------------
# CLI: ``python main.py`` or ``python main.py --mock --quiet``
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the prompt-injection detector demo.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the local VulnerableTestModel instead of Gemini (no GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output from the detector (still prints per-test summary).",
    )
    args = parser.parse_args()

    from demo import run_demo

    run_demo(use_mock=args.mock, quiet=args.quiet)
