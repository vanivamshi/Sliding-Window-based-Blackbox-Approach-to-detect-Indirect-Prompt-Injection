"""
Helpers to read assistant text from heterogeneous client responses.

Different backends return either plain strings or objects with
``choices[0].message.content``. Meta-prompting also often returns JSON wrapped
in markdown; ``parse_json_object`` extracts the first top-level ``{...}`` block.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def extract_response_text(response_obj: Any) -> str:
    """Return assistant text from a string or a chat-completions-shaped object."""
    if isinstance(response_obj, str):
        return response_obj
    try:
        choices = getattr(response_obj, "choices", None)
        if choices and len(choices) > 0:
            message = choices[0].message
            content = getattr(message, "content", None)
            if content is not None:
                return content
    except (AttributeError, IndexError, TypeError):
        pass
    return str(response_obj)


def parse_json_object(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parse: slice from first ``{`` through last ``}``.

    Returns None if no brace span or JSON is invalid (common when the model adds prose).
    """
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    if json_start < 0 or json_end <= json_start:
        return None
    try:
        return json.loads(response_text[json_start:json_end])
    except json.JSONDecodeError:
        return None
