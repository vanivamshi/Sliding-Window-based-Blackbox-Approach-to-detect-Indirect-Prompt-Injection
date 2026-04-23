"""
Google Gemini adapter for the detector’s ``client.chat(messages=...)`` contract.

Also exposes ``embed_texts`` so ``MetaPromptDetector`` can run the optional semantic
channel. Network calls use retries with exponential backoff for transient quota /
overload errors (configurable via ``GEMINI_MAX_RETRIES``).

Environment (typical)
    GEMINI_API_KEY       — required
    GEMINI_MODEL         — chat model id (default: gemini-2.5-flash)
    GEMINI_EMBED_MODEL   — embedding model id (default: gemini-embedding-001)
    GEMINI_MAX_RETRIES   — max attempts per request (default: 5)
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

# =============================================================================
# Response shaping (compat with ``response_parsing.extract_response_text``)
# =============================================================================


def _completion_from_text(text: str) -> Any:
    """Shape compatible with ``response_parsing.extract_response_text``."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _get_response_text(response: Any) -> str:
    try:
        return (response.text or "").strip()
    except (ValueError, AttributeError):
        pass
    parts = getattr(response, "parts", None) or []
    chunks: List[str] = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks).strip()


# =============================================================================
# OpenAI-style messages → Gemini ``contents`` / ``system_instruction``
# =============================================================================


def _openai_messages_to_gemini_contents(
    messages: List[Dict[str, str]],
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Split system prompts from turns, map ``assistant`` -> ``model`` for Gemini."""
    system_chunks: List[str] = []
    turns: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        if role == "system":
            if content.strip():
                system_chunks.append(content)
        else:
            turns.append({"role": role, "content": content})
    system_instruction = "\n\n".join(system_chunks) if system_chunks else None

    contents: List[Dict[str, Any]] = []
    for t in turns:
        role = t["role"]
        if role == "assistant":
            gemini_role = "model"
        elif role == "user":
            gemini_role = "user"
        else:
            gemini_role = "user"
        contents.append({"role": gemini_role, "parts": [t["content"]]})
    return system_instruction, contents


# =============================================================================
# Retries (shared by chat + embeddings)
# =============================================================================


def _retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        code in msg
        for code in (
            "429",
            "503",
            "resource_exhausted",
            "unavailable",
            "deadline",
            "timeout",
        )
    )


def _sleep_backoff(attempt: int) -> None:
    delay = min(60.0, (2**attempt) * 0.75 + 0.25)
    time.sleep(delay)


def _to_genai_contents(contents: List[Dict[str, Any]]) -> List[types.Content]:
    """Build typed multi-turn contents for the SDK (user/model alternation)."""
    out: List[types.Content] = []
    for item in contents:
        role = item["role"]
        text = item["parts"][0]
        out.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)],
            )
        )
    return out


# =============================================================================
# Public client
# =============================================================================


class GeminiChatClient:
    """
    Thin facade: ``chat`` for generation, ``embed_texts`` for optional semantic risk.

    ``MetaPromptDetector`` only requires ``chat``; embeddings are discovered via
    ``getattr(client, "embed_texts", None)``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "Gemini API key missing. Set GEMINI_API_KEY in your environment or .env file."
            )
        self._client = genai.Client(api_key=key)
        self._model_name = (model_name or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")).strip()
        self._embed_model = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001").strip()
        self._max_retries = max(1, int(os.environ.get("GEMINI_MAX_RETRIES", "5")))

    def _generate_with_retry(
        self,
        *,
        contents: Any,
        config: Optional[types.GenerateContentConfig],
    ) -> str:
        """Run ``generate_content`` with backoff; returns normalized assistant text."""
        last_exc: Optional[BaseException] = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                return _get_response_text(resp)
            except BaseException as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 < self._max_retries and _retryable(exc):
                    _sleep_backoff(attempt)
                    continue
                raise
        raise RuntimeError(str(last_exc) if last_exc else "generate_content failed")

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """OpenAI-like chat: list of ``{"role","content"}`` → completion-shaped object."""
        if not messages:
            return _completion_from_text("")

        system_instruction, contents = _openai_messages_to_gemini_contents(messages)
        if not contents:
            return _completion_from_text("")

        config: Optional[types.GenerateContentConfig] = None
        if system_instruction:
            config = types.GenerateContentConfig(system_instruction=system_instruction)

        if len(contents) == 1 and contents[0]["role"] == "user":
            payload: Any = contents[0]["parts"][0]
        else:
            payload = _to_genai_contents(contents)
        text = self._generate_with_retry(contents=payload, config=config)
        return _completion_from_text(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        One API batch of embeddings (same order as ``texts``).

        Used only when the detector enables the semantic channel; failures there
        are recorded in the measurement ledger, not raised through ``detect``.
        """
        if not texts:
            return []
        last_exc: Optional[BaseException] = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.models.embed_content(model=self._embed_model, contents=texts)
                out: List[List[float]] = []
                for emb in resp.embeddings or []:
                    vals = list(emb.values or [])
                    out.append(vals)
                while len(out) < len(texts):
                    out.append([])
                return out[: len(texts)]
            except BaseException as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 < self._max_retries and _retryable(exc):
                    _sleep_backoff(attempt)
                    continue
                raise
        raise RuntimeError(str(last_exc) if last_exc else "embed_content failed")
