import os
from collections.abc import Callable
from typing import Any

import requests

from src.config import DIGITAL_OCEAN_MODEL
from src.retry import run_with_backoff
from src.vertex_client import EMPTY_CHOICES_ERROR_MESSAGE, EmptyChoicesError, GemmaResponse


DIGITAL_OCEAN_API_KEY_ENV_NAME: str = "DO_KEY"
DIGITAL_OCEAN_CHAT_COMPLETIONS_URL: str = (
    "https://inference.do-ai.run/v1/chat/completions"
)
DIGITAL_OCEAN_TIMEOUT_SECONDS: int = 60
DIGITAL_OCEAN_MAX_TOKENS: int = 300


def is_retryable_error(error: Exception) -> bool:
    # ---------------------------------------------------------
    # Retry every DigitalOcean generation failure before skipping a row.
    # ---------------------------------------------------------
    return True


def ask_gemma4(
    prompt: str,
    system_prompt: str,
    on_retry: Callable[[Exception], None] | None = None,
    project_id: str | None = None,
) -> GemmaResponse:
    # ---------------------------------------------------------
    # Generate text through DigitalOcean's OpenAI-compatible API.
    # ---------------------------------------------------------
    api_key = os.getenv(DIGITAL_OCEAN_API_KEY_ENV_NAME)

    if api_key is None:
        raise RuntimeError(f"{DIGITAL_OCEAN_API_KEY_ENV_NAME} is not set in .env")

    def request_completion() -> dict[str, Any]:
        response = requests.post(
            DIGITAL_OCEAN_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DIGITAL_OCEAN_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "reasoning_effort": "none",
                "max_tokens": DIGITAL_OCEAN_MAX_TOKENS,
            },
            timeout=DIGITAL_OCEAN_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        data = response.json()

        if not data.get("choices"):
            raise EmptyChoicesError(EMPTY_CHOICES_ERROR_MESSAGE)

        return data

    data = run_with_backoff(request_completion, is_retryable_error, on_retry)
    choice = data["choices"][0]
    usage = data.get("usage")

    # ---------------------------------------------------------
    # Return generated text with token usage from DigitalOcean.
    # ---------------------------------------------------------
    return GemmaResponse(
        text=choice["message"].get("content") or "",
        output_tokens=usage.get("completion_tokens") if usage else None,
    )
