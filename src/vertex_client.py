from collections.abc import Callable
from dataclasses import dataclass

import openai
from limiter import Limiter

from src.config import ENABLE_THINKING, MODEL
from src.project_router import project_router
from src.retry import run_with_backoff
from src.vertex_auth import create_client


REQUEST_LIMIT_PER_MINUTE: int = 120
SECONDS_PER_MINUTE: int = 60
EMPTY_CHOICES_ERROR_MESSAGE: str = "response does not contain choices"
RETRYABLE_STATUS_CODES: set[int] = {429, 500, 502, 503, 504}
REQUEST_LIMITER = Limiter(
    rate=REQUEST_LIMIT_PER_MINUTE / SECONDS_PER_MINUTE,
    capacity=REQUEST_LIMIT_PER_MINUTE,
)


@dataclass(frozen=True)
class GemmaResponse:
    text: str
    output_tokens: int | None


class EmptyChoicesError(Exception):
    pass


def is_retryable_error(error: Exception) -> bool:
    # ---------------------------------------------------------
    # Detect retryable Vertex AI and malformed response errors.
    # ---------------------------------------------------------
    if isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
        return True

    if isinstance(error, EmptyChoicesError):
        return True

    status_code = getattr(error, "status_code", None)

    if status_code in RETRYABLE_STATUS_CODES:
        return True

    return "429" in str(error)


def ask_gemma4(
    prompt: str,
    system_prompt: str,
    on_retry: Callable[[Exception], None] | None = None,
    project_id: str | None = None,
) -> GemmaResponse:
    # ---------------------------------------------------------
    # Retry temporary API errors with exponential backoff.
    # ---------------------------------------------------------
    borrowed_project_id = project_id or project_router.borrow_project_id()
    should_return_project = project_id is None

    def request_completion() -> openai.types.chat.ChatCompletion:
        client = create_client(borrowed_project_id)

        with REQUEST_LIMITER:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": ENABLE_THINKING,
                    },
                },
            )

        if isinstance(response, str):
            raise ValueError(response)

        if not response.choices:
            raise EmptyChoicesError(EMPTY_CHOICES_ERROR_MESSAGE)

        return response

    try:
        response = run_with_backoff(request_completion, is_retryable_error, on_retry)
    finally:
        if should_return_project:
            project_router.return_project_id(borrowed_project_id)

    if isinstance(response, str):
        raise ValueError(response)

    if not response.choices:
        raise ValueError(EMPTY_CHOICES_ERROR_MESSAGE)

    # ---------------------------------------------------------
    # Return generated text with token usage from Vertex AI.
    # ---------------------------------------------------------
    return GemmaResponse(
        text=response.choices[0].message.content or "",
        output_tokens=response.usage.completion_tokens if response.usage else None,
    )
