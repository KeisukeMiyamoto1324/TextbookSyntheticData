import time
from dataclasses import dataclass

import openai
from google.auth import default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request

from src.config import ENABLE_THINKING, LOCATION, MODEL
from src.project_router import project_router
from src.retry import run_with_backoff


REQUEST_INTERVAL_SECONDS: float = 1.0
EMPTY_CHOICES_ERROR_MESSAGE: str = "response does not contain choices"


@dataclass(frozen=True)
class GemmaResponse:
    text: str
    output_tokens: int | None


class EmptyChoicesError(Exception):
    pass


def create_client(project_id: str) -> openai.OpenAI:
    # ---------------------------------------------------------
    # Refresh the access token before each API client creation.
    # This avoids using an expired token in long-running jobs.
    # ---------------------------------------------------------
    credentials: Credentials
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    if not credentials.valid or credentials.expired:
        credentials.refresh(Request())

    return openai.OpenAI(
        base_url=(
            f"https://aiplatform.googleapis.com/v1/projects/{project_id}"
            f"/locations/{LOCATION}/endpoints/openapi"
        ),
        api_key=credentials.token,
    )


def is_retryable_error(error: Exception) -> bool:
    # ---------------------------------------------------------
    # Detect retryable Vertex AI and malformed response errors.
    # ---------------------------------------------------------
    if isinstance(error, openai.RateLimitError):
        return True

    if isinstance(error, EmptyChoicesError):
        return True

    return "429" in str(error)


def ask_gemma4(prompt: str, system_prompt: str) -> GemmaResponse:
    # ---------------------------------------------------------
    # Retry temporary API errors with exponential backoff.
    # ---------------------------------------------------------
    def request_completion() -> openai.types.chat.ChatCompletion:
        project_id = project_router.next_project_id()
        client = create_client(project_id)
        time.sleep(REQUEST_INTERVAL_SECONDS)

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

    response = run_with_backoff(request_completion, is_retryable_error)

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
