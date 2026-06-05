import os
import time

import openai
from google.auth import default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request

from src.config import ENABLE_THINKING, LOCATION, MODEL, PROJECT_ID_ENV_NAME
from src.retry import run_with_backoff


REQUEST_INTERVAL_SECONDS: float = 1.0
EMPTY_CHOICES_ERROR_MESSAGE: str = "response does not contain choices"


class EmptyChoicesError(Exception):
    pass


def create_client() -> openai.OpenAI:
    # ---------------------------------------------------------
    # Refresh the access token before each API client creation.
    # This avoids using an expired token in long-running jobs.
    # ---------------------------------------------------------
    project_id = os.environ[PROJECT_ID_ENV_NAME]
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


def ask_gemma4(prompt: str, system_prompt: str) -> str:
    # ---------------------------------------------------------
    # Retry temporary API errors with exponential backoff.
    # ---------------------------------------------------------
    def request_completion() -> openai.types.chat.ChatCompletion:
        client = create_client()
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

    return response.choices[0].message.content or ""
