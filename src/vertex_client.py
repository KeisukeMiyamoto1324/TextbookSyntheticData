import os

import openai
from google.auth import default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request

from src.config import ENABLE_THINKING, LOCATION, MODEL, PROJECT_ID_ENV_NAME
from src.retry import run_with_backoff


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


def is_rate_limit_error(error: Exception) -> bool:
    # ---------------------------------------------------------
    # Detect Vertex AI rate limit errors from OpenAI SDK errors.
    # ---------------------------------------------------------
    if isinstance(error, openai.RateLimitError):
        return True

    return "429" in str(error)


def ask_gemma4(prompt: str, system_prompt: str) -> str:
    # ---------------------------------------------------------
    # Retry rate limit errors with exponential backoff.
    # ---------------------------------------------------------
    def request_completion() -> openai.types.chat.ChatCompletion:
        client = create_client()

        return client.chat.completions.create(
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

    response = run_with_backoff(request_completion, is_rate_limit_error)

    if isinstance(response, str):
        raise ValueError(response)

    if not response.choices:
        raise ValueError("response does not contain choices")

    return response.choices[0].message.content or ""
