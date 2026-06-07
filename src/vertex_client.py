import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock

import openai
from google.auth import default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request
from limiter import Limiter

from src.config import ENABLE_THINKING, LOCATION, MODEL
from src.project_router import project_router
from src.retry import run_with_backoff


REQUEST_INTERVAL_SECONDS: float = 1.0
EMPTY_CHOICES_ERROR_MESSAGE: str = "response does not contain choices"
RETRYABLE_STATUS_CODES: set[int] = {429, 500, 502, 503, 504}
AUTH_SCOPES: list[str] = ["https://www.googleapis.com/auth/cloud-platform"]
TOKEN_REFRESH_MARGIN_SECONDS: int = 300
REQUEST_LIMITER = Limiter(rate=1, capacity=1)
credentials_cache: Credentials | None = None
client_cache: dict[str, openai.OpenAI] = {}
auth_lock = Lock()


@dataclass(frozen=True)
class GemmaResponse:
    text: str
    output_tokens: int | None


class EmptyChoicesError(Exception):
    pass


def configure_project_slots(max_slots_per_project: int) -> None:
    # ---------------------------------------------------------
    # Set how many logical requests can use each project at once.
    # ---------------------------------------------------------
    project_router.set_max_slots_per_project(max_slots_per_project)


def should_refresh_credentials(credentials: Credentials) -> bool:
    # ---------------------------------------------------------
    # Refresh missing, invalid, expired, or nearly expired credentials.
    # ---------------------------------------------------------
    if not credentials.valid or credentials.expired:
        return True

    expiry = credentials.expiry

    if expiry is None:
        return False

    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)

    refresh_at = datetime.now(timezone.utc) + timedelta(
        seconds=TOKEN_REFRESH_MARGIN_SECONDS,
    )

    return expiry <= refresh_at


def create_client(project_id: str) -> openai.OpenAI:
    # ---------------------------------------------------------
    # Cache credentials and project clients across long-running jobs.
    # ---------------------------------------------------------
    global credentials_cache

    with auth_lock:
        if credentials_cache is None:
            credentials_cache, _ = default(scopes=AUTH_SCOPES)

        if should_refresh_credentials(credentials_cache):
            credentials_cache.refresh(Request())
            client_cache.clear()

        if project_id not in client_cache:
            client_cache[project_id] = openai.OpenAI(
                base_url=(
                    f"https://aiplatform.googleapis.com/v1/projects/{project_id}"
                    f"/locations/{LOCATION}/endpoints/openapi"
                ),
                api_key=credentials_cache.token,
            )

        return client_cache[project_id]


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
        time.sleep(REQUEST_INTERVAL_SECONDS)

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
