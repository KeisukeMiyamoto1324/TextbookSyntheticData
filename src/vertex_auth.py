from datetime import datetime, timedelta, timezone
from threading import Lock

import openai
from google.auth import default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request

from src.config import LOCATION


AUTH_SCOPES: list[str] = ["https://www.googleapis.com/auth/cloud-platform"]
TOKEN_REFRESH_MARGIN_SECONDS: int = 300
credentials_cache: Credentials | None = None
client_cache: dict[str, openai.OpenAI] = {}
auth_lock = Lock()


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
