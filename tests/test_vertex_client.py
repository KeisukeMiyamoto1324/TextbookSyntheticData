from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pytest

from src import vertex_client
from src.project_router import ProjectRouter


class FakeCompletions:
    def __init__(self) -> None:
        self.call_count = 0

    def create(self, **kwargs: object) -> object:
        # ---------------------------------------------------------
        # Fail once with a retryable error, then return a response.
        # ---------------------------------------------------------
        self.call_count += 1

        if self.call_count == 1:
            raise RuntimeError("429 rate limit")

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="rewrite"),
                ),
            ],
            usage=SimpleNamespace(completion_tokens=12),
        )


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(
            completions=completions,
        )


class FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeLimiter:
    def __init__(self) -> None:
        self.enter_count = 0

    def __enter__(self) -> None:
        # ---------------------------------------------------------
        # Count limiter usage without waiting during tests.
        # ---------------------------------------------------------
        self.enter_count += 1

    def __exit__(self, *args: object) -> None:
        pass


class FakeCredentials:
    def __init__(self, token: str, expiry: datetime) -> None:
        self.token = token
        self.expiry = expiry
        self.valid = True
        self.expired = False
        self.refresh_count = 0

    def refresh(self, request: object) -> None:
        # ---------------------------------------------------------
        # Simulate a refreshed long-lived access token.
        # ---------------------------------------------------------
        self.refresh_count += 1
        self.token = f"token-{self.refresh_count}"
        self.expiry = datetime.now(timezone.utc) + timedelta(hours=1)


def reset_auth_cache() -> None:
    # ---------------------------------------------------------
    # Clear module-level auth cache between credential tests.
    # ---------------------------------------------------------
    vertex_client.credentials_cache = None
    vertex_client.client_cache.clear()


def test_ask_gemma4_keeps_project_id_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify that retryable errors keep the same project ID.
    # ---------------------------------------------------------
    used_project_ids: list[str] = []
    completions = FakeCompletions()
    retry_errors: list[Exception] = []
    limiter = FakeLimiter()

    def fake_create_client(project_id: str) -> FakeClient:
        used_project_ids.append(project_id)
        return FakeClient(completions)

    monkeypatch.setattr(vertex_client, "project_router", ProjectRouter(["project-0", "project-1"]))
    monkeypatch.setattr(vertex_client, "create_client", fake_create_client)
    monkeypatch.setattr(vertex_client, "REQUEST_LIMITER", limiter)
    monkeypatch.setattr(vertex_client.time, "sleep", lambda seconds: None)
    monkeypatch.setattr("src.retry.time.sleep", lambda seconds: None)

    response = vertex_client.ask_gemma4(
        prompt="prompt",
        system_prompt="system",
        on_retry=retry_errors.append,
    )

    assert response.text == "rewrite"
    assert response.output_tokens == 12
    assert used_project_ids == ["project-0", "project-0"]
    assert len(retry_errors) == 1
    assert limiter.enter_count == 2


def test_create_client_reuses_cached_credentials_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify that valid credentials and clients are reused.
    # ---------------------------------------------------------
    reset_auth_cache()
    credentials = FakeCredentials(
        token="token-0",
        expiry=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    created_clients: list[object] = []

    def fake_default(scopes: list[str]) -> tuple[FakeCredentials, None]:
        return credentials, None

    def fake_openai(**kwargs: object) -> object:
        client = SimpleNamespace(**kwargs)
        created_clients.append(client)
        return client

    monkeypatch.setattr(vertex_client, "default", fake_default)
    monkeypatch.setattr(vertex_client.openai, "OpenAI", fake_openai)

    client_0 = vertex_client.create_client("project-0")
    client_1 = vertex_client.create_client("project-0")

    assert client_0 is client_1
    assert credentials.refresh_count == 0
    assert len(created_clients) == 1

    reset_auth_cache()


def test_create_client_refreshes_nearly_expired_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify that nearly expired credentials refresh and rebuild clients.
    # ---------------------------------------------------------
    reset_auth_cache()
    credentials = FakeCredentials(
        token="token-0",
        expiry=datetime.now(timezone.utc) + timedelta(seconds=10),
    )
    used_tokens: list[object] = []

    def fake_default(scopes: list[str]) -> tuple[FakeCredentials, None]:
        return credentials, None

    def fake_openai(**kwargs: object) -> object:
        used_tokens.append(kwargs["api_key"])
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(vertex_client, "default", fake_default)
    monkeypatch.setattr(vertex_client.openai, "OpenAI", fake_openai)

    vertex_client.create_client("project-0")

    assert credentials.refresh_count == 1
    assert used_tokens == ["token-1"]

    reset_auth_cache()


def test_vertex_retryable_error_detection() -> None:
    # ---------------------------------------------------------
    # Verify that temporary Vertex status errors are retried.
    # ---------------------------------------------------------
    assert vertex_client.is_retryable_error(FakeStatusError(429))
    assert vertex_client.is_retryable_error(FakeStatusError(503))
    assert not vertex_client.is_retryable_error(FakeStatusError(404))
