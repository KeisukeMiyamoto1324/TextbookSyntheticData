from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src import vertex_auth


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
    vertex_auth.credentials_cache = None
    vertex_auth.client_cache.clear()


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

    monkeypatch.setattr(vertex_auth, "default", fake_default)
    monkeypatch.setattr(vertex_auth.openai, "OpenAI", fake_openai)

    client_0 = vertex_auth.create_client("project-0")
    client_1 = vertex_auth.create_client("project-0")

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

    monkeypatch.setattr(vertex_auth, "default", fake_default)
    monkeypatch.setattr(vertex_auth.openai, "OpenAI", fake_openai)

    vertex_auth.create_client("project-0")

    assert credentials.refresh_count == 1
    assert used_tokens == ["token-1"]

    reset_auth_cache()
