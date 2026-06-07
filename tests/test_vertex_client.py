from types import SimpleNamespace

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


class FakeStringResponseCompletions:
    def __init__(self) -> None:
        self.call_count = 0

    def create(self, **kwargs: object) -> object:
        # ---------------------------------------------------------
        # Return malformed string responses before a valid response.
        # ---------------------------------------------------------
        self.call_count += 1

        if self.call_count <= 3:
            return "upstream connect error"

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


def test_ask_gemma4_retries_string_errors_with_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify malformed string errors wait 1, 4, and 16 seconds.
    # ---------------------------------------------------------
    sleep_seconds: list[int] = []
    completions = FakeStringResponseCompletions()
    limiter = FakeLimiter()

    def fake_create_client(project_id: str) -> FakeClient:
        return FakeClient(completions)

    monkeypatch.setattr(vertex_client, "project_router", ProjectRouter(["project-0"]))
    monkeypatch.setattr(vertex_client, "create_client", fake_create_client)
    monkeypatch.setattr(vertex_client, "REQUEST_LIMITER", limiter)
    monkeypatch.setattr("src.retry.time.sleep", sleep_seconds.append)

    response = vertex_client.ask_gemma4(
        prompt="prompt",
        system_prompt="system",
    )

    assert response.text == "rewrite"
    assert sleep_seconds == [1, 4, 16]
    assert limiter.enter_count == 4


def test_vertex_retries_every_generation_error() -> None:
    # ---------------------------------------------------------
    # Verify that every Vertex generation error is retried.
    # ---------------------------------------------------------
    assert vertex_client.is_retryable_error(FakeStatusError(429))
    assert vertex_client.is_retryable_error(FakeStatusError(503))
    assert vertex_client.is_retryable_error(FakeStatusError(404))
    assert vertex_client.is_retryable_error(ValueError("upstream connect error"))
