from typing import Any

import pytest

from src import digital_ocean_client


class FakeResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.data


def test_ask_gemma4_posts_digital_ocean_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify DigitalOcean request settings without real API access.
    # ---------------------------------------------------------
    requests: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        requests.append({"url": url, **kwargs})

        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "rewrite",
                        },
                    },
                ],
                "usage": {
                    "completion_tokens": 12,
                },
            }
        )

    monkeypatch.setenv("DO_KEY", "secret")
    monkeypatch.setattr(digital_ocean_client.requests, "post", fake_post)

    response = digital_ocean_client.ask_gemma4(
        prompt="prompt",
        system_prompt="system",
    )

    assert response.text == "rewrite"
    assert response.output_tokens == 12
    assert requests == [
        {
            "url": digital_ocean_client.DIGITAL_OCEAN_CHAT_COMPLETIONS_URL,
            "headers": {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            "json": {
                "model": "gemma-4-31B-it",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "prompt"},
                ],
                "reasoning_effort": "none",
                "max_tokens": 300,
            },
            "timeout": 60,
        }
    ]


def test_ask_gemma4_requires_digital_ocean_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify that missing API keys fail before sending a request.
    # ---------------------------------------------------------
    monkeypatch.delenv("DO_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DO_KEY is not set"):
        digital_ocean_client.ask_gemma4(
            prompt="prompt",
            system_prompt="system",
        )
