import json
from urllib.error import HTTPError, URLError

import pytest

from src.dataset_client import fetch_rows, is_dataset_retryable_error


class FakeResponse:
    def __init__(self, data: dict[str, object]) -> None:
        self.data = data

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.data).encode("utf-8")


def test_fetch_rows_retries_temporary_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify that temporary Dataset Viewer errors are retried.
    # ---------------------------------------------------------
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        calls.append(url)

        if len(calls) == 1:
            raise HTTPError(url, 500, "server error", {}, None)

        return FakeResponse(
            {
                "rows": [
                    {
                        "row_idx": 3,
                        "row": {
                            "id": "source-3",
                            "url": "https://example.com",
                            "text": "text",
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr("src.dataset_client.urlopen", fake_urlopen)
    monkeypatch.setattr("src.retry.time.sleep", lambda seconds: None)

    rows = fetch_rows(config="default", split="train", offset=0, length=1)

    assert len(calls) == 2
    assert rows[0]["row_idx"] == 3


def test_dataset_retryable_error_detection() -> None:
    # ---------------------------------------------------------
    # Verify that only temporary fetch errors are retryable.
    # ---------------------------------------------------------
    assert is_dataset_retryable_error(HTTPError("", 429, "", {}, None))
    assert is_dataset_retryable_error(HTTPError("", 503, "", {}, None))
    assert is_dataset_retryable_error(URLError("connection reset"))
    assert is_dataset_retryable_error(TimeoutError("timed out"))
    assert not is_dataset_retryable_error(HTTPError("", 404, "", {}, None))
