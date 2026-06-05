import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import urlopen

from src.retry import run_with_backoff


DATASET_NAME: str = "hotchpotch/fineweb-2-edu-japanese"
DATASET_ROWS_URL: str = "https://datasets-server.huggingface.co/rows"
BATCH_SIZE: int = 100


def is_rate_limit_error(error: Exception) -> bool:
    # ---------------------------------------------------------
    # Detect Hugging Face Dataset Viewer rate limit errors.
    # ---------------------------------------------------------
    return isinstance(error, HTTPError) and error.code == 429


def fetch_rows(config: str, split: str, offset: int, length: int) -> list[dict[str, Any]]:
    # ---------------------------------------------------------
    # Fetch a batch of rows from Hugging Face Dataset Viewer API.
    # ---------------------------------------------------------
    query = urlencode(
        {
            "dataset": DATASET_NAME,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )

    def request_rows() -> list[dict[str, Any]]:
        with urlopen(f"{DATASET_ROWS_URL}?{query}") as response:
            data = json.loads(response.read().decode("utf-8"))

        return list(data["rows"])

    return run_with_backoff(request_rows, is_rate_limit_error)


def iter_rows(
    config: str,
    split: str,
    start_offset: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    # ---------------------------------------------------------
    # Yield dataset rows while reducing API calls with batching.
    # ---------------------------------------------------------
    offset = start_offset

    while True:
        rows = fetch_rows(config=config, split=split, offset=offset, length=BATCH_SIZE)

        if not rows:
            break

        for row in rows:
            yield int(row["row_idx"]), dict(row["row"])

        offset += len(rows)
