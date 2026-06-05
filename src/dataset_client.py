import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET_NAME: str = "hotchpotch/fineweb-2-edu-japanese"
DATASET_ROWS_URL: str = "https://datasets-server.huggingface.co/rows"


def fetch_row(config: str, split: str, offset: int) -> dict[str, Any]:
    # ---------------------------------------------------------
    # Fetch one row from Hugging Face Dataset Viewer API.
    # ---------------------------------------------------------
    query = urlencode(
        {
            "dataset": DATASET_NAME,
            "config": config,
            "split": split,
            "offset": offset,
            "length": 1,
        }
    )

    with urlopen(f"{DATASET_ROWS_URL}?{query}") as response:
        data = json.loads(response.read().decode("utf-8"))

    return dict(data["rows"][0]["row"])
