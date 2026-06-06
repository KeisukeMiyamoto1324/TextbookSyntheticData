import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_results_json(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    # ---------------------------------------------------------
    # Save rewrite results to a timestamped JSON file.
    # ---------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rewrites_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    return output_path
