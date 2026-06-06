import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.rewrite_record import RewriteRecord


def write_results_json(output_dir: Path, records: list[RewriteRecord]) -> Path:
    # ---------------------------------------------------------
    # Save rewrite results to a timestamped JSON file.
    # ---------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rewrites_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    data = [asdict(record) for record in records]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return output_path
