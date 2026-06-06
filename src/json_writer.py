import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from src.rewrite_record import RewriteRecord


def build_results_jsonl_path(output_dir: Path) -> Path:
    # ---------------------------------------------------------
    # Build a timestamped JSON Lines output path.
    # ---------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / f"rewrites_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


class JsonlRecordWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.file = output_path.open("w", encoding="utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.file.close()

    def write(self, record: RewriteRecord) -> None:
        # ---------------------------------------------------------
        # Save one record immediately as a JSON Lines entry.
        # ---------------------------------------------------------
        json.dump(asdict(record), self.file, ensure_ascii=False)
        self.file.write("\n")
        self.file.flush()
