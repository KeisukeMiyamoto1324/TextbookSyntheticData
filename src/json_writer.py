import json
from dataclasses import asdict, dataclass
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from types import TracebackType
from typing import Self

from src.rewrite_record import RewriteRecord


@dataclass(frozen=True)
class JsonlResumeState:
    max_offset: int
    record_count: int


def build_results_jsonl_path(output_dir: Path) -> Path:
    # ---------------------------------------------------------
    # Build a timestamped JSON Lines output path.
    # ---------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / f"rewrites_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def read_resume_state(jsonl_path: Path) -> JsonlResumeState:
    # ---------------------------------------------------------
    # Read a JSONL file and return saved record count and offset.
    # ---------------------------------------------------------
    max_offset: int | None = None
    record_count = 0

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except JSONDecodeError as error:
                message = f"invalid JSON at line {line_number}: {error.msg}"
                raise ValueError(message) from error

            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not a JSON object")

            if "offset" not in record:
                raise ValueError(f"line {line_number} does not contain offset")

            offset = int(record["offset"])

            if max_offset is None or offset > max_offset:
                max_offset = offset

            record_count += 1

    if max_offset is None:
        raise ValueError("resume JSONL does not contain records")

    return JsonlResumeState(
        max_offset=max_offset,
        record_count=record_count,
    )


def read_max_offset(jsonl_path: Path) -> int:
    # ---------------------------------------------------------
    # Read a JSONL file and return its largest saved offset.
    # ---------------------------------------------------------
    return read_resume_state(jsonl_path).max_offset


class JsonlRecordWriter:
    def __init__(self, output_path: Path, append: bool = False) -> None:
        self.output_path = output_path
        needs_leading_newline = False

        if append and output_path.stat().st_size > 0:
            with output_path.open("rb") as file:
                file.seek(-1, 2)
                needs_leading_newline = file.read(1) != b"\n"

        mode = "a" if append else "w"
        self.file = output_path.open(mode, encoding="utf-8")

        if needs_leading_newline:
            self.file.write("\n")

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
