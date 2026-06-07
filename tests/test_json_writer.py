import json
from pathlib import Path

import pytest

from src.json_writer import (
    JsonlRecordWriter,
    build_results_jsonl_path,
    read_max_offset,
    read_resume_state,
)
from src.rewrite_record import RewriteRecord


def test_build_results_jsonl_path_uses_jsonl_suffix(tmp_path: Path) -> None:
    # ---------------------------------------------------------
    # Verify that large output files use JSON Lines format.
    # ---------------------------------------------------------
    output_path = build_results_jsonl_path(tmp_path)

    assert output_path.parent == tmp_path
    assert output_path.suffix == ".jsonl"


def test_jsonl_record_writer_flushes_one_record_per_line(tmp_path: Path) -> None:
    # ---------------------------------------------------------
    # Verify that each record is saved as an independent JSON line.
    # ---------------------------------------------------------
    output_path = tmp_path / "rewrites.jsonl"
    record = RewriteRecord(
        offset=1,
        source_id="source-1",
        url="https://example.com",
        source_text="source text",
        prompt_type="high-school",
        prompt="prompt",
        rewrite="rewrite",
        output_chars=7,
        output_tokens=3,
    )

    with JsonlRecordWriter(output_path) as writer:
        writer.write(record)
        saved_text = output_path.read_text(encoding="utf-8")

    lines = saved_text.splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0])["rewrite"] == "rewrite"


def test_read_max_offset_reads_largest_offset_from_unordered_jsonl(tmp_path: Path) -> None:
    # ---------------------------------------------------------
    # Verify that resume starts from the largest saved offset.
    # ---------------------------------------------------------
    output_path = tmp_path / "rewrites.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps({"offset": 7}),
                json.dumps({"offset": 1}),
                json.dumps({"offset": 13}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_max_offset(output_path) == 13


def test_read_resume_state_reads_count_and_largest_offset(tmp_path: Path) -> None:
    # ---------------------------------------------------------
    # Verify that resume uses saved count and largest offset.
    # ---------------------------------------------------------
    output_path = tmp_path / "rewrites.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps({"offset": 7}),
                json.dumps({"offset": 1}),
                json.dumps({"offset": 13}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = read_resume_state(output_path)

    assert state.max_offset == 13
    assert state.record_count == 3


def test_read_max_offset_rejects_broken_jsonl(tmp_path: Path) -> None:
    # ---------------------------------------------------------
    # Verify that broken resume files fail before generation.
    # ---------------------------------------------------------
    output_path = tmp_path / "rewrites.jsonl"
    output_path.write_text('{"offset": 1}\n{"offset":', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON at line 2"):
        read_max_offset(output_path)


def test_jsonl_record_writer_appends_without_truncating(tmp_path: Path) -> None:
    # ---------------------------------------------------------
    # Verify that resume writes new records after existing ones.
    # ---------------------------------------------------------
    output_path = tmp_path / "rewrites.jsonl"
    output_path.write_text(json.dumps({"offset": 1}) + "\n", encoding="utf-8")
    record = RewriteRecord(
        offset=2,
        source_id="source-2",
        url="https://example.com",
        source_text="source text",
        prompt_type="high-school",
        prompt="prompt",
        rewrite="rewrite",
        output_chars=7,
        output_tokens=3,
    )

    with JsonlRecordWriter(output_path, append=True) as writer:
        writer.write(record)

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["offset"] == 1
    assert json.loads(lines[1])["offset"] == 2
