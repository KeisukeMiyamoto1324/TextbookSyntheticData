import json
from pathlib import Path

from src.json_writer import JsonlRecordWriter, build_results_jsonl_path
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
