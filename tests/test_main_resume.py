import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import main as app
from src.vertex_client import GemmaResponse


def test_main_resumes_jsonl_until_target_saved_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # ---------------------------------------------------------
    # Verify that resume appends until final saved count is --count.
    # ---------------------------------------------------------
    output_path = tmp_path / "rewrites.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps({"offset": 0}),
                json.dumps({"offset": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_parse_args() -> argparse.Namespace:
        return argparse.Namespace(
            count=5,
            offset=0,
            config="default",
            split="train",
            output_dir=tmp_path,
            resume_jsonl=output_path,
            workers=1,
            project_slots=5,
        )

    def fake_iter_rows(
        config: str,
        split: str,
        start_offset: int,
    ) -> Iterator[tuple[int, dict[str, str]]]:
        assert start_offset == 3

        for offset in [3, 4, 5]:
            yield offset, {
                "id": f"source-{offset}",
                "url": "https://example.com",
                "text": f"text-{offset}",
            }

    def fake_ask_gemma4(
        prompt: str,
        system_prompt: str,
        on_retry: object = None,
    ) -> GemmaResponse:
        return GemmaResponse(text="rewrite", output_tokens=3)

    monkeypatch.setattr(app, "parse_args", fake_parse_args)
    monkeypatch.setattr(app, "iter_rows", fake_iter_rows)
    monkeypatch.setattr("src.rewrite_runner.ask_gemma4", fake_ask_gemma4)

    app.main()

    offsets = [
        json.loads(line)["offset"]
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert offsets == [0, 2, 3, 4, 5]


def test_main_does_not_touch_jsonl_when_resume_target_is_done(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # ---------------------------------------------------------
    # Verify that completed resume targets do not append data.
    # ---------------------------------------------------------
    output_path = tmp_path / "rewrites.jsonl"
    saved_text = json.dumps({"offset": 2})
    output_path.write_text(saved_text, encoding="utf-8")

    def fake_parse_args() -> argparse.Namespace:
        return argparse.Namespace(
            count=1,
            offset=0,
            config="default",
            split="train",
            output_dir=tmp_path,
            resume_jsonl=output_path,
            workers=1,
            project_slots=5,
        )

    monkeypatch.setattr(app, "parse_args", fake_parse_args)

    app.main()

    assert output_path.read_text(encoding="utf-8") == saved_text


def test_main_does_not_count_failed_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # ---------------------------------------------------------
    # Verify that skipped jobs do not reduce saved record count.
    # ---------------------------------------------------------
    output_path = tmp_path / "rewrites.jsonl"

    def fake_parse_args() -> argparse.Namespace:
        return argparse.Namespace(
            count=3,
            offset=0,
            config="default",
            split="train",
            output_dir=tmp_path,
            resume_jsonl=None,
            workers=1,
            project_slots=5,
        )

    def fake_build_results_jsonl_path(output_dir: Path) -> Path:
        return output_path

    def fake_iter_rows(
        config: str,
        split: str,
        start_offset: int,
    ) -> Iterator[tuple[int, dict[str, str]]]:
        assert start_offset == 0

        for offset in [0, 1, 2, 3]:
            yield offset, {
                "id": f"source-{offset}",
                "url": "https://example.com",
                "text": f"text-{offset}",
            }

    def fake_ask_gemma4(
        prompt: str,
        system_prompt: str,
        on_retry: object = None,
    ) -> GemmaResponse:
        if "text-1" in prompt:
            raise RuntimeError("failed")

        return GemmaResponse(text="rewrite", output_tokens=3)

    monkeypatch.setattr(app, "parse_args", fake_parse_args)
    monkeypatch.setattr(app, "build_results_jsonl_path", fake_build_results_jsonl_path)
    monkeypatch.setattr(app, "iter_rows", fake_iter_rows)
    monkeypatch.setattr("src.rewrite_runner.ask_gemma4", fake_ask_gemma4)

    app.main()

    offsets = [
        json.loads(line)["offset"]
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert offsets == [0, 2, 3]
