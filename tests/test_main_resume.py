import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import main as app
from src.vertex_client import GemmaResponse


class FakeProjectRouter:
    def get_project_ids(self) -> list[str]:
        # ---------------------------------------------------------
        # Return stable project IDs for main runner tests.
        # ---------------------------------------------------------
        return ["project-0"]


def patch_project_router(monkeypatch: pytest.MonkeyPatch) -> None:
    # ---------------------------------------------------------
    # Use test project IDs for progress and queue scheduling.
    # ---------------------------------------------------------
    fake_router = FakeProjectRouter()

    monkeypatch.setattr(app, "project_router", fake_router)
    monkeypatch.setattr("src.rewrite_runner.project_router", fake_router)


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
            provider="vertex",
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
        provider: str,
        prompt: str,
        system_prompt: str,
        on_retry: object = None,
        project_id: str | None = None,
    ) -> GemmaResponse:
        return GemmaResponse(text="rewrite", output_tokens=3)

    monkeypatch.setattr(app, "parse_args", fake_parse_args)
    monkeypatch.setattr(app, "iter_rows", fake_iter_rows)
    monkeypatch.setattr("src.rewrite_runner.ask_gemma4", fake_ask_gemma4)
    patch_project_router(monkeypatch)

    app.main()

    offsets = [
        json.loads(line)["offset"]
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert offsets == [0, 2, 3, 4, 5]


def test_main_progress_total_uses_requested_count_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # ---------------------------------------------------------
    # Verify that progress total shows requested count, not remaining count.
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
    add_task_kwargs: dict[str, object] = {}

    class FakeProgress:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def add_task(self, description: str, **kwargs: object) -> int:
            add_task_kwargs.update(kwargs)
            return 1

        def update(self, task_id: int, **kwargs: object) -> None:
            pass

    def fake_parse_args() -> argparse.Namespace:
        return argparse.Namespace(
            count=5,
            offset=0,
            config="default",
            split="train",
            output_dir=tmp_path,
            resume_jsonl=output_path,
            workers=1,
            provider="vertex",
        )

    def fake_iter_rows(
        config: str,
        split: str,
        start_offset: int,
    ) -> Iterator[tuple[int, dict[str, str]]]:
        for offset in [3, 4, 5]:
            yield offset, {
                "id": f"source-{offset}",
                "url": "https://example.com",
                "text": f"text-{offset}",
            }

    def fake_ask_gemma4(
        provider: str,
        prompt: str,
        system_prompt: str,
        on_retry: object = None,
        project_id: str | None = None,
    ) -> GemmaResponse:
        return GemmaResponse(text="rewrite", output_tokens=3)

    monkeypatch.setattr(app, "parse_args", fake_parse_args)
    monkeypatch.setattr(app, "iter_rows", fake_iter_rows)
    monkeypatch.setattr(app, "Progress", FakeProgress)
    monkeypatch.setattr("src.rewrite_runner.ask_gemma4", fake_ask_gemma4)
    patch_project_router(monkeypatch)

    app.main()

    assert add_task_kwargs["total"] == 5
    assert add_task_kwargs["completed"] == 2


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
            provider="vertex",
        )

    monkeypatch.setattr(app, "parse_args", fake_parse_args)
    patch_project_router(monkeypatch)

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
            provider="vertex",
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
        provider: str,
        prompt: str,
        system_prompt: str,
        on_retry: object = None,
        project_id: str | None = None,
    ) -> GemmaResponse:
        if "text-1" in prompt:
            raise RuntimeError("failed")

        return GemmaResponse(text="rewrite", output_tokens=3)

    monkeypatch.setattr(app, "parse_args", fake_parse_args)
    monkeypatch.setattr(app, "build_results_jsonl_path", fake_build_results_jsonl_path)
    monkeypatch.setattr(app, "iter_rows", fake_iter_rows)
    monkeypatch.setattr("src.rewrite_runner.ask_gemma4", fake_ask_gemma4)
    patch_project_router(monkeypatch)

    app.main()

    offsets = [
        json.loads(line)["offset"]
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert offsets == [0, 2, 3]
