import time

import pytest

from src.rewrite_runner import (
    RewriteFailure,
    RewriteJob,
    RewriteSuccess,
    iter_rewrite_job_queue,
    run_rewrite_jobs,
)
from src.vertex_client import GemmaResponse


def build_job(index: int) -> RewriteJob:
    # ---------------------------------------------------------
    # Build a minimal rewrite job for runner tests.
    # ---------------------------------------------------------
    return RewriteJob(
        index=index,
        offset=100 + index,
        item={
            "id": f"id-{index}",
            "url": f"https://example.com/{index}",
        },
        text=f"text-{index}",
        prompt_type="high-school",
        prompt=f"prompt-{index}",
    )


def test_run_rewrite_jobs_with_one_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    # ---------------------------------------------------------
    # Verify that the runner creates records with one worker.
    # ---------------------------------------------------------
    def fake_ask_gemma4(prompt: str, system_prompt: str) -> GemmaResponse:
        return GemmaResponse(text=f"rewrite for {prompt}", output_tokens=10)

    monkeypatch.setattr("src.rewrite_runner.ask_gemma4", fake_ask_gemma4)

    results = run_rewrite_jobs(
        jobs=[build_job(0)],
        workers=1,
        system_prompt="system",
    )

    assert len(results) == 1
    assert isinstance(results[0], RewriteSuccess)
    assert results[0].record.rewrite == "rewrite for prompt-0"


def test_run_rewrite_jobs_with_multiple_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    # ---------------------------------------------------------
    # Verify that multiple jobs can be processed by threads.
    # ---------------------------------------------------------
    def fake_ask_gemma4(prompt: str, system_prompt: str) -> GemmaResponse:
        return GemmaResponse(text=f"rewrite for {prompt}", output_tokens=10)

    monkeypatch.setattr("src.rewrite_runner.ask_gemma4", fake_ask_gemma4)

    results = run_rewrite_jobs(
        jobs=[build_job(0), build_job(1), build_job(2)],
        workers=4,
        system_prompt="system",
    )

    assert len(results) == 3
    assert all(isinstance(result, RewriteSuccess) for result in results)


def test_run_rewrite_jobs_keeps_successes_when_one_job_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify that one failed job does not remove successful jobs.
    # ---------------------------------------------------------
    def fake_ask_gemma4(prompt: str, system_prompt: str) -> GemmaResponse:
        if prompt == "prompt-1":
            raise RuntimeError("failed")

        return GemmaResponse(text=f"rewrite for {prompt}", output_tokens=10)

    monkeypatch.setattr("src.rewrite_runner.ask_gemma4", fake_ask_gemma4)

    results = run_rewrite_jobs(
        jobs=[build_job(0), build_job(1), build_job(2)],
        workers=4,
        system_prompt="system",
    )

    assert isinstance(results[0], RewriteSuccess)
    assert isinstance(results[1], RewriteFailure)
    assert isinstance(results[2], RewriteSuccess)
    assert results[1].message == "failed"


def test_run_rewrite_jobs_returns_input_order_after_out_of_order_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify that slow earlier jobs still appear first in results.
    # ---------------------------------------------------------
    def fake_ask_gemma4(prompt: str, system_prompt: str) -> GemmaResponse:
        if prompt == "prompt-0":
            time.sleep(0.02)

        return GemmaResponse(text=f"rewrite for {prompt}", output_tokens=10)

    monkeypatch.setattr("src.rewrite_runner.ask_gemma4", fake_ask_gemma4)

    results = run_rewrite_jobs(
        jobs=[build_job(0), build_job(1), build_job(2)],
        workers=4,
        system_prompt="system",
    )

    assert [result.index for result in results] == [0, 1, 2]


def test_iter_rewrite_job_queue_refills_worker_after_one_job_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ---------------------------------------------------------
    # Verify that a finished worker receives the next job immediately.
    # ---------------------------------------------------------
    jobs = iter([build_job(0), build_job(1), build_job(2)])

    def get_next_job(active_count: int) -> RewriteJob | None:
        try:
            return next(jobs)
        except StopIteration:
            return None

    def fake_ask_gemma4(prompt: str, system_prompt: str) -> GemmaResponse:
        if prompt == "prompt-0":
            time.sleep(0.05)

        return GemmaResponse(text=f"rewrite for {prompt}", output_tokens=10)

    monkeypatch.setattr("src.rewrite_runner.ask_gemma4", fake_ask_gemma4)

    results = list(
        iter_rewrite_job_queue(
            get_next_job=get_next_job,
            workers=2,
            system_prompt="system",
        )
    )

    assert [result.index for result in results] == [1, 2, 0]
