from collections.abc import Callable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from threading import Lock
from typing import Any

from src.rewrite_record import RewriteRecord
from src.vertex_client import ask_gemma4


@dataclass(frozen=True)
class RewriteJob:
    index: int
    offset: int
    item: dict[str, Any]
    text: str
    prompt_type: str
    prompt: str


@dataclass(frozen=True)
class RewriteSuccess:
    index: int
    job: RewriteJob
    record: RewriteRecord
    current_workers: int = 1


@dataclass(frozen=True)
class RewriteFailure:
    index: int
    offset: int
    message: str
    current_workers: int = 1


RewriteResult = RewriteSuccess | RewriteFailure
GetNextJob = Callable[[int], RewriteJob | None]
SUCCESS_SCALE_STEP: int = 3
REQUEST_FAILURE_SCALE_FACTOR: float = 0.8


class WorkerScaler:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        self.current_workers = 1
        self.success_streak = 0
        self.lock = Lock()

    @property
    def worker_count(self) -> int:
        # ---------------------------------------------------------
        # Return the active concurrency limit with thread safety.
        # ---------------------------------------------------------
        with self.lock:
            return self.current_workers

    def record_success(self) -> None:
        # ---------------------------------------------------------
        # Increase concurrency slowly after stable successful output.
        # ---------------------------------------------------------
        with self.lock:
            self.success_streak += 1

            if self.success_streak < SUCCESS_SCALE_STEP:
                return

            self.current_workers = min(self.max_workers, self.current_workers + 1)
            self.success_streak = 0

    def record_error(self, error: Exception) -> None:
        # ---------------------------------------------------------
        # Reduce concurrency when Vertex reports temporary pressure.
        # ---------------------------------------------------------
        with self.lock:
            self.success_streak = 0

            if is_request_failure_error(error):
                self.current_workers = max(
                    1,
                    int(self.current_workers * REQUEST_FAILURE_SCALE_FACTOR),
                )


def is_request_failure_error(error: Exception) -> bool:
    # ---------------------------------------------------------
    # Detect request failures that should reduce API pressure.
    # ---------------------------------------------------------
    status_code = getattr(error, "status_code", None)

    if status_code in {429, 500, 502, 503, 504}:
        return True

    error_text = str(error).lower()
    failure_words = ("429", "timeout", "timed out", "connection", "rate limit")

    return any(word in error_text for word in failure_words)


def run_rewrite_jobs(
    jobs: list[RewriteJob],
    workers: int,
    system_prompt: str,
) -> list[RewriteResult]:
    # ---------------------------------------------------------
    # Run rewrite jobs with threads and return results in input order.
    # ---------------------------------------------------------
    results: list[RewriteResult] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(_run_rewrite_job, job, system_prompt): job for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]

            try:
                record = future.result()
            except Exception as error:
                results.append(
                    RewriteFailure(
                        index=job.index,
                        offset=job.offset,
                        message=str(error),
                    )
                )
                continue

            results.append(
                RewriteSuccess(
                    index=job.index,
                    job=job,
                    record=record,
                )
            )

    return sorted(results, key=lambda result: result.index)


def iter_rewrite_job_queue(
    get_next_job: GetNextJob,
    workers: int,
    system_prompt: str,
) -> Iterator[RewriteResult]:
    # ---------------------------------------------------------
    # Keep workers busy while AIMD adjusts active concurrency.
    # ---------------------------------------------------------
    scaler = WorkerScaler(max_workers=workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job: dict[Future[RewriteRecord], RewriteJob] = {}

        def submit_available_jobs() -> None:
            while len(future_to_job) < scaler.worker_count:
                job = get_next_job(len(future_to_job))

                if job is None:
                    break

                future_to_job[
                    executor.submit(
                        _run_rewrite_job,
                        job,
                        system_prompt,
                        scaler.record_error,
                    )
                ] = job

        submit_available_jobs()

        while future_to_job:
            done, _ = wait(future_to_job, return_when=FIRST_COMPLETED)

            for future in done:
                job = future_to_job.pop(future)

                try:
                    record = future.result()
                except Exception as error:
                    scaler.record_error(error)
                    yield RewriteFailure(
                        index=job.index,
                        offset=job.offset,
                        message=str(error),
                        current_workers=scaler.worker_count,
                    )
                else:
                    scaler.record_success()
                    yield RewriteSuccess(
                        index=job.index,
                        job=job,
                        record=record,
                        current_workers=scaler.worker_count,
                    )

            submit_available_jobs()


def _run_rewrite_job(
    job: RewriteJob,
    system_prompt: str,
    on_retry: Callable[[Exception], None] | None = None,
) -> RewriteRecord:
    # ---------------------------------------------------------
    # Generate one rewrite and convert it to a record.
    # ---------------------------------------------------------
    response = ask_gemma4(
        prompt=job.prompt,
        system_prompt=system_prompt,
        on_retry=on_retry,
    )

    return RewriteRecord.from_generation(
        offset=job.offset,
        item=job.item,
        text=job.text,
        prompt_type=job.prompt_type,
        prompt=job.prompt,
        response=response,
    )
