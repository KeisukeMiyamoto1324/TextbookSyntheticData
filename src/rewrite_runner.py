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

from src.project_router import project_router
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
FAILURE_SCALE_STEP: int = 2
REQUEST_FAILURE_SCALE_FACTOR: float = 0.8


class WorkerScaler:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        self.current_workers = 1
        self.success_streak = 0
        self.failure_streak = 0
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
            self.failure_streak = 0

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

            if not is_request_failure_error(error):
                self.failure_streak = 0
                return

            self.failure_streak += 1

            if self.failure_streak < FAILURE_SCALE_STEP:
                return

            self.current_workers = max(
                1,
                int(self.current_workers * REQUEST_FAILURE_SCALE_FACTOR),
            )
            self.failure_streak = 0


class ProjectWorkerPool:
    def __init__(self, project_ids: list[str], max_workers_per_project: int) -> None:
        if not project_ids:
            raise ValueError("Google Cloud project ID is not set")

        self.project_ids = project_ids
        self.max_workers_per_project = max_workers_per_project
        self.scalers = {
            project_id: WorkerScaler(max_workers=max_workers_per_project)
            for project_id in project_ids
        }
        self.active_counts = {project_id: 0 for project_id in project_ids}
        self.index = 0

    @property
    def max_worker_count(self) -> int:
        # ---------------------------------------------------------
        # Return total maximum workers across all configured projects.
        # ---------------------------------------------------------
        return len(self.project_ids) * self.max_workers_per_project

    @property
    def worker_count(self) -> int:
        # ---------------------------------------------------------
        # Return total current AIMD workers across all projects.
        # ---------------------------------------------------------
        return sum(scaler.worker_count for scaler in self.scalers.values())

    def borrow_project_id(self) -> str | None:
        # ---------------------------------------------------------
        # Borrow one slot from the least busy available project.
        # ---------------------------------------------------------
        selected_project_id: str | None = None

        for step in range(len(self.project_ids)):
            project_index = (self.index + step) % len(self.project_ids)
            project_id = self.project_ids[project_index]
            active_count = self.active_counts[project_id]

            if active_count >= self.scalers[project_id].worker_count:
                continue

            if selected_project_id is None:
                selected_project_id = project_id
                continue

            if active_count < self.active_counts[selected_project_id]:
                selected_project_id = project_id

        if selected_project_id is None:
            return None

        self.active_counts[selected_project_id] += 1
        selected_index = self.project_ids.index(selected_project_id)
        self.index = (selected_index + 1) % len(self.project_ids)

        return selected_project_id

    def return_project_id(self, project_id: str) -> None:
        # ---------------------------------------------------------
        # Return one active project worker slot after job completion.
        # ---------------------------------------------------------
        self.active_counts[project_id] -= 1

    def record_success(self, project_id: str) -> None:
        # ---------------------------------------------------------
        # Scale only the project that handled the successful request.
        # ---------------------------------------------------------
        self.scalers[project_id].record_success()

    def record_error(self, project_id: str, error: Exception) -> None:
        # ---------------------------------------------------------
        # Scale only the project that handled the failed request.
        # ---------------------------------------------------------
        self.scalers[project_id].record_error(error)


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
    project_ids: list[str] | None = None,
) -> Iterator[RewriteResult]:
    # ---------------------------------------------------------
    # Keep per-project workers busy while AIMD adjusts concurrency.
    # ---------------------------------------------------------
    selected_project_ids = (
        project_router.get_project_ids()
        if project_ids is None
        else project_ids
    )
    pool = ProjectWorkerPool(
        project_ids=selected_project_ids,
        max_workers_per_project=workers,
    )

    with ThreadPoolExecutor(max_workers=pool.max_worker_count) as executor:
        future_to_job: dict[Future[RewriteRecord], tuple[RewriteJob, str]] = {}

        def submit_available_jobs() -> None:
            while len(future_to_job) < pool.worker_count:
                project_id = pool.borrow_project_id()

                if project_id is None:
                    break

                job = get_next_job(len(future_to_job))

                if job is None:
                    pool.return_project_id(project_id)
                    break

                future_to_job[
                    executor.submit(
                        _run_rewrite_job,
                        job,
                        system_prompt,
                        lambda error, selected_project_id=project_id: pool.record_error(
                            selected_project_id,
                            error,
                        ),
                        project_id,
                    )
                ] = (job, project_id)

        submit_available_jobs()

        while future_to_job:
            done, _ = wait(future_to_job, return_when=FIRST_COMPLETED)

            for future in done:
                job, project_id = future_to_job.pop(future)

                try:
                    record = future.result()
                except Exception as error:
                    pool.record_error(project_id, error)
                    pool.return_project_id(project_id)
                    yield RewriteFailure(
                        index=job.index,
                        offset=job.offset,
                        message=str(error),
                        current_workers=pool.worker_count,
                    )
                else:
                    pool.record_success(project_id)
                    pool.return_project_id(project_id)
                    yield RewriteSuccess(
                        index=job.index,
                        job=job,
                        record=record,
                        current_workers=pool.worker_count,
                    )

            submit_available_jobs()


def _run_rewrite_job(
    job: RewriteJob,
    system_prompt: str,
    on_retry: Callable[[Exception], None] | None = None,
    project_id: str | None = None,
) -> RewriteRecord:
    # ---------------------------------------------------------
    # Generate one rewrite and convert it to a record.
    # ---------------------------------------------------------
    response = ask_gemma4(
        prompt=job.prompt,
        system_prompt=system_prompt,
        on_retry=on_retry,
        project_id=project_id,
    )

    return RewriteRecord.from_generation(
        offset=job.offset,
        item=job.item,
        text=job.text,
        prompt_type=job.prompt_type,
        prompt=job.prompt,
        response=response,
    )
