from collections.abc import Callable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from typing import Any

from src.gemma_client import GenerationProvider
from src.gemma_client import ask_gemma4
from src.project_worker_pool import ProjectWorkerPool
from src.project_router import project_router
from src.rewrite_record import RewriteRecord


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


def run_rewrite_jobs(
    jobs: list[RewriteJob],
    workers: int,
    system_prompt: str,
    provider: GenerationProvider = "vertex",
) -> list[RewriteResult]:
    # ---------------------------------------------------------
    # Run rewrite jobs with threads and return results in input order.
    # ---------------------------------------------------------
    results: list[RewriteResult] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(_run_rewrite_job, job, system_prompt, provider): job
            for job in jobs
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
    provider: GenerationProvider = "vertex",
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
    selected_project_ids = (
        selected_project_ids
        if provider == "vertex"
        else ["digital-ocean"]
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
                        provider,
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
    provider: GenerationProvider,
    on_retry: Callable[[Exception], None] | None = None,
    project_id: str | None = None,
) -> RewriteRecord:
    # ---------------------------------------------------------
    # Generate one rewrite and convert it to a record.
    # ---------------------------------------------------------
    response = ask_gemma4(
        provider=provider,
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
