from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RewriteFailure:
    index: int
    offset: int
    message: str


RewriteResult = RewriteSuccess | RewriteFailure


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


def _run_rewrite_job(job: RewriteJob, system_prompt: str) -> RewriteRecord:
    # ---------------------------------------------------------
    # Generate one rewrite and convert it to a record.
    # ---------------------------------------------------------
    response = ask_gemma4(
        prompt=job.prompt,
        system_prompt=system_prompt,
    )

    return RewriteRecord.from_generation(
        offset=job.offset,
        item=job.item,
        text=job.text,
        prompt_type=job.prompt_type,
        prompt=job.prompt,
        response=response,
    )
