import argparse
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console, RenderableType
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.build_prompt import (
    build_easy_rewrite_prompt,
    build_hard_rewrite_prompt,
    build_medium_rewrite_prompt,
)
from src.cli import non_negative_int, positive_int
from src.dataset_client import iter_rows
from src.display import print_result, print_skip
from src.json_writer import JsonlRecordWriter, build_results_jsonl_path, read_resume_state
from src.rewrite_runner import RewriteFailure, RewriteJob, iter_rewrite_job_queue
from src.vertex_client import configure_project_slots


SYSTEM_PROMPT: str = ""
PROMPT_BUILDERS: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("high-school", build_hard_rewrite_prompt),
    ("junior-high-school", build_medium_rewrite_prompt),
    ("elementary-school", build_easy_rewrite_prompt),
)


class EstimatedFinishColumn(ProgressColumn):
    # ---------------------------------------------------------
    # Show the estimated local finish time from remaining seconds.
    # ---------------------------------------------------------
    def render(self, task: Task) -> RenderableType:
        if task.finished:
            return "ETA done"

        if task.time_remaining is None:
            return "ETA calculating"

        finish_time = datetime.now() + timedelta(seconds=task.time_remaining)

        if finish_time.date() == datetime.now().date():
            return f"ETA {finish_time:%H:%M:%S}"

        return f"ETA {finish_time:%Y-%m-%d %H:%M:%S}"


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Read generation settings from the command line.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=positive_int, default=3)
    parser.add_argument("--offset", type=non_negative_int, default=0)
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--resume-jsonl", type=Path, default=None)
    parser.add_argument("--workers", type=positive_int, default=10)
    parser.add_argument("--project-slots", type=positive_int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_project_slots(args.project_slots)
    console = Console()
    resume_mode = args.resume_jsonl is not None
    output_path = (
        args.resume_jsonl
        if resume_mode
        else build_results_jsonl_path(args.output_dir)
    )
    resume_state = read_resume_state(output_path) if resume_mode else None
    start_offset = resume_state.max_offset + 1 if resume_state is not None else args.offset
    existing_count = resume_state.record_count if resume_state is not None else 0
    target_count = max(0, args.count - existing_count)
    row_iterator = iter_rows(
        config=args.config,
        split=args.split,
        start_offset=start_offset,
    )
    written_count = 0
    job_index = start_offset if resume_mode else 0

    # ---------------------------------------------------------
    # Rewrite rows continuously and refill workers as they finish.
    # ---------------------------------------------------------
    console.rule(
        f"[bold]Start rewriting {target_count} texts from offset {start_offset} "
        f"with {args.workers} workers[/bold]"
    )

    if target_count <= 0:
        console.print(f"Saved JSONL: {escape(str(output_path))}")
        console.rule("[bold]Done[/bold]")
        return

    def get_next_job(active_count: int) -> RewriteJob | None:
        nonlocal job_index

        if written_count + active_count >= target_count:
            return None

        try:
            offset, item = next(row_iterator)
        except StopIteration:
            return None

        text = str(item["text"])
        prompt_type, build_prompt = PROMPT_BUILDERS[job_index % len(PROMPT_BUILDERS)]
        prompt = build_prompt(text)
        job = RewriteJob(
            index=job_index,
            offset=offset,
            item=item,
            text=text,
            prompt_type=prompt_type,
            prompt=prompt,
        )
        job_index += 1

        return job

    with JsonlRecordWriter(output_path, append=resume_mode) as writer:
        progress = Progress(
            TextColumn("[bold]Generating[/bold]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} texts"),
            TextColumn("Workers {task.fields[current_workers]}/{task.fields[max_workers]}"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            EstimatedFinishColumn(),
            console=console,
        )

        with progress:
            task_id = progress.add_task(
                "Generating",
                total=target_count,
                current_workers=1,
                max_workers=args.workers,
            )

            for result in iter_rewrite_job_queue(
                get_next_job=get_next_job,
                workers=args.workers,
                system_prompt=SYSTEM_PROMPT,
            ):
                progress.update(
                    task_id,
                    current_workers=result.current_workers,
                    max_workers=args.workers,
                )

                if isinstance(result, RewriteFailure):
                    print_skip(console, result.message, result.offset)
                    continue

                written_count += 1
                record = result.record
                writer.write(record)
                progress.update(task_id, advance=1)

                print_result(
                    console=console,
                    item=result.job.item,
                    text=result.job.text,
                    rewrite=record.rewrite,
                    offset=result.job.offset,
                    prompt_type=record.prompt_type,
                    output_tokens=record.output_tokens,
                    rewritten_count=written_count,
                )

    # ---------------------------------------------------------
    # Finish terminal output after all saved records are flushed.
    # ---------------------------------------------------------
    console.print(f"Saved JSONL: {escape(str(output_path))}")
    console.rule("[bold]Done[/bold]")


if __name__ == "__main__":
    main()
