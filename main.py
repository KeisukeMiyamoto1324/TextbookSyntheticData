import argparse
from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from src.build_prompt import (
    build_easy_rewrite_prompt,
    build_hard_rewrite_prompt,
    build_medium_rewrite_prompt,
)
from src.cli import non_negative_int, positive_int
from src.dataset_client import iter_rows
from src.display import print_result, print_skip
from src.json_writer import JsonlRecordWriter, build_results_jsonl_path
from src.rewrite_runner import RewriteFailure, RewriteJob, run_rewrite_jobs


SYSTEM_PROMPT: str = ""
PROMPT_BUILDERS: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("hard", build_hard_rewrite_prompt),
    ("medium", build_medium_rewrite_prompt),
    ("easy", build_easy_rewrite_prompt),
)


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
    parser.add_argument("--workers", type=positive_int, default=4)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    output_path = build_results_jsonl_path(args.output_dir)
    row_iterator = iter_rows(
        config=args.config,
        split=args.split,
        start_offset=args.offset,
    )
    written_count = 0
    job_index = 0

    # ---------------------------------------------------------
    # Rewrite rows in small batches and save each record immediately.
    # ---------------------------------------------------------
    console.rule(
        f"[bold]Start rewriting {args.count} texts from offset {args.offset} "
        f"with {args.workers} workers[/bold]"
    )

    with JsonlRecordWriter(output_path) as writer:
        while written_count < args.count:
            jobs: list[RewriteJob] = []

            for _ in range(args.workers):
                if written_count + len(jobs) >= args.count:
                    break

                try:
                    offset, item = next(row_iterator)
                except StopIteration:
                    break

                text = str(item["text"])
                prompt_type, build_prompt = PROMPT_BUILDERS[job_index % len(PROMPT_BUILDERS)]
                prompt = build_prompt(text)
                jobs.append(
                    RewriteJob(
                        index=job_index,
                        offset=offset,
                        item=item,
                        text=text,
                        prompt_type=prompt_type,
                        prompt=prompt,
                    )
                )
                job_index += 1

            if not jobs:
                break

            # ---------------------------------------------------------
            # Run only one small batch to avoid storing many records.
            # ---------------------------------------------------------
            with console.status("[bold]Rewriting texts...[/bold]"):
                results = run_rewrite_jobs(
                    jobs=jobs,
                    workers=args.workers,
                    system_prompt=SYSTEM_PROMPT,
                )

            for result in results:
                if isinstance(result, RewriteFailure):
                    print_skip(console, result.message, result.offset)
                    continue

                written_count += 1
                record = result.record
                writer.write(record)

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

                if written_count >= args.count:
                    break

    # ---------------------------------------------------------
    # Finish terminal output after all saved records are flushed.
    # ---------------------------------------------------------
    console.print(f"Saved JSONL: {output_path}")
    console.rule("[bold]Done[/bold]")


if __name__ == "__main__":
    main()
