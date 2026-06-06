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
from src.json_writer import write_results_json
from src.rewrite_record import RewriteRecord
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
    jobs: list[RewriteJob] = []
    records: list[RewriteRecord] = []

    # ---------------------------------------------------------
    # Read rows and prepare rewrite jobs before threaded generation.
    # ---------------------------------------------------------
    console.rule(
        f"[bold]Start rewriting {args.count} texts from offset {args.offset} "
        f"with {args.workers} workers[/bold]"
    )

    for offset, item in iter_rows(
        config=args.config,
        split=args.split,
        start_offset=args.offset,
    ):
        if len(jobs) >= args.count:
            break

        text = str(item["text"])
        prompt_type, build_prompt = PROMPT_BUILDERS[len(jobs) % len(PROMPT_BUILDERS)]
        prompt = build_prompt(text)
        jobs.append(
            RewriteJob(
                index=len(jobs),
                offset=offset,
                item=item,
                text=text,
                prompt_type=prompt_type,
                prompt=prompt,
            )
        )

    # ---------------------------------------------------------
    # Run prepared jobs while hiding thread handling in the runner.
    # ---------------------------------------------------------
    with console.status("[bold]Rewriting texts...[/bold]"):
        results = run_rewrite_jobs(
            jobs=jobs,
            workers=args.workers,
            system_prompt=SYSTEM_PROMPT,
        )

    rewritten_count = 0

    for result in results:
        if isinstance(result, RewriteFailure):
            print_skip(console, result.message, result.offset)
            continue

        rewritten_count += 1
        record = result.record
        records.append(record)

        print_result(
            console=console,
            item=result.job.item,
            text=result.job.text,
            rewrite=record.rewrite,
            offset=result.job.offset,
            prompt_type=record.prompt_type,
            output_tokens=record.output_tokens,
            rewritten_count=rewritten_count,
        )

    # ---------------------------------------------------------
    # Save JSON results and finish terminal output.
    # ---------------------------------------------------------
    output_path = write_results_json(args.output_dir, records)
    console.print(f"Saved JSON: {output_path}")
    console.rule("[bold]Done[/bold]")


if __name__ == "__main__":
    main()
