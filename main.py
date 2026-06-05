import argparse
from collections.abc import Callable

from rich.console import Console

from src.build_prompt import (
    build_easy_rewrite_prompt,
    build_hard_rewrite_prompt,
    build_medium_rewrite_prompt,
)
from src.cli import non_negative_int, positive_int
from src.dataset_client import iter_rows
from src.display import print_result, print_skip
from src.vertex_client import GemmaResponse, ask_gemma4


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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    rewritten_count = 0

    # ---------------------------------------------------------
    # Read rows until the requested number of valid texts is rewritten.
    # ---------------------------------------------------------
    console.rule(f"[bold]Start rewriting {args.count} texts from offset {args.offset}[/bold]")

    for offset, item in iter_rows(
        config=args.config,
        split=args.split,
        start_offset=args.offset,
    ):
        if rewritten_count >= args.count:
            break

        text = str(item["text"])
        responses: list[tuple[str, GemmaResponse]] = []

        # ---------------------------------------------------------
        # Rewrite one text with all prompt types.
        # ---------------------------------------------------------
        try:
            for prompt_type, build_prompt in PROMPT_BUILDERS:
                status = (
                    f"[bold]Rewriting item {rewritten_count + 1}/{args.count} "
                    f"with {prompt_type} prompt...[/bold]"
                )

                with console.status(status):
                    response = ask_gemma4(
                        prompt=build_prompt(text),
                        system_prompt=SYSTEM_PROMPT,
                    )

                responses.append((prompt_type, response))
        except Exception as error:
            print_skip(console, str(error), offset)
            continue

        rewritten_count += 1
        for prompt_type, response in responses:
            print_result(
                console=console,
                item=item,
                text=text,
                rewrite=response.text,
                offset=offset,
                prompt_type=prompt_type,
                output_tokens=response.output_tokens,
                rewritten_count=rewritten_count,
            )

    # ---------------------------------------------------------
    # Finish terminal output.
    # ---------------------------------------------------------
    console.rule("[bold]Done[/bold]")


if __name__ == "__main__":
    main()
