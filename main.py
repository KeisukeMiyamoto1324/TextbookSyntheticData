import argparse

from rich.console import Console

from src.build_prompt import build_rewrite_prompt
from src.cli import positive_int
from src.dataset_client import iter_rows
from src.display import print_result, print_skip
from src.vertex_client import ask_gemma4


SYSTEM_PROMPT: str = "あなたは日本語テキストを大学生向け教材文に書き換える専門家です。"


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Read generation settings from the command line.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=positive_int, default=3)
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
    console.rule(f"[bold]Start rewriting {args.count} texts[/bold]")

    for offset, item in iter_rows(config=args.config, split=args.split):
        if rewritten_count >= args.count:
            break

        text = str(item["text"])

        # ---------------------------------------------------------
        # Rewrite one text. Failed AI responses are skipped.
        # ---------------------------------------------------------
        try:
            with console.status(
                f"[bold]Rewriting item {rewritten_count + 1}/{args.count}...[/bold]"
            ):
                rewrite = ask_gemma4(
                    prompt=build_rewrite_prompt(text),
                    system_prompt=SYSTEM_PROMPT,
                )
        except Exception as error:
            print_skip(console, str(error), offset)
            continue

        rewritten_count += 1
        print_result(
            console=console,
            item=item,
            text=text,
            rewrite=rewrite,
            offset=offset,
            rewritten_count=rewritten_count,
        )

    # ---------------------------------------------------------
    # Finish terminal output.
    # ---------------------------------------------------------
    console.rule("[bold]Done[/bold]")


if __name__ == "__main__":
    main()
