import argparse
from itertools import count

from rich.console import Console

from src.build_prompt import build_prompt
from src.cli import positive_int
from src.dataset_client import fetch_row
from src.display import build_stats_table, print_result, print_skip
from src.judgement import SKIPPED_KEY, create_stats, update_stats
from src.vertex_client import ask_gemma4


SYSTEM_PROMPT: str = "あなたは日本語テキストの教育的価値を判定する専門家です。"
MAX_TEXT_LENGTH: int = 2048


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Read generation settings from the command line.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=positive_int, default=20)
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--split", type=str, default="train")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    stats = create_stats()
    judged_count = 0

    # ---------------------------------------------------------
    # Read rows until the requested number of valid texts is judged.
    # ---------------------------------------------------------
    console.rule(f"[bold]Start judging {args.count} texts[/bold]")

    for offset in count():
        if judged_count >= args.count:
            break

        item = fetch_row(config=args.config, split=args.split, offset=offset)
        text = str(item["text"])

        if len(text) > MAX_TEXT_LENGTH:
            print_skip(console, f"Text length is {len(text)} characters.", offset)
            stats[SKIPPED_KEY] += 1
            continue

        # ---------------------------------------------------------
        # Judge one text. Failed AI responses are skipped.
        # ---------------------------------------------------------
        try:
            with console.status(
                f"[bold]Judging item {judged_count + 1}/{args.count}...[/bold]"
            ):
                judgement = ask_gemma4(
                    prompt=build_prompt(text),
                    system_prompt=SYSTEM_PROMPT,
                )
        except Exception as error:
            print_skip(console, str(error), offset)
            stats[SKIPPED_KEY] += 1
            continue

        judged_count += 1
        update_stats(stats, judgement)
        print_result(
            console=console,
            item=item,
            text=text,
            judgement=judgement,
            offset=offset,
            judged_count=judged_count,
        )

    # ---------------------------------------------------------
    # Show final judgement statistics.
    # ---------------------------------------------------------
    console.rule("[bold]Done[/bold]")
    console.print(build_stats_table(stats))


if __name__ == "__main__":
    main()
