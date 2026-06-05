import argparse
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.build_prompt import build_prompt
from src.cli import positive_int
from src.vertex_client import ask_gemma4


DATASET_NAME: str = "hotchpotch/fineweb-2-edu-japanese"
DATASET_ROWS_URL: str = "https://datasets-server.huggingface.co/rows"
SYSTEM_PROMPT: str = "あなたは日本語テキストの教育的価値を判定する専門家です。"
TEXT_PREVIEW_LENGTH: int = 500


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Read generation settings from the command line.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=positive_int, default=20)
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--split", type=str, default="train")

    return parser.parse_args()


def fetch_row(config: str, split: str, offset: int) -> dict[str, Any]:
    # ---------------------------------------------------------
    # Fetch one row from Hugging Face Dataset Viewer API.
    # ---------------------------------------------------------
    query = urlencode(
        {
            "dataset": DATASET_NAME,
            "config": config,
            "split": split,
            "offset": offset,
            "length": 1,
        }
    )

    with urlopen(f"{DATASET_ROWS_URL}?{query}") as response:
        data = json.loads(response.read().decode("utf-8"))

    return dict(data["rows"][0]["row"])


def main() -> None:
    args = parse_args()
    console = Console()

    # ---------------------------------------------------------
    # Fetch dataset rows one by one and judge each text.
    # ---------------------------------------------------------
    console.rule(f"[bold]Start judging {args.count} texts[/bold]")

    for index in range(args.count):
        item = fetch_row(config=args.config, split=args.split, offset=index)
        text = str(item["text"])
        prompt = build_prompt(text)

        # ---------------------------------------------------------
        # Skip rows when the AI response is empty or malformed.
        # ---------------------------------------------------------
        try:
            with console.status(f"[bold]Judging item {index + 1}/{args.count}...[/bold]"):
                judgement = ask_gemma4(prompt=prompt, system_prompt=SYSTEM_PROMPT)
        except Exception as error:
            console.print(
                Panel(
                    str(error),
                    title=f"Skipped {index + 1}",
                    border_style="yellow",
                )
            )
            continue

        # ---------------------------------------------------------
        # Show each judgement with compact metadata and text preview.
        # ---------------------------------------------------------
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="bold cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("Index", str(index))
        table.add_row("ID", str(item["id"]))
        table.add_row("URL", str(item["url"]))
        table.add_row("Text", text[:TEXT_PREVIEW_LENGTH].replace("\n", " "))
        table.add_row("Judgement", judgement.strip())

        console.print(Panel(table, title=f"Result {index + 1}", border_style="green"))

    console.rule("[bold]Done[/bold]")


if __name__ == "__main__":
    main()
