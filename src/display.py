from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


TEXT_PREVIEW_LENGTH: int = 500


def build_result_table(item: dict[str, Any], text: str, rewrite: str, offset: int) -> Table:
    # ---------------------------------------------------------
    # Build compact rewrite output for terminal display.
    # ---------------------------------------------------------
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Offset", str(offset))
    table.add_row("ID", str(item["id"]))
    table.add_row("URL", str(item["url"]))
    table.add_row("Text", text[:TEXT_PREVIEW_LENGTH].replace("\n", " "))
    table.add_row("Rewrite", rewrite.strip())

    return table


def print_skip(console: Console, message: str, offset: int) -> None:
    # ---------------------------------------------------------
    # Print skipped row reason.
    # ---------------------------------------------------------
    console.print(
        Panel(
            message,
            title=f"Skipped offset {offset}",
            border_style="yellow",
        )
    )


def print_result(
    console: Console,
    item: dict[str, Any],
    text: str,
    rewrite: str,
    offset: int,
    rewritten_count: int,
) -> None:
    # ---------------------------------------------------------
    # Print one rewrite result.
    # ---------------------------------------------------------
    table = build_result_table(item=item, text=text, rewrite=rewrite, offset=offset)
    console.print(
        Panel(
            table,
            title=f"Result {rewritten_count}",
            border_style="green",
        )
    )
