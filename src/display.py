from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.judgement import (
    HIGH_LABEL,
    LOW_LABEL,
    PARSE_FAILED_KEY,
    SKIPPED_KEY,
    parse_judgement_label,
)


TEXT_PREVIEW_LENGTH: int = 500


def select_border_color(judgement: str) -> str:
    # ---------------------------------------------------------
    # Change border color by judgement label.
    # ---------------------------------------------------------
    label = parse_judgement_label(judgement)

    if label == HIGH_LABEL:
        return "green"

    if label == LOW_LABEL:
        return "red"

    return "yellow"


def build_result_table(item: dict[str, Any], text: str, judgement: str, offset: int) -> Table:
    # ---------------------------------------------------------
    # Build compact result output for terminal display.
    # ---------------------------------------------------------
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")
    table.add_row("Offset", str(offset))
    table.add_row("ID", str(item["id"]))
    table.add_row("URL", str(item["url"]))
    table.add_row("Text", text[:TEXT_PREVIEW_LENGTH].replace("\n", " "))
    table.add_row("Judgement", judgement.strip())

    return table


def build_stats_table(stats: dict[str, int]) -> Table:
    # ---------------------------------------------------------
    # Build summary statistics for terminal display.
    # ---------------------------------------------------------
    table = Table(title="Statistics")
    table.add_column("Label", style="bold cyan")
    table.add_column("Count", justify="right")
    table.add_row(HIGH_LABEL, str(stats[HIGH_LABEL]))
    table.add_row(LOW_LABEL, str(stats[LOW_LABEL]))
    table.add_row("Parse failed", str(stats[PARSE_FAILED_KEY]))
    table.add_row("Skipped", str(stats[SKIPPED_KEY]))

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
    judgement: str,
    offset: int,
    judged_count: int,
) -> None:
    # ---------------------------------------------------------
    # Print one judgement result.
    # ---------------------------------------------------------
    table = build_result_table(item=item, text=text, judgement=judgement, offset=offset)
    console.print(
        Panel(
            table,
            title=f"Result {judged_count}",
            border_style=select_border_color(judgement),
        )
    )
