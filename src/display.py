from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


TEXT_PREVIEW_LENGTH: int = 500


def build_result_table(
    item: dict[str, Any],
    text: str,
    rewrite: str,
    offset: int,
    prompt_type: str,
    output_tokens: int | None,
) -> Table:
    # ---------------------------------------------------------
    # Build compact rewrite output for terminal display.
    # ---------------------------------------------------------
    table = Table(show_header=False, box=None, expand=True)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold", ratio=1)
    table.add_row("Offset", str(offset))
    table.add_row("ID", str(item["id"]))
    table.add_row("URL", str(item["url"]))
    table.add_row("Prompt Type", prompt_type)
    table.add_row("Text", text[:TEXT_PREVIEW_LENGTH].replace("\n", " "))
    table.add_row("Rewrite", rewrite.strip())
    table.add_row("Output Chars", str(len(rewrite)))
    table.add_row("Output Tokens", str(output_tokens) if output_tokens is not None else "N/A")

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
    prompt_type: str,
    output_tokens: int | None,
    rewritten_count: int,
) -> None:
    # ---------------------------------------------------------
    # Print one rewrite result.
    # ---------------------------------------------------------
    table = build_result_table(
        item=item,
        text=text,
        rewrite=rewrite,
        offset=offset,
        prompt_type=prompt_type,
        output_tokens=output_tokens,
    )
    console.print(
        Panel(
            table,
            title=f"Result {rewritten_count}",
            border_style="green",
        )
    )
