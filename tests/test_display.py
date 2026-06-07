from rich.console import Console

from src.display import print_result, print_skip


def test_display_prints_generated_markup_as_plain_text() -> None:
    # ---------------------------------------------------------
    # Verify that generated bracket text does not break Rich.
    # ---------------------------------------------------------
    console = Console(record=True, width=100)

    print_result(
        console=console,
        item={
            "id": "source[/s]",
            "url": "https://example.com/[/s]",
        },
        text="source [/s] text",
        rewrite="rewrite [/s] text",
        offset=1,
        prompt_type="high-school[/s]",
        output_tokens=10,
        rewritten_count=1,
    )
    print_skip(console=console, message="skip [/s] reason", offset=2)

    output = console.export_text()

    assert "rewrite [/s] text" in output
    assert "skip [/s] reason" in output
