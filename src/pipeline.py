from itertools import count

from rich.console import Console

from src.build_prompt import build_prompt
from src.dataset_client import fetch_row
from src.display import build_stats_table, print_result, print_skip
from src.judgement import SKIPPED_KEY, create_stats, update_stats
from src.vertex_client import ask_gemma4


SYSTEM_PROMPT: str = "あなたは日本語テキストの教育的価値を判定する専門家です。"
MAX_TEXT_LENGTH: int = 2048


def judge_text(
    console: Console,
    text: str,
    count_size: int,
    judged_count: int,
    offset: int,
) -> str | None:
    # ---------------------------------------------------------
    # Ask AI to judge one text and return None on failure.
    # ---------------------------------------------------------
    try:
        with console.status(
            f"[bold]Judging item {judged_count + 1}/{count_size}...[/bold]"
        ):
            return ask_gemma4(
                prompt=build_prompt(text),
                system_prompt=SYSTEM_PROMPT,
            )
    except Exception as error:
        print_skip(console, str(error), offset)
        return None


def run_judgements(console: Console, count_size: int, config: str, split: str) -> None:
    # ---------------------------------------------------------
    # Fetch rows, skip invalid texts, judge valid texts, and show stats.
    # ---------------------------------------------------------
    console.rule(f"[bold]Start judging {count_size} texts[/bold]")

    judged_count = 0
    stats = create_stats()

    for offset in count():
        if judged_count >= count_size:
            break

        item = fetch_row(config=config, split=split, offset=offset)
        text = str(item["text"])

        if len(text) > MAX_TEXT_LENGTH:
            print_skip(console, f"Text length is {len(text)} characters.", offset)
            stats[SKIPPED_KEY] += 1
            continue

        judgement = judge_text(
            console=console,
            text=text,
            count_size=count_size,
            judged_count=judged_count,
            offset=offset,
        )

        if judgement is None:
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

    console.rule("[bold]Done[/bold]")
    console.print(build_stats_table(stats))
