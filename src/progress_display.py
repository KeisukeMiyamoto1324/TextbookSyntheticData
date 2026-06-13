from datetime import datetime, timedelta

from rich.console import RenderableType
from rich.progress import ProgressColumn, Task


def format_token_count(token_count: int) -> str:
    # ---------------------------------------------------------
    # Format generated tokens with compact K/M/B units.
    # ---------------------------------------------------------
    units = (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    )

    for threshold, suffix in units:
        if token_count >= threshold:
            value = token_count / threshold

            if value >= 100:
                return f"{value:.0f}{suffix} token"

            return f"{value:.1f}{suffix} token"

    return f"{token_count} token"


class TokenCountColumn(ProgressColumn):
    # ---------------------------------------------------------
    # Show compact cumulative generated output token count.
    # ---------------------------------------------------------
    def render(self, task: Task) -> RenderableType:
        return format_token_count(int(task.fields["total_output_tokens"]))


class EstimatedFinishColumn(ProgressColumn):
    # ---------------------------------------------------------
    # Show the estimated local finish time from remaining seconds.
    # ---------------------------------------------------------
    def render(self, task: Task) -> RenderableType:
        if task.finished:
            return "ETA done"

        if task.time_remaining is None:
            return "ETA calculating"

        finish_time = datetime.now() + timedelta(seconds=task.time_remaining)

        if finish_time.date() == datetime.now().date():
            return f"ETA {finish_time:%H:%M:%S}"

        return f"ETA {finish_time:%Y-%m-%d %H:%M:%S}"
