from datetime import datetime, timedelta

from rich.console import RenderableType
from rich.progress import ProgressColumn, Task


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
