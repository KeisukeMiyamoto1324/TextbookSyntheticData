import time
from collections.abc import Callable
from typing import TypeVar


MAX_RETRY_COUNT: int = 5

T = TypeVar("T")


def run_with_backoff(operation: Callable[[], T], should_retry: Callable[[Exception], bool]) -> T:
    # ---------------------------------------------------------
    # Retry temporary API errors with exponential backoff.
    # ---------------------------------------------------------
    for retry_count in range(MAX_RETRY_COUNT + 1):
        try:
            return operation()
        except Exception as error:
            if not should_retry(error) or retry_count >= MAX_RETRY_COUNT:
                raise

            time.sleep(2**retry_count)

    raise RuntimeError("retry loop ended unexpectedly")
