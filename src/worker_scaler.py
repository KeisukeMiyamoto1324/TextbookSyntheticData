from threading import Lock


SUCCESS_SCALE_STEP: int = 2
FAILURE_SCALE_STEP: int = 1
REQUEST_FAILURE_SCALE_FACTOR: float = 0.75


class WorkerScaler:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        self.current_workers = 1
        self.success_streak = 0
        self.failure_streak = 0
        self.lock = Lock()

    @property
    def worker_count(self) -> int:
        # ---------------------------------------------------------
        # Return the active concurrency limit with thread safety.
        # ---------------------------------------------------------
        with self.lock:
            return self.current_workers

    def record_success(self) -> None:
        # ---------------------------------------------------------
        # Increase concurrency slowly after stable successful output.
        # ---------------------------------------------------------
        with self.lock:
            self.success_streak += 1
            self.failure_streak = 0

            if self.success_streak < SUCCESS_SCALE_STEP:
                return

            self.current_workers = min(self.max_workers, self.current_workers + 1)
            self.success_streak = 0

    def record_error(self, error: Exception) -> None:
        # ---------------------------------------------------------
        # Reduce concurrency after consecutive request failures.
        # ---------------------------------------------------------
        with self.lock:
            self.success_streak = 0

            if not is_request_failure_error(error):
                self.failure_streak = 0
                return

            self.failure_streak += 1

            if self.failure_streak < FAILURE_SCALE_STEP:
                return

            self.current_workers = max(
                1,
                int(self.current_workers * REQUEST_FAILURE_SCALE_FACTOR),
            )
            self.failure_streak = 0


def is_request_failure_error(error: Exception) -> bool:
    # ---------------------------------------------------------
    # Detect request failures that should reduce API pressure.
    # ---------------------------------------------------------
    status_code = getattr(error, "status_code", None)

    if status_code in {429, 500, 502, 503, 504}:
        return True

    error_text = str(error).lower()
    failure_words = ("429", "timeout", "timed out", "connection", "rate limit")

    return any(word in error_text for word in failure_words)
