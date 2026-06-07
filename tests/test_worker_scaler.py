from src.worker_scaler import WorkerScaler


def test_worker_scaler_increases_after_success_streak() -> None:
    # ---------------------------------------------------------
    # Verify that AIMD increases concurrency after stable success.
    # ---------------------------------------------------------
    scaler = WorkerScaler(max_workers=3)

    for _ in range(3):
        scaler.record_success()

    assert scaler.current_workers == 2


def test_worker_scaler_does_not_exceed_max_workers() -> None:
    # ---------------------------------------------------------
    # Verify that AIMD respects the configured worker ceiling.
    # ---------------------------------------------------------
    scaler = WorkerScaler(max_workers=2)

    for _ in range(9):
        scaler.record_success()

    assert scaler.current_workers == 2


def test_worker_scaler_reduces_twenty_percent_on_request_failure() -> None:
    # ---------------------------------------------------------
    # Verify that two request failures reduce concurrency by twenty percent.
    # ---------------------------------------------------------
    scaler = WorkerScaler(max_workers=10)

    for _ in range(12):
        scaler.record_success()

    scaler.record_error(RuntimeError("429 rate limit"))
    assert scaler.current_workers == 5

    scaler.record_error(RuntimeError("429 rate limit"))

    assert scaler.current_workers == 4


def test_worker_scaler_reduces_twenty_percent_on_temporary_server_error() -> None:
    # ---------------------------------------------------------
    # Verify that temporary server errors use the same two-failure rule.
    # ---------------------------------------------------------
    class FakeStatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"status {status_code}")
            self.status_code = status_code

    scaler = WorkerScaler(max_workers=10)

    for _ in range(9):
        scaler.record_success()

    scaler.record_error(FakeStatusError(503))
    assert scaler.current_workers == 4

    scaler.record_error(FakeStatusError(503))

    assert scaler.current_workers == 3


def test_worker_scaler_success_resets_failure_streak() -> None:
    # ---------------------------------------------------------
    # Verify that failures must be consecutive to reduce workers.
    # ---------------------------------------------------------
    scaler = WorkerScaler(max_workers=10)

    for _ in range(12):
        scaler.record_success()

    scaler.record_error(RuntimeError("timeout"))
    scaler.record_success()
    scaler.record_error(RuntimeError("timeout"))

    assert scaler.current_workers == 5
