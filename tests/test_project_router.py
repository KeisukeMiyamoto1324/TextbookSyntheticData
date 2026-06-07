import pytest

from src.project_router import ProjectRouter, load_project_ids


def clear_project_id_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # ---------------------------------------------------------
    # Remove project ID settings loaded from the real .env file.
    # ---------------------------------------------------------
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    for index in range(20):
        monkeypatch.delenv(f"GOOGLE_CLOUD_PROJECT_{index}", raising=False)


def test_project_router_returns_project_ids_in_order() -> None:
    # ---------------------------------------------------------
    # Verify that project IDs are selected in round-robin order.
    # ---------------------------------------------------------
    router = ProjectRouter(["project-0", "project-1", "project-2"])

    assert router.next_project_id() == "project-0"
    assert router.next_project_id() == "project-1"
    assert router.next_project_id() == "project-2"
    assert router.next_project_id() == "project-0"


def test_project_router_requires_project_id() -> None:
    # ---------------------------------------------------------
    # Verify that missing project IDs fail with a clear error.
    # ---------------------------------------------------------
    router = ProjectRouter([])

    with pytest.raises(ValueError, match="Google Cloud project ID is not set"):
        router.next_project_id()


def test_project_router_borrows_least_busy_project() -> None:
    # ---------------------------------------------------------
    # Verify that borrowed slots are spread across projects.
    # ---------------------------------------------------------
    router = ProjectRouter(
        ["project-0", "project-1", "project-2"],
        max_slots_per_project=2,
    )

    assert router.borrow_project_id() == "project-0"
    assert router.borrow_project_id() == "project-1"
    assert router.borrow_project_id() == "project-2"
    assert router.borrow_project_id() == "project-0"
    assert router.borrow_project_id() == "project-1"


def test_project_router_returns_borrowed_slot() -> None:
    # ---------------------------------------------------------
    # Verify that returned slots are available for new requests.
    # ---------------------------------------------------------
    router = ProjectRouter(
        ["project-0", "project-1"],
        max_slots_per_project=1,
    )

    assert router.borrow_project_id() == "project-0"
    assert router.borrow_project_id() == "project-1"

    router.return_project_id("project-0")

    assert router.borrow_project_id() == "project-0"


def test_load_project_ids_reads_numbered_ids_first(monkeypatch: pytest.MonkeyPatch) -> None:
    # ---------------------------------------------------------
    # Verify that numbered project IDs are preferred.
    # ---------------------------------------------------------
    clear_project_id_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "legacy-project")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_0", "project-0")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_1", "project-1")

    assert load_project_ids() == ["project-0", "project-1"]


def test_load_project_ids_reads_legacy_single_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # ---------------------------------------------------------
    # Verify that the existing single project setting still works.
    # ---------------------------------------------------------
    clear_project_id_env(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "legacy-project")

    assert load_project_ids() == ["legacy-project"]
