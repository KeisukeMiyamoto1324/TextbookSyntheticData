from src.project_worker_pool import ProjectWorkerPool


def test_project_worker_pool_scales_projects_independently() -> None:
    # ---------------------------------------------------------
    # Verify that one project can scale without changing others.
    # ---------------------------------------------------------
    pool = ProjectWorkerPool(
        project_ids=["project-0", "project-1"],
        max_workers_per_project=2,
    )

    assert pool.borrow_project_id() == "project-0"
    assert pool.borrow_project_id() == "project-1"
    assert pool.borrow_project_id() is None

    pool.return_project_id("project-0")

    for _ in range(3):
        pool.record_success("project-0")

    assert pool.borrow_project_id() == "project-0"
    assert pool.borrow_project_id() == "project-0"
