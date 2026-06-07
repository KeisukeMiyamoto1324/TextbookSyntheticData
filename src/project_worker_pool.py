from src.worker_scaler import WorkerScaler


class ProjectWorkerPool:
    def __init__(self, project_ids: list[str], max_workers_per_project: int) -> None:
        if not project_ids:
            raise ValueError("Google Cloud project ID is not set")

        self.project_ids = project_ids
        self.max_workers_per_project = max_workers_per_project
        self.scalers = {
            project_id: WorkerScaler(max_workers=max_workers_per_project)
            for project_id in project_ids
        }
        self.active_counts = {project_id: 0 for project_id in project_ids}
        self.index = 0

    @property
    def max_worker_count(self) -> int:
        # ---------------------------------------------------------
        # Return total maximum workers across all configured projects.
        # ---------------------------------------------------------
        return len(self.project_ids) * self.max_workers_per_project

    @property
    def worker_count(self) -> int:
        # ---------------------------------------------------------
        # Return total current AIMD workers across all projects.
        # ---------------------------------------------------------
        return sum(scaler.worker_count for scaler in self.scalers.values())

    def borrow_project_id(self) -> str | None:
        # ---------------------------------------------------------
        # Borrow one slot from the least busy available project.
        # ---------------------------------------------------------
        selected_project_id: str | None = None

        for step in range(len(self.project_ids)):
            project_index = (self.index + step) % len(self.project_ids)
            project_id = self.project_ids[project_index]
            active_count = self.active_counts[project_id]

            if active_count >= self.scalers[project_id].worker_count:
                continue

            if selected_project_id is None:
                selected_project_id = project_id
                continue

            if active_count < self.active_counts[selected_project_id]:
                selected_project_id = project_id

        if selected_project_id is None:
            return None

        self.active_counts[selected_project_id] += 1
        selected_index = self.project_ids.index(selected_project_id)
        self.index = (selected_index + 1) % len(self.project_ids)

        return selected_project_id

    def return_project_id(self, project_id: str) -> None:
        # ---------------------------------------------------------
        # Return one active project worker slot after job completion.
        # ---------------------------------------------------------
        self.active_counts[project_id] -= 1

    def record_success(self, project_id: str) -> None:
        # ---------------------------------------------------------
        # Scale only the project that handled the successful request.
        # ---------------------------------------------------------
        self.scalers[project_id].record_success()

    def record_error(self, project_id: str, error: Exception) -> None:
        # ---------------------------------------------------------
        # Scale only the project that handled the failed request.
        # ---------------------------------------------------------
        self.scalers[project_id].record_error(error)
