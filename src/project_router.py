import os
from threading import Condition

from src.config import PROJECT_ID_ENV_NAME, PROJECT_ID_ENV_PREFIX


class ProjectRouter:
    def __init__(self, project_ids: list[str], max_slots_per_project: int = 5) -> None:
        self.project_ids = project_ids
        self.index = 0
        self.max_slots_per_project = max_slots_per_project
        self.active_counts = [0 for _ in project_ids]
        self.condition = Condition()

    def set_max_slots_per_project(self, max_slots_per_project: int) -> None:
        # ---------------------------------------------------------
        # Configure the maximum concurrent logical requests per project.
        # ---------------------------------------------------------
        with self.condition:
            self.max_slots_per_project = max_slots_per_project
            self.condition.notify_all()

    def next_project_id(self) -> str:
        # ---------------------------------------------------------
        # Return project IDs in order for parallel API requests.
        # ---------------------------------------------------------
        if not self.project_ids:
            raise ValueError("Google Cloud project ID is not set")

        with self.condition:
            project_id = self.project_ids[self.index]
            self.index = (self.index + 1) % len(self.project_ids)

        return project_id

    def borrow_project_id(self) -> str:
        # ---------------------------------------------------------
        # Borrow the least busy project slot for one logical request.
        # ---------------------------------------------------------
        if not self.project_ids:
            raise ValueError("Google Cloud project ID is not set")

        with self.condition:
            while True:
                project_index = self._find_available_project_index()

                if project_index is not None:
                    self.active_counts[project_index] += 1
                    self.index = (project_index + 1) % len(self.project_ids)
                    return self.project_ids[project_index]

                self.condition.wait()

    def return_project_id(self, project_id: str) -> None:
        # ---------------------------------------------------------
        # Return a borrowed project slot after final success or failure.
        # ---------------------------------------------------------
        with self.condition:
            if project_id not in self.project_ids:
                raise ValueError("borrowed project ID is unknown")

            project_index = self.project_ids.index(project_id)

            if self.active_counts[project_index] <= 0:
                raise ValueError("project slot was not borrowed")

            self.active_counts[project_index] -= 1
            self.condition.notify_all()

    def _find_available_project_index(self) -> int | None:
        # ---------------------------------------------------------
        # Select the least busy available project with round-robin ties.
        # ---------------------------------------------------------
        selected_index: int | None = None

        for step in range(len(self.project_ids)):
            project_index = (self.index + step) % len(self.project_ids)
            active_count = self.active_counts[project_index]

            if active_count >= self.max_slots_per_project:
                continue

            if selected_index is None:
                selected_index = project_index
                continue

            if active_count < self.active_counts[selected_index]:
                selected_index = project_index

        return selected_index


def load_project_ids() -> list[str]:
    # ---------------------------------------------------------
    # Read numbered project IDs first, then the legacy single ID.
    # ---------------------------------------------------------
    project_ids: list[str] = []
    project_keys = [
        key for key in os.environ
        if key.startswith(PROJECT_ID_ENV_PREFIX)
        and key[len(PROJECT_ID_ENV_PREFIX):].isdigit()
    ]
    project_keys = sorted(
        project_keys,
        key=lambda key: int(key[len(PROJECT_ID_ENV_PREFIX):]),
    )

    for key in project_keys:
        project_id = os.environ[key].strip()

        if project_id:
            project_ids.append(project_id)

    if project_ids:
        return project_ids

    project_id = os.environ.get(PROJECT_ID_ENV_NAME, "").strip()

    if project_id:
        return [project_id]

    return []


project_router = ProjectRouter(load_project_ids())
