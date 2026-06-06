import os
from threading import Lock

from src.config import PROJECT_ID_ENV_NAME, PROJECT_ID_ENV_PREFIX


class ProjectRouter:
    def __init__(self, project_ids: list[str]) -> None:
        self.project_ids = project_ids
        self.index = 0
        self.lock = Lock()

    def next_project_id(self) -> str:
        # ---------------------------------------------------------
        # Return project IDs in order for parallel API requests.
        # ---------------------------------------------------------
        if not self.project_ids:
            raise ValueError("Google Cloud project ID is not set")

        with self.lock:
            project_id = self.project_ids[self.index]
            self.index = (self.index + 1) % len(self.project_ids)

        return project_id


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
