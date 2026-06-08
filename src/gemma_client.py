from collections.abc import Callable
from typing import Literal

from src import digital_ocean_client, vertex_client
from src.vertex_client import GemmaResponse


GenerationProvider = Literal["vertex", "digital-ocean"]


def ask_gemma4(
    provider: GenerationProvider,
    prompt: str,
    system_prompt: str,
    on_retry: Callable[[Exception], None] | None = None,
    project_id: str | None = None,
) -> GemmaResponse:
    # ---------------------------------------------------------
    # Hide provider-specific generation details behind one API.
    # ---------------------------------------------------------
    if provider == "vertex":
        return vertex_client.ask_gemma4(
            prompt=prompt,
            system_prompt=system_prompt,
            on_retry=on_retry,
            project_id=project_id,
        )

    return digital_ocean_client.ask_gemma4(
        prompt=prompt,
        system_prompt=system_prompt,
        on_retry=on_retry,
        project_id=project_id,
    )
