import unicodedata
from dataclasses import dataclass
from typing import Any, Self

from src.vertex_client import GemmaResponse


@dataclass(frozen=True)
class RewriteRecord:
    offset: int
    source_id: str
    url: str
    source_text: str
    prompt_type: str
    prompt: str
    rewrite: str
    output_chars: int
    output_tokens: int | None

    @classmethod
    def from_generation(
        cls,
        offset: int,
        item: dict[str, Any],
        text: str,
        prompt_type: str,
        prompt: str,
        response: GemmaResponse,
    ) -> Self:
        # ---------------------------------------------------------
        # Normalize generated text for stable training data.
        # ---------------------------------------------------------
        rewrite = unicodedata.normalize("NFKC", response.text)

        return cls(
            offset=offset,
            source_id=str(item["id"]),
            url=str(item["url"]),
            source_text=text,
            prompt_type=prompt_type,
            prompt=prompt,
            rewrite=rewrite,
            output_chars=len(rewrite),
            output_tokens=response.output_tokens,
        )
