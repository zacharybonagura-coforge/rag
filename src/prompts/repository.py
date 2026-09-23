"""Load versioned markdown prompt templates and render them."""

from pathlib import Path

from models.chunk import ScoredChunk

TEMPLATES = Path(__file__).resolve().parent / "templates"


class PromptRepository:
    """Reads ``templates/{name}.md`` and fills ``{context}`` / ``{query}``."""

    def __init__(self, root: Path = TEMPLATES) -> None:
        self._root = root

    def render(self, name: str, query: str, hits: list[ScoredChunk]) -> str:
        """Load ``name`` and fill it."""
        path = self._root / f"{name}.md"
        template = path.read_text(encoding="utf-8")
        context = "\n\n".join(
            f"{hit.chunk.section_title}\n{hit.chunk.text}" for hit in hits
        )
        return template.format(context=context, query=query)
