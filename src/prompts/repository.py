"""Load versioned markdown prompt templates and render them."""

from pathlib import Path

from schemas.chunk import ScoredChunk

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
            _hit_block(hit.chunk) for hit in hits
        )
        return template.format(context=context, query=query)
        
def _hit_block(chunk) -> str:
    heading = f"{chunk.document} §{chunk.section} {chunk.section_title}"
    if chunk.subsection_title:
        heading += f" / {chunk.subsection_title}"
    return f"{heading}\n{chunk.text}"
