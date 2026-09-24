"""Document chunk models used by indexing and retrieval."""

from pydantic import BaseModel


class Chunk(BaseModel):
    """A text span from a source document, plus its embedding.
    Identifies where the span came from (document, version, section) and
    stores the vector produced by an EmbeddingAdapter.
    """

    chunk_id: str
    document: str
    version: str
    section: str
    section_title: str
    subsection: str = ""
    subsection_title: str = ""
    text: str
    part: int = 0
    page: int | None = None
    embedding: list[float]

class ScoredChunk(BaseModel):
    """A retrieved chunk paired with its similarity score."""

    chunk: Chunk
    score: float
