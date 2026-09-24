"""Response models for a single RAG run."""

from pydantic import BaseModel

from schemas.chunk import ScoredChunk


class Citation(BaseModel):
    """Source location of the chunk used to ground the answer."""

    document: str
    version: str
    section: str
    section_title: str
    subsection: str = ""
    subsection_title: str = ""
    part: int = 0
    page: int | None = None


class RetrievedChunkRef(BaseModel):
    """One retrieved neighbor, identified by section and cosine distance."""

    document: str
    version: str
    section: str
    section_title: str
    subsection: str = ""
    subsection_title: str = ""
    part: int = 0
    page: int | None = None
    distance: float


class RagResponse(BaseModel):
    """Generated answer plus the chunks that were retrieved."""

    query: str
    answer: str
    citation: Citation | None
    retrieved_chunks: list[RetrievedChunkRef]


def build_response(query: str, answer: str, hits: list[ScoredChunk]) -> RagResponse:
    """Attach a top-hit citation and retrieved refs to ``answer``."""
    top = hits[0] if hits else None
    return RagResponse(
        query=query,
        answer=answer,
        citation=(
            Citation(
                document=top.chunk.document,
                version=top.chunk.version,
                section=top.chunk.section,
                section_title=top.chunk.section_title,
                subsection=top.chunk.subsection,
                subsection_title=top.chunk.subsection_title,
                part=top.chunk.part,
                page=top.chunk.page,
            )
            if top
            else None
        ),
        retrieved_chunks=[
            RetrievedChunkRef(
                document=hit.chunk.document,
                version=hit.chunk.version,
                section=hit.chunk.section,
                section_title=hit.chunk.section_title,
                subsection=hit.chunk.subsection,
                subsection_title=hit.chunk.subsection_title,
                part=hit.chunk.part,
                page=hit.chunk.page,
                distance=hit.score,
            )
            for hit in hits
        ],
    )