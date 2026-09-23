"""Response models for a single RAG run."""

from pydantic import BaseModel

from models.chunk import ScoredChunk


class Citation(BaseModel):
    """Source location of the chunk used to ground the answer."""

    document: str
    version: str
    section: str
    section_title: str


class RetrievedChunkRef(BaseModel):
    """One retrieved neighbor, identified by section and cosine distance."""

    document: str
    version: str
    section: str
    section_title: str
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
                distance=hit.score,
            )
            for hit in hits
        ],
    )