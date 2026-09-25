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
    text: str = ""
    part: int = 0
    page: int | None = None
    score: float


class RagResponse(BaseModel):
    """Generated answer plus the chunks that were retrieved."""

    query: str
    answer: str
    citation: Citation | None
    retrieved_chunks: list[RetrievedChunkRef]


REFUSE = "The provided policy does not answer this question."


def _is_refuse(answer: str) -> bool:
    return answer.strip().lower() == REFUSE.lower()


def build_response(query: str, answer: str, hits: list[ScoredChunk]) -> RagResponse:
    """Attach a top-hit citation unless the model refused."""
    top = hits[0] if hits and not _is_refuse(answer) else None
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
                text=hit.chunk.text,
                part=hit.chunk.part,
                page=hit.chunk.page,
                score=hit.score,
            )
            for hit in hits
        ],
    )