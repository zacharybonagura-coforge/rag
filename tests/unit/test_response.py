from schemas.chunk import Chunk, ScoredChunk
from schemas.response import REFUSE, build_response

HOURS = Chunk(
    chunk_id="harbor-bike-shop-handbook:v1.0:section-1",
    document="harbor-bike-shop-handbook",
    version="1.0",
    section="1",
    section_title="Hours",
    text="Closed Sunday.",
    embedding=[],
)


def test_build_response_omits_citation_on_refuse() -> None:
    hits = [ScoredChunk(chunk=HOURS, score=0.1)]
    response = build_response("hours?", REFUSE, hits)
    assert response.citation is None
    assert response.retrieved_chunks[0].document == "harbor-bike-shop-handbook"


def test_build_response_omits_citation_when_no_hits() -> None:
    response = build_response("hours?", "The shop closes at 6pm.", [])
    assert response.citation is None
    assert response.retrieved_chunks == []
