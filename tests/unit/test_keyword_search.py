from collections.abc import Sequence

import pytest

from retrieve import (
    fuse_rrf,
    latest_chunks,
    retrieve_fused,
    retrieve_hybrid,
    search_keyword,
    tokenize,
)
from schemas.chunk import Chunk, ScoredChunk

RENTALS = Chunk(
    chunk_id="harbor-bike-shop-handbook:v1.0:section-2",
    document="harbor-bike-shop-handbook",
    version="1.0",
    section="2",
    section_title="Rentals",
    text="City bikes rent for $25 a day. Electric bikes rent for $45 a day. A photo ID is required.",
    embedding=[0.0] * 768,
)
HOURS = Chunk(
    chunk_id="harbor-bike-shop-handbook:v1.0:section-1",
    document="harbor-bike-shop-handbook",
    version="1.0",
    section="1",
    section_title="Hours",
    text="The shop is open Tuesday through Saturday from 10am to 6pm. We are closed Sunday and Monday.",
    embedding=[0.0] * 768,
)
RETURNS = Chunk(
    chunk_id="harbor-bike-shop-handbook:v1.0:section-5",
    document="harbor-bike-shop-handbook",
    version="1.0",
    section="5",
    section_title="Returns",
    text="Bring rentals back 15 minutes before close. Late returns are charged for an extra day.",
    embedding=[0.0] * 768,
)

CORPUS = [RETURNS, RENTALS, HOURS]


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("Do I need a photo ID to rent?") == ["photo", "id", "rent"]


def test_search_keyword_ranks_rentals_for_photo_id() -> None:
    hits = search_keyword("Do I need a photo ID to rent?", CORPUS)

    assert hits[0].chunk.section_title == "Rentals"
    assert hits[0].score > 0


def test_search_keyword_ranks_hours_for_sunday() -> None:
    hits = search_keyword("Are you open on Sunday?", CORPUS)

    assert [h.chunk.section_title for h in hits] == ["Hours"]


def test_search_keyword_drops_zero_overlap() -> None:
    hits = search_keyword("Sunday", CORPUS)

    assert all(h.chunk.section_title == "Hours" for h in hits)


def test_search_keyword_empty_query_or_corpus() -> None:
    assert search_keyword("What do I need?", CORPUS) == []
    assert search_keyword("Sunday", []) == []


def test_search_keyword_respects_k() -> None:
    hits = search_keyword("rent day", CORPUS, k=1)

    assert len(hits) == 1


def test_latest_chunks_keeps_highest_version_per_family() -> None:
    v1 = Chunk(
        chunk_id="gru-minion-handbook-v1:v1.0:section-2",
        document="gru-minion-handbook-v1",
        version="1.0",
        section="2",
        section_title="Banana Service",
        text="Each minion may take two bananas before noon.",
        embedding=[0.0] * 768,
    )
    v2 = Chunk(
        chunk_id="gru-minion-handbook-v2:v2.0:section-2",
        document="gru-minion-handbook-v2",
        version="2.0",
        section="2",
        section_title="Banana Service",
        text="Each minion may take one banana before noon.",
        embedding=[0.0] * 768,
    )
    other = Chunk(
        chunk_id="nefario-lab-safety-v1:v1.0:section-1",
        document="nefario-lab-safety-v1",
        version="1.0",
        section="1",
        section_title="Goggles",
        text="Wear goggles in the lab.",
        embedding=[0.0] * 768,
    )

    kept = latest_chunks([v1, v2, other])

    assert {c.document for c in kept} == {
        "gru-minion-handbook-v2",
        "nefario-lab-safety-v1",
    }


class FakeEmbedder:
    provider = "fake"
    model_id = "fake"
    dimension = 768

    def embed_query(self, query: str) -> list[float]:
        return [0.0] * 768

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]


class FakeStore:
    provider = "fake"

    def __init__(self, results: list[ScoredChunk], chunks: list[Chunk]) -> None:
        self.results = results
        self.chunks = chunks

    def save_chunks(self, chunks: list[Chunk]) -> None:
        return None

    def load_chunks(self) -> list[Chunk]:
        return self.chunks

    def search(self, query_embedding: list[float], k: int = 3) -> list[ScoredChunk]:
        return self.results[:k]


def test_retrieve_hybrid_returns_two_lists() -> None:
    vector_hits = [ScoredChunk(chunk=RETURNS, score=0.1)]
    store = FakeStore(vector_hits, CORPUS)

    vector, keyword = retrieve_hybrid(
        "Are you open on Sunday?",
        FakeEmbedder(),
        store,
        k=3,
    )

    assert [h.chunk.section_title for h in vector] == ["Returns"]
    assert [h.chunk.section_title for h in keyword] == ["Hours"]


def _hit(chunk: Chunk) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=0.0)


def test_fuse_rrf_keeps_shared_top_hit_first() -> None:
    vector = [_hit(RENTALS), _hit(RETURNS)]
    keyword = [_hit(RENTALS), _hit(HOURS)]

    fused = fuse_rrf(vector, keyword)

    assert fused[0].chunk.section_title == "Rentals"
    assert fused[0].score == pytest.approx(0.7 / 61 + 0.3 / 61)


def test_fuse_rrf_weights_change_runner_up() -> None:
    vector = [_hit(RENTALS), _hit(RETURNS)]
    keyword = [_hit(RENTALS), _hit(HOURS)]

    favor_vector = fuse_rrf(vector, keyword, vector_weight=0.7, keyword_weight=0.3)
    favor_keyword = fuse_rrf(vector, keyword, vector_weight=0.3, keyword_weight=0.7)

    assert [h.chunk.section_title for h in favor_vector] == [
        "Rentals",
        "Returns",
        "Hours",
    ]
    assert [h.chunk.section_title for h in favor_keyword] == [
        "Rentals",
        "Hours",
        "Returns",
    ]


def test_fuse_rrf_empty_side_keeps_other_ranking() -> None:
    vector = [_hit(RETURNS), _hit(RENTALS), _hit(HOURS)]

    fused = fuse_rrf(vector, [], k=2)

    assert [h.chunk.section_title for h in fused] == ["Returns", "Rentals"]


def test_fuse_rrf_respects_k() -> None:
    vector = [_hit(RETURNS), _hit(RENTALS), _hit(HOURS)]
    keyword = [_hit(HOURS)]

    assert len(fuse_rrf(vector, keyword, k=1)) == 1


def test_retrieve_fused_uses_pool_then_rrf() -> None:
    vector_hits = [_hit(RETURNS), _hit(RENTALS), _hit(HOURS)]
    store = FakeStore(vector_hits, CORPUS)

    fused = retrieve_fused(
        "Are you open on Sunday?",
        FakeEmbedder(),
        store,
        k=2,
        pool=10,
    )

    # Hours is vector #3 + BM25 #1, so RRF puts it first.
    assert [h.chunk.section_title for h in fused] == ["Hours", "Returns"]