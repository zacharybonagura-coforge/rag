from collections.abc import Sequence

import pytest

from retrieve import retrieve
from schemas.chunk import Chunk, ScoredChunk

RETURNS = Chunk(
    chunk_id="harbor-bike-shop-handbook:v1.0:section-5",
    document="harbor-bike-shop-handbook",
    version="1.0",
    section="5",
    section_title="Returns",
    text="Bring rentals back 15 minutes before close. Late returns are charged for an extra day.",
    embedding=[1.0] + [0.0] * 767,
)
RENTALS = Chunk(
    chunk_id="harbor-bike-shop-handbook:v1.0:section-2",
    document="harbor-bike-shop-handbook",
    version="1.0",
    section="2",
    section_title="Rentals",
    text="City bikes rent for $25 a day. Electric bikes rent for $45 a day. A photo ID is required.",
    embedding=[0.0, 1.0] + [0.0] * 766,
)
HOURS = Chunk(
    chunk_id="harbor-bike-shop-handbook:v1.0:section-1",
    document="harbor-bike-shop-handbook",
    version="1.0",
    section="1",
    section_title="Hours",
    text="The shop is open Tuesday through Saturday from 10am to 6pm. We are closed Sunday and Monday.",
    embedding=[0.0, 0.0, 1.0] + [0.0] * 765,
)


class FakeEmbedder:
    provider = "fake"
    model_id = "fake"
    dimension = 768

    def __init__(self) -> None:
        self.last_query: str | None = None

    def embed_query(self, query: str) -> list[float]:
        self.last_query = query
        return [0.5] + [0.0] * 767

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]


class FakeStore:
    provider = "fake"

    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.last_embedding: list[float] | None = None
        self.last_k: int | None = None

    def save_chunks(self, chunks: list[Chunk]) -> None:
        return None

    def load_chunks(self) -> list[Chunk]:
        return []

    def search(self, query_embedding: list[float], k: int = 3) -> list[ScoredChunk]:
        self.last_embedding = query_embedding
        self.last_k = k
        return self.results[:k]
    
    def search_all(self, query_embedding: list[float], k: int = 3) -> list[ScoredChunk]:
        self.last_embedding = query_embedding
        self.last_k = k
        return self.results[:k]


def test_retrieve_embeds_query_and_passes_k() -> None:
    embedder = FakeEmbedder()
    store = FakeStore(
        [
            ScoredChunk(chunk=RETURNS, score=0.1),
            ScoredChunk(chunk=RENTALS, score=0.4),
            ScoredChunk(chunk=HOURS, score=0.8),
        ]
    )
    query = "What if I return the bike late?"

    result = retrieve(query, embedder, store, k=2)

    assert embedder.last_query == query
    assert store.last_embedding == [0.5] + [0.0] * 767
    assert store.last_k == 2
    assert [h.chunk.section_title for h in result] == ["Returns", "Rentals"]


def test_retrieve_defaults_to_k_3() -> None:
    store = FakeStore([ScoredChunk(chunk=HOURS, score=0.2)])

    retrieve("What time does the shop close?", FakeEmbedder(), store)

    assert store.last_k == 3


def test_retrieve_returns_empty_when_store_empty() -> None:
    assert retrieve("Do you rent helmets?", FakeEmbedder(), FakeStore([])) == []


def test_retrieve_propagates_store_error() -> None:
    class BoomStore(FakeStore):
        def search(self, query_embedding: list[float], k: int = 3) -> list[ScoredChunk]:
            raise RuntimeError("search failed")

    with pytest.raises(RuntimeError, match="search failed"):
        retrieve("How much is an electric bike?", FakeEmbedder(), BoomStore([]))
