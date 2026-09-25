from collections.abc import Sequence

import pytest

import retrieve as retrieve_mod
from retrieve import (
    cross_encode_rerank,
    retrieve,
    retrieve_hybrid,
    retrieve_reranked,
    route_query,
)
from schemas.chunk import Chunk, ScoredChunk

V1 = Chunk(
    chunk_id="gru-minion-handbook-v1:v1.0:section-2",
    document="gru-minion-handbook-v1",
    version="1.0",
    section="2",
    section_title="Banana Service",
    text="Each minion may take two bananas before noon.",
    embedding=[0.0] * 768,
)
V2 = Chunk(
    chunk_id="gru-minion-handbook-v2:v2.0:section-2",
    document="gru-minion-handbook-v2",
    version="2.0",
    section="2",
    section_title="Banana Service",
    text="Each minion may take one bananas before noon.",
    embedding=[0.0] * 768,
)
OTHER = Chunk(
    chunk_id="nefario-lab-safety-v1:v1.0:section-1",
    document="nefario-lab-safety-v1",
    version="1.0",
    section="1",
    section_title="Goggles",
    text="Wear goggles in the lab.",
    embedding=[0.0] * 768,
)
GIRLS = Chunk(
    chunk_id="girls-house-rules-v1:v1.0:section-1",
    document="girls-house-rules-v1",
    version="1.0",
    section="1",
    section_title="Bedtime",
    text="Lights out at 8pm.",
    embedding=[0.0] * 768,
)
HARBOR = Chunk(
    chunk_id="harbor-bike-shop-handbook:v1.0:section-1",
    document="harbor-bike-shop-handbook",
    version="1.0",
    section="1",
    section_title="Hours",
    text="The shop closes at 6pm.",
    embedding=[0.0] * 768,
)

FACT = "How many bananas before noon?"
COMPARE = "How did the bananas change from v1 to v2?"


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

    def __init__(
        self,
        latest: list[ScoredChunk],
        all_hits: list[ScoredChunk],
        chunks: list[Chunk],
    ) -> None:
        self.latest = latest
        self.all_hits = all_hits
        self.chunks = chunks
        self.last_method: str | None = None

    def save_chunks(self, chunks: list[Chunk]) -> None:
        return None

    def load_chunks(self) -> list[Chunk]:
        return self.chunks

    def search(self, query_embedding: list[float], k: int = 3) -> list[ScoredChunk]:
        self.last_method = "search"
        return self.latest[:k]

    def search_all(self, query_embedding: list[float], k: int = 3) -> list[ScoredChunk]:
        self.last_method = "search_all"
        return self.all_hits[:k]


def _store() -> FakeStore:
    return FakeStore(
        latest=[ScoredChunk(chunk=V2, score=0.1)],
        all_hits=[
            ScoredChunk(chunk=V2, score=0.1),
            ScoredChunk(chunk=V1, score=0.2),
        ],
        chunks=[V1, V2, OTHER, GIRLS, HARBOR],
    )


@pytest.mark.parametrize(
    ("query", "scope"),
    [
        (FACT, "latest"),
        ("What time do minions report for the morning duty?", "latest"),
        (COMPARE, "all"),
        ("Compare morning briefing between the two handbook versions.", "all"),
        ("What changed in the old policy?", "all"),
    ],
)
def test_route_query(query: str, scope: str) -> None:
    assert route_query(query) == scope


def test_retrieve_default_scope_uses_search() -> None:
    store = _store()

    hits = retrieve(COMPARE, FakeEmbedder(), store)

    assert store.last_method == "search"
    assert [h.chunk.document for h in hits] == ["gru-minion-handbook-v2"]


def test_retrieve_all_scope_uses_search_all() -> None:
    store = _store()

    hits = retrieve(COMPARE, FakeEmbedder(), store, scope="all")

    assert store.last_method == "search_all"
    assert [h.chunk.document for h in hits] == [
        "gru-minion-handbook-v2",
        "gru-minion-handbook-v1",
    ]


def test_retrieve_hybrid_latest_drops_old_version() -> None:
    store = _store()

    vector, keyword = retrieve_hybrid(FACT, FakeEmbedder(), store)

    assert store.last_method == "search"
    assert [h.chunk.document for h in vector] == ["gru-minion-handbook-v2"]
    assert {h.chunk.document for h in keyword} == {"gru-minion-handbook-v2"}


def test_retrieve_hybrid_compare_query_keeps_both_versions() -> None:
    store = _store()

    vector, keyword = retrieve_hybrid(COMPARE, FakeEmbedder(), store)

    assert store.last_method == "search_all"
    assert {h.chunk.document for h in vector} == {
        "gru-minion-handbook-v1",
        "gru-minion-handbook-v2",
    }
    assert {h.chunk.document for h in keyword} == {
        "gru-minion-handbook-v1",
        "gru-minion-handbook-v2",
    }


def test_retrieve_hybrid_explicit_latest_overrides_compare_query() -> None:
    store = _store()

    _, keyword = retrieve_hybrid(COMPARE, FakeEmbedder(), store, scope="latest")

    assert store.last_method == "search"
    assert {h.chunk.document for h in keyword} == {"gru-minion-handbook-v2"}


def test_cross_encode_rerank_empty_hits() -> None:
    assert cross_encode_rerank("bananas?", [], k=3) == []


def test_cross_encode_rerank_reorders(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEncoder:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            assert pairs[0][0] == COMPARE
            return [0.2, 0.9]

    monkeypatch.setattr("retrieve._get_cross_encoder", lambda: FakeEncoder())
    hits = [
        ScoredChunk(chunk=V1, score=0.4),
        ScoredChunk(chunk=V2, score=0.3),
    ]

    ranked = cross_encode_rerank(COMPARE, hits, k=1)

    assert [h.chunk.document for h in ranked] == ["gru-minion-handbook-v2"]
    assert ranked[0].score == pytest.approx(0.9)


def test_retrieve_reranked_fuses_then_reranks(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEncoder:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.1] * len(pairs)

    monkeypatch.setattr("retrieve._get_cross_encoder", lambda: FakeEncoder())
    store = _store()

    hits = retrieve_reranked(COMPARE, FakeEmbedder(), store, k=2, pool=3)

    assert store.last_method == "search_all"
    assert {h.chunk.document for h in hits} <= {
        "gru-minion-handbook-v1",
        "gru-minion-handbook-v2",
    }
    assert len(hits) <= 2


def test_get_cross_encoder_lazy_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class FakeEncoder:
        def __init__(self, name: str) -> None:
            created.append(name)

    monkeypatch.setattr(retrieve_mod, "_cross_encoder", None)
    monkeypatch.setattr(retrieve_mod, "CrossEncoder", FakeEncoder)

    first = retrieve_mod._get_cross_encoder()
    second = retrieve_mod._get_cross_encoder()

    assert first is second
    assert created == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
