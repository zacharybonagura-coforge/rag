"""Retrieve nearest chunks for a natural-language query."""

from embeddings.base import EmbeddingAdapter
from models.chunk import ScoredChunk
from store.base import VectorStoreAdapter


def retrieve(
    query: str,
    embedder: EmbeddingAdapter,
    store: VectorStoreAdapter,
    k: int = 3,
) -> list[ScoredChunk]:
    """Embed ``query`` and return the ``k`` nearest stored chunks."""
    query_embedding = embedder.embed_query(query)
    return store.search(query_embedding, k)
