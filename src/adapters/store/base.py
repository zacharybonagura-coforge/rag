"""Vector store adapter protocol for persisting and searching chunks."""

from typing import Protocol

from schemas.chunk import Chunk, ScoredChunk


class VectorStoreAdapter(Protocol):
    """Interface for saving chunks and retrieving nearest neighbors.

    Implementations persist embedded chunks and rank them against a query
    vector. Callers can swap backends as long as they satisfy this protocol.
    """

    provider: str

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Persist a batch of embedded chunks."""
        ...

    def load_chunks(self) -> list[Chunk]:
        """Return all stored chunks."""
        ...

    def search(
        self,
        query_embedding: list[float],
        k: int = 3,
    ) -> list[ScoredChunk]:
        """Return the ``k`` chunks nearest to ``query_embedding``."""
        ...