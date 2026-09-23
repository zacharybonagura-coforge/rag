"""Protocol for turning text into embedding vectors."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingAdapter(Protocol):
    """Produces fixed-length vectors for queries and documents."""

    provider: str
    model_id: str
    dimension: int

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query. Returns a vector of length ``dimension``."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input."""
        ...