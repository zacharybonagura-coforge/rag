"""Sentence-Transformers implementation of EmbeddingAdapter."""

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingAdapter:
    """Local embedding adapter backed by a SentenceTransformer model."""

    provider = "sentence-transformers"

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        """Load the model and verify it produces 768-d embeddings."""
        
        self.model_id = model_id
        self._model = SentenceTransformer(model_id)
        dimension = self._model.get_embedding_dimension()
        if dimension is None or dimension != 768:
            raise ValueError(f"{model_id} is {dimension}-d; expected 768")
        self.dimension = dimension

    def embed_query(self, query: str) -> list[float]:
        """Encode one query and return an L2-normalized 768-d vector."""
        
        return self._model.encode(query, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of texts and return L2-normalized 768-d vectors."""
        
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()
