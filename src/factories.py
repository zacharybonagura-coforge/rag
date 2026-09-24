"""Build concrete adapters from Settings."""

from config import Settings
from embeddings.base import EmbeddingAdapter
from embeddings.sentence_transformer import SentenceTransformerEmbeddingAdapter
from enums import EmbeddingProvider, ModelProvider, StoreProvider
from generation.base import ModelAdapter
from generation.ollama import OllamaAdapter
from store.base import VectorStoreAdapter
from store.pgvector import PgVectorStoreAdapter


def build_embedder(settings: Settings) -> EmbeddingAdapter:
    """Return the embedding adapter selected by ``settings``."""
    if settings.embedding_provider is EmbeddingProvider.SENTENCE_TRANSFORMERS:
        embedder = SentenceTransformerEmbeddingAdapter(settings.embedding_model)
        if embedder.dimension != settings.embedding_dimension:
            raise ValueError(
                f"embedder is {embedder.dimension}-d; "
                f"config expects {settings.embedding_dimension}"
            )
        return embedder
    raise ValueError(f"unknown embedding provider: {settings.embedding_provider}")


def build_generator(settings: Settings) -> ModelAdapter:
    """Return the generation adapter selected by ``settings``."""
    if settings.generation_provider is ModelProvider.OLLAMA:
        return OllamaAdapter(
            model_id=settings.generation_model,
            host=settings.ollama_host,
        )
    raise ValueError(f"unknown generation provider: {settings.generation_provider}")


def build_store(settings: Settings) -> VectorStoreAdapter:
    """Return the vector store selected by ``settings``."""
    if settings.store_provider is StoreProvider.PGVECTOR:
        return PgVectorStoreAdapter(settings.database_url)
    raise ValueError(f"unknown store provider: {settings.store_provider}")