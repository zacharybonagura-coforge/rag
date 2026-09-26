"""Fixed names for adapters and prompt templates."""

from enum import StrEnum


class EmbeddingProvider(StrEnum):
    SENTENCE_TRANSFORMERS = "sentence-transformers"


class StoreProvider(StrEnum):
    PGVECTOR = "pgvector"


class ModelProvider(StrEnum):
    OLLAMA = "ollama"
