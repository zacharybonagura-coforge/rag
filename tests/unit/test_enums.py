import pytest

from enums import EmbeddingProvider, ModelProvider, StoreProvider


def test_provider_values_match_env_strings() -> None:
    assert EmbeddingProvider.SENTENCE_TRANSFORMERS == "sentence-transformers"
    assert ModelProvider.OLLAMA == "ollama"
    assert StoreProvider.PGVECTOR == "pgvector"


def test_providers_accept_their_string() -> None:
    assert EmbeddingProvider("sentence-transformers") is EmbeddingProvider.SENTENCE_TRANSFORMERS
    assert ModelProvider("ollama") is ModelProvider.OLLAMA
    assert StoreProvider("pgvector") is StoreProvider.PGVECTOR


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError):
        EmbeddingProvider("ollama")
    with pytest.raises(ValueError):
        ModelProvider("openai")
    with pytest.raises(ValueError):
        StoreProvider("chroma")
