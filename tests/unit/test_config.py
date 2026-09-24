from pathlib import Path

import pytest

from config import ROOT, Settings
from enums import EmbeddingProvider, ModelProvider, StoreProvider

SETTINGS_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "GENERATION_PROVIDER",
    "GENERATION_MODEL",
    "STORE_PROVIDER",
    "OLLAMA_HOST",
    "DATABASE_URL",
    "CORPUS_PATH",
    "RUNS_DIR",
    "RETRIEVE_K",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in SETTINGS_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_from_env_uses_lab_defaults(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

    settings = Settings.from_env()

    assert settings.embedding_provider is EmbeddingProvider.SENTENCE_TRANSFORMERS
    assert settings.embedding_model == "sentence-transformers/all-mpnet-base-v2"
    assert settings.embedding_dimension == 768
    assert settings.generation_provider is ModelProvider.OLLAMA
    assert settings.generation_model == "mistral:7b"
    assert settings.store_provider is StoreProvider.PGVECTOR
    assert settings.ollama_host == "http://host.docker.internal:11434"
    assert settings.database_url == "postgresql://rag:rag@localhost:5432/rag"
    assert settings.corpus_path == ROOT / "data/corpus"
    assert settings.runs_dir == ROOT / "runs"
    assert settings.retrieve_k == 3


def test_from_env_reads_overrides(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("DATABASE_URL", "postgresql://lab:lab@db:5432/lab")
    clean_env.setenv("EMBEDDING_MODEL", "other-mpnet")
    clean_env.setenv("EMBEDDING_DIMENSION", "512")
    clean_env.setenv("GENERATION_MODEL", "llama3:8b")
    clean_env.setenv("OLLAMA_HOST", "http://localhost:11434")
    clean_env.setenv("CORPUS_PATH", "data/corpus-tiny")
    clean_env.setenv("RUNS_DIR", "/tmp/runs")
    clean_env.setenv("RETRIEVE_K", "1")

    settings = Settings.from_env()

    assert settings.embedding_model == "other-mpnet"
    assert settings.embedding_dimension == 512
    assert settings.generation_model == "llama3:8b"
    assert settings.ollama_host == "http://localhost:11434"
    assert settings.corpus_path == Path("data/corpus-tiny")
    assert settings.runs_dir == Path("/tmp/runs")
    assert settings.retrieve_k == 1


def test_from_env_requires_database_url(clean_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(KeyError, match="DATABASE_URL"):
        Settings.from_env()


def test_from_env_rejects_unknown_provider(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
    clean_env.setenv("STORE_PROVIDER", "chroma")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_from_env_rejects_non_int_dimension(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
    clean_env.setenv("EMBEDDING_DIMENSION", "wide")

    with pytest.raises(ValueError):
        Settings.from_env()
