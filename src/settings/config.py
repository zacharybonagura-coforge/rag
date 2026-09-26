"""Runtime settings for the mini-corpus runner."""

import os
from dataclasses import dataclass
from pathlib import Path

from settings.enums import EmbeddingProvider, ModelProvider, StoreProvider

ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Paths and model hosts read from the environment, with lab defaults."""

    embedding_provider: EmbeddingProvider
    embedding_model: str
    embedding_dimension: int
    generation_provider: ModelProvider
    generation_model: str
    store_provider: StoreProvider
    ollama_host: str
    database_url: str
    corpus_path: Path
    runs_dir: Path
    retrieve_k: int
    retrieve_pool: int
    chunk_max_chars: int
    chunk_overlap: int
    hybrid_retrieve: bool
    cross_encode: bool
    prompt_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from env vars, falling back to local lab defaults."""
        return cls(
            embedding_provider=EmbeddingProvider(
                os.environ.get(
                    "EMBEDDING_PROVIDER",
                    EmbeddingProvider.SENTENCE_TRANSFORMERS,
                )
            ),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-mpnet-base-v2",
            ),
            embedding_dimension=int(os.environ.get("EMBEDDING_DIMENSION", "768")),
            generation_provider=ModelProvider(
                os.environ.get("GENERATION_PROVIDER", ModelProvider.OLLAMA)
            ),
            generation_model=os.environ.get("GENERATION_MODEL", "mistral:7b"),
            store_provider=StoreProvider(
                os.environ.get("STORE_PROVIDER", StoreProvider.PGVECTOR)
            ),
            ollama_host=os.environ.get(
                "OLLAMA_HOST", "http://host.docker.internal:11434"
            ),
            database_url=os.environ["DATABASE_URL"],
            corpus_path=Path(
                os.environ.get(
                    "CORPUS_PATH",
                    str(ROOT / "data/corpus"),
                )
            ),
            runs_dir=Path(os.environ.get("RUNS_DIR", str(ROOT / "runs"))),
            retrieve_k=int(os.environ.get("RETRIEVE_K", "3")),
            retrieve_pool=int(os.environ.get("RETRIEVE_POOL", "10")),
            chunk_max_chars=int(os.environ.get("CHUNK_MAX_CHARS", "250")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "80")),
            hybrid_retrieve=_env_bool("HYBRID_RETRIEVE", True),
            cross_encode=_env_bool("CROSS_ENCODE", True),
            prompt_name=os.environ.get("PROMPT_NAME", "policy.v3")
        )
