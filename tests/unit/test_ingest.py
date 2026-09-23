from collections.abc import Sequence
from pathlib import Path

import pytest

from ingest import chunk, ingest
from models.chunk import Chunk

CORPUS = Path("data/corpus-tiny/harbor-bike-shop-handbook.md")


class FakeEmbedder:
    provider = "fake"
    model_id = "fake"
    dimension = 768

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors

    def embed_query(self, query: str) -> list[float]:
        return [0.0] * self.dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.vectors is not None:
            return self.vectors
        return [[float(i)] + [0.0] * 767 for i, _ in enumerate(texts)]


def test_chunk_handbook_one_section_each() -> None:
    records = chunk(CORPUS, "1.0")

    assert [r["section_title"] for r in records] == [
        "Hours",
        "Rentals",
    ]
    assert records[0]["chunk_id"] == "harbor-bike-shop-handbook:v1.0:section-1"
    assert records[1]["chunk_id"] == "harbor-bike-shop-handbook:v1.0:section-2"
    assert records[0]["document"] == "harbor-bike-shop-handbook"
    assert records[0]["version"] == "1.0"
    assert "Tuesday" in records[0]["text"]
    assert all(r["section"] == str(i) for i, r in enumerate(records, start=1))


def test_chunk_skips_title() -> None:
    records = chunk(CORPUS, "1.0")

    assert not any("Handbook" in r["section_title"] for r in records)
    assert not any("Handbook" in r["text"] for r in records)


def test_ingest_builds_chunks_with_embeddings() -> None:
    chunks = ingest(CORPUS, FakeEmbedder(), "1.0")

    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].chunk_id == "harbor-bike-shop-handbook:v1.0:section-1"
    assert len(chunks[0].embedding) == 768
    assert chunks[1].section_title == "Rentals"


def test_chunk_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        chunk(tmp_path / "missing.md", "1.0")


def test_chunk_no_sections_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("# Title only\n\nPreamble with no sections.\n")

    assert chunk(path, "1.0") == []


def test_chunk_skips_empty_section(tmp_path: Path) -> None:
    path = tmp_path / "gaps.md"
    path.write_text(
        "# Doc\n\n## Hours\nOpen daily.\n\n## Empty\n\n## Rentals\nBring an ID.\n"
    )

    records = chunk(path, "1")

    assert [r["section_title"] for r in records] == ["Hours", "Rentals"]
    assert records[1]["chunk_id"] == "gaps:v1:section-2"


def test_ingest_raises_when_embedder_count_mismatches() -> None:
    embedder = FakeEmbedder(vectors=[[0.0] * 768])

    with pytest.raises(ValueError, match="shorter than argument 1"):
        ingest(CORPUS, embedder, "1.0")
