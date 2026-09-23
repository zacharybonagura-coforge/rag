"""Split a markdown handbook into one chunk per ## section."""

from pathlib import Path

from embeddings.base import EmbeddingAdapter
from models.chunk import Chunk


def chunk(path: Path, version: str) -> list[dict[str, str]]:
    """Read ``path`` and return one record per ``##`` section.

    ``chunk_id`` is ``{slug}:v{version}:section-{n}``.
    """
    slug = path.stem
    sections: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    body: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            if current is not None:
                current["text"] = "\n".join(body).strip()
                if current["text"]:
                    sections.append(current)
            current = {
                "section": str(len(sections) + 1),
                "section_title": line.removeprefix("## ").strip(),
            }
            body = []
            continue
        if current is not None:
            body.append(line)

    if current is not None:
        current["text"] = "\n".join(body).strip()
        if current["text"]:
            sections.append(current)

    for section in sections:
        section["document"] = slug
        section["version"] = version
        section["chunk_id"] = (
            f"{slug}:v{version}:section-{section['section']}"
        )
    return sections


def ingest(
    path: Path,
    embedder: EmbeddingAdapter,
    version: str,
) -> list[Chunk]:
    """Chunk ``path`` and embed each section body."""
    records = chunk(path, version)
    vectors = embedder.embed_documents([r["text"] for r in records])
    return [
        Chunk(**record, embedding=vector)
        for record, vector in zip(records, vectors, strict=True)
    ]
