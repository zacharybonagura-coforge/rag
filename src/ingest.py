"""Split a markdown handbook into one chunk per ## section."""

import re
from pathlib import Path

from embeddings.base import EmbeddingAdapter
from models.chunk import Chunk

VERSION_RE = re.compile(r"\(Version\s+([^)]+)\)", re.IGNORECASE)


def _version_from_title(title: str) -> str:
    match = VERSION_RE.search(title)
    if match is None:
        raise ValueError(f"title has no version: {title!r}")
    return match.group(1).strip()


def chunk(path: Path) -> list[dict[str, str]]:
    """Read ``path`` and return one record per ``##`` section.

    Version comes from the H1, e.g. ``(Version 1.0)``.
    ``chunk_id`` is ``{slug}:v{version}:section-{n}``.
    """
    slug = path.stem
    version: str | None = None
    sections: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    body: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            version = _version_from_title(line.removeprefix("# ").strip())
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

    if sections and version is None:
        raise ValueError(f"{path} has sections but no version in the H1")

    for section in sections:
        section["document"] = slug
        section["version"] = version or ""
        section["chunk_id"] = (
            f"{slug}:v{version}:section-{section['section']}"
        )
    return sections


def ingest(root: Path, embedder: EmbeddingAdapter) -> list[Chunk]:
    """Chunk every ``*.md`` under ``root`` and embed each section body."""
    paths = sorted(root.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no markdown files in {root}")
    records = [record for path in paths for record in chunk(path)]
    vectors = embedder.embed_documents([r["text"] for r in records])
    return [
        Chunk(**record, embedding=vector)
        for record, vector in zip(records, vectors, strict=True)
    ]
