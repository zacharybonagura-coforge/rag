"""Split handbooks into section and subsection chunks."""

import re
from pathlib import Path
from typing import Any, cast

from adapters.embeddings.base import EmbeddingAdapter
from blocks import load
from schemas.chunk import Chunk

# Match "(Version <value>)" in a title and capture the value, e.g. "(Version 3.2)" -> "3.2".
# \( literal "(", Version case-insensitive, \s+ one or more spaces,
# ([^)]+) capture one or more non-")" characters, \) literal ")".
VERSION_RE = re.compile(r"\(Version\s+([^)]+)\)", re.IGNORECASE)


def _version_from_title(title: str) -> str:
    match = VERSION_RE.search(title)
    if match is None:
        raise ValueError(f"title has no version: {title!r}")
    return match.group(1).strip()


def _soft_end(window: str) -> int:
    """Cut near the end of ``window`` on a line, sentence, or space."""
    region_start = len(window) * 4 // 5
    region = window[region_start:]
    for sep in ("\n\n", "\n", ". ", " "):
        idx = region.rfind(sep)
        if idx != -1:
            return region_start + idx + len(sep)
    return len(window)


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split ``text`` into windows of at most ``max_chars`` characters."""
    if len(text) <= max_chars:
        return [text]
    overlap = min(overlap, max_chars - 1)
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            end = start + _soft_end(text[start:end])
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        else:
            snapped = text.rfind("\n", start, next_start + 1)
            if snapped == -1:
                snapped = text.rfind(" ", start, next_start + 1)
            if snapped != -1:
                next_start = snapped + 1
        start = next_start
    return parts


def chunk(
    path: Path,
    *,
    max_chars: int = 500,
    overlap: int = 100,
) -> list[dict[str, str | int | bool | None]]:
    """Group ``path`` into one record per section, or per subsection when present.

    A section with no ``###`` headings stays one record:
    ``{slug}:v{version}:section-{n}``.
    Intro text before the first ``###`` is ``sub-0``. Each ``###`` is ``sub-1``,
    ``sub-2``, and so on. A unit longer than ``max_chars`` is split into
    overlapping parts, and those ids gain ``:part-{p}``.
    """
    slug = path.stem
    version: str | None = None
    records: list[dict[str, str | int | bool | None]] = []
    section_title = ""
    section_no: str | None = None
    next_section = 1
    sub_no = 0
    sub_title = ""
    in_section = False
    in_subsection = False
    body: list[str] = []
    page: int | None = None

    def emit(as_subsection: bool) -> None:
        nonlocal body, page, section_no, next_section
        text = "\n".join(body).strip()
        unit_page, body, page = page, [], None
        if not text:
            return
        if section_no is None:
            section_no = str(next_section)
            next_section += 1
        parts = _windows(text, max_chars, overlap)
        split = len(parts) > 1
        for index, part in enumerate(parts):
            records.append(
                {
                    "section": section_no,
                    "section_title": section_title,
                    "subsection": str(sub_no) if as_subsection else "",
                    "subsection_title": sub_title if as_subsection else "",
                    "part": index if split else 0,
                    "page": unit_page,
                    "text": part,
                    "split": split,
                }
            )

    for block in load(path):
        if block.kind == "title":
            version = _version_from_title(block.text)
        elif block.kind == "section":
            if in_section:
                emit(in_subsection)
            section_title, section_no = block.text, None
            sub_no, sub_title = 0, ""
            in_section, in_subsection = True, False
        elif block.kind == "subsection" and in_section:
            emit(True)
            in_subsection = True
            sub_no += 1
            sub_title = block.text
        elif block.kind == "body" and in_section:
            if page is None and block.page is not None:
                page = block.page
            body.append(block.text)

    if in_section:
        emit(in_subsection)

    if records and version is None:
        raise ValueError(f"{path} has sections but no version in the H1")

    for record in records:
        record["document"] = slug
        record["version"] = version or ""
        base = f"{slug}:v{version}:section-{record['section']}"
        if record["subsection"]:
            base += f":sub-{record['subsection']}"
        if record.pop("split"):
            base += f":part-{record['part']}"
        record["chunk_id"] = base
    return records


def ingest(
    root: Path,
    embedder: EmbeddingAdapter,
    *,
    max_chars: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """Chunk every ``*.md`` under ``root`` and embed each chunk body."""
    paths = sorted(root.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no markdown files in {root}")
    records = [
        record
        for path in paths
        for record in chunk(path, max_chars=max_chars, overlap=overlap)
    ]
    vectors = embedder.embed_documents([str(r["text"]) for r in records])
    return [
        Chunk(**cast(dict[str, Any], record), embedding=vector)
        for record, vector in zip(records, vectors, strict=True)
    ]
