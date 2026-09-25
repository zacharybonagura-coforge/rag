from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import pymupdf4llm
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

BlockKind = Literal["title", "section", "subsection", "body"]

@dataclass(frozen=True)
class Block:
    text: str
    kind: BlockKind
    page: int | None = None

_MD_HEADINGS: tuple[tuple[str, BlockKind], ...] = (
    ("### ", "subsection"),
    ("## ", "section"),
    ("# ", "title"),
)
_TABLE_SEP = re.compile(r"^:?-{3,}:?$")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


def _blocks_from_markdown_lines(
    lines: list[str], *, page: int | None = None
) -> list[Block]:
    blocks: list[Block] = []
    in_fence = False
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            blocks.append(Block(text=line, kind="body", page=page))
            continue
        if not in_fence:
            for prefix, heading_kind in _MD_HEADINGS:
                if line.startswith(prefix):
                    blocks.append(Block(
                        text=line.removeprefix(prefix).strip(),
                        kind=heading_kind,
                        page=page,
                    ))
                    break
            else:
                text = _plain_row(line)
                if text is None:
                    continue
                blocks.append(Block(text=text, kind="body", page=page))
        else:
            blocks.append(Block(text=line, kind="body", page=page))
    return blocks


def _plain_row(line: str) -> str | None:
    text = _MD_IMAGE.sub("", line).strip()
    if not text.startswith("|"):
        return text or None
    cells = [" ".join(part.split()) for part in text.strip("|").split("|")]
    if not cells or all(_TABLE_SEP.fullmatch(cell) for cell in cells):
        return None
    return " | ".join(cells) if any(cells) else None


def _table_rows(table: Table) -> list[Block]:
    """One body block per row, cells joined with `` | ``."""
    blocks: list[Block] = []
    for row in table.rows:
        cells = [" ".join(cell.text.split()) for cell in row.cells]
        if any(cells):
            blocks.append(Block(text=" | ".join(cells), kind="body"))
    return blocks


def _load_markdown(path: Path) -> list[Block]:
    """Turn markdown lines into blocks. ``page`` is always ``None``."""
    return _blocks_from_markdown_lines(
        path.read_text(encoding="utf-8").splitlines()
    )


def _page_number(meta: dict[str, object]) -> int | None:
    raw = meta.get("page_number", meta.get("page"))
    if isinstance(raw, int):
        return raw
    return None


def _load_pdf(path: Path) -> list[Block]:
    """Turn a PDF into markdown with pymupdf4llm, then into blocks."""
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, ignore_images=True)
    if not isinstance(pages, list):
        raise TypeError("expected page_chunks list from pymupdf4llm")
    blocks: list[Block] = []
    for page in pages:
        meta = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
        text = str(page.get("text") or "")
        blocks.extend(
            _blocks_from_markdown_lines(text.splitlines(), page=_page_number(meta))
        )
    return blocks


_DOCX_HEADINGS: dict[str, BlockKind] = {
    "Heading 1": "title",
    "Title": "title",
    "Heading 2": "section",
    "Heading 3": "subsection",
}


def _iter_docx_items(document: DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in document order."""
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)

def _load_docx(path: Path) -> list[Block]:
    """Turn Word paragraphs and tables into blocks. ``page`` is always ``None``."""
    blocks: list[Block] = []
    for item in _iter_docx_items(Document(str(path))):
        if isinstance(item, Table):
            blocks.extend(_table_rows(item))
            continue
        style = item.style.name if item.style is not None else ""
        kind = _DOCX_HEADINGS.get(style, "body")
        text = item.text.strip() if kind != "body" else item.text
        blocks.append(Block(text=text, kind=kind))
    return blocks





_LOADERS = {
    ".md": _load_markdown,
    ".docx": _load_docx,
    ".pdf": _load_pdf,
}


def load(path: Path) -> list[Block]:
    """Read ``path`` into blocks. Supports ``.md`` and ``.docx``."""
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"unsupported file type: {path.suffix}")
    return loader(path)
