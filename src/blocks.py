from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

def _load_markdown(path: Path) -> list[Block]:
    """Turn markdown lines into blocks. ``page`` is always ``None``."""
    blocks: list[Block] = []
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        kind: BlockKind = "body"
        text = line
        if not in_fence:
            for prefix, heading_kind in _MD_HEADINGS:
                if line.startswith(prefix):
                    kind = heading_kind
                    text = line.removeprefix(prefix).strip()
                    break
        blocks.append(Block(text=text, kind=kind))
    return blocks


def _iter_docx_items(document: DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in document order."""
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _table_rows(table: Table) -> list[Block]:
    """One body block per row, cells joined with `` | ``."""
    blocks: list[Block] = []
    for row in table.rows:
        cells = [" ".join(cell.text.split()) for cell in row.cells]
        if any(cells):
            blocks.append(Block(text=" | ".join(cells), kind="body"))
    return blocks


_DOCX_HEADINGS: dict[str, BlockKind] = {
    "Heading 1": "title",
    "Title": "title",
    "Heading 2": "section",
    "Heading 3": "subsection",
}


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
}

def load(path: Path) -> list[Block]:
    """Read ``path`` into blocks. Supports ``.md`` and ``.docx``."""
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"unsupported file type: {path.suffix}")
    return loader(path)
