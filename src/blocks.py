from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


def load(path: Path) -> list[Block]:
    """Read ``path`` into blocks. Only ``.md`` is supported so far."""
    if path.suffix == ".md":
        return _load_markdown(path)
    raise ValueError(f"unsupported file type: {path.suffix}")
