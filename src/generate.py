"""Generate an answer from a query and retrieved chunks."""

from adapters.generation.base import ModelAdapter
from harness.eval import gate_pii
from prompts.repository import PromptRepository
from schemas.chunk import ScoredChunk


def generate(
    query: str,
    hits: list[ScoredChunk],
    model: ModelAdapter,
    prompt_file: str = "mini.v1",
) -> str:
    """Render a prompt and return the model completion."""
    prompt_repo = PromptRepository()
    raw = model.generate(prompt_repo.render(prompt_file, query, hits))
    return gate_pii(raw)
