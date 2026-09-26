"""Language-model adapter protocol for prompt completion."""

from typing import Protocol


class ModelAdapter(Protocol):
    """Interface for turning a prompt into generated text.

    Implementations call a local or remote chat/completion model.
    Callers can swap providers as long as they satisfy this protocol.
    """

    provider: str
    model_id: str

    def generate(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""
        ...
