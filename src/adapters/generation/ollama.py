"""Ollama implementation of ModelAdapter."""

import httpx


class OllamaAdapter:
    """ModelAdapter backed by a local Ollama generate endpoint.

    Posts to ``{host}/api/generate`` with streaming off and temperature 0.
    """

    provider = "ollama"

    def __init__(
        self,
        model_id: str = "mistral:7b",
        host: str = "http://localhost:11434",
    ) -> None:
        """Use ``model_id`` and the Ollama ``host`` generate URL."""
        self.model_id = model_id
        self._url = f"{host}/api/generate"

    def generate(self, prompt: str) -> str:
        """Return Ollama's completion for ``prompt``."""
        response = httpx.post(
            self._url,
            json={
                "model": self.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "seed": 42},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
