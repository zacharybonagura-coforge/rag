import pytest

from generate import generate
from schemas.chunk import Chunk, ScoredChunk
from schemas.response import REFUSE

HOURS = Chunk(
    chunk_id="harbor-bike-shop-handbook:v1.0:section-1",
    document="harbor-bike-shop-handbook",
    version="1.0",
    section="1",
    section_title="Hours",
    text="The shop is open Tuesday through Saturday from 10am to 6pm.",
    embedding=[0.0] * 768,
)


class FakeModel:
    provider = "fake"
    model_id = "fake"

    def __init__(self, reply: str = "The shop closes at 6pm.") -> None:
        self.reply = reply
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.reply


def test_generate_renders_mini_v1_and_returns_model_text() -> None:
    model = FakeModel()
    query = "What time do you close?"
    hits = [ScoredChunk(chunk=HOURS, score=0.2)]

    answer = generate(query, hits, model)

    assert answer == "The shop closes at 6pm."
    assert model.last_prompt is not None
    assert query in model.last_prompt
    assert "Hours" in model.last_prompt
    assert HOURS.text in model.last_prompt
    assert "using only the context" in model.last_prompt


def test_generate_uses_named_prompt_file() -> None:
    model = FakeModel()
    generate("Are you open on Sunday?", [ScoredChunk(chunk=HOURS, score=0.3)], model, "mini.v1")

    assert model.last_prompt is not None
    assert "Are you open on Sunday?" in model.last_prompt


def test_generate_missing_prompt_raises() -> None:
    with pytest.raises(FileNotFoundError):
        generate("hours?", [ScoredChunk(chunk=HOURS, score=0.1)], FakeModel(), "missing.v9")


def test_generate_propagates_model_error() -> None:
    class BoomModel(FakeModel):
        def generate(self, prompt: str) -> str:
            raise RuntimeError("ollama down")

    with pytest.raises(RuntimeError, match="ollama down"):
        generate(
            "How much is a city bike?",
            [ScoredChunk(chunk=HOURS, score=0.1)],
            BoomModel(),
        )


def test_generate_replaces_pii_with_refuse() -> None:
    model = FakeModel("Agnes's social is 219-09-9999.")
    answer = generate(
        "What social is on the card?",
        [ScoredChunk(chunk=HOURS, score=0.1)],
        model,
    )
    assert answer == REFUSE
    assert "219" not in answer