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


V1 = Chunk(
    chunk_id="gru-minion-handbook-v1:v1.0:section-2",
    document="gru-minion-handbook-v1",
    version="1.0",
    section="2",
    section_title="Banana Service",
    text="Each minion may take two bananas before noon.",
    embedding=[0.0] * 768,
)
V2 = Chunk(
    chunk_id="gru-minion-handbook-v2:v2.0:section-2",
    document="gru-minion-handbook-v2",
    version="2.0",
    section="2",
    section_title="Banana Service",
    text="Each minion may take one banana before noon.",
    embedding=[0.0] * 768,
)
GIRLS = Chunk(
    chunk_id="girls-house-rules-v1:v1.0:section-1",
    document="girls-house-rules-v1",
    version="1.0",
    section="1",
    section_title="Bedtime",
    text="Lights out at 8pm.",
    embedding=[0.0] * 768,
)
HARBOR = Chunk(
    chunk_id="harbor-bike-shop-handbook:v1.0:section-1",
    document="harbor-bike-shop-handbook",
    version="1.0",
    section="1",
    section_title="Hours",
    text="The shop closes at 6pm.",
    embedding=[0.0] * 768,
)

def test_generate_renders_compare_prompt() -> None:
    model = FakeModel()
    query = "How did the bananas change from v1 to v2?"
    hits = [
        ScoredChunk(chunk=V1, score=0.2),
        ScoredChunk(chunk=V2, score=0.1),
    ]

    generate(query, hits, model, "policy.compare.v1")

    assert model.last_prompt is not None
    assert query in model.last_prompt
    assert "gru-minion-handbook-v1 §2 Banana Service" in model.last_prompt
    assert "gru-minion-handbook-v2 §2 Banana Service" in model.last_prompt
    assert "Do not prefer a higher-ranked excerpt" in model.last_prompt
    assert V1.text in model.last_prompt
    assert V2.text in model.last_prompt