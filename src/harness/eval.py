"""Score retrieval recall and answer key phrases against the golden set."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from schemas.chunk import Chunk, ScoredChunk
from schemas.response import REFUSE, RagResponse, RetrievedChunkRef

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = ROOT / "data/golden-corpus"
LATEST_GOLDEN = (
    "gru-minion-handbook-v2.json",
    "nefario-lab-safety-v1.json",
    "girls-house-rules-v1.json",
)


Phrase = str | tuple[str, ...]

@dataclass(frozen=True)
class GoldQuestion:
    id: str
    type: str
    query: str
    document: str
    gold_section: str
    gold_subsection: str
    must_include: tuple[Phrase, ...]
    must_exclude: tuple[Phrase, ...]
    retrieve_k: int


@dataclass(frozen=True)
class QuestionScore:
    id: str
    query: str
    recalled: bool
    include_ok: bool
    exclude_ok: bool
    hit_ids: tuple[str, ...]

    @property
    def answer_ok(self) -> bool:
        return self.include_ok and self.exclude_ok


@dataclass(frozen=True)
class EvalReport:
    scores: list[QuestionScore]
    retrieve_k: int

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def recall_at_k(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.recalled) / self.n

    @property
    def answer_pass(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.answer_ok) / self.n


def _phrases(raw: list) -> tuple[Phrase, ...]:
    return tuple(tuple(item) if isinstance(item, list) else item for item in raw)


def load_questions(paths: list[Path] | None = None) -> list[GoldQuestion]:
    """Load latest-version golden files (v2 + Nefario + girls)."""
    files = paths or [GOLDEN_DIR / name for name in LATEST_GOLDEN]
    questions: list[GoldQuestion] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        k = int(payload.get("retrieve_k", 5))
        for raw in payload["questions"]:
            questions.append(
                GoldQuestion(
                    id=raw["id"],
                    type=raw.get("type", "fact"),
                    query=raw["query"],
                    document=payload["document"],
                    gold_section=raw["gold_section"],
                    gold_subsection=raw.get("gold_subsection", ""),
                    must_include=_phrases(raw.get("must_include", [])),
                    must_exclude=_phrases(raw.get("must_exclude", [])),
                    retrieve_k=k,
                )
            )
    return questions


def chunk_id_candidates(ref: RetrievedChunkRef) -> set[str]:
    """IDs this ref might have, with and without ``:part-N``."""
    base = f"{ref.document}:v{ref.version}:section-{ref.section}"
    if ref.subsection:
        base += f":sub-{ref.subsection}"
    return {base, f"{base}:part-{ref.part}"}


def hit_ids_from_response(response: RagResponse, k: int) -> set[str]:
    ids: set[str] = set()
    for ref in response.retrieved_chunks[:k]:
        ids |= chunk_id_candidates(ref)
    return ids


def heading_match(chunk: Chunk, question: GoldQuestion) -> bool:
    if chunk.document != question.document:
        return False
    if chunk.section_title != question.gold_section:
        return False
    return not (
        question.gold_subsection
        and chunk.subsection_title != question.gold_subsection
    )


def score_retrieval(question: GoldQuestion, hits: list[ScoredChunk]) -> bool:
    """True if any gold chunk id is in the retrieved set."""
    if question.type == "unanswerable" or not question.gold_section:
        return True
    return any(heading_match(hit.chunk, question) for hit in hits[: question.retrieve_k])


_CITE_TAIL = re.compile(r"\s*\(cited from\b.*$", re.IGNORECASE | re.DOTALL)


def _answer_body(answer: str) -> str:
    return _CITE_TAIL.sub("", answer).strip()


def _phrase_hit(answer: str, item: Phrase) -> bool:
    text = _answer_body(answer).lower()
    if isinstance(item, str):
        return item.lower() in text
    return any(phrase.lower() in text for phrase in item)


def score_answer(question: GoldQuestion, answer: str) -> tuple[bool, bool]:
    """``(all must_include present, no must_exclude present)``."""
    include_ok = all(_phrase_hit(answer, item) for item in question.must_include)
    exclude_ok = all(not _phrase_hit(answer, item) for item in question.must_exclude)
    if pii_spans(answer):
        exclude_ok = False
    return include_ok, exclude_ok


def score_question(
    question: GoldQuestion,
    *,
    hits: list[ScoredChunk],
    answer: str,
) -> QuestionScore:
    include_ok, exclude_ok = score_answer(question, answer)
    return QuestionScore(
        id=question.id,
        query=question.query,
        recalled=score_retrieval(question, hits[: question.retrieve_k]),
        include_ok=include_ok,
        exclude_ok=exclude_ok,
        hit_ids=tuple(h.chunk.chunk_id for h in hits[: question.retrieve_k]),
    )


def _hits_from_response(response: RagResponse) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id="",
                document=ref.document,
                version=ref.version,
                section=ref.section,
                section_title=ref.section_title,
                subsection=ref.subsection,
                subsection_title=ref.subsection_title,
                text=ref.text,
                part=ref.part,
                page=ref.page,
                embedding=[],
            ),
            score=ref.score,
        )
        for ref in response.retrieved_chunks
    ]


def score_run(
    questions: list[GoldQuestion],
    responses: list[RagResponse],
) -> EvalReport:
    """Align questions to responses by ``query``."""
    by_query = {r.query: r for r in responses}
    scores = [
        score_question(
            question,
            hits=_hits_from_response(by_query[question.query]),
            answer=by_query[question.query].answer,
        )
        for question in questions
    ]
    k = questions[0].retrieve_k if questions else 5
    return EvalReport(scores=scores, retrieve_k=k)


_SSN_RE = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")
_SSN_COMPACT_RE = re.compile(r"\b\d{9}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"
)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_PEM_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = ord(ch) - 48
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def pii_spans(text: str) -> list[str]:
    """Return sensitive spans. Empty means the text is clean."""
    hits: list[str] = []
    hits.extend(_SSN_RE.findall(text))
    hits.extend(_SSN_COMPACT_RE.findall(text))
    hits.extend(_PHONE_RE.findall(text))
    hits.extend(_EMAIL_RE.findall(text))
    for raw in _CARD_RE.findall(text):
        digits = re.sub(r"\D", "", raw)
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            hits.append(raw)
    hits.extend(_IBAN_RE.findall(text))
    hits.extend(_AWS_KEY_RE.findall(text))
    hits.extend(_PEM_RE.findall(text))
    seen: set[str] = set()
    unique: list[str] = []
    for hit in hits:
        if hit not in seen:
            seen.add(hit)
            unique.append(hit)
    return unique


def gate_pii(answer: str) -> str:
    """Replace a leaked completion with the refuse sentence."""
    if pii_spans(answer):
        return REFUSE
    return answer