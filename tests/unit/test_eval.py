from dataclasses import replace

from harness.eval import (
    GoldQuestion,
    gate_pii,
    load_questions,
    pii_spans,
    score_answer,
    score_retrieval,
)
from schemas.chunk import Chunk, ScoredChunk
from schemas.response import REFUSE


def _q(
    *,
    type: str = "fact",
    gold_section: str = "Daily Operations",
    gold_subsection: str = "Morning Briefing",
) -> GoldQuestion:
    return replace(
        GoldQuestion(
            id="q",
            type="fact",
            query="q",
            document="gru-minion-handbook-v2",
            gold_section="Daily Operations",
            gold_subsection="Morning Briefing",
            must_include=("8:00",),
            must_exclude=("7:30",),
            retrieve_k=5,
        ),
        type=type,
        gold_section=gold_section,
        gold_subsection=gold_subsection,
    )


def _hit(document, section_title, subsection_title="") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id="x",
            document=document,
            version="2.0",
            section="2",
            section_title=section_title,
            subsection_title=subsection_title,
            text="",
            embedding=[],
        ),
        score=0.1,
    )


def test_golden_set_has_at_least_eight_questions() -> None:
    questions = load_questions()
    assert len(questions) >= 8
    for q in questions:
        assert q.query
        assert q.must_include
        if q.type != "unanswerable":
            assert q.gold_section


def test_recall_true_when_gold_heading_in_top_k() -> None:
    q = _q()
    hits = [
        _hit("nefario-lab-safety-v1", "Lab Access"),
        _hit("gru-minion-handbook-v2", "Daily Operations", "Morning Briefing"),
    ]
    assert score_retrieval(q, hits) is True


def test_recall_false_when_gold_heading_missing() -> None:
    q = _q()
    hits = [_hit("nefario-lab-safety-v1", "Lab Access")]
    assert score_retrieval(q, hits) is False


def test_unanswerable_counts_as_recalled() -> None:
    q = _q(type="unanswerable", gold_section="", gold_subsection="")
    assert score_retrieval(q, []) is True


def test_answer_requires_key_phrases_and_blocks_excludes() -> None:
    q = _q()
    assert score_answer(q, "Report at 8:00am. (cited from …)") == (True, True)
    assert score_answer(q, "Report at 7:30am.") == (False, False)


def test_pii_spans_finds_ssn_phone_email_and_card() -> None:
    text = (
        "SSN 219-09-9999 or 219099999. Call (212) 555-0100. "
        "Mail ada@gru.test. Card 4111 1111 1111 1111."
    )
    spans = pii_spans(text)
    assert "219-09-9999" in spans
    assert "219099999" in spans
    assert "(212) 555-0100" in spans
    assert "ada@gru.test" in spans
    assert "4111 1111 1111 1111" in spans
def test_pii_spans_ignores_handbook_tags() -> None:
    assert pii_spans("IEC-62443-3-3-M19-K2") == []
    assert pii_spans("UN-S-$!A4C9-77B0-QL") == []
    assert pii_spans("IBD-21434-Q7-8841CX") == []
    assert pii_spans(REFUSE) == []
def test_gate_pii_replaces_leak_and_keeps_clean_text() -> None:
    assert gate_pii("Agnes's social is 219-09-9999.") == REFUSE
    assert gate_pii("Lights stay dim after 8pm.") == "Lights stay dim after 8pm."
def test_score_answer_fails_exclude_when_ssn_leaks() -> None:
    q = _q()
    assert score_answer(q, "Report at 8:00am. SSN 219-09-9999.") == (True, False)