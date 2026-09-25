from dataclasses import replace

from harness.eval import (
    EvalReport,
    GoldQuestion,
    QuestionScore,
    chunk_id_candidates,
    gate_pii,
    hit_ids_from_response,
    load_questions,
    pii_spans,
    score_answer,
    score_question,
    score_retrieval,
    score_run,
    heading_match,
    _luhn_ok
)
from schemas.response import REFUSE, RagResponse, RetrievedChunkRef
from schemas.chunk import Chunk, ScoredChunk


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


def test_question_score_answer_ok() -> None:
    score = QuestionScore(
        id="q",
        query="q",
        recalled=True,
        include_ok=True,
        exclude_ok=False,
        hit_ids=(),
    )
    assert score.answer_ok is False


def test_empty_eval_report_rates_are_zero() -> None:
    report = EvalReport(scores=[], retrieve_k=5)
    assert report.n == 0
    assert report.recall_at_k == 0.0
    assert report.answer_pass == 0.0


def test_eval_report_rates_on_one_score() -> None:
    score = QuestionScore(
        id="q",
        query="q",
        recalled=True,
        include_ok=True,
        exclude_ok=True,
        hit_ids=(),
    )
    report = EvalReport(scores=[score], retrieve_k=5)
    assert report.n == 1
    assert report.recall_at_k == 1.0
    assert report.answer_pass == 1.0


def test_chunk_id_candidates_includes_sub_and_part() -> None:
    ref = RetrievedChunkRef(
        document="gru-minion-handbook-v2",
        version="2.0",
        section="2",
        section_title="Daily Operations",
        subsection="1",
        part=0,
        score=0.1,
    )
    assert chunk_id_candidates(ref) == {
        "gru-minion-handbook-v2:v2.0:section-2:sub-1",
        "gru-minion-handbook-v2:v2.0:section-2:sub-1:part-0",
    }


def test_hit_ids_from_response_respects_k() -> None:
    response = RagResponse(
        query="q",
        answer="ok",
        citation=None,
        retrieved_chunks=[
            RetrievedChunkRef(
                document="gru-minion-handbook-v2",
                version="2.0",
                section="2",
                section_title="Daily Operations",
                score=0.1,
            ),
            RetrievedChunkRef(
                document="nefario-lab-safety-v1",
                version="1.0",
                section="1",
                section_title="Lab Access",
                score=0.2,
            ),
        ],
    )
    ids = hit_ids_from_response(response, k=1)
    assert "gru-minion-handbook-v2:v2.0:section-2" in ids
    assert "nefario-lab-safety-v1:v1.0:section-1" not in ids


def test_recall_false_when_subsection_wrong() -> None:
    q = _q()
    hits = [_hit("gru-minion-handbook-v2", "Daily Operations", "Banana Service")]
    assert score_retrieval(q, hits) is False


def test_score_question_and_score_run_align_by_query() -> None:
    q = _q()
    hit = _hit("gru-minion-handbook-v2", "Daily Operations", "Morning Briefing")
    response = RagResponse(
        query="q",
        answer="Report at 8:00am.",
        citation=None,
        retrieved_chunks=[
            RetrievedChunkRef(
                document=hit.chunk.document,
                version=hit.chunk.version,
                section=hit.chunk.section,
                section_title=hit.chunk.section_title,
                subsection_title=hit.chunk.subsection_title,
                score=hit.score,
            )
        ],
    )
    scored = score_question(q, hits=[hit], answer=response.answer)
    report = score_run([q], [response])
    assert scored.recalled is True
    assert scored.answer_ok is True
    assert report.recall_at_k == 1.0


def test_score_answer_accepts_any_include_phrase() -> None:
    q = replace(_q(), must_include=(("8", "8:00"),), must_exclude=())
    assert score_answer(q, "Report at 8am.") == (True, True)


def test_pii_spans_ignores_invalid_card() -> None:
    assert pii_spans("Card 4111 1111 1111 1112.") == []


def test_heading_match_when_question_has_no_subsection() -> None:
    q = _q(gold_subsection="")
    chunk = _hit("gru-minion-handbook-v2", "Daily Operations", "Morning Briefing").chunk
    assert heading_match(chunk, q) is True


def test_luhn_handles_doubled_digit_over_nine() -> None:
    assert _luhn_ok("58") is False
    assert "5555 5555 5555 4444" in pii_spans("Card 5555 5555 5555 4444.")
