from dataclasses import replace

from harness.diagnose import classify, main
from harness.eval import GoldQuestion
from schemas.response import REFUSE


def _q(**kwargs) -> GoldQuestion:
    base = GoldQuestion(
        id="gru-briefing-time",
        type="fact",
        query="q",
        document="gru-minion-handbook-v2",
        gold_section="Daily Operations",
        gold_subsection="Morning Briefing",
        must_include=("8:00",),
        must_exclude=(),
        retrieve_k=3,
    )
    return replace(base, **kwargs)


def _result(*, hits: list[dict], answer: str, include_ok: bool, exclude_ok: bool) -> dict:
    return {
        "id": "gru-briefing-time",
        "retrieved_chunks": hits,
        "answer": answer,
        "include_ok": include_ok,
        "exclude_ok": exclude_ok,
    }


def _hit(section_title: str, subsection_title: str = "") -> dict:
    return {
        "document": "gru-minion-handbook-v2",
        "version": "2.0",
        "section": "2",
        "section_title": section_title,
        "subsection_title": subsection_title,
    }


def test_classify_skips_unanswerable() -> None:
    assert classify(_result(hits=[], answer="x", include_ok=True, exclude_ok=True), _q(type="unanswerable")) == []


def test_classify_skips_when_no_gold_section() -> None:
    assert classify(_result(hits=[], answer="x", include_ok=True, exclude_ok=True), _q(gold_section="")) == []


def test_classify_retrieval_mismatch_and_hallucination() -> None:
    result = _result(
        hits=[_hit("Lab Access")],
        answer="Minions report at noon.",
        include_ok=False,
        exclude_ok=True,
    )
    assert classify(result, _q()) == [
        "retrieval_mismatch",
        "hallucination_under_low_recall",
    ]


def test_classify_retrieval_mismatch_without_hallucination_when_refused() -> None:
    result = _result(
        hits=[_hit("Lab Access")],
        answer=REFUSE,
        include_ok=True,
        exclude_ok=True,
    )
    assert classify(result, _q()) == ["retrieval_mismatch"]


def test_classify_lost_in_the_middle() -> None:
    result = _result(
        hits=[
            _hit("Equipment"),
            _hit("Daily Operations", "Morning Briefing"),
            _hit("Banana Service"),
        ],
        answer="wrong",
        include_ok=False,
        exclude_ok=True,
    )
    assert classify(result, _q()) == ["lost_in_the_middle"]


def test_classify_ok_when_gold_is_first() -> None:
    result = _result(
        hits=[
            _hit("Daily Operations", "Morning Briefing"),
            _hit("Equipment"),
        ],
        answer="Report at 8:00.",
        include_ok=True,
        exclude_ok=True,
    )
    assert classify(result, _q()) == []


def test_main_prints_counts(tmp_path, capsys, monkeypatch) -> None:
    payload = tmp_path / "run.json"
    payload.write_text(
        '{"results": [{"id": "gru-briefing-time", "retrieved_chunks": '
        '[{"document": "x", "version": "1", "section": "1", '
        '"section_title": "Other", "subsection_title": ""}], '
        '"answer": "guess", "include_ok": false, "exclude_ok": true}]}\n'
    )
    monkeypatch.setattr(
        "harness.diagnose.load_questions",
        lambda: [_q()],
    )
    main(payload)
    out = capsys.readouterr().out
    assert "retrieval_mismatch=1" in out
    assert "hallucination_under_low_recall=1" in out
