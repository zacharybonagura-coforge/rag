import os

import pytest

from generate import generate
from harness.eval import EvalReport, load_questions, score_question
from ingest import ingest
from retrieve import cross_encode_rerank, retrieve, retrieve_fused, retrieve_reranked
from settings.config import Settings


def _hits(query, embedder, store, settings, k):
    pool = settings.retrieve_pool
    if settings.hybrid_retrieve and settings.cross_encode:
        return retrieve_reranked(query, embedder, store, k=k, pool=pool)
    if settings.hybrid_retrieve:
        return retrieve_fused(query, embedder, store, k=k, pool=pool)
    if settings.cross_encode:
        return cross_encode_rerank(query, retrieve(query, embedder, store, k=pool), k)
    return retrieve(query, embedder, store, k=k)


pytestmark = pytest.mark.integration

MIN_RECALL = 1.00
MIN_ANSWER = 0.9


@pytest.fixture(scope="module")
def stack():
    if "DATABASE_URL" not in os.environ:
        pytest.skip("DATABASE_URL is required for live retrieval")
    settings = Settings.from_env()
    from adapters.factories import build_embedder, build_generator, build_store

    embedder = build_embedder(settings)
    store = build_store(settings)
    model = build_generator(settings)
    chunks, _ = ingest(
        settings.corpus_path,
        embedder,
        max_chars=settings.chunk_max_chars,
        overlap=settings.chunk_overlap,
    )
    store.save_chunks(chunks)
    return settings, embedder, store, model


@pytest.fixture(scope="module")
def scored(stack):
    settings, embedder, store, model = stack
    questions = load_questions()
    assert len(questions) >= 8
    scores = []
    for question in questions:
        hits = _hits(question.query, embedder, store, settings, k=question.retrieve_k)
        answer = generate(question.query, hits, model, "policy.v2")
        scores.append(score_question(question, hits=hits, answer=answer))
    return EvalReport(scores=scores, retrieve_k=questions[0].retrieve_k)


def test_retrieval_recall_at_k(scored: EvalReport) -> None:
    missed = [s.id for s in scored.scores if not s.recalled]
    assert scored.recall_at_k >= MIN_RECALL, f"recall@{scored.retrieve_k}={scored.recall_at_k:.2f} missed={missed}"


def test_answers_contain_expected_keys(scored: EvalReport) -> None:
    failed = [s.id for s in scored.scores if not s.answer_ok]
    assert scored.answer_pass >= MIN_ANSWER, f"answer_pass={scored.answer_pass:.2f} failed={failed}"
