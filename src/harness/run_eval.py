"""Ingest the mixed corpus, run the golden set, score recall and answers."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from adapters.factories import build_embedder, build_generator, build_store
from generate import generate
from harness.eval import (
    EvalReport,
    load_questions,
    score_question,
)
from ingest import ingest
from retrieve import cross_encode_rerank, retrieve, retrieve_fused, retrieve_reranked
from schemas.response import RagResponse, build_response
from settings.config import Settings

ROOT = Path(__file__).resolve().parents[2]


def write_eval(
    report: EvalReport,
    responses: list[RagResponse],
    runs_dir: Path,
) -> Path:
    runs_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = runs_dir / f"eval-{stamp}.json"
    payload = {
        "retrieve_k": report.retrieve_k,
        "n": report.n,
        "recall_at_k": report.recall_at_k,
        "answer_pass": report.answer_pass,
        "results": [
            {
                "id": score.id,
                "recalled": score.recalled,
                "include_ok": score.include_ok,
                "exclude_ok": score.exclude_ok,
                **response.model_dump(),
            }
            for score, response in zip(report.scores, responses, strict=True)
        ],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    settings = Settings.from_env()
    embedder = build_embedder(settings)
    store = build_store(settings)
    model = build_generator(settings)
    questions = load_questions()
    corpus = settings.corpus_path

    chunks, documents = ingest(
        corpus,
        embedder,
        max_chars=settings.chunk_max_chars,
        overlap=settings.chunk_overlap,
    )
    store.save_chunks(chunks)
    for doc in documents:
        status = "ok" if doc.ok else doc.reason
        print(f"{doc.document}: {status}", file=sys.stderr)
    print(f"stored {len(chunks)} chunks from {corpus}", file=sys.stderr)

    prompt_file = "policy.v2"
    pool = settings.retrieve_pool
    scores = []
    responses = []
    for question in questions:
        if settings.hybrid_retrieve and settings.cross_encode:
            hits = retrieve_reranked(
                question.query, embedder, store,
                k=question.retrieve_k, pool=pool,
            )
        elif settings.hybrid_retrieve:
            hits = retrieve_fused(
                question.query, embedder, store,
                k=question.retrieve_k, pool=pool,
            )
        elif settings.cross_encode:
            hits = cross_encode_rerank(
                question.query,
                retrieve(question.query, embedder, store, k=pool),
                question.retrieve_k,
            )
        else:
            hits = retrieve(
                question.query, embedder, store, k=question.retrieve_k,
            )

        answer = generate(question.query, hits, model, prompt_file)
        score = score_question(question, hits=hits, answer=answer)
        scores.append(score)
        responses.append(build_response(question.query, answer, hits))
        mark_r = "Y" if score.recalled else "n"
        mark_a = "Y" if score.answer_ok else "n"
        print(f"{question.id:24} recall={mark_r}  answer={mark_a}  {answer}")

    report = EvalReport(scores=scores, retrieve_k=questions[0].retrieve_k)
    print(
        f"\nrecall@{report.retrieve_k}={report.recall_at_k:.2f} "
        f"({sum(s.recalled for s in scores)}/{report.n})  "
        f"answers={report.answer_pass:.2f} "
        f"({sum(s.answer_ok for s in scores)}/{report.n})",
        file=sys.stderr,
    )
    out = write_eval(report, responses, settings.runs_dir)
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
