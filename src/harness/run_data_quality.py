"""Ingest the mixed corpus and trace a version clash to source data."""

import json
import sys
from datetime import UTC, datetime

from adapters.factories import build_embedder, build_generator
from adapters.store.pgvector import PgVectorStoreAdapter
from generate import generate
from ingest import ingest
from schemas.chunk import ScoredChunk
from schemas.response import build_response
from settings.config import Settings

CASES = [
    {"id": "gru-briefing-time", "query": "What time do minions report for the morning briefing?"},
    {"id": "gru-moon-door-password", "query": "What is the moon-door password?"},
]


def _print_hits(label: str, hits: list[ScoredChunk]) -> None:
    print(f"\n{label}")
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        heading = f"{chunk.document} v{chunk.version} §{chunk.section_title}"
        if chunk.subsection_title:
            heading += f" / {chunk.subsection_title}"
        print(f"  {rank}  {heading}  {hit.score:.3f}")
        snippet = chunk.text.replace("\n", " ")
        print(f"      {snippet[:160]}")


def main() -> None:
    settings = Settings.from_env()
    embedder = build_embedder(settings)
    store = PgVectorStoreAdapter(settings.database_url)
    model = build_generator(settings)
    k = 3

    chunks, documents = ingest(
        settings.corpus_path,
        embedder,
        max_chars=500,
        overlap=100,
    )
    store.save_chunks(chunks)
    for doc in documents:
        print(f"{doc.document}: {'ok' if doc.ok else doc.reason}", file=sys.stderr)
    print(f"stored {len(chunks)} chunks (v1 and v2 both present)", file=sys.stderr)

    results = []
    for case in CASES:
        query = case["query"]
        vector = embedder.embed_query(query)
        all_hits = store.search_all(vector, k)
        latest_hits = store.search(vector, k)
        all_answer = generate(query, all_hits, model, "data-quality.v1")
        latest_answer = generate(query, latest_hits, model, "data-quality.v1")

        print(f"\n=== {case['id']} ===")
        print(query)
        _print_hits("search_all (every version)", all_hits)
        _print_hits("search (latest only)", latest_hits)
        print(f"\nanswer search_all:\n{all_answer}")
        print(f"\nanswer search:\n{latest_answer}")

        results.append(
            {
                "id": case["id"],
                "unfiltered": build_response(query, all_answer, all_hits).model_dump(),
                "latest": build_response(query, latest_answer, latest_hits).model_dump(),
            }
        )

    settings.runs_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = settings.runs_dir / f"data-quality-{stamp}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
