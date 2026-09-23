"""Ingest the tiny corpus, answer one question, and write the run to runs/."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from embeddings.sentence_transformer import SentenceTransformerEmbeddingAdapter
from generate import generate
from generation.ollama import OllamaAdapter
from ingest import ingest
from models.response import RagResponse, build_response
from retrieve import retrieve
from store.pgvector import PgVectorStoreAdapter

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/corpus-tiny/harbor-bike-shop-handbook.md"
RUNS_DIR = ROOT / "runs"
K = 3


def write_run(results: list[RagResponse], runs_dir: Path = RUNS_DIR) -> Path:
    """Write every result to one timestamped JSON file under ``runs_dir``."""
    runs_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = runs_dir / f"{stamp}.json"
    out.write_text(
        json.dumps([r.model_dump() for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    return out


QUERIES = [
    "What days is the shop open?",
    "Are you open on Sunday?",
    "What time do you close?",
    "How much is a city bike?",
    "How much does an electric bike cost?",
    "Do I need a photo ID to rent?",
]


def main() -> None:
    embedder = SentenceTransformerEmbeddingAdapter()
    store = PgVectorStoreAdapter()
    model = OllamaAdapter(host="http://host.docker.internal:11434")

    if not store.load_chunks():
        chunks = ingest(CORPUS, embedder)
        store.save_chunks(chunks)
    
    prompt_file = "mini.v1"
    results = []
    for query in QUERIES:
        hits = retrieve(query, embedder, store, k=K)
        answer = generate(query, hits, model, prompt_file)
        response = build_response(query, answer, hits)
        results.append(response)
        print(f"{query}\n{answer}\n")

    out = write_run(results)
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
