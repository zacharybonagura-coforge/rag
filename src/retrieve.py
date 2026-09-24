"""Retrieve nearest chunks for a natural-language query."""

import re

from rank_bm25 import BM25Okapi

from adapters.embeddings.base import EmbeddingAdapter
from adapters.store.base import VectorStoreAdapter
from schemas.chunk import Chunk, ScoredChunk

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "am", "was", "were", "do", "does",
    "did", "to", "of", "in", "on", "for", "and", "or", "if", "i", "you",
    "we", "what", "when", "where", "who", "how", "much", "many",
    "need",
})


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def document_family(document: str) -> str:
    return re.sub(r"-v[0-9]+$", "", document)


def version_key(version: str) -> tuple[int, ...]:
    if not version:
        return (0,)
    return tuple(int(part) for part in version.split("."))


def latest_chunks(chunks: list[Chunk]) -> list[Chunk]:
    best: dict[str, tuple[int, ...]] = {}
    for chunk in chunks:
        family = document_family(chunk.document)
        key = version_key(chunk.version)
        if family not in best or key > best[family]:
            best[family] = key
    return [
        c
        for c in chunks
        if version_key(c.version) == best[document_family(c.document)]
    ]

def search_keyword(
    query: str,
    chunks: list[Chunk],
    k: int = 3,
) -> list[ScoredChunk]:
    query_terms = tokenize(query)
    if not query_terms or not chunks:
        return []
    docs = [tokenize(f"{c.section_title} {c.text}") for c in chunks]
    scores = BM25Okapi(docs).get_scores(query_terms)
    scored = [
        ScoredChunk(chunk=chunk, score=float(score))
        for chunk, score in zip(chunks, scores, strict=True)
        if score > 0
    ]
    scored.sort(key=lambda hit: hit.score, reverse=True)
    return scored[:k]


def retrieve(
    query: str,
    embedder: EmbeddingAdapter,
    store: VectorStoreAdapter,
    k: int = 3,
) -> list[ScoredChunk]:
    """Embed ``query`` and return the ``k`` nearest stored chunks."""
    query_embedding = embedder.embed_query(query)
    return store.search(query_embedding, k)


def retrieve_hybrid(
    query: str,
    embedder: EmbeddingAdapter,
    store: VectorStoreAdapter,
    k: int = 3,
) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
    vector = retrieve(query, embedder, store, k)
    keyword = search_keyword(query, latest_chunks(store.load_chunks()), k)

    return vector, keyword
