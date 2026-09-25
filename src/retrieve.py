"""Retrieve nearest chunks for a natural-language query."""

import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

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


def fuse_rrf(
    vector: list[ScoredChunk],
    keyword: list[ScoredChunk],
    k: int = 3,
    rrf_k: int = 60,
    vector_weight: float = 0.65,
    keyword_weight: float = 0.35,
) -> list[ScoredChunk]:
    chunks: dict[str, ScoredChunk] = {}
    scores: dict[str, float] = {}

    for rank, hit in enumerate(vector, start=1):
        chunk_id = hit.chunk.chunk_id
        chunks[chunk_id] = hit
        scores[chunk_id] = scores.get(chunk_id, 0.0) + vector_weight / (rrf_k + rank)

    for rank, hit in enumerate(keyword, start=1):
        chunk_id = hit.chunk.chunk_id
        chunks[chunk_id] = hit
        scores[chunk_id] = scores.get(chunk_id, 0.0) + keyword_weight / (rrf_k + rank)

    fused = [
        ScoredChunk(chunk=chunks[chunk_id].chunk, score=score)
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused[:k]


def retrieve_fused(
    query: str,
    embedder: EmbeddingAdapter,
    store: VectorStoreAdapter,
    k: int = 3,
    pool: int = 10,
    vector_weight=0.65,
    keyword_weight=0.35
) -> list[ScoredChunk]:
    vector, keyword = retrieve_hybrid(query, embedder, store, k=pool)
    return fuse_rrf(
        vector,
        keyword,
        k=k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )


_cross_encoder: CrossEncoder | None = None

def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder

def cross_encode_rerank(query: str, hits: list[ScoredChunk], k: int) -> list[ScoredChunk]:
    """Reorder ``hits`` with a cross-encoder and keep the top ``k``."""
    if not hits:
        return []
    pairs = [
        (query, f"{hit.chunk.section_title} {hit.chunk.text}") for hit in hits
    ]
    _cross_encoder = _get_cross_encoder()
    scores = _cross_encoder.predict(pairs)
    ranked = [
        ScoredChunk(chunk=hit.chunk, score=float(score))
        for hit, score in zip(hits, scores, strict=True)
    ]
    ranked.sort(key=lambda hit: hit.score, reverse=True)
    return ranked[:k]


def retrieve_reranked(
    query: str,
    embedder: EmbeddingAdapter,
    store: VectorStoreAdapter,
    k: int = 3,
    pool: int = 10,
    vector_weight=0.65,
    keyword_weight=0.35
) -> list[ScoredChunk]:
    vector, keyword = retrieve_hybrid(query, embedder, store, k=pool)
    fused = fuse_rrf(
        vector,
        keyword,
        k=pool,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    return cross_encode_rerank(query, fused, k)
