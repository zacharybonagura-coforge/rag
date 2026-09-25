"""Interactive RAG loop. Type a question; type quit to exit."""

import sys

from adapters.factories import build_embedder, build_generator, build_store
from generate import generate
from ingest import ingest
from retrieve import retrieve_reranked, route_query
from schemas.response import build_response
from settings.config import Settings


def _print_response(response) -> None:
    print(f"\nanswer:\n{response.answer}\n")
    if response.citation is None:
        print("citation: none")
    else:
        c = response.citation
        heading = f"{c.document} v{c.version} §{c.section} {c.section_title}"
        if c.subsection_title:
            heading += f" / {c.subsection_title}"
        print(f"citation: {heading}")
    print("\nretrieved chunks:")
    for i, ref in enumerate(response.retrieved_chunks, start=1):
        heading = f"{ref.document} v{ref.version} §{ref.section} {ref.section_title}"
        if ref.subsection_title:
            heading += f" / {ref.subsection_title}"
        snippet = ref.text.replace("\n", " ")
        print(f"  {i}. score={ref.score:.4f}  {heading}")
        print(f"     {snippet[:200]}")
    print()


def main() -> None:
    settings = Settings.from_env()
    embedder = build_embedder(settings)
    store = build_store(settings)
    model = build_generator(settings)

    chunks, documents = ingest(
        settings.corpus_path,
        embedder,
        max_chars=settings.chunk_max_chars,
        overlap=settings.chunk_overlap,
    )
    store.save_chunks(chunks)
    for doc in documents:
        status = "ok" if doc.ok else doc.reason
        print(f"{doc.document}: {status}", file=sys.stderr)
    print(f"stored {len(chunks)} chunks. type a question, or quit.", file=sys.stderr)

    while True:
        query = input("query> ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break

        scope = route_query(query)
        prompt = "policy.compare.v1" if scope == "all" else settings.prompt_name
        hits = retrieve_reranked(
            query,
            embedder,
            store,
            k=settings.retrieve_k,
            pool=settings.retrieve_pool,
        )
        answer = generate(query, hits, model, prompt)
        _print_response(build_response(query, answer, hits))


if __name__ == "__main__":
    main()
