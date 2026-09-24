# RAG

Retrieval-augmented generation system. Chunks markdown handbooks into sections, embeds them with sentence transformers, stores vectors in pgvector, retrieves by cosine distance, and generates cited answers.

Default stack: `sentence-transformers/all-mpnet-base-v2` embeddings, Ollama (`mistral:7b`) for generation, Postgres with pgvector in Docker.

## Prerequisites

- Python 3.12+
- Docker
- [Ollama](https://ollama.com) (optional; required for generation)

Pull the model used by default:

```bash
ollama pull mistral:7b
```

## 1. Start Ollama (Mac)

Open the Ollama app, or run

```bash
ollama serve
```

Verify:

```bash
curl -s http://localhost:11434/api/tags
```



## 2. Start Postgres + pgvector (Docker on local terminal)

**First time** (creates the container):

```bash
docker run --name rag-db \
  -e POSTGRES_USER=YOUR_USER \
  -e POSTGRES_PASSWORD=YOUR_USER \
  -e POSTGRES_DB=rag \
  -p 5432:5432 \
  -v "$(pwd)/sql/init.sql:/docker-entrypoint-initdb.d/init.sql" \
  -d pgvector/pgvector:pg16
```

Later sessions:

```bash
docker start rag-db
```

Verify the table exists:

```bash
docker exec -it rag-db psql -U rag -d rag -c '\dt'
```



## 3. Install the Python app

```bash
uv sync
# or with pip:
# python3.12 -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
```



## Environment variables


| Variable              | Default                                   | Purpose                             |
| --------------------- | ----------------------------------------- | ----------------------------------- |
| `EMBEDDING_PROVIDER`  | `sentence-transformers`                   | Which embedding adapter to build    |
| `EMBEDDING_MODEL`     | `sentence-transformers/all-mpnet-base-v2` | Embedding model id                  |
| `EMBEDDING_DIMENSION` | `768`                                     | Expected embedding vector size      |
| `GENERATION_PROVIDER` | `ollama`                                  | Which generation adapter to build   |
| `GENERATION_MODEL`    | `mistral:7b`                              | Generation model id                 |
| `STORE_PROVIDER`      | `pgvector`                                | Which vector store adapter to build |
| `OLLAMA_HOST`         | `http://host.docker.internal:11434`       | Ollama base URL                     |
| `DATABASE_URL`        | `postgresql://rag:rag@localhost:5432/rag` | Postgres DSN for pgvector           |
| `CORPUS_PATH`         | `data/corpus-tiny`                        | Path to markdown corpus directory   |
| `RETRIEVE_K`          | `3`                                       | Number of nearest chunks to return  |
| `RUNS_DIR`            | `runs`                                    | Output directory for eval results   |


Provider values are validated as enums in `src/enums.py`. Currently supported: `sentence-transformers` (embeddings), `ollama` (generation), `pgvector` (storage).

## Results

Evaluation runs write timestamped JSON to `runs/{timestamp}.json`. See `[docs/mini-corpus-run.md](docs/mini-corpus-run.md)` for a readable example.

## Project layout

```text
src/
  ingest.py              # read corpus, embed chunks, store
  retrieve.py            # query embed + store.search
  generate.py            # grounded generation with Ollama
  run_mini_corpus.py     # 6-question evaluation CLI
  config.py              # environment settings
  enums.py               # provider enums
  adapters/
    embeddings/          # base and sentence-transformers adapter
    generation/          # base and Ollama adapter
    store/               # base and pgvector adapter
    factories.py         # build instances from config
  schemas/
    chunk.py             # Chunk, ScoredChunk models
    response.py          # RagResponse model
  prompts/
    repository.py        # prompt templates
data/
  corpus-tiny/           # example markdown handbook
docs/
  mini-corpus-run.md     # example evaluation results
pyproject.toml          # dependencies via uv
tests/                  # unit ans integration tests
```



## Development checks

From the repo root:

```bash
# Unit tests
pytest

# Type check
mypy src tests

# Lint
ruff check src tests
```



## Design choices

- **Sentence-Transformers embeddings** — local, no API key, 768-d vectors from `all-mpnet-base-v2`. Fast and competitive on semantic search.
- **Ollama generation** — local, CPU-capable, no auth needed. Mistral 7B is small enough for laptops while following grounded cite-or-refuse prompts.
- **pgvector** — Postgres-native vector extension. No separate vector DB; cosine distance (`<=>`) gives nearest-neighbor search. Schema auto-created on first run.
- **Pydantic schemas** — strict validation and serialization for chunks and responses. Ensures citations are always present.

