# RAG

Retrieval-augmented generation over policy handbooks. It loads Markdown, Word, and PDF, splits them into section and subsection chunks, embeds those chunks, stores them in Postgres with pgvector, retrieves the nearest chunks, and asks a local model for a short cited answer.

Default stack: `sentence-transformers/all-mpnet-base-v2` (768-d), Ollama `mistral:7b`, and Postgres 16 with pgvector.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker
- [Ollama](https://ollama.com), for generation and for the golden-set run

```bash
ollama pull mistral:7b
```

## Setup



### Ollama

Start the Ollama app, or run `ollama serve`. Check that it is up:

```bash
curl -s http://localhost:11434/api/tags
```

The app's default `OLLAMA_HOST` is `http://host.docker.internal:11434`, which reaches Ollama on the host from the devcontainer. When you run Python on the host, set `OLLAMA_HOST=http://localhost:11434`.

### Postgres

First time, on your local terminal, create the database and load `sql/init.sql` (the `chunks` table and the ivfflat cosine index):

```bash
docker run --name rag-db \
  -e POSTGRES_USER=rag \
  -e POSTGRES_PASSWORD=rag \
  -e POSTGRES_DB=rag \
  -p 5432:5432 \
  -v "$(pwd)/sql/init.sql:/docker-entrypoint-initdb.d/init.sql" \
  -d pgvector/pgvector:pg16
```

Later sessions:

```bash
docker start rag-db
```

```bash
docker exec rag-db psql -U rag -d rag -c '\dt'
```

The store replaces rows in `chunks`. It does not create the table.

### Python

```bash
uv sync
```

`DATABASE_URL` is required, but we will be running in container, so use host port:

```bash
export DATABASE_URL=postgresql://rag:rag@host.docker.internal:5432/rag
```



## Run

From the repo root, with `src` on `PYTHONPATH`:

```bash
# Six questions over data/corpus-tiny (Harbor Bike Shop). Vector search, prompt mini.v1.
PYTHONPATH=src uv run python -m harness.run_mini_corpus

# Golden set over data/corpus. Hybrid retrieval, prompt policy.v2.
# Search filters out duplicate documents, pulls latest version
# Prints recall@k and answer pass rate, writes runs/eval-<timestamp>.json.
PYTHONPATH=src uv run python -m harness.run_eval

# Same corpus, comparing two questions from different versions.
# Writes runs/data-quality-<timestamp>.json.
PYTHONPATH=src uv run python -m harness.run_data_quality
```

`runs/` is gitignored. A readable write-up of one corpus run is in [docs/corpus-run.md](docs/corpus-run.md).

## Changing settings for a run

`Settings.from_env()` reads the process environment when a harness starts. Export a variable for the rest of the shell session, or prefix one command.

```bash
export HYBRID_RETRIEVE=false
export CROSS_ENCODE=false
export CHUNK_MAX_CHARS=500
PYTHONPATH=src uv run python -m harness.run_eval
```

```bash
RETRIEVE_POOL=20 CORPUS_PATH=data/corpus-tiny \
  PYTHONPATH=src uv run python -m harness.run_mini_corpus
```

`HYBRID_RETRIEVE` and `CROSS_ENCODE` turn on for `1`, `true`, `yes`, or `on`. Any other value turns them off. Unset variables fall back to the defaults in the table above.

## Corpus

`data/corpus` is the mixed handbook set:


| File                                                     | Format                                 |
| -------------------------------------------------------- | -------------------------------------- |
| `gru-minion-handbook-v1.md`, `gru-minion-handbook-v2.md` | Markdown, two versions of one handbook |
| `nefario-lab-safety-v1.docx`                             | Word                                   |
| `girls-house-rules-v1.pdf`                               | PDF                                    |


Each document title must contain `(Version …)`. Chunk ids look like `gru-minion-handbook-v2:v2.0:section-2:sub-1:part-0`.

`data/corpus-tiny/harbor-bike-shop-handbook.md` is the small handbook used only by `run_mini_corpus`.

`data/golden-corpus/` holds the scored questions for the latest documents (`gru-minion-handbook-v2`, `nefario-lab-safety-v1`, and `girls-house-rules-v1`). Each question names the gold section, phrases the answer must include, and phrases it must exclude. A list of phrases means any one of them counts.

## Retrieval

`run_eval` and the integration test use the same path, controlled by settings (both on by default):

1. Embed the query and take a candidate pool (`RETRIEVE_POOL`, default 10) by cosine distance. `search` keeps only the latest version of each document family. `gru-minion-handbook-v1` and `-v2` are one family; Nefario and the house rules keep their own latest.
2. Score the same latest chunks with BM25.
3. Fuse the two lists with reciprocal rank fusion (vector weight 0.65, keyword weight 0.35).
4. Rerank the fused pool with `cross-encoder/ms-marco-MiniLM-L-6-v2` and keep the top k.

k comes from the golden file (`retrieve_k`, currently 5). `run_mini_corpus` skips fusion and rerank and uses vector search with k of 2. Generation uses `policy.v2` for the golden run and `mini.v1` for the tiny corpus. `policy.v2` tells the model to apply the excerpt, cite the heading, or answer exactly: `The provided policy does not answer this question.`

## Environment


| Variable              | Default                                   | Purpose                                                          |
| --------------------- | ----------------------------------------- | ---------------------------------------------------------------- |
| `DATABASE_URL`        | required                                  | Postgres DSN                                                     |
| `EMBEDDING_PROVIDER`  | `sentence-transformers`                   | Embedding adapter                                                |
| `EMBEDDING_MODEL`     | `sentence-transformers/all-mpnet-base-v2` | Embedding model id                                               |
| `EMBEDDING_DIMENSION` | `768`                                     | Expected vector size; startup fails if the model disagrees       |
| `GENERATION_PROVIDER` | `ollama`                                  | Generation adapter                                               |
| `GENERATION_MODEL`    | `mistral:7b`                              | Ollama model id                                                  |
| `STORE_PROVIDER`      | `pgvector`                                | Vector store adapter                                             |
| `OLLAMA_HOST`         | `http://host.docker.internal:11434`       | Ollama base URL                                                  |
| `CORPUS_PATH`         | `data/corpus`                             | Directory ingested by `run_eval` and `run_data_quality`          |
| `RUNS_DIR`            | `runs`                                    | JSON output directory                                            |
| `RETRIEVE_K`          | `3`                                       | Default neighbor count on `Settings`                             |
| `RETRIEVE_POOL`       | `10`                                      | Candidates before fusion and rerank                              |
| `CHUNK_MAX_CHARS`     | `250`                                     | Window size; longer sections split on a line, sentence, or space |
| `CHUNK_OVERLAP`       | `80`                                      | Overlap between windows                                          |
| `HYBRID_RETRIEVE`     | `true`                                    | Fuse vector and BM25 (`1`, `true`, `yes`, `on`)                  |
| `CROSS_ENCODE`        | `true`                                    | Rerank the fused pool                                            |


Provider names are enums in `src/settings/enums.py`. Supported values: `sentence-transformers`, `ollama`, `pgvector`.

## Layout

```text
src/
  blocks.py                  # Markdown, docx, and PDF to title/section/subsection/body
  ingest.py                  # chunk, embed, report unparsable files
  retrieve.py                # vector, BM25, RRF, cross-encoder
  generate.py                # render a prompt and call the model
  harness/
    run_mini_corpus.py       # tiny handbook, six questions
    run_eval.py              # golden-set recall and answer scores
    run_data_quality.py      # latest version vs every version
    eval.py                  # load golden questions and score them
  settings/
    config.py                # Settings.from_env()
    enums.py
  adapters/
    embeddings/              # sentence-transformers
    generation/              # Ollama
    store/                   # pgvector
    factories.py
  schemas/                   # Chunk, ScoredChunk, RagResponse
  prompts/templates/         # mini.v1, policy.v1, policy.v2
data/
  corpus/                    # mixed handbook corpus
  corpus-tiny/               # Harbor Bike Shop
  golden-corpus/             # scored questions
sql/init.sql                 # chunks table
docs/mini-corpus-run.md
tests/unit/                  # no services required
tests/eval/                  # live recall and answer gates
```



## Checks

```bash
uv run ruff check src tests
uv run mypy src tests
uv run pytest tests/unit
```

`tests/eval/test_golden_metrics.py` is marked `integration`. It ingests `CORPUS_PATH`, runs `policy.v2`, and requires recall at k of 1.00 and an answer pass rate of at least 0.90. It skips when `DATABASE_URL` is unset.

```bash
uv run pytest -m integration
```

CI (`.github/workflows/checks_rag.yaml`) runs `uv sync --frozen`, mypy, ruff, and pytest on pull requests and on pushes to `main`.

## Design

- **Local embeddings.** `all-mpnet-base-v2` needs no API key. The factory checks that the model dimension matches `EMBEDDING_DIMENSION`.
- **Local generation.** Ollama calls `/api/generate` with temperature 0 and a fixed seed. `mistral:7b` follows the cite-or-refuse prompts.
- **pgvector.** Cosine distance (`<=>`) is the vector ranker. An ivfflat index is created in `sql/init.sql`. `save_chunks` replaces the table contents.
- **Latest version wins.** Vector search keeps the highest version per document family so v1 and v2 of the same handbook do not both answer.
- **Hybrid retrieve plus rerank.** BM25 catches exact policy wording that cosine distance ranks low. The cross-encoder reorders the fused pool.
- **Heading-aware chunks.** Loaders turn Markdown headings, Word heading styles, and PDF text into sections and subsections. Long units are split with overlap. Answers cite those headings.

