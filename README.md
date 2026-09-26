# RAG

Retrieval-augmented generation over policy handbooks. It loads Markdown, Word, and PDF, splits them into section and subsection chunks, embeds those chunks, stores them in Postgres with pgvector, retrieves the nearest chunks (vector, optional BM25+RRF, optional MiniLM rerank), and asks a local model for a short cited answer.

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

`DATABASE_URL` is required. From the devcontainer, use the host port:

```bash
export DATABASE_URL=postgresql://rag:rag@host.docker.internal:5432/rag
```

On the host:

```bash
export DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
```

## Run

From the repo root, with `src` on `PYTHONPATH`:

```bash
# Two known chunks (Hours, Rentals). Vector search only, prompt mini.v1.
PYTHONPATH=src uv run python -m harness.run_mini_corpus

# Interactive Q&A over data/corpus. Hybrid + rerank, prompt PROMPT_NAME (policy.v3).
PYTHONPATH=src uv run python -m harness.run_app

# Golden set (21 questions). Hybrid + rerank. Prints recall@k and answer pass.
# Writes runs/eval-<timestamp>.json.
PYTHONPATH=src uv run python -m harness.run_eval

# Vector-only vs hybrid on locker-code queries. Writes runs/hybrid-codes-<timestamp>.json.
PYTHONPATH=src uv run python -m harness.run_hybrid_codes

# Unfiltered vs latest-version search (planted Gru v1/v2 clash). Cosine only.
# Prompt data-quality.v1, chunks 500/100. Writes runs/data-quality-<timestamp>.json.
PYTHONPATH=src uv run python -m harness.run_data_quality

# Label answer misses from an eval JSON.
PYTHONPATH=src uv run python -m harness.diagnose runs/eval-<timestamp>.json
```

`runs/` is gitignored. Copy a timestamped file into `docs/` if you want it in git. Readable write-ups:

- [docs/mini-corpus-run.md](docs/mini-corpus-run.md)
- [docs/eval-run.md](docs/eval-run.md)
- [docs/data-quality-run.md](docs/data-quality-run.md)

## Changing settings for a run

`Settings.from_env()` reads the process environment when a harness starts. Export a variable for the rest of the shell session, or prefix one command. Unset variables fall back to the Environment table below.

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

`HYBRID_RETRIEVE` and `CROSS_ENCODE` turn on for `1`, `true`, `yes`, or `on`. Any other value turns them off.

`run_mini_corpus` calls `ingest()` without chunk env overrides, so it uses `ingest.py` defaults (500/100) unless you export `CHUNK_MAX_CHARS` / `CHUNK_OVERLAP` and change that runner. `run_eval`, `run_app`, and `run_hybrid_codes` use the Settings defaults (250/80).

## Corpus

`data/corpus` is the mixed handbook set:


| File                                                     | Format                                 |
| -------------------------------------------------------- | -------------------------------------- |
| `gru-minion-handbook-v1.md`, `gru-minion-handbook-v2.md` | Markdown, two versions of one handbook |
| `nefario-lab-safety-v1.docx`                             | Word                                   |
| `girls-house-rules-v1.pdf`                               | PDF                                    |


Each document title must contain `(Version …)`. Chunk ids look like `gru-minion-handbook-v2:v2.0:section-2:sub-1:part-0`.

v1 is a planted outdated duplicate. v2 replaces it. Conflicting facts include morning briefing (7:30am vs 8:00am) and the moon door (password `PURPLE-KAZOO-9` vs welded shut). Shipped `search` keeps the latest version per document family. `search_all` keeps both and is what `run_data_quality` uses to surface the clash.

`data/corpus-tiny/harbor-bike-shop-handbook.md` is two sections (Hours, Rentals). `run_mini_corpus` uses it to prove embed-store-retrieve before the mixed corpus.

`data/golden-corpus/` holds scored questions for the latest documents (`gru-minion-handbook-v2`, `nefario-lab-safety-v1`, `girls-house-rules-v1`). Each question names the gold section, phrases the answer must include, and phrases it must exclude. A list of phrases means any one of them counts. Unanswerable items must refuse and must not cite.

## Retrieval

`run_eval`, `run_app`, and the integration test use the same path, controlled by settings (both flags on by default):

1. Embed the query and take a candidate pool (`RETRIEVE_POOL`, default 10) by cosine distance. `search` keeps only the latest version of each document family. `gru-minion-handbook-v1` and `-v2` are one family; Nefario and the house rules keep their own latest.
2. Score the same latest chunks with BM25 (`section_title` + `text`).
3. Fuse the two lists with reciprocal rank fusion (vector weight 0.65, keyword weight 0.35).
4. Rerank the fused pool with `cross-encoder/ms-marco-MiniLM-L-6-v2` and keep the top k.

k on the golden run comes from the golden file (`retrieve_k`, currently 5). `run_mini_corpus` skips fusion and rerank and uses vector search with k of 2.

`run_eval` also runs `route_query`: compare wording (`v1`, `v2`, `changed`, …) uses `search_all` and prompt `policy.compare.v1`. Other queries use latest-only search and `PROMPT_NAME` (default `policy.v3`). That prompt tells the model to apply the excerpt, cite the heading, or answer exactly: `The provided policy does not answer this question.`

`run_hybrid_codes` asks the same locker-code questions twice: `retrieve` (vector only) then `retrieve_fused` (no rerank). The rear-door latch query (`UN-S-$!A4C9-77B0-QL`) is the case where hybrid recovers a gold chunk that cosine-only misses.

## Environment


| Variable              | Default                                   | Purpose                                                                             |
| --------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------- |
| `DATABASE_URL`        | required                                  | Postgres DSN                                                                        |
| `EMBEDDING_PROVIDER`  | `sentence-transformers`                   | Embedding adapter                                                                   |
| `EMBEDDING_MODEL`     | `sentence-transformers/all-mpnet-base-v2` | Embedding model id                                                                  |
| `EMBEDDING_DIMENSION` | `768`                                     | Expected vector size; startup fails if the model disagrees                          |
| `GENERATION_PROVIDER` | `ollama`                                  | Generation adapter                                                                  |
| `GENERATION_MODEL`    | `mistral:7b`                              | Ollama model id                                                                     |
| `STORE_PROVIDER`      | `pgvector`                                | Vector store adapter                                                                |
| `OLLAMA_HOST`         | `http://host.docker.internal:11434`       | Ollama base URL                                                                     |
| `CORPUS_PATH`         | `data/corpus`                             | Directory ingested by `run_eval`, `run_app`, `run_hybrid_codes`, `run_data_quality` |
| `RUNS_DIR`            | `runs`                                    | JSON output directory                                                               |
| `RETRIEVE_K`          | `3`                                       | Default neighbor count on `Settings` (`run_app`)                                    |
| `RETRIEVE_POOL`       | `10`                                      | Candidates before fusion and rerank                                                 |
| `CHUNK_MAX_CHARS`     | `250`                                     | Window size; longer sections split on a line, sentence, or space                    |
| `CHUNK_OVERLAP`       | `80`                                      | Overlap between windows                                                             |
| `HYBRID_RETRIEVE`     | `true`                                    | Fuse vector and BM25 (`1`, `true`, `yes`, `on`)                                     |
| `CROSS_ENCODE`        | `true`                                    | Rerank the fused pool                                                               |
| `PROMPT_NAME`         | `policy.v3`                               | Generation template for `run_eval` / `run_app`                                      |


Provider names are enums in `src/settings/enums.py`. Supported values: `sentence-transformers`, `ollama`, `pgvector`.

## Layout

```text
src/
  blocks.py                  # Markdown, docx, and PDF to title/section/subsection/body
  ingest.py                  # chunk, embed, report unparsable files
  retrieve.py                # vector, BM25, RRF, cross-encoder, route_query
  generate.py                # render a prompt and call the model
  harness/
    run_mini_corpus.py       # tiny handbook, six questions
    run_app.py               # interactive Q&A
    run_eval.py              # golden-set recall and answer scores
    run_hybrid_codes.py      # vector-only vs hybrid
    run_data_quality.py      # latest version vs every version
    diagnose.py              # label eval-run answer misses
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
  prompts/templates/         # mini.v1, policy.v1–v3, policy.compare.v1, data-quality.v1
data/
  corpus/                    # mixed handbook corpus
  corpus-tiny/               # Harbor Bike Shop
  golden-corpus/             # scored questions (latest documents only)
sql/init.sql                 # chunks table
docs/mini-corpus-run.md
docs/eval-run.md
docs/data-quality-run.md
tests/unit/                  # no services required
tests/eval/                  # live recall and answer gates
.github/workflows/checks_rag.yaml
```

## Checks

```bash
uv run ruff check src tests
uv run mypy src tests
uv run pytest -m "not integration" --cov --cov-report=term-missing
```

Coverage `fail_under` is 100 on `src` (harness `run_*.py` and live adapters omitted).

`tests/eval/test_golden_metrics.py` is marked `integration`. It ingests `CORPUS_PATH`, runs `PROMPT_NAME` (default `policy.v3`), and requires recall at k of 1.00 and an answer pass rate of at least 0.90. It skips when `DATABASE_URL` is unset.

```bash
uv run pytest -m integration
```

CI (`.github/workflows/checks_rag.yaml`) runs on pull requests and on pushes to `main` and `feature/rag-pipeline`:

- `checks` (GitHub-hosted): `uv sync --frozen`, mypy, ruff, `pytest -m "not integration" --cov`.
- `live-eval` (self-hosted runner): `pytest -m integration` with `DATABASE_URL=postgresql://rag:rag@localhost:5432/rag` and `OLLAMA_HOST=http://localhost:11434`.

A recorded golden run is recall@5 = 1.00 (21/21) and answer pass = 0.90 (19/21). Two answer misses: `nefario-hum-scream` and `girls-saturday-chore`.

## Design

- **Local embeddings.** `all-mpnet-base-v2` needs no API key. The factory checks that the model dimension matches `EMBEDDING_DIMENSION`.
- **Local generation.** Ollama calls `/api/generate` with temperature 0 and a fixed seed. `mistral:7b` follows the cite-or-refuse prompts. `gate_pii` replaces a leaked SSN (or similar) with the refuse sentence.
- **pgvector.** Cosine distance (`<=>`) is the vector ranker. An ivfflat index is created in `sql/init.sql`. `save_chunks` replaces the table contents.
- **Latest version wins.** Vector search keeps the highest version per document family so v1 and v2 of the same handbook do not both answer. `search_all` is the unfiltered control.
- **Hybrid retrieve plus rerank.** BM25 catches exact policy wording that cosine distance ranks low. The cross-encoder reorders the fused pool.
- **Heading-aware chunks.** Loaders turn Markdown headings, Word heading styles, and PDF text into sections and subsections. Long units are split with overlap. Answers cite those headings. `build_response` attaches a citation from the top hit, or none on refuse.

