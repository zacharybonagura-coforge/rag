CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    document       TEXT NOT NULL,
    version        TEXT NOT NULL,
    section        TEXT NOT NULL,
    section_title  TEXT NOT NULL,
    text           TEXT NOT NULL,
    embedding      vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 1);