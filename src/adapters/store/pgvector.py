"""pgvector implementation of VectorStoreAdapter."""

import os

import psycopg
from pgvector.psycopg import register_vector

from schemas.chunk import Chunk, ScoredChunk


class PgVectorStoreAdapter:
    """VectorStoreAdapter backed by Postgres + pgvector.
    Connects with ``dsn`` or ``DATABASE_URL``. ``save_chunks`` replaces all
    rows. ``search`` ranks by cosine distance on the stored embedding column.
    """

    provider = "pgvector"

    def __init__(self, dsn: str | None = None) -> None:
        """Use ``dsn`` if given, otherwise ``DATABASE_URL``."""
        self._dsn = dsn or os.environ["DATABASE_URL"]

    def _connect(self) -> psycopg.Connection:
        """Open a connection with the pgvector type registered."""
        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Replace the ``chunks`` table with this batch."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks")
                for chunk in chunks:
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id, document, version, section,
                            section_title, text, subsection,
                            subsection_title, part, page, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            chunk.chunk_id,
                            chunk.document,
                            chunk.version,
                            chunk.section,
                            chunk.section_title,
                            chunk.text,
                            chunk.subsection,
                            chunk.subsection_title,
                            chunk.part,
                            chunk.page,
                            chunk.embedding,
                        ),
                    )
            conn.commit()

    def load_chunks(self) -> list[Chunk]:
        """Load every stored chunk, ordered by ``section``."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, document, version, section,
                    section_title, text, subsection,
                    subsection_title, part, page, embedding
                FROM chunks
                ORDER BY section
                """
            )
            rows = cur.fetchall()
        return [
            Chunk(
                chunk_id=row[0],
                document=row[1],
                version=row[2],
                section=row[3],
                section_title=row[4],
                text=row[5],
                subsection=row[6],
                subsection_title=row[7],
                part=row[8],
                page=row[9],
                embedding=list(row[10]),
            )
            for row in rows
        ]

    def search(
        self,
        query_embedding: list[float],
        k: int = 3,
    ) -> list[ScoredChunk]:
        """Return the ``k`` nearest chunks by cosine distance.
        Restricts hits to the latest version per document family. Family is
        the file stem with a trailing ``-vN`` stripped, so
        ``gru-minion-handbook-v1`` and ``-v2`` compete. Version is compared
        as ``int[]`` (``10.0`` > ``9.0``); an empty version is ``{0}``.
        Other families keep their own latest (Nefario/girls ``1.0`` stay).
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH latest AS (
                    SELECT
                        regexp_replace(document, '-v[0-9]+$', '') AS family,
                        max(
                            COALESCE(
                                string_to_array(NULLIF(version, ''), '.')::int[],
                                ARRAY[0]
                            )
                        ) AS ver
                    FROM chunks
                    GROUP BY 1
                )
                SELECT
                    c.*,
                    (c.embedding <=> %s::vector) AS distance
                FROM chunks c
                JOIN latest l
                ON l.family = regexp_replace(c.document, '-v[0-9]+$', '')
                AND COALESCE(
                        string_to_array(NULLIF(c.version, ''), '.')::int[],
                        ARRAY[0]
                    ) = l.ver
                ORDER BY c.embedding <=> %s::vector ASC
                LIMIT %s
                """,
                (query_embedding, query_embedding, k),
            )
            rows = cur.fetchall()
        return [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=row[0],
                    document=row[1],
                    version=row[2],
                    section=row[3],
                    section_title=row[4],
                    text=row[5],
                    subsection=row[6],
                    subsection_title=row[7],
                    part=row[8],
                    page=row[9],
                    embedding=list(row[10]),
                ),
                score=float(row[11]),
            )
            for row in rows
        ]
