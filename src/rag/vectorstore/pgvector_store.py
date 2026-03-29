"""
PgVectorStore.

PostgreSQL + pgvector를 사용하는 운영용 벡터 저장소.

테이블 스키마:
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE TABLE IF NOT EXISTS research_chunks (
      chunk_id     TEXT PRIMARY KEY,
      source_id    TEXT NOT NULL,
      source_type  TEXT NOT NULL,
      content      TEXT NOT NULL,
      chunk_index  INTEGER NOT NULL DEFAULT 0,
      embedding    vector(<dims>) NOT NULL,
      metadata     JSONB DEFAULT '{}'
  );
  CREATE INDEX IF NOT EXISTS idx_chunks_source_id
      ON research_chunks(source_id);

asyncpg 패키지가 없으면 ImportError를 발생시킨다.
개발/테스트 환경에서는 InMemoryVectorStore를 사용하라.
"""
import json
import logging
from typing import Any

from src.domain.models.embedded_document import EmbeddedDocument
from src.domain.models.search_result import SearchResult
from src.domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS research_chunks (
    chunk_id     TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    content      TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL DEFAULT 0,
    embedding    vector({dims}) NOT NULL,
    metadata     JSONB DEFAULT '{{}}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_source_id
    ON research_chunks(source_id);
"""


class PgVectorStore(VectorStorePort):
    """
    asyncpg 기반 pgvector 저장소.
    운영 환경에서 InMemoryVectorStore를 대체한다.
    """

    def __init__(self, pool: Any, embedding_dims: int = 1536) -> None:
        """
        Args:
            pool: asyncpg.Pool 인스턴스 (타입 힌트 생략으로 asyncpg 의존성 분리)
            embedding_dims: 임베딩 차원 수 (OpenAI ada-002 기본값 1536)
        """
        self._pool = pool
        self._dims = embedding_dims

    async def initialize(self) -> None:
        """테이블과 인덱스를 생성한다. 서버 시작 시 한 번 호출한다."""
        async with self._pool.acquire() as conn:
            await conn.execute(_DDL.format(dims=self._dims))
        logger.info("pgvector_store_initialized dims=%s", self._dims)

    async def upsert(self, docs: list[EmbeddedDocument]) -> int:
        if not docs:
            return 0

        rows = [
            (
                doc.chunk_id,
                doc.source_id,
                doc.source_type,
                doc.content,
                doc.chunk_index,
                f"[{','.join(str(v) for v in doc.embedding)}]",
                json.dumps(doc.metadata),
            )
            for doc in docs
        ]

        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO research_chunks
                    (chunk_id, source_id, source_type, content, chunk_index, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::vector, $7::jsonb)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content     = EXCLUDED.content,
                    embedding   = EXCLUDED.embedding,
                    chunk_index = EXCLUDED.chunk_index,
                    metadata    = EXCLUDED.metadata
                """,
                rows,
            )
        logger.info("pgvector_upsert count=%s", len(rows))
        return len(rows)

    async def similarity_search(
        self,
        query_vector: list[float],
        k: int = 5,
        source_type_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        pgvector의 <=> 연산자는 코사인 거리(0~2)를 반환한다.
        score = 1 - (cosine_distance / 2) 로 변환해 0~1 범위의 유사도를 반환한다.
        """
        vec_literal = f"[{','.join(str(v) for v in query_vector)}]"

        if source_type_filter:
            sql = """
                SELECT chunk_id, source_id, source_type, content, chunk_index,
                       embedding::text, metadata,
                       (embedding <=> $1::vector) AS cosine_distance
                FROM research_chunks
                WHERE source_type = $3
                ORDER BY cosine_distance
                LIMIT $2
            """
            params = (vec_literal, k, source_type_filter)
        else:
            sql = """
                SELECT chunk_id, source_id, source_type, content, chunk_index,
                       embedding::text, metadata,
                       (embedding <=> $1::vector) AS cosine_distance
                FROM research_chunks
                ORDER BY cosine_distance
                LIMIT $2
            """
            params = (vec_literal, k)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = [
            SearchResult(
                document=EmbeddedDocument(
                    chunk_id=row["chunk_id"],
                    source_id=row["source_id"],
                    source_type=row["source_type"],
                    content=row["content"],
                    chunk_index=row["chunk_index"],
                    embedding=_parse_pg_vector(row["embedding"]),
                    metadata=json.loads(row["metadata"]),
                ),
                score=max(0.0, min(1.0, 1.0 - float(row["cosine_distance"]) / 2.0)),
            )
            for row in rows
        ]
        logger.info(
            "pgvector_search k=%s filter=%s found=%s",
            k,
            source_type_filter,
            len(results),
        )
        return results

    async def delete_by_source(self, source_id: str) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM research_chunks WHERE source_id = $1",
                source_id,
            )
        count = int(result.split()[-1])
        logger.info("pgvector_delete source_id=%s removed=%s", source_id, count)
        return count


def _parse_pg_vector(text: str) -> list[float]:
    return [float(v) for v in text.strip("[]").split(",") if v.strip()]
