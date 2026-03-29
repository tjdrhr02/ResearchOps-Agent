"""
RAG Ingestion Pipeline 유닛 테스트.

테스트 대상:
  - SemanticChunker: 청킹 + overlap 검증
  - HashEmbedder: 결정적 출력 + 배치 처리 검증
  - InMemoryVectorStore: upsert / similarity_search / delete_by_source 검증
  - IngestionPipeline: 전체 흐름 통합 검증

모든 테스트는 외부 의존성 없이 동작한다.
"""
import pytest

from src.domain.models.embedded_document import EmbeddedDocument
from src.domain.models.source_document import SourceDocument
from src.rag.chunking.chunker import SemanticChunker
from src.rag.embedding.embedder import HashEmbedder
from src.rag.ingestion.ingestion_pipeline import IngestionPipeline
from src.rag.ingestion.ingestor import DocumentIngestor
from src.rag.vectorstore.in_memory_store import InMemoryVectorStore


def _make_doc(
    source_id: str = "doc-1",
    content: str = "Hello world. This is a test.",
    source_type: str = "blog",
) -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        source_type=source_type,
        title="Test Doc",
        url=f"https://example.com/{source_id}",
        content=content,
    )


# ──────────────────────────────────────────────
# SemanticChunker 테스트
# ──────────────────────────────────────────────

class TestSemanticChunker:
    @pytest.mark.asyncio
    async def test_single_short_doc_yields_one_chunk(self):
        chunker = SemanticChunker(chunk_size=500)
        doc = _make_doc(content="Short content.")
        chunks = await chunker.chunk([doc])
        assert len(chunks) == 1
        assert chunks[0].source_id == "doc-1"

    @pytest.mark.asyncio
    async def test_long_doc_yields_multiple_chunks(self):
        long_text = "\n\n".join([f"Paragraph {i}. " + "word " * 50 for i in range(10)])
        chunker = SemanticChunker(chunk_size=200, overlap_size=20)
        doc = _make_doc(content=long_text)
        chunks = await chunker.chunk([doc])
        assert len(chunks) > 1

    @pytest.mark.asyncio
    async def test_chunk_ids_are_unique(self):
        chunker = SemanticChunker(chunk_size=100)
        long_text = "\n\n".join(["Sentence. " * 20 for _ in range(5)])
        doc = _make_doc(content=long_text)
        chunks = await chunker.chunk([doc])
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_chunk_index_is_sequential(self):
        chunker = SemanticChunker(chunk_size=100)
        long_text = "\n\n".join(["Para " + "x " * 30 for _ in range(5)])
        doc = _make_doc(content=long_text)
        chunks = await chunker.chunk([doc])
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

    @pytest.mark.asyncio
    async def test_empty_content_doc_skipped(self):
        chunker = SemanticChunker()
        doc = _make_doc(content="")
        # 빈 content → title fallback → "Test Doc" → 1 chunk
        chunks = await chunker.chunk([doc])
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_multiple_docs(self):
        chunker = SemanticChunker(chunk_size=500)
        docs = [_make_doc(source_id=f"doc-{i}", content=f"Content {i}.") for i in range(3)]
        chunks = await chunker.chunk(docs)
        source_ids = {c.source_id for c in chunks}
        assert source_ids == {"doc-0", "doc-1", "doc-2"}

    @pytest.mark.asyncio
    async def test_source_type_preserved(self):
        chunker = SemanticChunker()
        doc = _make_doc(source_type="paper", content="Research content.")
        chunks = await chunker.chunk([doc])
        assert all(c.source_type == "paper" for c in chunks)


# ──────────────────────────────────────────────
# HashEmbedder 테스트
# ──────────────────────────────────────────────

class TestHashEmbedder:
    @pytest.mark.asyncio
    async def test_embed_returns_float_list(self):
        embedder = HashEmbedder(dims=8)
        vec = await embedder.embed("test")
        assert isinstance(vec, list)
        assert len(vec) == 8
        assert all(isinstance(v, float) for v in vec)

    @pytest.mark.asyncio
    async def test_embed_is_deterministic(self):
        embedder = HashEmbedder(dims=8)
        v1 = await embedder.embed("hello")
        v2 = await embedder.embed("hello")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_different_texts_differ(self):
        embedder = HashEmbedder(dims=8)
        v1 = await embedder.embed("hello")
        v2 = await embedder.embed("world")
        assert v1 != v2

    @pytest.mark.asyncio
    async def test_embed_batch_returns_correct_count(self):
        embedder = HashEmbedder(dims=8)
        texts = ["a", "b", "c"]
        vecs = await embedder.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 8 for v in vecs)

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        embedder = HashEmbedder()
        vecs = await embedder.embed_batch([])
        assert vecs == []

    @pytest.mark.asyncio
    async def test_langchain_compat_aembed_query(self):
        embedder = HashEmbedder(dims=8)
        vec = await embedder.aembed_query("hello")
        assert len(vec) == 8

    @pytest.mark.asyncio
    async def test_langchain_compat_aembed_documents(self):
        embedder = HashEmbedder(dims=8)
        vecs = await embedder.aembed_documents(["hello", "world"])
        assert len(vecs) == 2


# ──────────────────────────────────────────────
# InMemoryVectorStore 테스트
# ──────────────────────────────────────────────

def _make_embedded(chunk_id: str, source_id: str = "s1", embedding: list[float] | None = None) -> EmbeddedDocument:
    return EmbeddedDocument(
        chunk_id=chunk_id,
        source_id=source_id,
        source_type="blog",
        content=f"Content of {chunk_id}",
        embedding=embedding or [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        chunk_index=0,
    )


class TestInMemoryVectorStore:
    @pytest.mark.asyncio
    async def test_upsert_and_count(self):
        store = InMemoryVectorStore()
        docs = [_make_embedded(f"c{i}") for i in range(3)]
        count = await store.upsert(docs)
        assert count == 3
        assert len(store) == 3

    @pytest.mark.asyncio
    async def test_upsert_deduplicates_by_chunk_id(self):
        store = InMemoryVectorStore()
        doc = _make_embedded("c1")
        await store.upsert([doc])
        await store.upsert([doc])
        assert len(store) == 1

    @pytest.mark.asyncio
    async def test_similarity_search_returns_k(self):
        store = InMemoryVectorStore()
        docs = [_make_embedded(f"c{i}", embedding=[float(i) / 10] * 8) for i in range(5)]
        await store.upsert(docs)
        results = await store.similarity_search(query_vector=[0.2] * 8, k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_similarity_search_orders_by_cosine(self):
        store = InMemoryVectorStore()
        # c0: 쿼리와 동일 벡터 → 유사도 1.0
        # c1: 직교 벡터 → 유사도 낮음
        q = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        doc_close = _make_embedded("close", embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        doc_far = _make_embedded("far", embedding=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        await store.upsert([doc_close, doc_far])
        results = await store.similarity_search(query_vector=q, k=2)
        # similarity_search는 SearchResult를 반환한다 (VectorStorePort 변경 반영)
        assert results[0].document.chunk_id == "close"

    @pytest.mark.asyncio
    async def test_source_type_filter(self):
        store = InMemoryVectorStore()
        paper_doc = EmbeddedDocument(
            chunk_id="p1", source_id="s1", source_type="paper",
            content="paper", embedding=[0.5] * 8, chunk_index=0,
        )
        blog_doc = EmbeddedDocument(
            chunk_id="b1", source_id="s2", source_type="blog",
            content="blog", embedding=[0.5] * 8, chunk_index=0,
        )
        await store.upsert([paper_doc, blog_doc])
        results = await store.similarity_search(query_vector=[0.5] * 8, k=10, source_type_filter="paper")
        # SearchResult.document.source_type 로 접근한다
        assert all(r.document.source_type == "paper" for r in results)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_delete_by_source(self):
        store = InMemoryVectorStore()
        docs = [_make_embedded(f"c{i}", source_id="target") for i in range(3)]
        docs.append(_make_embedded("other", source_id="keep"))
        await store.upsert(docs)
        removed = await store.delete_by_source("target")
        assert removed == 3
        assert len(store) == 1

    @pytest.mark.asyncio
    async def test_similarity_search_empty_store(self):
        store = InMemoryVectorStore()
        results = await store.similarity_search([0.5] * 8, k=5)
        assert results == []


# ──────────────────────────────────────────────
# IngestionPipeline 통합 테스트
# ──────────────────────────────────────────────

class TestIngestionPipeline:
    def _make_pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            ingestor=DocumentIngestor(),
            chunker=SemanticChunker(chunk_size=300),
            embedder=HashEmbedder(dims=8),
            vector_store=InMemoryVectorStore(),
        )

    @pytest.mark.asyncio
    async def test_run_returns_embedded_docs(self):
        pipeline = self._make_pipeline()
        docs = [_make_doc(content="Paragraph one.\n\nParagraph two.\n\nParagraph three.")]
        embedded = await pipeline.run(docs)
        assert len(embedded) >= 1
        assert all(isinstance(e, EmbeddedDocument) for e in embedded)

    @pytest.mark.asyncio
    async def test_run_embedding_dims_consistent(self):
        pipeline = self._make_pipeline()
        docs = [_make_doc()]
        embedded = await pipeline.run(docs)
        dims = len(embedded[0].embedding)
        assert all(len(e.embedding) == dims for e in embedded)

    @pytest.mark.asyncio
    async def test_run_multiple_docs(self):
        pipeline = self._make_pipeline()
        docs = [_make_doc(source_id=f"doc-{i}", content=f"Content {i}.") for i in range(3)]
        embedded = await pipeline.run(docs)
        source_ids = {e.source_id for e in embedded}
        assert source_ids == {"doc-0", "doc-1", "doc-2"}

    @pytest.mark.asyncio
    async def test_run_empty_docs(self):
        pipeline = self._make_pipeline()
        embedded = await pipeline.run([])
        assert embedded == []

    @pytest.mark.asyncio
    async def test_delete_source_removes_chunks(self):
        store = InMemoryVectorStore()
        pipeline = IngestionPipeline(
            ingestor=DocumentIngestor(),
            chunker=SemanticChunker(chunk_size=300),
            embedder=HashEmbedder(dims=8),
            vector_store=store,
        )
        docs = [_make_doc(source_id="to-delete", content="Some content to index.")]
        await pipeline.run(docs)
        assert len(store) > 0
        await pipeline.delete_source("to-delete")
        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_chunk_ids_format(self):
        pipeline = self._make_pipeline()
        docs = [_make_doc(source_id="src-abc")]
        embedded = await pipeline.run(docs)
        for e in embedded:
            assert e.chunk_id.startswith("src-abc:chunk")
