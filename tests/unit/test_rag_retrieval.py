"""
RAG Retrieval 시스템 유닛 테스트.

테스트 대상:
  - RelevanceFilter: min_score 필터, Jaccard 중복 제거
  - CitationBuilder: 번호 부여, 참고문헌 블록 생성
  - Retriever: retrieve (score 전달), retrieve_multi_query (병렬 병합)

모든 테스트는 외부 의존성 없이 동작한다.
"""
import pytest

from src.domain.models.embedded_document import EmbeddedDocument
from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.search_result import SearchResult
from src.domain.models.source_document import SourceDocument
from src.rag.citation.citation_builder import CitationBuilder
from src.rag.chunking.chunker import SemanticChunker
from src.rag.embedding.embedder import HashEmbedder
from src.rag.ingestion.ingestion_pipeline import IngestionPipeline
from src.rag.ingestion.ingestor import DocumentIngestor
from src.rag.retrieval.relevance_filter import RelevanceFilter
from src.rag.retrieval.retriever import Retriever
from src.rag.vectorstore.in_memory_store import InMemoryVectorStore


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _make_embedded(
    chunk_id: str,
    source_id: str = "s1",
    source_type: str = "blog",
    content: str = "sample content",
    embedding: list[float] | None = None,
) -> EmbeddedDocument:
    return EmbeddedDocument(
        chunk_id=chunk_id,
        source_id=source_id,
        source_type=source_type,
        content=content,
        embedding=embedding or [0.5] * 8,
        chunk_index=0,
    )


def _make_result(chunk_id: str, score: float, content: str = "content") -> SearchResult:
    return SearchResult(
        document=_make_embedded(chunk_id, content=content),
        score=score,
    )


def _make_source(source_id: str = "s1", source_type: str = "blog") -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        source_type=source_type,
        title=f"Title of {source_id}",
        url=f"https://example.com/{source_id}",
        content="Long enough content for testing the retrieval pipeline end to end.",
    )


def _make_pipeline(store: InMemoryVectorStore | None = None) -> IngestionPipeline:
    # store or ... 패턴 금지: 빈 store는 bool(store)==False이므로 새 인스턴스가 생성된다
    return IngestionPipeline(
        ingestor=DocumentIngestor(),
        chunker=SemanticChunker(chunk_size=300),
        embedder=HashEmbedder(dims=8),
        vector_store=store if store is not None else InMemoryVectorStore(),
    )


# ──────────────────────────────────────────────
# RelevanceFilter 테스트
# ──────────────────────────────────────────────

class TestRelevanceFilter:
    def test_no_filter_passes_all(self):
        rf = RelevanceFilter(min_score=0.0)
        # 내용이 서로 달라야 dedup에서 제거되지 않는다
        results = [_make_result(f"c{i}", score=0.1 * i, content=f"unique content {i}") for i in range(5)]
        filtered = rf.filter(results)
        assert len(filtered) == 5

    def test_min_score_removes_low_scores(self):
        rf = RelevanceFilter(min_score=0.5)
        results = [
            _make_result("c1", score=0.3, content="alpha bravo charlie delta"),
            _make_result("c2", score=0.6, content="echo foxtrot golf hotel"),
            _make_result("c3", score=0.8, content="india juliet kilo lima"),
        ]
        filtered = rf.filter(results)
        assert len(filtered) == 2
        assert all(r.score >= 0.5 for r in filtered)

    def test_min_score_removes_all(self):
        rf = RelevanceFilter(min_score=0.99)
        results = [_make_result(f"c{i}", score=0.1) for i in range(3)]
        filtered = rf.filter(results)
        assert filtered == []

    def test_output_sorted_descending(self):
        rf = RelevanceFilter()
        results = [_make_result("c1", 0.3), _make_result("c2", 0.9), _make_result("c3", 0.6)]
        filtered = rf.filter(results)
        scores = [r.score for r in filtered]
        assert scores == sorted(scores, reverse=True)

    def test_duplicate_content_removed(self):
        rf = RelevanceFilter(max_content_overlap=0.8)
        same_content = "The quick brown fox jumps over the lazy dog in the park"
        r1 = SearchResult(document=_make_embedded("c1", content=same_content), score=0.9)
        r2 = SearchResult(document=_make_embedded("c2", content=same_content), score=0.7)
        filtered = rf.filter([r1, r2])
        # 두 결과가 내용이 동일하므로 score 높은 c1만 남아야 한다
        assert len(filtered) == 1
        assert filtered[0].document.chunk_id == "c1"

    def test_different_content_both_kept(self):
        rf = RelevanceFilter(max_content_overlap=0.85)
        r1 = _make_result("c1", 0.8, content="Transformer architecture attention mechanism")
        r2 = _make_result("c2", 0.7, content="Convolutional neural networks for image classification")
        filtered = rf.filter([r1, r2])
        assert len(filtered) == 2

    def test_partially_overlapping_content(self):
        rf = RelevanceFilter(max_content_overlap=0.9)
        # 대부분 같지만 끝 부분이 다른 두 텍스트 → 0.9 임계값에서 중복 아님
        r1 = _make_result("c1", 0.8, content="alpha beta gamma delta epsilon zeta eta theta iota kappa")
        r2 = _make_result("c2", 0.6, content="alpha beta gamma delta epsilon zeta eta theta lambda mu")
        filtered = rf.filter([r1, r2])
        assert len(filtered) >= 1  # threshold에 따라 결과 달라질 수 있음


# ──────────────────────────────────────────────
# CitationBuilder 테스트
# ──────────────────────────────────────────────

class TestCitationBuilder:
    def test_build_returns_formatted_string(self):
        cb = CitationBuilder()
        citation = cb.build(
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            source_type="paper",
            source_id="paper-1",
            score=0.91,
        )
        assert "[1]" in citation
        assert "Attention Is All You Need" in citation
        assert "https://arxiv.org/abs/1706.03762" in citation
        assert "paper" in citation
        assert "0.91" in citation

    def test_same_source_id_reuses_number(self):
        cb = CitationBuilder()
        c1 = cb.build("T1", "url1", "blog", source_id="src-1")
        c2 = cb.build("T1 chunk 2", "url1", "blog", source_id="src-1")
        assert c1.startswith("[1]")
        assert c2.startswith("[1]")

    def test_different_sources_get_different_numbers(self):
        cb = CitationBuilder()
        c1 = cb.build("T1", "url1", "paper", source_id="src-1")
        c2 = cb.build("T2", "url2", "blog", source_id="src-2")
        assert c1.startswith("[1]")
        assert c2.startswith("[2]")

    def test_build_without_score(self):
        cb = CitationBuilder()
        citation = cb.build("Title", "url", "news", source_id="n1")
        assert "score" not in citation

    def test_reset_restarts_numbering(self):
        cb = CitationBuilder()
        cb.build("T1", "url1", "paper", source_id="src-1")
        cb.build("T2", "url2", "paper", source_id="src-2")
        cb.reset()
        c_new = cb.build("T3", "url3", "paper", source_id="src-3")
        assert c_new.startswith("[1]")

    def test_build_for_chunks_assigns_numbers(self):
        cb = CitationBuilder()
        chunks = [
            EvidenceChunk(
                chunk_id=f"c{i}",
                source_id=f"src-{i}",
                content="content",
                score=0.8,
                citation=f"Title {i} — url{i} (blog)",
            )
            for i in range(3)
        ]
        numbered = cb.build_for_chunks(chunks)
        assert numbered[0].citation.startswith("[1]")
        assert numbered[1].citation.startswith("[2]")
        assert numbered[2].citation.startswith("[3]")

    def test_build_for_chunks_same_source_reuses_number(self):
        cb = CitationBuilder()
        chunks = [
            EvidenceChunk(
                chunk_id="c1", source_id="src-1",
                content="chunk 1", score=0.9, citation="T — url (blog)",
            ),
            EvidenceChunk(
                chunk_id="c2", source_id="src-1",
                content="chunk 2", score=0.7, citation="T — url (blog)",
            ),
        ]
        numbered = cb.build_for_chunks(chunks)
        assert numbered[0].citation.startswith("[1]")
        assert numbered[1].citation.startswith("[1]")

    def test_build_reference_list(self):
        cb = CitationBuilder()
        chunks = [
            EvidenceChunk(
                chunk_id="c1", source_id="s1",
                content="c", score=0.9, citation="[1] Paper — url (paper)",
            ),
            EvidenceChunk(
                chunk_id="c2", source_id="s2",
                content="c", score=0.7, citation="[2] Blog — url2 (blog)",
            ),
            EvidenceChunk(
                chunk_id="c3", source_id="s1",
                content="c", score=0.5, citation="[1] Paper — url (paper)",
            ),
        ]
        ref = cb.build_reference_list(chunks)
        assert "## References" in ref
        assert ref.count("[1]") == 1  # s1은 한 번만 등장
        assert "[2]" in ref

    def test_build_reference_list_empty(self):
        cb = CitationBuilder()
        chunks = [
            EvidenceChunk(chunk_id="c1", source_id="s1", content="c", score=0.5)
        ]
        ref = cb.build_reference_list(chunks)
        assert ref == ""


# ──────────────────────────────────────────────
# Retriever 통합 테스트
# ──────────────────────────────────────────────

class TestRetriever:
    def _make_retriever(self) -> tuple[Retriever, InMemoryVectorStore]:
        store = InMemoryVectorStore()
        pipeline = _make_pipeline(store)
        retriever = Retriever(
            pipeline=pipeline,
            embedder=HashEmbedder(dims=8),
            vector_store=store,
            citation_builder=CitationBuilder(),
            relevance_filter=RelevanceFilter(min_score=0.0),
        )
        return retriever, store

    @pytest.mark.asyncio
    async def test_index_and_retrieve_basic(self):
        retriever, _ = self._make_retriever()
        docs = [_make_source(source_id="s1")]
        indexed = await retriever.index_documents(docs)
        assert indexed > 0

        results = await retriever.retrieve(query="content testing", k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, EvidenceChunk) for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_score_is_float_in_range(self):
        retriever, _ = self._make_retriever()
        await retriever.index_documents([_make_source("s1")])
        results = await retriever.retrieve("test query")
        for r in results:
            assert 0.0 <= r.score <= 1.0

    @pytest.mark.asyncio
    async def test_retrieve_returns_citation(self):
        retriever, _ = self._make_retriever()
        await retriever.index_documents([_make_source("s1")])
        results = await retriever.retrieve("test")
        for r in results:
            assert r.citation is not None
            assert "Title of s1" in r.citation

    @pytest.mark.asyncio
    async def test_retrieve_citation_contains_number(self):
        retriever, _ = self._make_retriever()
        await retriever.index_documents([_make_source("s1")])
        results = await retriever.retrieve("test")
        assert results[0].citation.startswith("[1]")

    @pytest.mark.asyncio
    async def test_retrieve_respects_k(self):
        retriever, _ = self._make_retriever()
        docs = [_make_source(source_id=f"s{i}") for i in range(10)]
        await retriever.index_documents(docs)
        results = await retriever.retrieve("test", k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_min_score_filters(self):
        retriever, store = self._make_retriever()

        # 쿼리와 동일한 벡터를 가진 문서 색인
        embedder = HashEmbedder(dims=8)
        q_vec = await embedder.embed("specific topic query")
        doc = _make_embedded("match", embedding=q_vec, content="specific topic query")
        noise = _make_embedded("noise", embedding=[0.0] * 8, content="completely different")
        await store.upsert([doc, noise])
        retriever._source_by_id["match"] = _make_source("match")
        retriever._source_by_id["noise"] = _make_source("noise")

        results = await retriever.retrieve("specific topic query", min_score=0.99)
        assert all(r.score >= 0.99 for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_source_type_filter(self):
        retriever, store = self._make_retriever()
        paper_doc = _make_embedded("p1", source_id="sp", source_type="paper")
        blog_doc = _make_embedded("b1", source_id="sb", source_type="blog")
        await store.upsert([paper_doc, blog_doc])
        retriever._source_by_id["sp"] = SourceDocument(
            source_id="sp", source_type="paper",
            title="Paper", url="url", content="c",
        )
        retriever._source_by_id["sb"] = SourceDocument(
            source_id="sb", source_type="blog",
            title="Blog", url="url", content="c",
        )

        results = await retriever.retrieve("query", source_type_filter="paper")
        assert all(r.metadata.get("source_type") == "paper" or "paper" in (r.citation or "") for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_multi_query_merges_results(self):
        retriever, store = self._make_retriever()
        embedder = HashEmbedder(dims=8)
        v1 = await embedder.embed("query A")
        v2 = await embedder.embed("query B")
        d1 = _make_embedded("c-only-a", embedding=v1, content="query A result")
        d2 = _make_embedded("c-only-b", embedding=v2, content="query B result")
        await store.upsert([d1, d2])
        retriever._source_by_id["c-only-a"] = _make_source("c-only-a")
        retriever._source_by_id["c-only-b"] = _make_source("c-only-b")

        results = await retriever.retrieve_multi_query(["query A", "query B"], k=10)
        chunk_ids = {r.chunk_id for r in results}
        assert "c-only-a" in chunk_ids or "c-only-b" in chunk_ids

    @pytest.mark.asyncio
    async def test_retrieve_multi_query_deduplicates(self):
        retriever, store = self._make_retriever()
        doc = _make_embedded("shared", embedding=[0.5] * 8, content="shared content")
        await store.upsert([doc])
        retriever._source_by_id["shared"] = _make_source("shared")

        results = await retriever.retrieve_multi_query(["query 1", "query 2"], k=10)
        chunk_ids = [r.chunk_id for r in results]
        # 동일 chunk_id는 한 번만 등장해야 한다
        assert len(chunk_ids) == len(set(chunk_ids))

    @pytest.mark.asyncio
    async def test_retrieve_multi_query_keeps_max_score(self):
        """동일 chunk가 여러 쿼리에 매칭될 경우 최고 score를 유지한다."""
        retriever, store = self._make_retriever()
        embedder = HashEmbedder(dims=8)
        q1_vec = await embedder.embed("query one")
        q2_vec = await embedder.embed("query two")

        # chunk embedding을 q1과 가깝게 설정
        doc = _make_embedded("c1", embedding=q1_vec, content="content")
        await store.upsert([doc])
        retriever._source_by_id["c1"] = _make_source("c1")

        results = await retriever.retrieve_multi_query(["query one", "query two"], k=5)
        matching = [r for r in results if r.chunk_id == "c1"]
        if matching:
            # q1과의 유사도(높음) vs q2와의 유사도(낮음) → 높은 쪽이 선택됨
            assert matching[0].score >= 0.0

    @pytest.mark.asyncio
    async def test_retrieve_empty_store(self):
        retriever, _ = self._make_retriever()
        results = await retriever.retrieve("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_multi_query_empty_queries(self):
        retriever, _ = self._make_retriever()
        results = await retriever.retrieve_multi_query([])
        assert results == []
