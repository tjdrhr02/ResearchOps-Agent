"""
의존성 주입 팩토리.

FastAPI의 Depends() 패턴과 @lru_cache를 결합하여
싱글톤 의존성을 제공한다.

컴포넌트 연결 순서:
  InMemoryVectorStore
    └─ IngestionPipeline (Ingestor + Chunker + Embedder + VectorStore)
        └─ Retriever (Pipeline + Embedder + VectorStore + CitationBuilder)
            └─ ResearchWorkflowService
                └─ ResearchOrchestrator
"""
import logging
from functools import lru_cache

from src.infrastructure.settings.config import get_settings

logger = logging.getLogger(__name__)

from src.application.orchestrators.research_orchestrator import ResearchOrchestrator
from src.application.services.note_service import NoteService
from src.application.services.research_workflow_service import ResearchWorkflowService
from src.domain.ports.metrics_port import MetricsPort
from src.infrastructure.cache.redis_client import InMemoryRedisClient
from src.observability.metrics.collector import MetricsCollector
from src.rag.chunking.chunker import SemanticChunker
from src.rag.citation.citation_builder import CitationBuilder
from src.rag.embedding.embedder import HashEmbedder, LocalEmbedder
from src.rag.ingestion.ingestion_pipeline import IngestionPipeline
from src.rag.ingestion.ingestor import DocumentIngestor
from src.rag.retrieval.retriever import Retriever
from src.rag.vectorstore.in_memory_store import InMemoryVectorStore
from src.tools.implementations.fetch_article_content_tool import FetchArticleContentTool
from src.tools.implementations.save_research_note_tool import SaveResearchNoteTool
from src.tools.implementations.search_news_tool import SearchNewsTool
from src.tools.implementations.search_papers_tool import SearchPapersTool
from src.tools.implementations.search_saved_notes_tool import SearchSavedNotesTool
from src.tools.implementations.search_tech_blogs_tool import SearchTechBlogsTool


@lru_cache
def get_note_store() -> InMemoryRedisClient:
    return InMemoryRedisClient()


@lru_cache
def get_metrics_collector() -> MetricsPort:
    return MetricsCollector()


@lru_cache
def get_embedder():
    """
    LocalEmbedder(sentence-transformers, 384차원, API 키 불필요).
    sentence-transformers 미설치 시 HashEmbedder(개발용 stub)로 폴백.
    """
    try:
        return LocalEmbedder()
    except Exception as e:
        logger.warning("local_embedder_unavailable error=%s — falling back to HashEmbedder", e)
        return HashEmbedder()


@lru_cache
def get_vector_store() -> InMemoryVectorStore:
    """
    벡터 저장소. 운영 환경에서는 PgVectorStore(pool)로 교체한다.
    """
    return InMemoryVectorStore()


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        ingestor=DocumentIngestor(),
        chunker=SemanticChunker(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        metrics=get_metrics_collector(),
    )


def _build_chat_llm(model: str):
    """
    GOOGLE_API_KEY가 설정된 경우 ChatGoogleGenerativeAI 인스턴스를 반환한다.
    키가 없으면 None을 반환해 각 에이전트의 폴백 모드가 작동한다.
    """
    settings = get_settings()
    if not settings.google_api_key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0.3, google_api_key=settings.google_api_key)


@lru_cache
def get_research_orchestrator() -> ResearchOrchestrator:
    note_store = get_note_store()
    tools = [
        SearchPapersTool(),
        SearchTechBlogsTool(),
        SearchNewsTool(),
        FetchArticleContentTool(),
        SaveResearchNoteTool(note_store=note_store),
        SearchSavedNotesTool(note_store=note_store),
    ]

    retriever = Retriever(
        pipeline=get_ingestion_pipeline(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        citation_builder=CitationBuilder(),
    )

    settings = get_settings()
    model = settings.llm_model
    logger.info("llm_model=%s", model)
    workflow = ResearchWorkflowService(
        tools=tools,
        retriever=retriever,
        metrics=get_metrics_collector(),
        planner_llm=_build_chat_llm(model),
        synthesizer_llm=_build_chat_llm(model),
        reporter_llm=_build_chat_llm(model),
    )
    return ResearchOrchestrator(workflow=workflow, metrics=get_metrics_collector())


@lru_cache
def get_note_service() -> NoteService:
    note_store = get_note_store()
    return NoteService(
        save_tool=SaveResearchNoteTool(note_store=note_store),
        search_tool=SearchSavedNotesTool(note_store=note_store),
    )
