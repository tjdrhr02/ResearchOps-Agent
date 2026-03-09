from functools import lru_cache

from src.application.orchestrators.research_orchestrator import ResearchOrchestrator
from src.application.services.note_service import NoteService
from src.application.services.research_workflow_service import ResearchWorkflowService
from src.domain.ports.metrics_port import MetricsPort
from src.infrastructure.cache.redis_client import InMemoryRedisClient
from src.observability.metrics.collector import MetricsCollector
from src.rag.chunking.chunker import SimpleChunker
from src.rag.citation.citation_builder import CitationBuilder
from src.rag.embedding.embedder import HashEmbedder
from src.rag.ingestion.ingestor import DocumentIngestor
from src.rag.retrieval.retriever import Retriever
from src.rag.vectorstore.pgvector_store import InMemoryPgVectorStore
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

    vector_store = InMemoryPgVectorStore()
    retriever = Retriever(
        ingestor=DocumentIngestor(),
        chunker=SimpleChunker(),
        embedder=HashEmbedder(),
        vector_store=vector_store,
        citation_builder=CitationBuilder(),
    )

    workflow = ResearchWorkflowService(
        tools=tools,
        retriever=retriever,
        metrics=get_metrics_collector(),
    )
    return ResearchOrchestrator(workflow=workflow, metrics=get_metrics_collector())


@lru_cache
def get_note_service() -> NoteService:
    note_store = get_note_store()
    return NoteService(
        save_tool=SaveResearchNoteTool(note_store=note_store),
        search_tool=SearchSavedNotesTool(note_store=note_store),
    )
