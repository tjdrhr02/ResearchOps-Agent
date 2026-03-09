from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.source_document import SourceDocument
from src.domain.ports.retriever_port import RetrieverPort
from src.rag.chunking.chunker import SimpleChunker
from src.rag.citation.citation_builder import CitationBuilder
from src.rag.embedding.embedder import HashEmbedder
from src.rag.ingestion.ingestor import DocumentIngestor
from src.rag.vectorstore.pgvector_store import InMemoryPgVectorStore


class Retriever(RetrieverPort):
    def __init__(
        self,
        ingestor: DocumentIngestor,
        chunker: SimpleChunker,
        embedder: HashEmbedder,
        vector_store: InMemoryPgVectorStore,
        citation_builder: CitationBuilder,
    ) -> None:
        self.ingestor = ingestor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.citation_builder = citation_builder
        self._source_by_id: dict[str, SourceDocument] = {}

    async def index_documents(self, docs: list[SourceDocument]) -> int:
        ingested = await self.ingestor.ingest(docs)
        for doc in ingested:
            self._source_by_id[doc.source_id] = doc
        chunks = await self.chunker.chunk(ingested)
        rows: list[dict] = []
        for chunk in chunks:
            embedding = await self.embedder.embed(chunk["content"])
            rows.append({**chunk, "embedding": embedding})
        return await self.vector_store.upsert(rows)

    async def retrieve(self, query: str, k: int = 5) -> list[EvidenceChunk]:
        query_vector = await self.embedder.embed(query)
        rows = await self.vector_store.similarity_search(query_vector=query_vector, k=k)
        results: list[EvidenceChunk] = []
        for row in rows:
            source = self._source_by_id.get(row["source_id"])
            citation = None
            if source:
                citation = self.citation_builder.build(
                    title=source.title,
                    url=source.url,
                    source_type=source.source_type,
                )
            results.append(
                EvidenceChunk(
                    chunk_id=row["chunk_id"],
                    source_id=row["source_id"],
                    content=row["content"],
                    score=1.0,
                    citation=citation,
                )
            )
        return results
