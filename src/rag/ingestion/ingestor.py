from src.domain.models.source_document import SourceDocument


class DocumentIngestor:
    async def ingest(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        cleaned: list[SourceDocument] = []
        for doc in docs:
            cleaned.append(
                SourceDocument(
                    source_id=doc.source_id,
                    source_type=doc.source_type,
                    title=doc.title.strip(),
                    url=doc.url,
                    content=doc.content.strip(),
                    metadata=doc.metadata,
                )
            )
        return cleaned
