from src.domain.models.source_document import SourceDocument


class SimpleChunker:
    async def chunk(self, docs: list[SourceDocument], chunk_size: int = 300) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        for doc in docs:
            content = doc.content or doc.title
            for idx in range(0, max(len(content), 1), chunk_size):
                chunks.append(
                    {
                        "chunk_id": f"{doc.source_id}:{idx // chunk_size}",
                        "source_id": doc.source_id,
                        "content": content[idx : idx + chunk_size],
                    }
                )
        return chunks
