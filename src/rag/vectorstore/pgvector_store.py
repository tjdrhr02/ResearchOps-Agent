class InMemoryPgVectorStore:
    def __init__(self) -> None:
        self._rows: list[dict] = []

    async def upsert(self, rows: list[dict]) -> int:
        self._rows.extend(rows)
        return len(rows)

    async def similarity_search(self, query_vector: list[float], k: int) -> list[dict]:
        # Placeholder scoring logic for local development.
        sorted_rows = sorted(
            self._rows,
            key=lambda item: abs(item["embedding"][0] - query_vector[0]),
        )
        return sorted_rows[:k]
