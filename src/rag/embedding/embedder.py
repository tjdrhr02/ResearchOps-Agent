class HashEmbedder:
    async def embed(self, text: str) -> list[float]:
        # Lightweight deterministic embedding stub for skeleton stage.
        return [float((sum(ord(ch) for ch in text) % 1000) / 1000.0)]
