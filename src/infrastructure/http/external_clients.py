from dataclasses import dataclass


@dataclass
class ExternalHttpClient:
    timeout_seconds: float = 5.0

    async def get(self, url: str) -> dict:
        return {"url": url, "status": "ok"}
