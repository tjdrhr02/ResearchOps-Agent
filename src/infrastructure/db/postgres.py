from dataclasses import dataclass


@dataclass
class PostgresClient:
    dsn: str

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None
