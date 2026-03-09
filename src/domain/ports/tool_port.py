from abc import ABC, abstractmethod
from typing import Any


class ToolPort(ABC):
    name: str

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
