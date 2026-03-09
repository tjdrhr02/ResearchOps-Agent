import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from src.domain.errors.exceptions import ToolExecutionError
from src.domain.ports.tool_port import ToolPort

logger = logging.getLogger(__name__)


class ToolContract(ToolPort, ABC):
    name: str = "base_tool"
    timeout_seconds: float = 8.0
    retry_count: int = 2

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_schema(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def _run(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_count + 2):
            try:
                result = await asyncio.wait_for(self._run(payload), timeout=self.timeout_seconds)
                logger.info("tool_success name=%s attempt=%s", self.name, attempt)
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("tool_failure name=%s attempt=%s error=%s", self.name, attempt, exc)
        raise ToolExecutionError(f"Tool failed: {self.name}") from last_error
