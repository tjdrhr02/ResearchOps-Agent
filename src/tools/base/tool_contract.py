"""
ResearchBaseTool: LangChain BaseTool 기반 공통 Tool 계약.

모든 ResearchOps Tool은 이 클래스를 상속한다.

공통 기능
- Pydantic input/output schema 강제
- asyncio.wait_for 기반 timeout
- tenacity 기반 retry (지수 백오프)
- 구조화 logging (tool=, query=, elapsed_ms=)
- LangChain BaseTool과 호환 (ainvoke/arun)
"""
import asyncio
import logging
import time
from abc import abstractmethod
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.domain.errors.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)


class ResearchBaseTool(BaseTool):
    """
    LangChain BaseTool을 상속하는 공통 Tool 추상 클래스.

    하위 클래스는 args_schema, output_schema_cls, _execute 를 구현한다.
    timeout_seconds / retry_count 는 클래스 변수로 재정의 가능하다.
    """

    timeout_seconds: float = 8.0
    retry_count: int = 2

    # ------------------------------------------------------------------
    # 하위 클래스가 구현해야 하는 메서드
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def output_schema(self) -> dict[str, Any]:
        """출력 필드 명세를 딕셔너리로 반환한다."""
        raise NotImplementedError

    @abstractmethod
    async def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """실제 비즈니스 로직. timeout/retry 래퍼 안에서 호출된다."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # LangChain BaseTool 필수 구현
    # ------------------------------------------------------------------

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ResearchBaseTool은 비동기(_arun)만 지원합니다.")

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """LangChain 비동기 체인에서 호출되는 진입점."""
        payload = kwargs if kwargs else (args[0] if args else {})
        if isinstance(payload, BaseModel):
            payload = payload.model_dump()
        result = await self.run(payload=payload)
        return str(result)

    # ------------------------------------------------------------------
    # 내부 공통 run (CollectorAgent 등 직접 호출 경로)
    # ------------------------------------------------------------------

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        """timeout + retry + logging 공통 래퍼."""
        start = time.perf_counter()
        try:
            result = await self._run_with_retry(payload)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "tool_success tool=%s elapsed_ms=%.2f",
                self.name,
                elapsed_ms,
            )
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "tool_failed tool=%s elapsed_ms=%.2f error=%s",
                self.name,
                elapsed_ms,
                exc,
            )
            raise ToolExecutionError(f"Tool failed after retries: {self.name}") from exc

    async def _run_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_count + 2):
            try:
                result = await asyncio.wait_for(
                    self._execute(payload),
                    timeout=self.timeout_seconds,
                )
                if attempt > 1:
                    logger.info(
                        "tool_retry_success tool=%s attempt=%s",
                        self.name,
                        attempt,
                    )
                return result
            except asyncio.TimeoutError as exc:
                last_error = exc
                logger.warning(
                    "tool_timeout tool=%s attempt=%s timeout_seconds=%.1f",
                    self.name,
                    attempt,
                    self.timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "tool_attempt_failed tool=%s attempt=%s error=%s",
                    self.name,
                    attempt,
                    exc,
                )
        raise ToolExecutionError(f"All attempts exhausted: {self.name}") from last_error


# 이전 인터페이스와 호환을 위한 alias
ToolContract = ResearchBaseTool
