import logging
import time
from typing import Any, Protocol

from src.domain.ports.metrics_port import MetricsPort
from src.infrastructure.llm.model_router import ModelRouter
from src.infrastructure.llm.token_tracker import TokenUsageTracker
from src.infrastructure.llm.types import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class LangChainRunnable(Protocol):
    async def ainvoke(self, input_data: Any) -> Any:
        raise NotImplementedError


class LangChainClient:
    """
    공통 LLM 클라이언트.
    - LangChain runnable model 호출
    - task_type 기반 model routing
    - token usage tracking
    - structured logging
    """

    def __init__(
        self,
        models: dict[str, LangChainRunnable],
        router: ModelRouter,
        metrics: MetricsPort | None = None,
    ) -> None:
        self.models = models
        self.router = router
        self.metrics = metrics
        self.token_tracker = TokenUsageTracker(metrics=metrics)

    async def invoke(self, request: LLMRequest) -> LLMResponse:
        model_name = self.router.route(
            task_type=request.task_type,
            preferred_model=request.preferred_model,
        )
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Unknown model selected by router: {model_name}")

        input_payload = self._build_input_payload(request=request)
        start = time.perf_counter()
        raw_output = await model.ainvoke(input_payload)
        latency_ms = (time.perf_counter() - start) * 1000

        text = self._extract_text(raw_output)
        prompt_tokens, completion_tokens, total_tokens = self._extract_or_estimate_usage(
            raw_output=raw_output,
            prompt=request.prompt,
            completion=text,
        )

        response = LLMResponse(
            text=text,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )
        self.token_tracker.track(response)
        if self.metrics:
            self.metrics.observe_latency("llm_latency_ms", latency_ms)

        logger.info(
            "llm_invoke_success model=%s task=%s total_tokens=%s latency_ms=%.2f",
            model_name,
            request.task_type,
            total_tokens,
            latency_ms,
        )
        return response

    def _build_input_payload(self, request: LLMRequest) -> Any:
        # LangChain chat model에서 일반적으로 지원하는 message 형태를 사용한다.
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def _extract_text(self, raw_output: Any) -> str:
        if hasattr(raw_output, "content"):
            content = getattr(raw_output, "content")
            if isinstance(content, str):
                return content
        if isinstance(raw_output, str):
            return raw_output
        return str(raw_output)

    def _extract_or_estimate_usage(
        self,
        raw_output: Any,
        prompt: str,
        completion: str,
    ) -> tuple[int, int, int]:
        response_metadata = getattr(raw_output, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage", {})
        prompt_tokens = int(token_usage.get("prompt_tokens", 0))
        completion_tokens = int(token_usage.get("completion_tokens", 0))
        total_tokens = int(token_usage.get("total_tokens", 0))

        if total_tokens > 0:
            return prompt_tokens, completion_tokens, total_tokens

        # usage 정보가 없는 모델을 위한 보수적 추정치 (약 4 chars/token)
        estimated_prompt = max(1, len(prompt) // 4)
        estimated_completion = max(1, len(completion) // 4)
        estimated_total = estimated_prompt + estimated_completion
        return estimated_prompt, estimated_completion, estimated_total
