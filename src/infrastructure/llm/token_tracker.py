from src.domain.ports.metrics_port import MetricsPort
from src.infrastructure.llm.types import LLMResponse


class TokenUsageTracker:
    def __init__(self, metrics: MetricsPort | None = None) -> None:
        self.metrics = metrics

    def track(self, response: LLMResponse) -> None:
        if not self.metrics:
            return
        self.metrics.increment("token_usage", response.total_tokens)
        self.metrics.increment("prompt_token_usage", response.prompt_tokens)
        self.metrics.increment("completion_token_usage", response.completion_tokens)
