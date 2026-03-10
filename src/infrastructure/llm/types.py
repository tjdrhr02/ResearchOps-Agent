from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    prompt: str
    system_prompt: str | None = None
    task_type: str = "default"
    preferred_model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
