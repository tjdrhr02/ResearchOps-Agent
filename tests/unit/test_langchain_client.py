import pytest

from src.infrastructure.llm.langchain_client import LangChainClient
from src.infrastructure.llm.model_router import ModelRouter
from src.infrastructure.llm.types import LLMRequest
from src.observability.metrics.collector import MetricsCollector


class FakeLangChainMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            }
        }


class FakeLangChainModel:
    async def ainvoke(self, input_data):  # noqa: ANN001
        return FakeLangChainMessage(content="structured response")


@pytest.mark.asyncio
async def test_langchain_client_routes_model_and_tracks_tokens() -> None:
    metrics = MetricsCollector()
    router = ModelRouter(default_model="gpt-4o-mini", task_model_map={"planner": "gpt-4o-mini"})
    client = LangChainClient(
        models={"gpt-4o-mini": FakeLangChainModel()},
        router=router,
        metrics=metrics,
    )

    response = await client.invoke(
        LLMRequest(prompt="Plan research for agentic RAG", task_type="planner"),
    )
    assert response.model_name == "gpt-4o-mini"
    assert response.total_tokens == 20

    snapshot = metrics.snapshot()
    assert snapshot["token_usage"] == 20
    assert snapshot["prompt_token_usage"] == 12
    assert snapshot["completion_token_usage"] == 8
