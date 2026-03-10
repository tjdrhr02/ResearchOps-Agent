from src.domain.ports.metrics_port import MetricsPort
from src.infrastructure.llm.langchain_client import LangChainClient, LangChainRunnable
from src.infrastructure.llm.model_router import ModelRouter
from src.infrastructure.settings.config import Settings


def create_llm_client(
    models: dict[str, LangChainRunnable],
    settings: Settings,
    metrics: MetricsPort | None = None,
) -> LangChainClient:
    router = ModelRouter(
        default_model=settings.llm_default_model,
        task_model_map={
            "planner": settings.llm_planner_model,
            "synthesizer": settings.llm_synthesizer_model,
        },
    )
    return LangChainClient(models=models, router=router, metrics=metrics)
