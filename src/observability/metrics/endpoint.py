from fastapi import APIRouter, Depends

from src.api.dependencies import get_metrics_collector
from src.domain.ports.metrics_port import MetricsPort

router = APIRouter()


@router.get("/metrics")
def metrics(metrics_collector: MetricsPort = Depends(get_metrics_collector)) -> dict[str, float]:
    return metrics_collector.snapshot()
