import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

_EXPORT_INTERVAL = 60
_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")

# MetricsCollector key → Cloud Monitoring custom metric type
_METRIC_MAP = {
    "total_requests": "custom.googleapis.com/researchops/total_requests",
    "failed_jobs": "custom.googleapis.com/researchops/failed_jobs",
    "request_latency_ms_avg": "custom.googleapis.com/researchops/request_latency_ms_avg",
}


async def start_export_loop() -> None:
    """MetricsCollector 스냅샷을 60초마다 Cloud Monitoring에 푸시한다."""
    if not _PROJECT_ID:
        logger.warning("GOOGLE_CLOUD_PROJECT unset — Cloud Monitoring export disabled")
        return

    try:
        from google.cloud import monitoring_v3
        from src.api.dependencies import get_metrics_collector

        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{_PROJECT_ID}"
        collector = get_metrics_collector()

        while True:
            await asyncio.sleep(_EXPORT_INTERVAL)
            _push_snapshot(client, project_name, collector.snapshot())
    except ImportError:
        logger.warning("google-cloud-monitoring not installed — Cloud Monitoring export disabled")
    except asyncio.CancelledError:
        pass


def _push_snapshot(client, project_name: str, snapshot: dict[str, float]) -> None:
    from google.cloud import monitoring_v3
    from google.protobuf import timestamp_pb2

    ts = timestamp_pb2.Timestamp()
    ts.seconds = int(time.time())

    interval = monitoring_v3.TimeInterval()
    interval.end_time = ts

    series_list = []
    for key, metric_type in _METRIC_MAP.items():
        value = snapshot.get(key, 0.0)
        series = monitoring_v3.TimeSeries()
        series.metric.type = metric_type
        series.resource.type = "global"
        point = monitoring_v3.Point()
        point.interval = interval
        point.value.double_value = value
        series.points = [point]
        series_list.append(series)

    try:
        client.create_time_series(name=project_name, time_series=series_list)
        logger.debug("Cloud Monitoring: pushed %d metrics", len(series_list))
    except Exception as exc:
        logger.warning("Cloud Monitoring push failed: %s", exc)
