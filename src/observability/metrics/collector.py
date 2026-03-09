from collections import defaultdict

from src.domain.ports.metrics_port import MetricsPort


class MetricsCollector(MetricsPort):
    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.latency_sums: dict[str, float] = defaultdict(float)
        self.latency_counts: dict[str, float] = defaultdict(float)

    def increment(self, metric_name: str, value: int = 1) -> None:
        self.counters[metric_name] += value

    def observe_latency(self, metric_name: str, milliseconds: float) -> None:
        self.latency_sums[metric_name] += milliseconds
        self.latency_counts[metric_name] += 1

    def snapshot(self) -> dict[str, float]:
        data = dict(self.counters)
        for key, total in self.latency_sums.items():
            count = self.latency_counts[key]
            data[f"{key}_avg"] = total / count if count else 0.0
        return data
