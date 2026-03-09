from abc import ABC, abstractmethod


class MetricsPort(ABC):
    @abstractmethod
    def increment(self, metric_name: str, value: int = 1) -> None:
        raise NotImplementedError

    @abstractmethod
    def observe_latency(self, metric_name: str, milliseconds: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> dict[str, float]:
        raise NotImplementedError
