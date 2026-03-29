"""
ResearchJob.

연구 작업의 생명주기를 추적하는 도메인 모델.

상태 전이:
  pending → running → completed
                    → failed

status는 Orchestrator가 외부에서 직접 업데이트하지 않고
mark_running / mark_completed / mark_failed 메서드를 통해 변경한다.
"""
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.domain.models.research_brief import ResearchBrief
from src.domain.models.source_document import SourceDocument

_VALID_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"running"},
    "running": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
}


class ResearchJob(BaseModel):
    job_id: str
    query: str
    status: str = "pending"
    brief: ResearchBrief | None = None
    sources: list[SourceDocument] = Field(default_factory=list)
    error_message: str | None = None
    elapsed_ms: float = 0.0
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    completed_at: str | None = None
    step_trace: list[dict] = Field(
        default_factory=list,
        description="각 워크플로우 단계의 실행 결과 목록",
    )

    model_config = {"arbitrary_types_allowed": True}

    def mark_running(self) -> None:
        self._transition("running")

    def mark_completed(
        self,
        brief: ResearchBrief,
        sources: list[SourceDocument],
        elapsed_ms: float,
    ) -> None:
        self._transition("completed")
        self.brief = brief
        self.sources = sources
        self.elapsed_ms = elapsed_ms
        self.completed_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def mark_failed(self, error_message: str, elapsed_ms: float) -> None:
        self._transition("failed")
        self.error_message = error_message
        self.elapsed_ms = elapsed_ms
        self.completed_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def add_step(
        self,
        step: str,
        status: str,
        elapsed_ms: float,
        detail: str = "",
        error: str | None = None,
    ) -> None:
        self.step_trace.append(
            {
                "step": step,
                "status": status,
                "elapsed_ms": round(elapsed_ms, 2),
                "detail": detail,
                "error": error,
            }
        )

    def _transition(self, target: str) -> None:
        allowed = _VALID_TRANSITIONS.get(self.status, set())
        if target not in allowed:
            raise ValueError(
                f"Invalid status transition: {self.status!r} → {target!r}"
            )
        self.status = target
