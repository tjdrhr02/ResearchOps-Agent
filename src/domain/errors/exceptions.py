"""
ResearchOps 커스텀 예외 계층.

워크플로우 단계별로 예외를 분리해 오류 원인을 명확히 추적한다.

계층 구조:
  ResearchOpsError          (기본)
    WorkflowError           (워크플로우 단계 오류 기반)
      PlannerError          (Planner Agent 실패 → 워크플로우 중단)
      CollectorError        (Collector Agent 실패 → 부분 계속 가능)
      RAGError              (RAG 인덱싱/검색 실패 → 빈 evidence로 계속)
      SynthesizerError      (Synthesizer Agent 실패)
      ReporterError         (Reporter Agent 실패 → 워크플로우 중단)
    ToolExecutionError      (Tool 계층 실행 오류)
    RetrievalError          (RAG retrieval 오류)
    JobNotFoundError        (존재하지 않는 job_id 조회)
"""


class ResearchOpsError(Exception):
    """ResearchOps 서비스 기본 예외."""


class WorkflowError(ResearchOpsError):
    """워크플로우 단계 오류 기반 클래스."""

    def __init__(self, step: str, message: str, cause: Exception | None = None) -> None:
        self.step = step
        self.cause = cause
        super().__init__(f"[{step}] {message}")


class PlannerError(WorkflowError):
    """Planner Agent 실패. 복구 불가 — 워크플로우를 중단한다."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(step="planner", message=message, cause=cause)


class CollectorError(WorkflowError):
    """Collector Agent 실패. 부분 복구 가능 — 빈 문서로 계속 진행한다."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(step="collector", message=message, cause=cause)


class RAGError(WorkflowError):
    """RAG 인덱싱 또는 검색 실패. 부분 복구 가능 — 빈 evidence로 계속 진행한다."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(step="rag", message=message, cause=cause)


class SynthesizerError(WorkflowError):
    """Synthesizer Agent 실패."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(step="synthesizer", message=message, cause=cause)


class ReporterError(WorkflowError):
    """Reporter Agent 실패. 복구 불가 — 워크플로우를 중단한다."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(step="reporter", message=message, cause=cause)


class ToolExecutionError(ResearchOpsError):
    """Tool 계층 실행 오류."""


class RetrievalError(ResearchOpsError):
    """RAG retrieval 오류."""


class JobNotFoundError(ResearchOpsError):
    """존재하지 않는 job_id 조회."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Job not found: {job_id}")
