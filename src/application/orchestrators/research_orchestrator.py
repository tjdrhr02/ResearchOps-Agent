"""
ResearchOrchestrator.

ResearchOps Agent 워크플로우의 최상위 제어 컴포넌트.

역할:
  - ResearchJob 생명주기 관리 (pending → running → completed/failed)
  - ResearchWorkflowService 호출 및 결과 매핑
  - 전체 요청 latency 측정 및 메트릭 기록
  - Job 저장소 관리 (인메모리, asyncio.Lock으로 동시성 안전)
  - 최대 저장 Job 수 제한 (MAX_JOBS, FIFO 방식 제거)

API Layer는 Orchestrator를 통해서만 워크플로우에 접근한다.
WorkflowService의 내부 단계(Planner/Collector/RAG/Synthesizer/Reporter)는
Orchestrator에서 직접 참조하지 않는다.
"""
import asyncio
import logging
import time
from collections import OrderedDict
from uuid import uuid4

from src.application.dto.agent_io import WorkflowResult
from src.application.services.research_workflow_service import ResearchWorkflowService
from src.domain.errors.exceptions import JobNotFoundError, PlannerError, ReporterError, WorkflowError
from src.domain.models.research_job import ResearchJob
from src.domain.ports.metrics_port import MetricsPort

logger = logging.getLogger(__name__)

MAX_JOBS = 1000  # 인메모리 최대 Job 보관 수


class ResearchOrchestrator:
    """
    워크플로우 실행 및 Job 관리 컴포넌트.

    FastAPI 의존성 주입으로 싱글톤 인스턴스를 사용하므로
    동시 요청 시 _jobs 딕셔너리를 asyncio.Lock으로 보호한다.
    """

    def __init__(self, workflow: ResearchWorkflowService, metrics: MetricsPort) -> None:
        self.workflow = workflow
        self.metrics = metrics
        self._jobs: OrderedDict[str, ResearchJob] = OrderedDict()
        self._lock = asyncio.Lock()

    # ──────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ──────────────────────────────────────────────────────────────

    async def submit(self, user_query: str, max_sources: int) -> ResearchJob:
        """
        연구 작업을 백그라운드로 제출하고 pending 상태의 Job을 즉시 반환한다.

        클라이언트는 반환된 job_id로 GET /research/{job_id}를 polling해
        completed 또는 failed 상태가 될 때까지 결과를 확인한다.
        """
        job = ResearchJob(job_id=str(uuid4()), query=user_query)
        await self._store_job(job)

        logger.info(
            "orchestrator_submit job_id=%s query=%r max_sources=%s",
            job.job_id,
            user_query[:60],
            max_sources,
        )

        asyncio.create_task(
            self._run_workflow(job=job, user_query=user_query, max_sources=max_sources),
            name=f"workflow-{job.job_id}",
        )
        return job

    async def _run_workflow(self, job: ResearchJob, user_query: str, max_sources: int) -> None:
        """백그라운드 태스크: 워크플로우를 실행하고 job 상태를 갱신한다."""
        t0 = time.perf_counter()
        job.mark_running()

        try:
            result: WorkflowResult = await self.workflow.run(
                user_query=user_query,
                max_sources=max_sources,
            )
            elapsed_ms = _ms(t0)
            job.mark_completed(
                brief=result.brief,
                sources=result.sources,
                elapsed_ms=elapsed_ms,
            )
            for step in result.trace.steps:
                job.add_step(
                    step=step.step,
                    status=step.status,
                    elapsed_ms=step.elapsed_ms,
                    detail=step.detail,
                    error=step.error,
                )
            self.metrics.observe_latency("request_latency_ms", elapsed_ms)
            logger.info(
                "orchestrator_complete job_id=%s elapsed_ms=%.1f trace=[%s]",
                job.job_id,
                elapsed_ms,
                result.trace.summary(),
            )
        except Exception as exc:
            elapsed_ms = _ms(t0)
            job.mark_failed(error_message=str(exc), elapsed_ms=elapsed_ms)
            self.metrics.increment("failed_jobs")
            self.metrics.observe_latency("request_latency_ms", elapsed_ms)
            logger.error(
                "orchestrator_failed job_id=%s error=%s elapsed_ms=%.1f",
                job.job_id,
                exc,
                elapsed_ms,
            )

    async def run(self, user_query: str, max_sources: int) -> ResearchJob:
        """
        새 연구 작업을 생성하고 전체 워크플로우를 실행한다.

        1. pending 상태의 ResearchJob 생성 및 저장
        2. running으로 전이 후 WorkflowService 실행
        3. 성공 → completed, 실패 → failed 로 전이
        4. 전체 요청 latency 기록

        Returns:
            완료(또는 실패) 상태의 ResearchJob

        Raises:
            PlannerError: Planner Agent 실패 (plan 생성 불가)
            ReporterError: Reporter Agent 실패 (brief 생성 불가)
            WorkflowError: 기타 워크플로우 단계 오류
        """
        job = ResearchJob(job_id=str(uuid4()), query=user_query)
        await self._store_job(job)

        logger.info(
            "orchestrator_start job_id=%s query=%r max_sources=%s",
            job.job_id,
            user_query[:60],
            max_sources,
        )

        t0 = time.perf_counter()
        job.mark_running()

        try:
            result: WorkflowResult = await self.workflow.run(
                user_query=user_query,
                max_sources=max_sources,
            )
            elapsed_ms = _ms(t0)

            job.mark_completed(
                brief=result.brief,
                sources=result.sources,
                elapsed_ms=elapsed_ms,
            )
            # WorkflowTrace의 각 단계 결과를 Job에 복사
            for step in result.trace.steps:
                job.add_step(
                    step=step.step,
                    status=step.status,
                    elapsed_ms=step.elapsed_ms,
                    detail=step.detail,
                    error=step.error,
                )

            self.metrics.observe_latency("request_latency_ms", elapsed_ms)
            logger.info(
                "orchestrator_complete job_id=%s elapsed_ms=%.1f trace=[%s]",
                job.job_id,
                elapsed_ms,
                result.trace.summary(),
            )
            return job

        except (PlannerError, ReporterError, WorkflowError) as exc:
            elapsed_ms = _ms(t0)
            job.mark_failed(error_message=str(exc), elapsed_ms=elapsed_ms)
            self.metrics.increment("failed_jobs")
            self.metrics.observe_latency("request_latency_ms", elapsed_ms)
            logger.error(
                "orchestrator_failed job_id=%s step=%s error=%s elapsed_ms=%.1f",
                job.job_id,
                getattr(exc, "step", "unknown"),
                exc,
                elapsed_ms,
            )
            raise

        except Exception as exc:
            elapsed_ms = _ms(t0)
            job.mark_failed(error_message=str(exc), elapsed_ms=elapsed_ms)
            self.metrics.increment("failed_jobs")
            self.metrics.observe_latency("request_latency_ms", elapsed_ms)
            logger.error(
                "orchestrator_unexpected_error job_id=%s error=%s elapsed_ms=%.1f",
                job.job_id,
                exc,
                elapsed_ms,
            )
            raise

    async def get_job(self, job_id: str) -> ResearchJob:
        """
        job_id로 ResearchJob을 조회한다.

        Raises:
            JobNotFoundError: 존재하지 않는 job_id
        """
        async with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def get_job_sync(self, job_id: str) -> ResearchJob | None:
        """동기 컨텍스트(라우터 등)에서 job을 조회한다. 없으면 None 반환."""
        return self._jobs.get(job_id)

    async def list_jobs(
        self,
        status_filter: str | None = None,
        limit: int = 20,
    ) -> list[ResearchJob]:
        """
        저장된 Job 목록을 최신순으로 반환한다.

        Args:
            status_filter: "pending" | "running" | "completed" | "failed" 중 하나
            limit: 최대 반환 수
        """
        async with self._lock:
            jobs = list(reversed(list(self._jobs.values())))

        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]
        return jobs[:limit]

    async def job_count(self) -> int:
        async with self._lock:
            return len(self._jobs)

    # ──────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────

    async def _store_job(self, job: ResearchJob) -> None:
        async with self._lock:
            if len(self._jobs) >= MAX_JOBS:
                # FIFO: 가장 오래된 완료/실패 job 제거
                removed = self._evict_oldest_terminal_job()
                if removed:
                    logger.info("orchestrator_evict job_id=%s", removed)

            self._jobs[job.job_id] = job

    def _evict_oldest_terminal_job(self) -> str | None:
        """완료 또는 실패 상태인 가장 오래된 Job을 제거하고 그 job_id를 반환한다."""
        for job_id, job in self._jobs.items():
            if job.status in ("completed", "failed"):
                del self._jobs[job_id]
                return job_id
        return None


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
