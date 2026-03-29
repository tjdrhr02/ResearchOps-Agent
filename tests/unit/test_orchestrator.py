"""
ResearchOrchestrator 유닛 테스트.

테스트 대상:
  - ResearchJob 상태 전이 (pending → running → completed/failed)
  - WorkflowTrace / StepResult 기록
  - ResearchOrchestrator Job 생명주기 관리
  - ResearchWorkflowService 단계별 오류 격리
  - 커스텀 예외 계층 (PlannerError, CollectorError, ...)

모든 테스트는 외부 LLM 및 DB 없이 동작한다.
"""
import asyncio

import pytest

from src.application.dto.agent_io import (
    CollectorOutput,
    PlannerOutput,
    SynthesizerOutput,
    WorkflowResult,
    WorkflowTrace,
)
from src.application.orchestrators.research_orchestrator import ResearchOrchestrator
from src.domain.errors.exceptions import (
    CollectorError,
    JobNotFoundError,
    PlannerError,
    RAGError,
    ReporterError,
    SynthesizerError,
    WorkflowError,
)
from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.research_brief import BriefMetadata, ResearchBrief
from src.domain.models.research_job import ResearchJob
from src.domain.models.research_plan import ResearchPlan
from src.domain.models.source_document import SourceDocument


# ──────────────────────────────────────────────
# 헬퍼 / 가짜 객체
# ──────────────────────────────────────────────

def _make_brief(summary: str = "Test summary.") -> ResearchBrief:
    return ResearchBrief(
        executive_summary=summary,
        key_trends=["Trend A"],
        evidence=["Evidence snippet."],
        source_comparison=["Papers vs blogs."],
        open_questions=["What next?"],
        citations=["[1] Source"],
        metadata=BriefMetadata(research_question="test", research_type="trend_analysis"),
    )


def _make_workflow_result(brief: ResearchBrief | None = None) -> WorkflowResult:
    b = brief or _make_brief()
    trace = WorkflowTrace()
    for step in ["planner", "collector", "rag_index", "rag_retrieve", "synthesizer", "reporter"]:
        trace.add(step=step, status="success", elapsed_ms=10.0)
    return WorkflowResult(brief=b, sources=[], trace=trace)


class FakeMetrics:
    def __init__(self) -> None:
        self.counters: dict[str, float] = {}
        self.latencies: list[tuple[str, float]] = []

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def observe_latency(self, name: str, ms: float) -> None:
        self.latencies.append((name, ms))

    def snapshot(self) -> dict:
        return dict(self.counters)


class FakeWorkflow:
    """성공하는 가짜 WorkflowService."""

    def __init__(self, result: WorkflowResult | None = None) -> None:
        self._result = result or _make_workflow_result()
        self.call_count = 0

    async def run(self, user_query: str, max_sources: int) -> WorkflowResult:
        self.call_count += 1
        return self._result


class FailingWorkflow:
    """특정 예외를 발생시키는 가짜 WorkflowService."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def run(self, user_query: str, max_sources: int) -> WorkflowResult:
        raise self._exc


def _make_orchestrator(workflow=None, metrics=None) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        workflow=workflow or FakeWorkflow(),
        metrics=metrics or FakeMetrics(),
    )


# ──────────────────────────────────────────────
# ResearchJob 상태 전이 테스트
# ──────────────────────────────────────────────

class TestResearchJobTransitions:
    def _make_job(self) -> ResearchJob:
        return ResearchJob(job_id="j1", query="test query")

    def test_initial_status_is_pending(self):
        job = self._make_job()
        assert job.status == "pending"

    def test_mark_running(self):
        job = self._make_job()
        job.mark_running()
        assert job.status == "running"

    def test_mark_completed(self):
        job = self._make_job()
        job.mark_running()
        job.mark_completed(brief=_make_brief(), sources=[], elapsed_ms=123.4)
        assert job.status == "completed"
        assert job.elapsed_ms == 123.4
        assert job.completed_at is not None

    def test_mark_failed(self):
        job = self._make_job()
        job.mark_running()
        job.mark_failed(error_message="Something went wrong", elapsed_ms=50.0)
        assert job.status == "failed"
        assert job.error_message == "Something went wrong"
        assert job.elapsed_ms == 50.0

    def test_invalid_transition_pending_to_completed(self):
        job = self._make_job()
        with pytest.raises(ValueError, match="Invalid status transition"):
            job.mark_completed(brief=_make_brief(), sources=[], elapsed_ms=0)

    def test_invalid_transition_completed_to_running(self):
        job = self._make_job()
        job.mark_running()
        job.mark_completed(brief=_make_brief(), sources=[], elapsed_ms=0)
        with pytest.raises(ValueError, match="Invalid status transition"):
            job.mark_running()

    def test_add_step_appends_to_trace(self):
        job = self._make_job()
        job.add_step("planner", "success", 12.3, detail="queries=3")
        assert len(job.step_trace) == 1
        assert job.step_trace[0]["step"] == "planner"
        assert job.step_trace[0]["elapsed_ms"] == 12.3

    def test_add_multiple_steps(self):
        job = self._make_job()
        for step in ["planner", "collector", "synthesizer"]:
            job.add_step(step, "success", 10.0)
        assert len(job.step_trace) == 3

    def test_created_at_is_set(self):
        job = self._make_job()
        assert job.created_at != ""
        assert "T" in job.created_at


# ──────────────────────────────────────────────
# WorkflowTrace 테스트
# ──────────────────────────────────────────────

class TestWorkflowTrace:
    def test_add_step(self):
        trace = WorkflowTrace()
        trace.add("planner", "success", 15.0, detail="queries=3")
        assert len(trace.steps) == 1
        assert trace.steps[0].step == "planner"
        assert trace.steps[0].status == "success"

    def test_has_failure_false_when_all_success(self):
        trace = WorkflowTrace()
        trace.add("planner", "success", 10.0)
        trace.add("collector", "success", 20.0)
        assert trace.has_failure() is False

    def test_has_failure_true_when_any_failed(self):
        trace = WorkflowTrace()
        trace.add("planner", "success", 10.0)
        trace.add("collector", "failed", 5.0, error="timeout")
        assert trace.has_failure() is True

    def test_summary_format(self):
        trace = WorkflowTrace()
        trace.add("planner", "success", 10.0)
        trace.add("collector", "partial", 25.0)
        summary = trace.summary()
        assert "planner=success" in summary
        assert "collector=partial" in summary

    def test_summary_empty(self):
        trace = WorkflowTrace()
        assert trace.summary() == ""


# ──────────────────────────────────────────────
# 커스텀 예외 테스트
# ──────────────────────────────────────────────

class TestCustomExceptions:
    def test_planner_error_has_step(self):
        exc = PlannerError("plan failed")
        assert exc.step == "planner"
        assert "planner" in str(exc)

    def test_collector_error_has_step(self):
        exc = CollectorError("collection failed")
        assert exc.step == "collector"

    def test_rag_error_has_step(self):
        exc = RAGError("retrieval failed")
        assert exc.step == "rag"

    def test_synthesizer_error_has_step(self):
        exc = SynthesizerError("synthesis failed")
        assert exc.step == "synthesizer"

    def test_reporter_error_has_step(self):
        exc = ReporterError("brief failed")
        assert exc.step == "reporter"

    def test_workflow_error_preserves_cause(self):
        cause = ValueError("root cause")
        exc = PlannerError("wrapper", cause=cause)
        assert exc.cause is cause

    def test_job_not_found_error(self):
        exc = JobNotFoundError("job-xyz")
        assert "job-xyz" in str(exc)
        assert exc.job_id == "job-xyz"

    def test_workflow_error_is_research_ops_error(self):
        exc = PlannerError("test")
        from src.domain.errors.exceptions import ResearchOpsError
        assert isinstance(exc, ResearchOpsError)


# ──────────────────────────────────────────────
# ResearchOrchestrator 테스트
# ──────────────────────────────────────────────

class TestResearchOrchestrator:
    @pytest.mark.asyncio
    async def test_run_returns_completed_job(self):
        orch = _make_orchestrator()
        job = await orch.run("test query", max_sources=5)
        assert isinstance(job, ResearchJob)
        assert job.status == "completed"

    @pytest.mark.asyncio
    async def test_run_job_has_brief(self):
        orch = _make_orchestrator()
        job = await orch.run("test query", max_sources=5)
        assert job.brief is not None
        assert job.brief.executive_summary != ""

    @pytest.mark.asyncio
    async def test_run_job_has_elapsed_ms(self):
        orch = _make_orchestrator()
        job = await orch.run("test query", max_sources=5)
        assert job.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_run_job_has_step_trace(self):
        orch = _make_orchestrator()
        job = await orch.run("test query", max_sources=5)
        assert len(job.step_trace) > 0
        step_names = [s["step"] for s in job.step_trace]
        assert "planner" in step_names
        assert "reporter" in step_names

    @pytest.mark.asyncio
    async def test_run_job_stored(self):
        orch = _make_orchestrator()
        job = await orch.run("test query", max_sources=5)
        stored = orch.get_job_sync(job.job_id)
        assert stored is not None
        assert stored.job_id == job.job_id

    @pytest.mark.asyncio
    async def test_run_records_latency_metric(self):
        metrics = FakeMetrics()
        orch = _make_orchestrator(metrics=metrics)
        await orch.run("test query", max_sources=5)
        latency_names = [name for name, _ in metrics.latencies]
        assert "request_latency_ms" in latency_names

    @pytest.mark.asyncio
    async def test_planner_error_marks_job_failed(self):
        orch = _make_orchestrator(
            workflow=FailingWorkflow(PlannerError("plan failed"))
        )
        with pytest.raises(PlannerError):
            await orch.run("bad query", max_sources=5)

        jobs = await orch.list_jobs()
        assert jobs[0].status == "failed"
        assert "plan failed" in jobs[0].error_message

    @pytest.mark.asyncio
    async def test_reporter_error_marks_job_failed(self):
        orch = _make_orchestrator(
            workflow=FailingWorkflow(ReporterError("brief failed"))
        )
        with pytest.raises(ReporterError):
            await orch.run("test query", max_sources=5)

        jobs = await orch.list_jobs()
        assert jobs[0].status == "failed"

    @pytest.mark.asyncio
    async def test_failed_job_increments_metric(self):
        metrics = FakeMetrics()
        orch = _make_orchestrator(
            workflow=FailingWorkflow(PlannerError("fail")),
            metrics=metrics,
        )
        with pytest.raises(PlannerError):
            await orch.run("test", max_sources=3)
        assert metrics.counters.get("failed_jobs", 0) >= 1

    @pytest.mark.asyncio
    async def test_get_job_sync_returns_none_for_unknown(self):
        orch = _make_orchestrator()
        assert orch.get_job_sync("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_job_raises_for_unknown(self):
        orch = _make_orchestrator()
        with pytest.raises(JobNotFoundError):
            await orch.get_job("nonexistent-id")

    @pytest.mark.asyncio
    async def test_list_jobs_returns_all(self):
        orch = _make_orchestrator()
        await orch.run("query A", max_sources=3)
        await orch.run("query B", max_sources=3)
        jobs = await orch.list_jobs()
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_list_jobs_status_filter(self):
        orch = _make_orchestrator()
        await orch.run("query A", max_sources=3)
        completed = await orch.list_jobs(status_filter="completed")
        assert all(j.status == "completed" for j in completed)

    @pytest.mark.asyncio
    async def test_list_jobs_limit(self):
        orch = _make_orchestrator()
        for i in range(5):
            await orch.run(f"query {i}", max_sources=3)
        jobs = await orch.list_jobs(limit=3)
        assert len(jobs) == 3

    @pytest.mark.asyncio
    async def test_list_jobs_most_recent_first(self):
        orch = _make_orchestrator()
        await orch.run("first query", max_sources=3)
        await orch.run("second query", max_sources=3)
        jobs = await orch.list_jobs()
        assert jobs[0].query == "second query"

    @pytest.mark.asyncio
    async def test_job_count(self):
        orch = _make_orchestrator()
        assert await orch.job_count() == 0
        await orch.run("query", max_sources=3)
        assert await orch.job_count() == 1

    @pytest.mark.asyncio
    async def test_concurrent_runs_safe(self):
        """동시 요청 시 Job 저장소 동시성 안전 검증."""
        orch = _make_orchestrator()
        tasks = [orch.run(f"query {i}", max_sources=3) for i in range(5)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert await orch.job_count() == 5
        job_ids = {j.job_id for j in results}
        assert len(job_ids) == 5  # 모두 고유한 job_id

    @pytest.mark.asyncio
    async def test_unexpected_error_marks_job_failed(self):
        orch = _make_orchestrator(
            workflow=FailingWorkflow(RuntimeError("unexpected"))
        )
        with pytest.raises(RuntimeError):
            await orch.run("test", max_sources=3)
        jobs = await orch.list_jobs()
        assert jobs[0].status == "failed"
