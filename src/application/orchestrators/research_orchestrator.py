import time
from uuid import uuid4

from src.application.dto.agent_io import WorkflowResult
from src.application.services.research_workflow_service import ResearchWorkflowService
from src.domain.models.research_job import ResearchJob
from src.domain.ports.metrics_port import MetricsPort


class ResearchOrchestrator:
    def __init__(self, workflow: ResearchWorkflowService, metrics: MetricsPort) -> None:
        self.workflow = workflow
        self.metrics = metrics
        self._jobs: dict[str, ResearchJob] = {}

    async def run(self, user_query: str, max_sources: int) -> ResearchJob:
        start = time.perf_counter()
        try:
            result: WorkflowResult = await self.workflow.run(user_query=user_query, max_sources=max_sources)
            job = ResearchJob(
                job_id=str(uuid4()),
                query=user_query,
                status="completed",
                brief=result.brief,
                sources=result.sources,
            )
            self._jobs[job.job_id] = job
            self.metrics.increment("failed_jobs", 0)
            return job
        except Exception:
            self.metrics.increment("failed_jobs", 1)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics.observe_latency("request_latency_ms", elapsed_ms)

    def get_job(self, job_id: str) -> ResearchJob | None:
        return self._jobs.get(job_id)
