"""
ResearchWorkflowService.

Planner → Collector → RAG → Synthesizer → Reporter 순서로
각 Agent를 호출하고 단계별 결과를 WorkflowTrace에 기록한다.

오류 격리 전략:
  Planner 실패    → PlannerError 발생, 워크플로우 중단 (plan 없이는 진행 불가)
  Collector 실패  → CollectorError 로깅, 빈 문서 목록으로 계속
  RAG 실패        → RAGError 로깅, 빈 evidence로 계속
  Synthesizer 실패 → SynthesizerError 로깅, 빈 synthesis로 계속
  Reporter 실패   → ReporterError 발생, 워크플로우 중단 (brief 없이 반환 불가)
"""
import logging
import time

from src.agents.collector.collector_agent import CollectorAgent
from src.agents.planner.planner_agent import PlannerAgent
from src.agents.reporter.reporter_agent import ReporterAgent
from src.agents.synthesizer.synthesizer_agent import SynthesizerAgent
from src.application.dto.agent_io import (
    CollectorOutput,
    PlannerOutput,
    SynthesizerOutput,
    WorkflowResult,
    WorkflowTrace,
)
from src.domain.errors.exceptions import (
    CollectorError,
    PlannerError,
    RAGError,
    ReporterError,
    SynthesizerError,
)
from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.ports.metrics_port import MetricsPort
from src.domain.ports.retriever_port import RetrieverPort
from src.domain.ports.tool_port import ToolPort

logger = logging.getLogger(__name__)


class ResearchWorkflowService:
    def __init__(
        self,
        tools: list[ToolPort],
        retriever: RetrieverPort,
        metrics: MetricsPort,
        planner_llm=None,
        synthesizer_llm=None,
        reporter_llm=None,
    ) -> None:
        self.planner = PlannerAgent(llm=planner_llm)
        self.collector = CollectorAgent(tools=tools)
        self.synthesizer = SynthesizerAgent(llm=synthesizer_llm)
        self.reporter = ReporterAgent(llm=reporter_llm)
        self.retriever = retriever
        self.metrics = metrics

    async def run(self, user_query: str, max_sources: int) -> WorkflowResult:
        self.metrics.increment("total_requests")
        trace = WorkflowTrace()

        # ── Step 1: Planner ──────────────────────────────────────────
        planner_output = await self._run_planner(user_query, trace)

        # ── Step 2: Collector ────────────────────────────────────────
        collector_output = await self._run_collector(planner_output, max_sources, trace)

        # ── Step 3: RAG Index + Retrieve ────────────────────────────
        evidence = await self._run_rag(
            planner_output=planner_output,
            collector_output=collector_output,
            user_query=user_query,
            max_sources=max_sources,
            trace=trace,
        )

        # ── Step 4: Synthesizer ──────────────────────────────────────
        synthesis = await self._run_synthesizer(
            planner_output=planner_output,
            collector_output=collector_output,
            evidence=evidence,
            trace=trace,
        )

        # ── Step 5: Reporter ─────────────────────────────────────────
        brief = await self._run_reporter(
            planner_output=planner_output,
            synthesis=synthesis,
            source_count=len(collector_output.documents),
            trace=trace,
        )

        logger.info(
            "workflow_complete query=%r trace=[%s]",
            user_query[:60],
            trace.summary(),
        )
        return WorkflowResult(
            brief=brief,
            sources=collector_output.documents,
            trace=trace,
        )

    # ────────────────────────────────────────────────────────────────
    # 단계별 실행 메서드
    # ────────────────────────────────────────────────────────────────

    async def _run_planner(self, user_query: str, trace: WorkflowTrace) -> PlannerOutput:
        t0 = time.perf_counter()
        try:
            output = await self.planner.run(user_query=user_query)
            elapsed = _ms(t0)
            trace.add(
                step="planner",
                status="success",
                elapsed_ms=elapsed,
                detail=f"research_type={output.plan.research_type} queries={len(output.plan.queries)}",
            )
            logger.info(
                "step_planner status=success queries=%s elapsed_ms=%.1f",
                len(output.plan.queries),
                elapsed,
            )
            return output
        except Exception as exc:
            elapsed = _ms(t0)
            trace.add(step="planner", status="failed", elapsed_ms=elapsed, error=str(exc))
            logger.error("step_planner status=failed error=%s", exc)
            self.metrics.increment("failed_steps")
            raise PlannerError("Failed to generate research plan", cause=exc) from exc

    async def _run_collector(
        self,
        planner_output: PlannerOutput,
        max_sources: int,
        trace: WorkflowTrace,
    ) -> CollectorOutput:
        t0 = time.perf_counter()
        try:
            output = await self.collector.run(
                planner_output=planner_output, max_sources=max_sources
            )
            elapsed = _ms(t0)
            trace.add(
                step="collector",
                status="success",
                elapsed_ms=elapsed,
                detail=f"documents={len(output.documents)}",
            )
            logger.info(
                "step_collector status=success documents=%s elapsed_ms=%.1f",
                len(output.documents),
                elapsed,
            )
            return output
        except Exception as exc:
            elapsed = _ms(t0)
            trace.add(step="collector", status="failed", elapsed_ms=elapsed, error=str(exc))
            logger.warning(
                "step_collector status=failed error=%s — continuing with empty docs", exc
            )
            self.metrics.increment("failed_steps")
            raise CollectorError("Data collection failed", cause=exc) from exc

    async def _run_rag(
        self,
        planner_output: PlannerOutput,
        collector_output: CollectorOutput,
        user_query: str,
        max_sources: int,
        trace: WorkflowTrace,
    ) -> list[EvidenceChunk]:
        plan = planner_output.plan

        # 3a. 인덱싱
        t0 = time.perf_counter()
        indexed = 0
        try:
            if collector_output.documents:
                indexed = await self.retriever.index_documents(collector_output.documents)
                self.metrics.increment("retrieval_count", indexed)
            elapsed = _ms(t0)
            trace.add(
                step="rag_index",
                status="success",
                elapsed_ms=elapsed,
                detail=f"indexed={indexed}",
            )
            logger.info(
                "step_rag_index status=success indexed=%s elapsed_ms=%.1f",
                indexed,
                elapsed,
            )
        except Exception as exc:
            elapsed = _ms(t0)
            trace.add(step="rag_index", status="failed", elapsed_ms=elapsed, error=str(exc))
            logger.warning(
                "step_rag_index status=failed error=%s — skipping retrieval", exc
            )
            self.metrics.increment("failed_steps")
            return []

        # 3b. 검색
        t0 = time.perf_counter()
        try:
            evidence = await self.retriever.retrieve_multi_query(
                queries=plan.queries or [user_query],
                k=min(5, max_sources),
            )
            elapsed = _ms(t0)
            trace.add(
                step="rag_retrieve",
                status="success",
                elapsed_ms=elapsed,
                detail=f"evidence={len(evidence)}",
            )
            logger.info(
                "step_rag_retrieve status=success evidence=%s elapsed_ms=%.1f",
                len(evidence),
                elapsed,
            )
            return evidence
        except Exception as exc:
            elapsed = _ms(t0)
            trace.add(step="rag_retrieve", status="failed", elapsed_ms=elapsed, error=str(exc))
            logger.warning(
                "step_rag_retrieve status=failed error=%s — continuing with empty evidence", exc
            )
            self.metrics.increment("failed_steps")
            raise RAGError("Retrieval failed", cause=exc) from exc

    async def _run_synthesizer(
        self,
        planner_output: PlannerOutput,
        collector_output: CollectorOutput,
        evidence: list[EvidenceChunk],
        trace: WorkflowTrace,
    ) -> SynthesizerOutput:
        plan = planner_output.plan
        t0 = time.perf_counter()
        try:
            output = await self.synthesizer.run(
                collected=collector_output,
                retrieved=evidence,
                research_topic=plan.question,
                research_objective=plan.objective,
            )
            elapsed = _ms(t0)
            trace.add(
                step="synthesizer",
                status="success",
                elapsed_ms=elapsed,
                detail=f"claims={len(output.claims)} comparisons={len(output.comparisons)}",
            )
            logger.info(
                "step_synthesizer status=success claims=%s elapsed_ms=%.1f",
                len(output.claims),
                elapsed,
            )
            return output
        except Exception as exc:
            elapsed = _ms(t0)
            trace.add(step="synthesizer", status="failed", elapsed_ms=elapsed, error=str(exc))
            logger.warning(
                "step_synthesizer status=failed error=%s — using empty synthesis", exc
            )
            self.metrics.increment("failed_steps")
            raise SynthesizerError("Synthesis failed", cause=exc) from exc

    async def _run_reporter(
        self,
        planner_output: PlannerOutput,
        synthesis: SynthesizerOutput,
        source_count: int,
        trace: WorkflowTrace,
    ):
        t0 = time.perf_counter()
        try:
            brief = await self.reporter.run(
                planner_output=planner_output,
                synthesis=synthesis,
                source_count=source_count,
            )
            elapsed = _ms(t0)
            trace.add(
                step="reporter",
                status="success",
                elapsed_ms=elapsed,
                detail=f"trends={len(brief.key_trends)} citations={len(brief.citations)}",
            )
            logger.info(
                "step_reporter status=success trends=%s citations=%s elapsed_ms=%.1f",
                len(brief.key_trends),
                len(brief.citations),
                elapsed,
            )
            return brief
        except Exception as exc:
            elapsed = _ms(t0)
            trace.add(step="reporter", status="failed", elapsed_ms=elapsed, error=str(exc))
            logger.error("step_reporter status=failed error=%s", exc)
            self.metrics.increment("failed_steps")
            raise ReporterError("Brief generation failed", cause=exc) from exc


def _ms(t0: float) -> float:
    """perf_counter 기준 경과 밀리초를 반환한다."""
    return (time.perf_counter() - t0) * 1000
