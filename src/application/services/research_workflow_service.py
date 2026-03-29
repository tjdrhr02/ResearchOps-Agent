from src.agents.collector.collector_agent import CollectorAgent
from src.agents.planner.planner_agent import PlannerAgent
from src.agents.reporter.reporter_agent import ReporterAgent
from src.agents.synthesizer.synthesizer_agent import SynthesizerAgent
from src.application.dto.agent_io import WorkflowResult
from src.domain.ports.metrics_port import MetricsPort
from src.domain.ports.retriever_port import RetrieverPort
from src.domain.ports.tool_port import ToolPort


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
        planner_output = await self.planner.run(user_query=user_query)
        collector_output = await self.collector.run(planner_output=planner_output, max_sources=max_sources)
        indexed = await self.retriever.index_documents(collector_output.documents)
        self.metrics.increment("retrieval_count", indexed)

        # 멀티쿼리 retrieval: Planner가 생성한 모든 쿼리로 검색
        plan = planner_output.plan
        evidence = await self.retriever.retrieve_multi_query(
            queries=plan.queries or [user_query],
            k=min(5, max_sources),
        )

        synthesis = await self.synthesizer.run(
            collected=collector_output,
            retrieved=evidence,
            research_topic=plan.question,
            research_objective=plan.objective,
        )
        brief = await self.reporter.run(
            planner_output=planner_output,
            synthesis=synthesis,
            source_count=len(collector_output.documents),
        )
        return WorkflowResult(brief=brief, sources=collector_output.documents)
