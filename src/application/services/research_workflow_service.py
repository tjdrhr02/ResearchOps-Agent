from src.agents.collector.collector_agent import CollectorAgent
from src.agents.planner.planner_agent import PlannerAgent
from src.agents.reporter.reporter_agent import ReporterAgent
from src.agents.synthesizer.synthesizer_agent import SynthesizerAgent
from src.application.dto.agent_io import WorkflowResult
from src.domain.ports.metrics_port import MetricsPort
from src.domain.ports.retriever_port import RetrieverPort
from src.domain.ports.tool_port import ToolPort


class ResearchWorkflowService:
    def __init__(self, tools: list[ToolPort], retriever: RetrieverPort, metrics: MetricsPort) -> None:
        self.planner = PlannerAgent()
        self.collector = CollectorAgent(tools=tools)
        self.synthesizer = SynthesizerAgent()
        self.reporter = ReporterAgent()
        self.retriever = retriever
        self.metrics = metrics

    async def run(self, user_query: str, max_sources: int) -> WorkflowResult:
        self.metrics.increment("total_requests")
        planner_output = await self.planner.run(user_query=user_query)
        collector_output = await self.collector.run(planner_output=planner_output, max_sources=max_sources)
        indexed = await self.retriever.index_documents(collector_output.documents)
        self.metrics.increment("retrieval_count", indexed)
        evidence = await self.retriever.retrieve(query=user_query, k=min(5, max_sources))
        synthesis = await self.synthesizer.run(collected=collector_output, retrieved=evidence)
        brief = await self.reporter.run(planner_output=planner_output, synthesis=synthesis)
        return WorkflowResult(brief=brief, sources=collector_output.documents)
