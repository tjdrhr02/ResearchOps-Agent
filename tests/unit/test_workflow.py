import pytest

from src.api.dependencies import get_research_orchestrator


@pytest.mark.asyncio
async def test_research_workflow_runs() -> None:
    orchestrator = get_research_orchestrator()
    job = await orchestrator.run("LLM observability best practices", max_sources=5)
    assert job.job_id
    assert job.brief.executive_summary
    assert len(job.brief.key_trends) >= 1
    assert len(job.brief.citations) >= 1
