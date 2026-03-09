from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_research_orchestrator
from src.api.schemas.request import ResearchRunRequest
from src.api.schemas.response import (
    ResearchJobResponse,
    ResearchRunResponse,
    ResearchSourcesResponse,
)
from src.application.orchestrators.research_orchestrator import ResearchOrchestrator

router = APIRouter()


@router.post("/run", response_model=ResearchRunResponse)
async def run_research(
    request: ResearchRunRequest,
    orchestrator: ResearchOrchestrator = Depends(get_research_orchestrator),
) -> ResearchRunResponse:
    job = await orchestrator.run(user_query=request.user_query, max_sources=request.max_sources)
    return ResearchRunResponse(job_id=job.job_id, status=job.status, brief=job.brief)


@router.get("/{job_id}", response_model=ResearchJobResponse)
async def get_research_job(
    job_id: str,
    orchestrator: ResearchOrchestrator = Depends(get_research_orchestrator),
) -> ResearchJobResponse:
    job = orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return ResearchJobResponse(job=job)


@router.get("/{job_id}/sources", response_model=ResearchSourcesResponse)
async def get_research_sources(
    job_id: str,
    orchestrator: ResearchOrchestrator = Depends(get_research_orchestrator),
) -> ResearchSourcesResponse:
    job = orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return ResearchSourcesResponse(job_id=job.job_id, sources=job.sources)
