from fastapi import APIRouter, Depends, HTTPException, status as http_status

from src.api.dependencies import get_research_orchestrator
from src.api.schemas.request import ResearchRunRequest
from src.api.schemas.response import (
    ResearchJobResponse,
    ResearchRunResponse,
    ResearchSourcesResponse,
)
from src.application.orchestrators.research_orchestrator import ResearchOrchestrator

router = APIRouter()


@router.post("/run", response_model=ResearchRunResponse, status_code=http_status.HTTP_202_ACCEPTED)
async def run_research(
    request: ResearchRunRequest,
    orchestrator: ResearchOrchestrator = Depends(get_research_orchestrator),
) -> ResearchRunResponse:
    job = await orchestrator.submit(user_query=request.user_query, max_sources=request.max_sources)
    return ResearchRunResponse(
        job_id=job.job_id,
        status=job.status,
        message=f"Job accepted. Poll GET /research/{job.job_id} for results.",
    )


@router.get("/{job_id}", response_model=ResearchJobResponse)
async def get_research_job(
    job_id: str,
    orchestrator: ResearchOrchestrator = Depends(get_research_orchestrator),
) -> ResearchJobResponse:
    job = orchestrator.get_job_sync(job_id)
    if not job:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return ResearchJobResponse(job=job)


@router.get("/{job_id}/sources", response_model=ResearchSourcesResponse)
async def get_research_sources(
    job_id: str,
    orchestrator: ResearchOrchestrator = Depends(get_research_orchestrator),
) -> ResearchSourcesResponse:
    job = orchestrator.get_job_sync(job_id)
    if not job:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    return ResearchSourcesResponse(job_id=job.job_id, sources=job.sources)
