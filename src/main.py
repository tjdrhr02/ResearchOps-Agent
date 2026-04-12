from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()  # 프로젝트 루트의 .env 파일을 자동 로드

from src.api.routers.notes import router as notes_router
from src.api.routers.research import router as research_router
from src.observability.metrics.endpoint import router as metrics_router


def create_app() -> FastAPI:
    app = FastAPI(title="ResearchOps Agent", version="0.1.0")
    app.include_router(research_router, prefix="/research", tags=["research"])
    app.include_router(notes_router, prefix="/notes", tags=["notes"])
    app.include_router(metrics_router, tags=["observability"])
    return app


app = create_app()
