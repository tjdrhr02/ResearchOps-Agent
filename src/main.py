import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from src.api.routers.health import router as health_router
from src.api.routers.notes import router as notes_router
from src.api.routers.research import router as research_router
from src.observability.logging.structured_logger import configure_logging
from src.observability.metrics.cloud_monitoring_exporter import start_export_loop
from src.observability.metrics.endpoint import router as metrics_router

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(start_export_loop())
    yield
    task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(title="ResearchOps Agent", version="0.1.0", lifespan=lifespan)
    app.include_router(health_router, tags=["health"])
    app.include_router(research_router, prefix="/research", tags=["research"])
    app.include_router(notes_router, prefix="/notes", tags=["notes"])
    app.include_router(metrics_router, tags=["observability"])
    return app


app = create_app()
