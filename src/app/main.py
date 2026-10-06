"""FastAPI application setup."""

from __future__ import annotations

import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.deps import require_api_key
from app.api.routes import documents, health, ocr, search, translate
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.database import init_database
from app.rag import embeddings

logger = get_logger(__name__)

WEB_DIR = Path(__file__).parent / "web"


def _open_browser_when_ready(url: str) -> None:
    """Open the interface shortly after startup, once the server accepts connections."""
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare the database and warm the embedding model before serving traffic."""
    settings = get_settings()
    init_database()
    embeddings.warm_up()

    if settings.open_browser:
        _open_browser_when_ready(settings.browser_url)

    logger.info("Interface available at %s", settings.browser_url)
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Build the application, its middleware and its routes."""
    settings = get_settings()

    app = FastAPI(
        title="Indic Document Intelligence",
        version="1.0.0",
        description=(
            "Recognise, correct, translate and question multilingual documents. "
            "Supports English and nine Indian languages."
        ),
        lifespan=lifespan,
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        """Turn application errors into their declared status codes."""
        if exc.status_code >= 500:
            logger.error("%s on %s: %s", type(exc).__name__, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": type(exc).__name__, "detail": exc.message},
        )

    guarded = [Depends(require_api_key)]
    for router in (health.router, documents.router, ocr.router, translate.router, search.router):
        app.include_router(router, prefix="/api", dependencies=guarded)

    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the browser interface."""
        return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
