"""
FastAPI Enterprise Application for Graph-Based Attack Path Predictor AI.
Exposes REST endpoints under /api/v1 and manages database/ML lifecycle.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.database import engine, Base
from backend.app.api.router import api_router
from backend.app.services.graph_service import get_path_generator, get_risk_scorer
from backend.seed_demo_data import seed_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for database initialization and GNN checkpoint loading."""
    logger.info("Initializing SQLite database tables...")
    Base.metadata.create_all(bind=engine)

    logger.info("Verifying GNN inference engine and checkpoints...")
    try:
        _ = get_path_generator()
        _ = get_risk_scorer()
        logger.info("GNN inference models ready.")
    except Exception as e:
        logger.warning("GNN initialization warning: %s", e)

    logger.info("Ensuring demo data 'Operation Golden Vineyard' is seeded...")
    try:
        seed_database()
    except Exception as e:
        logger.error("Error seeding initial data: %s", e)

    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Cybersecurity Attack Path Predictor AI with GraphSAGE GNN and XAI Explanations",
    lifespan=lifespan,
)

# CORS middleware for local Vite frontend and network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Master API Router
app.include_router(api_router)


@app.get("/api/health", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
def health_check():
    """Health check endpoint validating API and ML readiness."""
    ckpt_exists = settings.best_checkpoint_path.exists()
    return JSONResponse(
        content={
            "status": "online",
            "version": "1.0.0",
            "gnn_checkpoint_present": ckpt_exists,
            "architecture": "GraphSAGE 2-Layer Inductive GNN",
            "scoring_formula": "Risk = Likelihood x Impact",
            "research_disclaimer": "Synthetic defensive research only. No real systems involved.",
        }
    )


# Mount built frontend static files if present (supports unified single-port deployment)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.app.config import PROJECT_ROOT

DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", tags=["Frontend"])
    def serve_spa(full_path: str):
        file_path = DIST_DIR / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")
