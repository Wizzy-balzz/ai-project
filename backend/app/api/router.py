"""Master API router aggregating all v1 sub-routers."""
from __future__ import annotations

from fastapi import APIRouter
from backend.app.api.scenarios import router as scenarios_router
from backend.app.api.nodes import router as nodes_router
from backend.app.api.edges import router as edges_router
from backend.app.api.paths import router as paths_router
from backend.app.api.explain import router as explain_router
from backend.app.api.simulate import router as simulate_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(scenarios_router)
api_router.include_router(nodes_router)
api_router.include_router(edges_router)
api_router.include_router(paths_router)
api_router.include_router(explain_router)
api_router.include_router(simulate_router)
