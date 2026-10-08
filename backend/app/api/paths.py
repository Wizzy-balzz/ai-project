"""API endpoints for attack paths and trajectories."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.scenario import Scenario
from backend.app.schemas.path import TopKPathsResponse
from backend.app.services.graph_service import calculate_scenario_paths

router = APIRouter(tags=["Paths"])


@router.get("/scenarios/{scenario_id}/paths", response_model=TopKPathsResponse)
def get_scenario_paths(
    scenario_id: str,
    k: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Retrieve top-K ranked attack paths and risk trajectories for a scenario."""
    sc = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not sc:
        raise HTTPException(status_code=404, detail="Scenario not found")

    return calculate_scenario_paths(db, scenario_id, k=k)
