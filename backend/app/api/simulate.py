"""API endpoints for What-If defense simulation."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.scenario import Scenario
from backend.app.schemas.simulation import SimulationRequest, SimulationResult
from backend.app.services.graph_service import simulate_what_if_defense

router = APIRouter(tags=["Simulation"])


@router.post("/scenarios/{scenario_id}/simulate", response_model=SimulationResult)
def simulate_mitigations(
    scenario_id: str,
    payload: SimulationRequest,
    db: Session = Depends(get_db),
):
    """Execute counterfactual What-If defense simulation on a scenario."""
    sc = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not sc:
        raise HTTPException(status_code=404, detail="Scenario not found")

    return simulate_what_if_defense(db, scenario_id, payload)
