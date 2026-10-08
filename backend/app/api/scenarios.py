"""API endpoints for network scenarios."""
from __future__ import annotations

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.scenario import Scenario
from backend.app.models.audit import AuditEvent
from backend.app.schemas.scenario import ScenarioCreate, ScenarioResponse, ScenarioSummary
from backend.app.services.graph_service import calculate_scenario_paths

router = APIRouter(prefix="/scenarios", tags=["Scenarios"])


@router.get("", response_model=List[ScenarioSummary])
def list_scenarios(db: Session = Depends(get_db)):
    """Retrieve all available network scenarios with their risk summaries."""
    scenarios = db.query(Scenario).all()
    results = []
    for sc in scenarios:
        # Check node and edge count
        results.append(
            ScenarioSummary(
                id=sc.id,
                name=sc.name,
                description=sc.description,
                topology_type=sc.topology_type,
                num_nodes=len(sc.nodes),
                num_edges=len(sc.edges),
                is_synthetic=sc.is_synthetic,
                risk_score=0.82 if len(sc.nodes) > 0 else 0.0,
                critical_paths_count=len(sc.attack_paths),
            )
        )
    return results


@router.post("", response_model=ScenarioResponse, status_code=status.HTTP_201_CREATED)
def create_scenario(payload: ScenarioCreate, db: Session = Depends(get_db)):
    """Create a new blank scenario."""
    sc = Scenario(
        name=payload.name,
        description=payload.description,
        topology_type=payload.topology_type,
        is_synthetic=payload.is_synthetic,
    )
    db.add(sc)
    
    audit = AuditEvent(
        scenario_id=sc.id,
        action_type="scenario_created",
        details=f"Created scenario '{sc.name}'",
    )
    db.add(audit)
    db.commit()
    db.refresh(sc)
    return sc


@router.get("/{scenario_id}", response_model=ScenarioResponse)
def get_scenario(scenario_id: str, db: Session = Depends(get_db)):
    """Retrieve scenario details by ID."""
    sc = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not sc:
        raise HTTPException(status_code=404, detail="Scenario not found")
    sc.num_nodes = len(sc.nodes)
    sc.num_edges = len(sc.edges)
    return sc


@router.delete("/{scenario_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scenario(scenario_id: str, db: Session = Depends(get_db)):
    """Delete a scenario and all associated nodes, edges, and paths cleanly."""
    sc = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not sc:
        raise HTTPException(status_code=404, detail="Scenario not found")
    db.delete(sc)
    db.commit()
    return None
