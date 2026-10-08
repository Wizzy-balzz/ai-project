"""API endpoints for network edges."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.scenario import Scenario
from backend.app.models.node import NetworkNode
from backend.app.models.edge import NetworkEdge
from backend.app.models.audit import AuditEvent
from backend.app.schemas.graph import EdgeCreate, GraphTopologyResponse
from backend.app.services.graph_service import (
    get_react_flow_topology,
    calculate_scenario_paths,
)

router = APIRouter(tags=["Edges"])


@router.post("/scenarios/{scenario_id}/edges", status_code=status.HTTP_201_CREATED)
def create_edge(
    scenario_id: str,
    payload: EdgeCreate,
    db: Session = Depends(get_db),
):
    """Create a directed edge between two existing nodes in a scenario."""
    sc = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not sc:
        raise HTTPException(status_code=404, detail="Scenario not found")

    src = db.query(NetworkNode).filter(NetworkNode.id == payload.source_id).first()
    tgt = db.query(NetworkNode).filter(NetworkNode.id == payload.target_id).first()
    if not src or not tgt:
        raise HTTPException(status_code=400, detail="Source or target node does not exist")

    edge = NetworkEdge(
        scenario_id=scenario_id,
        source_id=payload.source_id,
        target_id=payload.target_id,
        edge_type=payload.edge_type,
        is_firewalled=payload.is_firewalled,
        is_encrypted=payload.is_encrypted,
        anomaly_score=payload.anomaly_score,
        port=payload.port,
    )
    db.add(edge)
    sc.num_edges += 1

    audit = AuditEvent(
        scenario_id=scenario_id,
        action_type="edge_created",
        details=f"Created edge {src.ip_address} -> {tgt.ip_address}",
    )
    db.add(audit)
    db.commit()

    # Recalculate paths
    try:
        calculate_scenario_paths(db, scenario_id, k=5)
    except Exception:
        pass

    return get_react_flow_topology(db, scenario_id)


@router.delete("/edges/{edge_id}", status_code=status.HTTP_200_OK)
def delete_edge(edge_id: str, db: Session = Depends(get_db)):
    """Delete an edge and recalculate attack paths."""
    edge = db.query(NetworkEdge).filter(NetworkEdge.id == edge_id).first()
    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found")

    scenario_id = edge.scenario_id
    db.delete(edge)

    audit = AuditEvent(
        scenario_id=scenario_id,
        action_type="edge_deleted",
        details=f"Deleted edge {edge_id}",
    )
    db.add(audit)
    db.commit()

    try:
        calculate_scenario_paths(db, scenario_id, k=5)
    except Exception:
        pass

    return {
        "success": True,
        "message": f"Successfully deleted edge {edge_id}",
        "scenario_id": scenario_id,
        "topology": get_react_flow_topology(db, scenario_id),
    }


@router.patch("/edges/{edge_id}/toggle-firewall", status_code=status.HTTP_200_OK)
def toggle_firewall(edge_id: str, db: Session = Depends(get_db)):
    """Toggle firewall state on an edge to block/unblock traversal."""
    edge = db.query(NetworkEdge).filter(NetworkEdge.id == edge_id).first()
    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found")

    edge.is_firewalled = not edge.is_firewalled
    db.commit()

    try:
        calculate_scenario_paths(db, edge.scenario_id, k=5)
    except Exception:
        pass

    return {
        "success": True,
        "edge_id": edge.id,
        "is_firewalled": edge.is_firewalled,
        "topology": get_react_flow_topology(db, edge.scenario_id),
    }
