"""API endpoints for network nodes."""
from __future__ import annotations

import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.scenario import Scenario
from backend.app.models.node import NetworkNode
from backend.app.models.edge import NetworkEdge
from backend.app.models.attack_path import AttackPath
from backend.app.models.audit import AuditEvent
from backend.app.schemas.graph import NodeCreate, ReactFlowNode, GraphTopologyResponse
from backend.app.services.graph_service import (
    get_react_flow_topology,
    calculate_scenario_paths,
)

router = APIRouter(tags=["Nodes"])


@router.get("/scenarios/{scenario_id}/nodes", response_model=GraphTopologyResponse)
def get_scenario_nodes(
    scenario_id: str,
    active_path: str = None,
    db: Session = Depends(get_db),
):
    """Retrieve all nodes and edges formatted for @xyflow/react canvas."""
    sc = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not sc:
        raise HTTPException(status_code=404, detail="Scenario not found")

    path_node_ids = []
    if active_path:
        db_path = db.query(AttackPath).filter(AttackPath.id == active_path).first()
        if db_path:
            try:
                path_node_ids = json.loads(db_path.path_sequence)
            except Exception:
                path_node_ids = []

    return get_react_flow_topology(db, scenario_id, active_path_node_ids=path_node_ids)


@router.post("/scenarios/{scenario_id}/nodes", status_code=status.HTTP_201_CREATED)
def create_node(
    scenario_id: str,
    payload: NodeCreate,
    db: Session = Depends(get_db),
):
    """Create a new network node within a scenario."""
    sc = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not sc:
        raise HTTPException(status_code=404, detail="Scenario not found")

    node = NetworkNode(
        scenario_id=scenario_id,
        ip_address=payload.ip_address,
        role=payload.role,
        os_type=payload.os_type,
        cvss_score=payload.cvss_score,
        asset_value=payload.asset_value,
        is_entry=payload.is_entry,
        is_target=payload.is_target,
        hostname=payload.hostname or payload.ip_address,
        vulnerabilities=json.dumps(payload.vulnerabilities or []),
        pos_x=payload.pos_x or 100.0,
        pos_y=payload.pos_y or 100.0,
    )
    db.add(node)
    sc.num_nodes += 1

    audit = AuditEvent(
        scenario_id=scenario_id,
        action_type="node_created",
        details=f"Created node {payload.ip_address} ({payload.role})",
    )
    db.add(audit)
    db.commit()
    db.refresh(node)

    # Recalculate paths dynamically if we have entry and target
    try:
        calculate_scenario_paths(db, scenario_id, k=5)
    except Exception:
        pass

    return get_react_flow_topology(db, scenario_id)


@router.delete("/nodes/{node_id}", status_code=status.HTTP_200_OK)
def delete_node(node_id: str, db: Session = Depends(get_db)):
    """
    Clean cascade delete for a node:
    Removes the node, removes all incident edges, and recalculates paths dynamically.
    """
    node = db.query(NetworkNode).filter(NetworkNode.id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    scenario_id = node.scenario_id
    node_label = f"{node.ip_address} ({node.role})"

    # Remove incident edges
    edges_to_remove = (
        db.query(NetworkEdge)
        .filter((NetworkEdge.source_id == node_id) | (NetworkEdge.target_id == node_id))
        .all()
    )
    for edge in edges_to_remove:
        db.delete(edge)

    db.delete(node)

    # Log audit event
    audit = AuditEvent(
        scenario_id=scenario_id,
        action_type="node_deleted",
        details=f"Cascade deleted node {node_label} and {len(edges_to_remove)} connected edges",
    )
    db.add(audit)
    db.commit()

    # Recalculate paths dynamically
    try:
        calculate_scenario_paths(db, scenario_id, k=5)
    except Exception:
        pass

    return {
        "success": True,
        "message": f"Successfully deleted node {node_label} and {len(edges_to_remove)} edges.",
        "scenario_id": scenario_id,
        "topology": get_react_flow_topology(db, scenario_id),
    }


@router.post("/scenarios/{scenario_id}/nodes/clear", status_code=status.HTTP_200_OK)
def clear_all_nodes(scenario_id: str, db: Session = Depends(get_db)):
    """Wipe all nodes, edges, and paths for a scenario to start with an empty canvas."""
    sc = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not sc:
        raise HTTPException(status_code=404, detail="Scenario not found")

    db.query(NetworkEdge).filter(NetworkEdge.scenario_id == scenario_id).delete()
    db.query(NetworkNode).filter(NetworkNode.scenario_id == scenario_id).delete()
    db.query(AttackPath).filter(AttackPath.scenario_id == scenario_id).delete()

    sc.num_nodes = 0
    sc.num_edges = 0

    audit = AuditEvent(
        scenario_id=scenario_id,
        action_type="nodes_cleared",
        details="Cleared all nodes and edges from canvas",
    )
    db.add(audit)
    db.commit()

    return {
        "success": True,
        "message": "All nodes and edges cleared. Canvas is empty.",
        "scenario_id": scenario_id,
        "topology": get_react_flow_topology(db, scenario_id),
    }
