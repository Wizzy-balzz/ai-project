"""Pydantic schemas formatting nodes and edges for @xyflow/react directly."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0


class NodeData(BaseModel):
    id: str
    scenario_id: str
    ip: str
    role: str
    os: str
    cvss: float
    asset_value: float
    is_entry: bool
    is_target: bool
    hostname: str = ""
    vulnerabilities: list[str] = Field(default_factory=list)
    gnn_probability: float = 0.0
    lateral_velocity: float = 0.0
    threat_level: str = "LOW"
    in_critical_path: bool = False
    is_isolated: bool = False


class ReactFlowNode(BaseModel):
    id: str
    type: str = "customNode"
    position: Position
    data: NodeData


class EdgeData(BaseModel):
    id: str
    scenario_id: str
    source_id: str
    target_id: str
    edge_type: str = "network_access"
    is_firewalled: bool = False
    is_encrypted: bool = False
    anomaly_score: float = 0.0
    port: str = "443"
    in_critical_path: bool = False
    is_blocked: bool = False


class ReactFlowEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str = "customEdge"
    animated: bool = False
    data: EdgeData


class GraphTopologyResponse(BaseModel):
    scenario_id: str
    scenario_name: str
    nodes: list[ReactFlowNode]
    edges: list[ReactFlowEdge]
    stats: dict[str, Any] = Field(default_factory=dict)


class NodeCreate(BaseModel):
    ip_address: str
    role: str = "workstation"
    os_type: str = "linux"
    cvss_score: float = 0.0
    asset_value: float = 0.5
    is_entry: bool = False
    is_target: bool = False
    hostname: Optional[str] = ""
    vulnerabilities: Optional[list[str]] = None
    pos_x: Optional[float] = 100.0
    pos_y: Optional[float] = 100.0


class EdgeCreate(BaseModel):
    source_id: str
    target_id: str
    edge_type: str = "network_access"
    is_firewalled: bool = False
    is_encrypted: bool = False
    anomaly_score: float = 0.0
    port: str = "443"
