"""Pydantic schemas for network scenarios."""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class ScenarioBase(BaseModel):
    name: str
    description: str = ""
    topology_type: str = "enterprise"
    is_synthetic: bool = True


class ScenarioCreate(ScenarioBase):
    pass


class ScenarioResponse(ScenarioBase):
    id: str
    num_nodes: int = 0
    num_edges: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ScenarioSummary(BaseModel):
    id: str
    name: str
    description: str
    topology_type: str
    num_nodes: int
    num_edges: int
    is_synthetic: bool
    risk_score: float = 0.0
    critical_paths_count: int = 0
