"""Pydantic schemas for What-If simulation and defense actions."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    isolated_node_ids: list[str] = Field(default_factory=list)
    blocked_edge_ids: list[str] = Field(default_factory=list)
    patched_cves: list[str] = Field(default_factory=list)


class SimulationResult(BaseModel):
    scenario_id: str
    original_risk: float
    simulated_risk: float
    risk_drop_percentage: float
    total_original_paths: int
    severed_paths_count: int
    remaining_paths_count: int
    active_defenses: list[dict[str, Any]]
    recommendation: str
    is_mitigated: bool
