"""Pydantic schemas for attack paths, risk scores, and trajectories."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class HopDetail(BaseModel):
    step_number: int
    source_id: str
    source_ip: str
    source_role: str
    target_id: str
    target_ip: str
    target_role: str
    edge_type: str = "network_access"
    vulnerability_cve: str = "SYN-2026-1042"
    cvss_score: float = 7.5
    step_likelihood: float = 0.85
    action_narrative: str = ""


class FactorAttribution(BaseModel):
    gnn_weight: float = 0.35
    edge_weight: float = 0.35
    cvss_severity: float = 0.30
    node_susceptibility: float = 0.85
    edge_permeability: float = 0.70
    vulnerability_exploitability: float = 0.88


class AttackTrajectoryResponse(BaseModel):
    id: str
    scenario_id: str
    node_ids: list[str]
    node_ips: list[str]
    node_roles: list[str]
    hops: int
    likelihood: float
    impact: float
    risk_score: float
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    hop_details: list[HopDetail] = Field(default_factory=list)
    factor_attribution: FactorAttribution = Field(default_factory=FactorAttribution)
    kill_chain_narrative: str = ""
    chokepoints: list[str] = Field(default_factory=list)


class TopKPathsResponse(BaseModel):
    scenario_id: str
    total_candidates: int
    top_k_count: int
    paths: list[AttackTrajectoryResponse]
    max_risk_score: float
    average_risk_score: float
    critical_count: int
    high_count: int
