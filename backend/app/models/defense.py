"""SQLAlchemy model for defense mitigations."""
from __future__ import annotations

from sqlalchemy import Column, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseModel


class DefenseMitigation(BaseModel):
    """Database model representing what-if defense configurations."""

    __tablename__ = "defense_mitigations"

    scenario_id = Column(String(64), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    defense_type = Column(String(64), nullable=False)  # isolate_node, block_edge, patch_cve
    target_node_id = Column(String(64), nullable=True)
    target_edge_id = Column(String(64), nullable=True)
    risk_reduction = Column(Float, default=0.0, nullable=False)

    # Relationship to parent scenario
    scenario = relationship("Scenario", back_populates="defenses")
