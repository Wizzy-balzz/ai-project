"""SQLAlchemy model for computed attack paths."""
from __future__ import annotations

from sqlalchemy import Column, String, Integer, Float, Text, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseModel


class AttackPath(BaseModel):
    """Database model representing an enumerated multi-hop attack trajectory."""

    __tablename__ = "attack_paths"

    scenario_id = Column(String(64), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    path_sequence = Column(Text, nullable=False)  # JSON serialized list of node IDs
    hops = Column(Integer, nullable=False)        # len(nodes) - 1
    likelihood = Column(Float, nullable=False)    # 0.0 - 1.0
    impact = Column(Float, nullable=False)        # 0.0 - 1.0
    risk_score = Column(Float, nullable=False)    # likelihood * impact
    severity = Column(String(16), nullable=False) # CRITICAL, HIGH, MEDIUM, LOW

    # Relationship to parent scenario
    scenario = relationship("Scenario", back_populates="attack_paths")
