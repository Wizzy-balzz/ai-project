"""SQLAlchemy model for network edges."""
from __future__ import annotations

from sqlalchemy import Column, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseModel


class NetworkEdge(BaseModel):
    """Database model representing a directed edge between network nodes."""

    __tablename__ = "network_edges"

    scenario_id = Column(String(64), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(String(64), ForeignKey("network_nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    target_id = Column(String(64), ForeignKey("network_nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    edge_type = Column(String(64), default="network_access", nullable=False)
    is_firewalled = Column(Boolean, default=False, nullable=False)
    is_encrypted = Column(Boolean, default=False, nullable=False)
    anomaly_score = Column(Float, default=0.0, nullable=False)
    port = Column(String(16), default="443", nullable=False)

    # Relationship to parent scenario
    scenario = relationship("Scenario", back_populates="edges")
