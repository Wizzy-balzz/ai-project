"""SQLAlchemy model for audit logging."""
from __future__ import annotations

from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseModel


class AuditEvent(BaseModel):
    """Database model logging system modifications, node deletions, and user actions."""

    __tablename__ = "audit_events"

    scenario_id = Column(String(64), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=True, index=True)
    action_type = Column(String(64), nullable=False) # node_added, node_deleted, edge_added, simulation_run, etc.
    details = Column(Text, default="", nullable=False)
    operator = Column(String(64), default="analyst", nullable=False)

    # Relationship to parent scenario
    scenario = relationship("Scenario", back_populates="audit_events")
