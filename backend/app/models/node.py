"""SQLAlchemy model for network nodes."""
from __future__ import annotations

from sqlalchemy import Column, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseModel


class NetworkNode(BaseModel):
    """Database model representing a node within a network topology."""

    __tablename__ = "network_nodes"

    scenario_id = Column(String(64), ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True)
    ip_address = Column(String(64), nullable=False)
    role = Column(String(64), default="workstation", nullable=False)  # workstation, server, router, database, etc.
    os_type = Column(String(64), default="linux", nullable=False)     # linux, windows, cisco_ios, etc.
    cvss_score = Column(Float, default=0.0, nullable=False)           # 0.0 - 10.0
    asset_value = Column(Float, default=0.5, nullable=False)         # 0.0 - 1.0
    is_entry = Column(Boolean, default=False, nullable=False)
    is_target = Column(Boolean, default=False, nullable=False)
    
    # Metadata and canvas coordinates
    hostname = Column(String(128), default="", nullable=False)
    vulnerabilities = Column(String(512), default="[]", nullable=False)  # JSON array string of CVEs
    pos_x = Column(Float, default=100.0, nullable=False)
    pos_y = Column(Float, default=100.0, nullable=False)

    # Relationship to parent scenario
    scenario = relationship("Scenario", back_populates="nodes")
