"""SQLAlchemy model for network scenarios."""
from __future__ import annotations

from sqlalchemy import Column, String, Integer, Boolean, Text
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseModel


class Scenario(BaseModel):
    """Database model representing a network scenario / topology."""

    __tablename__ = "scenarios"

    name = Column(String(128), nullable=False)
    description = Column(Text, default="", nullable=False)
    topology_type = Column(String(64), default="enterprise", nullable=False)
    num_nodes = Column(Integer, default=0, nullable=False)
    num_edges = Column(Integer, default=0, nullable=False)
    is_synthetic = Column(Boolean, default=True, nullable=False)

    # Relationships with full cascade on delete
    nodes = relationship("NetworkNode", back_populates="scenario", cascade="all, delete-orphan")
    edges = relationship("NetworkEdge", back_populates="scenario", cascade="all, delete-orphan")
    attack_paths = relationship("AttackPath", back_populates="scenario", cascade="all, delete-orphan")
    defenses = relationship("DefenseMitigation", back_populates="scenario", cascade="all, delete-orphan")
    audit_events = relationship("AuditEvent", back_populates="scenario", cascade="all, delete-orphan")
