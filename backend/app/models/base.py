"""Shared base model with timestamps and standard identifiers."""
from __future__ import annotations

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime
from backend.app.database import Base


class BaseModel(Base):
    """Abstract declarative base class providing UUID primary key and timestamps."""

    __abstract__ = True

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
