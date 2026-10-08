"""Configuration management for Attack Path Predictor AI."""
from __future__ import annotations

import os
from pathlib import Path

# Use pydantic_settings if installed; otherwise fall back smoothly to pydantic BaseModel
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel as BaseSettings

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Application settings and configuration parameters."""

    app_name: str = "Graph-Based Attack Path Predictor AI"
    environment: str = os.environ.get("ENVIRONMENT", "production")
    debug: bool = os.environ.get("DEBUG", "").lower() in ("true", "1")
    
    # Network ports and hosts
    backend_port: int = int(os.environ.get("BACKEND_PORT", 8000))
    backend_host: str = os.environ.get("BACKEND_HOST", "0.0.0.0")
    frontend_port: int = int(os.environ.get("FRONTEND_PORT", 5173))
    
    # SQLite Database URI
    database_url: str = os.environ.get(
        "SQLITE_URL",
        f"sqlite:///{PROJECT_ROOT}/attack_predictor.db",
    )
    
    # GNN Checkpoints directory
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    best_checkpoint_path: Path = PROJECT_ROOT / "checkpoints" / "best_attack_gnn.pt"

    # CORS Origins
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ]


settings = Settings()
