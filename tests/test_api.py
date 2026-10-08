"""
test_api.py
-----------
Integration tests for Phase 7 FastAPI Backend & Phase 8 Dashboard endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "gnn_checkpoint_present" in data
    assert len(data["phases_active"]) == 8


def test_scenarios_endpoint():
    res = client.get("/api/scenarios")
    assert res.status_code == 200
    scenarios = res.json()
    assert len(scenarios) == 3
    ids = [s["id"] for s in scenarios]
    assert "small" in ids
    assert "medium" in ids
    assert "large" in ids


def test_graph_endpoint_valid():
    res = client.get("/api/graph/small")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 10
    assert len(data["edges"]) >= 15


def test_graph_endpoint_invalid():
    res = client.get("/api/graph/nonexistent")
    assert res.status_code == 404


def test_analyze_endpoint():
    res = client.post(
        "/api/analyze",
        json={"scenario_id": "small", "top_k": 3, "max_path_length": 4},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["scenario_id"] == "small"
    assert "network_risk_score" in data
    assert "network_risk_tier" in data
    assert len(data["top_k_paths"]) <= 3
    assert len(data["critical_assets_at_risk"]) > 0
    assert len(data["explanations"]) == len(data["top_k_paths"])
    assert len(data["chokepoints"]) > 0
    assert len(data["recommended_mitigations"]) > 0


def test_simulate_endpoint():
    res = client.post(
        "/api/simulate",
        json={
            "scenario_id": "small",
            "mitigation_id": "mit-test",
            "action_type": "isolate_node",
            "target_element": "internet-gateway",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "original_network_risk" in data
    assert "new_network_risk" in data
    assert "risk_reduction_pct" in data
    assert "severed_paths_count" in data


def test_dashboard_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Graph-Based Attack Path Predictor AI" in res.text
