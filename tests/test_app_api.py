"""
tests/test_app_api.py
---------------------
Tests for FastAPI backend/app REST endpoints:
- Scenarios CRUD
- Nodes with cascade deletion and clear
- Edges with cascade checks
- Attack paths and XAI explanations
- What-If defense simulation
- Health check
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["gnn_checkpoint_present"] is True


def test_scenarios_listing():
    res = client.get("/api/v1/scenarios")
    assert res.status_code == 200
    scenarios = res.json()
    assert len(scenarios) > 0
    golden = next((s for s in scenarios if "Golden Vineyard" in s["name"]), None)
    assert golden is not None
    assert golden["num_nodes"] >= 6


def test_scenario_nodes_and_topology():
    sc_res = client.get("/api/v1/scenarios")
    scenario_id = sc_res.json()[0]["id"]

    res = client.get(f"/api/v1/scenarios/{scenario_id}/nodes")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0


def test_paths_endpoint():
    sc_res = client.get("/api/v1/scenarios")
    scenario_id = sc_res.json()[0]["id"]

    res = client.get(f"/api/v1/scenarios/{scenario_id}/paths?k=5")
    assert res.status_code == 200
    data = res.json()
    assert "paths" in data
    assert len(data["paths"]) > 0
    top_path = data["paths"][0]
    assert top_path["hops"] >= 1
    assert 0.0 <= top_path["risk_score"] <= 1.0


def test_explain_path_endpoint():
    sc_res = client.get("/api/v1/scenarios")
    scenario_id = sc_res.json()[0]["id"]

    paths_res = client.get(f"/api/v1/scenarios/{scenario_id}/paths?k=1")
    path_id = paths_res.json()["paths"][0]["id"]

    res = client.get(f"/api/v1/paths/{path_id}/explain")
    assert res.status_code == 200
    data = res.json()
    assert "kill_chain_narrative" in data
    assert "factor_attribution" in data
    assert len(data["kill_chain_narrative"]) > 10


def test_simulate_defense_endpoint():
    sc_res = client.get("/api/v1/scenarios")
    scenario_id = sc_res.json()[0]["id"]

    nodes_res = client.get(f"/api/v1/scenarios/{scenario_id}/nodes")
    node_id_to_isolate = nodes_res.json()["nodes"][1]["id"]

    payload = {
        "isolated_node_ids": [node_id_to_isolate],
        "blocked_edge_ids": [],
        "patched_cves": ["SYN-2026-9910"],
    }
    res = client.post(f"/api/v1/scenarios/{scenario_id}/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "risk_drop_percentage" in data
    assert "severed_paths_count" in data


def test_node_creation_and_cascade_deletion():
    # 1. Create a blank scenario
    sc_create = client.post(
        "/api/v1/scenarios",
        json={
            "name": "Test Cascade Scenario",
            "description": "Scenario to test node deletion",
            "topology_type": "enterprise",
            "is_synthetic": True,
        },
    )
    assert sc_create.status_code == 201
    sc_id = sc_create.json()["id"]

    # 2. Create Node A and Node B
    n_a = client.post(
        f"/api/v1/scenarios/{sc_id}/nodes",
        json={
            "ip_address": "10.10.10.1",
            "role": "workstation",
            "os_type": "windows",
            "cvss_score": 7.0,
            "asset_value": 0.3,
            "is_entry": True,
            "is_target": False,
        },
    )
    assert n_a.status_code == 201

    n_b = client.post(
        f"/api/v1/scenarios/{sc_id}/nodes",
        json={
            "ip_address": "10.10.10.2",
            "role": "database",
            "os_type": "linux",
            "cvss_score": 9.5,
            "asset_value": 1.0,
            "is_entry": False,
            "is_target": True,
        },
    )
    assert n_b.status_code == 201
    node_b_id = n_b.json()["nodes"][1]["id"]
    node_a_id = n_b.json()["nodes"][0]["id"]

    # 3. Create Edge A -> B
    e_ab = client.post(
        f"/api/v1/scenarios/{sc_id}/edges",
        json={
            "source_id": node_a_id,
            "target_id": node_b_id,
            "edge_type": "network_access",
            "is_firewalled": False,
            "is_encrypted": False,
            "port": "5432",
        },
    )
    assert e_ab.status_code == 201
    assert len(e_ab.json()["edges"]) == 1

    # 4. Cascade delete Node A
    del_res = client.delete(f"/api/v1/nodes/{node_a_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 5. Verify Node A and Edge A->B are gone
    topo_res = client.get(f"/api/v1/scenarios/{sc_id}/nodes")
    topo_nodes = topo_res.json()["nodes"]
    topo_edges = topo_res.json()["edges"]
    assert len(topo_nodes) == 1
    assert topo_nodes[0]["id"] == node_b_id
    assert len(topo_edges) == 0

    # 6. Clear canvas
    clear_res = client.post(f"/api/v1/scenarios/{sc_id}/nodes/clear")
    assert clear_res.status_code == 200
    assert len(clear_res.json()["topology"]["nodes"]) == 0

    # 7. Verify simulation handles blank / cleared scenario gracefully without crashing
    sim_empty = client.post(
        f"/api/v1/scenarios/{sc_id}/simulate",
        json={"isolated_node_ids": [], "blocked_edge_ids": [], "patched_cves": []},
    )
    assert sim_empty.status_code == 200
    sim_data = sim_empty.json()
    assert sim_data["simulated_risk"] == 0.0
    assert sim_data["is_mitigated"] is True

