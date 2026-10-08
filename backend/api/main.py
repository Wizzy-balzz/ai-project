"""
main.py
-------
Phase 7 & 8: FastAPI Backend & Proactive Defense Service.

Serves REST API endpoints for:
- System Health & Model Status
- Scenario Catalog (Small, Medium, Large synthetic topologies)
- Cytoscape Graph Topology Data
- End-to-end Attack Path Analysis (GNN + DFS + Phase 5 Risk + Phase 6 XAI)
- What-If Mitigation Simulation
- Interactive Web Dashboard
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.core.graph_builder import GraphBuilder
from backend.core.synthetic_data import SyntheticNetworkGenerator
from backend.ml.explainer import MitigationAction, PathExplainer
from backend.ml.path_generator import PathGenerator
from backend.ml.path_models import CandidateAttackPath, PathConfig
from backend.ml.risk_scorer import RiskScorer, ScoredAttackPath

logger = logging.getLogger("backend_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Graph-Based Attack Path Predictor AI",
    description="Defensive cybersecurity research platform modeling attack paths using GNNs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_attack_gnn.pt"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# In-memory singletons for fast response times
_builder = GraphBuilder()
_scorer = RiskScorer()
_explainer = PathExplainer(_scorer)
_generator = PathGenerator(
    config=PathConfig(max_path_length=5, max_paths_per_entry=40, max_total_paths=500),
    checkpoint_path=str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
)

# Cache scenarios
_scenarios_cache: Dict[str, Any] = {}


def get_scenario(scenario_id: str):
    """Retrieve or generate synthetic topology by scenario ID."""
    sid = scenario_id.lower()
    if sid in _scenarios_cache:
        return _scenarios_cache[sid]

    gen = SyntheticNetworkGenerator(seed=42)
    if sid == "small":
        topo = gen.generate_small()
    elif sid == "medium":
        topo = gen.generate_medium()
    elif sid == "large":
        topo = gen.generate_large()
    else:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")

    _scenarios_cache[sid] = topo
    return topo


def topology_to_cytoscape(topo) -> Dict[str, List[Dict[str, Any]]]:
    """Convert NetworkTopology to Cytoscape.js elements."""
    G = _builder.build(topo)
    nodes = []
    edges = []

    for nid, d in G.nodes(data=True):
        vuln_count = d.get("vuln_count", 0)
        max_cvss = d.get("max_cvss", 0.0)
        is_entry = bool(d.get("is_entry_point", 0))
        is_crit = bool(d.get("is_critical_asset", 0))

        # Node styling class
        if is_entry:
            ntype = "entry"
        elif is_crit:
            ntype = "critical"
        elif vuln_count > 0 and max_cvss >= 7.0:
            ntype = "vulnerable"
        else:
            ntype = "normal"

        nodes.append({
            "data": {
                "id": nid,
                "label": d.get("label", nid),
                "type": ntype,
                "node_role": d.get("node_type", "server"),
                "ip": d.get("ip_address", "192.168.1.1"),
                "is_entry": is_entry,
                "is_critical": is_crit,
                "max_cvss": max_cvss,
                "vuln_count": vuln_count,
            }
        })

    for u, v, d in G.edges(data=True):
        edge_id = f"{u}->{v}"
        has_fw = bool(d.get("firewall_rule", 0))
        is_enc = bool(d.get("is_encrypted", 0))
        atk_wt = d.get("attack_weight", 0.5)

        edges.append({
            "data": {
                "id": edge_id,
                "source": u,
                "target": v,
                "has_firewall": has_fw,
                "is_encrypted": is_enc,
                "attack_weight": round(atk_wt, 3),
                "edge_type": d.get("edge_type", "network_link"),
            }
        })

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    scenario_id: str = Field(default="small", description="small, medium, or large")
    top_k: int = Field(default=5, ge=1, le=20)
    max_path_length: int = Field(default=5, ge=1, le=8)


class SimulateRequest(BaseModel):
    scenario_id: str = Field(default="small")
    mitigation_id: str
    action_type: str
    target_element: str


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "version": "1.0.0",
        "gnn_checkpoint_present": CHECKPOINT_PATH.exists(),
        "phases_active": [
            "Phase 1: Graph Construction",
            "Phase 2: PyG Data Pipeline",
            "Phase 3: GraphSAGE GNN",
            "Phase 4: Attack Path Generation",
            "Phase 5: Risk Scoring & Top-K",
            "Phase 6: Explainable AI & Simulator",
            "Phase 7: FastAPI Backend",
            "Phase 8: Proactive Defense Dashboard",
        ],
        "research_disclaimer": "Synthetic defensive research only. No real systems involved.",
    }


@app.get("/api/scenarios")
def list_scenarios():
    return [
        {
            "id": "small",
            "name": "Small SOHO Network",
            "description": "10 nodes, 3 entry points, 2 critical targets (Domain Controller & Database).",
            "nodes": 10,
            "complexity": "Standard",
        },
        {
            "id": "medium",
            "name": "Medium Enterprise Network",
            "description": "25-30 nodes with segmented DMZ, internal servers, and workstations.",
            "nodes": 25,
            "complexity": "Moderate",
        },
        {
            "id": "large",
            "name": "Large Multi-Segment Enterprise",
            "description": "50+ nodes with industrial IoT/OT, cloud interconnects, and strict perimeter.",
            "nodes": 50,
            "complexity": "Complex",
        },
    ]


@app.get("/api/graph/{scenario_id}")
def get_graph(scenario_id: str):
    topo = get_scenario(scenario_id)
    return topology_to_cytoscape(topo)


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    topo = get_scenario(req.scenario_id)
    G = _builder.build(topo)

    # Configure path generator
    cfg = PathConfig(
        max_path_length=req.max_path_length,
        max_paths_per_entry=40,
        max_total_paths=1000,
    )
    generator = PathGenerator(
        config=cfg,
        checkpoint_path=str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
    )

    candidates, summary = generator.generate(topo)

    # Phase 5: Risk Scoring & Top-K
    top_paths, critical_assets, network_risk = _scorer.score_topology(
        candidates, top_k=req.top_k
    )

    # Phase 6: Explainable AI & Mitigations
    explanations = [_explainer.explain_path(p, G).to_dict() for p in top_paths]
    chokepoints = [cp.to_dict() for cp in _explainer.find_chokepoints(top_paths, G)]
    mitigations = [m.to_dict() for m in _explainer.generate_mitigations(top_paths, G)]

    return {
        "scenario_id": req.scenario_id,
        "topology_name": topo.name,
        "summary": summary.to_dict(),
        "network_risk_score": round(network_risk, 4),
        "network_risk_tier": _scorer._get_tier(network_risk),
        "top_k_paths": [p.to_dict() for p in top_paths],
        "critical_assets_at_risk": [ca.to_dict() for ca in critical_assets],
        "explanations": explanations,
        "chokepoints": chokepoints,
        "recommended_mitigations": mitigations,
    }


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    topo = get_scenario(req.scenario_id)
    candidates, _ = _generator.generate(topo)

    mit = MitigationAction(
        mitigation_id=req.mitigation_id,
        title=f"Mitigation on {req.target_element}",
        action_type=req.action_type,
        target_element=req.target_element,
        target_label=req.target_element,
        description="Simulated action",
        priority="HIGH",
        severed_paths_count=0,
        severed_path_ids=[],
        estimated_risk_reduction_pct=0.0,
    )

    res = _explainer.simulate_mitigation(candidates, mit)
    return res


# ---------------------------------------------------------------------------
# Dashboard Frontend Serving
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Graph-Based Attack Path Predictor AI API Online</h1><p>Dashboard not found.</p>")


# Mount static folder
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=False)
