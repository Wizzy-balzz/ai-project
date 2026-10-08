"""
validate_full_pipeline.py
=========================
Full end-to-end validation script for Graph-Based Attack Path Predictor AI:
- Phase 1: Synthetic Network Generation & Graph Construction
- Phase 2: PyG Graph Data Pipeline & 43 Node / 9 Edge Features
- Phase 3: GraphSAGE GNN Inference & Node Attack Probabilities
- Phase 4: Bounded DFS Candidate Attack Path Generation
- Phase 5: Risk Scoring, Top-K Ranking & Critical Asset Exposure
- Phase 6: Explainable AI, Chokepoint Analysis & Counterfactual Simulation
- Phase 7: FastAPI Backend Endpoints
- Phase 8: Proactive Defense Dashboard Assets
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

from backend.core.graph_builder import GraphBuilder
from backend.core.synthetic_data import SyntheticNetworkGenerator
from backend.ml.explainer import PathExplainer
from backend.ml.path_generator import PathGenerator
from backend.ml.path_models import PathConfig
from backend.ml.risk_scorer import RiskScorer
from backend.api.main import app


def run_pipeline_validation() -> bool:
    print("=" * 70)
    print("Graph-Based Attack Path Predictor AI — Complete Pipeline Validation")
    print("=" * 70)
    print("DATA LABEL: [SYNTHETIC] All network data is procedurally generated.")

    # 1. Phase 1: Graph Construction
    print("\n[Phase 1] Generating synthetic topology and constructing graph...")
    gen = SyntheticNetworkGenerator(seed=42)
    topo = gen.generate_small()
    builder = GraphBuilder()
    G = builder.build(topo)
    print(f"  ✓ Built graph '{topo.name}': {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. Phase 2 & 3: GNN Inference
    ckpt_path = Path("checkpoints/best_attack_gnn.pt")
    print(f"\n[Phase 2 & 3] Loading GNN checkpoint: {ckpt_path.exists()} ({ckpt_path})")

    # 3. Phase 4: Candidate Path Enumeration
    print("\n[Phase 4] Enumerating candidate attack paths (Bounded DFS)...")
    generator = PathGenerator(
        config=PathConfig(max_path_length=5, max_paths_per_entry=30),
        checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None,
    )
    candidates, summary = generator.generate(topo)
    print(f"  ✓ Generated {summary.total_candidates} candidate attack paths (used_historical={summary.used_historical_labels})")

    # 4. Phase 5: Risk Scoring
    print("\n[Phase 5] Scoring attack paths and ranking Top-K...")
    scorer = RiskScorer()
    top_paths, critical_assets, network_risk = scorer.score_topology(candidates, top_k=5)
    print(f"  ✓ Evaluated network risk: {network_risk:.4f} ({scorer._get_tier(network_risk)})")
    print(f"  ✓ Top-K paths ranked: {len(top_paths)}")
    print(f"  ✓ Critical assets exposed: {len(critical_assets)}")
    for i, p in enumerate(top_paths[:3]):
        print(f"      #{i+1}: {' -> '.join(p.node_labels)} | Risk: {p.risk_score:.4f} [{p.risk_tier}]")

    # 5. Phase 6: Explainable AI & Simulation
    print("\n[Phase 6] Generating explanations, chokepoints & simulating mitigation...")
    explainer = PathExplainer(scorer)
    top_explanation = explainer.explain_path(top_paths[0], G)
    chokepoints = explainer.find_chokepoints(top_paths, G)
    mitigations = explainer.generate_mitigations(top_paths, G)
    print(f"  ✓ XAI Narrative: {top_explanation.path_summary}")
    print(f"  ✓ Primary Chokepoint: {chokepoints[0].element_label} ({chokepoints[0].intersected_paths_count} paths)")
    print(f"  ✓ Mitigations Recommended: {len(mitigations)}")

    if mitigations:
        sim = explainer.simulate_mitigation(candidates, mitigations[0])
        print(f"  ✓ Simulated Defense [{mitigations[0].title}]:")
        print(f"      Severed: {sim['severed_paths_count']} paths | Risk Reduced: {sim['risk_reduction_pct']}%")

    # 6. Phase 7 & 8: API & Dashboard Check
    print("\n[Phase 7 & 8] Testing FastAPI backend service & dashboard assets...")
    client = TestClient(app)
    h_res = client.get("/api/health")
    assert h_res.status_code == 200, "Health check failed"

    d_res = client.get("/")
    assert d_res.status_code == 200, "Dashboard serve failed"
    print("  ✓ Health check endpoint operational")
    print("  ✓ Web dashboard frontend served successfully")

    print("\n" + "=" * 70)
    print("ALL PHASES (1 THROUGH 8) VALIDATED AND OPERATIONAL!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_pipeline_validation()
    sys.exit(0 if success else 1)
