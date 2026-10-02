"""
Phase 1 Validation Script
=========================
Run this script to verify Phase 1 end-to-end before running pytest.
It exercises the full pipeline: synthetic data → graph → features.
"""

import sys
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_phase1_smoke_test():
    print("=" * 60)
    print("Graph-Based Attack Path Predictor — Phase 1 Smoke Test")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Step 1: Generate synthetic topology
    # -------------------------------------------------------------------
    print("\n[1/4] Generating synthetic network topology (seed=42)...")
    from backend.core.synthetic_data import SyntheticNetworkGenerator
    gen = SyntheticNetworkGenerator(seed=42)
    topo = gen.generate_small()

    assert topo.is_synthetic, "FAIL: is_synthetic must be True"
    print(f"      Topology: '{topo.name}'")
    print(f"      Nodes:    {topo.node_count}")
    print(f"      Edges:    {topo.edge_count}")
    print(f"      Entry points:    {len(topo.entry_points)}")
    print(f"      Critical assets: {len(topo.critical_assets)}")
    print(f"      Telemetry records: {len(topo.telemetry)}")
    print(f"      Historical attacks: {len(topo.historical_attacks)}")
    print("      [SYNTHETIC DATA LABEL: ALL CVE IDs use SYN- prefix]")
    print("  ✓ Topology generated")

    # -------------------------------------------------------------------
    # Step 2: Build graph
    # -------------------------------------------------------------------
    print("\n[2/4] Building NetworkX DiGraph...")
    from backend.core.graph_builder import GraphBuilder
    builder = GraphBuilder()
    G = builder.build(topo)

    print(f"      Directed graph nodes: {G.number_of_nodes()}")
    print(f"      Directed graph edges: {G.number_of_edges()}")

    validation = builder.validate_graph(G)
    print(f"      Has entry points:     {validation['has_entry_points']}")
    print(f"      Has critical assets:  {validation['has_critical_assets']}")
    print(f"      Weakly connected:     {validation['is_weakly_connected']}")
    print(f"      Has self-loops:       {validation['has_self_loops']}")
    print(f"      Any attack path:      {validation['any_path_exists']}")

    for pair in validation['reachability']:
        status = "✓" if pair['reachable'] else "✗"
        print(f"      {status} {pair['from']} → {pair['to']}")

    assert validation['has_entry_points'], "FAIL: No entry points in graph"
    assert validation['has_critical_assets'], "FAIL: No critical assets in graph"
    assert validation['any_path_exists'], "FAIL: No path from entry to critical asset"
    print("  ✓ Graph built and validated")

    # -------------------------------------------------------------------
    # Step 3: Extract features
    # -------------------------------------------------------------------
    print("\n[3/4] Extracting node features...")
    from backend.core.feature_extractor import FeatureExtractor, FEATURE_DIM
    extractor = FeatureExtractor(normalize=True)
    features, node_index = extractor.extract(G)

    print(f"      Feature matrix shape: {features.shape}")
    print(f"      Feature dimension:    {FEATURE_DIM}")
    print(f"      Node index length:    {len(node_index)}")

    import numpy as np
    assert features.shape == (G.number_of_nodes(), FEATURE_DIM), "FAIL: Wrong feature shape"
    assert features.dtype.name == "float32", "FAIL: Features not float32"
    assert float(features.min()) >= 0.0, "FAIL: Normalized features below 0"
    assert float(features.max()) <= 1.0 + 1e-5, "FAIL: Normalized features above 1"
    print("  ✓ Features extracted and validated")

    # -------------------------------------------------------------------
    # Step 4: Graph statistics
    # -------------------------------------------------------------------
    print("\n[4/4] Computing graph statistics...")
    stats = extractor.get_graph_statistics(G)
    for k, v in stats.items():
        print(f"      {k}: {v}")
    print("  ✓ Statistics computed")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 1 Smoke Test: ALL CHECKS PASSED ✓")
    print("=" * 60)
    print("\nNote: All data is [SYNTHETIC]. No real systems were involved.")
    return True


if __name__ == "__main__":
    try:
        success = run_phase1_smoke_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
