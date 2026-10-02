"""
Phase 4 Validation Script
=========================
Run this script to verify Phase 4 end-to-end:
- PathConfig validation
- Bounded DFS candidate path enumeration
- Path length definition (edges count)
- Structural validation (8 rules)
- Edge feature preservation (9-dim per hop)
- GNN probability integration (checkpoint evaluation)
- Determinism and leakage prevention checks
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("phase4_smoke_test")


def run_phase4_smoke_test() -> bool:
    print("=" * 65)
    print("Graph-Based Attack Path Predictor — Phase 4 Smoke Test")
    print("=" * 65)
    print("DATA LABEL: [SYNTHETIC] All topologies are procedurally generated.")

    # -------------------------------------------------------------------
    # Step 1: Import modules and check PathConfig
    # -------------------------------------------------------------------
    print("\n[1/6] Checking PathConfig and Data Structures...")
    from backend.ml.path_models import PathConfig, CandidateAttackPath, PathGenerationSummary
    from backend.ml.path_generator import PathGenerator, PathValidator, _extract_edge_feature_vector

    config = PathConfig(
        min_path_length=1,
        max_path_length=5,
        max_paths_per_entry=50,
        max_total_paths=500,
        require_critical_terminal=False,
    )
    print(f"      min_path_length: {config.min_path_length} (edge count)")
    print(f"      max_path_length: {config.max_path_length} (depth bound)")
    print(f"      max_paths_per_entry: {config.max_paths_per_entry}")
    print("  ✓ PathConfig initialized successfully")

    # -------------------------------------------------------------------
    # Step 2: Generate synthetic network topology
    # -------------------------------------------------------------------
    print("\n[2/6] Generating synthetic network topology (seed=42)...")
    from backend.core.synthetic_data import SyntheticNetworkGenerator
    from backend.core.graph_builder import GraphBuilder

    gen = SyntheticNetworkGenerator(seed=42)
    topo = gen.generate_small()
    builder = GraphBuilder()
    G = builder.build(topo)
    print(f"      Topology: '{topo.name}' ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    print(f"      Entry points: {[n for n, d in G.nodes(data=True) if d.get('is_entry_point') == 1]}")
    print(f"      Critical assets: {[n for n, d in G.nodes(data=True) if d.get('is_critical_asset') == 1]}")
    print("  ✓ Graph built and verified")

    # -------------------------------------------------------------------
    # Step 3: Run Candidate Attack Path Generator
    # -------------------------------------------------------------------
    print("\n[3/6] Enumerating candidate attack paths (Bounded DFS)...")
    ckpt_path = Path("checkpoints/best_attack_gnn.pt")
    ckpt_available = ckpt_path.exists()
    print(f"      Phase 3 Checkpoint present: {ckpt_available} ({ckpt_path})")

    generator = PathGenerator(config=config, checkpoint_path=str(ckpt_path) if ckpt_available else None)
    candidates, summary = generator.generate(topo)

    print(f"      Total candidates generated: {summary.total_candidates}")
    print(f"      Critical-terminal candidates: {summary.critical_terminal_count}")
    print(f"      GNN probabilities attached: {summary.gnn_probabilities_attached}")
    print(f"      Used historical labels: {summary.used_historical_labels} (MUST be False)")
    print(f"      Generation truncated: {summary.generation_truncated}")

    assert not summary.used_historical_labels, "FAIL: Leakage detected: historical labels were used!"
    assert summary.total_candidates > 0, "FAIL: Expected at least one candidate path in small topology"
    print("  ✓ Candidate path enumeration complete")

    # -------------------------------------------------------------------
    # Step 4: Validate candidate paths against structural rules
    # -------------------------------------------------------------------
    print("\n[4/6] Validating candidate paths with PathValidator...")
    validator = PathValidator(G, config)
    for c in candidates:
        v_res = validator.validate(c)
        assert v_res.is_valid, f"FAIL: Candidate {c.path_id} failed validation: {v_res.errors}"
        assert c.path_length == len(c.node_ids) - 1, "FAIL: path_length must equal len(node_ids) - 1"
        assert len(c.edge_pairs) == c.path_length, "FAIL: edge_pairs length mismatch"
        assert len(c.edge_features) == c.path_length, "FAIL: edge_features length mismatch"
        for ef in c.edge_features:
            assert len(ef) == 9, "FAIL: Each edge feature vector must have 9 elements"
        if ckpt_available:
            for p in c.node_attack_probabilities:
                assert 0.0 <= p <= 1.0, f"FAIL: Probability out of bounds: {p}"

    print(f"      Verified {len(candidates)} candidate paths against all 8 structural rules.")
    print("  ✓ Structural validation passed")

    # -------------------------------------------------------------------
    # Step 5: Test Determinism
    # -------------------------------------------------------------------
    print("\n[5/6] Testing determinism...")
    candidates2, summary2 = generator.generate(topo)
    assert len(candidates) == len(candidates2), "FAIL: Path counts differ between runs"
    for c1, c2 in zip(candidates, candidates2):
        assert c1.node_ids == c2.node_ids, "FAIL: Candidate path ordering is not deterministic"
    print("  ✓ Determinism verified: identical paths in identical order")

    # -------------------------------------------------------------------
    # Step 6: Sample Candidate Display
    # -------------------------------------------------------------------
    print("\n[6/6] Inspecting Sample Candidate Path:")
    sample = candidates[0]
    print(f"      Path ID: {sample.path_id}")
    print(f"      Traversal: {' -> '.join(sample.node_labels)}")
    print(f"      Node IDs:  {' -> '.join(sample.node_ids)}")
    print(f"      Path length (edges): {sample.path_length}")
    print(f"      Reaches critical asset: {sample.reaches_critical_asset}")
    if ckpt_available:
        print(f"      GNN Attack Probs: {[round(p, 4) for p in sample.node_attack_probabilities]}")
    print(f"      Structural metadata: {sample.structural_metadata}")

    print("\n" + "=" * 65)
    print("ALL PHASE 4 SMOKE TEST CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_phase4_smoke_test()
    sys.exit(0 if success else 1)
