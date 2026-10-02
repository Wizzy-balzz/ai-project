"""
run_phase4_paths.py
-------------------
Generates candidate attack paths from the synthetic dataset.

Loads the Phase 3 GNN checkpoint, runs path enumeration for all three
dataset splits, and saves results to results/phase4_candidates.json.

ALL DATA IS SYNTHETIC.
NO HISTORICAL ATTACK LABELS ARE USED FOR CANDIDATE GENERATION.
ALL GNN PROBABILITIES COME FROM THE ACTUAL TRAINED MODEL.

Usage:
    python run_phase4_paths.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase4_run")


def main() -> int:
    logger.info("=" * 65)
    logger.info("Graph-Based Attack Path Predictor — Phase 4 Path Generation")
    logger.info("=" * 65)
    logger.info("DATA LABEL: [SYNTHETIC] All topologies are procedurally generated.")
    logger.info("TASK:       Candidate attack-path enumeration via bounded DFS.")
    logger.info("LEAKAGE:    Historical attack labels NOT used for generation.")

    # -----------------------------------------------------------------------
    # Imports
    # -----------------------------------------------------------------------
    from backend.core.synthetic_data import SyntheticNetworkGenerator
    from backend.core.data_models import NetworkTopology
    from backend.ml.path_generator import PathGenerator
    from backend.ml.path_models import PathConfig

    CHECKPOINT = "checkpoints/best_attack_gnn.pt"
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------
    config = PathConfig(
        min_path_length=1,
        max_path_length=5,
        max_paths_per_entry=50,
        max_total_paths=2000,
        require_critical_terminal=False,
        allow_noncritical_terminal=True,
        sort_by="path_str",
    )

    logger.info("\n--- Path Generation Configuration ---")
    for k, v in config.to_dict().items():
        logger.info("  %s: %s", k, v)

    # -----------------------------------------------------------------------
    # Topology generation (same seeds as Phase 3 training)
    # -----------------------------------------------------------------------
    logger.info("\n--- Generating [SYNTHETIC] Topologies ---")

    # Generate the same 16 topologies used in Phase 3 dataset
    # (8 small + 6 medium + 2 large, seed_offset=0)
    all_topologies = []
    scenario_idx = 0
    for scenario_type, n_scenarios, gen_method in [
        ("small", 8, "generate_small"),
        ("medium", 6, "generate_medium"),
        ("large", 2, "generate_large"),
    ]:
        for i in range(n_scenarios):
            seed = scenario_idx
            gen = SyntheticNetworkGenerator(seed=seed)
            topo: NetworkTopology = getattr(gen, gen_method)()
            all_topologies.append({
                "topo": topo,
                "scenario_type": scenario_type,
                "seed": seed,
            })
            scenario_idx += 1
            logger.info(
                "  Topology %s[%d] seed=%d: '%s' (%d nodes, %d edges)",
                scenario_type, i, seed, topo.name, topo.node_count, topo.edge_count,
            )

    # -----------------------------------------------------------------------
    # Check checkpoint availability
    # -----------------------------------------------------------------------
    ckpt_available = Path(CHECKPOINT).exists()
    if ckpt_available:
        logger.info("\n--- Phase 3 Checkpoint: FOUND (%s) ---", CHECKPOINT)
    else:
        logger.warning(
            "\n--- Phase 3 Checkpoint: NOT FOUND ---\n"
            "  Generating paths WITHOUT GNN probabilities.\n"
            "  Run run_phase3_training.py first for full integration.\n"
        )

    # -----------------------------------------------------------------------
    # Path generation
    # -----------------------------------------------------------------------
    logger.info("\n--- Generating Candidate Attack Paths ---")
    generator = PathGenerator(
        config=config,
        checkpoint_path=CHECKPOINT if ckpt_available else None,
    )

    all_results = []
    total_candidates = 0
    total_critical = 0
    total_truncated = 0

    for entry in all_topologies:
        topo = entry["topo"]
        scenario_type = entry["scenario_type"]
        seed = entry["seed"]

        candidates, summary = generator.generate(topo)
        total_candidates += summary.total_candidates
        total_critical += summary.critical_terminal_count
        if summary.generation_truncated:
            total_truncated += 1

        logger.info(
            "  [%s seed=%d] '%s': %d candidates "
            "(%d reach critical, truncated=%s)",
            scenario_type, seed, topo.name,
            summary.total_candidates,
            summary.critical_terminal_count,
            summary.generation_truncated,
        )

        # Verify no historical labels used
        assert not summary.used_historical_labels, \
            "BUG: historical labels used in path generation!"

        all_results.append({
            "scenario_type": scenario_type,
            "seed": seed,
            "summary": summary.to_dict(),
            "sample_candidates": [
                c.to_dict() for c in candidates[:3]  # save first 3 as samples
            ],
            "all_candidate_paths": [
                {
                    "path_id": c.path_id,
                    "node_ids": c.node_ids,
                    "node_labels": c.node_labels,
                    "node_attack_probabilities": c.node_attack_probabilities,
                    "path_length": c.path_length,
                    "reaches_critical_asset": c.reaches_critical_asset,
                    "structural_metadata": c.structural_metadata,
                }
                for c in candidates
            ],
        })

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output_path = RESULTS_DIR / "phase4_candidates.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": config.to_dict(),
            "checkpoint_used": CHECKPOINT if ckpt_available else None,
            "gnn_probabilities_attached": ckpt_available,
            "used_historical_labels": False,
            "total_scenarios": len(all_topologies),
            "total_candidates": total_candidates,
            "total_critical_terminal": total_critical,
            "total_truncated_scenarios": total_truncated,
            "scenarios": all_results,
        }, f, indent=2)
    logger.info("\nResults saved to %s", output_path)

    # -----------------------------------------------------------------------
    # Validation: check all candidates are structurally valid
    # -----------------------------------------------------------------------
    logger.info("\n--- Structural Validation ---")
    from backend.core.graph_builder import GraphBuilder
    from backend.ml.path_generator import PathValidator

    n_invalid = 0
    n_checked = 0
    for entry in all_topologies:
        topo = entry["topo"]
        builder = GraphBuilder()
        G = builder.build(topo)
        validator = PathValidator(G, config)
        candidates, _ = generator.generate(topo)
        for c in candidates:
            result = validator.validate(c.node_ids)
            if not result.valid:
                logger.error("INVALID PATH in '%s': %s — %s",
                             topo.name, c.node_ids, result.errors)
                n_invalid += 1
            n_checked += 1

    logger.info("  Validated %d candidate paths: %d invalid.", n_checked, n_invalid)

    # -----------------------------------------------------------------------
    # Determinism verification
    # -----------------------------------------------------------------------
    logger.info("\n--- Determinism Verification ---")
    topo_check = all_topologies[0]["topo"]
    run1_candidates, _ = generator.generate(topo_check)
    run2_candidates, _ = generator.generate(topo_check)
    keys1 = [c.path_key for c in run1_candidates]
    keys2 = [c.path_key for c in run2_candidates]
    deterministic = keys1 == keys2
    logger.info("  Deterministic output: %s", deterministic)

    # -----------------------------------------------------------------------
    # Sample output
    # -----------------------------------------------------------------------
    logger.info("\n--- Sample Candidate Paths (first topology) ---")
    first_entry = all_topologies[0]["topo"]
    sample_candidates, _ = generator.generate(first_entry)
    for i, c in enumerate(sample_candidates[:5]):
        prob_str = " → ".join(f"{p:.3f}" for p in c.node_attack_probabilities)
        logger.info(
            "  Path %d: %s", i + 1,
            " → ".join(c.node_labels),
        )
        logger.info("    node_probs: [%s]", prob_str)
        logger.info(
            "    length=%d  reaches_critical=%s  mean_prob=%.3f",
            c.path_length,
            c.reaches_critical_asset,
            c.mean_attack_probability,
        )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 4 COMPLETE")
    logger.info("=" * 65)
    logger.info("Generation algorithm:    Bounded DFS (depth-first search)")
    logger.info("Configuration:           max_depth=%d, max_per_entry=%d, max_total=%d",
                config.max_path_length, config.max_paths_per_entry, config.max_total_paths)
    logger.info("Scenarios processed:     %d", len(all_topologies))
    logger.info("Total candidates:        %d", total_candidates)
    logger.info("Reach critical assets:   %d", total_critical)
    logger.info("Truncated scenarios:     %d", total_truncated)
    logger.info("Invalid paths:           %d", n_invalid)
    logger.info("Deterministic:           %s", deterministic)
    logger.info("GNN probs attached:      %s", ckpt_available)
    logger.info("Historical labels used:  NO (guaranteed by design)")
    logger.info("Output:                  results/phase4_candidates.json")
    logger.info("")
    logger.info("IMPORTANT:")
    logger.info("  Phase 4 generates CANDIDATE paths — it does NOT rank them.")
    logger.info("  Phase 5 will define the risk scoring and Top-K ranking.")
    logger.info("  [SYNTHETIC DATA] — no real systems involved.")

    return 0 if n_invalid == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
