"""
run_phase3_training.py
----------------------
Runs a complete, reproducible Phase 3 training experiment on synthetic data.

ALL DATA IS SYNTHETIC.
ALL METRICS ARE COMPUTED FROM ACTUAL MODEL RUNS.
NO METRICS ARE FABRICATED.

Usage:
    python run_phase3_training.py

Output:
    checkpoints/best_attack_gnn.pt     — trained model checkpoint
    results/phase3_eval_results.json   — evaluation metrics (all three splits)
    results/phase3_training_history.json — per-epoch training history
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase3_run")


def main() -> int:
    logger.info("=" * 65)
    logger.info("Graph-Based Attack Path Predictor — Phase 3 Training Run")
    logger.info("=" * 65)
    logger.info("DATA LABEL: [SYNTHETIC] All topologies are procedurally generated.")
    logger.info("ML TASK:    Node-level binary attack involvement classification")
    logger.info("MODEL:      AttackGNN (GraphSAGE backbone)")

    # ------------------------------------------------------------------
    # Step 1: Imports
    # ------------------------------------------------------------------
    from backend.core.feature_extractor import FEATURE_DIM
    from backend.ml.dataset import EDGE_FEATURE_DIM, build_dataset
    from backend.ml.evaluator import Evaluator
    from backend.ml.model import AttackGNN
    from backend.ml.trainer import TrainingConfig, Trainer, set_seeds

    # ------------------------------------------------------------------
    # Step 2: Configuration
    # ------------------------------------------------------------------
    config = TrainingConfig(
        random_seed=42,
        hidden_channels=64,
        num_layers=2,
        dropout=0.3,
        learning_rate=5e-3,
        weight_decay=1e-4,
        epochs=200,
        patience=20,
        use_class_weights=True,
        checkpoint_dir="checkpoints",
        checkpoint_name="best_attack_gnn.pt",
    )

    logger.info("\n--- Training Configuration ---")
    for k, v in config.to_dict().items():
        logger.info("  %s: %s", k, v)

    set_seeds(config.random_seed)

    # ------------------------------------------------------------------
    # Step 3: Dataset
    # ------------------------------------------------------------------
    logger.info("\n--- Building [SYNTHETIC] Dataset ---")
    dataset = build_dataset(
        n_small=8,
        n_medium=6,
        n_large=2,
        seed_offset=0,
        split_seed=42,
        normalize_features=True,
        validate=True,
    )

    logger.info("Dataset summary:")
    for k, v in dataset.summary().items():
        logger.info("  %s: %s", k, v)

    # Verify leakage
    leakage = dataset.check_leakage()
    if not leakage["no_leakage"]:
        logger.error("DATA LEAKAGE DETECTED: %s", leakage)
        return 1
    logger.info("  leakage_check: PASSED (no topology overlap across splits)")

    # Warn about all-positive scenarios (documented behavior, not suppressed)
    all_pos_train = sum(
        1 for d in dataset.train_data
        if hasattr(d, "negative_count") and d.negative_count == 0
    )
    if all_pos_train > 0:
        logger.warning(
            "WARNING: %d training scenario(s) have all-positive labels. "
            "This occurs in small (10-node) topologies where 8 attack paths "
            "with 1-3 random intermediate hops from 5 intermediate nodes "
            "cover all nodes. These scenarios contribute valid loss signal "
            "but do NOT contribute to ROC-AUC / PR-AUC computation "
            "(single-class is handled gracefully by the Evaluator). "
            "No scenarios are silently deleted.",
            all_pos_train,
        )

    # ------------------------------------------------------------------
    # Step 4: Model
    # ------------------------------------------------------------------
    logger.info("\n--- Model Architecture ---")
    model = AttackGNN(
        in_channels=FEATURE_DIM,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    arch = model.architecture_summary()
    for k, v in arch.items():
        logger.info("  %s: %s", k, v)

    # ------------------------------------------------------------------
    # Step 5: Training
    # ------------------------------------------------------------------
    logger.info("\n--- Training ---")
    trainer = Trainer(config)
    history = trainer.train(model, dataset.train_data, dataset.val_data)

    logger.info("Training complete:")
    logger.info("  epochs_run:      %d", history.epochs_run)
    logger.info("  best_epoch:      %d", history.best_epoch)
    logger.info("  best_val_loss:   %.6f", history.best_val_loss)
    logger.info("  stopped_early:   %s", history.stopped_early)
    logger.info("  total_time_s:    %.1f", history.total_train_time_s)

    if history.train_losses:
        logger.info("  first_train_loss: %.6f", history.train_losses[0])
        logger.info("  last_train_loss:  %.6f", history.train_losses[-1])

    # ------------------------------------------------------------------
    # Step 6: Evaluation
    # ------------------------------------------------------------------
    logger.info("\n--- Evaluation (ALL METRICS FROM ACTUAL MODEL RUNS) ---")
    evaluator = Evaluator(model)

    Path("results").mkdir(exist_ok=True)
    results = evaluator.evaluate_all_splits(
        dataset.train_data,
        dataset.val_data,
        dataset.test_data,
        output_path="results/phase3_eval_results.json",
    )

    for split_name, result in results.items():
        logger.info("\n  [%s split]", split_name.upper())
        logger.info("    n_scenarios: %d",  result.n_scenarios)
        logger.info("    n_nodes:     %d",  result.n_nodes)
        logger.info("    n_positive:  %d",  result.n_positive)
        logger.info("    n_negative:  %d",  result.n_negative)
        logger.info("    accuracy:    %.4f", result.accuracy)
        logger.info("    precision:   %s",
                    f"{result.precision:.4f}" if result.precision is not None else "N/A")
        logger.info("    recall:      %s",
                    f"{result.recall:.4f}" if result.recall is not None else "N/A")
        logger.info("    F1:          %s",
                    f"{result.f1:.4f}" if result.f1 is not None else "N/A")
        logger.info("    ROC-AUC:     %s",
                    f"{result.roc_auc:.4f}" if result.roc_auc_available else
                    f"NOT AVAILABLE — {result.roc_auc_unavailable_reason[:80]}")
        logger.info("    PR-AUC:      %s",
                    f"{result.pr_auc:.4f}" if result.pr_auc_available else
                    f"NOT AVAILABLE — {result.pr_auc_unavailable_reason[:80]}")
        if result.confusion_matrix:
            logger.info("    confusion_matrix: %s", result.confusion_matrix)

    # ------------------------------------------------------------------
    # Step 7: Save training history
    # ------------------------------------------------------------------
    history_path = "results/phase3_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history.to_dict(), f, indent=2)
    logger.info("\nTraining history saved to %s", history_path)

    # ------------------------------------------------------------------
    # Step 8: Reproducibility verification
    # ------------------------------------------------------------------
    logger.info("\n--- Reproducibility Verification ---")
    logger.info("Re-running training with same seed to verify reproducibility...")
    set_seeds(config.random_seed)
    model2 = AttackGNN(
        in_channels=FEATURE_DIM,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    config2 = TrainingConfig(**{**config.to_dict(), "checkpoint_name": "repro_check.pt"})
    history2 = Trainer(config2).train(model2, dataset.train_data, dataset.val_data)

    losses_match = history.train_losses[:5] == history2.train_losses[:5]
    logger.info("  First 5 train losses match: %s", losses_match)
    if not losses_match:
        logger.warning("  Run 1: %s", history.train_losses[:5])
        logger.warning("  Run 2: %s", history2.train_losses[:5])

    # Clean up repro checkpoint
    repro_ckpt = Path("checkpoints/repro_check.pt")
    if repro_ckpt.exists():
        repro_ckpt.unlink()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 3 TRAINING RUN COMPLETE")
    logger.info("=" * 65)
    logger.info("Checkpoint:    checkpoints/best_attack_gnn.pt")
    logger.info("Eval results:  results/phase3_eval_results.json")
    logger.info("History:       results/phase3_training_history.json")
    logger.info("")
    logger.info("IMPORTANT — Phase 3 scope:")
    logger.info("  This model performs NODE-LEVEL ATTACK INVOLVEMENT CLASSIFICATION.")
    logger.info("  It does NOT produce complete multi-hop attack paths.")
    logger.info("  Attack path construction is a Phase 4+ concern.")
    logger.info("  [SYNTHETIC DATA] — no real systems involved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
