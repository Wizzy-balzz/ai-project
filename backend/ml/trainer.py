"""
trainer.py
----------
Reproducible training pipeline for AttackGNN.

Design decisions:
- All random seeds are set at the start of training and logged.
- Class weights are computed ONLY from training data labels, never from
  validation or test labels. This prevents any form of label leakage.
- Early stopping monitors validation loss with a configurable patience window.
- Training loop aggregates predictions across multiple graph scenarios in
  each epoch (one gradient step per scenario per epoch).
- Best model weights are saved by validation loss (not accuracy), because
  accuracy is misleading on imbalanced datasets.
- Checkpoint format: Python dict saved via torch.save — fully reproducible.
- All numeric results are primitive Python types for JSON serialization.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from backend.ml.model import AttackGNN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    All hyperparameters and training settings in one place.

    No magic numbers in training code — everything comes from here.
    """
    # Reproducibility
    random_seed: int = 42

    # Model architecture
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.3

    # Optimiser
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4

    # Training schedule
    epochs: int = 200
    patience: int = 20          # Early stopping patience (epochs)

    # Class imbalance
    use_class_weights: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best_attack_gnn.pt"

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For CPU determinism
    torch.use_deterministic_algorithms(True, warn_only=True)


def compute_class_weights(train_data: List[Data]) -> Optional[torch.Tensor]:
    """
    Compute class weights from TRAINING data labels ONLY.

    Uses inverse-frequency weighting:
      weight[c] = total_samples / (num_classes * count[c])

    Returns None if class weights should not be applied (e.g., both classes
    have zero samples — pathological case handled gracefully).

    IMPORTANT: This function must NEVER be called with validation or test data.

    Args:
        train_data: List of PyG Data objects from the TRAINING split only.

    Returns:
        torch.Tensor of shape (2,) with float weights, or None.
    """
    all_labels = []
    for d in train_data:
        if d.y is not None:
            all_labels.append(d.y)

    if not all_labels:
        logger.warning("No labels found in training data — disabling class weights.")
        return None

    y_all = torch.cat(all_labels, dim=0)
    n_total = y_all.numel()
    n_pos = int((y_all == 1).sum().item())
    n_neg = int((y_all == 0).sum().item())

    logger.info(
        "Training label distribution: %d positive (%.1f%%), %d negative (%.1f%%) "
        "— computed from training split ONLY.",
        n_pos, 100.0 * n_pos / n_total if n_total > 0 else 0.0,
        n_neg, 100.0 * n_neg / n_total if n_total > 0 else 0.0,
    )

    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "Training data contains only one class (%s). "
            "Class weights disabled — single-class training is degenerate.",
            "positive" if n_neg == 0 else "negative",
        )
        return None

    # Inverse frequency weights
    weight_neg = n_total / (2.0 * n_neg)
    weight_pos = n_total / (2.0 * n_pos)
    weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float)
    logger.info("Class weights: neg=%.4f, pos=%.4f", weight_neg, weight_pos)
    return weights


def _scenario_is_single_class(data: Data) -> bool:
    """Return True if a Data object has only one unique label (0 or 1)."""
    if data.y is None or data.y.numel() == 0:
        return True
    return len(data.y.unique()) == 1


# ---------------------------------------------------------------------------
# Training History
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    """Records metrics for every epoch."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    total_train_time_s: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Reproducible training pipeline for AttackGNN.

    Handles:
    - Seed setting
    - Class weight computation (training labels only)
    - Per-epoch training over multiple graph scenarios
    - Validation loss monitoring
    - Early stopping
    - Best checkpoint saving
    - Training history recording

    Usage:
        config = TrainingConfig(epochs=100, patience=15)
        trainer = Trainer(config)
        model = AttackGNN(in_channels=43)
        history = trainer.train(model, dataset.train_data, dataset.val_data)
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Trainer device: %s", self.device)

    def _get_criterion(self, train_data: List[Data]) -> nn.CrossEntropyLoss:
        """Build the loss function, optionally with class weights."""
        if self.config.use_class_weights:
            weights = compute_class_weights(train_data)
        else:
            weights = None

        if weights is not None:
            return nn.CrossEntropyLoss(weight=weights.to(self.device))
        return nn.CrossEntropyLoss()

    def _train_epoch(
        self,
        model: AttackGNN,
        train_data: List[Data],
        optimizer: torch.optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
    ) -> Tuple[float, float]:
        """
        Run one training epoch across all training scenarios.

        Each scenario is one gradient step. Scenarios with a single class
        still contribute to training (loss is valid), but their contribution
        to accuracy metrics is logged separately.

        Returns:
            (mean_loss, accuracy) over all training scenarios.
        """
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0
        skipped = 0

        for data in train_data:
            if data.x is None or data.y is None:
                skipped += 1
                continue

            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            y = data.y.to(self.device)

            optimizer.zero_grad()
            logits = model(x, edge_index)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients on small graphs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == y).sum().item())
            total_nodes += y.numel()

        if skipped > 0:
            logger.debug("Skipped %d training scenarios with missing data.", skipped)

        n_scenarios = len(train_data) - skipped
        avg_loss = total_loss / max(n_scenarios, 1)
        accuracy = total_correct / max(total_nodes, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: AttackGNN,
        data_list: List[Data],
        criterion: nn.CrossEntropyLoss,
    ) -> Tuple[float, float]:
        """
        Evaluate model on a list of scenarios without gradient updates.

        Returns:
            (mean_loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0
        n_valid = 0

        for data in data_list:
            if data.x is None or data.y is None:
                continue

            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            y = data.y.to(self.device)

            logits = model(x, edge_index)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == y).sum().item())
            total_nodes += y.numel()
            n_valid += 1

        avg_loss = total_loss / max(n_valid, 1)
        accuracy = total_correct / max(total_nodes, 1)
        return avg_loss, accuracy

    def train(
        self,
        model: AttackGNN,
        train_data: List[Data],
        val_data: List[Data],
    ) -> TrainingHistory:
        """
        Full training loop with early stopping.

        Args:
            model:      AttackGNN instance (untrained).
            train_data: List of training PyG Data objects.
            val_data:   List of validation PyG Data objects.

        Returns:
            TrainingHistory with per-epoch metrics and best epoch info.

        Side effects:
            Saves best model checkpoint to config.checkpoint_dir /
            config.checkpoint_name.
        """
        if not train_data:
            raise ValueError("train_data is empty.")
        if not val_data:
            raise ValueError("val_data is empty.")

        set_seeds(self.config.random_seed)
        model = model.to(self.device)

        criterion = self._get_criterion(train_data)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history = TrainingHistory()
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        start_time = time.time()

        checkpoint_path = Path(self.config.checkpoint_dir) / self.config.checkpoint_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting training: %d epochs, patience=%d, lr=%.5f, seed=%d",
            self.config.epochs, self.config.patience,
            self.config.learning_rate, self.config.random_seed,
        )

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self._train_epoch(
                model, train_data, optimizer, criterion
            )
            val_loss, val_acc = self._eval_epoch(model, val_data, criterion)

            history.train_losses.append(round(train_loss, 6))
            history.val_losses.append(round(val_loss, 6))
            history.train_accuracies.append(round(train_acc, 6))
            history.val_accuracies.append(round(val_acc, 6))
            history.epochs_run = epoch

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                    "train_acc=%.3f  val_acc=%.3f",
                    epoch, self.config.epochs,
                    train_loss, val_loss, train_acc, val_acc,
                )

            # Early stopping check
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                history.best_epoch = epoch
                history.best_val_loss = round(best_val_loss, 6)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d, best_val_loss=%.4f).",
                        epoch, self.config.patience, best_val_loss,
                    )
                    history.stopped_early = True
                    break

        history.total_train_time_s = round(time.time() - start_time, 2)

        # Restore best weights and save checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)

        checkpoint = {
            "model_state_dict": best_state if best_state else model.state_dict(),
            "config": self.config.to_dict(),
            "history": history.to_dict(),
            "architecture": model.architecture_summary(),
            "random_seed": self.config.random_seed,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info("Checkpoint saved to %s", checkpoint_path)

        logger.info(
            "Training complete: best_epoch=%d, best_val_loss=%.4f, "
            "total_time=%.1fs",
            history.best_epoch, history.best_val_loss, history.total_train_time_s,
        )

        return history


def load_checkpoint(checkpoint_path: str, device: Optional[str] = None) -> Tuple[AttackGNN, dict]:
    """
    Load a saved AttackGNN checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device:         'cpu', 'cuda', or None (auto-detect).

    Returns:
        (model, checkpoint_dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]
    arch = checkpoint.get("architecture", {})

    model = AttackGNN(
        in_channels=arch.get("in_channels", 43),
        hidden_channels=config_dict.get("hidden_channels", 64),
        num_layers=config_dict.get("num_layers", 2),
        dropout=config_dict.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Loaded checkpoint from %s (best_epoch=%s)",
                checkpoint_path,
                checkpoint.get("history", {}).get("best_epoch", "?"))
    return model, checkpoint
