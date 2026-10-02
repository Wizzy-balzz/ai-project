"""
evaluator.py
------------
Evaluation pipeline for AttackGNN — node-level binary classification.

Design decisions:
- All metrics are computed from actual model predictions; nothing is fabricated.
- ROC-AUC and PR-AUC require both classes to be present in the evaluated split.
  If a split contains only one class, these metrics are reported as None with
  an explanation — NOT fabricated or suppressed.
- Confusion matrix is always reported (valid even for single-class splits).
- Accuracy is reported but highlighted as unreliable for imbalanced data.
- Evaluation is NEVER run on training data during test reporting.
- All numeric results are Python primitives for JSON serialization.
- The evaluator is a pure function — it does not modify the model.

Metric definitions (binary classification, class 1 = "attack involved"):
  Accuracy:   (TP + TN) / N
  Precision:  TP / (TP + FP)   — how many predicted positives are real
  Recall:     TP / (TP + FN)   — how many real positives are caught
  F1-score:   2 * P * R / (P + R)
  ROC-AUC:    area under the ROC curve (requires both classes)
  PR-AUC:     area under Precision-Recall curve (preferred for imbalanced data)

These are real sklearn metrics computed from real model outputs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from torch_geometric.data import Data

from backend.ml.model import AttackGNN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Holds all evaluation metrics for one split.

    Fields marked Optional are None when they cannot be computed
    (e.g., ROC-AUC with a single-class split).
    """
    split_name: str = ""                    # "train" | "val" | "test"
    n_scenarios: int = 0                    # Number of graph scenarios evaluated
    n_nodes: int = 0                        # Total nodes evaluated
    n_positive: int = 0                     # Nodes with y=1
    n_negative: int = 0                     # Nodes with y=0

    # Core metrics
    accuracy: float = 0.0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    # Ranking metrics (require both classes)
    roc_auc: Optional[float] = None
    roc_auc_available: bool = False
    roc_auc_unavailable_reason: str = ""

    pr_auc: Optional[float] = None
    pr_auc_available: bool = False
    pr_auc_unavailable_reason: str = ""

    # Confusion matrix: [[TN, FP], [FN, TP]]
    confusion_matrix: List[List[int]] = field(default_factory=list)

    # Class imbalance warning
    is_single_class: bool = False
    single_class_warning: str = ""

    # Data provenance
    data_is_synthetic: bool = True

    def to_dict(self) -> Dict:
        d = {
            "split_name": self.split_name,
            "n_scenarios": self.n_scenarios,
            "n_nodes": self.n_nodes,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "accuracy": round(self.accuracy, 6) if self.accuracy is not None else None,
            "precision": round(self.precision, 6) if self.precision is not None else None,
            "recall": round(self.recall, 6) if self.recall is not None else None,
            "f1": round(self.f1, 6) if self.f1 is not None else None,
            "roc_auc": round(self.roc_auc, 6) if self.roc_auc is not None else None,
            "roc_auc_available": self.roc_auc_available,
            "roc_auc_unavailable_reason": self.roc_auc_unavailable_reason,
            "pr_auc": round(self.pr_auc, 6) if self.pr_auc is not None else None,
            "pr_auc_available": self.pr_auc_available,
            "pr_auc_unavailable_reason": self.pr_auc_unavailable_reason,
            "confusion_matrix": self.confusion_matrix,
            "is_single_class": self.is_single_class,
            "single_class_warning": self.single_class_warning,
            "data_is_synthetic": self.data_is_synthetic,
        }
        return d


# ---------------------------------------------------------------------------
# Prediction collector
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: AttackGNN,
    data_list: List[Data],
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on all scenarios and collect predictions.

    Args:
        model:     Trained AttackGNN (must be in eval mode).
        data_list: List of PyG Data objects.
        device:    Torch device string.

    Returns:
        (y_true, y_pred, y_prob)
        - y_true: Ground truth labels, shape (N_total,), int array
        - y_pred: Predicted labels,    shape (N_total,), int array
        - y_prob: P(class=1),          shape (N_total,), float array
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for data in data_list:
        if data.x is None or data.y is None:
            logger.warning("Skipping scenario with missing x or y.")
            continue

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y

        logits = model(x, edge_index)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_true.append(y.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_prob.append(probs[:, 1].cpu().numpy())  # P(attack involved)

    if not all_true:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Computes evaluation metrics for AttackGNN on a dataset split.

    All metrics are derived from actual model predictions.
    No metrics are fabricated or hard-coded.

    Usage:
        evaluator = Evaluator(model, device="cpu")
        result = evaluator.evaluate(dataset.test_data, split_name="test")
        print(json.dumps(result.to_dict(), indent=2))
    """

    def __init__(self, model: AttackGNN, device: Optional[str] = None) -> None:
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)

    def evaluate(
        self,
        data_list: List[Data],
        split_name: str = "eval",
    ) -> EvaluationResult:
        """
        Evaluate model on a list of graph scenarios.

        Computes:
          accuracy, precision, recall, F1 (always)
          ROC-AUC, PR-AUC (only when both classes present)
          confusion matrix (always)

        Args:
            data_list:  PyG Data objects for this split (train, val, or test).
            split_name: Label for the split (used in logging and result).

        Returns:
            EvaluationResult with all metrics.
        """
        result = EvaluationResult(
            split_name=split_name,
            n_scenarios=len(data_list),
            data_is_synthetic=True,
        )

        y_true, y_pred, y_prob = collect_predictions(
            self.model, data_list, self.device
        )

        if len(y_true) == 0:
            logger.warning("[%s] No valid scenarios to evaluate.", split_name)
            return result

        result.n_nodes = int(len(y_true))
        result.n_positive = int((y_true == 1).sum())
        result.n_negative = int((y_true == 0).sum())

        # Single-class detection
        unique_classes = set(y_true.tolist())
        if len(unique_classes) < 2:
            result.is_single_class = True
            result.single_class_warning = (
                f"Split '{split_name}' contains only class(es) {unique_classes}. "
                "ROC-AUC and PR-AUC require both classes and cannot be computed. "
                "This is a dataset characteristic — the split lacks negative (or positive) "
                "examples across all scenarios in this split."
            )
            logger.warning("[%s] %s", split_name, result.single_class_warning)

        # Accuracy (always valid)
        result.accuracy = float(accuracy_score(y_true, y_pred))

        # Precision, Recall, F1 — handle zero-division gracefully
        try:
            result.precision = float(precision_score(
                y_true, y_pred, zero_division=0
            ))
            result.recall = float(recall_score(
                y_true, y_pred, zero_division=0
            ))
            result.f1 = float(f1_score(
                y_true, y_pred, zero_division=0
            ))
        except Exception as e:
            logger.warning("[%s] Precision/Recall/F1 computation failed: %s", split_name, e)

        # Confusion matrix: [[TN, FP], [FN, TP]]
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            result.confusion_matrix = cm.tolist()
        except Exception as e:
            logger.warning("[%s] Confusion matrix failed: %s", split_name, e)
            result.confusion_matrix = []

        # ROC-AUC (requires both classes)
        if result.is_single_class:
            result.roc_auc = None
            result.roc_auc_available = False
            result.roc_auc_unavailable_reason = (
                f"Only class(es) {unique_classes} present in '{split_name}' split. "
                "ROC-AUC is undefined for single-class distributions."
            )
        else:
            try:
                result.roc_auc = float(roc_auc_score(y_true, y_prob))
                result.roc_auc_available = True
            except ValueError as e:
                result.roc_auc = None
                result.roc_auc_available = False
                result.roc_auc_unavailable_reason = str(e)

        # PR-AUC (requires both classes)
        if result.is_single_class:
            result.pr_auc = None
            result.pr_auc_available = False
            result.pr_auc_unavailable_reason = (
                f"Only class(es) {unique_classes} present in '{split_name}' split. "
                "PR-AUC is undefined for single-class distributions."
            )
        else:
            try:
                result.pr_auc = float(average_precision_score(y_true, y_prob))
                result.pr_auc_available = True
            except ValueError as e:
                result.pr_auc = None
                result.pr_auc_available = False
                result.pr_auc_unavailable_reason = str(e)

        logger.info(
            "[%s] N=%d  pos=%d  neg=%d  acc=%.4f  prec=%s  rec=%s  F1=%s  "
            "ROC-AUC=%s  PR-AUC=%s",
            split_name, result.n_nodes, result.n_positive, result.n_negative,
            result.accuracy,
            f"{result.precision:.4f}" if result.precision is not None else "N/A",
            f"{result.recall:.4f}" if result.recall is not None else "N/A",
            f"{result.f1:.4f}" if result.f1 is not None else "N/A",
            f"{result.roc_auc:.4f}" if result.roc_auc is not None else "N/A",
            f"{result.pr_auc:.4f}" if result.pr_auc is not None else "N/A",
        )

        return result

    def evaluate_all_splits(
        self,
        train_data: List[Data],
        val_data: List[Data],
        test_data: List[Data],
        output_path: Optional[str] = None,
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate model on all three splits and optionally save to JSON.

        Args:
            train_data:  Training scenarios.
            val_data:    Validation scenarios.
            test_data:   Test scenarios.
            output_path: If provided, saves results as JSON to this path.

        Returns:
            dict with keys "train", "val", "test" mapping to EvaluationResult.
        """
        results = {
            "train": self.evaluate(train_data, split_name="train"),
            "val":   self.evaluate(val_data,   split_name="val"),
            "test":  self.evaluate(test_data,  split_name="test"),
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            serializable = {k: v.to_dict() for k, v in results.items()}
            with open(output_path, "w") as f:
                json.dump(serializable, f, indent=2)
            logger.info("Evaluation results saved to %s", output_path)

        return results
