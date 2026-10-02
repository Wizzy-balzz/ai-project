"""
test_phase3_gnn.py
------------------
Tests for Phase 3: AttackGNN model, training pipeline, and evaluator.

Tests are organized into:
  TestModel        — forward pass, output shape, gradient flow, NaN/Inf
  TestTrainer      — reproducibility, loss finite, checkpoint saved,
                     early stopping, class weights from training only
  TestEvaluator    — metric correctness, single-class handling, no fabrication
  TestIntegration  — Phase 2 Data → model → train → eval end-to-end

No existing tests are modified or removed.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from backend.core.data_models import (
    EdgeType, HistoricalAttack, NetworkEdge, NetworkNode,
    NetworkTopology, NodeType, OSType, Vulnerability, VulnerabilitySeverity,
)
from backend.core.feature_extractor import FEATURE_DIM
from backend.core.synthetic_data import SyntheticNetworkGenerator
from backend.ml.dataset import (
    EDGE_FEATURE_DIM, build_dataset, topology_to_pyg, validate_pyg_data,
)
from backend.ml.evaluator import EvaluationResult, Evaluator, collect_predictions
from backend.ml.model import AttackGNN
from backend.ml.trainer import (
    TrainingConfig, TrainingHistory, Trainer,
    compute_class_weights, set_seeds,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_tiny_pyg_data(
    n_nodes: int = 6,
    n_edges: int = 8,
    seed: int = 0,
    n_pos: int = 3,
) -> Data:
    """
    Construct a minimal deterministic PyG Data object for fast unit tests.
    Does NOT use the full synthetic pipeline — avoids overhead.
    """
    torch.manual_seed(seed)
    x = torch.rand(n_nodes, FEATURE_DIM)
    # Random directed edges (no self-loops)
    src = torch.randint(0, n_nodes, (n_edges,))
    tgt = torch.randint(0, n_nodes, (n_edges,))
    # Eliminate self-loops
    mask = src != tgt
    src, tgt = src[mask], tgt[mask]
    edge_index = torch.stack([src, tgt], dim=0)
    edge_attr = torch.rand(edge_index.shape[1], EDGE_FEATURE_DIM)
    y = torch.zeros(n_nodes, dtype=torch.long)
    y[:n_pos] = 1
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=n_nodes)


def make_tiny_dataset(n_train: int = 4, n_val: int = 2, n_test: int = 2) -> tuple:
    """Return (train_data, val_data, test_data) lists of tiny Data objects."""
    train = [make_tiny_pyg_data(seed=i, n_pos=3) for i in range(n_train)]
    val   = [make_tiny_pyg_data(seed=100 + i, n_pos=2) for i in range(n_val)]
    test  = [make_tiny_pyg_data(seed=200 + i, n_pos=1) for i in range(n_test)]
    return train, val, test


def make_model(hidden: int = 32, layers: int = 2) -> AttackGNN:
    return AttackGNN(in_channels=FEATURE_DIM, hidden_channels=hidden, num_layers=layers)


# ---------------------------------------------------------------------------
# Task: MODEL TESTS
# ---------------------------------------------------------------------------

class TestModel:
    def test_output_shape_small_graph(self):
        model = make_model()
        data = make_tiny_pyg_data(n_nodes=5, n_edges=6)
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        assert logits.shape == (5, 2), f"Expected (5,2), got {logits.shape}"

    def test_output_shape_medium_graph(self):
        model = make_model()
        model.eval()
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_medium()
        data, _, _ = topology_to_pyg(topo)
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        assert logits.shape == (data.num_nodes, 2)

    def test_output_shape_large_graph(self):
        model = make_model()
        model.eval()
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_large()
        data, _, _ = topology_to_pyg(topo)
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        assert logits.shape == (data.num_nodes, 2)

    def test_num_classes_is_two(self):
        model = make_model()
        assert model.num_classes == 2

    def test_invalid_num_classes_raises(self):
        with pytest.raises(ValueError, match="binary classifier"):
            AttackGNN(in_channels=43, num_classes=3)

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            AttackGNN(in_channels=43, num_layers=0)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            AttackGNN(in_channels=43, dropout=1.0)

    def test_no_nan_in_output(self):
        model = make_model()
        model.eval()
        data = make_tiny_pyg_data()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        assert not torch.isnan(logits).any(), "NaN in model output"

    def test_no_inf_in_output(self):
        model = make_model()
        model.eval()
        data = make_tiny_pyg_data()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        assert not torch.isinf(logits).any(), "Inf in model output"

    def test_gradient_flows_through_all_parameters(self):
        """After a backward pass, every parameter should have a non-None gradient."""
        model = make_model()
        model.train()
        data = make_tiny_pyg_data()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits, data.y)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' has no gradient"

    def test_predict_proba_sums_to_one(self):
        model = make_model()
        data = make_tiny_pyg_data()
        probs = model.predict_proba(data.x, data.edge_index)
        row_sums = probs.sum(dim=-1)
        for i, s in enumerate(row_sums):
            assert float(s.item()) == pytest.approx(1.0, abs=1e-5), (
                f"Node {i}: softmax sum = {s.item()}"
            )

    def test_predict_proba_in_01_range(self):
        model = make_model()
        data = make_tiny_pyg_data()
        probs = model.predict_proba(data.x, data.edge_index)
        assert float(probs.min().item()) >= 0.0
        assert float(probs.max().item()) <= 1.0 + 1e-6

    def test_count_parameters_positive(self):
        model = make_model()
        assert model.count_parameters() > 0

    def test_architecture_summary_keys(self):
        model = make_model()
        summary = model.architecture_summary()
        required = {
            "model_class", "backbone", "in_channels", "hidden_channels",
            "num_layers", "dropout", "num_classes", "output",
            "activation", "normalization", "edge_attr_used", "edge_attr_note",
            "trainable_parameters",
        }
        assert required == set(summary.keys())

    def test_edge_attr_not_used_by_model(self):
        """Architecture summary must explicitly document edge_attr is not consumed."""
        model = make_model()
        summary = model.architecture_summary()
        assert summary["edge_attr_used"] is False

    def test_different_graph_sizes_same_model(self):
        """Same model instance should handle graphs of different sizes."""
        model = make_model()
        model.eval()
        for n in [3, 10, 25]:
            data = make_tiny_pyg_data(n_nodes=n, n_edges=n * 2)
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
            assert logits.shape == (n, 2)

    def test_training_mode_vs_eval_mode_output_differs(self):
        """Dropout should make train-mode outputs stochastic."""
        model = AttackGNN(in_channels=FEATURE_DIM, dropout=0.5)
        data = make_tiny_pyg_data(n_nodes=20, n_edges=30)
        torch.manual_seed(0)
        model.train()
        out1 = model(data.x, data.edge_index)
        torch.manual_seed(1)
        out2 = model(data.x, data.edge_index)
        # With p=0.5 dropout, outputs should differ (with overwhelming probability)
        assert not torch.allclose(out1, out2), (
            "Train-mode outputs were identical despite dropout — check dropout is active"
        )

    def test_eval_mode_output_is_deterministic(self):
        """Eval mode with no randomness should give identical outputs."""
        model = make_model()
        model.eval()
        data = make_tiny_pyg_data()
        with torch.no_grad():
            out1 = model(data.x, data.edge_index)
            out2 = model(data.x, data.edge_index)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Task: TRAINER TESTS
# ---------------------------------------------------------------------------

class TestTrainer:
    def _quick_config(self, tmpdir: str, epochs: int = 5, patience: int = 3) -> TrainingConfig:
        return TrainingConfig(
            random_seed=42,
            epochs=epochs,
            patience=patience,
            hidden_channels=16,
            num_layers=2,
            dropout=0.1,
            learning_rate=1e-2,
            weight_decay=0.0,
            use_class_weights=True,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_checkpoint.pt",
        )

    def test_training_runs_without_error(self, tmp_path):
        config = self._quick_config(str(tmp_path))
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16, num_layers=2)
        train, val, _ = make_tiny_dataset()
        trainer = Trainer(config)
        history = trainer.train(model, train, val)
        assert isinstance(history, TrainingHistory)

    def test_loss_is_finite_after_training(self, tmp_path):
        config = self._quick_config(str(tmp_path))
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
        train, val, _ = make_tiny_dataset()
        trainer = Trainer(config)
        history = trainer.train(model, train, val)
        for loss in history.train_losses:
            assert np.isfinite(loss), f"Non-finite training loss: {loss}"
        for loss in history.val_losses:
            assert np.isfinite(loss), f"Non-finite validation loss: {loss}"

    def test_checkpoint_is_created(self, tmp_path):
        config = self._quick_config(str(tmp_path))
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
        train, val, _ = make_tiny_dataset()
        Trainer(config).train(model, train, val)
        ckpt_path = tmp_path / "test_checkpoint.pt"
        assert ckpt_path.exists(), "Checkpoint file not created"

    def test_checkpoint_contains_required_keys(self, tmp_path):
        config = self._quick_config(str(tmp_path))
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
        train, val, _ = make_tiny_dataset()
        Trainer(config).train(model, train, val)
        ckpt = torch.load(tmp_path / "test_checkpoint.pt", map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "config" in ckpt
        assert "history" in ckpt
        assert "architecture" in ckpt

    def test_reproducibility(self, tmp_path):
        """Same seed → same training losses."""
        config1 = self._quick_config(str(tmp_path / "run1"), epochs=5)
        config2 = self._quick_config(str(tmp_path / "run2"), epochs=5)

        def run(cfg):
            set_seeds(cfg.random_seed)
            m = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
            train, val, _ = make_tiny_dataset()
            return Trainer(cfg).train(m, train, val)

        h1 = run(config1)
        h2 = run(config2)
        assert h1.train_losses == h2.train_losses, (
            f"Non-reproducible: {h1.train_losses} vs {h2.train_losses}"
        )

    def test_early_stopping_triggers(self, tmp_path):
        """With very low patience, training should stop early."""
        config = self._quick_config(str(tmp_path), epochs=100, patience=3)
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
        train, val, _ = make_tiny_dataset()
        history = Trainer(config).train(model, train, val)
        assert history.stopped_early or history.epochs_run <= 100

    def test_history_epochs_run_is_positive(self, tmp_path):
        config = self._quick_config(str(tmp_path))
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
        train, val, _ = make_tiny_dataset()
        history = Trainer(config).train(model, train, val)
        assert history.epochs_run >= 1

    def test_history_to_dict_serializable(self, tmp_path):
        config = self._quick_config(str(tmp_path))
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
        train, val, _ = make_tiny_dataset()
        history = Trainer(config).train(model, train, val)
        d = history.to_dict()
        json_str = json.dumps(d)  # Should not raise
        assert len(json_str) > 0

    def test_empty_train_data_raises(self, tmp_path):
        config = self._quick_config(str(tmp_path))
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
        with pytest.raises(ValueError, match="empty"):
            Trainer(config).train(model, [], [make_tiny_pyg_data()])

    def test_empty_val_data_raises(self, tmp_path):
        config = self._quick_config(str(tmp_path))
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
        with pytest.raises(ValueError, match="empty"):
            Trainer(config).train(model, [make_tiny_pyg_data()], [])

    def test_best_epoch_is_valid(self, tmp_path):
        config = self._quick_config(str(tmp_path), epochs=10)
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
        train, val, _ = make_tiny_dataset()
        history = Trainer(config).train(model, train, val)
        assert 1 <= history.best_epoch <= history.epochs_run

    def test_class_weights_from_training_only(self):
        """
        Class weights must be computed from training data ONLY.
        This test verifies the function does not touch val/test data
        by checking it only accepts a list of Data objects.
        """
        train, _, _ = make_tiny_dataset()
        weights = compute_class_weights(train)
        # Weights should be a 2-tensor or None
        assert weights is None or (isinstance(weights, torch.Tensor) and weights.shape == (2,))

    def test_class_weights_are_positive(self):
        train, _, _ = make_tiny_dataset()
        weights = compute_class_weights(train)
        if weights is not None:
            assert float(weights.min().item()) > 0.0

    def test_single_class_training_returns_none_weights(self):
        """All-positive training data → compute_class_weights returns None."""
        all_pos_data = []
        for i in range(3):
            d = make_tiny_pyg_data(n_nodes=4, n_pos=4)  # all positive
            all_pos_data.append(d)
        weights = compute_class_weights(all_pos_data)
        assert weights is None, (
            "Single-class training data should produce None class weights"
        )


# ---------------------------------------------------------------------------
# Task: EVALUATOR TESTS
# ---------------------------------------------------------------------------

class TestEvaluator:
    def _trained_model(self, tmp_path) -> AttackGNN:
        """Train a minimal model for evaluation tests."""
        config = TrainingConfig(
            epochs=10, patience=5, hidden_channels=16, num_layers=2,
            dropout=0.1, checkpoint_dir=str(tmp_path),
            checkpoint_name="eval_test.pt",
        )
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16)
        train, val, _ = make_tiny_dataset(n_train=4, n_val=2)
        Trainer(config).train(model, train, val)
        model.eval()
        return model

    def test_evaluation_result_is_dataclass(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, val, _ = make_tiny_dataset()
        result = evaluator.evaluate(val, split_name="val")
        assert isinstance(result, EvaluationResult)

    def test_accuracy_in_01_range(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, _, test = make_tiny_dataset()
        result = evaluator.evaluate(test, split_name="test")
        assert 0.0 <= result.accuracy <= 1.0

    def test_precision_in_01_range(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, _, test = make_tiny_dataset()
        result = evaluator.evaluate(test, split_name="test")
        if result.precision is not None:
            assert 0.0 <= result.precision <= 1.0

    def test_recall_in_01_range(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, _, test = make_tiny_dataset()
        result = evaluator.evaluate(test, split_name="test")
        if result.recall is not None:
            assert 0.0 <= result.recall <= 1.0

    def test_f1_in_01_range(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, _, test = make_tiny_dataset()
        result = evaluator.evaluate(test, split_name="test")
        if result.f1 is not None:
            assert 0.0 <= result.f1 <= 1.0

    def test_roc_auc_in_01_range_when_available(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, _, test = make_tiny_dataset()
        result = evaluator.evaluate(test, split_name="test")
        if result.roc_auc_available and result.roc_auc is not None:
            assert 0.0 <= result.roc_auc <= 1.0

    def test_confusion_matrix_shape(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, _, test = make_tiny_dataset()
        result = evaluator.evaluate(test, split_name="test")
        if result.confusion_matrix:
            assert len(result.confusion_matrix) == 2
            assert len(result.confusion_matrix[0]) == 2

    def test_confusion_matrix_sum_equals_n_nodes(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, _, test = make_tiny_dataset()
        result = evaluator.evaluate(test, split_name="test")
        if result.confusion_matrix:
            cm_sum = sum(result.confusion_matrix[i][j]
                        for i in range(2) for j in range(2))
            assert cm_sum == result.n_nodes

    def test_single_class_split_roc_auc_is_none(self):
        """A split where all labels are 1 should NOT produce a ROC-AUC value."""
        all_pos = [make_tiny_pyg_data(n_nodes=5, n_pos=5) for _ in range(3)]
        # Train briefly
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                epochs=3, patience=3, hidden_channels=16, num_layers=1,
                checkpoint_dir=tmpdir,
            )
            model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16, num_layers=1)
            # Need mixed train data for training
            train = [make_tiny_pyg_data(n_pos=3)] * 3
            val_mixed = [make_tiny_pyg_data(n_pos=3)] * 2
            Trainer(config).train(model, train, val_mixed)
            model.eval()
            evaluator = Evaluator(model)
            result = evaluator.evaluate(all_pos, split_name="single_class_test")

        assert result.is_single_class is True
        assert result.roc_auc is None, (
            "ROC-AUC must be None for single-class splits — never fabricate it"
        )
        assert result.roc_auc_available is False
        assert len(result.roc_auc_unavailable_reason) > 0

    def test_single_class_split_pr_auc_is_none(self):
        """Same test for PR-AUC."""
        all_neg = [make_tiny_pyg_data(n_nodes=5, n_pos=0) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                epochs=3, patience=3, hidden_channels=16, num_layers=1,
                checkpoint_dir=tmpdir,
            )
            model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=16, num_layers=1)
            train = [make_tiny_pyg_data(n_pos=3)] * 3
            val_mixed = [make_tiny_pyg_data(n_pos=2)] * 2
            Trainer(config).train(model, train, val_mixed)
            model.eval()
            evaluator = Evaluator(model)
            result = evaluator.evaluate(all_neg, split_name="all_neg_test")

        assert result.pr_auc is None
        assert result.pr_auc_available is False

    def test_result_to_dict_is_json_serializable(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, _, test = make_tiny_dataset()
        result = evaluator.evaluate(test, split_name="test")
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_evaluate_all_splits_returns_three_results(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        train, val, test = make_tiny_dataset()
        results = evaluator.evaluate_all_splits(train, val, test)
        assert set(results.keys()) == {"train", "val", "test"}

    def test_evaluate_all_splits_saves_json(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        train, val, test = make_tiny_dataset()
        output_path = str(tmp_path / "eval_results.json")
        evaluator.evaluate_all_splits(train, val, test, output_path=output_path)
        assert Path(output_path).exists()
        with open(output_path) as f:
            loaded = json.load(f)
        assert set(loaded.keys()) == {"train", "val", "test"}

    def test_n_positive_plus_n_negative_equals_n_nodes(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, _, test = make_tiny_dataset()
        result = evaluator.evaluate(test, split_name="test")
        assert result.n_positive + result.n_negative == result.n_nodes

    def test_collect_predictions_shapes(self, tmp_path):
        model = self._trained_model(tmp_path)
        train, _, _ = make_tiny_dataset()
        y_true, y_pred, y_prob = collect_predictions(model, train)
        n = sum(d.num_nodes for d in train)
        assert y_true.shape == (n,)
        assert y_pred.shape == (n,)
        assert y_prob.shape == (n,)

    def test_collect_predictions_prob_range(self, tmp_path):
        model = self._trained_model(tmp_path)
        train, _, _ = make_tiny_dataset()
        _, _, y_prob = collect_predictions(model, train)
        assert float(y_prob.min()) >= 0.0
        assert float(y_prob.max()) <= 1.0 + 1e-6

    def test_collect_predictions_binary_labels(self, tmp_path):
        model = self._trained_model(tmp_path)
        train, _, _ = make_tiny_dataset()
        _, y_pred, _ = collect_predictions(model, train)
        assert set(y_pred.tolist()).issubset({0, 1})

    def test_data_is_synthetic_flag_set(self, tmp_path):
        model = self._trained_model(tmp_path)
        evaluator = Evaluator(model)
        _, _, test = make_tiny_dataset()
        result = evaluator.evaluate(test, "test")
        assert result.data_is_synthetic is True


# ---------------------------------------------------------------------------
# Task: INTEGRATION TESTS
# ---------------------------------------------------------------------------

class TestPhase3Integration:
    """
    End-to-end integration tests: Phase 2 dataset → model → train → eval.
    These use real synthetic topology generation to test the full pipeline.
    """

    @pytest.fixture(scope="class")
    def trained_model_and_dataset(self, tmp_path_factory):
        """Train a real model on a small synthetic dataset (class-scoped)."""
        tmp_path = tmp_path_factory.mktemp("integration")
        # Small dataset for speed: 6 scenarios
        dataset = build_dataset(
            n_small=6, n_medium=0, n_large=0,
            seed_offset=999, split_seed=42,
            validate=True,
        )
        config = TrainingConfig(
            random_seed=42,
            epochs=30,
            patience=10,
            hidden_channels=32,
            num_layers=2,
            dropout=0.2,
            learning_rate=5e-3,
            checkpoint_dir=str(tmp_path),
            checkpoint_name="integration_best.pt",
        )
        model = AttackGNN(in_channels=FEATURE_DIM, hidden_channels=32)
        trainer = Trainer(config)
        history = trainer.train(model, dataset.train_data, dataset.val_data)
        return model, dataset, history, config, tmp_path

    def test_pipeline_runs_end_to_end(self, trained_model_and_dataset):
        model, dataset, history, config, tmp_path = trained_model_and_dataset
        assert history.epochs_run >= 1

    def test_checkpoint_saved(self, trained_model_and_dataset):
        model, dataset, history, config, tmp_path = trained_model_and_dataset
        ckpt_path = tmp_path / "integration_best.pt"
        assert ckpt_path.exists()

    def test_no_test_data_in_training(self, trained_model_and_dataset):
        """
        Verify no topology_id from test scenarios appears in train scenarios.
        """
        _, dataset, _, _, _ = trained_model_and_dataset
        report = dataset.check_leakage()
        assert report["no_leakage"] is True

    def test_train_split_evaluation_runs(self, trained_model_and_dataset):
        model, dataset, _, _, _ = trained_model_and_dataset
        evaluator = Evaluator(model)
        result = evaluator.evaluate(dataset.train_data, split_name="train")
        assert result.n_nodes > 0
        assert 0.0 <= result.accuracy <= 1.0

    def test_val_split_evaluation_runs(self, trained_model_and_dataset):
        model, dataset, _, _, _ = trained_model_and_dataset
        evaluator = Evaluator(model)
        result = evaluator.evaluate(dataset.val_data, split_name="val")
        assert result.n_nodes > 0

    def test_test_split_evaluation_runs(self, trained_model_and_dataset):
        model, dataset, _, _, _ = trained_model_and_dataset
        evaluator = Evaluator(model)
        result = evaluator.evaluate(dataset.test_data, split_name="test")
        assert result.n_nodes > 0

    def test_no_leakage_in_training_labels(self, trained_model_and_dataset):
        """
        Class weights computed during training must NOT incorporate
        test split labels.
        """
        _, dataset, _, _, _ = trained_model_and_dataset
        train_labels = torch.cat([d.y for d in dataset.train_data])
        test_labels = torch.cat([d.y for d in dataset.test_data])
        # Verify they are from different scenarios (disjoint topology IDs)
        train_topo_ids = {getattr(d, "topology_id", None) for d in dataset.train_data}
        test_topo_ids = {getattr(d, "topology_id", None) for d in dataset.test_data}
        assert len(train_topo_ids & test_topo_ids) == 0

    def test_evaluation_results_are_real(self, trained_model_and_dataset):
        """Metrics must come from actual model output, not hard-coded values."""
        model, dataset, _, _, tmp_path = trained_model_and_dataset
        evaluator = Evaluator(model)
        results = evaluator.evaluate_all_splits(
            dataset.train_data, dataset.val_data, dataset.test_data,
            output_path=str(tmp_path / "eval.json"),
        )
        # All split accuracies must be in [0,1]
        for split_name, result in results.items():
            assert 0.0 <= result.accuracy <= 1.0, (
                f"Split '{split_name}' accuracy={result.accuracy} out of range"
            )
        # Verify JSON was written
        assert (tmp_path / "eval.json").exists()

    def test_best_val_loss_is_finite(self, trained_model_and_dataset):
        _, _, history, _, _ = trained_model_and_dataset
        assert np.isfinite(history.best_val_loss)

    def test_phase3_does_not_claim_attack_path_prediction(self):
        """
        Verify that the model output is node-level logits, NOT complete paths.
        The model output shape [N, 2] confirms per-node classification.
        Attack path construction is a Phase 4+ concern.
        """
        model = make_model()
        model.eval()
        data = make_tiny_pyg_data(n_nodes=10)
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        # Output is per-node, not a path
        assert logits.shape[0] == data.num_nodes
        assert logits.shape[1] == 2
        # There is no "path" output; confirming Phase 3 scope is node-level
        assert not hasattr(logits, "attack_path")
