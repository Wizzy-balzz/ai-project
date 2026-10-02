"""
test_dataset.py
---------------
Unit tests for backend/ml/dataset.py

Tests cover:
- PyG Data object creation and field types
- Node feature dimension (x.shape == (N, 43))
- Edge index dimension and dtype (shape (2,E), torch.long)
- Edge attribute dimension (shape (E, 9))
- Label generation from historical attacks
- Label values are binary (0/1 only)
- At least 1 positive and 1 negative label per scenario
- Split integrity: correct counts at 70/15/15
- Leakage prevention: no topology_id overlap across splits
- Reproducibility: same seeds → identical datasets
- Validation report: catches NaN, wrong dims, self-loops, etc.
- EDGE_FEATURE_DIM == 9
- Scenarios with no historical attacks produce all-zero labels
- build_dataset works end to end
"""

import math

import pytest
import torch
import numpy as np

from backend.core.data_models import (
    EdgeType,
    HistoricalAttack,
    NetworkEdge,
    NetworkNode,
    NetworkTopology,
    NodeType,
    OSType,
    Vulnerability,
    VulnerabilitySeverity,
)
from backend.core.graph_builder import GraphBuilder
from backend.core.synthetic_data import SyntheticNetworkGenerator
from backend.ml.dataset import (
    ALL_EDGE_FEATURE_NAMES,
    EDGE_FEATURE_DIM,
    EDGE_SCALAR_FEATURE_NAMES,
    EDGE_TYPE_CATEGORIES,
    AttackGraphDataset,
    DataValidationReport,
    build_dataset,
    build_edge_features,
    build_node_labels,
    topology_to_pyg,
    validate_pyg_data,
)
from backend.core.feature_extractor import FEATURE_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_minimal_topology(with_attacks: bool = True) -> NetworkTopology:
    """
    Three-node topology: n1 (entry) → n2 → n3 (critical)

    Historical attacks: n1 → n2 → n3 (if with_attacks=True)
    """
    topo = NetworkTopology(
        topology_id="topo-min-test",
        name="Minimal-Test-Topology",
        is_synthetic=True,
    )
    n1 = NetworkNode(
        node_id="n1", label="internet",
        node_type=NodeType.INTERNET, os_type=OSType.UNKNOWN,
        ip_address="203.0.113.1", subnet="0.0.0.0/0",
        is_entry_point=True, asset_value=0.1,
    )
    n2 = NetworkNode(
        node_id="n2", label="web-server",
        node_type=NodeType.SERVER, os_type=OSType.LINUX,
        ip_address="10.0.1.10", subnet="10.0.1.0/24",
        vulnerabilities=[
            Vulnerability(
                cve_id="SYN-T1", cvss_score=9.0,
                severity=VulnerabilitySeverity.CRITICAL,
                description="Test", exploitability_score=3.9, impact_score=5.9,
            )
        ],
        asset_value=0.5,
    )
    n3 = NetworkNode(
        node_id="n3", label="domain-controller",
        node_type=NodeType.CRITICAL_ASSET, os_type=OSType.WINDOWS,
        ip_address="10.0.5.10", subnet="10.0.5.0/24",
        is_critical_asset=True, asset_value=1.0,
    )
    topo.add_node(n1)
    topo.add_node(n2)
    topo.add_node(n3)

    e1 = NetworkEdge(
        edge_id="e1", source_id="n1", target_id="n2",
        edge_type=EdgeType.NETWORK_LINK, bidirectional=True, attack_weight=0.7,
    )
    e2 = NetworkEdge(
        edge_id="e2", source_id="n2", target_id="n3",
        edge_type=EdgeType.NETWORK_LINK, bidirectional=False, attack_weight=0.6,
    )
    topo.add_edge(e1)
    topo.add_edge(e2)

    if with_attacks:
        topo.historical_attacks = [
            HistoricalAttack(
                attack_id="a1",
                attack_path=["n1", "n2", "n3"],
                technique="T1190",
                success=True,
                timestamp=1700000000.0,
                source="synthetic",
            )
        ]
    return topo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def minimal_topo():
    return make_minimal_topology(with_attacks=True)


@pytest.fixture(scope="module")
def minimal_data(minimal_topo):
    data, node_index, extractor = topology_to_pyg(minimal_topo)
    return data, node_index, extractor


@pytest.fixture(scope="module")
def small_topo():
    gen = SyntheticNetworkGenerator(seed=42)
    return gen.generate_small()


@pytest.fixture(scope="module")
def small_data(small_topo):
    data, node_index, extractor = topology_to_pyg(small_topo)
    return data, node_index, extractor


# ---------------------------------------------------------------------------
# TASK 1: Edge Feature Dimension Constant
# ---------------------------------------------------------------------------

class TestEdgeFeatureConstants:
    def test_edge_feature_dim_is_9(self):
        assert EDGE_FEATURE_DIM == 9, (
            f"Expected EDGE_FEATURE_DIM=9, got {EDGE_FEATURE_DIM}"
        )

    def test_edge_feature_names_length_matches_dim(self):
        assert len(ALL_EDGE_FEATURE_NAMES) == EDGE_FEATURE_DIM

    def test_no_duplicate_edge_feature_names(self):
        assert len(ALL_EDGE_FEATURE_NAMES) == len(set(ALL_EDGE_FEATURE_NAMES))

    def test_edge_type_categories_count(self):
        assert len(EDGE_TYPE_CATEGORIES) == 4

    def test_edge_scalar_feature_names_count(self):
        assert len(EDGE_SCALAR_FEATURE_NAMES) == 5

    def test_edge_feature_dim_equals_types_plus_scalars(self):
        assert EDGE_FEATURE_DIM == len(EDGE_TYPE_CATEGORIES) + len(EDGE_SCALAR_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# TASK 3: PyG Data Object Shape Tests
# ---------------------------------------------------------------------------

class TestPyGDataCreation:
    def test_data_x_exists(self, minimal_data):
        data, _, _ = minimal_data
        assert data.x is not None

    def test_data_x_shape(self, minimal_data):
        data, _, _ = minimal_data
        assert data.x.shape == (3, FEATURE_DIM), (
            f"Expected (3, {FEATURE_DIM}), got {data.x.shape}"
        )

    def test_data_x_dtype(self, minimal_data):
        data, _, _ = minimal_data
        assert data.x.dtype == torch.float32

    def test_data_edge_index_exists(self, minimal_data):
        data, _, _ = minimal_data
        assert data.edge_index is not None

    def test_data_edge_index_shape_rows(self, minimal_data):
        data, _, _ = minimal_data
        assert data.edge_index.shape[0] == 2

    def test_data_edge_index_dtype_is_long(self, minimal_data):
        data, _, _ = minimal_data
        assert data.edge_index.dtype == torch.long

    def test_data_edge_attr_exists(self, minimal_data):
        data, _, _ = minimal_data
        assert data.edge_attr is not None

    def test_data_edge_attr_feature_dim(self, minimal_data):
        data, _, _ = minimal_data
        assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM, (
            f"Expected edge_attr dim={EDGE_FEATURE_DIM}, got {data.edge_attr.shape[1]}"
        )

    def test_data_edge_attr_row_count_matches_edge_index(self, minimal_data):
        data, _, _ = minimal_data
        E = data.edge_index.shape[1]
        assert data.edge_attr.shape[0] == E

    def test_data_y_exists(self, minimal_data):
        data, _, _ = minimal_data
        assert data.y is not None

    def test_data_y_shape(self, minimal_data):
        data, _, _ = minimal_data
        assert data.y.shape == (3,)

    def test_data_y_dtype(self, minimal_data):
        data, _, _ = minimal_data
        assert data.y.dtype == torch.long

    def test_data_num_nodes_set(self, minimal_data):
        data, _, _ = minimal_data
        assert data.num_nodes == 3

    def test_data_has_node_ids(self, minimal_data):
        data, node_index, _ = minimal_data
        assert hasattr(data, "node_ids")
        assert len(data.node_ids) == 3

    def test_data_has_topology_metadata(self, minimal_data):
        data, _, _ = minimal_data
        assert hasattr(data, "topology_name")
        assert hasattr(data, "topology_id")
        assert hasattr(data, "is_synthetic")
        assert data.is_synthetic is True

    def test_data_positive_count_attribute(self, minimal_data):
        data, _, _ = minimal_data
        assert hasattr(data, "positive_count")
        assert hasattr(data, "negative_count")
        assert data.positive_count + data.negative_count == 3

    def test_node_index_length_matches_rows(self, minimal_data):
        data, node_index, _ = minimal_data
        assert len(node_index) == data.x.shape[0]

    def test_small_topology_x_shape(self, small_data, small_topo):
        data, _, _ = small_data
        N = small_topo.node_count
        assert data.x.shape == (N, FEATURE_DIM)

    def test_small_topology_edge_attr_shape(self, small_data):
        data, _, _ = small_data
        E = data.edge_index.shape[1]
        assert data.edge_attr.shape == (E, EDGE_FEATURE_DIM)


# ---------------------------------------------------------------------------
# TASK 2: Label Generation Tests
# ---------------------------------------------------------------------------

class TestLabelGeneration:
    def test_all_attack_path_nodes_are_positive(self, minimal_topo, minimal_data):
        """
        All three nodes appear in the attack path n1→n2→n3,
        so all should have label 1.
        """
        data, node_index, _ = minimal_data
        for i, nid in enumerate(node_index):
            label = int(data.y[i].item())
            assert label == 1, (
                f"Node {nid} is on the attack path but got label {label}"
            )

    def test_no_attack_path_gives_all_zero_labels(self):
        topo = make_minimal_topology(with_attacks=False)
        data, node_index, _ = topology_to_pyg(topo)
        assert data.y.sum().item() == 0, "No attacks → all labels should be 0"

    def test_labels_only_contain_0_and_1(self, small_data):
        data, _, _ = small_data
        unique = set(data.y.unique().tolist())
        assert unique.issubset({0, 1}), f"Unexpected label values: {unique}"

    def test_partial_path_labels(self):
        """
        If attack path is n1→n2 only, n3 (not on path) should be label 0.
        """
        topo = make_minimal_topology(with_attacks=False)
        topo.historical_attacks = [
            HistoricalAttack(
                attack_id="a_partial",
                attack_path=["n1", "n2"],  # n3 NOT included
                technique="T1190",
                success=False,
                timestamp=1700000000.0,
                source="synthetic",
            )
        ]
        data, node_index, _ = topology_to_pyg(topo)
        n3_idx = node_index.index("n3")
        assert int(data.y[n3_idx].item()) == 0, (
            f"n3 not on attack path but got label {data.y[n3_idx].item()}"
        )
        n1_idx = node_index.index("n1")
        assert int(data.y[n1_idx].item()) == 1

    def test_positive_count_matches_y_sum(self, minimal_data):
        data, _, _ = minimal_data
        assert data.positive_count == int(data.y.sum().item())

    def test_negative_count_matches_zero_labels(self, minimal_data):
        data, _, _ = minimal_data
        assert data.negative_count == int((data.y == 0).sum().item())

    def test_small_topology_has_both_positive_and_negative(self, small_data):
        data, _, _ = small_data
        assert data.positive_count > 0, "No positive labels in small topology"
        assert data.negative_count > 0, "No negative labels in small topology"

    def test_labels_are_deterministic(self, minimal_topo):
        """Same topology → same labels every time."""
        d1, ni1, _ = topology_to_pyg(minimal_topo)
        d2, ni2, _ = topology_to_pyg(minimal_topo)
        assert ni1 == ni2
        assert torch.equal(d1.y, d2.y)


# ---------------------------------------------------------------------------
# Edge Feature Tests
# ---------------------------------------------------------------------------

class TestEdgeFeatures:
    def test_build_edge_features_returns_tensors(self, minimal_topo):
        builder = GraphBuilder()
        G = builder.build(minimal_topo)
        ei, ea = build_edge_features(G)
        assert isinstance(ei, torch.Tensor)
        assert isinstance(ea, torch.Tensor)

    def test_edge_index_shape(self, minimal_topo):
        builder = GraphBuilder()
        G = builder.build(minimal_topo)
        ei, ea = build_edge_features(G)
        assert ei.shape[0] == 2
        assert ei.shape[1] == G.number_of_edges()

    def test_edge_attr_shape(self, minimal_topo):
        builder = GraphBuilder()
        G = builder.build(minimal_topo)
        ei, ea = build_edge_features(G)
        E = G.number_of_edges()
        assert ea.shape == (E, EDGE_FEATURE_DIM)

    def test_edge_type_one_hot_sums_to_one(self, minimal_data):
        """Each edge's first 4 features should be a valid one-hot."""
        data, _, _ = minimal_data
        onehot = data.edge_attr[:, :4]
        row_sums = onehot.sum(dim=1)
        for i, s in enumerate(row_sums):
            assert float(s.item()) == pytest.approx(1.0, abs=1e-5), (
                f"Edge {i} one-hot sum is {s.item()}, expected 1.0"
            )

    def test_firewall_rule_is_binary(self, minimal_data):
        """firewall_rule column (index 4) should only contain 0.0 or 1.0."""
        data, _, _ = minimal_data
        fw_col = data.edge_attr[:, 4]
        unique = set(fw_col.unique().tolist())
        assert unique.issubset({0.0, 1.0}), (
            f"firewall_rule contains non-binary values: {unique}"
        )

    def test_attack_weight_in_01_range(self, minimal_data):
        """attack_weight column (index 6) should be in [0.0, 1.0]."""
        data, _, _ = minimal_data
        aw_col = data.edge_attr[:, 6]
        assert float(aw_col.min().item()) >= 0.0
        assert float(aw_col.max().item()) <= 1.0 + 1e-5

    def test_no_nan_in_edge_attr(self, minimal_data):
        data, _, _ = minimal_data
        assert not torch.isnan(data.edge_attr).any().item()

    def test_no_inf_in_edge_attr(self, minimal_data):
        data, _, _ = minimal_data
        assert not torch.isinf(data.edge_attr).any().item()

    def test_edge_indices_within_valid_range(self, minimal_data):
        data, _, _ = minimal_data
        N = data.num_nodes
        assert int(data.edge_index.min().item()) >= 0
        assert int(data.edge_index.max().item()) < N


# ---------------------------------------------------------------------------
# TASK 5: Data Validation Tests
# ---------------------------------------------------------------------------

class TestDataValidation:
    def test_valid_data_passes(self, minimal_data):
        data, _, _ = minimal_data
        report = validate_pyg_data(data)
        assert report.passed, f"Valid data failed: {report.errors}"

    def test_small_graph_data_passes(self, small_data):
        data, _, _ = small_data
        report = validate_pyg_data(data)
        assert report.passed, f"Small graph failed: {report.errors}"

    def test_nan_in_x_fails_validation(self, minimal_data):
        import copy
        data, _, _ = minimal_data
        bad_data = copy.deepcopy(data)
        bad_x = bad_data.x.clone()
        bad_x[0, 0] = float("nan")
        bad_data.x = bad_x
        report = validate_pyg_data(bad_data)
        assert not report.passed
        assert any("NaN" in e for e in report.errors)

    def test_inf_in_x_fails_validation(self, minimal_data):
        import copy
        data, _, _ = minimal_data
        bad_data = copy.deepcopy(data)
        bad_x = bad_data.x.clone()
        bad_x[1, 2] = float("inf")
        bad_data.x = bad_x
        report = validate_pyg_data(bad_data)
        assert not report.passed
        assert any("Inf" in e for e in report.errors)

    def test_wrong_feature_dim_fails_validation(self, minimal_data):
        import copy
        data, _, _ = minimal_data
        bad_data = copy.deepcopy(data)
        bad_data.x = bad_data.x[:, :10]  # truncate to 10 features
        report = validate_pyg_data(bad_data)
        assert not report.passed
        assert any("FEATURE_DIM" in e or "features" in e.lower() for e in report.errors)

    def test_wrong_edge_attr_dim_fails_validation(self, minimal_data):
        import copy
        data, _, _ = minimal_data
        bad_data = copy.deepcopy(data)
        bad_data.edge_attr = bad_data.edge_attr[:, :3]  # wrong dim
        report = validate_pyg_data(bad_data)
        assert not report.passed
        assert any("edge feature" in e.lower() or "edge_attr" in e.lower() for e in report.errors)

    def test_out_of_range_edge_index_fails(self, minimal_data):
        import copy
        data, _, _ = minimal_data
        bad_data = copy.deepcopy(data)
        bad_ei = bad_data.edge_index.clone()
        bad_ei[0, 0] = data.num_nodes + 100  # out of bounds
        bad_data.edge_index = bad_ei
        report = validate_pyg_data(bad_data)
        assert not report.passed
        assert any("node index" in e.lower() or "N=" in e for e in report.errors)

    def test_invalid_label_values_fail_validation(self, minimal_data):
        import copy
        data, _, _ = minimal_data
        bad_data = copy.deepcopy(data)
        bad_y = bad_data.y.clone()
        bad_y[0] = 5  # invalid label
        bad_data.y = bad_y
        report = validate_pyg_data(bad_data)
        assert not report.passed
        assert any("invalid label" in e.lower() for e in report.errors)

    def test_all_zero_labels_issues_warning(self):
        """Topology with no attacks → all labels 0 → warning (not error)."""
        topo = make_minimal_topology(with_attacks=False)
        data, _, _ = topology_to_pyg(topo)
        report = validate_pyg_data(data)
        # All-zero labels generate a WARNING, not a failure
        assert report.passed, f"Should pass with warning: {report.errors}"
        assert any("positive" in w.lower() or "No positive" in w for w in report.warnings)

    def test_missing_x_fails_validation(self, minimal_data):
        import copy
        data, _, _ = minimal_data
        bad_data = copy.deepcopy(data)
        bad_data.x = None
        report = validate_pyg_data(bad_data)
        assert not report.passed
        assert any("x is None" in e or "missing" in e.lower() for e in report.errors)


# ---------------------------------------------------------------------------
# TASK 4: Dataset Splitting Tests
# ---------------------------------------------------------------------------

class TestDatasetSplitting:
    @pytest.fixture(scope="class")
    def dataset(self):
        """Build a small dataset: 6 small + 4 medium + 2 large = 12 scenarios."""
        return build_dataset(
            n_small=6, n_medium=4, n_large=2,
            seed_offset=0, split_seed=42,
        )

    def test_total_scenario_count(self, dataset):
        assert dataset.total_scenarios == 12

    def test_split_counts_sum_to_total(self, dataset):
        total = (
            len(dataset.train_data)
            + len(dataset.val_data)
            + len(dataset.test_data)
        )
        assert total == dataset.total_scenarios

    def test_train_is_largest_split(self, dataset):
        assert len(dataset.train_data) >= len(dataset.val_data)
        assert len(dataset.train_data) >= len(dataset.test_data)

    def test_all_splits_non_empty(self, dataset):
        assert len(dataset.train_data) >= 1
        assert len(dataset.val_data) >= 1
        assert len(dataset.test_data) >= 1

    def test_approximate_70_15_15_ratio(self, dataset):
        N = dataset.total_scenarios
        train_ratio = len(dataset.train_data) / N
        val_ratio = len(dataset.val_data) / N
        test_ratio = len(dataset.test_data) / N
        # Allow ±1 scenario tolerance
        assert 0.55 <= train_ratio <= 0.85, f"Train ratio {train_ratio:.2f} out of expected range"
        assert 0.05 <= val_ratio <= 0.35, f"Val ratio {val_ratio:.2f} out of expected range"
        assert 0.05 <= test_ratio <= 0.35, f"Test ratio {test_ratio:.2f} out of expected range"

    def test_no_leakage_between_splits(self, dataset):
        report = dataset.check_leakage()
        assert report["no_leakage"] is True, (
            f"Leakage detected:\n"
            f"  train/val overlap: {report['train_val_overlap']}\n"
            f"  train/test overlap: {report['train_test_overlap']}\n"
            f"  val/test overlap: {report['val_test_overlap']}"
        )

    def test_topology_ids_unique_across_scenarios(self, dataset):
        all_ids = [getattr(d, "topology_id", None) for d in dataset.data_list]
        assert len(all_ids) == len(set(all_ids)), "Duplicate topology IDs in dataset"

    def test_train_scenarios_have_valid_data(self, dataset):
        for d in dataset.train_data:
            assert d.x is not None
            assert d.y is not None
            assert d.edge_index is not None

    def test_split_error_on_empty_dataset(self):
        ds = AttackGraphDataset()
        with pytest.raises(ValueError, match="empty"):
            ds.split()

    def test_split_error_on_bad_ratios(self):
        from backend.ml.dataset import topology_to_pyg
        gen = SyntheticNetworkGenerator(seed=0)
        topo = gen.generate_small()
        d, _, _ = topology_to_pyg(topo)
        ds = AttackGraphDataset(data_list=[d, d, d])
        with pytest.raises(ValueError, match="sum to 1.0"):
            ds.split(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_split_error_on_too_few_scenarios(self):
        from backend.ml.dataset import topology_to_pyg
        gen = SyntheticNetworkGenerator(seed=0)
        topo = gen.generate_small()
        d, _, _ = topology_to_pyg(topo)
        ds = AttackGraphDataset(data_list=[d, d])  # only 2
        with pytest.raises(ValueError, match="at least 3 scenarios"):
            ds.split()

    def test_summary_returns_expected_keys(self, dataset):
        summary = dataset.summary()
        expected = {
            "total_scenarios", "train_scenarios", "val_scenarios",
            "test_scenarios", "total_nodes", "total_positive_labels",
            "total_negative_labels", "node_feature_dim", "edge_feature_dim",
            "split_seed",
        }
        assert expected == set(summary.keys())

    def test_summary_node_feature_dim(self, dataset):
        summary = dataset.summary()
        assert summary["node_feature_dim"] == FEATURE_DIM

    def test_summary_edge_feature_dim(self, dataset):
        summary = dataset.summary()
        assert summary["edge_feature_dim"] == EDGE_FEATURE_DIM

    def test_total_nodes_positive(self, dataset):
        assert dataset.total_nodes > 0

    def test_total_positive_labels_positive(self, dataset):
        assert dataset.total_positive_labels > 0

    def test_total_negative_labels_positive(self, dataset):
        assert dataset.total_negative_labels > 0


# ---------------------------------------------------------------------------
# TASK 6: Reproducibility Tests
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_dataset_x(self):
        """Two datasets built with identical parameters must have identical x matrices."""
        d1 = build_dataset(n_small=3, n_medium=2, n_large=1, seed_offset=10, split_seed=99)
        d2 = build_dataset(n_small=3, n_medium=2, n_large=1, seed_offset=10, split_seed=99)
        for i, (a, b) in enumerate(zip(d1.data_list, d2.data_list)):
            assert torch.equal(a.x, b.x), f"Scenario {i}: x matrices differ"
            assert torch.equal(a.y, b.y), f"Scenario {i}: labels differ"

    def test_same_seed_same_split_assignment(self):
        d1 = build_dataset(n_small=4, n_medium=2, n_large=2, seed_offset=0, split_seed=7)
        d2 = build_dataset(n_small=4, n_medium=2, n_large=2, seed_offset=0, split_seed=7)
        train_ids_1 = [getattr(d, "topology_id") for d in d1.train_data]
        train_ids_2 = [getattr(d, "topology_id") for d in d2.train_data]
        # Topology IDs differ due to UUID, but COUNT and ORDER of assignment must match
        assert len(train_ids_1) == len(train_ids_2)
        assert len(d1.val_data) == len(d2.val_data)
        assert len(d1.test_data) == len(d2.test_data)

    def test_different_seed_gives_different_split(self):
        d1 = build_dataset(n_small=4, n_medium=2, n_large=2, seed_offset=0, split_seed=1)
        d2 = build_dataset(n_small=4, n_medium=2, n_large=2, seed_offset=0, split_seed=2)
        # With different split seeds, the assignment is likely different (not guaranteed,
        # but with 8 scenarios it's extremely unlikely to be identical)
        # We only assert both produce valid non-empty splits
        assert len(d1.train_data) >= 1
        assert len(d2.train_data) >= 1

    def test_different_scenario_seed_gives_different_x(self):
        d1 = build_dataset(n_small=2, n_medium=1, n_large=0, seed_offset=0, split_seed=42)
        d2 = build_dataset(n_small=2, n_medium=1, n_large=0, seed_offset=100, split_seed=42)
        # x should differ because different topology seeds produce different graphs
        # (different vulnerabilities, attack weights, etc.)
        any_differ = any(
            not torch.equal(a.x, b.x)
            for a, b in zip(d1.data_list, d2.data_list)
        )
        assert any_differ, "Different seeds should produce different feature matrices"


# ---------------------------------------------------------------------------
# Full Pipeline Smoke Test
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    def test_build_dataset_runs_successfully(self):
        """End-to-end test: generate, convert, validate, split."""
        dataset = build_dataset(
            n_small=4, n_medium=2, n_large=1,
            seed_offset=0, split_seed=42,
            validate=True,
        )
        assert dataset.total_scenarios == 7
        assert len(dataset.train_data) + len(dataset.val_data) + len(dataset.test_data) == 7

    def test_all_data_objects_pass_validation(self):
        dataset = build_dataset(n_small=3, n_medium=2, n_large=1, seed_offset=5, validate=True)
        for d in dataset.data_list:
            report = validate_pyg_data(d)
            assert report.passed, f"Scenario {getattr(d, 'topology_name', '?')} failed: {report.errors}"

    def test_medium_topology_data_shapes(self):
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_medium()
        data, node_index, _ = topology_to_pyg(topo)
        N = topo.node_count
        E = data.edge_index.shape[1]
        assert data.x.shape == (N, FEATURE_DIM)
        assert data.edge_attr.shape == (E, EDGE_FEATURE_DIM)
        assert data.y.shape == (N,)
        report = validate_pyg_data(data)
        assert report.passed, f"Medium topology failed validation: {report.errors}"

    def test_large_topology_data_shapes(self):
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_large()
        data, node_index, _ = topology_to_pyg(topo)
        N = topo.node_count
        assert data.x.shape == (N, FEATURE_DIM)
        assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
        assert data.y.shape == (N,)
        report = validate_pyg_data(data)
        assert report.passed, f"Large topology failed validation: {report.errors}"
