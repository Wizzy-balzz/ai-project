"""
test_feature_extractor.py
-------------------------
Unit tests for backend/core/feature_extractor.py

Tests cover:
- Feature matrix shape correctness (N × FEATURE_DIM)
- Node index alignment (row i → node_index[i])
- Feature names length matches FEATURE_DIM
- Normalization: all values in [0, 1] after MinMax scaling
- One-hot encodings: sum = 1 per node per category group
- Structural features: non-negative values
- Entry point and critical asset flags are correctly encoded
- graph_statistics returns expected keys
- feature_summary returns expected structure
- Calling get_feature_summary before extract raises RuntimeError
"""

import pytest
import numpy as np
import networkx as nx

from backend.core.data_models import (
    EdgeType,
    NetworkEdge,
    NetworkNode,
    NetworkTopology,
    NodeType,
    OSType,
    Vulnerability,
    VulnerabilitySeverity,
)
from backend.core.feature_extractor import (
    ALL_FEATURE_NAMES,
    FEATURE_DIM,
    FeatureExtractor,
    NODE_TYPE_CATEGORIES,
    OS_TYPE_CATEGORIES,
    RAW_SCALAR_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
)
from backend.core.graph_builder import GraphBuilder
from backend.core.synthetic_data import SyntheticNetworkGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_three_node_graph() -> nx.DiGraph:
    """n1 (entry) → n2 → n3 (critical)"""
    topo = NetworkTopology(topology_id="t1", name="Test", is_synthetic=True)

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
                is_exploited_in_wild=True,
            )
        ],
        asset_value=0.5,
        services=["http:80", "https:443", "ssh:22"],
    )
    n3 = NetworkNode(
        node_id="n3", label="domain-controller",
        node_type=NodeType.CRITICAL_ASSET, os_type=OSType.WINDOWS,
        ip_address="10.0.5.10", subnet="10.0.5.0/24",
        is_critical_asset=True, asset_value=1.0,
        services=["ldap:389", "rdp:3389", "smb:445"],
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

    builder = GraphBuilder()
    return builder.build(topo)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_node_graph():
    return make_three_node_graph()


@pytest.fixture
def extractor():
    return FeatureExtractor(normalize=True)


@pytest.fixture
def extracted(extractor, three_node_graph):
    features, node_index = extractor.extract(three_node_graph)
    return features, node_index, extractor


@pytest.fixture(scope="module")
def small_extracted():
    gen = SyntheticNetworkGenerator(seed=42)
    topo = gen.generate_small()
    builder = GraphBuilder()
    G = builder.build(topo)
    extractor = FeatureExtractor(normalize=True)
    features, node_index = extractor.extract(G)
    return features, node_index, extractor, G


# ---------------------------------------------------------------------------
# Feature Dimension Tests
# ---------------------------------------------------------------------------

class TestFeatureDimensions:
    def test_feature_dim_constant_equals_feature_names_length(self):
        assert FEATURE_DIM == len(ALL_FEATURE_NAMES), (
            f"FEATURE_DIM={FEATURE_DIM} but len(ALL_FEATURE_NAMES)={len(ALL_FEATURE_NAMES)}"
        )

    def test_feature_names_no_duplicates(self):
        assert len(ALL_FEATURE_NAMES) == len(set(ALL_FEATURE_NAMES)), (
            "Duplicate feature names detected"
        )

    def test_feature_matrix_shape_three_nodes(self, extracted):
        features, node_index, _ = extracted
        assert features.shape == (3, FEATURE_DIM), (
            f"Expected shape (3, {FEATURE_DIM}), got {features.shape}"
        )

    def test_feature_matrix_shape_small_graph(self, small_extracted):
        features, node_index, _, _ = small_extracted
        n = len(node_index)
        assert features.shape == (n, FEATURE_DIM)

    def test_node_index_length_matches_rows(self, extracted):
        features, node_index, _ = extracted
        assert len(node_index) == features.shape[0]

    def test_dtype_is_float32(self, extracted):
        features, _, _ = extracted
        assert features.dtype == np.float32


# ---------------------------------------------------------------------------
# Normalization Tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalized_values_in_01_range(self, extracted):
        features, _, _ = extracted
        assert np.all(features >= 0.0), "Some features are below 0 after normalization"
        assert np.all(features <= 1.0 + 1e-6), "Some features exceed 1 after normalization"

    def test_small_graph_normalized_values_in_01_range(self, small_extracted):
        features, _, _, _ = small_extracted
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0 + 1e-6)

    def test_unnormalized_extractor_raw_values(self, three_node_graph):
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        # CVSS score of 9.0 should appear un-normalized
        n2_idx = node_index.index("n2")
        # max_cvss is RAW_SCALAR index 4
        max_cvss_idx = RAW_SCALAR_FEATURE_NAMES.index("max_cvss")
        assert features[n2_idx, max_cvss_idx] == pytest.approx(9.0, abs=0.01)


# ---------------------------------------------------------------------------
# Feature Value Correctness Tests
# ---------------------------------------------------------------------------

class TestFeatureValues:
    def test_entry_point_feature_positive(self, three_node_graph):
        """n1 is entry point; its raw (unscaled) is_entry_point feature should be > 0."""
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        n1_idx = node_index.index("n1")
        ep_idx = RAW_SCALAR_FEATURE_NAMES.index("is_entry_point")
        assert features[n1_idx, ep_idx] == 1.0

    def test_non_entry_point_feature_zero(self, three_node_graph):
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        n2_idx = node_index.index("n2")
        ep_idx = RAW_SCALAR_FEATURE_NAMES.index("is_entry_point")
        assert features[n2_idx, ep_idx] == 0.0

    def test_critical_asset_feature_positive(self, three_node_graph):
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        n3_idx = node_index.index("n3")
        ca_idx = RAW_SCALAR_FEATURE_NAMES.index("is_critical_asset")
        assert features[n3_idx, ca_idx] == 1.0

    def test_vuln_count_correct_raw(self, three_node_graph):
        """n2 has 1 vulnerability."""
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        n2_idx = node_index.index("n2")
        vc_idx = RAW_SCALAR_FEATURE_NAMES.index("vuln_count")
        assert features[n2_idx, vc_idx] == 1.0

    def test_max_cvss_correct_raw(self, three_node_graph):
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        n2_idx = node_index.index("n2")
        cvss_idx = RAW_SCALAR_FEATURE_NAMES.index("max_cvss")
        assert features[n2_idx, cvss_idx] == pytest.approx(9.0, abs=0.01)

    def test_has_ssh_flag(self, three_node_graph):
        """n2 has ssh:22 in services."""
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        n2_idx = node_index.index("n2")
        ssh_idx = RAW_SCALAR_FEATURE_NAMES.index("has_ssh")
        assert features[n2_idx, ssh_idx] == 1.0

    def test_has_rdp_flag(self, three_node_graph):
        """n3 has rdp:3389 in services."""
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        n3_idx = node_index.index("n3")
        rdp_idx = RAW_SCALAR_FEATURE_NAMES.index("has_rdp")
        assert features[n3_idx, rdp_idx] == 1.0


# ---------------------------------------------------------------------------
# One-Hot Encoding Tests
# ---------------------------------------------------------------------------

class TestOneHotEncoding:
    def _get_onehot_offset(self, feature_group: str) -> int:
        """Return the column offset for a feature group in ALL_FEATURE_NAMES."""
        if feature_group == "nodetype":
            return len(RAW_SCALAR_FEATURE_NAMES)
        elif feature_group == "ostype":
            return len(RAW_SCALAR_FEATURE_NAMES) + len(NODE_TYPE_CATEGORIES)
        raise ValueError(f"Unknown group: {feature_group}")

    def test_node_type_onehot_sums_to_one(self, three_node_graph):
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        offset = self._get_onehot_offset("nodetype")
        dim = len(NODE_TYPE_CATEGORIES)
        for i in range(features.shape[0]):
            row_sum = features[i, offset:offset + dim].sum()
            assert row_sum == pytest.approx(1.0, abs=1e-5), (
                f"Node type one-hot sum != 1.0 for node at row {i}: {row_sum}"
            )

    def test_os_type_onehot_sums_to_one(self, three_node_graph):
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        offset = self._get_onehot_offset("ostype")
        dim = len(OS_TYPE_CATEGORIES)
        for i in range(features.shape[0]):
            row_sum = features[i, offset:offset + dim].sum()
            assert row_sum == pytest.approx(1.0, abs=1e-5), (
                f"OS type one-hot sum != 1.0 for node at row {i}: {row_sum}"
            )

    def test_internet_node_type_encoded_correctly(self, three_node_graph):
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        n1_idx = node_index.index("n1")
        offset = self._get_onehot_offset("nodetype")
        internet_col = offset + NODE_TYPE_CATEGORIES.index("internet")
        assert features[n1_idx, internet_col] == 1.0

    def test_critical_asset_node_type_encoded_correctly(self, three_node_graph):
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        n3_idx = node_index.index("n3")
        offset = self._get_onehot_offset("nodetype")
        ca_col = offset + NODE_TYPE_CATEGORIES.index("critical_asset")
        assert features[n3_idx, ca_col] == 1.0


# ---------------------------------------------------------------------------
# Structural Feature Tests
# ---------------------------------------------------------------------------

class TestStructuralFeatures:
    def _get_structural_offset(self) -> int:
        return (
            len(RAW_SCALAR_FEATURE_NAMES)
            + len(NODE_TYPE_CATEGORIES)
            + len(OS_TYPE_CATEGORIES)
        )

    def test_structural_features_non_negative(self, extracted):
        features, node_index, _ = extracted
        offset = self._get_structural_offset()
        structural = features[:, offset:]
        # After normalization all values should be >= 0
        assert np.all(structural >= 0.0)

    def test_in_degree_positive_for_non_entry(self, three_node_graph):
        """n2 and n3 should have incoming edges → positive in_degree."""
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        offset = self._get_structural_offset()
        in_deg_idx = offset + STRUCTURAL_FEATURE_NAMES.index("in_degree")
        n2_idx = node_index.index("n2")
        n3_idx = node_index.index("n3")
        assert features[n2_idx, in_deg_idx] > 0
        assert features[n3_idx, in_deg_idx] > 0

    def test_out_degree_zero_for_sink(self, three_node_graph):
        """n3 is a sink (no outgoing non-reverse edges — but bidirectionality on e1 adds n2→n1)."""
        # n3 has no outgoing edges in our simple graph
        extractor = FeatureExtractor(normalize=False)
        features, node_index = extractor.extract(three_node_graph)
        offset = self._get_structural_offset()
        out_deg_idx = offset + STRUCTURAL_FEATURE_NAMES.index("out_degree")
        n3_idx = node_index.index("n3")
        assert features[n3_idx, out_deg_idx] == 0.0


# ---------------------------------------------------------------------------
# Summary & Statistics Tests
# ---------------------------------------------------------------------------

class TestSummaryAndStatistics:
    def test_get_feature_summary_returns_dict(self, extracted, three_node_graph):
        features, node_index, extractor = extracted
        summary = extractor.get_feature_summary(three_node_graph)
        assert isinstance(summary, dict)
        assert len(summary) == 3

    def test_summary_has_all_feature_names(self, extracted, three_node_graph):
        features, node_index, extractor = extracted
        summary = extractor.get_feature_summary(three_node_graph)
        for node_id, info in summary.items():
            assert set(info["features"].keys()) == set(ALL_FEATURE_NAMES)

    def test_get_feature_summary_before_extract_raises(self, three_node_graph):
        extractor = FeatureExtractor()
        with pytest.raises(RuntimeError, match="Call extract()"):
            extractor.get_feature_summary(three_node_graph)

    def test_get_graph_statistics_keys(self, extracted, three_node_graph):
        _, _, extractor = extracted
        stats = extractor.get_graph_statistics(three_node_graph)
        expected_keys = {
            "node_count", "edge_count", "entry_point_count",
            "critical_asset_count", "graph_density",
            "avg_shortest_path_length", "total_vulnerability_count",
            "avg_vuln_per_node", "max_cvss_in_graph",
            "avg_cvss_in_graph", "avg_edge_attack_weight",
            "total_anomaly_events",
        }
        assert expected_keys == set(stats.keys())

    def test_statistics_node_count_correct(self, extracted, three_node_graph):
        _, _, extractor = extracted
        stats = extractor.get_graph_statistics(three_node_graph)
        assert stats["node_count"] == 3

    def test_statistics_max_cvss_correct(self, extracted, three_node_graph):
        _, _, extractor = extracted
        stats = extractor.get_graph_statistics(three_node_graph)
        assert stats["max_cvss_in_graph"] == pytest.approx(9.0, abs=0.01)

    def test_empty_graph_raises(self):
        extractor = FeatureExtractor()
        G = nx.DiGraph()
        with pytest.raises(ValueError, match="empty graph"):
            extractor.extract(G)


# ---------------------------------------------------------------------------
# Integration Test: Small Synthetic Graph
# ---------------------------------------------------------------------------

class TestFeatureExtractorIntegration:
    def test_small_graph_feature_matrix_shape(self, small_extracted):
        features, node_index, _, G = small_extracted
        assert features.shape[0] == G.number_of_nodes()
        assert features.shape[1] == FEATURE_DIM

    def test_small_graph_feature_extractor_has_scaler(self, small_extracted):
        _, _, extractor, _ = small_extracted
        assert extractor.scaler is not None

    def test_small_graph_raw_features_stored(self, small_extracted):
        _, _, extractor, _ = small_extracted
        assert extractor.raw_features is not None
        assert extractor.raw_features.shape[1] == FEATURE_DIM
