"""
test_graph_builder.py
---------------------
Unit tests for backend/core/graph_builder.py

Tests cover:
- Graph node/edge count after build
- Bidirectional edge expansion
- Node attribute correctness
- Edge attribute correctness (including telemetry merge)
- Validation report correctness
- Empty topology raises ValueError
- Telemetry aggregation
- Historical attack signal on edges
"""

import pytest
import networkx as nx

from backend.core.data_models import (
    EdgeType,
    HistoricalAttack,
    NetworkEdge,
    NetworkNode,
    NetworkTopology,
    NodeType,
    OSType,
    TelemetryRecord,
    Vulnerability,
    VulnerabilitySeverity,
)
from backend.core.graph_builder import GraphBuilder, _aggregate_telemetry, _aggregate_historical_attacks
from backend.core.synthetic_data import SyntheticNetworkGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simple_topology() -> NetworkTopology:
    """
    n1 (entry) --[e1]--> n2 --[e2]--> n3 (critical)
    """
    topo = NetworkTopology(
        topology_id="topo-test",
        name="Test-Topology",
        is_synthetic=True,
    )
    n1 = NetworkNode(
        node_id="n1", label="entry-node",
        node_type=NodeType.INTERNET, os_type=OSType.UNKNOWN,
        ip_address="203.0.113.1", subnet="0.0.0.0/0",
        is_entry_point=True, asset_value=0.1,
    )
    n2 = NetworkNode(
        node_id="n2", label="middle-node",
        node_type=NodeType.SERVER, os_type=OSType.LINUX,
        ip_address="10.0.1.10", subnet="10.0.1.0/24",
        vulnerabilities=[
            Vulnerability(
                cve_id="SYN-TEST-01", cvss_score=8.5,
                severity=VulnerabilitySeverity.HIGH,
                description="High severity test vuln",
                exploitability_score=3.4, impact_score=5.9,
            )
        ],
        asset_value=0.5,
    )
    n3 = NetworkNode(
        node_id="n3", label="critical-db",
        node_type=NodeType.DATABASE, os_type=OSType.LINUX,
        ip_address="10.0.4.10", subnet="10.0.4.0/24",
        is_critical_asset=True, asset_value=0.95,
    )
    topo.add_node(n1)
    topo.add_node(n2)
    topo.add_node(n3)

    e1 = NetworkEdge(
        edge_id="e1", source_id="n1", target_id="n2",
        edge_type=EdgeType.NETWORK_LINK, bidirectional=True,
        attack_weight=0.7, firewall_rule=True,
    )
    e2 = NetworkEdge(
        edge_id="e2", source_id="n2", target_id="n3",
        edge_type=EdgeType.NETWORK_LINK, bidirectional=False,
        attack_weight=0.6,
    )
    topo.add_edge(e1)
    topo.add_edge(e2)

    # Add telemetry
    topo.telemetry = [
        TelemetryRecord(
            record_id="r1", source_id="n1", target_id="n2",
            timestamp=1700000000.0, bytes_sent=5000, packet_count=50,
            is_anomalous=True, alert_type="port_scan",
        ),
        TelemetryRecord(
            record_id="r2", source_id="n1", target_id="n2",
            timestamp=1700001000.0, bytes_sent=100, packet_count=5,
            is_anomalous=False,
        ),
    ]

    # Add historical attacks
    topo.historical_attacks = [
        HistoricalAttack(
            attack_id="a1",
            attack_path=["n1", "n2", "n3"],
            technique="T1190",
            success=True,
            timestamp=1700000000.0,
            source="synthetic",
        ),
        HistoricalAttack(
            attack_id="a2",
            attack_path=["n1", "n2", "n3"],
            technique="T1021",
            success=True,
            timestamp=1700005000.0,
            source="synthetic",
        ),
    ]

    return topo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder():
    return GraphBuilder()


@pytest.fixture
def simple_topo():
    return make_simple_topology()


@pytest.fixture
def simple_graph(builder, simple_topo):
    return builder.build(simple_topo)


@pytest.fixture(scope="module")
def small_graph():
    gen = SyntheticNetworkGenerator(seed=42)
    topo = gen.generate_small()
    builder = GraphBuilder()
    return builder.build(topo)


# ---------------------------------------------------------------------------
# Basic Graph Construction Tests
# ---------------------------------------------------------------------------

class TestGraphConstruction:
    def test_node_count_matches_topology(self, simple_graph, simple_topo):
        assert simple_graph.number_of_nodes() == simple_topo.node_count

    def test_directed_graph_type(self, simple_graph):
        assert isinstance(simple_graph, nx.DiGraph)

    def test_bidirectional_edge_creates_two_edges(self, simple_graph):
        # e1 is bidirectional: n1->n2 and n2->n1 should both exist
        assert simple_graph.has_edge("n1", "n2")
        assert simple_graph.has_edge("n2", "n1")

    def test_unidirectional_edge_creates_one_edge(self, simple_graph):
        # e2 is not bidirectional: n2->n3 should exist, n3->n2 should not
        assert simple_graph.has_edge("n2", "n3")
        assert not simple_graph.has_edge("n3", "n2")

    def test_total_edge_count(self, simple_graph):
        # e1 bidirectional = 2 edges, e2 unidirectional = 1 edge → total 3
        assert simple_graph.number_of_edges() == 3

    def test_graph_metadata_stored(self, simple_graph):
        assert simple_graph.graph["topology_name"] == "Test-Topology"
        assert simple_graph.graph["is_synthetic"] is True

    def test_empty_topology_raises(self, builder):
        empty_topo = NetworkTopology(topology_id="empty", name="Empty")
        with pytest.raises(ValueError, match="empty topology"):
            builder.build(empty_topo)


# ---------------------------------------------------------------------------
# Node Attribute Tests
# ---------------------------------------------------------------------------

class TestNodeAttributes:
    def test_entry_point_flag_set(self, simple_graph):
        assert simple_graph.nodes["n1"]["is_entry_point"] == 1

    def test_non_entry_point_flag_zero(self, simple_graph):
        assert simple_graph.nodes["n2"]["is_entry_point"] == 0

    def test_critical_asset_flag_set(self, simple_graph):
        assert simple_graph.nodes["n3"]["is_critical_asset"] == 1

    def test_non_critical_flag_zero(self, simple_graph):
        assert simple_graph.nodes["n1"]["is_critical_asset"] == 0

    def test_vulnerability_count_correct(self, simple_graph):
        # n2 has 1 vulnerability
        assert simple_graph.nodes["n2"]["vuln_count"] == 1

    def test_max_cvss_correct(self, simple_graph):
        assert simple_graph.nodes["n2"]["max_cvss"] == 8.5

    def test_node_type_stored(self, simple_graph):
        assert simple_graph.nodes["n1"]["node_type"] == "internet"
        assert simple_graph.nodes["n3"]["node_type"] == "database"

    def test_asset_value_stored(self, simple_graph):
        assert simple_graph.nodes["n3"]["asset_value"] == 0.95

    def test_all_nodes_have_label(self, simple_graph):
        for node_id in simple_graph.nodes:
            assert "label" in simple_graph.nodes[node_id]


# ---------------------------------------------------------------------------
# Edge Attribute Tests
# ---------------------------------------------------------------------------

class TestEdgeAttributes:
    def test_attack_weight_stored(self, simple_graph):
        edge = simple_graph.edges["n1", "n2"]
        assert "attack_weight" in edge
        assert 0.0 <= edge["attack_weight"] <= 1.0

    def test_firewall_rule_encoded(self, simple_graph):
        # e1 has firewall_rule=True
        edge = simple_graph.edges["n1", "n2"]
        assert edge["firewall_rule"] == 1

    def test_edge_type_stored(self, simple_graph):
        edge = simple_graph.edges["n1", "n2"]
        assert edge["edge_type"] == "network_link"

    def test_telemetry_anomaly_count_merged(self, simple_graph):
        # 2 telemetry records on n1->n2, 1 anomalous
        edge = simple_graph.edges["n1", "n2"]
        assert edge["anomaly_count"] == 1
        assert edge["telemetry_record_count"] == 2

    def test_telemetry_bytes_merged(self, simple_graph):
        edge = simple_graph.edges["n1", "n2"]
        assert edge["total_bytes"] == 5100  # 5000 + 100

    def test_historical_attack_count_merged(self, simple_graph):
        # 2 attacks traversed n1->n2
        edge = simple_graph.edges["n1", "n2"]
        assert edge["historical_attack_count"] == 2

    def test_historical_attack_count_second_edge(self, simple_graph):
        # 2 attacks also traversed n2->n3
        edge = simple_graph.edges["n2", "n3"]
        assert edge["historical_attack_count"] == 2

    def test_no_telemetry_edge_has_zero_anomaly(self, simple_graph):
        # n2->n3 has no telemetry
        edge = simple_graph.edges["n2", "n3"]
        assert edge["anomaly_count"] == 0


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_aggregate_telemetry(self, simple_topo):
        agg = _aggregate_telemetry(simple_topo)
        key = ("n1", "n2")
        assert key in agg
        assert agg[key]["anomaly_count"] == 1
        assert agg[key]["total_bytes"] == 5100
        assert agg[key]["record_count"] == 2

    def test_aggregate_historical_attacks(self, simple_topo):
        counts = _aggregate_historical_attacks(simple_topo)
        assert counts[("n1", "n2")] == 2
        assert counts[("n2", "n3")] == 2


# ---------------------------------------------------------------------------
# GraphBuilder Utility Method Tests
# ---------------------------------------------------------------------------

class TestGraphBuilderUtilities:
    def test_get_entry_nodes(self, builder, simple_graph):
        entries = builder.get_entry_nodes(simple_graph)
        assert "n1" in entries
        assert len(entries) == 1

    def test_get_critical_nodes(self, builder, simple_graph):
        crits = builder.get_critical_nodes(simple_graph)
        assert "n3" in crits
        assert len(crits) == 1

    def test_validate_graph_returns_dict(self, builder, simple_graph):
        results = builder.validate_graph(simple_graph)
        assert isinstance(results, dict)

    def test_validate_graph_has_entry_points(self, builder, simple_graph):
        results = builder.validate_graph(simple_graph)
        assert results["has_entry_points"] is True

    def test_validate_graph_has_critical_assets(self, builder, simple_graph):
        results = builder.validate_graph(simple_graph)
        assert results["has_critical_assets"] is True

    def test_validate_graph_no_self_loops(self, builder, simple_graph):
        results = builder.validate_graph(simple_graph)
        assert results["has_self_loops"] is False

    def test_validate_graph_reachability(self, builder, simple_graph):
        results = builder.validate_graph(simple_graph)
        assert results["any_path_exists"] is True

    def test_validate_graph_reachability_details(self, builder, simple_graph):
        results = builder.validate_graph(simple_graph)
        # There should be exactly 1 entry × 1 critical = 1 pair
        assert len(results["reachability"]) == 1
        assert results["reachability"][0]["reachable"] is True


# ---------------------------------------------------------------------------
# Integration with Synthetic Data
# ---------------------------------------------------------------------------

class TestGraphBuilderIntegration:
    def test_small_graph_built_successfully(self, small_graph):
        assert small_graph.number_of_nodes() >= 8

    def test_small_graph_is_directed(self, small_graph):
        assert isinstance(small_graph, nx.DiGraph)

    def test_small_graph_has_entry_points(self, small_graph):
        builder = GraphBuilder()
        assert len(builder.get_entry_nodes(small_graph)) >= 1

    def test_small_graph_has_critical_assets(self, small_graph):
        builder = GraphBuilder()
        assert len(builder.get_critical_nodes(small_graph)) >= 1

    def test_small_graph_all_nodes_have_attributes(self, small_graph):
        required_attrs = [
            "label", "node_type", "is_entry_point", "is_critical_asset",
            "vuln_count", "max_cvss", "asset_value",
        ]
        for node_id in small_graph.nodes:
            d = small_graph.nodes[node_id]
            for attr in required_attrs:
                assert attr in d, (
                    f"Node {node_id} missing attribute '{attr}'"
                )

    def test_small_graph_all_edges_have_attributes(self, small_graph):
        required_attrs = [
            "edge_type", "attack_weight", "firewall_rule",
            "anomaly_count", "historical_attack_count",
        ]
        for u, v, d in small_graph.edges(data=True):
            for attr in required_attrs:
                assert attr in d, (
                    f"Edge ({u},{v}) missing attribute '{attr}'"
                )
