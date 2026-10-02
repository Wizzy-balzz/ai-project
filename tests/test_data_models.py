"""
test_data_models.py
-------------------
Unit tests for backend/core/data_models.py

Tests cover:
- Vulnerability: CVSS validation, severity derivation, serialization
- NetworkNode: asset_value validation, computed properties, serialization
- NetworkEdge: self-loop rejection, attack_weight validation, serialization
- TelemetryRecord: basic serialization
- HistoricalAttack: minimum path length enforcement, serialization
- NetworkTopology: node/edge management, entry point filtering, serialization
"""

import pytest

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
    make_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vuln(cvss: float = 7.5) -> Vulnerability:
    return Vulnerability(
        cve_id="SYN-TEST-0001",
        cvss_score=cvss,
        severity=Vulnerability.severity_from_cvss(cvss),
        description="Test vulnerability",
        exploitability_score=3.0,
        impact_score=4.0,
    )


def make_node(
    node_id: str = "n1",
    is_entry: bool = False,
    is_critical: bool = False,
    asset_value: float = 0.5,
    vulns: list = None,
) -> NetworkNode:
    return NetworkNode(
        node_id=node_id,
        label=f"node-{node_id}",
        node_type=NodeType.SERVER,
        os_type=OSType.LINUX,
        ip_address="10.0.1.1",
        subnet="10.0.1.0/24",
        vulnerabilities=vulns or [],
        is_entry_point=is_entry,
        is_critical_asset=is_critical,
        asset_value=asset_value,
    )


def make_edge(src: str = "n1", tgt: str = "n2", attack_weight: float = 0.5) -> NetworkEdge:
    return NetworkEdge(
        edge_id="e1",
        source_id=src,
        target_id=tgt,
        edge_type=EdgeType.NETWORK_LINK,
        attack_weight=attack_weight,
    )


# ---------------------------------------------------------------------------
# Vulnerability Tests
# ---------------------------------------------------------------------------

class TestVulnerability:
    def test_valid_creation(self):
        v = make_vuln(7.5)
        assert v.cvss_score == 7.5
        assert v.severity == VulnerabilitySeverity.HIGH

    def test_cvss_out_of_range_raises(self):
        with pytest.raises(ValueError, match="CVSS score must be"):
            Vulnerability(
                cve_id="SYN-BAD", cvss_score=10.1,
                severity=VulnerabilitySeverity.CRITICAL,
                description="bad", exploitability_score=0.0, impact_score=0.0,
            )

    def test_cvss_negative_raises(self):
        with pytest.raises(ValueError, match="CVSS score must be"):
            Vulnerability(
                cve_id="SYN-BAD", cvss_score=-1.0,
                severity=VulnerabilitySeverity.LOW,
                description="bad", exploitability_score=0.0, impact_score=0.0,
            )

    def test_exploitability_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Exploitability score"):
            Vulnerability(
                cve_id="SYN-BAD", cvss_score=5.0,
                severity=VulnerabilitySeverity.MEDIUM,
                description="bad", exploitability_score=4.0, impact_score=0.0,
            )

    def test_impact_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Impact score"):
            Vulnerability(
                cve_id="SYN-BAD", cvss_score=5.0,
                severity=VulnerabilitySeverity.MEDIUM,
                description="bad", exploitability_score=0.0, impact_score=7.0,
            )

    @pytest.mark.parametrize("score, expected_severity", [
        (0.0, VulnerabilitySeverity.NONE),
        (0.1, VulnerabilitySeverity.LOW),
        (3.9, VulnerabilitySeverity.LOW),
        (4.0, VulnerabilitySeverity.MEDIUM),
        (6.9, VulnerabilitySeverity.MEDIUM),
        (7.0, VulnerabilitySeverity.HIGH),
        (8.9, VulnerabilitySeverity.HIGH),
        (9.0, VulnerabilitySeverity.CRITICAL),
        (10.0, VulnerabilitySeverity.CRITICAL),
    ])
    def test_severity_from_cvss(self, score, expected_severity):
        assert Vulnerability.severity_from_cvss(score) == expected_severity

    def test_to_dict_keys(self):
        v = make_vuln()
        d = v.to_dict()
        expected_keys = {
            "cve_id", "cvss_score", "severity", "description",
            "exploitability_score", "impact_score",
            "is_exploited_in_wild", "patch_available",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_severity_is_string(self):
        v = make_vuln(9.5)
        assert isinstance(v.to_dict()["severity"], str)
        assert v.to_dict()["severity"] == "critical"


# ---------------------------------------------------------------------------
# NetworkNode Tests
# ---------------------------------------------------------------------------

class TestNetworkNode:
    def test_valid_creation(self):
        n = make_node()
        assert n.node_id == "n1"
        assert n.max_cvss == 0.0

    def test_asset_value_out_of_range_raises(self):
        with pytest.raises(ValueError, match="asset_value"):
            make_node(asset_value=1.5)

    def test_asset_value_negative_raises(self):
        with pytest.raises(ValueError, match="asset_value"):
            make_node(asset_value=-0.1)

    def test_max_cvss_no_vulns(self):
        n = make_node()
        assert n.max_cvss == 0.0

    def test_max_cvss_with_vulns(self):
        vulns = [make_vuln(5.0), make_vuln(8.5), make_vuln(3.0)]
        n = make_node(vulns=vulns)
        assert n.max_cvss == 8.5

    def test_has_critical_vuln_false(self):
        vulns = [make_vuln(7.5)]  # HIGH, not CRITICAL
        n = make_node(vulns=vulns)
        assert not n.has_critical_vuln

    def test_has_critical_vuln_true(self):
        vuln = Vulnerability(
            cve_id="SYN-CRIT", cvss_score=9.8,
            severity=VulnerabilitySeverity.CRITICAL,
            description="critical", exploitability_score=3.9, impact_score=5.9,
        )
        n = make_node(vulns=[vuln])
        assert n.has_critical_vuln

    def test_has_exploited_in_wild_false(self):
        n = make_node(vulns=[make_vuln()])
        assert not n.has_exploited_in_wild

    def test_has_exploited_in_wild_true(self):
        vuln = Vulnerability(
            cve_id="SYN-EW", cvss_score=9.0,
            severity=VulnerabilitySeverity.CRITICAL,
            description="wild", exploitability_score=3.9, impact_score=5.9,
            is_exploited_in_wild=True,
        )
        n = make_node(vulns=[vuln])
        assert n.has_exploited_in_wild

    def test_to_dict_contains_computed_props(self):
        n = make_node(vulns=[make_vuln()])
        d = n.to_dict()
        assert "max_cvss" in d
        assert "has_critical_vuln" in d
        assert "has_exploited_in_wild" in d

    def test_to_dict_vulns_are_dicts(self):
        n = make_node(vulns=[make_vuln()])
        d = n.to_dict()
        assert isinstance(d["vulnerabilities"], list)
        assert isinstance(d["vulnerabilities"][0], dict)


# ---------------------------------------------------------------------------
# NetworkEdge Tests
# ---------------------------------------------------------------------------

class TestNetworkEdge:
    def test_valid_creation(self):
        e = make_edge()
        assert e.edge_id == "e1"

    def test_self_loop_raises(self):
        with pytest.raises(ValueError, match="Self-loops"):
            make_edge(src="n1", tgt="n1")

    def test_attack_weight_out_of_range_raises(self):
        with pytest.raises(ValueError, match="attack_weight"):
            make_edge(attack_weight=1.5)

    def test_attack_weight_negative_raises(self):
        with pytest.raises(ValueError, match="attack_weight"):
            make_edge(attack_weight=-0.1)

    def test_attack_weight_boundary(self):
        e_zero = make_edge(attack_weight=0.0)
        e_one = make_edge(attack_weight=1.0)
        assert e_zero.attack_weight == 0.0
        assert e_one.attack_weight == 1.0

    def test_to_dict_keys(self):
        e = make_edge()
        d = e.to_dict()
        expected_keys = {
            "edge_id", "source_id", "target_id", "edge_type",
            "bidirectional", "protocol", "port", "firewall_rule",
            "is_encrypted", "bandwidth_mbps", "latency_ms", "attack_weight",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_edge_type_is_string(self):
        e = make_edge()
        assert isinstance(e.to_dict()["edge_type"], str)


# ---------------------------------------------------------------------------
# TelemetryRecord Tests
# ---------------------------------------------------------------------------

class TestTelemetryRecord:
    def test_valid_creation(self):
        rec = TelemetryRecord(
            record_id="r1", source_id="n1", target_id="n2",
            timestamp=1700000000.0, bytes_sent=1024, packet_count=10,
        )
        assert rec.record_id == "r1"
        assert not rec.is_anomalous

    def test_to_dict(self):
        rec = TelemetryRecord(
            record_id="r1", source_id="n1", target_id="n2",
            timestamp=1700000000.0,
        )
        d = rec.to_dict()
        assert d["record_id"] == "r1"
        assert d["is_anomalous"] is False
        assert d["alert_type"] is None


# ---------------------------------------------------------------------------
# HistoricalAttack Tests
# ---------------------------------------------------------------------------

class TestHistoricalAttack:
    def test_valid_creation(self):
        atk = HistoricalAttack(
            attack_id="a1",
            attack_path=["n1", "n2", "n3"],
            technique="T1190",
            success=True,
            timestamp=1700000000.0,
            source="synthetic",
        )
        assert atk.attack_id == "a1"
        assert len(atk.attack_path) == 3

    def test_single_node_path_raises(self):
        with pytest.raises(ValueError, match="at least 2 nodes"):
            HistoricalAttack(
                attack_id="a1",
                attack_path=["n1"],
                technique="T1190",
                success=True,
                timestamp=1700000000.0,
            )

    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="at least 2 nodes"):
            HistoricalAttack(
                attack_id="a1",
                attack_path=[],
                technique="T1190",
                success=True,
                timestamp=1700000000.0,
            )

    def test_to_dict(self):
        atk = HistoricalAttack(
            attack_id="a1",
            attack_path=["n1", "n2"],
            technique="T1021",
            success=False,
            timestamp=1700000000.0,
            source="synthetic",
        )
        d = atk.to_dict()
        assert d["source"] == "synthetic"
        assert d["attack_path"] == ["n1", "n2"]


# ---------------------------------------------------------------------------
# NetworkTopology Tests
# ---------------------------------------------------------------------------

class TestNetworkTopology:
    def _make_topology(self) -> NetworkTopology:
        topo = NetworkTopology(
            topology_id="t1",
            name="Test-Topology",
            is_synthetic=True,
        )
        n1 = make_node("n1", is_entry=True)
        n2 = make_node("n2")
        n3 = make_node("n3", is_critical=True)
        topo.add_node(n1)
        topo.add_node(n2)
        topo.add_node(n3)
        topo.add_edge(make_edge("n1", "n2"))
        e2 = NetworkEdge(
            edge_id="e2", source_id="n2", target_id="n3",
            edge_type=EdgeType.NETWORK_LINK,
        )
        topo.add_edge(e2)
        return topo

    def test_node_count(self):
        topo = self._make_topology()
        assert topo.node_count == 3

    def test_edge_count(self):
        topo = self._make_topology()
        assert topo.edge_count == 2

    def test_entry_points(self):
        topo = self._make_topology()
        assert len(topo.entry_points) == 1
        assert topo.entry_points[0].node_id == "n1"

    def test_critical_assets(self):
        topo = self._make_topology()
        assert len(topo.critical_assets) == 1
        assert topo.critical_assets[0].node_id == "n3"

    def test_add_edge_unknown_source_raises(self):
        topo = self._make_topology()
        with pytest.raises(ValueError, match="Source node"):
            bad_edge = NetworkEdge(
                edge_id="e_bad", source_id="NONEXISTENT", target_id="n1",
                edge_type=EdgeType.NETWORK_LINK,
            )
            topo.add_edge(bad_edge)

    def test_add_edge_unknown_target_raises(self):
        topo = self._make_topology()
        with pytest.raises(ValueError, match="Target node"):
            bad_edge = NetworkEdge(
                edge_id="e_bad2", source_id="n1", target_id="NONEXISTENT",
                edge_type=EdgeType.NETWORK_LINK,
            )
            topo.add_edge(bad_edge)

    def test_is_synthetic_flag(self):
        topo = self._make_topology()
        assert topo.is_synthetic is True

    def test_to_dict_structure(self):
        topo = self._make_topology()
        d = topo.to_dict()
        assert d["topology_id"] == "t1"
        assert d["is_synthetic"] is True
        assert "nodes" in d
        assert "edges" in d
        assert "telemetry" in d
        assert "historical_attacks" in d
        assert d["node_count"] == 3
        assert d["edge_count"] == 2


# ---------------------------------------------------------------------------
# make_id Tests
# ---------------------------------------------------------------------------

class TestMakeId:
    def test_returns_string(self):
        assert isinstance(make_id(), str)

    def test_prefix_applied(self):
        uid = make_id("node-")
        assert uid.startswith("node-")

    def test_unique_ids(self):
        ids = {make_id() for _ in range(100)}
        assert len(ids) == 100
