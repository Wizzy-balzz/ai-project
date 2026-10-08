"""
test_phase6_xai.py
------------------
Unit tests for Phase 6: Explainable AI, Chokepoint Analysis & Mitigation Simulation.
"""

import networkx as nx
import pytest

from backend.ml.explainer import (
    CriticalChokepoint,
    MitigationAction,
    PathExplanation,
    PathExplainer,
)
from backend.ml.path_models import CandidateAttackPath
from backend.ml.risk_scorer import RiskScorer, ScoredAttackPath


def _build_test_graph_and_paths():
    # Build a simple directed graph:
    # gateway -> web -> db (critical)
    # gateway -> jump -> db (critical)
    G = nx.DiGraph()
    G.add_node("gw", label="Gateway", is_entry_point=1, is_critical_asset=0, max_cvss=0.0)
    G.add_node("web", label="Web Server", is_entry_point=0, is_critical_asset=0, max_cvss=9.8)
    G.add_node("jump", label="Jump Host", is_entry_point=0, is_critical_asset=0, max_cvss=5.0)
    G.add_node("db", label="Database", is_entry_point=0, is_critical_asset=1, max_cvss=4.0)

    G.add_edge("gw", "web", edge_type="network_link", firewall_rule=0, is_encrypted=0, attack_weight=0.8, anomaly_count=2)
    G.add_edge("web", "db", edge_type="exploit_path", firewall_rule=0, is_encrypted=0, attack_weight=0.9, anomaly_count=0)
    G.add_edge("gw", "jump", edge_type="network_link", firewall_rule=1, is_encrypted=1, attack_weight=0.3, anomaly_count=0)
    G.add_edge("jump", "db", edge_type="network_link", firewall_rule=0, is_encrypted=1, attack_weight=0.4, anomaly_count=0)

    # Candidate 1: gw -> web -> db
    c1 = CandidateAttackPath(
        path_id="path-web",
        node_ids=["gw", "web", "db"],
        edge_pairs=[("gw", "web"), ("web", "db")],
        node_labels=["Gateway", "Web Server", "Database"],
        node_attack_probabilities=[0.9, 0.9, 0.9],
        edge_features=[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
        ],
        path_length=2,
        entry_node="gw",
        terminal_node="db",
        reaches_critical_asset=True,
        structural_metadata={"mean_max_cvss": 4.6},
    )

    # Candidate 2: gw -> jump -> db
    c2 = CandidateAttackPath(
        path_id="path-jump",
        node_ids=["gw", "jump", "db"],
        edge_pairs=[("gw", "jump"), ("jump", "db")],
        node_labels=["Gateway", "Jump Host", "Database"],
        node_attack_probabilities=[0.4, 0.4, 0.4],
        edge_features=[
            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.3, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0],
        ],
        path_length=2,
        entry_node="gw",
        terminal_node="db",
        reaches_critical_asset=True,
        structural_metadata={"mean_max_cvss": 3.0},
    )

    scorer = RiskScorer()
    sp1 = scorer.score_path(c1)
    sp2 = scorer.score_path(c2)

    return G, [c1, c2], [sp1, sp2]


class TestPathExplainer:
    def setup_method(self):
        self.G, self.candidates, self.scored = _build_test_graph_and_paths()
        self.explainer = PathExplainer()

    def test_explain_path_narrative(self):
        exp = self.explainer.explain_path(self.scored[0], self.G)
        assert isinstance(exp, PathExplanation)
        assert exp.path_id == "path-web"
        assert "Gateway" in exp.path_summary
        assert "Database" in exp.path_summary
        assert len(exp.hop_explanations) == 2
        assert exp.weakest_node == "web"  # Has CVSS 9.8
        assert exp.weakest_edge == ("web", "db")

    def test_hop_explanations_weaknesses(self):
        exp = self.explainer.explain_path(self.scored[0], self.G)
        hop1 = exp.hop_explanations[0]
        assert hop1.step == 1
        assert hop1.source_id == "gw"
        assert hop1.target_id == "web"
        assert any("CVSS 9.8" in w for w in hop1.exploited_weaknesses)
        assert any("Missing firewall" in w for w in hop1.exploited_weaknesses)
        assert hop1.anomaly_count == 2

    def test_find_chokepoints(self):
        chokepoints = self.explainer.find_chokepoints(self.scored, self.G)
        assert len(chokepoints) > 0

        # "gw" and "db" appear in both paths
        gw_cps = [cp for cp in chokepoints if cp.element_id == "gw"]
        assert len(gw_cps) == 1
        assert gw_cps[0].intersected_paths_count == 2

        db_cps = [cp for cp in chokepoints if cp.element_id == "db"]
        assert len(db_cps) == 1
        assert db_cps[0].intersected_paths_count == 2

    def test_generate_mitigations(self):
        mitigations = self.explainer.generate_mitigations(self.scored, self.G)
        assert len(mitigations) > 0
        for m in mitigations:
            assert m.priority in ("IMMEDIATE", "HIGH", "MEDIUM")
            assert m.severed_paths_count >= 1
            assert m.estimated_risk_reduction_pct > 0.0

    def test_simulate_mitigation_sever_edge(self):
        # Create a mitigation targeting the web->db edge
        mit = MitigationAction(
            mitigation_id="test-mit-1",
            title="Deploy firewall web->db",
            action_type="deploy_firewall",
            target_element="web->db",
            target_label="Web Server → Database",
            description="Block web to db",
            priority="HIGH",
            severed_paths_count=1,
            severed_path_ids=["path-web"],
            estimated_risk_reduction_pct=50.0,
        )

        sim = self.explainer.simulate_mitigation(self.candidates, mit)
        assert sim["original_paths_count"] == 2
        assert sim["remaining_paths_count"] == 1
        assert sim["severed_paths_count"] == 1
        assert "path-web" in sim["severed_path_ids"]
        assert sim["new_network_risk"] < sim["original_network_risk"]
        assert sim["risk_reduction_pct"] > 0.0

    def test_simulate_mitigation_isolate_node(self):
        # Isolate "gw" which is the entry point for both paths
        mit = MitigationAction(
            mitigation_id="test-mit-2",
            title="Isolate gateway",
            action_type="isolate_node",
            target_element="gw",
            target_label="Gateway",
            description="Isolate gateway",
            priority="IMMEDIATE",
            severed_paths_count=2,
            severed_path_ids=["path-web", "path-jump"],
            estimated_risk_reduction_pct=100.0,
        )

        sim = self.explainer.simulate_mitigation(self.candidates, mit)
        assert sim["remaining_paths_count"] == 0
        assert sim["severed_paths_count"] == 2
        assert sim["new_network_risk"] == 0.0
        assert sim["risk_reduction_pct"] == 100.0

    def test_serialization(self):
        exp = self.explainer.explain_path(self.scored[0], self.G)
        d = exp.to_dict()
        assert "path_id" in d
        assert "path_summary" in d
        assert "factor_breakdown" in d
        assert "hop_explanations" in d
        assert len(d["hop_explanations"]) == 2
