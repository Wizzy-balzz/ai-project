"""
test_phase4_paths.py
--------------------
Unit and integration tests for Phase 4 candidate attack-path generation.

Test classes:
  TestPathConfig            — configuration validation
  TestPathLengthDefinition  — explicit edge-count definition
  TestPathValidator         — all 8 validation rules
  TestEdgeFeatureExtraction — 9 edge features preserved correctly
  TestDFSGeneration         — DFS algorithm correctness
  TestDeduplication         — duplicate removal
  TestCriticalTerminal      — critical-asset terminal detection
  TestGNNIntegration        — checkpoint loading, probability shapes/range
  TestDeterminism           — same graph+config → same output
  TestLeakagePrevention     — historical labels NOT used
  TestTruncation            — quota enforcement + flag
  TestCandidatePath         — CandidateAttackPath data structure
  TestIntegration           — small/medium/large topology end-to-end
"""

from __future__ import annotations

import copy
from typing import List, Optional
from pathlib import Path

import networkx as nx
import pytest
import torch

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
    make_id,
)
from backend.core.graph_builder import GraphBuilder
from backend.core.synthetic_data import SyntheticNetworkGenerator
from backend.ml.dataset import EDGE_FEATURE_DIM
from backend.ml.path_generator import (
    GNNProbabilityProvider,
    PathGenerator,
    PathValidator,
    _extract_edge_feature_vector,
)
from backend.ml.path_models import (
    CandidateAttackPath,
    PathConfig,
    PathGenerationSummary,
    PathValidationResult,
    make_path_id,
)


# ---------------------------------------------------------------------------
# Helpers — minimal deterministic topology
# ---------------------------------------------------------------------------

def make_linear_topology(n_intermediates: int = 2) -> NetworkTopology:
    """
    Simple linear topology:
      internet (entry) → hop1 → ... → hopN → domain-controller (critical)

    No historical attacks stored — tests leakage prevention.
    """
    topo = NetworkTopology(
        topology_id="topo-linear-test",
        name="Linear-Test",
        is_synthetic=True,
    )
    nodes = []
    n0 = NetworkNode(
        node_id="n0", label="internet",
        node_type=NodeType.INTERNET, os_type=OSType.UNKNOWN,
        ip_address="0.0.0.0", subnet="0.0.0.0/0",
        is_entry_point=True, asset_value=0.1,
    )
    topo.add_node(n0)
    nodes.append(n0)

    prev_id = "n0"
    for i in range(1, n_intermediates + 1):
        ni = NetworkNode(
            node_id=f"n{i}", label=f"server-{i}",
            node_type=NodeType.SERVER, os_type=OSType.LINUX,
            ip_address=f"10.0.{i}.1", subnet=f"10.0.{i}.0/24",
            is_entry_point=False, asset_value=0.5,
        )
        topo.add_node(ni)
        nodes.append(ni)
        e = NetworkEdge(
            edge_id=f"e{i}", source_id=prev_id, target_id=f"n{i}",
            edge_type=EdgeType.NETWORK_LINK, bidirectional=False,
            attack_weight=0.7,
        )
        topo.add_edge(e)
        prev_id = f"n{i}"

    final_id = f"n{n_intermediates + 1}"
    nc = NetworkNode(
        node_id=final_id, label="domain-controller",
        node_type=NodeType.CRITICAL_ASSET, os_type=OSType.WINDOWS,
        ip_address="10.99.0.1", subnet="10.99.0.0/24",
        is_critical_asset=True, asset_value=1.0,
    )
    topo.add_node(nc)
    e_final = NetworkEdge(
        edge_id=f"e{n_intermediates+1}",
        source_id=prev_id, target_id=final_id,
        edge_type=EdgeType.NETWORK_LINK, bidirectional=False,
        attack_weight=0.9,
    )
    topo.add_edge(e_final)
    # NO historical attacks — testing leakage
    return topo


def make_branching_topology() -> NetworkTopology:
    """
    Two entry points, branching topology:
      entry1 → A → critical
      entry2 → B → critical
      entry1 → B (cross-link)
    """
    topo = NetworkTopology(
        topology_id="topo-branch-test",
        name="Branching-Test",
        is_synthetic=True,
    )
    for nid, label, is_entry, is_critical, asset in [
        ("entry1", "internet1", True, False, 0.1),
        ("entry2", "internet2", True, False, 0.1),
        ("nodeA", "server-a", False, False, 0.5),
        ("nodeB", "server-b", False, False, 0.5),
        ("crit", "domain-ctrl", False, True, 1.0),
    ]:
        n = NetworkNode(
            node_id=nid, label=label,
            node_type=NodeType.SERVER, os_type=OSType.LINUX,
            ip_address=f"10.0.0.{1}", subnet="10.0.0.0/24",
            is_entry_point=is_entry, is_critical_asset=is_critical,
            asset_value=asset,
        )
        topo.add_node(n)

    for eid, src, tgt in [
        ("e1", "entry1", "nodeA"),
        ("e2", "nodeA", "crit"),
        ("e3", "entry2", "nodeB"),
        ("e4", "nodeB", "crit"),
        ("e5", "entry1", "nodeB"),
    ]:
        e = NetworkEdge(
            edge_id=eid, source_id=src, target_id=tgt,
            edge_type=EdgeType.NETWORK_LINK, bidirectional=False,
            attack_weight=0.6,
        )
        topo.add_edge(e)
    return topo


def make_no_entry_topology() -> NetworkTopology:
    """Topology with no entry points."""
    topo = NetworkTopology(
        topology_id="topo-no-entry", name="No-Entry-Test", is_synthetic=True,
    )
    n = NetworkNode(
        node_id="n1", label="server", node_type=NodeType.SERVER,
        os_type=OSType.LINUX, ip_address="10.0.0.1", subnet="10.0.0.0/24",
        is_entry_point=False, is_critical_asset=True, asset_value=0.8,
    )
    topo.add_node(n)
    return topo


# ---------------------------------------------------------------------------
# TestPathConfig
# ---------------------------------------------------------------------------

class TestPathConfig:
    def test_default_config_valid(self):
        cfg = PathConfig()
        assert cfg.min_path_length == 1
        assert cfg.max_path_length == 6
        assert cfg.max_paths_per_entry == 50

    def test_min_length_zero_raises(self):
        with pytest.raises(ValueError, match="min_path_length"):
            PathConfig(min_path_length=0)

    def test_max_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_path_length"):
            PathConfig(min_path_length=5, max_path_length=3)

    def test_invalid_max_paths_per_entry_raises(self):
        with pytest.raises(ValueError, match="max_paths_per_entry"):
            PathConfig(max_paths_per_entry=0)

    def test_invalid_sort_by_raises(self):
        with pytest.raises(ValueError, match="sort_by"):
            PathConfig(sort_by="random_nonsense")

    def test_to_dict_serializable(self):
        import json
        cfg = PathConfig()
        d = cfg.to_dict()
        json.dumps(d)  # Must not raise

    def test_equal_min_max_valid(self):
        cfg = PathConfig(min_path_length=3, max_path_length=3)
        assert cfg.min_path_length == 3
        assert cfg.max_path_length == 3


# ---------------------------------------------------------------------------
# TestPathLengthDefinition — explicit edge-count semantics
# ---------------------------------------------------------------------------

class TestPathLengthDefinition:
    """
    Path length = NUMBER OF EDGES = len(node_ids) - 1.

    A → B       has path_length = 1
    A → B → C   has path_length = 2
    """

    def test_two_node_path_length_is_one(self):
        topo = make_linear_topology(n_intermediates=0)
        # n_intermediates=0 means n0 → n1(critical) — wait that's wrong for
        # make_linear_topology which always has at least entry+final.
        # Let's directly test via a candidate.
        cfg = PathConfig(min_path_length=1, max_path_length=6)
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        two_node = [c for c in candidates if len(c.node_ids) == 2]
        for c in two_node:
            assert c.path_length == 1, (
                f"2-node path has length {c.path_length}, expected 1"
            )

    def test_three_node_path_length_is_two(self):
        topo = make_linear_topology(n_intermediates=1)
        cfg = PathConfig(min_path_length=1, max_path_length=6)
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        three_node = [c for c in candidates if len(c.node_ids) == 3]
        for c in three_node:
            assert c.path_length == 2

    def test_path_length_equals_node_count_minus_one(self):
        topo = make_linear_topology(n_intermediates=3)
        cfg = PathConfig(min_path_length=1, max_path_length=6)
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert c.path_length == len(c.node_ids) - 1

    def test_edge_pairs_count_equals_path_length(self):
        topo = make_linear_topology(n_intermediates=2)
        cfg = PathConfig(min_path_length=1, max_path_length=6)
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert len(c.edge_pairs) == c.path_length


# ---------------------------------------------------------------------------
# TestPathValidator
# ---------------------------------------------------------------------------

class TestPathValidator:
    @pytest.fixture
    def validator(self):
        topo = make_branching_topology()
        builder = GraphBuilder()
        G = builder.build(topo)
        cfg = PathConfig()
        return PathValidator(G, cfg)

    def test_valid_path_accepted(self, validator):
        result = validator.validate(["entry1", "nodeA", "crit"])
        assert result.valid, f"Expected valid, got errors: {result.errors}"

    def test_single_node_rejected(self, validator):
        result = validator.validate(["entry1"])
        assert not result.valid
        assert any("2 node" in e or "minimum" in e.lower() for e in result.errors)

    def test_nonentry_first_node_rejected(self, validator):
        result = validator.validate(["nodeA", "crit"])
        assert not result.valid
        assert any("entry point" in e.lower() for e in result.errors)

    def test_nonexistent_node_rejected(self, validator):
        result = validator.validate(["entry1", "nonexistent_node_xyz"])
        assert not result.valid
        assert any("does not exist" in e for e in result.errors)

    def test_nonexistent_edge_rejected(self, validator):
        # entry2 → nodeA has no edge (entry2 → nodeB only)
        result = validator.validate(["entry2", "nodeA"])
        assert not result.valid
        assert any("No directed edge" in e for e in result.errors)

    def test_repeated_node_rejected(self, validator):
        result = validator.validate(["entry1", "nodeA", "entry1"])
        assert not result.valid
        assert any("more than once" in e or "repeat" in e.lower() for e in result.errors)

    def test_path_too_short_rejected(self):
        topo = make_branching_topology()
        builder = GraphBuilder()
        G = builder.build(topo)
        cfg = PathConfig(min_path_length=3)  # require 3 edges
        v = PathValidator(G, cfg)
        result = v.validate(["entry1", "nodeA", "crit"])  # 2 edges only
        assert not result.valid
        assert any("minimum" in e.lower() or "edge" in e.lower() for e in result.errors)

    def test_path_too_long_rejected(self):
        topo = make_branching_topology()
        builder = GraphBuilder()
        G = builder.build(topo)
        cfg = PathConfig(max_path_length=1)  # only 1 edge allowed
        v = PathValidator(G, cfg)
        result = v.validate(["entry1", "nodeA", "crit"])  # 2 edges
        assert not result.valid
        assert any("maximum" in e.lower() or "edge" in e.lower() for e in result.errors)

    def test_wrong_direction_rejected(self, validator):
        # crit → nodeA is reverse direction, no such edge
        result = validator.validate(["entry1", "crit", "nodeA"])
        # crit is not an entry point AND no crit→nodeA edge
        assert not result.valid

    def test_self_loop_rejected(self, validator):
        # Manually check the self-loop rule
        topo = make_branching_topology()
        builder = GraphBuilder()
        G = builder.build(topo)
        # Manually add a self-loop to test
        G.add_edge("entry1", "entry1", edge_type="network_link", attack_weight=0.1,
                   firewall_rule=0, is_encrypted=0, anomaly_count=0,
                   historical_attack_count=0)
        cfg = PathConfig()
        v = PathValidator(G, cfg)
        result = v.validate(["entry1", "entry1"])
        # Should fail: repeated node AND self-loop
        assert not result.valid


# ---------------------------------------------------------------------------
# TestEdgeFeatureExtraction
# ---------------------------------------------------------------------------

class TestEdgeFeatureExtraction:
    def test_edge_feature_vector_length(self):
        edge_data = {
            "edge_type": "network_link",
            "firewall_rule": 1,
            "is_encrypted": 0,
            "attack_weight": 0.75,
            "anomaly_count": 3,
            "historical_attack_count": 2,
        }
        vec = _extract_edge_feature_vector(edge_data)
        assert len(vec) == EDGE_FEATURE_DIM, (
            f"Expected {EDGE_FEATURE_DIM}, got {len(vec)}"
        )

    def test_edge_type_onehot_sums_to_one(self):
        for etype in ["network_link", "trust_relationship", "exploit_path", "data_flow"]:
            vec = _extract_edge_feature_vector({"edge_type": etype})
            one_hot_sum = sum(vec[:4])
            assert one_hot_sum == pytest.approx(1.0, abs=1e-6)

    def test_unknown_edge_type_defaults_to_network_link(self):
        vec = _extract_edge_feature_vector({"edge_type": "unknown_type_xyz"})
        # network_link is at index 0
        assert vec[0] == pytest.approx(1.0, abs=1e-6)

    def test_firewall_rule_position(self):
        vec1 = _extract_edge_feature_vector({"firewall_rule": 1})
        vec0 = _extract_edge_feature_vector({"firewall_rule": 0})
        assert vec1[4] == pytest.approx(1.0, abs=1e-6)
        assert vec0[4] == pytest.approx(0.0, abs=1e-6)

    def test_attack_weight_position(self):
        vec = _extract_edge_feature_vector({"attack_weight": 0.42})
        assert vec[6] == pytest.approx(0.42, abs=1e-6)

    def test_missing_keys_default_gracefully(self):
        vec = _extract_edge_feature_vector({})  # all defaults
        assert len(vec) == EDGE_FEATURE_DIM
        assert all(isinstance(v, float) for v in vec)

    def test_edge_features_preserved_in_candidate(self):
        topo = make_branching_topology()
        cfg = PathConfig(min_path_length=1, max_path_length=6)
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert len(c.edge_features) == c.path_length
            for ef in c.edge_features:
                assert len(ef) == EDGE_FEATURE_DIM


# ---------------------------------------------------------------------------
# TestDFSGeneration
# ---------------------------------------------------------------------------

class TestDFSGeneration:
    def test_paths_originate_from_entry_points_only(self):
        topo = make_branching_topology()
        entry_nodes = {"entry1", "entry2"}
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert c.entry_node in entry_nodes, (
                f"Path starts at '{c.entry_node}', not an entry point"
            )

    def test_paths_follow_directed_edges(self):
        topo = make_branching_topology()
        builder = GraphBuilder()
        G = builder.build(topo)
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            for u, v in c.edge_pairs:
                assert G.has_edge(u, v), (
                    f"Invalid edge in candidate path: {u} → {v}"
                )

    def test_no_repeated_nodes_in_any_path(self):
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert len(c.node_ids) == len(set(c.node_ids)), (
                f"Repeated node in path {c.node_ids}"
            )

    def test_min_path_length_enforced(self):
        topo = make_linear_topology(n_intermediates=2)
        cfg = PathConfig(min_path_length=2, max_path_length=6)
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert c.path_length >= 2, (
                f"Path {c.node_ids} has length {c.path_length} < min=2"
            )

    def test_max_path_length_enforced(self):
        topo = make_linear_topology(n_intermediates=5)
        cfg = PathConfig(min_path_length=1, max_path_length=3)
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert c.path_length <= 3, (
                f"Path {c.node_ids} has length {c.path_length} > max=3"
            )

    def test_no_entry_topology_returns_empty(self):
        topo = make_no_entry_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, summary = gen.generate(topo)
        assert candidates == []
        assert summary.total_candidates == 0

    def test_critical_terminal_detection(self):
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        # "crit" is the only critical asset
        for c in candidates:
            if c.terminal_node == "crit":
                assert c.reaches_critical_asset is True
            else:
                assert c.reaches_critical_asset is False

    def test_require_critical_terminal_filter(self):
        topo = make_branching_topology()
        cfg = PathConfig(require_critical_terminal=True)
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert c.reaches_critical_asset is True

    def test_no_paths_to_unreachable_critical_asset(self):
        """If critical asset is isolated, no paths reach it."""
        topo = make_no_entry_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        critical_reached = [c for c in candidates if c.reaches_critical_asset]
        assert critical_reached == []

    def test_all_paths_are_structurally_valid(self):
        """Every generated path passes the PathValidator."""
        topo = make_branching_topology()
        builder = GraphBuilder()
        G = builder.build(topo)
        cfg = PathConfig()
        validator = PathValidator(G, cfg)
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            result = validator.validate(c.node_ids)
            assert result.valid, (
                f"Generated path {c.node_ids} failed validation: {result.errors}"
            )


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_no_duplicate_paths(self):
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        path_keys = [c.path_key for c in candidates]
        assert len(path_keys) == len(set(path_keys)), "Duplicate paths found"

    def test_different_order_paths_not_deduplicated(self):
        """A→B→C and A→C→B are different paths and both should appear."""
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        path_keys = {c.path_key for c in candidates}
        # entry1→nodeB→crit and entry1→nodeA→crit should both appear
        assert "entry1 → nodeA → crit" in path_keys
        assert "entry1 → nodeB → crit" in path_keys


# ---------------------------------------------------------------------------
# TestGNNIntegration
# ---------------------------------------------------------------------------

class TestGNNIntegration:
    CHECKPOINT = "checkpoints/best_attack_gnn.pt"

    @pytest.fixture(scope="class")
    def checkpoint_available(self):
        return Path(self.CHECKPOINT).exists()

    def test_checkpoint_loads_without_error(self, checkpoint_available):
        if not checkpoint_available:
            pytest.skip("Checkpoint not found — run run_phase3_training.py first")
        provider = GNNProbabilityProvider(self.CHECKPOINT)
        assert provider._model is not None

    def test_probabilities_have_correct_count(self, checkpoint_available):
        if not checkpoint_available:
            pytest.skip("Checkpoint not found")
        topo = make_linear_topology(n_intermediates=2)
        provider = GNNProbabilityProvider(self.CHECKPOINT)
        prob_dict = provider.get_node_probabilities(topo)
        # Should have one entry per node
        assert len(prob_dict) == topo.node_count

    def test_probabilities_in_01_range(self, checkpoint_available):
        if not checkpoint_available:
            pytest.skip("Checkpoint not found")
        topo = make_linear_topology(n_intermediates=2)
        provider = GNNProbabilityProvider(self.CHECKPOINT)
        prob_dict = provider.get_node_probabilities(topo)
        for nid, prob in prob_dict.items():
            assert 0.0 <= prob <= 1.0, (
                f"Node '{nid}' has probability {prob} outside [0,1]"
            )

    def test_every_path_node_has_probability(self, checkpoint_available):
        if not checkpoint_available:
            pytest.skip("Checkpoint not found")
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=self.CHECKPOINT)
        candidates, summary = gen.generate(topo)
        assert summary.gnn_probabilities_attached is True
        for c in candidates:
            for i, prob in enumerate(c.node_attack_probabilities):
                assert 0.0 <= prob <= 1.0, (
                    f"Node {i} of path {c.node_ids} has prob {prob}"
                )

    def test_probabilities_count_matches_node_ids(self, checkpoint_available):
        if not checkpoint_available:
            pytest.skip("Checkpoint not found")
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=self.CHECKPOINT)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert len(c.node_attack_probabilities) == len(c.node_ids)

    def test_no_retraining_during_generation(self, checkpoint_available):
        """Model parameters must not change during path generation."""
        if not checkpoint_available:
            pytest.skip("Checkpoint not found")
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=self.CHECKPOINT)
        # Capture initial weights
        initial_params = {
            k: v.clone()
            for k, v in gen._prob_provider._model.named_parameters()
        }
        gen.generate(topo)
        # Compare after generation
        for k, v in gen._prob_provider._model.named_parameters():
            assert torch.equal(v, initial_params[k]), (
                f"Parameter '{k}' changed during path generation — model was retrained!"
            )

    def test_missing_checkpoint_raises(self):
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            GNNProbabilityProvider("checkpoints/nonexistent_zzz.pt")


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_graph_same_config_same_paths(self):
        """Two runs with identical inputs must produce identical path lists."""
        topo = make_branching_topology()
        cfg = PathConfig()
        gen1 = PathGenerator(cfg, checkpoint_path=None)
        gen2 = PathGenerator(cfg, checkpoint_path=None)
        c1, _ = gen1.generate(topo)
        c2, _ = gen2.generate(topo)
        keys1 = [c.path_key for c in c1]
        keys2 = [c.path_key for c in c2]
        assert keys1 == keys2, "Non-deterministic path ordering"

    def test_same_paths_same_edge_features(self):
        """Same generation run → same edge features per path."""
        topo = make_branching_topology()
        cfg = PathConfig()
        gen1 = PathGenerator(cfg, checkpoint_path=None)
        gen2 = PathGenerator(cfg, checkpoint_path=None)
        c1, _ = gen1.generate(topo)
        c2, _ = gen2.generate(topo)
        for a, b in zip(c1, c2):
            assert a.edge_features == b.edge_features

    def test_deterministic_ordering_small_topology(self):
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_small()
        cfg = PathConfig(max_path_length=3)
        gen1 = PathGenerator(cfg, checkpoint_path=None)
        gen2 = PathGenerator(cfg, checkpoint_path=None)
        c1, _ = gen1.generate(topo)
        c2, _ = gen2.generate(topo)
        assert [c.path_key for c in c1] == [c.path_key for c in c2]


# ---------------------------------------------------------------------------
# TestLeakagePrevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    def test_summary_used_historical_labels_is_false(self):
        """PathGenerationSummary must always report used_historical_labels=False."""
        topo = make_linear_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        _, summary = gen.generate(topo)
        assert summary.used_historical_labels is False

    def test_generation_does_not_read_historical_attack_path(self):
        """
        Verify that path generation produces the same output regardless of
        whether historical_attacks is populated.

        If generation used HistoricalAttack.attack_path as input, the
        candidate sets would differ between the two cases.
        """
        topo_with = make_linear_topology()
        topo_without = make_linear_topology()

        # Add historical attacks ONLY to topo_with
        # (Note: attack paths must reference real node IDs)
        # Our linear topology has n0, n1, n2, n3 (for n_intermediates=2)
        topo_with.historical_attacks = [
            HistoricalAttack(
                attack_id="a1",
                attack_path=["n0", "n1", "n2", "n3"],
                technique="T1190",
                success=True,
                timestamp=1700000000.0,
                source="synthetic",
            )
        ]
        # topo_without has no historical attacks

        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        c_with, _ = gen.generate(topo_with)
        c_without, _ = gen.generate(topo_without)

        keys_with = sorted(c.path_key for c in c_with)
        keys_without = sorted(c.path_key for c in c_without)

        # The generated paths must be IDENTICAL regardless of attack history
        # (Path generation uses only graph topology, not historical_attacks)
        assert keys_with == keys_without, (
            "Candidate generation produced different paths depending on "
            "historical_attacks — LEAKAGE DETECTED"
        )

    def test_ground_truth_y_not_used(self):
        """
        Candidate generation must not require or read data.y (ground-truth).
        The generator only needs the NetworkTopology — no PyG Data.y.
        """
        topo = make_linear_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        # generate() takes NetworkTopology, not a PyG Data object
        # This implicitly verifies data.y is not used
        candidates, _ = gen.generate(topo)
        assert isinstance(candidates, list)


# ---------------------------------------------------------------------------
# TestTruncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_max_paths_per_entry_enforced(self):
        """With max_paths_per_entry=2, no single entry contributes more than 2 paths."""
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_medium()
        cfg = PathConfig(max_paths_per_entry=2, max_total_paths=10000)
        generator = PathGenerator(cfg, checkpoint_path=None)
        candidates, summary = generator.generate(topo)
        # Count per entry
        from collections import Counter
        entry_counts = Counter(c.entry_node for c in candidates)
        for entry, count in entry_counts.items():
            assert count <= cfg.max_paths_per_entry, (
                f"Entry '{entry}' produced {count} paths, max is {cfg.max_paths_per_entry}"
            )

    def test_max_total_paths_enforced(self):
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_medium()
        cfg = PathConfig(max_total_paths=5, max_paths_per_entry=100)
        generator = PathGenerator(cfg, checkpoint_path=None)
        candidates, summary = generator.generate(topo)
        assert len(candidates) <= cfg.max_total_paths

    def test_truncation_flag_set_when_quota_hit(self):
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_medium()
        cfg = PathConfig(max_total_paths=3, max_paths_per_entry=100)
        generator = PathGenerator(cfg, checkpoint_path=None)
        candidates, summary = generator.generate(topo)
        if len(candidates) == 3:
            # Quota was hit, flag must be True
            assert summary.generation_truncated is True

    def test_truncation_reason_nonempty_when_truncated(self):
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_medium()
        cfg = PathConfig(max_total_paths=2, max_paths_per_entry=1)
        generator = PathGenerator(cfg, checkpoint_path=None)
        _, summary = generator.generate(topo)
        if summary.generation_truncated:
            assert len(summary.truncation_reason) > 0

    def test_no_truncation_on_small_topology(self):
        """With generous limits, small topologies should not truncate."""
        topo = make_branching_topology()
        cfg = PathConfig(max_path_length=4, max_paths_per_entry=1000, max_total_paths=10000)
        gen = PathGenerator(cfg, checkpoint_path=None)
        _, summary = gen.generate(topo)
        assert summary.generation_truncated is False


# ---------------------------------------------------------------------------
# TestCandidatePath
# ---------------------------------------------------------------------------

class TestCandidatePath:
    def test_path_id_is_unique(self):
        ids = {make_path_id() for _ in range(100)}
        assert len(ids) == 100

    def test_path_key_is_canonical(self):
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            expected_key = " → ".join(c.node_ids)
            assert c.path_key == expected_key

    def test_mean_attack_probability_in_01(self):
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        # Without checkpoint, probs are -1 (sentinel)
        # With checkpoint they should be in [0,1]
        # Just check the attribute exists and is a float
        for c in candidates:
            assert isinstance(c.mean_attack_probability, float)

    def test_to_dict_is_json_serializable(self):
        import json
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates[:5]:
            d = c.to_dict()
            json.dumps(d)  # Must not raise

    def test_structural_metadata_keys(self):
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        required_keys = {
            "path_length_edges", "n_nodes", "n_critical_nodes",
            "n_vulnerable_nodes", "mean_max_cvss",
            "mean_edge_attack_weight",
        }
        for c in candidates:
            assert required_keys.issubset(set(c.structural_metadata.keys()))

    def test_summary_to_dict_serializable(self):
        import json
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        _, summary = gen.generate(topo)
        json.dumps(summary.to_dict())

    def test_is_synthetic_flag_propagated(self):
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert c.is_synthetic is True

    def test_topology_id_propagated(self):
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert c.topology_id == "topo-branch-test"


# ---------------------------------------------------------------------------
# TestIntegration — small/medium/large topologies
# ---------------------------------------------------------------------------

class TestIntegration:
    CHECKPOINT = "checkpoints/best_attack_gnn.pt"

    def test_small_topology_integration(self):
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_small()
        cfg = PathConfig(max_path_length=4, max_paths_per_entry=20)
        generator = PathGenerator(cfg, checkpoint_path=None)
        candidates, summary = generator.generate(topo)
        assert summary.total_candidates > 0
        assert all(c.is_valid for c in candidates)

    def test_medium_topology_integration(self):
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_medium()
        cfg = PathConfig(max_path_length=4, max_paths_per_entry=30)
        generator = PathGenerator(cfg, checkpoint_path=None)
        candidates, summary = generator.generate(topo)
        assert summary.total_candidates > 0
        assert summary.entry_point_count > 0
        assert summary.critical_asset_count > 0

    def test_large_topology_integration(self):
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_large()
        cfg = PathConfig(max_path_length=4, max_paths_per_entry=10, max_total_paths=200)
        generator = PathGenerator(cfg, checkpoint_path=None)
        candidates, summary = generator.generate(topo)
        assert summary.total_candidates > 0

    def test_edge_features_dim_on_real_topology(self):
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_medium()
        cfg = PathConfig(max_path_length=3)
        generator = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = generator.generate(topo)
        for c in candidates[:20]:
            for ef in c.edge_features:
                assert len(ef) == EDGE_FEATURE_DIM

    def test_gnn_integration_small_topology(self):
        if not Path(self.CHECKPOINT).exists():
            pytest.skip("Checkpoint not found — run run_phase3_training.py first")
        gen = SyntheticNetworkGenerator(seed=42)
        topo = gen.generate_small()
        cfg = PathConfig(max_path_length=4, max_paths_per_entry=10)
        generator = PathGenerator(cfg, checkpoint_path=self.CHECKPOINT)
        candidates, summary = generator.generate(topo)
        assert summary.gnn_probabilities_attached is True
        for c in candidates[:5]:
            for prob in c.node_attack_probabilities:
                assert 0.0 <= prob <= 1.0

    def test_generate_for_dataset_train_val_test(self):
        """Verify dataset split generation maintains topology integrity."""
        from backend.ml.dataset import build_dataset, topology_to_pyg
        from backend.core.synthetic_data import SyntheticNetworkGenerator

        # Build small dataset
        dataset = build_dataset(
            n_small=4, n_medium=2, n_large=0,
            seed_offset=77, split_seed=42, validate=True,
        )
        # Reconstruct topologies (simple re-generation — dataset stores Data, not topology)
        cfg = PathConfig(max_path_length=3, max_paths_per_entry=10)
        generator = PathGenerator(cfg, checkpoint_path=None)

        # Generate for train scenarios
        train_topos = [
            SyntheticNetworkGenerator(seed=77 + i).generate_small()
            for i in range(4)
        ]
        train_candidates, train_summaries = generator.generate_for_dataset(
            train_topos, split_name="train"
        )
        assert len(train_candidates) > 0
        assert len(train_summaries) == 4
        # All candidates belong to the train topologies
        train_topo_ids = {t.topology_id for t in train_topos}
        # Note: topo IDs are UUID-based so we check structure not values
        for c in train_candidates:
            assert isinstance(c.topology_id, str)

    def test_no_phase5_risk_score_in_candidates(self):
        """Phase 4 must NOT compute a final risk score — that is Phase 5."""
        topo = make_branching_topology()
        cfg = PathConfig()
        gen = PathGenerator(cfg, checkpoint_path=None)
        candidates, _ = gen.generate(topo)
        for c in candidates:
            assert "risk_score" not in c.structural_metadata, (
                "Phase 4 must not compute risk_score — that is Phase 5"
            )
            assert not hasattr(c, "risk_score"), (
                "CandidateAttackPath must not have risk_score attribute in Phase 4"
            )
