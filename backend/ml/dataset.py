"""
dataset.py
----------
PyTorch Geometric data pipeline for the Graph-Based Attack Path Predictor.

=============================================================================
TASK 1 — 43-FEATURE AUDIT (full index table, no duplicates)
=============================================================================

Node feature vector layout per node: shape (43,)

Block A — Raw Scalar Features (indices 0–19)  [source: NetworkNode attributes]
─────────────────────────────────────────────────────────────────────────────
Idx  Name                      Type     Meaning
 0   is_entry_point            binary   1 if attackers can reach node directly
 1   is_critical_asset         binary   1 if high-value target
 2   asset_value               float    Business value [0.0–1.0]
 3   vuln_count                int      Number of known vulnerabilities
 4   max_cvss                  float    Highest CVSS v3 score on node [0.0–10.0]
 5   has_critical_vuln         binary   1 if any vuln is CRITICAL (CVSS ≥ 9.0)
 6   has_exploited_in_wild     binary   1 if any vuln is actively exploited
 7   critical_vuln_count       int      Count of CRITICAL severity vulns
 8   high_vuln_count           int      Count of HIGH severity vulns
 9   unpatched_vuln_count      int      Count of vulns with no patch available
10   exploited_in_wild_count   int      Count of vulns exploited in the wild
11   avg_cvss                  float    Mean CVSS score across all vulns
12   avg_exploitability        float    Mean CVSS exploitability sub-score
13   service_count             int      Number of open services/ports
14   has_rdp                   binary   1 if RDP (port 3389) is open
15   has_ssh                   binary   1 if SSH (port 22) is open
16   has_smb                   binary   1 if SMB (port 445) is open
17   has_http                  binary   1 if HTTP (port 80) is open
18   has_https                 binary   1 if HTTPS (port 443) is open
19   compromised               binary   1 if node is currently flagged compromised

Block B — Node Type One-Hot (indices 20–27)  [source: NodeType enum]
─────────────────────────────────────────────────────────────────────────────
20   nodetype_workstation      binary   Node is a workstation
21   nodetype_server           binary   Node is a server
22   nodetype_database         binary   Node is a database
23   nodetype_router           binary   Node is a router
24   nodetype_firewall         binary   Node is a firewall
25   nodetype_internet         binary   Node is an internet gateway (entry)
26   nodetype_iot_device       binary   Node is an IoT/OT device
27   nodetype_critical_asset   binary   Node is a critical asset (DC/SCADA)

Block C — OS Type One-Hot (indices 28–32)  [source: OSType enum]
─────────────────────────────────────────────────────────────────────────────
28   ostype_windows            binary   Node runs Windows
29   ostype_linux              binary   Node runs Linux
30   ostype_macos              binary   Node runs macOS
31   ostype_firmware           binary   Node runs Firmware (IoT/router)
32   ostype_unknown            binary   OS type unknown

Block D — Structural Graph Features (indices 33–42)  [source: graph topology]
─────────────────────────────────────────────────────────────────────────────
33   in_degree                 int      Number of incoming directed edges
34   out_degree                int      Number of outgoing directed edges
35   total_degree              int      in_degree + out_degree
36   pagerank                  float    PageRank score (α=0.85, attack_weight)
37   betweenness_centrality    float    Approximate betweenness centrality
38   closeness_centrality      float    Closeness centrality
39   clustering_coeff          float    Clustering coefficient (undirected)
40   attack_weighted_in_degree float    Sum of attack_weight of incoming edges
41   anomaly_signal            float    Sum of anomaly_count of incoming edges
42   historical_attack_exposure float   Sum of historical_attack_count on in-edges

Total: 43 features

=============================================================================
TASK 2 — ML TARGET DEFINITION
=============================================================================

What the GNN predicts: NODE-LEVEL BINARY LABEL

    y[i] ∈ {0, 1}   for each node i in the graph

  Positive class (y=1): "This node appears on at least one recorded attack path"
  Negative class (y=0): "This node does NOT appear on any recorded attack path"

Rationale:
  - Historical attack paths already exist in the topology (HistoricalAttack).
  - Each attack path is an ordered list of node IDs traversed by an attacker.
  - Any node that appears in ANY position of any attack path is a positive.
  - This is the simplest valid label derivable from existing data that teaches
    the GNN "which nodes tend to appear in real (simulated) attack paths".
  - Entry points and critical assets always appear in at least one path, so
    they are naturally positive — but many intermediate nodes are also positive,
    giving the GNN a useful signal about lateral movement structure.

Label generation algorithm (see _build_node_labels()):
  1. Collect all node IDs that appear in any position in any HistoricalAttack.
  2. For each node in the graph: y = 1 if node in attacked_set else 0.
  3. Labels are deterministic given the same topology + attack history.

This is NOT arbitrary — it directly encodes which nodes were traversed by
simulated attackers according to the synthetic kill-chain logic in Phase 1.

Edge features used as edge_attr in PyG Data:
  - edge_type_network_link    (one-hot)
  - edge_type_trust           (one-hot)
  - edge_type_exploit         (one-hot)
  - edge_type_data_flow       (one-hot)
  - firewall_rule             (binary)
  - is_encrypted              (binary)
  - attack_weight             (float)
  - anomaly_count             (int → float)
  - historical_attack_count   (int → float)
  Total edge feature dim: 9

=============================================================================
TASK 4 — DATASET SPLITTING DESIGN
=============================================================================

What is split: GRAPH SCENARIOS (not individual nodes)

Rationale for scenario-level splits:
  - A single graph encodes ALL attack paths from the same synthetic scenario.
  - Splitting individual nodes from the SAME graph would create leakage: the
    GNN could infer labels for "test" nodes by reading their neighbors'
    structural features which were computed from the full graph.
  - Splitting GRAPH SCENARIOS means train/val/test contain completely different
    networks (different topologies, different attack paths). The GNN must
    generalize to unseen network structures.

Split strategy:
  - Generate N scenarios across small/medium/large topologies with different seeds.
  - Split scenarios 70/15/15 by count (seeded shuffle for reproducibility).
  - Each scenario → one PyG Data object.
  - No node-level overlap between splits (different topologies, different node IDs).

Minimum recommended: 20 scenarios (14 train / 3 val / 3 test).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from backend.core.data_models import NetworkTopology
from backend.core.feature_extractor import (
    ALL_FEATURE_NAMES,
    FEATURE_DIM,
    FeatureExtractor,
)
from backend.core.graph_builder import GraphBuilder
from backend.core.synthetic_data import SyntheticNetworkGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edge feature constants
# ---------------------------------------------------------------------------

EDGE_TYPE_CATEGORIES = [
    "network_link",
    "trust_relationship",
    "exploit_path",
    "data_flow",
]

EDGE_SCALAR_FEATURE_NAMES = [
    "firewall_rule",
    "is_encrypted",
    "attack_weight",
    "anomaly_count",
    "historical_attack_count",
]

EDGE_TYPE_FEATURE_NAMES = [f"edgetype_{c}" for c in EDGE_TYPE_CATEGORIES]

ALL_EDGE_FEATURE_NAMES = EDGE_TYPE_FEATURE_NAMES + EDGE_SCALAR_FEATURE_NAMES
EDGE_FEATURE_DIM = len(ALL_EDGE_FEATURE_NAMES)  # 4 + 5 = 9


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def build_node_labels(G: nx.DiGraph, topology: NetworkTopology) -> torch.Tensor:
    """
    Generate binary node labels from historical attack paths.

    y[i] = 1 if node i appears on at least one historical attack path
    y[i] = 0 otherwise

    This is derived directly from HistoricalAttack.attack_path lists in the
    topology — NOT invented or hard-coded.

    Args:
        G:        The NetworkX DiGraph (determines node ordering).
        topology: The source NetworkTopology (contains historical_attacks).

    Returns:
        torch.Tensor of shape (N,) with dtype torch.long (0 or 1).
    """
    # Collect all node IDs that appear in any attack path
    attacked_nodes: set = set()
    for attack in topology.historical_attacks:
        for node_id in attack.attack_path:
            attacked_nodes.add(node_id)

    node_order = list(G.nodes())
    labels = [1 if nid in attacked_nodes else 0 for nid in node_order]
    return torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Edge feature extraction
# ---------------------------------------------------------------------------

def build_edge_features(G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge_index and edge_attr tensors from a NetworkX DiGraph.

    edge_index: shape (2, E) — COO format, [source_indices, target_indices]
    edge_attr:  shape (E, EDGE_FEATURE_DIM) — one feature vector per edge

    Edge features (9 total):
      [0:4]  One-hot edge type (network_link / trust / exploit / data_flow)
      [4]    firewall_rule  (binary)
      [5]    is_encrypted   (binary)
      [6]    attack_weight  (float)
      [7]    anomaly_count  (float)
      [8]    historical_attack_count (float)

    Node ordering follows list(G.nodes()), same as feature matrix ordering.

    Returns:
        (edge_index, edge_attr)
    """
    node_order = list(G.nodes())
    node_to_idx = {nid: i for i, nid in enumerate(node_order)}
    etype_to_idx = {c: i for i, c in enumerate(EDGE_TYPE_CATEGORIES)}

    src_list, tgt_list = [], []
    attr_list = []

    for u, v, d in G.edges(data=True):
        src_list.append(node_to_idx[u])
        tgt_list.append(node_to_idx[v])

        # One-hot edge type
        etype_vec = [0.0] * len(EDGE_TYPE_CATEGORIES)
        etype_raw = d.get("edge_type", "network_link")
        etype_idx = etype_to_idx.get(etype_raw, 0)
        etype_vec[etype_idx] = 1.0

        # Scalar edge features
        scalar_vec = [
            float(d.get("firewall_rule", 0)),
            float(d.get("is_encrypted", 0)),
            float(d.get("attack_weight", 0.5)),
            float(d.get("anomaly_count", 0)),
            float(d.get("historical_attack_count", 0)),
        ]

        attr_list.append(etype_vec + scalar_vec)

    if not src_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.float)

    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# PyG Data conversion
# ---------------------------------------------------------------------------

def topology_to_pyg(
    topology: NetworkTopology,
    normalize_features: bool = True,
) -> Tuple[Data, List[str], FeatureExtractor]:
    """
    Convert a NetworkTopology into a PyTorch Geometric Data object.

    Produces:
        data.x          — node feature matrix, shape (N, 43), float32
        data.edge_index — COO edge index,       shape (2, E), int64
        data.edge_attr  — edge feature matrix,  shape (E, 9),  float32
        data.y          — node labels,           shape (N,),    int64
        data.num_nodes  — N (set explicitly for PyG compatibility)

    Also stores on data:
        data.node_ids        — list[str]  original node IDs in row order
        data.topology_id     — str        topology identifier
        data.topology_name   — str        human-readable name
        data.is_synthetic    — bool       always True for our data
        data.positive_count  — int        nodes with y=1
        data.negative_count  — int        nodes with y=0

    Args:
        topology:           Source NetworkTopology.
        normalize_features: Whether to MinMax-normalize x. Default True.

    Returns:
        (data, node_index, extractor)
        - data:       torch_geometric.data.Data object
        - node_index: list of node IDs in feature row order
        - extractor:  fitted FeatureExtractor (holds scaler for inverse transforms)
    """
    builder = GraphBuilder()
    G = builder.build(topology)

    # --- Node features ---
    extractor = FeatureExtractor(normalize=normalize_features)
    x_np, node_index = extractor.extract(G)
    x = torch.tensor(x_np, dtype=torch.float)

    # --- Edge structure + features ---
    edge_index, edge_attr = build_edge_features(G)

    # --- Labels ---
    y = build_node_labels(G, topology)

    # --- Assemble Data ---
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=G.number_of_nodes(),
    )

    # Store metadata as attributes (PyG Data supports arbitrary attributes)
    data.node_ids = node_index
    data.topology_id = topology.topology_id
    data.topology_name = topology.name
    data.is_synthetic = topology.is_synthetic
    data.positive_count = int(y.sum().item())
    data.negative_count = int((y == 0).sum().item())

    return data, node_index, extractor


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class DataValidationReport:
    """Results of a single PyG Data object validation pass."""
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_pyg_data(data: Data) -> DataValidationReport:
    """
    Validate a PyG Data object for correctness before model training.

    Checks:
      1. x exists and has correct feature dimension
      2. edge_index has shape (2, E) and valid integer dtype
      3. edge_attr has correct edge feature dimension
      4. y exists, has N entries, only contains 0/1
      5. No NaN or Inf values in x, edge_attr
      6. edge_index node indices are within [0, N-1]
      7. No self-loops in edge_index
      8. At least one positive (y=1) label
      9. At least one negative (y=0) label
     10. num_nodes is set and matches x.shape[0]

    Returns:
        DataValidationReport with .passed, .errors, .warnings
    """
    report = DataValidationReport()
    N = data.num_nodes

    # 1. x existence and shape
    if data.x is None:
        report.fail("data.x is None — node features missing.")
        return report  # Cannot continue without x

    if data.x.shape[0] != N:
        report.fail(
            f"data.x has {data.x.shape[0]} rows but num_nodes={N}."
        )
    if data.x.shape[1] != FEATURE_DIM:
        report.fail(
            f"data.x has {data.x.shape[1]} features but expected FEATURE_DIM={FEATURE_DIM}."
        )

    # 2. edge_index
    if data.edge_index is None:
        report.fail("data.edge_index is None — edge structure missing.")
    else:
        if data.edge_index.shape[0] != 2:
            report.fail(
                f"edge_index must have shape (2, E), got {data.edge_index.shape}."
            )
        if data.edge_index.dtype not in (torch.long, torch.int64):
            report.fail(
                f"edge_index must be torch.long, got {data.edge_index.dtype}."
            )
        if data.edge_index.numel() > 0:
            min_idx = int(data.edge_index.min().item())
            max_idx = int(data.edge_index.max().item())
            if min_idx < 0:
                report.fail(f"edge_index contains negative node index: {min_idx}.")
            if max_idx >= N:
                report.fail(
                    f"edge_index references node index {max_idx} but N={N}."
                )
            # Self-loops
            src, tgt = data.edge_index[0], data.edge_index[1]
            self_loops = int((src == tgt).sum().item())
            if self_loops > 0:
                report.fail(f"edge_index contains {self_loops} self-loop(s).")

    # 3. edge_attr
    if data.edge_attr is None:
        report.warn("data.edge_attr is None — edge features will not be used.")
    else:
        if data.edge_index is not None and data.edge_index.numel() > 0:
            E = data.edge_index.shape[1]
            if data.edge_attr.shape[0] != E:
                report.fail(
                    f"edge_attr has {data.edge_attr.shape[0]} rows but E={E}."
                )
            if data.edge_attr.shape[1] != EDGE_FEATURE_DIM:
                report.fail(
                    f"edge_attr has {data.edge_attr.shape[1]} edge features "
                    f"but expected {EDGE_FEATURE_DIM}."
                )

    # 4. Labels
    if data.y is None:
        report.fail("data.y is None — labels missing.")
    else:
        if data.y.shape[0] != N:
            report.fail(f"data.y has {data.y.shape[0]} entries but N={N}.")
        unique_labels = set(data.y.unique().tolist())
        invalid = unique_labels - {0, 1}
        if invalid:
            report.fail(f"data.y contains invalid label values: {invalid}.")

    # 5. NaN / Inf checks
    if data.x is not None:
        if torch.isnan(data.x).any().item():
            report.fail("data.x contains NaN values.")
        if torch.isinf(data.x).any().item():
            report.fail("data.x contains Inf values.")
    if data.edge_attr is not None:
        if torch.isnan(data.edge_attr).any().item():
            report.fail("data.edge_attr contains NaN values.")
        if torch.isinf(data.edge_attr).any().item():
            report.fail("data.edge_attr contains Inf values.")

    # 6–9: Label balance
    if data.y is not None and data.y.numel() > 0:
        pos = int((data.y == 1).sum().item())
        neg = int((data.y == 0).sum().item())
        if pos == 0:
            report.warn(
                "No positive labels (y=1) — no attack paths recorded? "
                "Check that topology has historical_attacks."
            )
        if neg == 0:
            report.warn(
                "No negative labels (y=0) — all nodes on attack paths. "
                "Labels may not provide useful signal."
            )

    # 10. num_nodes
    if data.num_nodes != N:
        report.fail(
            f"data.num_nodes={data.num_nodes} does not match x.shape[0]={N}."
        )

    return report


# ---------------------------------------------------------------------------
# Dataset generation + scenario-level splitting
# ---------------------------------------------------------------------------

@dataclass
class AttackGraphDataset:
    """
    A collection of PyG Data objects, one per network scenario.

    Supports train/val/test scenario-level splitting with leakage prevention.

    Split methodology:
      - Each Data object corresponds to one full network topology (scenario).
      - Scenarios are shuffled with a fixed random seed, then split 70/15/15.
      - All nodes, edges, and labels within a scenario are kept together.
      - There is ZERO node-level overlap between splits because each scenario
        has a completely different set of node IDs (UUID-based).

    Attributes:
        data_list:    All Data objects (all scenarios).
        train_data:   Training set (70% of scenarios).
        val_data:     Validation set (15% of scenarios).
        test_data:    Test set (15% of scenarios).
        split_seed:   Random seed used for splitting (for reproducibility).
        metadata:     Per-scenario metadata dicts.
    """
    data_list: List[Data] = field(default_factory=list)
    train_data: List[Data] = field(default_factory=list)
    val_data: List[Data] = field(default_factory=list)
    test_data: List[Data] = field(default_factory=list)
    split_seed: int = 42
    metadata: List[Dict] = field(default_factory=list)

    @property
    def total_scenarios(self) -> int:
        return len(self.data_list)

    @property
    def total_nodes(self) -> int:
        return sum(d.num_nodes for d in self.data_list)

    @property
    def total_positive_labels(self) -> int:
        return sum(getattr(d, "positive_count", 0) for d in self.data_list)

    @property
    def total_negative_labels(self) -> int:
        return sum(getattr(d, "negative_count", 0) for d in self.data_list)

    def split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        """
        Split data_list into train/val/test at the scenario level.

        Raises ValueError if ratios don't sum to 1.0 or if there are
        too few scenarios to have at least 1 in each split.

        Leakage prevention:
          - Splitting is done on SCENARIO index, not on node index.
          - Each scenario's node IDs are UUID-based and unique across scenarios,
            so there is no shared node identity across splits.
          - Graph topology and attack paths are entirely within each scenario.
        """
        if not self.data_list:
            raise ValueError("data_list is empty — generate scenarios first.")

        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total:.4f} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
            )

        N = len(self.data_list)
        if N < 3:
            raise ValueError(
                f"Need at least 3 scenarios to split (got {N}). "
                "Generate more scenarios."
            )

        # Seed-controlled shuffle — reproducible
        indices = list(range(N))
        rng = random.Random(self.split_seed)
        rng.shuffle(indices)

        n_train = max(1, int(N * train_ratio))
        n_val = max(1, int(N * val_ratio))
        n_test = N - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = N - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        self.train_data = [self.data_list[i] for i in train_idx]
        self.val_data = [self.data_list[i] for i in val_idx]
        self.test_data = [self.data_list[i] for i in test_idx]

        logger.info(
            "Dataset split: %d train / %d val / %d test scenarios (seed=%d).",
            len(self.train_data), len(self.val_data),
            len(self.test_data), self.split_seed,
        )

    def check_leakage(self) -> Dict:
        """
        Verify that no node IDs (topology IDs) appear in more than one split.

        Returns a dict with leakage report.
        """
        def get_ids(split_data: List[Data]) -> set:
            return {getattr(d, "topology_id", None) for d in split_data}

        train_ids = get_ids(self.train_data)
        val_ids = get_ids(self.val_data)
        test_ids = get_ids(self.test_data)

        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids

        no_leakage = (
            len(train_val_overlap) == 0
            and len(train_test_overlap) == 0
            and len(val_test_overlap) == 0
        )

        return {
            "no_leakage": no_leakage,
            "train_val_overlap": list(train_val_overlap),
            "train_test_overlap": list(train_test_overlap),
            "val_test_overlap": list(val_test_overlap),
            "train_topology_count": len(train_ids),
            "val_topology_count": len(val_ids),
            "test_topology_count": len(test_ids),
        }

    def summary(self) -> Dict:
        """Return a human-readable summary dict of the dataset."""
        return {
            "total_scenarios": self.total_scenarios,
            "train_scenarios": len(self.train_data),
            "val_scenarios": len(self.val_data),
            "test_scenarios": len(self.test_data),
            "total_nodes": self.total_nodes,
            "total_positive_labels": self.total_positive_labels,
            "total_negative_labels": self.total_negative_labels,
            "node_feature_dim": FEATURE_DIM,
            "edge_feature_dim": EDGE_FEATURE_DIM,
            "split_seed": self.split_seed,
        }


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_dataset(
    n_small: int = 10,
    n_medium: int = 7,
    n_large: int = 3,
    seed_offset: int = 0,
    split_seed: int = 42,
    normalize_features: bool = True,
    validate: bool = True,
) -> AttackGraphDataset:
    """
    Generate a full multi-scenario dataset and split it.

    Each scenario uses a unique random seed (seed_offset + scenario_index)
    for reproducibility. Scenario seeds are determined solely by:
      - n_small, n_medium, n_large
      - seed_offset

    Args:
        n_small:           Number of small (≈10-node) topology scenarios.
        n_medium:          Number of medium (≈30-node) topology scenarios.
        n_large:           Number of large (≈60-node) topology scenarios.
        seed_offset:       Base seed offset for all scenario generators.
        split_seed:        Seed for train/val/test shuffling.
        normalize_features: Apply MinMax normalization to node features.
        validate:          Run DataValidationReport on each Data object.

    Returns:
        AttackGraphDataset with train/val/test splits populated.

    Raises:
        ValueError if any Data object fails validation.
    """
    dataset = AttackGraphDataset(split_seed=split_seed)
    scenario_idx = 0

    for scenario_type, n_scenarios, gen_method in [
        ("small", n_small, "generate_small"),
        ("medium", n_medium, "generate_medium"),
        ("large", n_large, "generate_large"),
    ]:
        for i in range(n_scenarios):
            seed = seed_offset + scenario_idx
            gen = SyntheticNetworkGenerator(seed=seed)
            topology: NetworkTopology = getattr(gen, gen_method)()

            data, node_index, extractor = topology_to_pyg(
                topology, normalize_features=normalize_features
            )

            if validate:
                report = validate_pyg_data(data)
                if not report.passed:
                    raise ValueError(
                        f"Scenario {scenario_type}[{i}] (seed={seed}) "
                        f"failed validation:\n  " + "\n  ".join(report.errors)
                    )
                if report.warnings:
                    for w in report.warnings:
                        logger.warning("Scenario %s[%d] seed=%d: %s", scenario_type, i, seed, w)

            meta = {
                "scenario_type": scenario_type,
                "seed": seed,
                "topology_id": topology.topology_id,
                "topology_name": topology.name,
                "node_count": data.num_nodes,
                "edge_count": data.edge_index.shape[1],
                "positive_labels": int(data.positive_count),
                "negative_labels": int(data.negative_count),
                "positive_ratio": round(
                    data.positive_count / data.num_nodes, 4
                ) if data.num_nodes > 0 else 0.0,
            }
            dataset.data_list.append(data)
            dataset.metadata.append(meta)

            logger.info(
                "Scenario %s[%d] seed=%d: %d nodes, %d edges, "
                "%d pos / %d neg labels",
                scenario_type, i, seed,
                data.num_nodes, data.edge_index.shape[1],
                data.positive_count, data.negative_count,
            )
            scenario_idx += 1

    dataset.split()
    return dataset
