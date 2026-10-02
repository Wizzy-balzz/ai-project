"""
path_models.py
--------------
Data structures for Phase 4 candidate attack path generation.

Design decisions:
- PathConfig is the single source of truth for all generation parameters.
  No magic numbers anywhere in the generation code.
- CandidateAttackPath is an immutable-style dataclass holding all information
  needed for Phase 5 risk scoring: node IDs, GNN probabilities, edge features,
  structural metadata.
- PathGenerationSummary aggregates metadata about a full generation run,
  including truncation flags (NOT silently suppressed).
- All structures serialize to plain dicts (JSON-compatible).
- Consistent with existing data_models.py style (dataclasses, to_dict()).

Path length is defined as the NUMBER OF EDGES traversed:
  [A] → [B] → [C]  has path_length = 2
  [A] → [B]         has path_length = 1

This matches standard graph theory convention and is tested explicitly.

IMPORTANT — Leakage prevention:
  CandidateAttackPath NEVER stores or reads HistoricalAttack.attack_path.
  The 'is_synthetic' flag records data provenance.
  'topology_id' allows later evaluation to link candidates to scenarios.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Path Generation Configuration
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    """
    Configuration for candidate path generation.

    All generation parameters are centralised here.
    No magic numbers in path_generator.py.

    Attributes:
        min_path_length:          Minimum edges in a valid path (≥1).
        max_path_length:          Maximum edges (cutoff for DFS). Default 6.
        max_paths_per_entry:      Maximum candidates per entry node. Default 50.
        max_total_paths:          Hard cap on total candidates. Default 2000.
        require_critical_terminal: If True, only emit paths ending at a critical asset.
        allow_noncritical_terminal: If False, discard non-critical terminals.
                                    (Has no effect when require_critical_terminal=True.)
        sort_by:                  Deterministic sort key for output ordering.
                                  Options: "path_str" (lexicographic on node IDs),
                                           "length", "mean_prob"
    """
    min_path_length: int = 1
    max_path_length: int = 6
    max_paths_per_entry: int = 50
    max_total_paths: int = 2000
    require_critical_terminal: bool = False
    allow_noncritical_terminal: bool = True
    sort_by: str = "path_str"   # Deterministic default

    def __post_init__(self) -> None:
        if self.min_path_length < 1:
            raise ValueError(
                f"min_path_length must be ≥1 (number of edges), got {self.min_path_length}."
            )
        if self.max_path_length < self.min_path_length:
            raise ValueError(
                f"max_path_length ({self.max_path_length}) must be ≥ "
                f"min_path_length ({self.min_path_length})."
            )
        if self.max_paths_per_entry < 1:
            raise ValueError(
                f"max_paths_per_entry must be ≥1, got {self.max_paths_per_entry}."
            )
        if self.sort_by not in ("path_str", "length", "mean_prob"):
            raise ValueError(
                f"sort_by must be 'path_str', 'length', or 'mean_prob', "
                f"got '{self.sort_by}'."
            )

    def to_dict(self) -> Dict:
        return {
            "min_path_length": self.min_path_length,
            "max_path_length": self.max_path_length,
            "max_paths_per_entry": self.max_paths_per_entry,
            "max_total_paths": self.max_total_paths,
            "require_critical_terminal": self.require_critical_terminal,
            "allow_noncritical_terminal": self.allow_noncritical_terminal,
            "sort_by": self.sort_by,
        }


# ---------------------------------------------------------------------------
# Candidate Attack Path
# ---------------------------------------------------------------------------

@dataclass
class CandidateAttackPath:
    """
    A single candidate attack path through the network graph.

    Path length = number of edges = len(node_ids) - 1.

    Fields:
        path_id:
            Unique UUID identifier for this path.

        node_ids:
            Ordered list of node IDs. len(node_ids) ≥ 2.
            node_ids[0]  = entry node
            node_ids[-1] = terminal node

        edge_pairs:
            List of (source_node_id, target_node_id) tuples for each hop.
            len(edge_pairs) == len(node_ids) - 1.

        node_labels:
            Human-readable labels for each node (parallel to node_ids).

        node_attack_probabilities:
            GNN-predicted P(node involved in attack) for each node.
            Values in [0.0, 1.0]. Computed from the trained Phase 3 model.
            len == len(node_ids).
            NEVER hard-coded or fabricated.

        edge_features:
            List of edge feature vectors (one per hop).
            Each vector has length == EDGE_FEATURE_DIM (9).
            Values come from data.edge_attr.

        path_length:
            Number of edges = len(node_ids) - 1.

        entry_node:
            node_id of the first node (must be an entry point).

        terminal_node:
            node_id of the last node.

        reaches_critical_asset:
            True if terminal_node has is_critical_asset == 1.

        is_valid:
            True if the path passed all validation checks.

        topology_id:
            Links back to the source NetworkTopology for evaluation tracing.

        is_synthetic:
            Data provenance flag. Always True for our data.

        structural_metadata:
            Pre-computed path features for Phase 5 risk scoring.
            Does NOT include a final risk score (Phase 5 concern).
    """
    path_id: str
    node_ids: List[str]
    edge_pairs: List[Tuple[str, str]]
    node_labels: List[str]
    node_attack_probabilities: List[float]
    edge_features: List[List[float]]  # shape: (path_length, EDGE_FEATURE_DIM)
    path_length: int                  # number of edges = len(node_ids) - 1
    entry_node: str
    terminal_node: str
    reaches_critical_asset: bool
    is_valid: bool = True
    topology_id: str = ""
    is_synthetic: bool = True
    structural_metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "path_id": self.path_id,
            "node_ids": self.node_ids,
            "edge_pairs": [list(e) for e in self.edge_pairs],
            "node_labels": self.node_labels,
            "node_attack_probabilities": [round(p, 6) for p in self.node_attack_probabilities],
            "edge_features": [[round(v, 6) for v in ef] for ef in self.edge_features],
            "path_length": self.path_length,
            "entry_node": self.entry_node,
            "terminal_node": self.terminal_node,
            "reaches_critical_asset": self.reaches_critical_asset,
            "is_valid": self.is_valid,
            "topology_id": self.topology_id,
            "is_synthetic": self.is_synthetic,
            "structural_metadata": self.structural_metadata,
        }

    @property
    def mean_attack_probability(self) -> float:
        if not self.node_attack_probabilities:
            return 0.0
        return sum(self.node_attack_probabilities) / len(self.node_attack_probabilities)

    @property
    def min_attack_probability(self) -> float:
        if not self.node_attack_probabilities:
            return 0.0
        return min(self.node_attack_probabilities)

    @property
    def max_attack_probability(self) -> float:
        if not self.node_attack_probabilities:
            return 0.0
        return max(self.node_attack_probabilities)

    @property
    def path_key(self) -> str:
        """Canonical string representation for deduplication."""
        return " → ".join(self.node_ids)


# ---------------------------------------------------------------------------
# Path Validation Result
# ---------------------------------------------------------------------------

@dataclass
class PathValidationResult:
    """Result of validating a single candidate path."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def to_dict(self) -> Dict:
        return {"valid": self.valid, "errors": self.errors}


# ---------------------------------------------------------------------------
# Path Generation Summary
# ---------------------------------------------------------------------------

@dataclass
class PathGenerationSummary:
    """
    Summary of a complete path generation run for one topology scenario.

    Truncation information is NEVER suppressed — if any generation limit
    was hit, generation_truncated is True and truncation_reason explains why.
    """
    topology_id: str = ""
    topology_name: str = ""
    is_synthetic: bool = True

    # Counts
    total_candidates: int = 0
    critical_terminal_count: int = 0
    noncritical_terminal_count: int = 0
    entry_point_count: int = 0
    critical_asset_count: int = 0

    # Truncation
    generation_truncated: bool = False
    truncation_reason: str = ""

    # Generation inputs
    config: Dict = field(default_factory=dict)
    graph_node_count: int = 0
    graph_edge_count: int = 0

    # Whether model probabilities were attached
    gnn_probabilities_attached: bool = False
    gnn_checkpoint_used: Optional[str] = None

    # Edge features
    edge_features_attached: bool = False

    # Whether generation used historical attack labels (MUST be False)
    used_historical_labels: bool = False

    def to_dict(self) -> Dict:
        return {
            "topology_id": self.topology_id,
            "topology_name": self.topology_name,
            "is_synthetic": self.is_synthetic,
            "total_candidates": self.total_candidates,
            "critical_terminal_count": self.critical_terminal_count,
            "noncritical_terminal_count": self.noncritical_terminal_count,
            "entry_point_count": self.entry_point_count,
            "critical_asset_count": self.critical_asset_count,
            "generation_truncated": self.generation_truncated,
            "truncation_reason": self.truncation_reason,
            "config": self.config,
            "graph_node_count": self.graph_node_count,
            "graph_edge_count": self.graph_edge_count,
            "gnn_probabilities_attached": self.gnn_probabilities_attached,
            "gnn_checkpoint_used": self.gnn_checkpoint_used,
            "edge_features_attached": self.edge_features_attached,
            "used_historical_labels": self.used_historical_labels,
        }


def make_path_id() -> str:
    """Generate a unique path identifier."""
    return "path-" + str(uuid.uuid4()).replace("-", "")[:12]
