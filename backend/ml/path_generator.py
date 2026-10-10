"""
path_generator.py
-----------------
Candidate attack-path generator for the Graph-Based Attack Path Predictor.

=============================================================================
ALGORITHM: Bounded Depth-First Search (DFS) with path deduplication
=============================================================================

Why bounded DFS?
- The network graph is directed; BFS would require storing all frontier paths
  simultaneously, which is memory-intensive on dense graphs.
- DFS with a depth bound (max_path_length edges) naturally prunes the search
  space without requiring all paths to be held in memory at once.
- Backtracking naturally handles cycles and repeated-node detection.
- The output order is deterministic when neighbors are sorted.

Algorithm per entry node:
  1. Start at entry node with visited={entry}, current_path=[entry].
  2. At each step, iterate over sorted(G.successors(current_node)).
  3. If next_node not in visited AND path length < max_path_length:
     a. Append next_node to current_path, add to visited.
     b. If path length >= min_path_length: emit current_path as candidate.
     c. Recurse to step 2.
     d. Backtrack: pop next_node from visited and current_path.
  4. Stop when per-entry quota (max_paths_per_entry) or total quota
     (max_total_paths) is reached.

Determinism:
  - Successors are sorted lexicographically by node ID at every DFS step.
  - Given the same graph and PathConfig, the same candidates are always
    produced in the same order.
  - No random sampling is used during generation.

Leakage prevention:
  - The generator reads ONLY: graph topology, node attributes, edge attributes,
    GNN model predictions.
  - It does NOT read: HistoricalAttack.attack_path, data.y (ground truth labels).
  - This invariant is documented and tested.

=============================================================================
GNN PROBABILITY INTEGRATION
=============================================================================

The trained Phase 3 checkpoint is loaded ONCE at PathGenerator construction.
For each topology scenario:
  1. Build the PyG Data object (x, edge_index).
  2. Run forward pass → logits [N, 2].
  3. Apply softmax → P(involved) = probs[:, 1] for each node.
  4. Store as a node_id → probability dict.
  5. Attach per-node probabilities to each CandidateAttackPath.

Probabilities come from the ACTUAL MODEL — never hard-coded.
The model is NOT retrained during path generation.

=============================================================================
EDGE FEATURE PRESERVATION
=============================================================================

For each (u → v) hop in a candidate path:
  - Retrieve edge attributes from G[u][v] (dict stored by GraphBuilder).
  - Extract the 9 edge features in EDGE_FEATURE_DIM order.
  - Store as a list of floats on the CandidateAttackPath.

These are the SAME 9 edge features used in Phase 2/3 (EDGE_FEATURE_DIM = 9).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import torch
import torch.nn.functional as F

from backend.core.graph_builder import GraphBuilder
from backend.core.data_models import NetworkTopology
from backend.ml.dataset import (
    EDGE_FEATURE_DIM,
    EDGE_TYPE_CATEGORIES,
    EDGE_SCALAR_FEATURE_NAMES,
    build_edge_features,
    topology_to_pyg,
)
from backend.ml.model import AttackGNN
from backend.ml.path_models import (
    CandidateAttackPath,
    PathConfig,
    PathGenerationSummary,
    PathValidationResult,
    make_path_id,
)
from backend.ml.trainer import load_checkpoint
from backend.core.feature_extractor import FEATURE_DIM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge feature extraction helpers
# ---------------------------------------------------------------------------

# The order of edge features must match EXACTLY what was built in Phase 2.
# Phase 2 dataset.py: [onehot_type(4)] + [firewall, encrypted, atk_wt, anomaly, hist_atk]

_EDGE_FEATURE_ORDER = (
    [f"edgetype_{c}" for c in EDGE_TYPE_CATEGORIES]  # 4
    + list(EDGE_SCALAR_FEATURE_NAMES)                 # 5
)  # total = 9

assert len(_EDGE_FEATURE_ORDER) == EDGE_FEATURE_DIM, (
    f"Edge feature order mismatch: {len(_EDGE_FEATURE_ORDER)} vs {EDGE_FEATURE_DIM}"
)

_ETYPE_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(EDGE_TYPE_CATEGORIES)}


def _extract_edge_feature_vector(edge_data: dict) -> List[float]:
    """
    Extract a 9-dimensional edge feature vector from a NetworkX edge attribute dict.

    Matches the exact same encoding used in Phase 2 dataset.build_edge_features().

    Args:
        edge_data: dict of edge attributes from G[u][v].

    Returns:
        List[float] of length EDGE_FEATURE_DIM (9).
    """
    # One-hot edge type (4 dims)
    etype_vec = [0.0] * len(EDGE_TYPE_CATEGORIES)
    etype_raw = edge_data.get("edge_type", "network_link")
    etype_idx = _ETYPE_TO_IDX.get(etype_raw, 0)
    etype_vec[etype_idx] = 1.0

    # Scalar features (5 dims)
    scalar_vec = [
        float(edge_data.get("firewall_rule", 0)),
        float(edge_data.get("is_encrypted", 0)),
        float(edge_data.get("attack_weight", 0.5)),
        float(edge_data.get("anomaly_count", 0)),
        float(edge_data.get("historical_attack_count", 0)),
    ]

    return etype_vec + scalar_vec


# ---------------------------------------------------------------------------
# Path Validator
# ---------------------------------------------------------------------------

class PathValidator:
    """
    Validates a candidate attack path against the graph.

    Rules (from spec):
      1. Path contains at least 2 nodes (min_path_length edges ≥ 1).
      2. First node is a graph entry point.
      3. Every node exists in the graph.
      4. Every consecutive pair is a valid directed edge.
      5. No repeated nodes.
      6. No self-loops (adjacent identical nodes).
      7. Path length (edges) within configured bounds.
      8. path_length field matches actual len(node_ids) - 1.

    Invalid paths are REJECTED, not silently repaired.
    """

    def __init__(self, G: nx.DiGraph, config: PathConfig) -> None:
        self.G = G
        self.config = config
        self._entry_nodes: Set[str] = {
            n for n, d in G.nodes(data=True) if d.get("is_entry_point", 0) == 1
        }

    def validate(self, node_ids: List[str]) -> PathValidationResult:
        """
        Validate a candidate path.

        Args:
            node_ids: Ordered list of node IDs representing the path.

        Returns:
            PathValidationResult with valid=True/False and any error messages.
        """
        result = PathValidationResult()
        n = len(node_ids)

        # Rule 1: At least 2 nodes
        if n < 2:
            result.fail(
                f"Path has {n} node(s) — minimum is 2 (at least 1 edge required)."
            )
            return result  # Can't check further

        # Rule 2: First node is an entry point
        if node_ids[0] not in self._entry_nodes:
            result.fail(
                f"First node '{node_ids[0]}' is not an entry point."
            )

        # Rule 3: All nodes exist
        for nid in node_ids:
            if nid not in self.G.nodes:
                result.fail(f"Node '{nid}' does not exist in the graph.")

        # Rule 4: All consecutive pairs are valid directed edges
        for i in range(n - 1):
            u, v = node_ids[i], node_ids[i + 1]
            if u in self.G.nodes and v in self.G.nodes:
                if not self.G.has_edge(u, v):
                    result.fail(
                        f"No directed edge from '{u}' to '{v}' (hop {i}→{i+1})."
                    )

        # Rule 5: No repeated nodes
        if len(node_ids) != len(set(node_ids)):
            seen = set()
            for nid in node_ids:
                if nid in seen:
                    result.fail(f"Node '{nid}' appears more than once in the path.")
                    break
                seen.add(nid)

        # Rule 6: No self-loops (adjacent identical nodes)
        for i in range(n - 1):
            if node_ids[i] == node_ids[i + 1]:
                result.fail(
                    f"Self-loop at position {i}: node '{node_ids[i]}' → same node."
                )

        # Rule 7: Path length (edges) within bounds
        edge_count = n - 1
        if edge_count < self.config.min_path_length:
            result.fail(
                f"Path has {edge_count} edge(s); minimum is {self.config.min_path_length}."
            )
        if edge_count > self.config.max_path_length:
            result.fail(
                f"Path has {edge_count} edge(s); maximum is {self.config.max_path_length}."
            )

        return result


# ---------------------------------------------------------------------------
# GNN Probability Provider
# ---------------------------------------------------------------------------

class GNNProbabilityProvider:
    """
    Loads a trained Phase 3 checkpoint and provides per-node attack probabilities.

    This class:
    - Loads the model ONCE at construction.
    - For each topology, runs a single forward pass to get node probabilities.
    - Does NOT retrain the model.
    - Does NOT fabricate probabilities.

    Probabilities are P(node_i = attack_involved) = softmax(logits)[:, 1].
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[AttackGNN] = None
        self._checkpoint_info: dict = {}
        self._load()

    def _load(self) -> None:
        """Load checkpoint once."""
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"Phase 3 checkpoint not found: {self.checkpoint_path}\n"
                "Run run_phase3_training.py first."
            )
        self._model, self._checkpoint_info = load_checkpoint(
            self.checkpoint_path, device=self.device
        )
        self._model.eval()
        logger.info(
            "GNNProbabilityProvider: loaded checkpoint '%s' "
            "(best_epoch=%s, hidden=%s, layers=%s)",
            self.checkpoint_path,
            self._checkpoint_info.get("history", {}).get("best_epoch", "?"),
            self._checkpoint_info.get("config", {}).get("hidden_channels", "?"),
            self._checkpoint_info.get("config", {}).get("num_layers", "?"),
        )

    @torch.no_grad()
    def get_node_probabilities(
        self,
        topology: NetworkTopology,
    ) -> Dict[str, float]:
        """
        Compute P(attack_involved) for every node in the topology.

        Uses the trained model — NO retraining, NO fabrication.

        Args:
            topology: The source NetworkTopology.

        Returns:
            dict mapping node_id → float probability in [0.0, 1.0].
        """
        data, node_index, _ = topology_to_pyg(topology, normalize_features=True)
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        logits = self._model(x, edge_index)               # [N, 2]
        probs = F.softmax(logits, dim=-1)[:, 1]           # [N] P(attack_involved)

        prob_dict = {
            node_id: float(probs[i].item())
            for i, node_id in enumerate(node_index)
        }
        return prob_dict

    @property
    def checkpoint_path_str(self) -> str:
        return self.checkpoint_path


# ---------------------------------------------------------------------------
# Path Generator
# ---------------------------------------------------------------------------

class PathGenerator:
    """
    Generates candidate attack paths using bounded DFS.

    Usage:
        config  = PathConfig(max_path_length=5, max_paths_per_entry=30)
        gen     = PathGenerator(config, checkpoint_path="checkpoints/best_attack_gnn.pt")
        paths, summary = gen.generate(topology)

    The generator:
      - Never reads HistoricalAttack.attack_path.
      - Never reads data.y (ground-truth labels).
      - Reads only: graph topology, node/edge attributes, trained GNN predictions.
      - Is deterministic: same graph + config + checkpoint → same output.
    """

    def __init__(
        self,
        config: PathConfig,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            config:          Path generation configuration.
            checkpoint_path: Path to Phase 3 .pt checkpoint. If None, no
                             GNN probabilities are attached (probabilities set to -1.0).
            device:          Torch device.
        """
        self.config = config
        self.device = device or "cpu"
        self._builder = GraphBuilder()

        self._prob_provider: Optional[GNNProbabilityProvider] = None
        if checkpoint_path is not None:
            self._prob_provider = GNNProbabilityProvider(checkpoint_path, device=self.device)
            logger.info("PathGenerator: GNN checkpoint loaded from '%s'.", checkpoint_path)
        else:
            logger.warning(
                "PathGenerator: no checkpoint provided. "
                "Node probabilities will be None (not attached)."
            )

    def _dfs_enumerate(
        self,
        G: nx.DiGraph,
        current_node: str,
        visited: Set[str],
        current_path: List[str],
        results: List[List[str]],
        per_entry_count_ref: List[int],  # mutable int via list wrapper
        total_count_ref: List[int],      # mutable int via list wrapper
        truncated_ref: List[bool],
    ) -> None:
        """
        Depth-first search enumerating simple paths.

        Modifies results in-place.
        Uses mutable wrappers for per-entry and total counts.
        Stops when quotas are exceeded.
        Neighbors are sorted for determinism.
        """
        current_depth = len(current_path) - 1  # number of edges so far

        # Emit if we've reached minimum depth
        if current_depth >= self.config.min_path_length:
            results.append(list(current_path))
            per_entry_count_ref[0] += 1
            total_count_ref[0] += 1

        # Check quotas BEFORE recursing
        if per_entry_count_ref[0] >= self.config.max_paths_per_entry:
            truncated_ref[0] = True
            return
        if total_count_ref[0] >= self.config.max_total_paths:
            truncated_ref[0] = True
            return

        # Stop recursion if we've hit max depth
        if current_depth >= self.config.max_path_length:
            return

        # Sort successors for deterministic ordering
        for neighbor in sorted(G.successors(current_node)):
            if neighbor in visited:
                continue  # No repeated nodes
            visited.add(neighbor)
            current_path.append(neighbor)

            self._dfs_enumerate(
                G, neighbor, visited, current_path,
                results, per_entry_count_ref, total_count_ref, truncated_ref,
            )

            current_path.pop()
            visited.remove(neighbor)

            # Re-check quotas after each recursive return
            if per_entry_count_ref[0] >= self.config.max_paths_per_entry:
                truncated_ref[0] = True
                return
            if total_count_ref[0] >= self.config.max_total_paths:
                truncated_ref[0] = True
                return

    def _build_candidate(
        self,
        G: nx.DiGraph,
        node_ids: List[str],
        prob_dict: Optional[Dict[str, float]],
        topology_id: str,
        is_synthetic: bool,
    ) -> CandidateAttackPath:
        """
        Construct a CandidateAttackPath from a raw path (list of node IDs).

        Attaches:
        - node labels (from graph attributes)
        - GNN probabilities (from prob_dict)
        - edge feature vectors (from graph edge attributes)
        - structural metadata
        """
        n = len(node_ids)
        path_length = n - 1

        # Node labels
        node_labels = [
            G.nodes[nid].get("label", nid) for nid in node_ids
        ]

        # GNN probabilities
        if prob_dict is not None:
            node_probs = [float(prob_dict.get(nid, 0.5)) for nid in node_ids]
        else:
            node_probs = [-1.0] * n  # sentinel: no model loaded

        # Edge pairs and features
        edge_pairs: List[Tuple[str, str]] = []
        edge_feats: List[List[float]] = []
        for i in range(path_length):
            u, v = node_ids[i], node_ids[i + 1]
            edge_pairs.append((u, v))
            if G.has_edge(u, v):
                edge_feats.append(_extract_edge_feature_vector(G[u][v]))
            else:
                edge_feats.append([0.0] * EDGE_FEATURE_DIM)

        # Terminal is critical asset?
        terminal_attrs = G.nodes[node_ids[-1]]
        reaches_critical = bool(terminal_attrs.get("is_critical_asset", 0))

        # Structural metadata (features for Phase 5, NOT a risk score)
        n_critical_nodes = sum(
            1 for nid in node_ids if G.nodes[nid].get("is_critical_asset", 0)
        )
        n_vulnerable_nodes = sum(
            1 for nid in node_ids if G.nodes[nid].get("vuln_count", 0) > 0
        )
        mean_cvss = (
            sum(G.nodes[nid].get("max_cvss", 0.0) for nid in node_ids) / n
        )
        mean_attack_weight = (
            sum(ef[6] for ef in edge_feats) / len(edge_feats)
            if edge_feats else 0.0
        )
        valid_probs = [p for p in node_probs if p >= 0.0]
        structural_metadata = {
            "path_length_edges": path_length,
            "n_nodes": n,
            "n_critical_nodes": n_critical_nodes,
            "n_vulnerable_nodes": n_vulnerable_nodes,
            "mean_max_cvss": round(mean_cvss, 4),
            "mean_edge_attack_weight": round(mean_attack_weight, 4),
            "mean_node_attack_prob": round(sum(valid_probs) / len(valid_probs), 6)
                                     if valid_probs else None,
            "min_node_attack_prob": round(min(valid_probs), 6)
                                    if valid_probs else None,
            "max_node_attack_prob": round(max(valid_probs), 6)
                                    if valid_probs else None,
        }

        return CandidateAttackPath(
            path_id=make_path_id(),
            node_ids=node_ids,
            edge_pairs=edge_pairs,
            node_labels=node_labels,
            node_attack_probabilities=node_probs,
            edge_features=edge_feats,
            path_length=path_length,
            entry_node=node_ids[0],
            terminal_node=node_ids[-1],
            reaches_critical_asset=reaches_critical,
            is_valid=True,
            topology_id=topology_id,
            is_synthetic=is_synthetic,
            structural_metadata=structural_metadata,
        )

    def _sort_candidates(
        self,
        candidates: List[CandidateAttackPath],
    ) -> List[CandidateAttackPath]:
        """Sort candidates deterministically by config.sort_by key."""
        if self.config.sort_by == "path_str":
            return sorted(candidates, key=lambda p: p.path_key)
        elif self.config.sort_by == "length":
            return sorted(candidates, key=lambda p: (p.path_length, p.path_key))
        elif self.config.sort_by == "mean_prob":
            return sorted(
                candidates,
                key=lambda p: (-p.mean_attack_probability, p.path_key),
            )
        return candidates

    def generate(
        self,
        topology: NetworkTopology,
    ) -> Tuple[List[CandidateAttackPath], PathGenerationSummary]:
        """
        Generate candidate attack paths for one topology.

        IMPORTANT: Does NOT read HistoricalAttack.attack_path.
                   Does NOT read ground-truth labels (data.y).

        Args:
            topology: A NetworkTopology object.

        Returns:
            (candidates, summary)
            - candidates: list of CandidateAttackPath objects.
            - summary:    PathGenerationSummary with stats and truncation info.
        """
        summary = PathGenerationSummary(
            topology_id=topology.topology_id,
            topology_name=topology.name,
            is_synthetic=topology.is_synthetic,
            config=self.config.to_dict(),
            used_historical_labels=False,  # Always False — guaranteed by design
        )

        # Check for empty topology
        if not topology.nodes:
            logger.warning(
                "Topology '%s' has no nodes — returning empty candidate set.",
                topology.name,
            )
            summary.graph_node_count = 0
            summary.graph_edge_count = 0
            summary.entry_point_count = 0
            summary.critical_asset_count = 0
            summary.truncation_reason = "Empty topology (no nodes)."
            return [], summary

        # Build the directed graph (existing Phase 1 logic, not duplicated)
        G = self._builder.build(topology)
        summary.graph_node_count = G.number_of_nodes()
        summary.graph_edge_count = G.number_of_edges()

        # Get entry points from graph attributes (NOT from HistoricalAttack)
        entry_nodes = sorted([
            n for n, d in G.nodes(data=True) if d.get("is_entry_point", 0) == 1
        ])
        critical_nodes = set(
            n for n, d in G.nodes(data=True) if d.get("is_critical_asset", 0) == 1
        )
        summary.entry_point_count = len(entry_nodes)
        summary.critical_asset_count = len(critical_nodes)

        if not entry_nodes:
            logger.warning(
                "Topology '%s' has no entry points — returning empty candidate set.",
                topology.name,
            )
            summary.truncation_reason = "No entry points in topology."
            return [], summary

        # Get GNN probabilities (from trained model — NOT from labels)
        prob_dict: Optional[Dict[str, float]] = None
        if self._prob_provider is not None:
            try:
                prob_dict = self._prob_provider.get_node_probabilities(topology)
                summary.gnn_probabilities_attached = True
                summary.gnn_checkpoint_used = self._prob_provider.checkpoint_path_str
            except Exception as e:
                logger.error("GNN probability computation failed: %s", e)
                summary.gnn_probabilities_attached = False

        summary.edge_features_attached = True  # always extracted from graph

        # Enumerate candidate paths via bounded DFS
        all_raw_paths: List[List[str]] = []
        total_count_ref = [0]      # mutable int wrapper
        any_truncated = [False]    # mutable bool wrapper
        truncation_reasons = []

        for entry in entry_nodes:
            per_entry_results: List[List[str]] = []
            entry_truncated = [False]
            per_entry_count_ref = [0]

            self._dfs_enumerate(
                G=G,
                current_node=entry,
                visited={entry},
                current_path=[entry],
                results=per_entry_results,
                per_entry_count_ref=per_entry_count_ref,
                total_count_ref=total_count_ref,
                truncated_ref=entry_truncated,
            )

            if entry_truncated[0]:
                any_truncated[0] = True
                entry_label = G.nodes[entry].get("label", entry)
                truncation_reasons.append(
                    f"Entry '{entry_label}' hit quota "
                    f"(max_paths_per_entry={self.config.max_paths_per_entry} "
                    f"or max_total_paths={self.config.max_total_paths})."
                )

            all_raw_paths.extend(per_entry_results)

            if total_count_ref[0] >= self.config.max_total_paths:
                any_truncated[0] = True
                truncation_reasons.append(
                    f"Global max_total_paths={self.config.max_total_paths} reached."
                )
                break

        summary.generation_truncated = any_truncated[0]
        summary.truncation_reason = " | ".join(truncation_reasons) if truncation_reasons else ""

        # Apply critical-terminal filter (after generation, not during)
        if self.config.require_critical_terminal:
            all_raw_paths = [p for p in all_raw_paths if p[-1] in critical_nodes]
        elif not self.config.allow_noncritical_terminal:
            all_raw_paths = [p for p in all_raw_paths if p[-1] in critical_nodes]

        # Deduplicate by canonical path key
        seen_keys: Set[str] = set()
        unique_raw: List[List[str]] = []
        for p in all_raw_paths:
            key = " → ".join(p)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_raw.append(p)

        # Build CandidateAttackPath objects
        candidates: List[CandidateAttackPath] = [
            self._build_candidate(
                G, raw_path, prob_dict,
                topology.topology_id, topology.is_synthetic,
            )
            for raw_path in unique_raw
        ]

        # Deterministic sort
        candidates = self._sort_candidates(candidates)

        # Update summary counts
        summary.total_candidates = len(candidates)
        summary.critical_terminal_count = sum(
            1 for c in candidates if c.reaches_critical_asset
        )
        summary.noncritical_terminal_count = (
            summary.total_candidates - summary.critical_terminal_count
        )

        logger.info(
            "Topology '%s': generated %d candidate paths "
            "(%d reach critical assets, truncated=%s).",
            topology.name, summary.total_candidates,
            summary.critical_terminal_count, summary.generation_truncated,
        )

        return candidates, summary

    def generate_for_dataset(
        self,
        topologies: List[NetworkTopology],
        split_name: str = "all",
    ) -> Tuple[List[CandidateAttackPath], List[PathGenerationSummary]]:
        """
        Generate candidates for a list of topology scenarios (one split).

        Maintains train/val/test integrity: this function is called separately
        for each split. It does not have access to other splits' data.

        Args:
            topologies:  List of NetworkTopology objects for one split.
            split_name:  Label for logging ("train", "val", "test").

        Returns:
            (all_candidates, all_summaries)
        """
        all_candidates: List[CandidateAttackPath] = []
        all_summaries: List[PathGenerationSummary] = []

        logger.info(
            "Generating candidates for '%s' split: %d topologies.",
            split_name, len(topologies),
        )

        for topo in topologies:
            paths, summary = self.generate(topo)
            all_candidates.extend(paths)
            all_summaries.append(summary)

        total = len(all_candidates)
        critical = sum(1 for c in all_candidates if c.reaches_critical_asset)
        truncated = sum(1 for s in all_summaries if s.generation_truncated)

        logger.info(
            "Split '%s': %d total candidates, %d reach critical assets, "
            "%d truncated scenarios.",
            split_name, total, critical, truncated,
        )

        return all_candidates, all_summaries
