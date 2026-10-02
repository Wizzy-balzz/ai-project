"""
feature_extractor.py
--------------------
Extracts and normalizes graph-level and node-level features from a NetworkX DiGraph.

Design Decisions:
- Features are split into three tiers:
    1. Node-level raw features (from graph node attributes)
    2. Node-level structural features (from graph topology: degree, centrality)
    3. Graph-level summary statistics
- Normalization uses sklearn's MinMaxScaler/StandardScaler.
  Scalers are fitted on the full graph so that node features are comparable.
- Feature names are tracked alongside the numeric arrays for interpretability
  and downstream explainability (Phase 5+).
- NumPy arrays are returned for direct use in PyTorch tensor conversion (Phase 2).
- All intermediate steps are documented so that GNN researchers can reproduce
  or modify the feature engineering pipeline.

Feature Vector Structure (per node):
  [raw node attributes] + [structural graph features]

Where raw node attributes include:
  - Vulnerability counts and scores
  - Node type one-hot encoding
  - OS type one-hot encoding
  - Service flags
  - Entry point / critical asset flags

Structural features include:
  - In-degree, out-degree
  - PageRank
  - Betweenness centrality (approximate for large graphs)
  - Closeness centrality
  - Clustering coefficient (undirected projection)
  - Attack-weighted in-degree (sum of attack_weight of incoming edges)
  - Anomaly signal (sum of anomaly_count of incoming edges)
  - Historical attack exposure (sum of historical_attack_count of incoming edges)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants: Feature Names
# ---------------------------------------------------------------------------

# Node type one-hot categories (must match NodeType enum order for reproducibility)
NODE_TYPE_CATEGORIES = [
    "workstation", "server", "database", "router",
    "firewall", "internet", "iot_device", "critical_asset",
]

# OS type one-hot categories
OS_TYPE_CATEGORIES = [
    "windows", "linux", "macos", "firmware", "unknown",
]

RAW_SCALAR_FEATURE_NAMES = [
    "is_entry_point",
    "is_critical_asset",
    "asset_value",
    "vuln_count",
    "max_cvss",
    "has_critical_vuln",
    "has_exploited_in_wild",
    "critical_vuln_count",
    "high_vuln_count",
    "unpatched_vuln_count",
    "exploited_in_wild_count",
    "avg_cvss",
    "avg_exploitability",
    "service_count",
    "has_rdp",
    "has_ssh",
    "has_smb",
    "has_http",
    "has_https",
    "compromised",
]

NODE_TYPE_FEATURE_NAMES = [f"nodetype_{c}" for c in NODE_TYPE_CATEGORIES]
OS_TYPE_FEATURE_NAMES = [f"ostype_{c}" for c in OS_TYPE_CATEGORIES]

STRUCTURAL_FEATURE_NAMES = [
    "in_degree",
    "out_degree",
    "total_degree",
    "pagerank",
    "betweenness_centrality",
    "closeness_centrality",
    "clustering_coeff",
    "attack_weighted_in_degree",
    "anomaly_signal",
    "historical_attack_exposure",
]

ALL_FEATURE_NAMES = (
    RAW_SCALAR_FEATURE_NAMES
    + NODE_TYPE_FEATURE_NAMES
    + OS_TYPE_FEATURE_NAMES
    + STRUCTURAL_FEATURE_NAMES
)

FEATURE_DIM = len(ALL_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Extracts, encodes, and normalizes node feature vectors from a NetworkX DiGraph.

    Usage:
        extractor = FeatureExtractor()
        features, node_index = extractor.extract(G)
        # features: np.ndarray of shape (N, FEATURE_DIM)
        # node_index: list of node IDs in row order

    The extractor also exposes:
        extractor.feature_names  -> list of feature names
        extractor.scaler         -> fitted MinMaxScaler (for inverse transform)
        extractor.raw_features   -> unscaled feature matrix (before normalization)
    """

    def __init__(self, normalize: bool = True, pagerank_alpha: float = 0.85) -> None:
        self.normalize = normalize
        self.pagerank_alpha = pagerank_alpha
        self.scaler: Optional[MinMaxScaler] = None
        self.feature_names: List[str] = ALL_FEATURE_NAMES
        self.raw_features: Optional[np.ndarray] = None
        self.node_index: Optional[List[str]] = None

    # -----------------------------------------------------------------------
    # Step 1: Raw Scalar Features
    # -----------------------------------------------------------------------

    def _extract_raw_scalar_features(self, G: nx.DiGraph, node_order: List[str]) -> np.ndarray:
        """Extract scalar attributes from graph node data."""
        n = len(node_order)
        dim = len(RAW_SCALAR_FEATURE_NAMES)
        matrix = np.zeros((n, dim), dtype=np.float32)

        for i, node_id in enumerate(node_order):
            d = G.nodes[node_id]
            for j, fname in enumerate(RAW_SCALAR_FEATURE_NAMES):
                matrix[i, j] = float(d.get(fname, 0.0))

        return matrix

    # -----------------------------------------------------------------------
    # Step 2: One-Hot Encodings
    # -----------------------------------------------------------------------

    def _encode_node_types(self, G: nx.DiGraph, node_order: List[str]) -> np.ndarray:
        """One-hot encode node type."""
        n = len(node_order)
        dim = len(NODE_TYPE_CATEGORIES)
        matrix = np.zeros((n, dim), dtype=np.float32)
        cat_to_idx = {c: i for i, c in enumerate(NODE_TYPE_CATEGORIES)}
        for i, node_id in enumerate(node_order):
            ntype = G.nodes[node_id].get("node_type", "workstation")
            idx = cat_to_idx.get(ntype, 0)
            matrix[i, idx] = 1.0
        return matrix

    def _encode_os_types(self, G: nx.DiGraph, node_order: List[str]) -> np.ndarray:
        """One-hot encode OS type."""
        n = len(node_order)
        dim = len(OS_TYPE_CATEGORIES)
        matrix = np.zeros((n, dim), dtype=np.float32)
        cat_to_idx = {c: i for i, c in enumerate(OS_TYPE_CATEGORIES)}
        for i, node_id in enumerate(node_order):
            ostype = G.nodes[node_id].get("os_type", "unknown")
            idx = cat_to_idx.get(ostype, cat_to_idx["unknown"])
            matrix[i, idx] = 1.0
        return matrix

    # -----------------------------------------------------------------------
    # Step 3: Structural Features
    # -----------------------------------------------------------------------

    def _extract_structural_features(self, G: nx.DiGraph, node_order: List[str]) -> np.ndarray:
        """
        Compute graph-structure-based features for each node.

        These features capture each node's topological role in the network,
        which is critical for the GNN to learn attack propagation patterns.
        """
        n = len(node_order)
        dim = len(STRUCTURAL_FEATURE_NAMES)
        matrix = np.zeros((n, dim), dtype=np.float32)

        # Degree features
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())

        # PageRank — measures how "central" a node is to network flow
        try:
            pagerank = nx.pagerank(G, alpha=self.pagerank_alpha, weight="attack_weight")
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank failed to converge; using uniform values.")
            pagerank = {n: 1.0 / len(G.nodes) for n in G.nodes}

        # Betweenness centrality — nodes on many shortest paths are attack bottlenecks
        # Using approximate version (k samples) for scalability on large graphs
        k_approx = min(len(G.nodes), max(10, len(G.nodes) // 5))
        try:
            betweenness = nx.betweenness_centrality(G, k=k_approx, normalized=True, seed=42)
        except Exception as e:
            logger.warning("Betweenness centrality failed: %s — using zeros.", e)
            betweenness = {node: 0.0 for node in G.nodes}

        # Closeness centrality — how easily a node can reach others
        try:
            closeness = nx.closeness_centrality(G)
        except Exception as e:
            logger.warning("Closeness centrality failed: %s — using zeros.", e)
            closeness = {node: 0.0 for node in G.nodes}

        # Clustering coefficient (requires undirected projection)
        try:
            G_undirected = G.to_undirected()
            clustering = nx.clustering(G_undirected)
        except Exception as e:
            logger.warning("Clustering coefficient failed: %s — using zeros.", e)
            clustering = {node: 0.0 for node in G.nodes}

        for i, node_id in enumerate(node_order):
            # Attack-weighted in-degree: sum of attack_weight of all incoming edges
            attack_w_in = sum(
                G[pred][node_id].get("attack_weight", 0.0)
                for pred in G.predecessors(node_id)
            )

            # Anomaly signal: total anomaly count from incoming edges
            anomaly_sig = sum(
                G[pred][node_id].get("anomaly_count", 0)
                for pred in G.predecessors(node_id)
            )

            # Historical attack exposure
            historical_exp = sum(
                G[pred][node_id].get("historical_attack_count", 0)
                for pred in G.predecessors(node_id)
            )

            matrix[i, 0] = float(in_deg.get(node_id, 0))
            matrix[i, 1] = float(out_deg.get(node_id, 0))
            matrix[i, 2] = float(in_deg.get(node_id, 0) + out_deg.get(node_id, 0))
            matrix[i, 3] = float(pagerank.get(node_id, 0.0))
            matrix[i, 4] = float(betweenness.get(node_id, 0.0))
            matrix[i, 5] = float(closeness.get(node_id, 0.0))
            matrix[i, 6] = float(clustering.get(node_id, 0.0))
            matrix[i, 7] = float(attack_w_in)
            matrix[i, 8] = float(anomaly_sig)
            matrix[i, 9] = float(historical_exp)

        return matrix

    # -----------------------------------------------------------------------
    # Step 4: Normalization
    # -----------------------------------------------------------------------

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply MinMax normalization to scale all features to [0, 1].

        One-hot encoded columns are already in [0, 1], but normalizing them
        is harmless (they remain unchanged if all values are 0 or 1).
        """
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        normalized = self.scaler.fit_transform(features)
        return normalized.astype(np.float32)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def extract(self, G: nx.DiGraph) -> Tuple[np.ndarray, List[str]]:
        """
        Extract the full feature matrix from a graph.

        Args:
            G: NetworkX DiGraph built by GraphBuilder.

        Returns:
            Tuple of:
                features:   np.ndarray of shape (N, FEATURE_DIM)
                node_index: list of node IDs in row order (row i -> node_index[i])
        """
        if G.number_of_nodes() == 0:
            raise ValueError("Cannot extract features from an empty graph.")

        node_order = list(G.nodes())
        self.node_index = node_order

        logger.info("Extracting features for %d nodes...", len(node_order))

        # Build feature matrix by concatenating all feature blocks
        raw_scalar = self._extract_raw_scalar_features(G, node_order)
        node_type_enc = self._encode_node_types(G, node_order)
        os_type_enc = self._encode_os_types(G, node_order)
        structural = self._extract_structural_features(G, node_order)

        features = np.concatenate(
            [raw_scalar, node_type_enc, os_type_enc, structural], axis=1
        )
        self.raw_features = features.copy()

        if self.normalize:
            features = self._normalize(features)

        logger.info(
            "Feature matrix shape: %s | Feature dim: %d",
            features.shape, FEATURE_DIM,
        )
        return features, node_order

    def get_feature_summary(self, G: nx.DiGraph) -> Dict:
        """
        Return a human-readable summary of node features for a graph.

        Useful for debugging and the explainability phase.
        Must be called after extract().
        """
        if self.raw_features is None or self.node_index is None:
            raise RuntimeError("Call extract() before get_feature_summary().")

        summary = {}
        for i, node_id in enumerate(self.node_index):
            label = G.nodes[node_id].get("label", node_id)
            raw_row = self.raw_features[i]
            summary[node_id] = {
                "label": label,
                "features": {
                    name: round(float(raw_row[j]), 4)
                    for j, name in enumerate(self.feature_names)
                },
            }
        return summary

    def get_graph_statistics(self, G: nx.DiGraph) -> Dict:
        """
        Compute graph-level summary statistics.

        These are not used directly as GNN input but are valuable for
        dashboard reporting and anomaly detection baseline comparisons.
        """
        entry_nodes = [n for n, d in G.nodes(data=True) if d.get("is_entry_point", 0) == 1]
        critical_nodes = [n for n, d in G.nodes(data=True) if d.get("is_critical_asset", 0) == 1]

        vuln_counts = [d.get("vuln_count", 0) for _, d in G.nodes(data=True)]
        cvss_scores = [d.get("max_cvss", 0.0) for _, d in G.nodes(data=True)]
        attack_weights = [
            d.get("attack_weight", 0.0)
            for _, _, d in G.edges(data=True)
        ]
        anomaly_counts = [
            d.get("anomaly_count", 0)
            for _, _, d in G.edges(data=True)
        ]

        density = nx.density(G)
        try:
            avg_shortest = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            avg_shortest = None  # Graph may not be strongly connected

        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "entry_point_count": len(entry_nodes),
            "critical_asset_count": len(critical_nodes),
            "graph_density": round(density, 6),
            "avg_shortest_path_length": (
                round(avg_shortest, 4) if avg_shortest is not None else "N/A (disconnected)"
            ),
            "total_vulnerability_count": int(sum(vuln_counts)),
            "avg_vuln_per_node": round(float(np.mean(vuln_counts)), 3),
            "max_cvss_in_graph": round(float(max(cvss_scores)), 2),
            "avg_cvss_in_graph": round(float(np.mean(cvss_scores)), 3),
            "avg_edge_attack_weight": round(float(np.mean(attack_weights)), 4) if attack_weights else 0.0,
            "total_anomaly_events": int(sum(anomaly_counts)),
        }
