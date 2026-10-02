"""
graph_builder.py
----------------
Converts a NetworkTopology into a NetworkX DiGraph for downstream processing.

Design Decisions:
- DiGraph (directed) is used because attack traversal is asymmetric.
  An attacker moving from node A to node B does not imply the reverse is valid.
- Node attributes are stored directly on the graph for easy feature extraction.
- Bidirectional edges in NetworkEdge are expanded into two directed edges.
- The builder is stateless: given the same topology, it produces the same graph.
- Telemetry signals are aggregated per (source, target) pair and stored as
  edge attributes (anomaly_count, total_bytes, packet_count).
- All NetworkX attributes use primitive Python types (int, float, str, bool)
  to ensure JSON serialization compatibility for the API in later phases.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import networkx as nx

from backend.core.data_models import (
    EdgeType,
    NetworkEdge,
    NetworkNode,
    NetworkTopology,
    NodeType,
    OSType,
    VulnerabilitySeverity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry Aggregation
# ---------------------------------------------------------------------------

def _aggregate_telemetry(topology: NetworkTopology) -> Dict[Tuple[str, str], Dict]:
    """
    Aggregate telemetry records per directed (source_id, target_id) pair.

    Returns a dict: (source_id, target_id) -> {
        "anomaly_count": int,
        "total_bytes": int,
        "total_packets": int,
        "record_count": int,
        "alert_types": list[str],
    }
    """
    agg: Dict[Tuple[str, str], Dict] = {}
    for rec in topology.telemetry:
        key = (rec.source_id, rec.target_id)
        if key not in agg:
            agg[key] = {
                "anomaly_count": 0,
                "total_bytes": 0,
                "total_packets": 0,
                "record_count": 0,
                "alert_types": [],
            }
        agg[key]["record_count"] += 1
        agg[key]["total_bytes"] += rec.bytes_sent
        agg[key]["total_packets"] += rec.packet_count
        if rec.is_anomalous:
            agg[key]["anomaly_count"] += 1
            if rec.alert_type:
                agg[key]["alert_types"].append(rec.alert_type)
    return agg


def _aggregate_historical_attacks(topology: NetworkTopology) -> Dict[Tuple[str, str], int]:
    """
    Count how many historical attacks traversed each directed edge.

    Returns a dict: (source_id, target_id) -> attack_traversal_count
    """
    counts: Dict[Tuple[str, str], int] = {}
    for attack in topology.historical_attacks:
        path = attack.attack_path
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Node Attribute Builder
# ---------------------------------------------------------------------------

def _build_node_attributes(node: NetworkNode) -> Dict:
    """
    Extract serializable attributes from a NetworkNode for the graph.

    These attributes serve as raw input features for the GNN in Phase 2.
    All values must be Python primitives (int, float, str, bool).
    """
    return {
        # Identity
        "label": node.label,
        "node_type": node.node_type.value,
        "os_type": node.os_type.value,
        "ip_address": node.ip_address,
        "subnet": node.subnet,
        # Structural role
        "is_entry_point": int(node.is_entry_point),
        "is_critical_asset": int(node.is_critical_asset),
        "asset_value": float(node.asset_value),
        # Vulnerability features
        "vuln_count": len(node.vulnerabilities),
        "max_cvss": float(node.max_cvss),
        "has_critical_vuln": int(node.has_critical_vuln),
        "has_exploited_in_wild": int(node.has_exploited_in_wild),
        "critical_vuln_count": sum(
            1 for v in node.vulnerabilities
            if v.severity == VulnerabilitySeverity.CRITICAL
        ),
        "high_vuln_count": sum(
            1 for v in node.vulnerabilities
            if v.severity == VulnerabilitySeverity.HIGH
        ),
        "unpatched_vuln_count": sum(
            1 for v in node.vulnerabilities if not v.patch_available
        ),
        "exploited_in_wild_count": sum(
            1 for v in node.vulnerabilities if v.is_exploited_in_wild
        ),
        "avg_cvss": (
            sum(v.cvss_score for v in node.vulnerabilities) / len(node.vulnerabilities)
            if node.vulnerabilities else 0.0
        ),
        "avg_exploitability": (
            sum(v.exploitability_score for v in node.vulnerabilities) / len(node.vulnerabilities)
            if node.vulnerabilities else 0.0
        ),
        # Services
        "service_count": len(node.services),
        "has_rdp": int(any("rdp" in s for s in node.services)),
        "has_ssh": int(any("ssh" in s for s in node.services)),
        "has_smb": int(any("smb" in s for s in node.services)),
        "has_http": int(any("http" in s and "https" not in s for s in node.services)),
        "has_https": int(any("https" in s for s in node.services)),
        # Simulation state
        "compromised": int(node.compromised),
    }


# ---------------------------------------------------------------------------
# Edge Attribute Builder
# ---------------------------------------------------------------------------

def _build_edge_attributes(
    edge: NetworkEdge,
    telemetry_agg: Dict[Tuple[str, str], Dict],
    attack_counts: Dict[Tuple[str, str], int],
) -> Dict:
    """
    Extract serializable attributes from a NetworkEdge for the graph.

    Telemetry and historical attack signals are merged here.
    """
    key = (edge.source_id, edge.target_id)
    tel = telemetry_agg.get(key, {})

    return {
        "edge_type": edge.edge_type.value,
        "protocol": edge.protocol,
        "port": edge.port if edge.port is not None else -1,
        "firewall_rule": int(edge.firewall_rule),
        "is_encrypted": int(edge.is_encrypted),
        "bandwidth_mbps": float(edge.bandwidth_mbps),
        "latency_ms": float(edge.latency_ms),
        # Baseline attack weight (pre-GNN; updated by GNN in Phase 2+)
        "attack_weight": float(edge.attack_weight),
        # Telemetry signals
        "anomaly_count": int(tel.get("anomaly_count", 0)),
        "total_bytes": int(tel.get("total_bytes", 0)),
        "total_packets": int(tel.get("total_packets", 0)),
        "telemetry_record_count": int(tel.get("record_count", 0)),
        # Historical attack signal
        "historical_attack_count": int(attack_counts.get(key, 0)),
    }


# ---------------------------------------------------------------------------
# Main Graph Builder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Converts a NetworkTopology into a NetworkX DiGraph.

    Usage:
        builder = GraphBuilder()
        G = builder.build(topology)

    The resulting graph has:
    - Nodes with attributes from _build_node_attributes()
    - Directed edges with attributes from _build_edge_attributes()
    - Bidirectional NetworkEdge entries expanded to two directed edges
    - Graph-level metadata stored in G.graph

    The graph is reproducible: given the same topology input,
    the same graph is produced.
    """

    def build(self, topology: NetworkTopology) -> nx.DiGraph:
        """
        Build and return a NetworkX DiGraph from the given topology.

        Args:
            topology: A fully populated NetworkTopology object.

        Returns:
            nx.DiGraph with node and edge attributes ready for GNN input.
        """
        if not topology.nodes:
            raise ValueError("Cannot build graph from an empty topology (no nodes).")

        G = nx.DiGraph()

        # Store topology-level metadata on the graph object
        G.graph["topology_id"] = topology.topology_id
        G.graph["topology_name"] = topology.name
        G.graph["is_synthetic"] = topology.is_synthetic
        G.graph["description"] = topology.description

        # Pre-aggregate telemetry and attack history
        telemetry_agg = _aggregate_telemetry(topology)
        attack_counts = _aggregate_historical_attacks(topology)

        # --- Add Nodes ---
        for node_id, node in topology.nodes.items():
            attrs = _build_node_attributes(node)
            G.add_node(node_id, **attrs)
            logger.debug("Added node %s (%s)", node_id, node.label)

        # --- Add Edges ---
        edge_count_added = 0
        for edge_id, edge in topology.edges.items():
            # Validate that both endpoints exist
            if edge.source_id not in G.nodes:
                logger.warning(
                    "Edge %s references unknown source node %s — skipping.",
                    edge_id, edge.source_id,
                )
                continue
            if edge.target_id not in G.nodes:
                logger.warning(
                    "Edge %s references unknown target node %s — skipping.",
                    edge_id, edge.target_id,
                )
                continue

            attrs = _build_edge_attributes(edge, telemetry_agg, attack_counts)

            # Primary directed edge
            G.add_edge(edge.source_id, edge.target_id, edge_id=edge_id, **attrs)
            edge_count_added += 1

            # Reverse edge for bidirectional links
            if edge.bidirectional:
                reverse_key = (edge.target_id, edge.source_id)
                reverse_tel = telemetry_agg.get(reverse_key, {})
                reverse_attacks = attack_counts.get(reverse_key, 0)
                reverse_attrs = {**attrs}
                reverse_attrs["anomaly_count"] = int(reverse_tel.get("anomaly_count", 0))
                reverse_attrs["total_bytes"] = int(reverse_tel.get("total_bytes", 0))
                reverse_attrs["total_packets"] = int(reverse_tel.get("total_packets", 0))
                reverse_attrs["telemetry_record_count"] = int(reverse_tel.get("record_count", 0))
                reverse_attrs["historical_attack_count"] = int(reverse_attacks)
                G.add_edge(
                    edge.target_id, edge.source_id,
                    edge_id=f"{edge_id}_rev",
                    **reverse_attrs,
                )
                edge_count_added += 1

        logger.info(
            "Built graph for topology '%s': %d nodes, %d directed edges.",
            topology.name, G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    def get_entry_nodes(self, G: nx.DiGraph) -> list:
        """Return node IDs where is_entry_point == 1."""
        return [n for n, d in G.nodes(data=True) if d.get("is_entry_point", 0) == 1]

    def get_critical_nodes(self, G: nx.DiGraph) -> list:
        """Return node IDs where is_critical_asset == 1."""
        return [n for n, d in G.nodes(data=True) if d.get("is_critical_asset", 0) == 1]

    def get_subgraph_between(
        self,
        G: nx.DiGraph,
        source_nodes: list,
        target_nodes: list,
    ) -> nx.DiGraph:
        """
        Return a subgraph containing all simple paths from any source to any target.

        Useful for limiting GNN scope to the attack-relevant subgraph.
        Note: This can be expensive for large graphs with many paths.
        """
        relevant_nodes = set(source_nodes) | set(target_nodes)
        for src in source_nodes:
            for tgt in target_nodes:
                if nx.has_path(G, src, tgt):
                    for path in nx.all_simple_paths(G, src, tgt, cutoff=6):
                        relevant_nodes.update(path)
        return G.subgraph(relevant_nodes).copy()

    def validate_graph(self, G: nx.DiGraph) -> Dict:
        """
        Run basic validation checks on the built graph.

        Returns a dict of validation results (all should pass for a well-formed graph).
        """
        results = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "has_entry_points": len(self.get_entry_nodes(G)) > 0,
            "has_critical_assets": len(self.get_critical_nodes(G)) > 0,
            "is_weakly_connected": nx.is_weakly_connected(G),
            "has_self_loops": bool(list(nx.selfloop_edges(G))),
            "entry_point_count": len(self.get_entry_nodes(G)),
            "critical_asset_count": len(self.get_critical_nodes(G)),
        }

        # Check reachability from entry points to critical assets
        entry_nodes = self.get_entry_nodes(G)
        critical_nodes = self.get_critical_nodes(G)
        reachability_pairs = []
        for src in entry_nodes:
            for tgt in critical_nodes:
                reachable = nx.has_path(G, src, tgt)
                reachability_pairs.append({
                    "from": G.nodes[src].get("label", src),
                    "to": G.nodes[tgt].get("label", tgt),
                    "reachable": reachable,
                })
        results["reachability"] = reachability_pairs
        results["any_path_exists"] = any(p["reachable"] for p in reachability_pairs)

        return results
