"""
data_models.py
--------------
Core data models for the Graph-Based Attack Path Predictor.

Design Decisions:
- Dataclasses are used for lightweight, type-annotated value objects.
- Enums enforce valid node types and vulnerability severities.
- CVSS scores use the standard 0.0–10.0 range (CVSS v3).
- All models are serializable to dicts for downstream use (graph construction, API).
- No database ORM is introduced at this phase; persistence is file-based (Phase 6+).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """
    Types of nodes in the network graph.
    """
    WORKSTATION = "workstation"
    SERVER = "server"
    DATABASE = "database"
    ROUTER = "router"
    FIREWALL = "firewall"
    INTERNET = "internet"          # Entry point from external networks
    IOT_DEVICE = "iot_device"
    CRITICAL_ASSET = "critical_asset"  # High-value target (e.g., domain controller)


class VulnerabilitySeverity(str, Enum):
    """
    CVSS v3 qualitative severity ratings.
    Reference: https://www.first.org/cvss/specification-document
    """
    NONE = "none"          # CVSS 0.0
    LOW = "low"            # CVSS 0.1–3.9
    MEDIUM = "medium"      # CVSS 4.0–6.9
    HIGH = "high"          # CVSS 7.0–8.9
    CRITICAL = "critical"  # CVSS 9.0–10.0


class EdgeType(str, Enum):
    """
    Types of edges representing network relationships.
    """
    NETWORK_LINK = "network_link"          # Physical/logical network connection
    TRUST_RELATIONSHIP = "trust_relationship"  # e.g., AD trust, SSH key access
    EXPLOIT_PATH = "exploit_path"          # Known or inferred attack traversal edge
    DATA_FLOW = "data_flow"                # Data transfer between nodes


class OSType(str, Enum):
    """
    Operating system categories for nodes.
    """
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    FIRMWARE = "firmware"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Vulnerability
# ---------------------------------------------------------------------------

@dataclass
class Vulnerability:
    """
    Represents a known software vulnerability on a network node.

    Attributes:
        cve_id:        CVE identifier (e.g., "CVE-2021-44228"). May be synthetic.
        cvss_score:    CVSS v3 base score (0.0–10.0).
        severity:      Qualitative severity derived from cvss_score.
        description:   Human-readable description.
        exploitability_score: CVSS v3 exploitability sub-score (0.0–3.9).
        impact_score:  CVSS v3 impact sub-score (0.0–6.0).
        is_exploited_in_wild: Whether this CVE has known active exploitation.
        patch_available: Whether a patch exists.
    """
    cve_id: str
    cvss_score: float
    severity: VulnerabilitySeverity
    description: str
    exploitability_score: float = 0.0
    impact_score: float = 0.0
    is_exploited_in_wild: bool = False
    patch_available: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.cvss_score <= 10.0):
            raise ValueError(f"CVSS score must be 0.0–10.0, got {self.cvss_score}")
        if not (0.0 <= self.exploitability_score <= 3.9):
            raise ValueError(
                f"Exploitability score must be 0.0–3.9, got {self.exploitability_score}"
            )
        if not (0.0 <= self.impact_score <= 6.0):
            raise ValueError(f"Impact score must be 0.0–6.0, got {self.impact_score}")

    @classmethod
    def severity_from_cvss(cls, cvss_score: float) -> VulnerabilitySeverity:
        """Derive qualitative severity from a CVSS v3 score."""
        if cvss_score == 0.0:
            return VulnerabilitySeverity.NONE
        elif cvss_score < 4.0:
            return VulnerabilitySeverity.LOW
        elif cvss_score < 7.0:
            return VulnerabilitySeverity.MEDIUM
        elif cvss_score < 9.0:
            return VulnerabilitySeverity.HIGH
        else:
            return VulnerabilitySeverity.CRITICAL

    def to_dict(self) -> Dict:
        return {
            "cve_id": self.cve_id,
            "cvss_score": self.cvss_score,
            "severity": self.severity.value,
            "description": self.description,
            "exploitability_score": self.exploitability_score,
            "impact_score": self.impact_score,
            "is_exploited_in_wild": self.is_exploited_in_wild,
            "patch_available": self.patch_available,
        }


# ---------------------------------------------------------------------------
# Network Node
# ---------------------------------------------------------------------------

@dataclass
class NetworkNode:
    """
    Represents a single host or network device in the topology.

    Attributes:
        node_id:         Unique identifier (UUID string).
        label:           Human-readable name (e.g., "web-server-01").
        node_type:       Category of device.
        os_type:         Operating system.
        ip_address:      IPv4/IPv6 address (may be fictional in synthetic mode).
        subnet:          CIDR subnet (e.g., "192.168.1.0/24").
        services:        Open services/ports (e.g., ["ssh:22", "http:80"]).
        vulnerabilities: List of known vulnerabilities on this node.
        is_entry_point:  True if an attacker can directly reach this node.
        is_critical_asset: True if this node is a high-value target.
        compromised:     Used during simulation; False by default.
        asset_value:     Relative business value (0.0–1.0) for risk scoring.
    """
    node_id: str
    label: str
    node_type: NodeType
    os_type: OSType
    ip_address: str
    subnet: str
    services: List[str] = field(default_factory=list)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    is_entry_point: bool = False
    is_critical_asset: bool = False
    compromised: bool = False
    asset_value: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.asset_value <= 1.0):
            raise ValueError(f"asset_value must be 0.0–1.0, got {self.asset_value}")

    @property
    def max_cvss(self) -> float:
        """Highest CVSS score among all vulnerabilities on this node."""
        if not self.vulnerabilities:
            return 0.0
        return max(v.cvss_score for v in self.vulnerabilities)

    @property
    def has_critical_vuln(self) -> bool:
        """True if any vulnerability is CRITICAL severity."""
        return any(
            v.severity == VulnerabilitySeverity.CRITICAL for v in self.vulnerabilities
        )

    @property
    def has_exploited_in_wild(self) -> bool:
        """True if any vulnerability is actively exploited in the wild."""
        return any(v.is_exploited_in_wild for v in self.vulnerabilities)

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type.value,
            "os_type": self.os_type.value,
            "ip_address": self.ip_address,
            "subnet": self.subnet,
            "services": self.services,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "is_entry_point": self.is_entry_point,
            "is_critical_asset": self.is_critical_asset,
            "compromised": self.compromised,
            "asset_value": self.asset_value,
            "max_cvss": self.max_cvss,
            "has_critical_vuln": self.has_critical_vuln,
            "has_exploited_in_wild": self.has_exploited_in_wild,
        }


# ---------------------------------------------------------------------------
# Network Edge
# ---------------------------------------------------------------------------

@dataclass
class NetworkEdge:
    """
    Represents a directed connection between two network nodes.

    Design Note:
        Edges are directed to model the asymmetric nature of network access
        and attack traversal. An undirected physical link is represented as
        two directed edges (bidirectional=True creates both).

    Attributes:
        edge_id:         Unique identifier.
        source_id:       Source node ID.
        target_id:       Target node ID.
        edge_type:       Category of relationship.
        bidirectional:   If True, traffic flows both ways.
        protocol:        Network protocol (e.g., "tcp", "udp", "icmp").
        port:            Destination port (None if not applicable).
        firewall_rule:   Whether a firewall rule governs this connection.
        is_encrypted:    Whether traffic is encrypted (e.g., TLS/SSH).
        bandwidth_mbps:  Link bandwidth for telemetry-based features.
        latency_ms:      Average link latency.
        attack_weight:   Pre-computed attack traversal likelihood (0.0–1.0).
                         Updated by the GNN in later phases.
    """
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    bidirectional: bool = True
    protocol: str = "tcp"
    port: Optional[int] = None
    firewall_rule: bool = False
    is_encrypted: bool = False
    bandwidth_mbps: float = 100.0
    latency_ms: float = 1.0
    attack_weight: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.attack_weight <= 1.0):
            raise ValueError(f"attack_weight must be 0.0–1.0, got {self.attack_weight}")
        if self.source_id == self.target_id:
            raise ValueError("Self-loops are not allowed in the network graph.")

    def to_dict(self) -> Dict:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "bidirectional": self.bidirectional,
            "protocol": self.protocol,
            "port": self.port,
            "firewall_rule": self.firewall_rule,
            "is_encrypted": self.is_encrypted,
            "bandwidth_mbps": self.bandwidth_mbps,
            "latency_ms": self.latency_ms,
            "attack_weight": self.attack_weight,
        }


# ---------------------------------------------------------------------------
# Network Telemetry Record
# ---------------------------------------------------------------------------

@dataclass
class TelemetryRecord:
    """
    A single network telemetry observation (e.g., from an IDS/SIEM).

    This represents the kind of data that would be collected from real network
    monitoring tools. In Phase 1, this is used to annotate edges with
    behavioral signals.

    Attributes:
        record_id:      Unique identifier.
        source_id:      Source node ID.
        target_id:      Target node ID.
        timestamp:      Unix timestamp of observation.
        bytes_sent:     Volume of data in this flow.
        packet_count:   Number of packets.
        is_anomalous:   Flagged as anomalous by an IDS/SIEM.
        alert_type:     Type of alert if anomalous (e.g., "port_scan", "brute_force").
    """
    record_id: str
    source_id: str
    target_id: str
    timestamp: float
    bytes_sent: int = 0
    packet_count: int = 0
    is_anomalous: bool = False
    alert_type: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "record_id": self.record_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "timestamp": self.timestamp,
            "bytes_sent": self.bytes_sent,
            "packet_count": self.packet_count,
            "is_anomalous": self.is_anomalous,
            "alert_type": self.alert_type,
        }


# ---------------------------------------------------------------------------
# Historical Attack Record
# ---------------------------------------------------------------------------

@dataclass
class HistoricalAttack:
    """
    A recorded multi-step attack path from historical data or simulation.

    Used in Phase 2+ for supervised GNN training labels.

    Attributes:
        attack_id:    Unique identifier.
        attack_path:  Ordered list of node IDs representing traversal sequence.
        technique:    MITRE ATT&CK technique (e.g., "T1021 - Remote Services").
        success:      Whether the attack reached its final target.
        timestamp:    Unix timestamp when attack was recorded.
        source:       Data provenance label (e.g., "synthetic", "NVD", "CTF").
    """
    attack_id: str
    attack_path: List[str]        # Ordered node IDs
    technique: str
    success: bool
    timestamp: float
    source: str = "synthetic"     # Always document data provenance

    def __post_init__(self) -> None:
        if len(self.attack_path) < 2:
            raise ValueError("An attack path must have at least 2 nodes.")

    def to_dict(self) -> Dict:
        return {
            "attack_id": self.attack_id,
            "attack_path": self.attack_path,
            "technique": self.technique,
            "success": self.success,
            "timestamp": self.timestamp,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Network Topology Container
# ---------------------------------------------------------------------------

@dataclass
class NetworkTopology:
    """
    Top-level container for the full network representation.

    Attributes:
        topology_id:    Unique identifier for this topology snapshot.
        name:           Human-readable name (e.g., "Corporate-LAN-v1").
        nodes:          Dict mapping node_id -> NetworkNode.
        edges:          Dict mapping edge_id -> NetworkEdge.
        telemetry:      List of telemetry records.
        historical_attacks: List of historical attack records.
        is_synthetic:   Explicit flag — True if this topology is synthetic.
        description:    Free-text description of the network scenario.
    """
    topology_id: str
    name: str
    nodes: Dict[str, NetworkNode] = field(default_factory=dict)
    edges: Dict[str, NetworkEdge] = field(default_factory=dict)
    telemetry: List[TelemetryRecord] = field(default_factory=list)
    historical_attacks: List[HistoricalAttack] = field(default_factory=list)
    is_synthetic: bool = True
    description: str = ""

    @property
    def entry_points(self) -> List[NetworkNode]:
        """Nodes reachable directly by an external attacker."""
        return [n for n in self.nodes.values() if n.is_entry_point]

    @property
    def critical_assets(self) -> List[NetworkNode]:
        """High-value target nodes."""
        return [n for n in self.nodes.values() if n.is_critical_asset]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def add_node(self, node: NetworkNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: NetworkEdge) -> None:
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node '{edge.source_id}' not in topology.")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node '{edge.target_id}' not in topology.")
        self.edges[edge.edge_id] = edge

    def to_dict(self) -> Dict:
        return {
            "topology_id": self.topology_id,
            "name": self.name,
            "is_synthetic": self.is_synthetic,
            "description": self.description,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": {eid: e.to_dict() for eid, e in self.edges.items()},
            "telemetry": [t.to_dict() for t in self.telemetry],
            "historical_attacks": [a.to_dict() for a in self.historical_attacks],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_id(prefix: str = "") -> str:
    """Generate a short UUID-based identifier."""
    uid = str(uuid.uuid4()).replace("-", "")[:12]
    return f"{prefix}{uid}" if prefix else uid
