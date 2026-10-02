"""
synthetic_data.py
-----------------
Generates labeled synthetic network topologies for development and testing.

⚠️  DATA PROVENANCE WARNING:
    ALL data produced by this module is SYNTHETIC.
    It does not represent any real network, organization, or infrastructure.
    CVE IDs used here follow the format SYN-XXXX-XXXXX to distinguish them
    from real NVD entries. Real CVE IDs are NOT used.

Design Decisions:
- Random seed is always explicitly set and returned with the topology for
  reproducibility. Every generated topology includes `is_synthetic=True`.
- Three scenario presets are provided: small (10 nodes), medium (30 nodes),
  large (60 nodes). These simulate a tiered corporate network.
- The generator produces nodes across multiple subnets to simulate network
  segmentation, which is essential for realistic attack path modeling.
- Historical attack paths are generated with fixed structural logic
  (Internet → DMZ → Internal → Critical Asset) to provide valid training
  labels, not random noise.
"""

from __future__ import annotations

import random
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

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
# Synthetic Vulnerability Library
# ---------------------------------------------------------------------------
# CVE IDs use SYN- prefix to explicitly mark them as synthetic.
# CVSS scores reflect real-world severity distributions for research realism.

SYNTHETIC_VULN_LIBRARY: List[Dict] = [
    {
        "cve_id": "SYN-2021-44228",
        "cvss_score": 10.0,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Remote code execution via JNDI injection (Log4Shell analog)",
        "exploitability_score": 3.9,
        "impact_score": 6.0,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2021-26855",
        "cvss_score": 9.8,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Server-Side Request Forgery in mail server (ProxyLogon analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.9,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2020-1472",
        "cvss_score": 10.0,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Privilege escalation via Netlogon cryptography flaw (Zerologon analog)",
        "exploitability_score": 3.9,
        "impact_score": 6.0,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2019-0708",
        "cvss_score": 9.8,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Remote code execution via RDP without authentication (BlueKeep analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.9,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2022-30190",
        "cvss_score": 7.8,
        "severity": VulnerabilitySeverity.HIGH,
        "description": "Code execution via MSDT diagnostic tool (Follina analog)",
        "exploitability_score": 3.4,
        "impact_score": 5.9,
        "is_exploited_in_wild": False,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2021-34527",
        "cvss_score": 8.8,
        "severity": VulnerabilitySeverity.HIGH,
        "description": "Remote code execution via Windows Print Spooler (PrintNightmare analog)",
        "exploitability_score": 3.8,
        "impact_score": 5.9,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2017-5638",
        "cvss_score": 10.0,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Remote code execution via content-type header parsing (Struts2 analog)",
        "exploitability_score": 3.9,
        "impact_score": 6.0,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2018-7600",
        "cvss_score": 9.8,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Remote code execution in CMS framework (Drupalgeddon2 analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.9,
        "is_exploited_in_wild": False,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2020-0796",
        "cvss_score": 10.0,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Wormable remote code execution via SMBv3 (SMBGhost analog)",
        "exploitability_score": 3.9,
        "impact_score": 6.0,
        "is_exploited_in_wild": False,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2022-22965",
        "cvss_score": 9.8,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Remote code execution in Spring framework via data binding (Spring4Shell analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.9,
        "is_exploited_in_wild": False,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2019-11510",
        "cvss_score": 10.0,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Arbitrary file read in VPN appliance (Pulse Secure analog)",
        "exploitability_score": 3.9,
        "impact_score": 6.0,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2021-22986",
        "cvss_score": 9.8,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Remote code execution via REST API in load balancer (F5 BIG-IP analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.9,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2023-23397",
        "cvss_score": 9.8,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Privilege escalation via NTLM hash leak in mail client (Outlook analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.9,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2020-5902",
        "cvss_score": 10.0,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Remote code execution via management interface in network device (F5 TMUI analog)",
        "exploitability_score": 3.9,
        "impact_score": 6.0,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2018-11776",
        "cvss_score": 8.1,
        "severity": VulnerabilitySeverity.HIGH,
        "description": "Remote code execution in web framework namespace handling (Struts2 RCE analog)",
        "exploitability_score": 3.6,
        "impact_score": 5.9,
        "is_exploited_in_wild": False,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2019-19781",
        "cvss_score": 9.8,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Path traversal and RCE in VPN appliance (Citrix ADC analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.9,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2021-3156",
        "cvss_score": 7.8,
        "severity": VulnerabilitySeverity.HIGH,
        "description": "Local privilege escalation via heap overflow in sudo (Baron Samedit analog)",
        "exploitability_score": 3.4,
        "impact_score": 5.9,
        "is_exploited_in_wild": False,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2022-26134",
        "cvss_score": 9.8,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Remote code execution via OGNL injection in wiki software (Confluence analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.9,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2016-0167",
        "cvss_score": 7.8,
        "severity": VulnerabilitySeverity.HIGH,
        "description": "Privilege escalation via Windows kernel vulnerability",
        "exploitability_score": 3.4,
        "impact_score": 5.9,
        "is_exploited_in_wild": False,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2023-20198",
        "cvss_score": 10.0,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Authentication bypass in network OS management interface (IOS XE analog)",
        "exploitability_score": 3.9,
        "impact_score": 6.0,
        "is_exploited_in_wild": True,
        "patch_available": False,
    },
    {
        "cve_id": "SYN-2022-1388",
        "cvss_score": 9.8,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Authentication bypass and RCE in network appliance REST API (F5 BIG-IP iControl analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.9,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2014-6271",
        "cvss_score": 9.8,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Remote code execution via crafted environment variables in bash (Shellshock analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.9,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2023-4966",
        "cvss_score": 9.4,
        "severity": VulnerabilitySeverity.CRITICAL,
        "description": "Session token leakage in VPN appliance (Citrix Bleed analog)",
        "exploitability_score": 3.9,
        "impact_score": 5.2,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-2021-40444",
        "cvss_score": 8.8,
        "severity": VulnerabilitySeverity.HIGH,
        "description": "Remote code execution via malicious Office document",
        "exploitability_score": 3.8,
        "impact_score": 5.9,
        "is_exploited_in_wild": True,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-MEDIUM-001",
        "cvss_score": 6.5,
        "severity": VulnerabilitySeverity.MEDIUM,
        "description": "SQL injection in internal web application",
        "exploitability_score": 2.8,
        "impact_score": 3.6,
        "is_exploited_in_wild": False,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-MEDIUM-002",
        "cvss_score": 5.3,
        "severity": VulnerabilitySeverity.MEDIUM,
        "description": "Information disclosure via directory listing in web server",
        "exploitability_score": 3.9,
        "impact_score": 1.4,
        "is_exploited_in_wild": False,
        "patch_available": True,
    },
    {
        "cve_id": "SYN-LOW-001",
        "cvss_score": 3.1,
        "severity": VulnerabilitySeverity.LOW,
        "description": "Weak TLS cipher suite negotiation",
        "exploitability_score": 1.6,
        "impact_score": 1.4,
        "is_exploited_in_wild": False,
        "patch_available": True,
    },
]


# ---------------------------------------------------------------------------
# Node Templates
# ---------------------------------------------------------------------------

def _make_vulnerability(vuln_dict: Dict) -> Vulnerability:
    """Instantiate a Vulnerability from the library dict."""
    return Vulnerability(
        cve_id=vuln_dict["cve_id"],
        cvss_score=vuln_dict["cvss_score"],
        severity=vuln_dict["severity"],
        description=vuln_dict["description"],
        exploitability_score=vuln_dict["exploitability_score"],
        impact_score=vuln_dict["impact_score"],
        is_exploited_in_wild=vuln_dict["is_exploited_in_wild"],
        patch_available=vuln_dict["patch_available"],
    )


def _sample_vulns(rng: random.Random, count: int) -> List[Vulnerability]:
    """Sample `count` vulnerabilities from the synthetic library."""
    sampled = rng.sample(SYNTHETIC_VULN_LIBRARY, min(count, len(SYNTHETIC_VULN_LIBRARY)))
    return [_make_vulnerability(v) for v in sampled]


# ---------------------------------------------------------------------------
# Subnet Definitions
# ---------------------------------------------------------------------------

SUBNETS = {
    "dmz":      "10.0.1.0/24",
    "internal": "10.0.2.0/24",
    "corp":     "10.0.3.0/24",
    "db":       "10.0.4.0/24",
    "mgmt":     "10.0.5.0/24",
    "internet": "0.0.0.0/0",
}


def _ip(subnet_key: str, host: int) -> str:
    """Generate a synthetic IP address within a named subnet."""
    bases = {
        "dmz": "10.0.1",
        "internal": "10.0.2",
        "corp": "10.0.3",
        "db": "10.0.4",
        "mgmt": "10.0.5",
        "internet": "203.0.113",  # TEST-NET-3 (RFC 5737) — not routable
    }
    return f"{bases[subnet_key]}.{host}"


# ---------------------------------------------------------------------------
# Core Generator
# ---------------------------------------------------------------------------

class SyntheticNetworkGenerator:
    """
    Generates synthetic network topologies for research and testing.

    All topologies have:
    - Explicit `is_synthetic=True` flag
    - CVE IDs with SYN- prefix
    - Reproducible output via fixed random seeds
    - Layered subnet structure (Internet → DMZ → Internal → DB/MGMT)

    Usage:
        gen = SyntheticNetworkGenerator(seed=42)
        topology = gen.generate_small()
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def _reset_rng(self) -> None:
        """Reset RNG to ensure reproducible generation within a session."""
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)

    def _make_node_id(self) -> str:
        return make_id("node-")

    def _make_edge_id(self) -> str:
        return make_id("edge-")

    def _make_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.NETWORK_LINK,
        bidirectional: bool = True,
        protocol: str = "tcp",
        port: Optional[int] = None,
        firewall_rule: bool = False,
        is_encrypted: bool = False,
    ) -> NetworkEdge:
        attack_weight = float(self._np_rng.uniform(0.1, 0.9))
        if firewall_rule:
            attack_weight *= 0.5  # Firewall reduces traversal likelihood
        if is_encrypted:
            attack_weight *= 0.8  # Encryption slightly reduces attack surface
        attack_weight = round(min(max(attack_weight, 0.0), 1.0), 4)
        return NetworkEdge(
            edge_id=self._make_edge_id(),
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            bidirectional=bidirectional,
            protocol=protocol,
            port=port,
            firewall_rule=firewall_rule,
            is_encrypted=is_encrypted,
            bandwidth_mbps=float(self._np_rng.choice([10, 100, 1000])),
            latency_ms=float(round(self._np_rng.uniform(0.5, 20.0), 2)),
            attack_weight=attack_weight,
        )

    def _generate_telemetry(
        self,
        nodes: List[NetworkNode],
        edges: List[NetworkEdge],
        n_records: int = 50,
    ) -> List[TelemetryRecord]:
        """Generate synthetic telemetry records for the given topology."""
        records = []
        base_time = 1_700_000_000.0  # Fixed epoch for reproducibility
        for i in range(n_records):
            edge = self._rng.choice(edges)
            is_anomalous = self._rng.random() < 0.15  # 15% anomaly rate
            alert_types = ["port_scan", "brute_force", "lateral_movement",
                           "data_exfiltration", "c2_beacon", None]
            records.append(TelemetryRecord(
                record_id=make_id("tel-"),
                source_id=edge.source_id,
                target_id=edge.target_id,
                timestamp=base_time + i * self._rng.uniform(10, 300),
                bytes_sent=int(self._np_rng.integers(64, 1_000_000)),
                packet_count=int(self._np_rng.integers(1, 10_000)),
                is_anomalous=is_anomalous,
                alert_type=self._rng.choice(alert_types[:-1]) if is_anomalous else None,
            ))
        return records

    def _generate_attack_paths(
        self,
        entry_points: List[NetworkNode],
        critical_assets: List[NetworkNode],
        intermediate_nodes: List[NetworkNode],
        n_attacks: int = 10,
    ) -> List[HistoricalAttack]:
        """
        Generate synthetic historical attack paths.

        Attack structure follows a simplified kill chain:
          Entry Point → (1–3 intermediate hops) → Critical Asset

        All paths are labeled source="synthetic".
        """
        if not entry_points or not critical_assets or not intermediate_nodes:
            return []

        mitre_techniques = [
            "T1190 - Exploit Public-Facing Application",
            "T1021 - Remote Services",
            "T1078 - Valid Accounts",
            "T1059 - Command and Scripting Interpreter",
            "T1548 - Abuse Elevation Control Mechanism",
            "T1210 - Exploitation of Remote Services",
            "T1557 - Adversary-in-the-Middle",
            "T1003 - OS Credential Dumping",
        ]

        attacks = []
        base_time = 1_700_000_000.0
        for i in range(n_attacks):
            entry = self._rng.choice(entry_points)
            target = self._rng.choice(critical_assets)
            n_hops = self._rng.randint(1, min(3, len(intermediate_nodes)))
            hops = self._rng.sample(intermediate_nodes, n_hops)
            path = [entry.node_id] + [h.node_id for h in hops] + [target.node_id]
            attacks.append(HistoricalAttack(
                attack_id=make_id("atk-"),
                attack_path=path,
                technique=self._rng.choice(mitre_techniques),
                success=self._rng.random() > 0.3,  # 70% success rate
                timestamp=base_time + i * self._rng.uniform(3600, 86400),
                source="synthetic",
            ))
        return attacks

    # -----------------------------------------------------------------------
    # Public Generation Methods
    # -----------------------------------------------------------------------

    def generate_small(self) -> NetworkTopology:
        """
        Generate a small synthetic network (≈10 nodes).

        Scenario: Small office/home office (SOHO) with a DMZ web server,
        internal workstations, a file server, and a domain controller.

        [SYNTHETIC DATA]
        """
        self._reset_rng()
        topology = NetworkTopology(
            topology_id=make_id("topo-"),
            name="SOHO-Network-Synthetic-Small",
            is_synthetic=True,
            description=(
                "[SYNTHETIC] Small office network with 10 nodes. "
                "Scenario: external attacker targeting domain controller via web DMZ."
            ),
        )

        # --- Nodes ---
        internet = NetworkNode(
            node_id=self._make_node_id(),
            label="internet-gateway",
            node_type=NodeType.INTERNET,
            os_type=OSType.UNKNOWN,
            ip_address=_ip("internet", 1),
            subnet=SUBNETS["internet"],
            services=[],
            vulnerabilities=[],
            is_entry_point=True,
            is_critical_asset=False,
            asset_value=0.1,
        )

        firewall = NetworkNode(
            node_id=self._make_node_id(),
            label="firewall-01",
            node_type=NodeType.FIREWALL,
            os_type=OSType.LINUX,
            ip_address=_ip("dmz", 1),
            subnet=SUBNETS["dmz"],
            services=["iptables"],
            vulnerabilities=[],
            is_entry_point=False,
            is_critical_asset=False,
            asset_value=0.6,
        )

        web_server = NetworkNode(
            node_id=self._make_node_id(),
            label="web-server-01",
            node_type=NodeType.SERVER,
            os_type=OSType.LINUX,
            ip_address=_ip("dmz", 10),
            subnet=SUBNETS["dmz"],
            services=["http:80", "https:443", "ssh:22"],
            vulnerabilities=_sample_vulns(self._rng, 2),
            is_entry_point=True,
            is_critical_asset=False,
            asset_value=0.4,
        )

        mail_server = NetworkNode(
            node_id=self._make_node_id(),
            label="mail-server-01",
            node_type=NodeType.SERVER,
            os_type=OSType.WINDOWS,
            ip_address=_ip("dmz", 11),
            subnet=SUBNETS["dmz"],
            services=["smtp:25", "imap:143", "https:443"],
            vulnerabilities=_sample_vulns(self._rng, 2),
            is_entry_point=True,
            is_critical_asset=False,
            asset_value=0.5,
        )

        ws1 = NetworkNode(
            node_id=self._make_node_id(),
            label="workstation-01",
            node_type=NodeType.WORKSTATION,
            os_type=OSType.WINDOWS,
            ip_address=_ip("internal", 10),
            subnet=SUBNETS["internal"],
            services=["smb:445", "rdp:3389"],
            vulnerabilities=_sample_vulns(self._rng, 1),
            is_entry_point=False,
            is_critical_asset=False,
            asset_value=0.3,
        )

        ws2 = NetworkNode(
            node_id=self._make_node_id(),
            label="workstation-02",
            node_type=NodeType.WORKSTATION,
            os_type=OSType.WINDOWS,
            ip_address=_ip("internal", 11),
            subnet=SUBNETS["internal"],
            services=["smb:445"],
            vulnerabilities=[],
            is_entry_point=False,
            is_critical_asset=False,
            asset_value=0.3,
        )

        file_server = NetworkNode(
            node_id=self._make_node_id(),
            label="file-server-01",
            node_type=NodeType.SERVER,
            os_type=OSType.WINDOWS,
            ip_address=_ip("internal", 20),
            subnet=SUBNETS["internal"],
            services=["smb:445", "nfs:2049"],
            vulnerabilities=_sample_vulns(self._rng, 2),
            is_entry_point=False,
            is_critical_asset=False,
            asset_value=0.6,
        )

        db_server = NetworkNode(
            node_id=self._make_node_id(),
            label="db-server-01",
            node_type=NodeType.DATABASE,
            os_type=OSType.LINUX,
            ip_address=_ip("db", 10),
            subnet=SUBNETS["db"],
            services=["mysql:3306", "ssh:22"],
            vulnerabilities=_sample_vulns(self._rng, 1),
            is_entry_point=False,
            is_critical_asset=True,
            asset_value=0.85,
        )

        domain_controller = NetworkNode(
            node_id=self._make_node_id(),
            label="domain-controller-01",
            node_type=NodeType.CRITICAL_ASSET,
            os_type=OSType.WINDOWS,
            ip_address=_ip("mgmt", 10),
            subnet=SUBNETS["mgmt"],
            services=["ldap:389", "kerberos:88", "smb:445", "rdp:3389"],
            vulnerabilities=_sample_vulns(self._rng, 2),
            is_entry_point=False,
            is_critical_asset=True,
            asset_value=1.0,
        )

        router = NetworkNode(
            node_id=self._make_node_id(),
            label="core-router-01",
            node_type=NodeType.ROUTER,
            os_type=OSType.FIRMWARE,
            ip_address=_ip("internal", 1),
            subnet=SUBNETS["internal"],
            services=["snmp:161"],
            vulnerabilities=[],
            is_entry_point=False,
            is_critical_asset=False,
            asset_value=0.7,
        )

        all_nodes = [
            internet, firewall, web_server, mail_server,
            ws1, ws2, file_server, db_server, domain_controller, router,
        ]
        for node in all_nodes:
            topology.add_node(node)

        # --- Edges ---
        edges_to_add = [
            # Internet → DMZ
            self._make_edge(internet.node_id, firewall.node_id, firewall_rule=True, port=443),
            self._make_edge(internet.node_id, web_server.node_id, firewall_rule=True, port=443),
            self._make_edge(internet.node_id, mail_server.node_id, firewall_rule=True, port=25),
            # DMZ → Internal
            self._make_edge(firewall.node_id, router.node_id, firewall_rule=True),
            self._make_edge(web_server.node_id, router.node_id, firewall_rule=True),
            self._make_edge(mail_server.node_id, router.node_id, firewall_rule=True),
            # Internal distribution
            self._make_edge(router.node_id, ws1.node_id),
            self._make_edge(router.node_id, ws2.node_id),
            self._make_edge(router.node_id, file_server.node_id),
            # Internal → DB
            self._make_edge(file_server.node_id, db_server.node_id, firewall_rule=True, port=3306),
            self._make_edge(ws1.node_id, db_server.node_id, firewall_rule=True, port=3306),
            # Internal → MGMT (Domain Controller)
            self._make_edge(ws1.node_id, domain_controller.node_id, port=389),
            self._make_edge(ws2.node_id, domain_controller.node_id, port=389),
            self._make_edge(file_server.node_id, domain_controller.node_id, port=445),
            # Trust / lateral movement
            self._make_edge(
                web_server.node_id, file_server.node_id,
                edge_type=EdgeType.TRUST_RELATIONSHIP, is_encrypted=True,
            ),
        ]
        for edge in edges_to_add:
            topology.add_edge(edge)

        # --- Telemetry ---
        topology.telemetry = self._generate_telemetry(
            list(topology.nodes.values()), edges_to_add, n_records=40
        )

        # --- Historical Attacks ---
        entry_points = topology.entry_points
        critical_assets = topology.critical_assets
        intermediate = [
            n for n in topology.nodes.values()
            if not n.is_entry_point and not n.is_critical_asset
        ]
        topology.historical_attacks = self._generate_attack_paths(
            entry_points, critical_assets, intermediate, n_attacks=8
        )

        return topology

    def generate_medium(self) -> NetworkTopology:
        """
        Generate a medium synthetic network (≈30 nodes).

        Scenario: Mid-sized enterprise with segmented zones, multiple servers,
        IoT devices, and two critical assets.

        [SYNTHETIC DATA]
        """
        self._reset_rng()
        topology = NetworkTopology(
            topology_id=make_id("topo-"),
            name="Enterprise-Network-Synthetic-Medium",
            is_synthetic=True,
            description=(
                "[SYNTHETIC] Medium enterprise network with ~30 nodes. "
                "Scenario: APT-style lateral movement toward ERP database and AD."
            ),
        )

        nodes: List[NetworkNode] = []

        # Internet + Firewall
        internet = NetworkNode(
            node_id=self._make_node_id(), label="internet-gateway",
            node_type=NodeType.INTERNET, os_type=OSType.UNKNOWN,
            ip_address=_ip("internet", 1), subnet=SUBNETS["internet"],
            is_entry_point=True, asset_value=0.1,
        )
        nodes.append(internet)

        fw1 = NetworkNode(
            node_id=self._make_node_id(), label="firewall-perimeter",
            node_type=NodeType.FIREWALL, os_type=OSType.LINUX,
            ip_address=_ip("dmz", 1), subnet=SUBNETS["dmz"],
            asset_value=0.7,
        )
        nodes.append(fw1)

        # DMZ Servers (5)
        dmz_labels = [
            ("web-server-01", NodeType.SERVER, OSType.LINUX,
             _ip("dmz", 10), ["http:80", "https:443"], True, 0.4),
            ("web-server-02", NodeType.SERVER, OSType.LINUX,
             _ip("dmz", 11), ["https:443"], True, 0.4),
            ("mail-server-01", NodeType.SERVER, OSType.WINDOWS,
             _ip("dmz", 12), ["smtp:25", "imap:143"], True, 0.5),
            ("ftp-server-01", NodeType.SERVER, OSType.LINUX,
             _ip("dmz", 13), ["ftp:21"], True, 0.35),
            ("vpn-gateway-01", NodeType.SERVER, OSType.LINUX,
             _ip("dmz", 14), ["openvpn:1194", "ssh:22"], True, 0.55),
        ]
        dmz_nodes = []
        for label, ntype, ostype, ip, svcs, is_entry, aval in dmz_labels:
            n = NetworkNode(
                node_id=self._make_node_id(), label=label,
                node_type=ntype, os_type=ostype, ip_address=ip,
                subnet=SUBNETS["dmz"], services=svcs,
                vulnerabilities=_sample_vulns(self._rng, self._rng.randint(1, 3)),
                is_entry_point=is_entry, asset_value=aval,
            )
            nodes.append(n)
            dmz_nodes.append(n)

        # Core Router
        router = NetworkNode(
            node_id=self._make_node_id(), label="core-router",
            node_type=NodeType.ROUTER, os_type=OSType.FIRMWARE,
            ip_address=_ip("internal", 1), subnet=SUBNETS["internal"],
            asset_value=0.8,
        )
        nodes.append(router)

        # Internal Servers (6)
        internal_server_specs = [
            ("app-server-01", NodeType.SERVER, OSType.LINUX,
             _ip("internal", 20), ["http:8080", "ssh:22"], 0.5),
            ("app-server-02", NodeType.SERVER, OSType.LINUX,
             _ip("internal", 21), ["http:8080"], 0.5),
            ("file-server-01", NodeType.SERVER, OSType.WINDOWS,
             _ip("internal", 22), ["smb:445", "nfs:2049"], 0.6),
            ("backup-server-01", NodeType.SERVER, OSType.LINUX,
             _ip("internal", 23), ["rsync:873", "ssh:22"], 0.65),
            ("monitoring-server", NodeType.SERVER, OSType.LINUX,
             _ip("internal", 24), ["http:9090", "https:443"], 0.6),
            ("print-server-01", NodeType.SERVER, OSType.WINDOWS,
             _ip("internal", 25), ["lpd:515", "smb:445"], 0.35),
        ]
        internal_servers = []
        for label, ntype, ostype, ip, svcs, aval in internal_server_specs:
            n = NetworkNode(
                node_id=self._make_node_id(), label=label,
                node_type=ntype, os_type=ostype, ip_address=ip,
                subnet=SUBNETS["internal"], services=svcs,
                vulnerabilities=_sample_vulns(self._rng, self._rng.randint(0, 2)),
                asset_value=aval,
            )
            nodes.append(n)
            internal_servers.append(n)

        # Workstations (8)
        workstations = []
        for i in range(1, 9):
            os_choice = self._rng.choice([OSType.WINDOWS, OSType.LINUX, OSType.MACOS])
            svcs = ["smb:445", "rdp:3389"] if os_choice == OSType.WINDOWS else ["ssh:22"]
            n = NetworkNode(
                node_id=self._make_node_id(), label=f"workstation-{i:02d}",
                node_type=NodeType.WORKSTATION, os_type=os_choice,
                ip_address=_ip("corp", 10 + i), subnet=SUBNETS["corp"],
                services=svcs,
                vulnerabilities=_sample_vulns(self._rng, self._rng.randint(0, 2)),
                asset_value=0.3,
            )
            nodes.append(n)
            workstations.append(n)

        # IoT Devices (3)
        iot_devices = []
        for i in range(1, 4):
            n = NetworkNode(
                node_id=self._make_node_id(), label=f"iot-device-{i:02d}",
                node_type=NodeType.IOT_DEVICE, os_type=OSType.FIRMWARE,
                ip_address=_ip("internal", 50 + i), subnet=SUBNETS["internal"],
                services=["http:80", "telnet:23"],
                vulnerabilities=_sample_vulns(self._rng, 2),
                asset_value=0.2,
            )
            nodes.append(n)
            iot_devices.append(n)

        # DB Servers (2)
        db1 = NetworkNode(
            node_id=self._make_node_id(), label="erp-db-server",
            node_type=NodeType.DATABASE, os_type=OSType.LINUX,
            ip_address=_ip("db", 10), subnet=SUBNETS["db"],
            services=["postgresql:5432", "ssh:22"],
            vulnerabilities=_sample_vulns(self._rng, 1),
            is_critical_asset=True, asset_value=0.9,
        )
        db2 = NetworkNode(
            node_id=self._make_node_id(), label="customer-db-server",
            node_type=NodeType.DATABASE, os_type=OSType.WINDOWS,
            ip_address=_ip("db", 11), subnet=SUBNETS["db"],
            services=["mssql:1433"],
            vulnerabilities=_sample_vulns(self._rng, 2),
            is_critical_asset=True, asset_value=0.95,
        )
        nodes.extend([db1, db2])

        # Domain Controller
        dc = NetworkNode(
            node_id=self._make_node_id(), label="domain-controller-01",
            node_type=NodeType.CRITICAL_ASSET, os_type=OSType.WINDOWS,
            ip_address=_ip("mgmt", 10), subnet=SUBNETS["mgmt"],
            services=["ldap:389", "kerberos:88", "smb:445", "rdp:3389"],
            vulnerabilities=_sample_vulns(self._rng, 2),
            is_critical_asset=True, asset_value=1.0,
        )
        nodes.append(dc)

        for node in nodes:
            topology.add_node(node)

        # --- Edges ---
        edges: List[NetworkEdge] = []

        # Internet → DMZ
        edges.append(self._make_edge(internet.node_id, fw1.node_id, firewall_rule=True))
        for dmz_node in dmz_nodes:
            edges.append(self._make_edge(
                internet.node_id, dmz_node.node_id, firewall_rule=True
            ))

        # Firewall → Router
        edges.append(self._make_edge(fw1.node_id, router.node_id, firewall_rule=True))

        # DMZ → Router (limited)
        for dmz_node in dmz_nodes[:3]:
            edges.append(self._make_edge(
                dmz_node.node_id, router.node_id, firewall_rule=True
            ))

        # Router → Internal servers
        for srv in internal_servers:
            edges.append(self._make_edge(router.node_id, srv.node_id))

        # Router → Workstations
        for ws in workstations:
            edges.append(self._make_edge(router.node_id, ws.node_id))

        # Router → IoT
        for iot in iot_devices:
            edges.append(self._make_edge(router.node_id, iot.node_id))

        # Internal servers → DBs
        for srv in internal_servers[:3]:
            edges.append(self._make_edge(
                srv.node_id, db1.node_id, firewall_rule=True, port=5432
            ))
            edges.append(self._make_edge(
                srv.node_id, db2.node_id, firewall_rule=True, port=1433
            ))

        # Workstations → DC
        for ws in workstations:
            edges.append(self._make_edge(ws.node_id, dc.node_id, port=389))

        # Some lateral movement edges (trust relationships)
        for i in range(0, len(internal_servers) - 1, 2):
            edges.append(self._make_edge(
                internal_servers[i].node_id,
                internal_servers[i + 1].node_id,
                edge_type=EdgeType.TRUST_RELATIONSHIP,
                is_encrypted=True,
            ))

        for edge in edges:
            topology.add_edge(edge)

        # Telemetry
        topology.telemetry = self._generate_telemetry(
            list(topology.nodes.values()), edges, n_records=100
        )

        # Historical attacks
        entry_pts = topology.entry_points
        crits = topology.critical_assets
        intermediate = [
            n for n in topology.nodes.values()
            if not n.is_entry_point and not n.is_critical_asset
        ]
        topology.historical_attacks = self._generate_attack_paths(
            entry_pts, crits, intermediate, n_attacks=20
        )

        return topology

    def generate_large(self) -> NetworkTopology:
        """
        Generate a large synthetic network (≈60 nodes).

        Scenario: Large enterprise / data center with multiple VLANs,
        production servers, OT/IoT segment, and multiple critical assets.

        [SYNTHETIC DATA]
        """
        self._reset_rng()
        topology = NetworkTopology(
            topology_id=make_id("topo-"),
            name="DataCenter-Network-Synthetic-Large",
            is_synthetic=True,
            description=(
                "[SYNTHETIC] Large data center network with ~60 nodes. "
                "Scenario: Nation-state APT targeting manufacturing OT systems and SCADA."
            ),
        )

        nodes: List[NetworkNode] = []

        # Internet
        internet = NetworkNode(
            node_id=self._make_node_id(), label="internet",
            node_type=NodeType.INTERNET, os_type=OSType.UNKNOWN,
            ip_address=_ip("internet", 1), subnet=SUBNETS["internet"],
            is_entry_point=True, asset_value=0.1,
        )
        nodes.append(internet)

        # 2 Firewalls
        fws = []
        for i in range(1, 3):
            fw = NetworkNode(
                node_id=self._make_node_id(), label=f"firewall-0{i}",
                node_type=NodeType.FIREWALL, os_type=OSType.LINUX,
                ip_address=_ip("dmz", i), subnet=SUBNETS["dmz"],
                asset_value=0.8,
            )
            nodes.append(fw)
            fws.append(fw)

        # 2 Routers
        routers = []
        for i in range(1, 3):
            r = NetworkNode(
                node_id=self._make_node_id(), label=f"core-router-0{i}",
                node_type=NodeType.ROUTER, os_type=OSType.FIRMWARE,
                ip_address=_ip("internal", i), subnet=SUBNETS["internal"],
                asset_value=0.8,
            )
            nodes.append(r)
            routers.append(r)

        # 8 DMZ servers
        dmz_nodes = []
        for i in range(1, 9):
            svcs = self._rng.sample(
                ["http:80", "https:443", "smtp:25", "ftp:21", "ssh:22", "rdp:3389"], 2
            )
            n = NetworkNode(
                node_id=self._make_node_id(), label=f"dmz-server-{i:02d}",
                node_type=NodeType.SERVER, os_type=self._rng.choice(list(OSType)),
                ip_address=_ip("dmz", 10 + i), subnet=SUBNETS["dmz"],
                services=svcs,
                vulnerabilities=_sample_vulns(self._rng, self._rng.randint(1, 3)),
                is_entry_point=True, asset_value=0.4,
            )
            nodes.append(n)
            dmz_nodes.append(n)

        # 15 Internal servers
        internal_servers = []
        for i in range(1, 16):
            n = NetworkNode(
                node_id=self._make_node_id(), label=f"internal-server-{i:02d}",
                node_type=NodeType.SERVER,
                os_type=self._rng.choice([OSType.LINUX, OSType.WINDOWS]),
                ip_address=_ip("internal", 20 + i), subnet=SUBNETS["internal"],
                services=["http:8080", "ssh:22"],
                vulnerabilities=_sample_vulns(self._rng, self._rng.randint(0, 3)),
                asset_value=round(self._np_rng.uniform(0.3, 0.7), 2),
            )
            nodes.append(n)
            internal_servers.append(n)

        # 20 Workstations
        workstations = []
        for i in range(1, 21):
            os_c = self._rng.choice([OSType.WINDOWS, OSType.LINUX, OSType.MACOS])
            n = NetworkNode(
                node_id=self._make_node_id(), label=f"workstation-{i:02d}",
                node_type=NodeType.WORKSTATION, os_type=os_c,
                ip_address=_ip("corp", 10 + i), subnet=SUBNETS["corp"],
                services=["smb:445"] if os_c == OSType.WINDOWS else ["ssh:22"],
                vulnerabilities=_sample_vulns(self._rng, self._rng.randint(0, 2)),
                asset_value=0.3,
            )
            nodes.append(n)
            workstations.append(n)

        # 5 IoT / OT Devices
        iot_devices = []
        for i in range(1, 6):
            n = NetworkNode(
                node_id=self._make_node_id(), label=f"iot-ot-device-{i:02d}",
                node_type=NodeType.IOT_DEVICE, os_type=OSType.FIRMWARE,
                ip_address=_ip("internal", 60 + i), subnet=SUBNETS["internal"],
                services=["http:80", "modbus:502"],
                vulnerabilities=_sample_vulns(self._rng, 2),
                asset_value=0.25,
            )
            nodes.append(n)
            iot_devices.append(n)

        # 4 DB Servers
        db_servers = []
        for i in range(1, 5):
            n = NetworkNode(
                node_id=self._make_node_id(), label=f"db-server-{i:02d}",
                node_type=NodeType.DATABASE,
                os_type=self._rng.choice([OSType.LINUX, OSType.WINDOWS]),
                ip_address=_ip("db", 10 + i), subnet=SUBNETS["db"],
                services=["postgresql:5432"],
                vulnerabilities=_sample_vulns(self._rng, self._rng.randint(1, 2)),
                is_critical_asset=True, asset_value=round(self._np_rng.uniform(0.8, 1.0), 2),
            )
            nodes.append(n)
            db_servers.append(n)

        # 3 Critical MGMT Nodes
        dc = NetworkNode(
            node_id=self._make_node_id(), label="domain-controller-01",
            node_type=NodeType.CRITICAL_ASSET, os_type=OSType.WINDOWS,
            ip_address=_ip("mgmt", 10), subnet=SUBNETS["mgmt"],
            services=["ldap:389", "kerberos:88", "smb:445"],
            vulnerabilities=_sample_vulns(self._rng, 2),
            is_critical_asset=True, asset_value=1.0,
        )
        scada = NetworkNode(
            node_id=self._make_node_id(), label="scada-server-01",
            node_type=NodeType.CRITICAL_ASSET, os_type=OSType.WINDOWS,
            ip_address=_ip("mgmt", 11), subnet=SUBNETS["mgmt"],
            services=["http:80", "modbus:502", "dnp3:20000"],
            vulnerabilities=_sample_vulns(self._rng, 3),
            is_critical_asset=True, asset_value=1.0,
        )
        backup_ctrl = NetworkNode(
            node_id=self._make_node_id(), label="backup-controller",
            node_type=NodeType.CRITICAL_ASSET, os_type=OSType.LINUX,
            ip_address=_ip("mgmt", 12), subnet=SUBNETS["mgmt"],
            services=["ssh:22", "rsync:873"],
            vulnerabilities=_sample_vulns(self._rng, 1),
            is_critical_asset=True, asset_value=0.9,
        )
        nodes.extend([dc, scada, backup_ctrl])

        for node in nodes:
            topology.add_node(node)

        # --- Edges ---
        edges: List[NetworkEdge] = []

        edges.append(self._make_edge(internet.node_id, fws[0].node_id, firewall_rule=True))
        edges.append(self._make_edge(internet.node_id, fws[1].node_id, firewall_rule=True))
        for dmz in dmz_nodes:
            edges.append(self._make_edge(internet.node_id, dmz.node_id, firewall_rule=True))
        for fw in fws:
            for r in routers:
                edges.append(self._make_edge(fw.node_id, r.node_id, firewall_rule=True))
        for dmz in dmz_nodes[:4]:
            edges.append(self._make_edge(dmz.node_id, routers[0].node_id, firewall_rule=True))
        for dmz in dmz_nodes[4:]:
            edges.append(self._make_edge(dmz.node_id, routers[1].node_id, firewall_rule=True))
        for srv in internal_servers:
            r = self._rng.choice(routers)
            edges.append(self._make_edge(r.node_id, srv.node_id))
        for ws in workstations:
            r = self._rng.choice(routers)
            edges.append(self._make_edge(r.node_id, ws.node_id))
        for iot in iot_devices:
            edges.append(self._make_edge(routers[0].node_id, iot.node_id))
        for srv in internal_servers[:8]:
            for db in db_servers[:2]:
                edges.append(self._make_edge(srv.node_id, db.node_id, firewall_rule=True))
        for ws in workstations[:10]:
            edges.append(self._make_edge(ws.node_id, dc.node_id, port=389))
        for iot in iot_devices:
            edges.append(self._make_edge(iot.node_id, scada.node_id, firewall_rule=True))
        for srv in internal_servers[:3]:
            edges.append(self._make_edge(
                srv.node_id, backup_ctrl.node_id,
                edge_type=EdgeType.TRUST_RELATIONSHIP, is_encrypted=True
            ))

        for edge in edges:
            topology.add_edge(edge)

        topology.telemetry = self._generate_telemetry(
            list(topology.nodes.values()), edges, n_records=200
        )
        entry_pts = topology.entry_points
        crits = topology.critical_assets
        intermediate = [
            n for n in topology.nodes.values()
            if not n.is_entry_point and not n.is_critical_asset
        ]
        topology.historical_attacks = self._generate_attack_paths(
            entry_pts, crits, intermediate, n_attacks=40
        )

        return topology
