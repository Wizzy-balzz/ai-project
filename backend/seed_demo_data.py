"""
seed_demo_data.py
-----------------
Seeds "Operation Golden Vineyard (Synthetic Enterprise)" topology into SQLite.
Guarantees standard primary keys, complete deletability via the canvas,
and pre-calculated Top-K attack trajectories.
"""
from __future__ import annotations

import json
import logging
import os
import sys

# Ensure root directory in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.database import engine, SessionLocal, Base
from backend.app.models.scenario import Scenario
from backend.app.models.node import NetworkNode
from backend.app.models.edge import NetworkEdge
from backend.app.models.attack_path import AttackPath
from backend.app.models.audit import AuditEvent
from backend.app.services.graph_service import calculate_scenario_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_database():
    """Create tables if missing and seed the default Operation Golden Vineyard topology."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        existing = db.query(Scenario).filter(Scenario.name == "Operation Golden Vineyard").first()
        if existing:
            logger.info("Demo scenario 'Operation Golden Vineyard' already exists (id=%s).", existing.id)
            return existing.id

        logger.info("Seeding 'Operation Golden Vineyard' demo scenario...")

        scenario = Scenario(
            id="scenario-golden-vineyard-01",
            name="Operation Golden Vineyard",
            description="Synthetic 12-Node Multi-Tier DMZ and Internal Database Topology",
            topology_type="enterprise",
            num_nodes=12,
            num_edges=16,
            is_synthetic=True,
        )
        db.add(scenario)

        # 12 Detailed Nodes matching Specifications 98-104 + multi-tier enterprise nodes
        nodes_def = [
            # Specification Nodes 1 - 6
            {
                "id": "node-01-workstation",
                "ip": "192.168.1.10",
                "role": "workstation",
                "os": "windows",
                "cvss": 7.8,
                "asset_value": 0.2,
                "is_entry": True,
                "is_target": False,
                "hostname": "WS-FINANCE-01",
                "vulns": ["SYN-2026-1042", "SYN-2026-0819"],
                "pos_x": 80.0,
                "pos_y": 120.0,
            },
            {
                "id": "node-02-router",
                "ip": "10.0.0.1",
                "role": "router",
                "os": "firmware",
                "cvss": 4.5,
                "asset_value": 0.5,
                "is_entry": False,
                "is_target": False,
                "hostname": "RT-PERIMETER-01",
                "vulns": ["SYN-2026-3390"],
                "pos_x": 300.0,
                "pos_y": 140.0,
            },
            {
                "id": "node-03-webserver",
                "ip": "10.0.1.50",
                "role": "server",
                "os": "linux",
                "cvss": 9.2,
                "asset_value": 0.6,
                "is_entry": False,
                "is_target": False,
                "hostname": "SRV-DMZ-WEB01",
                "vulns": ["SYN-2026-7811", "SYN-2026-9021"],
                "pos_x": 520.0,
                "pos_y": 140.0,
            },
            {
                "id": "node-04-appproxy",
                "ip": "10.0.2.15",
                "role": "server",
                "os": "linux",
                "cvss": 6.1,
                "asset_value": 0.7,
                "is_entry": False,
                "is_target": False,
                "hostname": "SRV-INTERNAL-PROXY01",
                "vulns": ["SYN-2026-4412"],
                "pos_x": 740.0,
                "pos_y": 200.0,
            },
            {
                "id": "node-05-coredb",
                "ip": "10.0.3.100",
                "role": "database",
                "os": "linux",
                "cvss": 9.8,
                "asset_value": 1.0,
                "is_entry": False,
                "is_target": True,
                "hostname": "DB-ENTERPRISE-CORE",
                "vulns": ["SYN-2026-9910", "SYN-2026-8801"],
                "pos_x": 980.0,
                "pos_y": 140.0,
            },
            {
                "id": "node-06-backupvault",
                "ip": "10.0.3.105",
                "role": "database",
                "os": "linux",
                "cvss": 8.5,
                "asset_value": 0.9,
                "is_entry": False,
                "is_target": True,
                "hostname": "DB-BACKUP-VAULT",
                "vulns": ["SYN-2026-6632"],
                "pos_x": 980.0,
                "pos_y": 320.0,
            },
            # Additional Enterprise Segment Nodes (Nodes 7-12)
            {
                "id": "node-07-vpn",
                "ip": "192.168.1.1",
                "role": "firewall",
                "os": "firmware",
                "cvss": 5.0,
                "asset_value": 0.4,
                "is_entry": True,
                "is_target": False,
                "hostname": "FW-REMOTE-VPN",
                "vulns": ["SYN-2026-2104"],
                "pos_x": 80.0,
                "pos_y": 320.0,
            },
            {
                "id": "node-08-devws",
                "ip": "192.168.1.25",
                "role": "workstation",
                "os": "macos",
                "cvss": 6.4,
                "asset_value": 0.3,
                "is_entry": False,
                "is_target": False,
                "hostname": "WS-DEV-ENG01",
                "vulns": ["SYN-2026-5120"],
                "pos_x": 300.0,
                "pos_y": 340.0,
            },
            {
                "id": "node-09-ldap",
                "ip": "10.0.2.10",
                "role": "server",
                "os": "windows",
                "cvss": 7.5,
                "asset_value": 0.8,
                "is_entry": False,
                "is_target": False,
                "hostname": "DC-PRIMARY-LDAP",
                "vulns": ["SYN-2026-4011"],
                "pos_x": 520.0,
                "pos_y": 340.0,
            },
            {
                "id": "node-10-mail",
                "ip": "10.0.1.20",
                "role": "server",
                "os": "linux",
                "cvss": 5.8,
                "asset_value": 0.5,
                "is_entry": False,
                "is_target": False,
                "hostname": "SRV-DMZ-MAIL",
                "vulns": ["SYN-2026-3098"],
                "pos_x": 520.0,
                "pos_y": -40.0,
            },
            {
                "id": "node-11-ci-runner",
                "ip": "10.0.2.80",
                "role": "server",
                "os": "linux",
                "cvss": 6.8,
                "asset_value": 0.6,
                "is_entry": False,
                "is_target": False,
                "hostname": "SRV-CI-RUNNER01",
                "vulns": ["SYN-2026-7221"],
                "pos_x": 740.0,
                "pos_y": 380.0,
            },
            {
                "id": "node-12-filestore",
                "ip": "10.0.3.50",
                "role": "server",
                "os": "linux",
                "cvss": 5.2,
                "asset_value": 0.7,
                "is_entry": False,
                "is_target": False,
                "hostname": "NAS-ENTERPRISE-SHARE",
                "vulns": ["SYN-2026-1189"],
                "pos_x": 980.0,
                "pos_y": -20.0,
            },
        ]

        for nd in nodes_def:
            node = NetworkNode(
                id=nd["id"],
                scenario_id=scenario.id,
                ip_address=nd["ip"],
                role=nd["role"],
                os_type=nd["os"],
                cvss_score=nd["cvss"],
                asset_value=nd["asset_value"],
                is_entry=nd["is_entry"],
                is_target=nd["is_target"],
                hostname=nd["hostname"],
                vulnerabilities=json.dumps(nd["vulns"]),
                pos_x=nd["pos_x"],
                pos_y=nd["pos_y"],
            )
            db.add(node)

        # Directed Edges (Requirements 105: Node 1 -> 2 -> 3 -> 4 -> 5 and 4 -> 6)
        edges_def = [
            # Flagship Critical Trajectory
            ("edge-01-02", "node-01-workstation", "node-02-router", "network_access", False, False, 0.45, "80"),
            ("edge-02-03", "node-02-router", "node-03-webserver", "network_access", False, False, 0.60, "443"),
            ("edge-03-04", "node-03-webserver", "node-04-appproxy", "exploit_pivot", False, False, 0.78, "8443"),
            ("edge-04-05", "node-04-appproxy", "node-05-coredb", "database_access", False, True, 0.85, "5432"),
            ("edge-04-06", "node-04-appproxy", "node-06-backupvault", "database_access", False, True, 0.72, "3306"),
            
            # Alternative & Lateral Trajectories
            ("edge-07-02", "node-07-vpn", "node-02-router", "network_access", True, True, 0.20, "1194"),
            ("edge-07-08", "node-07-vpn", "node-08-devws", "network_access", False, False, 0.50, "22"),
            ("edge-08-09", "node-08-devws", "node-09-ldap", "credential_access", False, False, 0.65, "389"),
            ("edge-09-04", "node-09-ldap", "node-04-appproxy", "trust_relationship", False, False, 0.55, "443"),
            ("edge-09-05", "node-09-ldap", "node-05-coredb", "admin_access", True, True, 0.60, "5432"),
            ("edge-02-10", "node-02-router", "node-10-mail", "network_access", False, False, 0.30, "25"),
            ("edge-10-04", "node-10-mail", "node-04-appproxy", "internal_api", False, False, 0.40, "8080"),
            ("edge-08-11", "node-08-devws", "node-11-ci-runner", "git_push", False, False, 0.50, "22"),
            ("edge-11-06", "node-11-ci-runner", "node-06-backupvault", "backup_agent", True, True, 0.40, "3306"),
            ("edge-03-12", "node-03-webserver", "node-12-filestore", "nfs_mount", False, False, 0.35, "2049"),
            ("edge-12-05", "node-12-filestore", "node-05-coredb", "database_export", False, False, 0.45, "5432"),
        ]

        for eid, src, tgt, etype, fw, enc, anom, port in edges_def:
            edge = NetworkEdge(
                id=eid,
                scenario_id=scenario.id,
                source_id=src,
                target_id=tgt,
                edge_type=etype,
                is_firewalled=fw,
                is_encrypted=enc,
                anomaly_score=anom,
                port=port,
            )
            db.add(edge)

        audit = AuditEvent(
            scenario_id=scenario.id,
            action_type="scenario_seeded",
            details="Seeded initial 12-node Operation Golden Vineyard topology",
        )
        db.add(audit)
        db.commit()

        # Precalculate attack paths & risk scoring (Requirement 106)
        logger.info("Computing initial GNN attack paths and risk scores...")
        calculate_scenario_paths(db, scenario.id, k=5)
        logger.info("Seeding complete successfully.")
        return scenario.id

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
