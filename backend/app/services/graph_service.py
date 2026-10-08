"""Graph service integrating database models with GNN inference, path generation, risk scoring, and XAI."""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple
import networkx as nx

from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.models.scenario import Scenario
from backend.app.models.node import NetworkNode
from backend.app.models.edge import NetworkEdge
from backend.app.models.attack_path import AttackPath
from backend.app.models.audit import AuditEvent
from backend.app.schemas.graph import (
    ReactFlowNode,
    ReactFlowEdge,
    Position,
    NodeData,
    EdgeData,
    GraphTopologyResponse,
)
from backend.app.schemas.path import (
    AttackTrajectoryResponse,
    TopKPathsResponse,
    HopDetail,
    FactorAttribution,
)
from backend.app.schemas.simulation import SimulationRequest, SimulationResult
from backend.core.data_models import (
    NetworkTopology,
    NetworkNode as CoreNode,
    NetworkEdge as CoreEdge,
    NodeType,
    OSType,
    EdgeType,
    Vulnerability,
    VulnerabilitySeverity,
)
from backend.core.graph_builder import GraphBuilder
from backend.ml.path_generator import PathGenerator, PathConfig
from backend.ml.risk_scorer import RiskScorer, RiskScoringConfig, ScoredAttackPath
from backend.ml.explainer import PathExplainer

logger = logging.getLogger(__name__)

# Global singletons for ML components
_path_generator: Optional[PathGenerator] = None
_risk_scorer: Optional[RiskScorer] = None
_explainer: Optional[PathExplainer] = None


def get_path_generator() -> PathGenerator:
    """Retrieve or initialize the singleton PathGenerator with GNN checkpoint."""
    global _path_generator
    if _path_generator is None:
        ckpt = settings.best_checkpoint_path if settings.best_checkpoint_path.exists() else None
        config = PathConfig(max_path_length=7, min_path_length=2, max_paths_per_entry=150, max_total_paths=500)
        _path_generator = PathGenerator(config=config, checkpoint_path=ckpt)
    return _path_generator


def get_risk_scorer() -> RiskScorer:
    """Retrieve or initialize the deterministic RiskScorer."""
    global _risk_scorer
    if _risk_scorer is None:
        cfg = RiskScoringConfig(w_node=0.35, w_edge=0.35, w_vuln=0.30, length_decay=0.95, top_k=8)
        _risk_scorer = RiskScorer(config=cfg)
    return _risk_scorer


def get_explainer() -> PathExplainer:
    """Retrieve or initialize the XAI Explainer."""
    global _explainer
    if _explainer is None:
        _explainer = PathExplainer(scorer=get_risk_scorer())
    return _explainer


def _parse_role(role_str: str) -> NodeType:
    clean = (role_str or "").strip().lower()
    mapping = {
        "workstation": NodeType.WORKSTATION,
        "server": NodeType.SERVER,
        "router": NodeType.ROUTER,
        "firewall": NodeType.FIREWALL,
        "database": NodeType.DATABASE,
        "internet": NodeType.INTERNET,
        "iot_device": NodeType.IOT_DEVICE,
        "critical_asset": NodeType.CRITICAL_ASSET,
    }
    return mapping.get(clean, NodeType.SERVER)


def _parse_os(os_str: str) -> OSType:
    clean = (os_str or "").strip().lower()
    mapping = {
        "windows": OSType.WINDOWS,
        "linux": OSType.LINUX,
        "macos": OSType.MACOS,
        "firmware": OSType.FIRMWARE,
    }
    return mapping.get(clean, OSType.LINUX)


def db_to_network_topology(
    db: Session,
    scenario_id: str,
    exclude_node_ids: Optional[set[str]] = None,
    exclude_edge_ids: Optional[set[str]] = None,
    patched_cves: Optional[set[str]] = None,
) -> Tuple[NetworkTopology, Scenario]:
    """Convert SQLite models for a scenario into an internal NetworkTopology dataclass."""
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        raise ValueError(f"Scenario not found: {scenario_id}")

    exclude_nodes = exclude_node_ids or set()
    exclude_edges = exclude_edge_ids or set()
    patched = patched_cves or set()

    db_nodes = (
        db.query(NetworkNode)
        .filter(NetworkNode.scenario_id == scenario_id)
        .all()
    )
    db_edges = (
        db.query(NetworkEdge)
        .filter(NetworkEdge.scenario_id == scenario_id)
        .all()
    )

    core_nodes: List[Node] = []
    for dn in db_nodes:
        if dn.id in exclude_nodes:
            continue

        # Parse vulnerabilities
        vulns: List[Vulnerability] = []
        try:
            cve_list = json.loads(dn.vulnerabilities) if dn.vulnerabilities else []
        except Exception:
            cve_list = []

        cvss = dn.cvss_score
        for cve_name in cve_list:
            if cve_name in patched:
                continue
            vulns.append(
                Vulnerability(
                    cve_id=cve_name,
                    cvss_score=cvss if cvss > 0 else 7.5,
                    severity=Vulnerability.severity_from_cvss(cvss if cvss > 0 else 7.5),
                    description=f"Synthetic vulnerability {cve_name} on {dn.ip_address}",
                    exploitability_score=min(3.9, (cvss / 10.0) * 3.5),
                    impact_score=min(6.0, (cvss / 10.0) * 5.5),
                )
            )

        if not vulns and cvss > 0:
            cve_name = f"SYN-2026-{abs(hash(dn.id)) % 8999 + 1000}"
            if cve_name not in patched:
                vulns.append(
                    Vulnerability(
                        cve_id=cve_name,
                        cvss_score=cvss,
                        severity=Vulnerability.severity_from_cvss(cvss),
                        description=f"Synthetic CVE on {dn.ip_address}",
                        exploitability_score=min(3.9, (cvss / 10.0) * 3.5),
                        impact_score=min(6.0, (cvss / 10.0) * 5.5),
                    )
                )

        core_node = CoreNode(
            node_id=dn.id,
            label=dn.hostname or dn.ip_address,
            node_type=_parse_role(dn.role),
            os_type=_parse_os(dn.os_type),
            ip_address=dn.ip_address,
            subnet="10.0.0.0/24",
            services=["https:443", "ssh:22"] if dn.role != "workstation" else ["rdp:3389"],
            vulnerabilities=vulns,
            is_entry_point=dn.is_entry,
            is_critical_asset=dn.is_target,
            compromised=False,
            asset_value=dn.asset_value,
        )
        core_nodes.append(core_node)

    valid_node_ids = {n.node_id for n in core_nodes}

    core_edges: List[CoreEdge] = []
    for de in db_edges:
        if de.id in exclude_edges:
            continue
        if de.source_id not in valid_node_ids or de.target_id not in valid_node_ids:
            continue

        core_edge = CoreEdge(
            edge_id=de.id,
            source_id=de.source_id,
            target_id=de.target_id,
            edge_type=EdgeType.EXPLOIT_PATH if "exploit" in de.edge_type else EdgeType.NETWORK_LINK,
            bidirectional=False,
            port=int(de.port) if de.port.isdigit() else 443,
            protocol="tcp",
            firewall_rule=de.is_firewalled,
            is_encrypted=de.is_encrypted,
            attack_weight=0.85 if not de.is_firewalled else 0.40,
        )
        core_edges.append(core_edge)

    topology = NetworkTopology(
        topology_id=scenario.id,
        name=scenario.name,
        nodes={n.node_id: n for n in core_nodes},
        edges={e.edge_id: e for e in core_edges},
        is_synthetic=scenario.is_synthetic,
    )
    return topology, scenario


def calculate_scenario_paths(
    db: Session,
    scenario_id: str,
    k: int = 5,
) -> TopKPathsResponse:
    """Generate, risk-score, and store candidate attack trajectories."""
    topology, scenario = db_to_network_topology(db, scenario_id)
    if not topology.nodes:
        return TopKPathsResponse(
            scenario_id=scenario_id,
            total_candidates=0,
            top_k_count=0,
            paths=[],
            max_risk_score=0.0,
            average_risk_score=0.0,
            critical_count=0,
            high_count=0,
        )

    generator = get_path_generator()
    scorer = get_risk_scorer()
    explainer = get_explainer()

    # Generate candidate paths using bounded DFS & GNN probabilities
    candidates, summary = generator.generate(topology)
    
    # Deterministic risk scoring
    ranked_paths: List[ScoredAttackPath] = scorer.rank_paths(candidates, top_k=k)

    # Build node lookup for labels and roles
    node_map = topology.nodes

    G = GraphBuilder().build(topology)
    response_paths: List[AttackTrajectoryResponse] = []
    
    # Clear old paths and insert updated paths in database
    db.query(AttackPath).filter(AttackPath.scenario_id == scenario_id).delete()

    for idx, scored in enumerate(ranked_paths):
        cand = scored.candidate
        node_ids = cand.node_ids
        node_ips = [node_map[nid].ip_address if nid in node_map else nid for nid in node_ids]
        node_roles = [node_map[nid].node_type.value if nid in node_map else "server" for nid in node_ids]

        # Generate XAI narrative and hop breakdown
        path_explanation = explainer.explain_path(scored, G)

        hop_details: List[HopDetail] = []
        for h_idx in range(len(node_ids) - 1):
            src_id = node_ids[h_idx]
            tgt_id = node_ids[h_idx + 1]
            src_node = node_map.get(src_id)
            tgt_node = node_map.get(tgt_id)

            hop_details.append(
                HopDetail(
                    step_number=h_idx + 1,
                    source_id=src_id,
                    source_ip=src_node.ip_address if src_node else src_id,
                    source_role=src_node.node_type.value if src_node else "node",
                    target_id=tgt_id,
                    target_ip=tgt_node.ip_address if tgt_node else tgt_id,
                    target_role=tgt_node.node_type.value if tgt_node else "node",
                    edge_type="network_access",
                    vulnerability_cve=tgt_node.vulnerabilities[0].cve_id if tgt_node and tgt_node.vulnerabilities else "SYN-2026-1042",
                    cvss_score=tgt_node.max_cvss if tgt_node else 7.5,
                    step_likelihood=round(scored.node_score, 3),
                    action_narrative=f"Pivot from {src_node.label if src_node else src_id} to {tgt_node.label if tgt_node else tgt_id} via port 443.",
                )
            )

        attr = FactorAttribution(
            gnn_weight=0.35,
            edge_weight=0.35,
            cvss_severity=0.30,
            node_susceptibility=round(scored.node_score, 3),
            edge_permeability=round(scored.edge_score, 3),
            vulnerability_exploitability=round(scored.vuln_score, 3),
        )

        traj_resp = AttackTrajectoryResponse(
            id=f"path-{scenario_id[:8]}-{idx+1:03d}",
            scenario_id=scenario_id,
            node_ids=node_ids,
            node_ips=node_ips,
            node_roles=node_roles,
            hops=len(node_ids) - 1,
            likelihood=round(scored.likelihood_score, 4),
            impact=round(scored.impact_score, 4),
            risk_score=round(scored.risk_score, 4),
            severity=scored.risk_tier,
            hop_details=hop_details,
            factor_attribution=attr,
            kill_chain_narrative=path_explanation.path_summary,
            chokepoints=node_ids[1:-1] if len(node_ids) > 2 else [],
        )
        response_paths.append(traj_resp)

        # Store in DB
        db_path = AttackPath(
            id=traj_resp.id,
            scenario_id=scenario_id,
            path_sequence=json.dumps(node_ids),
            hops=traj_resp.hops,
            likelihood=traj_resp.likelihood,
            impact=traj_resp.impact,
            risk_score=traj_resp.risk_score,
            severity=traj_resp.severity,
        )
        db.add(db_path)

    db.commit()

    max_risk = max((p.risk_score for p in response_paths), default=0.0)
    avg_risk = sum((p.risk_score for p in response_paths)) / len(response_paths) if response_paths else 0.0
    critical_cnt = sum(1 for p in response_paths if p.severity == "CRITICAL")
    high_cnt = sum(1 for p in response_paths if p.severity == "HIGH")

    return TopKPathsResponse(
        scenario_id=scenario_id,
        total_candidates=len(candidates),
        top_k_count=len(response_paths),
        paths=response_paths,
        max_risk_score=round(max_risk, 4),
        average_risk_score=round(avg_risk, 4),
        critical_count=critical_cnt,
        high_count=high_cnt,
    )


def get_react_flow_topology(
    db: Session,
    scenario_id: str,
    active_path_node_ids: Optional[list[str]] = None,
    isolated_node_ids: Optional[list[str]] = None,
    blocked_edge_ids: Optional[list[str]] = None,
) -> GraphTopologyResponse:
    """Format nodes and edges into @xyflow/react direct specifications with automatic layered layout."""
    scenario = db.query(Scenario).filter(Scenario.id == scenario_id).first()
    if not scenario:
        raise ValueError(f"Scenario not found: {scenario_id}")

    db_nodes = (
        db.query(NetworkNode)
        .filter(NetworkNode.scenario_id == scenario_id)
        .all()
    )
    db_edges = (
        db.query(NetworkEdge)
        .filter(NetworkEdge.scenario_id == scenario_id)
        .all()
    )

    active_set = set(active_path_node_ids or [])
    active_edges: set[Tuple[str, str]] = set()
    if active_path_node_ids and len(active_path_node_ids) >= 2:
        for i in range(len(active_path_node_ids) - 1):
            active_edges.add((active_path_node_ids[i], active_path_node_ids[i + 1]))

    isolated_set = set(isolated_node_ids or [])
    blocked_set = set(blocked_edge_ids or [])

    # Layout generation: partition nodes into tiers (entry -> intermediate -> targets)
    tier_entry: List[NetworkNode] = []
    tier_internal: List[NetworkNode] = []
    tier_target: List[NetworkNode] = []

    for n in db_nodes:
        if n.is_entry:
            tier_entry.append(n)
        elif n.is_target:
            tier_target.append(n)
        else:
            tier_internal.append(n)

    flow_nodes: List[ReactFlowNode] = []

    def place_tier(nodes: List[NetworkNode], col_x: float, start_y: float = 120.0, gap_y: float = 160.0):
        for idx, n in enumerate(nodes):
            x = n.pos_x if n.pos_x and n.pos_x != 100.0 else col_x
            y = n.pos_y if n.pos_y and n.pos_y != 100.0 else (start_y + idx * gap_y)

            try:
                vuln_list = json.loads(n.vulnerabilities) if n.vulnerabilities else []
            except Exception:
                vuln_list = []

            threat_level = "CRITICAL" if n.cvss_score >= 8.5 else ("HIGH" if n.cvss_score >= 6.5 else ("MEDIUM" if n.cvss_score >= 4.0 else "LOW"))

            flow_nodes.append(
                ReactFlowNode(
                    id=n.id,
                    type="customNode",
                    position=Position(x=x, y=y),
                    data=NodeData(
                        id=n.id,
                        scenario_id=scenario_id,
                        ip=n.ip_address,
                        role=n.role,
                        os=n.os_type,
                        cvss=n.cvss_score,
                        asset_value=n.asset_value,
                        is_entry=n.is_entry,
                        is_target=n.is_target,
                        hostname=n.hostname or n.ip_address,
                        vulnerabilities=vuln_list,
                        gnn_probability=round(0.40 + (n.cvss_score / 10.0) * 0.55, 3),
                        lateral_velocity=round(0.30 + (n.asset_value * 0.65), 3),
                        threat_level=threat_level,
                        in_critical_path=n.id in active_set,
                        is_isolated=n.id in isolated_set,
                    ),
                )
            )

    place_tier(tier_entry, col_x=100.0)
    place_tier(tier_internal, col_x=500.0)
    place_tier(tier_target, col_x=900.0)

    flow_edges: List[ReactFlowEdge] = []
    for e in db_edges:
        is_in_path = (e.source_id, e.target_id) in active_edges
        is_blocked = e.id in blocked_set

        flow_edges.append(
            ReactFlowEdge(
                id=e.id,
                source=e.source_id,
                target=e.target_id,
                type="customEdge",
                animated=is_in_path and not is_blocked,
                data=EdgeData(
                    id=e.id,
                    scenario_id=scenario_id,
                    source_id=e.source_id,
                    target_id=e.target_id,
                    edge_type=e.edge_type,
                    is_firewalled=e.is_firewalled,
                    is_encrypted=e.is_encrypted,
                    anomaly_score=e.anomaly_score,
                    port=e.port,
                    in_critical_path=is_in_path,
                    is_blocked=is_blocked,
                ),
            )
        )

    stats = {
        "num_nodes": len(flow_nodes),
        "num_edges": len(flow_edges),
        "entry_points": len(tier_entry),
        "critical_targets": len(tier_target),
    }

    return GraphTopologyResponse(
        scenario_id=scenario_id,
        scenario_name=scenario.name,
        nodes=flow_nodes,
        edges=flow_edges,
        stats=stats,
    )


def simulate_what_if_defense(
    db: Session,
    scenario_id: str,
    req: SimulationRequest,
) -> SimulationResult:
    """Execute counterfactual What-If simulation measuring severed paths and risk reduction."""
    # 1. Baseline analysis
    base_res = calculate_scenario_paths(db, scenario_id, k=10)
    base_risk = base_res.max_risk_score
    base_paths_count = base_res.total_candidates

    # 2. Counterfactual evaluation
    topology_sim, scenario = db_to_network_topology(
        db,
        scenario_id,
        exclude_node_ids=set(req.isolated_node_ids),
        exclude_edge_ids=set(req.blocked_edge_ids),
        patched_cves=set(req.patched_cves),
    )

    generator = get_path_generator()
    scorer = get_risk_scorer()

    sim_candidates, _ = generator.generate(topology_sim)
    sim_ranked = scorer.rank_paths(sim_candidates, top_k=10)

    sim_risk = sim_ranked[0].risk_score if sim_ranked else 0.0
    sim_paths_count = len(sim_candidates)

    severed_count = max(0, base_paths_count - sim_paths_count)
    if base_risk > 0.0:
        drop_pct = max(0.0, min(100.0, ((base_risk - sim_risk) / base_risk) * 100.0))
    else:
        drop_pct = 100.0 if sim_risk == 0.0 else 0.0

    active_defenses = []
    for nid in req.isolated_node_ids:
        active_defenses.append({"type": "isolate_node", "target_id": nid, "status": "active"})
    for eid in req.blocked_edge_ids:
        active_defenses.append({"type": "block_edge", "target_id": eid, "status": "active"})
    for cve in req.patched_cves:
        active_defenses.append({"type": "patch_cve", "target_id": cve, "status": "applied"})

    rec = "Defense plan successfully neutralizes primary exploit trajectories." if drop_pct > 30 else "Apply additional firewall block rules to sever remaining lateral movement routes."

    # Audit log
    audit = AuditEvent(
        scenario_id=scenario_id,
        action_type="what_if_simulation",
        details=json.dumps({
            "isolated": req.isolated_node_ids,
            "blocked": req.blocked_edge_ids,
            "patched": req.patched_cves,
            "risk_drop_pct": round(drop_pct, 2),
        }),
    )
    db.add(audit)
    db.commit()

    return SimulationResult(
        scenario_id=scenario_id,
        original_risk=round(base_risk, 4),
        simulated_risk=round(sim_risk, 4),
        risk_drop_percentage=round(drop_pct, 2),
        total_original_paths=base_paths_count,
        severed_paths_count=severed_count,
        remaining_paths_count=sim_paths_count,
        active_defenses=active_defenses,
        recommendation=rec,
        is_mitigated=drop_pct >= 50.0 or sim_risk < 0.35,
    )
