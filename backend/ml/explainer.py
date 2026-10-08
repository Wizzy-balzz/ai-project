"""
explainer.py
------------
Phase 6: Explainable AI (XAI) & Proactive Mitigation Simulator.

Provides:
1. Path Explanations: Plain-English kill-chain narratives, step-by-step hop
   audits, and quantitative factor attributions (GNN vs CVSS vs Edge Controls).
2. Chokepoint Identification: Identifies graph pivot nodes and edges where
   multiple high-risk attack paths converge.
3. Actionable Mitigations: Generates prioritized defensive recommendations
   (patch CVE, deploy firewall, close open service, isolate node).
4. Counterfactual What-If Simulation: Simulates applying a defensive mitigation
   and computes the exact risk reduction and severed attack paths.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from backend.ml.path_models import CandidateAttackPath
from backend.ml.risk_scorer import RiskScorer, ScoredAttackPath


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class HopExplanation:
    """Step-by-step breakdown of a single edge hop in an attack path."""
    step: int
    source_id: str
    source_label: str
    target_id: str
    target_label: str
    edge_type: str
    has_firewall: bool
    is_encrypted: bool
    attack_weight: float
    anomaly_count: int
    exploited_weaknesses: List[str]
    traversal_mechanism: str

    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "source_id": self.source_id,
            "source_label": self.source_label,
            "target_id": self.target_id,
            "target_label": self.target_label,
            "edge_type": self.edge_type,
            "has_firewall": self.has_firewall,
            "is_encrypted": self.is_encrypted,
            "attack_weight": round(self.attack_weight, 4),
            "anomaly_count": self.anomaly_count,
            "exploited_weaknesses": self.exploited_weaknesses,
            "traversal_mechanism": self.traversal_mechanism,
        }


@dataclass
class PathExplanation:
    """Complete XAI explanation for a scored attack path."""
    path_id: str
    path_summary: str
    risk_score: float
    risk_tier: str
    factor_breakdown: Dict[str, float]
    hop_explanations: List[HopExplanation]
    weakest_node: str
    weakest_edge: Tuple[str, str]

    def to_dict(self) -> Dict:
        return {
            "path_id": self.path_id,
            "path_summary": self.path_summary,
            "risk_score": round(self.risk_score, 4),
            "risk_tier": self.risk_tier,
            "factor_breakdown": {k: round(v, 4) for k, v in self.factor_breakdown.items()},
            "hop_explanations": [h.to_dict() for h in self.hop_explanations],
            "weakest_node": self.weakest_node,
            "weakest_edge": list(self.weakest_edge),
        }


@dataclass
class CriticalChokepoint:
    """Network node or edge where high-risk attack paths intersect."""
    element_type: str              # "node" or "edge"
    element_id: str                # node_id or "u->v"
    element_label: str
    intersected_paths_count: int
    intersected_path_ids: List[str]
    max_path_risk: float

    def to_dict(self) -> Dict:
        return {
            "element_type": self.element_type,
            "element_id": self.element_id,
            "element_label": self.element_label,
            "intersected_paths_count": self.intersected_paths_count,
            "intersected_path_ids": self.intersected_path_ids,
            "max_path_risk": round(self.max_path_risk, 4),
        }


@dataclass
class MitigationAction:
    """Actionable defensive recommendation."""
    mitigation_id: str
    title: str
    action_type: str               # "deploy_firewall", "patch_vulnerability", "isolate_node"
    target_element: str            # node_id or "u->v"
    target_label: str
    description: str
    priority: str                  # "IMMEDIATE", "HIGH", "MEDIUM"
    severed_paths_count: int
    severed_path_ids: List[str]
    estimated_risk_reduction_pct: float

    def to_dict(self) -> Dict:
        return {
            "mitigation_id": self.mitigation_id,
            "title": self.title,
            "action_type": self.action_type,
            "target_element": self.target_element,
            "target_label": self.target_label,
            "description": self.description,
            "priority": self.priority,
            "severed_paths_count": self.severed_paths_count,
            "severed_path_ids": self.severed_path_ids,
            "estimated_risk_reduction_pct": round(self.estimated_risk_reduction_pct, 2),
        }


# ---------------------------------------------------------------------------
# Explainer Engine
# ---------------------------------------------------------------------------

class PathExplainer:
    """Generates explanations, identifies bottlenecks, and simulates defenses."""

    def __init__(self, scorer: Optional[RiskScorer] = None) -> None:
        self.scorer = scorer or RiskScorer()

    def explain_path(
        self,
        scored_path: ScoredAttackPath,
        G: nx.DiGraph,
    ) -> PathExplanation:
        """Generate human-readable and quantitative explanations for a path."""
        c = scored_path.candidate
        hops = len(c.edge_pairs)

        # Factor attribution
        tot = scored_path.node_score + scored_path.edge_score + scored_path.vuln_score
        if tot > 0:
            factor_weights = {
                "gnn_node_likelihood": scored_path.node_score / tot,
                "edge_traversal_exploitability": scored_path.edge_score / tot,
                "cvss_vulnerability_severity": scored_path.vuln_score / tot,
                "target_asset_criticality": scored_path.impact_score,
            }
        else:
            factor_weights = {
                "gnn_node_likelihood": 0.33,
                "edge_traversal_exploitability": 0.33,
                "cvss_vulnerability_severity": 0.34,
                "target_asset_criticality": scored_path.impact_score,
            }

        # Hop-by-hop breakdown
        hop_exps: List[HopExplanation] = []
        max_edge_risk = -1.0
        weakest_edge = ("", "")

        for i, (u, v) in enumerate(c.edge_pairs):
            ef = c.edge_features[i] if i < len(c.edge_features) else [0.0] * 9
            u_label = G.nodes[u].get("label", u) if u in G else u
            v_label = G.nodes[v].get("label", v) if v in G else v

            # Edge attributes
            edge_data = G[u][v] if G.has_edge(u, v) else {}
            has_fw = bool(ef[4] > 0.5) if len(ef) > 4 else False
            is_enc = bool(ef[5] > 0.5) if len(ef) > 5 else False
            atk_wt = ef[6] if len(ef) > 6 else 0.5
            anom = int(ef[7]) if len(ef) > 7 else 0

            # Determine weaknesses
            weaknesses = []
            target_node_data = G.nodes.get(v, {})
            max_cvss = target_node_data.get("max_cvss", 0.0)
            if max_cvss >= 7.0:
                weaknesses.append(f"Target node has CVSS {max_cvss} vulnerability")
            if not has_fw:
                weaknesses.append("Missing firewall control on link")
            if not is_enc:
                weaknesses.append("Unencrypted network channel")
            if anom > 0:
                weaknesses.append(f"Active telemetry anomaly ({anom} anomalous flows)")

            mech = f"Traverse from {u_label} to {v_label}"
            if not has_fw and not is_enc:
                mech += " via uninspected, unencrypted pathway"
            elif not has_fw:
                mech += " without perimeter firewall filtering"

            hop_exps.append(
                HopExplanation(
                    step=i + 1,
                    source_id=u,
                    source_label=u_label,
                    target_id=v,
                    target_label=v_label,
                    edge_type=edge_data.get("edge_type", "network_link"),
                    has_firewall=has_fw,
                    is_encrypted=is_enc,
                    attack_weight=atk_wt,
                    anomaly_count=anom,
                    exploited_weaknesses=weaknesses,
                    traversal_mechanism=mech,
                )
            )

            if atk_wt > max_edge_risk:
                max_edge_risk = atk_wt
                weakest_edge = (u, v)

        # Identify weakest node (highest CVSS or highest GNN prob)
        weakest_node = c.node_ids[0]
        max_node_cvss = -1.0
        for nid in c.node_ids:
            cvss = G.nodes.get(nid, {}).get("max_cvss", 0.0)
            if cvss > max_node_cvss:
                max_node_cvss = cvss
                weakest_node = nid

        # Summary narrative
        entry_lbl = c.node_labels[0]
        term_lbl = c.node_labels[-1]
        crit_str = "critical asset" if c.reaches_critical_asset else "internal target"
        summary = (
            f"Attack originates at entry point '{entry_lbl}', traverses {hops} network hop(s), "
            f"and targets {crit_str} '{term_lbl}' with {scored_path.risk_tier} risk "
            f"(score: {scored_path.risk_score:.2f})."
        )

        return PathExplanation(
            path_id=c.path_id,
            path_summary=summary,
            risk_score=scored_path.risk_score,
            risk_tier=scored_path.risk_tier,
            factor_breakdown=factor_weights,
            hop_explanations=hop_exps,
            weakest_node=weakest_node,
            weakest_edge=weakest_edge,
        )

    def find_chokepoints(
        self,
        top_paths: List[ScoredAttackPath],
        G: Optional[nx.DiGraph] = None,
    ) -> List[CriticalChokepoint]:
        """Find nodes and edges that intersect the largest number of high-risk paths."""
        if not top_paths:
            return []

        node_counts: Dict[str, List[ScoredAttackPath]] = {}
        edge_counts: Dict[Tuple[str, str], List[ScoredAttackPath]] = {}

        for sp in top_paths:
            # Nodes
            for nid in sp.node_ids:
                node_counts.setdefault(nid, []).append(sp)
            # Edges
            for edge in sp.candidate.edge_pairs:
                edge_counts.setdefault(edge, []).append(sp)

        chokepoints: List[CriticalChokepoint] = []

        # Add node chokepoints (excluding entry points if intermediate exist)
        for nid, paths in node_counts.items():
            if len(paths) >= 1:
                label = (
                    G.nodes[nid].get("label", nid)
                    if G and nid in G.nodes
                    else nid
                )
                max_r = max(p.risk_score for p in paths)
                chokepoints.append(
                    CriticalChokepoint(
                        element_type="node",
                        element_id=nid,
                        element_label=label,
                        intersected_paths_count=len(paths),
                        intersected_path_ids=[p.path_id for p in paths],
                        max_path_risk=max_r,
                    )
                )

        # Add edge chokepoints
        for (u, v), paths in edge_counts.items():
            if len(paths) >= 1:
                u_lbl = G.nodes[u].get("label", u) if G and u in G.nodes else u
                v_lbl = G.nodes[v].get("label", v) if G and v in G.nodes else v
                max_r = max(p.risk_score for p in paths)
                chokepoints.append(
                    CriticalChokepoint(
                        element_type="edge",
                        element_id=f"{u}->{v}",
                        element_label=f"{u_lbl} → {v_lbl}",
                        intersected_paths_count=len(paths),
                        intersected_path_ids=[p.path_id for p in paths],
                        max_path_risk=max_r,
                    )
                )

        # Sort chokepoints: most intersected paths descending, then max risk
        chokepoints.sort(
            key=lambda cp: (-cp.intersected_paths_count, -cp.max_path_risk)
        )
        return chokepoints

    def generate_mitigations(
        self,
        top_paths: List[ScoredAttackPath],
        G: nx.DiGraph,
    ) -> List[MitigationAction]:
        """Generate ranked mitigation actions that sever or reduce high-risk attack paths."""
        if not top_paths:
            return []

        chokepoints = self.find_chokepoints(top_paths, G)
        mitigations: List[MitigationAction] = []
        total_top_paths = len(top_paths)

        seen_targets: Set[str] = set()

        for cp in chokepoints:
            if cp.element_id in seen_targets:
                continue
            seen_targets.add(cp.element_id)

            severed = cp.intersected_paths_count
            reduction_pct = (severed / total_top_paths) * 100.0

            if cp.element_type == "edge":
                u, v = cp.element_id.split("->")
                u_lbl = G.nodes[u].get("label", u) if u in G.nodes else u
                v_lbl = G.nodes[v].get("label", v) if v in G.nodes else v

                priority = "IMMEDIATE" if reduction_pct >= 50.0 else "HIGH"
                mitigations.append(
                    MitigationAction(
                        mitigation_id=f"mit-fw-{uuid.uuid4().hex[:6]}",
                        title=f"Deploy Firewall Inspection on {u_lbl} → {v_lbl}",
                        action_type="deploy_firewall",
                        target_element=cp.element_id,
                        target_label=cp.element_label,
                        description=(
                            f"Enforce egress filtering and IPS between {u_lbl} and {v_lbl} "
                            f"to sever {severed} attack path(s)."
                        ),
                        priority=priority,
                        severed_paths_count=severed,
                        severed_path_ids=cp.intersected_path_ids,
                        estimated_risk_reduction_pct=reduction_pct,
                    )
                )

            elif cp.element_type == "node":
                node_data = G.nodes.get(cp.element_id, {})
                max_cvss = node_data.get("max_cvss", 0.0)
                is_crit = bool(node_data.get("is_critical_asset", 0))

                if max_cvss >= 7.0:
                    priority = "IMMEDIATE" if max_cvss >= 9.0 else "HIGH"
                    mitigations.append(
                        MitigationAction(
                            mitigation_id=f"mit-patch-{uuid.uuid4().hex[:6]}",
                            title=f"Patch High/Critical CVEs on {cp.element_label}",
                            action_type="patch_vulnerability",
                            target_element=cp.element_id,
                            target_label=cp.element_label,
                            description=(
                                f"Remediate CVSS {max_cvss} vulnerability on {cp.element_label}. "
                                f"Hardens pivot node across {severed} attack path(s)."
                            ),
                            priority=priority,
                            severed_paths_count=severed,
                            severed_path_ids=cp.intersected_path_ids,
                            estimated_risk_reduction_pct=reduction_pct * 0.8,
                        )
                    )
                elif not is_crit:
                    mitigations.append(
                        MitigationAction(
                            mitigation_id=f"mit-isolate-{uuid.uuid4().hex[:6]}",
                            title=f"Segment/Isolate Pivot Node {cp.element_label}",
                            action_type="isolate_node",
                            target_element=cp.element_id,
                            target_label=cp.element_label,
                            description=(
                                f"Restrict lateral traversal through pivot node {cp.element_label}. "
                                f"Protects downstream assets on {severed} attack path(s)."
                            ),
                            priority="HIGH" if severed >= 2 else "MEDIUM",
                            severed_paths_count=severed,
                            severed_path_ids=cp.intersected_path_ids,
                            estimated_risk_reduction_pct=reduction_pct,
                        )
                    )

        # Rank mitigations by severed paths descending
        mitigations.sort(
            key=lambda m: (-m.severed_paths_count, -m.estimated_risk_reduction_pct)
        )
        return mitigations[:6]

    def simulate_mitigation(
        self,
        candidate_paths: List[CandidateAttackPath],
        mitigation: MitigationAction,
    ) -> Dict:
        """
        Simulate applying a mitigation action.

        Returns:
            Dict containing before/after comparison and severed paths.
        """
        orig_scored = [self.scorer.score_path(c) for c in candidate_paths]
        orig_risk = max((p.risk_score for p in orig_scored), default=0.0)

        severed_ids = set()
        remaining_candidates: List[CandidateAttackPath] = []

        target = mitigation.target_element

        for c in candidate_paths:
            sever = False
            if mitigation.action_type in ("deploy_firewall", "sever_edge"):
                if "->" in target:
                    u, v = target.split("->")
                    if (u, v) in c.edge_pairs:
                        sever = True
            elif mitigation.action_type in ("isolate_node", "patch_vulnerability"):
                if target in c.node_ids:
                    # Isolating or removing vulnerability breaks traversal
                    sever = True

            if sever:
                severed_ids.add(c.path_id)
            else:
                remaining_candidates.append(c)

        # Recompute risk with remaining candidates
        rem_scored = [self.scorer.score_path(c) for c in remaining_candidates]
        new_risk = max((p.risk_score for p in rem_scored), default=0.0)

        risk_drop_pct = (
            ((orig_risk - new_risk) / orig_risk * 100.0) if orig_risk > 0 else 0.0
        )

        return {
            "mitigation_id": mitigation.mitigation_id,
            "title": mitigation.title,
            "target_element": mitigation.target_element,
            "original_paths_count": len(candidate_paths),
            "remaining_paths_count": len(remaining_candidates),
            "severed_paths_count": len(severed_ids),
            "severed_path_ids": sorted(severed_ids),
            "original_network_risk": round(orig_risk, 4),
            "new_network_risk": round(new_risk, 4),
            "risk_reduction_pct": round(max(0.0, risk_drop_pct), 2),
        }
