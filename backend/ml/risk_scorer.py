"""
risk_scorer.py
--------------
Phase 5: Attack Path Risk Scoring & Top-K Ranking.

Cybersecurity Risk Formula:
    Risk = Likelihood × Impact

Where:
    1. Likelihood:
       - Node Compromise Likelihood (S_node): Blended GNN attack probability
         (70% path mean + 30% bottleneck node minimum).
       - Edge Traversal Exploitability (S_edge): Edge attack weight adjusted for
         firewall presence (penalty 0.6x), encryption (penalty 0.8x),
         exploit path indicator (bonus 1.2x), and anomaly activity.
       - Vulnerability Severity (S_vuln): Mean & max CVSS normalized to [0, 1].
       - Length Decay: Sequential multi-hop execution discount (0.95^(len - 1)).
       Likelihood = (0.35 * S_node + 0.35 * S_edge + 0.30 * S_vuln) * Length_Decay

    2. Impact:
       - Critical Asset Terminal: 1.0 baseline (high business impact).
       - Non-Critical Terminal: scaled by terminal asset_value (0.3 - 0.5 default).
       - Impact = 0.40 + 0.60 * target_criticality

    Composite Risk Score:
       Risk_Score = clamp(Likelihood * Impact, 0.0, 1.0)

Risk Tiers:
    - CRITICAL: >= 0.70
    - HIGH:     >= 0.45 and < 0.70
    - MEDIUM:   >= 0.25 and < 0.45
    - LOW:      < 0.25
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.ml.path_models import CandidateAttackPath


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskScoringConfig:
    """Configuration for path-level cybersecurity risk scoring."""
    w_node: float = 0.35
    w_edge: float = 0.35
    w_vuln: float = 0.30
    length_decay: float = 0.95
    firewall_penalty: float = 0.60
    encryption_penalty: float = 0.80
    exploit_type_multiplier: float = 1.20
    top_k: int = 5

    def __post_init__(self) -> None:
        total_w = self.w_node + self.w_edge + self.w_vuln
        if abs(total_w - 1.0) > 1e-4:
            raise ValueError(f"Weights must sum to 1.0, got {total_w}")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ScoredAttackPath:
    """A candidate attack path augmented with Phase 5 risk metrics."""
    candidate: CandidateAttackPath
    risk_score: float              # [0.0, 1.0]
    risk_tier: str                 # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    likelihood_score: float        # [0.0, 1.0]
    impact_score: float            # [0.0, 1.0]
    node_score: float              # [0.0, 1.0]
    edge_score: float              # [0.0, 1.0]
    vuln_score: float              # [0.0, 1.0]

    @property
    def path_id(self) -> str:
        return self.candidate.path_id

    @property
    def node_ids(self) -> List[str]:
        return self.candidate.node_ids

    @property
    def node_labels(self) -> List[str]:
        return self.candidate.node_labels

    @property
    def path_length(self) -> int:
        return self.candidate.path_length

    @property
    def entry_node(self) -> str:
        return self.candidate.entry_node

    @property
    def terminal_node(self) -> str:
        return self.candidate.terminal_node

    @property
    def reaches_critical_asset(self) -> bool:
        return self.candidate.reaches_critical_asset

    def to_dict(self) -> Dict:
        c_dict = self.candidate.to_dict()
        c_dict.update({
            "risk_score": round(self.risk_score, 4),
            "risk_tier": self.risk_tier,
            "likelihood_score": round(self.likelihood_score, 4),
            "impact_score": round(self.impact_score, 4),
            "node_score": round(self.node_score, 4),
            "edge_score": round(self.edge_score, 4),
            "vuln_score": round(self.vuln_score, 4),
        })
        return c_dict


@dataclass
class CriticalAssetRisk:
    """Risk summary for a critical asset exposed to attack paths."""
    asset_id: str
    asset_label: str
    max_risk_score: float
    risk_tier: str
    total_reaching_paths: int
    top_path_ids: List[str]
    targeting_entry_points: List[str]

    def to_dict(self) -> Dict:
        return {
            "asset_id": self.asset_id,
            "asset_label": self.asset_label,
            "max_risk_score": round(self.max_risk_score, 4),
            "risk_tier": self.risk_tier,
            "total_reaching_paths": self.total_reaching_paths,
            "top_path_ids": self.top_path_ids,
            "targeting_entry_points": self.targeting_entry_points,
        }


# ---------------------------------------------------------------------------
# Scorer Engine
# ---------------------------------------------------------------------------

class RiskScorer:
    """Computes cybersecurity risk scores for attack paths and network assets."""

    def __init__(self, config: Optional[RiskScoringConfig] = None) -> None:
        self.config = config or RiskScoringConfig()

    @staticmethod
    def _get_tier(score: float) -> str:
        if score >= 0.70:
            return "CRITICAL"
        if score >= 0.45:
            return "HIGH"
        if score >= 0.25:
            return "MEDIUM"
        return "LOW"

    def score_path(self, candidate: CandidateAttackPath) -> ScoredAttackPath:
        """Score a single candidate attack path deterministically."""
        meta = candidate.structural_metadata

        # 1. Node score (from GNN probabilities or fallback)
        valid_probs = [p for p in candidate.node_attack_probabilities if p >= 0.0]
        if valid_probs:
            mean_prob = sum(valid_probs) / len(valid_probs)
            min_prob = min(valid_probs)
            node_score = 0.70 * mean_prob + 0.30 * min_prob
        else:
            # Fallback if model probabilities absent
            mean_cvss = meta.get("mean_max_cvss", 5.0)
            node_score = min(1.0, mean_cvss / 10.0)

        # 2. Edge exploitability score
        # Edge feature layout: [etype(4), firewall, encrypted, attack_weight, anomaly, hist]
        hop_scores: List[float] = []
        for ef in candidate.edge_features:
            if len(ef) >= 9:
                is_exploit = ef[2] > 0.5
                has_firewall = ef[4] > 0.5
                is_enc = ef[5] > 0.5
                atk_wt = ef[6]
                anomaly = ef[7]

                score = atk_wt
                if is_exploit:
                    score *= self.config.exploit_type_multiplier
                if has_firewall:
                    score *= self.config.firewall_penalty
                if is_enc:
                    score *= self.config.encryption_penalty
                if anomaly > 0:
                    score = min(1.0, score + 0.15)
                hop_scores.append(min(1.0, max(0.0, score)))
            else:
                hop_scores.append(0.5)

        edge_score = (sum(hop_scores) / len(hop_scores)) if hop_scores else 0.5

        # 3. Vulnerability score
        mean_cvss = meta.get("mean_max_cvss", 0.0)
        vuln_score = min(1.0, mean_cvss / 10.0)

        # 4. Multi-hop decay (longer chains face higher failure / detection chance)
        hops = max(1, candidate.path_length)
        decay = self.config.length_decay ** (hops - 1)

        # Likelihood
        likelihood = (
            self.config.w_node * node_score
            + self.config.w_edge * edge_score
            + self.config.w_vuln * vuln_score
        ) * decay
        likelihood = min(1.0, max(0.0, likelihood))

        # Impact
        if candidate.reaches_critical_asset:
            target_criticality = 1.0
        else:
            target_criticality = 0.40  # non-critical terminal default

        impact = 0.40 + 0.60 * target_criticality

        # Final risk
        final_risk = min(1.0, max(0.0, likelihood * impact))
        tier = self._get_tier(final_risk)

        return ScoredAttackPath(
            candidate=candidate,
            risk_score=final_risk,
            risk_tier=tier,
            likelihood_score=likelihood,
            impact_score=impact,
            node_score=node_score,
            edge_score=edge_score,
            vuln_score=vuln_score,
        )

    def rank_paths(
        self,
        candidates: List[CandidateAttackPath],
        top_k: Optional[int] = None,
        critical_only: bool = False,
    ) -> List[ScoredAttackPath]:
        """Score and rank candidate paths descending by risk score."""
        k = top_k if top_k is not None else self.config.top_k
        scored = [self.score_path(c) for c in candidates]

        if critical_only:
            scored = [p for p in scored if p.reaches_critical_asset]

        # Deterministic sort: risk descending, then path_id
        scored.sort(key=lambda p: (-p.risk_score, p.path_id))
        return scored[:k]

    def assess_critical_assets(
        self,
        scored_paths: List[ScoredAttackPath],
    ) -> List[CriticalAssetRisk]:
        """Aggregate attack risk across all critical asset targets."""
        asset_map: Dict[str, List[ScoredAttackPath]] = {}
        for sp in scored_paths:
            if sp.reaches_critical_asset:
                term = sp.terminal_node
                asset_map.setdefault(term, []).append(sp)

        results: List[CriticalAssetRisk] = []
        for asset_id, paths in sorted(asset_map.items()):
            label = paths[0].node_labels[-1]
            max_risk = max(p.risk_score for p in paths)
            entry_set = sorted({p.entry_node for p in paths})
            top_ids = [p.path_id for p in sorted(paths, key=lambda p: -p.risk_score)[:3]]

            results.append(
                CriticalAssetRisk(
                    asset_id=asset_id,
                    asset_label=label,
                    max_risk_score=max_risk,
                    risk_tier=self._get_tier(max_risk),
                    total_reaching_paths=len(paths),
                    top_path_ids=top_ids,
                    targeting_entry_points=entry_set,
                )
            )

        # Sort assets by highest risk descending
        results.sort(key=lambda a: (-a.max_risk_score, a.asset_id))
        return results

    def score_topology(
        self,
        candidates: List[CandidateAttackPath],
        top_k: Optional[int] = None,
    ) -> Tuple[List[ScoredAttackPath], List[CriticalAssetRisk], float]:
        """
        Complete Phase 5 evaluation on a candidate set.

        Returns:
            (top_k_paths, critical_assets_at_risk, overall_network_risk)
        """
        all_scored = [self.score_path(c) for c in candidates]
        all_scored.sort(key=lambda p: (-p.risk_score, p.path_id))

        k = top_k if top_k is not None else self.config.top_k
        top_paths = all_scored[:k]
        critical_assets = self.assess_critical_assets(all_scored)

        # Network risk = max path risk if exists, else 0.0
        overall_risk = top_paths[0].risk_score if top_paths else 0.0

        return top_paths, critical_assets, overall_risk
