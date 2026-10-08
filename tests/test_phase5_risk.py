"""
test_phase5_risk.py
-------------------
Unit tests for Phase 5: Risk Scoring & Top-K Attack Paths.
"""

import pytest

from backend.ml.path_models import CandidateAttackPath
from backend.ml.risk_scorer import (
    CriticalAssetRisk,
    RiskScorer,
    RiskScoringConfig,
    ScoredAttackPath,
)


def _make_dummy_candidate(
    path_id: str = "p1",
    path_length: int = 2,
    reaches_critical: bool = True,
    mean_cvss: float = 8.0,
    has_firewall: bool = False,
    is_encrypted: bool = False,
    is_exploit: bool = False,
    node_probs: tuple = (0.8, 0.8, 0.8),
    attack_weight: float = 0.7,
) -> CandidateAttackPath:
    node_ids = [f"n{i}" for i in range(path_length + 1)]
    edge_pairs = [(node_ids[i], node_ids[i + 1]) for i in range(path_length)]

    edge_feats = []
    for _ in range(path_length):
        # [etype(4), firewall, encrypted, attack_weight, anomaly, hist]
        ef = [
            0.0 if is_exploit else 1.0,  # network_link
            0.0,
            1.0 if is_exploit else 0.0,  # exploit_path
            0.0,
            1.0 if has_firewall else 0.0,
            1.0 if is_encrypted else 0.0,
            attack_weight,
            0.0,
            0.0,
        ]
        edge_feats.append(ef)

    return CandidateAttackPath(
        path_id=path_id,
        node_ids=node_ids,
        edge_pairs=edge_pairs,
        node_labels=[f"Label-{nid}" for nid in node_ids],
        node_attack_probabilities=list(node_probs[:path_length + 1]),
        edge_features=edge_feats,
        path_length=path_length,
        entry_node=node_ids[0],
        terminal_node=node_ids[-1],
        reaches_critical_asset=reaches_critical,
        structural_metadata={"mean_max_cvss": mean_cvss},
    )


class TestRiskScoringConfig:
    def test_default_config(self):
        cfg = RiskScoringConfig()
        assert abs(cfg.w_node + cfg.w_edge + cfg.w_vuln - 1.0) < 1e-4

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError):
            RiskScoringConfig(w_node=0.5, w_edge=0.5, w_vuln=0.5)


class TestRiskScorerLogic:
    def setup_method(self):
        self.scorer = RiskScorer()

    def test_score_bounds(self):
        cand = _make_dummy_candidate()
        scored = self.scorer.score_path(cand)
        assert 0.0 <= scored.risk_score <= 1.0
        assert 0.0 <= scored.likelihood_score <= 1.0
        assert 0.0 <= scored.impact_score <= 1.0
        assert scored.risk_tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW")

    def test_critical_asset_increases_risk(self):
        c_crit = _make_dummy_candidate(reaches_critical=True)
        c_noncrit = _make_dummy_candidate(reaches_critical=False)
        s_crit = self.scorer.score_path(c_crit)
        s_noncrit = self.scorer.score_path(c_noncrit)

        assert s_crit.impact_score > s_noncrit.impact_score
        assert s_crit.risk_score > s_noncrit.risk_score

    def test_firewall_decreases_risk(self):
        c_open = _make_dummy_candidate(has_firewall=False)
        c_firewalled = _make_dummy_candidate(has_firewall=True)
        s_open = self.scorer.score_path(c_open)
        s_firewalled = self.scorer.score_path(c_firewalled)

        assert s_open.edge_score > s_firewalled.edge_score
        assert s_open.risk_score > s_firewalled.risk_score

    def test_encryption_decreases_risk(self):
        c_plain = _make_dummy_candidate(is_encrypted=False)
        c_enc = _make_dummy_candidate(is_encrypted=True)
        s_plain = self.scorer.score_path(c_plain)
        s_enc = self.scorer.score_path(c_enc)

        assert s_plain.edge_score > s_enc.edge_score
        assert s_plain.risk_score > s_enc.risk_score

    def test_exploit_path_increases_risk(self):
        c_normal = _make_dummy_candidate(is_exploit=False)
        c_exploit = _make_dummy_candidate(is_exploit=True)
        s_normal = self.scorer.score_path(c_normal)
        s_exploit = self.scorer.score_path(c_exploit)

        assert s_exploit.edge_score > s_normal.edge_score
        assert s_exploit.risk_score > s_normal.risk_score

    def test_cvss_increases_risk(self):
        c_low = _make_dummy_candidate(mean_cvss=2.0)
        c_high = _make_dummy_candidate(mean_cvss=9.8)
        s_low = self.scorer.score_path(c_low)
        s_high = self.scorer.score_path(c_high)

        assert s_high.vuln_score > s_low.vuln_score
        assert s_high.risk_score > s_low.risk_score

    def test_multi_hop_decay(self):
        c_short = _make_dummy_candidate(path_length=1)
        c_long = _make_dummy_candidate(path_length=4)
        s_short = self.scorer.score_path(c_short)
        s_long = self.scorer.score_path(c_long)

        # Longer paths have execution decay discount
        assert s_short.likelihood_score > s_long.likelihood_score


class TestTopKRanking:
    def setup_method(self):
        self.scorer = RiskScorer()

    def test_ranking_order(self):
        c1 = _make_dummy_candidate("p1", mean_cvss=3.0)
        c2 = _make_dummy_candidate("p2", mean_cvss=9.5)
        c3 = _make_dummy_candidate("p3", mean_cvss=6.0)

        ranked = self.scorer.rank_paths([c1, c2, c3], top_k=3)
        assert len(ranked) == 3
        assert ranked[0].path_id == "p2"
        assert ranked[1].path_id == "p3"
        assert ranked[2].path_id == "p1"
        assert ranked[0].risk_score >= ranked[1].risk_score >= ranked[2].risk_score

    def test_top_k_truncation(self):
        candidates = [_make_dummy_candidate(f"p{i}") for i in range(10)]
        ranked = self.scorer.rank_paths(candidates, top_k=4)
        assert len(ranked) == 4

    def test_critical_assets_assessment(self):
        c1 = _make_dummy_candidate("p1", reaches_critical=True, mean_cvss=9.0)
        c2 = _make_dummy_candidate("p2", reaches_critical=True, mean_cvss=5.0)
        c3 = _make_dummy_candidate("p3", reaches_critical=False)

        scored = [self.scorer.score_path(c) for c in [c1, c2, c3]]
        assets = self.scorer.assess_critical_assets(scored)

        assert len(assets) == 1
        assert assets[0].total_reaching_paths == 2
        assert assets[0].max_risk_score == scored[0].risk_score

    def test_serialization(self):
        cand = _make_dummy_candidate()
        scored = self.scorer.score_path(cand)
        d = scored.to_dict()
        assert "risk_score" in d
        assert "risk_tier" in d
        assert "likelihood_score" in d
        assert "impact_score" in d
