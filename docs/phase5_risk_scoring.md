# Phase 5 — Attack Path Risk Scoring & Top-K Ranking

> **Data label**: ALL datasets are **[SYNTHETIC]**.  
> No real network infrastructure, real vulnerabilities, or real attack records are involved.

---

## 1. Objective

Transform unranked candidate attack paths (Phase 4) into prioritized, quantified cybersecurity risks using a composite formula combining machine learning signals, vulnerability severity metrics, network perimeter defenses, and target business impact.

---

## 2. Cybersecurity Risk Formula

Standard defensive risk formula:

```
Risk = Likelihood × Impact
```

### A. Likelihood Calculation

Likelihood accounts for four key security dimensions:

1. **Node Compromise Likelihood ($S_{\text{node}}$)**:
   Blended GNN attack-involvement probability across the path:
   $$S_{\text{node}} = 0.70 \times \text{mean}(P) + 0.30 \times \min(P)$$
   *(Accounts for average node exposure while penalizing hard bottleneck nodes).*

2. **Edge Traversal Exploitability ($S_{\text{edge}}$)**:
   For each hop $(u \to v)$, traversal likelihood is adjusted based on security controls:
   - Baseline edge `attack_weight` $\in [0, 1]$
   - **Firewall Penalty**: $\times 0.60$ if perimeter firewall governs connection
   - **Encryption Penalty**: $\times 0.80$ if link is encrypted
   - **Exploit Type Multiplier**: $\times 1.20$ if classified as an active exploit pathway
   - **Telemetry Anomaly Boost**: $+0.15$ if anomalous network traffic flows are observed

3. **Vulnerability Severity ($S_{\text{vuln}}$)**:
   Mean CVSS v3 score along the path normalized to $[0.0, 1.0]$:
   $$S_{\text{vuln}} = \frac{\text{mean\_max\_cvss}}{10.0}$$

4. **Multi-Hop Execution Decay**:
   Longer multi-hop attack paths have higher operational complexity and higher defender detection probability:
   $$\text{Decay} = 0.95^{(\text{path\_length} - 1)}$$

**Composite Likelihood**:
$$\text{Likelihood} = (0.35 \times S_{\text{node}} + 0.35 \times S_{\text{edge}} + 0.30 \times S_{\text{vuln}}) \times \text{Decay}$$

### B. Impact Calculation

Target asset business criticality:
- **Critical Asset Terminal**: $I_{\text{target}} = 1.0$ (Domain Controller, Database, SCADA)
- **Non-Critical Terminal**: $I_{\text{target}} = 0.40$
$$\text{Impact} = 0.40 + 0.60 \times I_{\text{target}}$$

### C. Final Risk Score & Severity Tiers

$$\text{Risk\_Score} = \text{clamp}(\text{Likelihood} \times \text{Impact}, 0.0, 1.0)$$

| Risk Score Range | Severity Tier |
|---|---|
| $\ge 0.70$ | **CRITICAL** |
| $0.45 \le \text{Score} < 0.70$ | **HIGH** |
| $0.25 \le \text{Score} < 0.45$ | **MEDIUM** |
| $< 0.25$ | **LOW** |

---

## 3. Top-K Ranking & Critical Asset Assessment

- **Top-K Ranking**: Paths are sorted deterministically by `risk_score` descending, with unique `path_id` as the secondary tie-breaker.
- **Critical Assets Exposure**: Aggregates all paths targeting high-value assets, reporting maximum exposure risk, targeting entry points, and total reaching paths.
