# Phase 6 — Explainable AI (XAI) & Proactive Mitigation Simulator

> **Data label**: ALL datasets are **[SYNTHETIC]**.  
> No real network infrastructure, real vulnerabilities, or real attack records are involved.

---

## 1. Objective

Provide transparency and actionable defensive guidance for identified attack paths by:
1. Explaining the kill chain narrative and quantitative factor contributions.
2. Identifying graph chokepoints where multiple attack paths converge.
3. Generating prioritized, actionable mitigation recommendations.
4. Simulating the effect of defenses on overall network risk and severed attack paths.

---

## 2. Key Components

### A. Kill-Chain Explanation
- **Plain-English Summary**: Outlines origin, lateral movement mechanism, and high-value target reached.
- **Hop-by-Hop Breakdown**: Details protocol, open services (RDP, SSH, SMB), firewall status, and CVSS vulnerabilities at each step.
- **Factor Attribution**: Quantifies the percentage contribution of GNN probability, CVSS score, and missing perimeter controls.

### B. Graph Chokepoint Identification
- Analyzes node and edge intersections across Top-K attack paths.
- Ranks network elements by intersection frequency and maximum path risk to find optimal defensive leverage points.

### C. Proactive Mitigation Simulator
- Simulates applying defensive actions:
  - **Firewall Rule Insertion**: Blocks edges traversed by attack paths.
  - **Vulnerability Patching**: Remediates high-CVSS CVEs on pivot hosts.
  - **Node Isolation**: Segments compromised or vulnerable hosts.
- Calculates counterfactual network risk reduction percentage and exact count of severed attack paths.
