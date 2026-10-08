# Project Context & Technical Architecture

## 1. Executive Summary

The **Graph-Based Attack Path Predictor AI** is a cybersecurity research platform that models computer networks as directed graphs and employs Graph Neural Networks (GNNs) combined with bounded graph search to predict, evaluate, and explain likely multi-step attack trajectories from network entry points to critical business assets.

The platform provides security analysts and defenders with:
- **Node-level attack involvement predictions** via an inductive GraphSAGE GNN.
- **Enumerated multi-hop attack routes** through network topologies.
- **Quantified path risk scores** using a formal $Likelihood \times Impact$ model.
- **Explainable AI (XAI)** featuring kill-chain narratives, factor attribution, and chokepoint detection.
- **Proactive What-If Defense Simulation** that measures how firewalls, patches, or node isolations reduce overall network risk.

---

## 2. Design Philosophy & Ethical Boundaries

1. **Defensive Cybersecurity Only**: The system is designed to identify exposure, prioritize hardening, and simulate remediations. It never performs live exploitation.
2. **Synthetic Data Provenance**: All network structures, vulnerabilities, and telemetry records are procedurally generated and tagged with the `SYN-` CVE prefix.
3. **Scientific Integrity**:
   - Zero data leakage across train/validation/test splits (splits occur strictly at whole-graph scenario boundaries).
   - Metrics are never fabricated: degenerate single-class splits report ROC-AUC and PR-AUC as `None` with explicit logged explanations.
   - Historical attack ground-truth labels are never read during candidate path enumeration or risk scoring.

---

## 3. End-to-End Pipeline (Phases 1–8)

### Phase 1 — Project Foundation & Graph Construction
- **Location**: `backend/core/data_models.py`, `graph_builder.py`, `synthetic_data.py`
- **Output**: Constructs directed NetworkX graphs (`nx.DiGraph`) representing computer networks with IP addresses, node roles, open ports, CVSS vulnerability parameters, and telemetry flow records.

### Phase 2 — GNN Input & Feature Engineering
- **Location**: `backend/core/feature_extractor.py`, `backend/ml/dataset.py`
- **Features**:
  - **43 Node Features**: Raw scalars (CVSS, ports, asset value), one-hot node types (workstation, server, DB, router), one-hot OS types (Windows, Linux, macOS), and structural topological features (PageRank, centrality, in/out degrees).
  - **9 Edge Features**: One-hot edge categories (network link, trust, exploit, data flow), firewall rules, encryption status, baseline attack weight, and telemetry anomalies.
- **Output**: PyTorch Geometric `Data(x, edge_index, edge_attr, y)` objects.

### Phase 3 — GraphSAGE Inductive GNN Architecture
- **Location**: `backend/ml/model.py`, `trainer.py`, `evaluator.py`
- **Architecture**: `Linear(43 -> 64) -> BatchNorm -> SAGEConv(64 -> 64) -> SAGEConv(64 -> 64) -> Linear(64 -> 2)`.
- **Target**: Predicts binary node attack-involvement probability $P(\text{attack\_involved}) \in [0.0, 1.0]$.
- **Weight Checkpoint**: Saved at `checkpoints/best_attack_gnn.pt`.

### Phase 4 — Candidate Attack Path Generation
- **Location**: `backend/ml/path_models.py`, `backend/ml/path_generator.py`
- **Algorithm**: Bounded Depth-First Search (DFS) with neighbor lexicographical sorting for strict determinism.
- **Constraints**: Simple paths (no repeated nodes or self-loops), strictly directed hops originating from entry points. Edge-count convention: `path_length = len(node_ids) - 1`.

### Phase 5 — Risk Scoring & Top-K Ranking
- **Location**: `backend/ml/risk_scorer.py`
- **Formula**:
  $$\text{Likelihood} = (0.35 \cdot S_{\text{node}} + 0.35 \cdot S_{\text{edge}} + 0.30 \cdot S_{\text{vuln}}) \cdot 0.95^{(\text{hops} - 1)}$$
  $$\text{Impact} = 0.40 + 0.60 \cdot I_{\text{target}}$$
  $$\text{Risk\_Score} = \text{clamp}(\text{Likelihood} \cdot \text{Impact}, 0.0, 1.0)$$
- **Output**: Scored attack paths ranked descending into CRITICAL, HIGH, MEDIUM, and LOW severity tiers. Aggregates critical assets at risk.

### Phase 6 — Explainable AI & Mitigation Simulator
- **Location**: `backend/ml/explainer.py`
- **Components**:
  - **Kill-Chain Narratives**: Plain-language summaries of attacker traversal.
  - **Factor Attribution**: Relative weight breakdown between GNN probability, CVSS severity, and edge perimeter controls.
  - **Chokepoint Detection**: Graph cut analysis identifying the pivot nodes and edges where attack paths converge.
  - **What-If Simulation**: Computes severed path count and percentage drop in network risk when applying specific defenses.

### Phase 7 & 8 — FastAPI Service & Interactive Dashboard
- **Location**: `backend/api/main.py`, `backend/api/static/index.html`
- **Delivery**:
  - Unified FastAPI service serving both REST endpoints and frontend static assets.
  - Interactive Cytoscape.js dark-mode SOC dashboard with real-time path animation, hover inspection, scenario selection, and one-click defense simulation.

---

## 4. Key Invariants for Future Modifications

1. **Path Length Convention**: Path length always denotes the number of directed edges traversed (`path_length = len(node_ids) - 1`).
2. **Quota Wrappers**: Quotas inside recursive DFS routines must always use mutable list wrappers (`per_entry_count_ref = [0]`) to ensure accurate accumulation across recursive branches.
3. **Single-Command Execution**: The launcher [`run.py`](file:///home/triotreek/ai-project/run.py) binds to `0.0.0.0:8000` to ensure seamless local and network-wide accessibility out of the box.
4. **Test Suite Integrity**: Always verify changes against the full pytest suite (`python3 -m pytest`). All 395 tests must pass with zero failures.
