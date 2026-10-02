# Phase 3 — GNN Model: Design, Training & Evaluation

> **Data label**: ALL datasets used in Phase 3 are **[SYNTHETIC]**.  
> No real network infrastructure, real vulnerabilities, or real attack records are involved.

---

## 1. Objective

Train a Graph Neural Network that learns to identify which nodes in a network
graph are likely to appear in attack paths, using historical (synthetic) attack
data as training labels.

---

## 2. ML Problem Definition

**Task**: Node-level binary classification  
**Unit of prediction**: Individual network node  
**This is NOT**: Complete multi-hop attack path prediction (Phase 4+)

The GNN learns a function:

```
f(x_i, N(i), G) → y_i ∈ {0, 1}
```

Where:
- `x_i` = 43-dimensional feature vector for node i
- `N(i)` = neighborhood of node i in the directed network graph
- `G` = full directed graph structure
- `y_i` = 1 if node i appeared in ≥1 historical attack path, 0 otherwise

---

## 3. Input Representation

Each network topology is converted to a PyTorch Geometric `Data` object:

| Field | Shape | dtype | Contents |
|-------|-------|-------|----------|
| `data.x` | (N, 43) | float32 | Node feature matrix |
| `data.edge_index` | (2, E) | int64 | Directed COO edge list |
| `data.edge_attr` | (E, 9) | float32 | Edge feature matrix |
| `data.y` | (N,) | int64 | Binary node labels {0,1} |
| `data.num_nodes` | int | — | N |

---

## 4. Node Features (43 total)

### Block A — Raw Scalar Features (indices 0–19)

| Idx | Name | Type | Meaning |
|-----|------|------|---------|
| 0 | `is_entry_point` | binary | Directly reachable by attacker |
| 1 | `is_critical_asset` | binary | High-value target |
| 2 | `asset_value` | float | Business value [0,1] |
| 3 | `vuln_count` | int | Number of vulnerabilities |
| 4 | `max_cvss` | float | Highest CVSS v3 score |
| 5 | `has_critical_vuln` | binary | Any CVSS ≥9.0 vuln |
| 6 | `has_exploited_in_wild` | binary | Any actively exploited vuln |
| 7 | `critical_vuln_count` | int | Count of CRITICAL vulns |
| 8 | `high_vuln_count` | int | Count of HIGH vulns |
| 9 | `unpatched_vuln_count` | int | Unpatched vuln count |
| 10 | `exploited_in_wild_count` | int | Actively exploited count |
| 11 | `avg_cvss` | float | Mean CVSS across vulns |
| 12 | `avg_exploitability` | float | Mean CVSS exploitability sub-score |
| 13 | `service_count` | int | Number of open services |
| 14 | `has_rdp` | binary | Port 3389 open |
| 15 | `has_ssh` | binary | Port 22 open |
| 16 | `has_smb` | binary | Port 445 open |
| 17 | `has_http` | binary | Port 80 open |
| 18 | `has_https` | binary | Port 443 open |
| 19 | `compromised` | binary | Currently flagged compromised |

### Block B — Node Type One-Hot (indices 20–27)

Encodes: workstation, server, database, router, firewall, internet, iot_device, critical_asset

### Block C — OS Type One-Hot (indices 28–32)

Encodes: windows, linux, macos, firmware, unknown

### Block D — Structural Features (indices 33–42)

| Idx | Name | Meaning |
|-----|------|---------|
| 33 | `in_degree` | Incoming directed edges |
| 34 | `out_degree` | Outgoing directed edges |
| 35 | `total_degree` | in + out |
| 36 | `pagerank` | PageRank (α=0.85) |
| 37 | `betweenness_centrality` | Approximate betweenness |
| 38 | `closeness_centrality` | Closeness centrality |
| 39 | `clustering_coeff` | Clustering coefficient (undirected) |
| 40 | `attack_weighted_in_degree` | Σ attack_weight on in-edges |
| 41 | `anomaly_signal` | Σ anomaly_count on in-edges |
| 42 | `historical_attack_exposure` | Σ historical_attack_count on in-edges |

---

## 5. Edge Features (9 total)

| Idx | Name | Meaning |
|-----|------|---------|
| 0 | `edgetype_network_link` | One-hot edge type |
| 1 | `edgetype_trust_relationship` | One-hot edge type |
| 2 | `edgetype_exploit_path` | One-hot edge type |
| 3 | `edgetype_data_flow` | One-hot edge type |
| 4 | `firewall_rule` | Firewall governs this connection |
| 5 | `is_encrypted` | Traffic is encrypted |
| 6 | `attack_weight` | Baseline traversal likelihood |
| 7 | `anomaly_count` | Telemetry anomaly count on this edge |
| 8 | `historical_attack_count` | Times traversed in historical attacks |

**Note**: Edge features are stored in `data.edge_attr` for future use (Phase 4 path scoring). GraphSAGE in Phase 3 does NOT consume `edge_attr` in message passing (see Section 8).

---

## 6. Target Definition

```
y[i] = 1  if node i appears in ≥1 historical attack path
y[i] = 0  otherwise
```

**Label generation** (`build_node_labels()`):
1. Collect the union of all node IDs from all `HistoricalAttack.attack_path` lists in the topology.
2. For each node in the graph: assign label 1 if in the union set, 0 otherwise.
3. Labels are deterministic for a given topology + attack history.

**Positive class** = "this node was traversed by at least one attack"  
**Negative class** = "this node was not traversed by any recorded attack"

---

## 7. Train/Validation/Test Methodology

**Split unit: full graph scenarios** (not individual nodes).

| Split | Ratio | Description |
|-------|-------|-------------|
| Train | 70% | Model learns from these topologies |
| Validation | 15% | Early stopping monitored on this set |
| Test | 15% | Final evaluation — never touched during training |

**Leakage prevention**:
- Splitting is done at the topology/scenario level.
- Each scenario has UUID-based node IDs unique across all scenarios.
- No node ID, label, or graph structural information crosses split boundaries.
- Class weights are computed **only** from training labels.

---

## 8. Architecture Decision

### Decision: GraphSAGE (SAGEConv) — not edge-aware GNN

**Options evaluated**:

| Option | Model | Edge features in message passing |
|--------|-------|----------------------------------|
| A | GraphSAGE (SAGEConv) | ❌ Node features only |
| B | NNConv / GATv2 | ✅ Directly uses edge_attr |

**Why GraphSAGE was selected**:

1. **Edge signals already encoded in node features**: Features 40–42 (`attack_weighted_in_degree`, `anomaly_signal`, `historical_attack_exposure`) aggregate the three most discriminative edge attributes (attack_weight, anomaly_count, historical_attack_count) into the node feature vector **before** GNN message passing. This means the GNN already sees these signals.

2. **Small graphs, high overfitting risk**: With 10–60 nodes per synthetic scenario, an edge-parameterised MLP (NNConv) adds parameters that are unlikely to generalise without many more training scenarios.

3. **Inductive learning**: GraphSAGE learns a transferable aggregation function that works on unseen graph structures — exactly what scenario-level splitting requires.

4. **Simplicity and reproducibility**: GraphSAGE is well-studied and its behaviour is predictable and well-documented.

**Edge features are NOT lost**: `data.edge_attr` is preserved in the PyG Data objects and will be used directly in Phase 4 path scoring.

---

## 9. Model Architecture

```
AttackGNN
─────────────────────────────────────────────
Input:   x ∈ ℝ^{N×43}
         edge_index ∈ ℤ^{2×E}

Layer 0: Linear(43 → hidden_channels)
         BatchNorm1d(hidden_channels)
         ReLU
         Dropout(p=dropout)

Layer 1: SAGEConv(hidden → hidden, aggr='mean')
         BatchNorm1d(hidden)
         ReLU
         Dropout(p=dropout)

Layer 2: SAGEConv(hidden → hidden, aggr='mean')
         BatchNorm1d(hidden)
         ReLU
         Dropout(p=dropout)

Output:  Linear(hidden → 2)
         → logits ∈ ℝ^{N×2}

─────────────────────────────────────────────
Default config:
  hidden_channels = 64
  num_layers      = 2
  dropout         = 0.3
  activation      = ReLU
  weight_init     = Kaiming uniform
─────────────────────────────────────────────
```

**Outputs**: Raw logits `[N, 2]`. Softmax applied externally for probabilities.

---

## 10. Training Procedure

1. Fix all random seeds (Python, NumPy, PyTorch) at `seed=42`.
2. Compute class weights from **training labels only** (inverse-frequency).
3. Optimizer: Adam (`lr=5e-3`, `weight_decay=1e-4`).
4. Gradient clipping: `max_norm=1.0` (prevents exploding gradients on small graphs).
5. Early stopping: monitor validation loss, patience = 20 epochs.
6. Save best checkpoint (lowest validation loss).

---

## 11. Loss Function

**CrossEntropyLoss** with optional class weights:

```python
weight[class=0] = N_total / (2 × N_negative)
weight[class=1] = N_total / (2 × N_positive)
```

Class weights are computed from training labels ONLY and disabled automatically
if training data contains only one class (degenerate case handled gracefully).

---

## 12. Class Imbalance Handling

**The all-positive label warning** occurs in small (10-node) topologies because:
- The topology has only 5 intermediate nodes.
- 8 attack paths with 1–3 random hops from 5 nodes statistically cover all 5 intermediate nodes.
- Combined with 3 entry points + 2 critical assets = 10 total = all positive.

**How this is handled**:
- These scenarios are **not silently deleted**.
- They contribute valid CrossEntropyLoss signal (even single-class loss is valid).
- The `Evaluator` detects single-class splits and reports ROC-AUC / PR-AUC as "NOT AVAILABLE" with an explanation — never fabricated.
- Class weights are disabled for single-class training sets (logged, not silent).
- Larger topologies (medium/large) always have both classes.

---

## 13. Evaluation Metrics

| Metric | Valid condition | Notes |
|--------|----------------|-------|
| Accuracy | Always | Misleading for imbalanced — reported but not relied upon |
| Precision | Always | zero_division=0 |
| Recall | Always | zero_division=0 |
| F1-score | Always | Macro, zero_division=0 |
| Confusion matrix | Always | [[TN, FP], [FN, TP]] |
| ROC-AUC | Both classes present | Reported as None otherwise |
| PR-AUC | Both classes present | Preferred metric for imbalanced |

---

## 14. Reproducibility

- Fixed `random_seed=42` for all: Python RNG, NumPy, PyTorch.
- `torch.use_deterministic_algorithms(True, warn_only=True)` enabled.
- Dataset generation is seed-controlled (`SyntheticNetworkGenerator(seed=k)`).
- Checkpoint saves full config + architecture + training history.
- Re-running `run_phase3_training.py` with same config produces identical losses.

---

## 15. Limitations

1. **Synthetic data only**: The GNN is trained on procedurally generated topologies. Real-world performance is unknown and would require authorized real-network data.
2. **All-positive scenarios**: Small topologies with ≤10 nodes frequently produce all-positive labels, reducing the diversity of negative examples in training.
3. **Node-level scope**: The model identifies attack-involved nodes — it does NOT rank or sequence them into complete paths (Phase 4).
4. **GraphSAGE limitations**: SAGEConv uses mean aggregation which treats all neighbors equally. Attention-weighted aggregation may improve precision on heterogeneous networks.
5. **No temporal modeling**: The GNN treats the graph as static. Time-series attack data is not exploited.

---

## 16. Why This Is Not Yet Complete Attack-Path Prediction

Phase 3 produces per-node probability scores:

```
P(node_i = attack_involved) ∈ [0.0, 1.0]
```

This is **not** a ranked list of complete multi-hop attack sequences.

To construct candidate attack paths (Phase 4), these node scores must be:
1. Combined with the graph structure (edge_index, edge_attr)
2. Used to weight path search algorithms (Dijkstra / beam search on attack-weighted edges)
3. Scored end-to-end for complete path risk
4. Ranked to produce Top-K attack paths

Phase 3 provides the **node-level attack involvement signal** that feeds into this pipeline. Phase 4 builds on top of it.

---

## Files Created

| File | Purpose |
|------|---------|
| `backend/ml/model.py` | AttackGNN implementation + ADR |
| `backend/ml/trainer.py` | Reproducible training pipeline |
| `backend/ml/evaluator.py` | Metric computation (no fabrication) |
| `backend/ml/__init__.py` | Package exports |
| `tests/test_phase3_gnn.py` | ~60 unit + integration tests |
| `run_phase3_training.py` | Full training execution script |
| `docs/phase3_gnn.md` | This document |
