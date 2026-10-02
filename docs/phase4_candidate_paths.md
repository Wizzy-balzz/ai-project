# Phase 4 — Candidate Attack-Path Generation

> **Data label**: ALL datasets are **[SYNTHETIC]**.  
> No real network infrastructure, real vulnerabilities, or real attack records are involved.

> **Scope**: Phase 4 generates **candidate paths**. It does **not rank them** as final predicted attack paths. Path ranking is a Phase 5 concern.

---

## 1. Objective

Move from GNN node-level attack-involvement probabilities (Phase 3) to explicitly
enumerated candidate multi-hop attack paths through the network graph.

**Pipeline position**:
```
Phase 3: P(node_i = attack_involved) for every node
               ↓
Phase 4: Enumerate structurally valid candidate paths from entry → target
               ↓
Phase 5: Risk scoring + Top-K ranking (NOT Phase 4)
```

---

## 2. Candidate Path Definition

A candidate attack path is a **directed, simple path** through the network graph:

```
[node_0, node_1, ..., node_k]
```

Where:
- `node_0` is a network entry point (`is_entry_point = True`)
- `node_k` is the terminal node (preferably a critical asset)
- Every consecutive pair `(node_i, node_{i+1})` is a valid **directed edge** in the graph
- All nodes are **unique** within the path (no cycles)
- Path length is defined as the **number of edges** = `k`

### Path Length Convention

```
A → B         path_length = 1  (one edge)
A → B → C     path_length = 2  (two edges)
```

`path_length = len(node_ids) - 1`

This is the standard graph-theory edge-count convention, tested explicitly.

---

## 3. Entry-Point Selection

Entry points are identified from existing node metadata:

```python
G.nodes[nid].get("is_entry_point") == 1
```

Source: `NetworkNode.is_entry_point` stored by `GraphBuilder` as a node attribute.

If a topology has **no entry points**:
- Return empty candidate set
- Report condition in `PathGenerationSummary`
- No paths fabricated

---

## 4. Critical-Asset Handling

Critical assets are identified from:

```python
G.nodes[nid].get("is_critical_asset") == 1
```

- `CandidateAttackPath.reaches_critical_asset` = True when terminal node is a critical asset.
- The generator can optionally filter to only emit critical-terminal paths via `PathConfig.require_critical_terminal`.
- If no path from any entry point reaches any critical asset in the bounded depth, the candidate set is empty but correctly reported — not fabricated.

---

## 5. Path Enumeration Algorithm

**Algorithm**: Bounded Depth-First Search (DFS)

### Why DFS?

| Criterion | DFS | BFS |
|-----------|-----|-----|
| Memory footprint | O(depth × branching) — low | O(all frontier paths) — high |
| Early termination | Natural with backtracking | Requires explicit pruning |
| Simple path constraint | Visited-set per path | Requires per-path tracking |
| Output ordering | Lexicographic sort applied | Naturally breadth-ordered |

For directed graphs with depth bounds, DFS with backtracking is the standard approach.

### DFS Pseudocode

```
def dfs(node, visited, current_path):
    if len(current_path) - 1 >= min_length:
        emit current_path as candidate

    if len(current_path) - 1 >= max_length:
        return  # depth bound

    for neighbor in sorted(G.successors(node)):  # sorted for determinism
        if neighbor not in visited:
            visited.add(neighbor)
            current_path.append(neighbor)
            dfs(neighbor, visited, current_path)
            current_path.pop()
            visited.remove(neighbor)
```

### Sorting for Determinism

Neighbors are visited in **lexicographic order by node ID** at every DFS step.
This guarantees identical output for identical inputs without relying on dict/set ordering.

---

## 6. Path-Length Definition

| | Value |
|--|-------|
| Definition | Number of directed edges traversed |
| Unit | edges |
| Formula | `path_length = len(node_ids) - 1` |
| Edge-pairs count | `len(edge_pairs) == path_length` |
| `edge_features` count | `len(edge_features) == path_length` |

---

## 7. Duplicate Handling

Deduplication uses a **canonical path key**:

```python
path_key = " → ".join(node_ids)
```

- Two paths with the same ordered node sequence → **same key → deduplicated**.
- Two paths with different orderings (e.g., `A→B→C` vs `A→C→B`) → **different keys → both kept**.
- Direction is preserved.

---

## 8. GNN Probability Integration

### How probabilities are obtained

1. Load the trained Phase 3 checkpoint **once** at `PathGenerator` construction.
2. For each topology scenario, build the PyG Data object `(x, edge_index)`.
3. Run one forward pass: `logits = model(x, edge_index)` → shape `[N, 2]`.
4. Apply softmax: `probs = softmax(logits)[:, 1]` → `P(attack_involved)` per node.
5. Store as `{node_id: probability}` dict.
6. Attach to each `CandidateAttackPath.node_attack_probabilities`.

### Guarantees

- Model is **never retrained** during Phase 4.
- Probabilities are **never hard-coded or fabricated**.
- If no checkpoint is available, probabilities are set to sentinel value `-1.0` and `gnn_probabilities_attached = False` is reported in the summary.
- All attached probabilities are in `[0.0, 1.0]`.

---

## 9. Edge-Feature Preservation

For each `(u → v)` hop in a candidate path, the full 9-dimensional edge feature vector is extracted from `G[u][v]`:

| Idx | Feature | Source |
|-----|---------|--------|
| 0–3 | One-hot edge type | `edge_type` attribute |
| 4 | `firewall_rule` | `G[u][v]["firewall_rule"]` |
| 5 | `is_encrypted` | `G[u][v]["is_encrypted"]` |
| 6 | `attack_weight` | `G[u][v]["attack_weight"]` |
| 7 | `anomaly_count` | `G[u][v]["anomaly_count"]` |
| 8 | `historical_attack_count` | `G[u][v]["historical_attack_count"]` |

These match exactly the Phase 2 `EDGE_FEATURE_DIM = 9` encoding.

`CandidateAttackPath.edge_features` has shape `(path_length, 9)`.

---

## 10. Leakage Prevention

**What is NOT used during generation**:
- `HistoricalAttack.attack_path` — never read by the generator
- `data.y` (PyG ground-truth labels) — generator receives `NetworkTopology`, not PyG Data
- Validation/test labels — generator is called separately per split

**What IS used**:
- Graph topology (node/edge structure)
- Node attributes: `is_entry_point`, `is_critical_asset`, `vuln_count`, etc.
- Edge attributes: `attack_weight`, `edge_type`, etc.
- Trained GNN model predictions (probabilities only)

**Verification**: `PathGenerationSummary.used_historical_labels` is always `False`.  
**Test**: The leakage test generates paths with and without `historical_attacks` populated — output must be identical.

---

## 11. Scalability Controls

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_path_length` | 1 edge | Discard trivial single-node inputs |
| `max_path_length` | 6 edges | DFS depth bound |
| `max_paths_per_entry` | 50 | Quota per entry node |
| `max_total_paths` | 2000 | Hard cap across all entries |
| `require_critical_terminal` | False | Filter to critical targets only |

When a quota is hit:
- Generation stops for that entry node (or globally).
- `PathGenerationSummary.generation_truncated = True`.
- `PathGenerationSummary.truncation_reason` explains which quota was hit.
- Existing valid candidates are kept.
- **Nothing is silently discarded.**

---

## 12. Determinism

Given identical inputs:
- Same `NetworkTopology`
- Same `PathConfig`
- Same trained checkpoint

The generator produces:
- The **same candidate paths** in the **same order** on every run.

This is achieved by:
1. Sorting DFS neighbors lexicographically at every recursion step.
2. Sorting the final candidate list by `PathConfig.sort_by` (default: `"path_str"` = lexicographic on path key).
3. Using `torch.no_grad()` and `model.eval()` for deterministic inference.

---

## 13. Candidate Path Data Structure

```python
@dataclass
class CandidateAttackPath:
    path_id:                    str          # Unique UUID-based ID
    node_ids:                   List[str]    # Ordered node IDs
    edge_pairs:                 List[Tuple]  # (src, tgt) per hop
    node_labels:                List[str]    # Human-readable labels
    node_attack_probabilities:  List[float]  # GNN P(attack) per node
    edge_features:              List[List[float]]  # (path_length, 9)
    path_length:                int          # Number of edges
    entry_node:                 str          # node_ids[0]
    terminal_node:              str          # node_ids[-1]
    reaches_critical_asset:     bool
    is_valid:                   bool         # Passed all validation checks
    topology_id:                str          # Links to source scenario
    is_synthetic:               bool         # Always True
    structural_metadata:        dict         # Path features for Phase 5
```

### `structural_metadata` keys

| Key | Meaning |
|-----|---------|
| `path_length_edges` | Number of edges |
| `n_nodes` | Number of nodes |
| `n_critical_nodes` | Nodes with `is_critical_asset=True` |
| `n_vulnerable_nodes` | Nodes with `vuln_count > 0` |
| `mean_max_cvss` | Mean of per-node max_cvss |
| `mean_edge_attack_weight` | Mean edge `attack_weight` |
| `mean_node_attack_prob` | Mean GNN probability |
| `min_node_attack_prob` | Bottleneck node probability |
| `max_node_attack_prob` | Highest-risk node probability |

These are **features**, not a risk score. Phase 5 will define the risk score.

---

## 14. Limitations

1. **DFS combinatorial explosion**: On dense graphs with many entry points and long max_depth, the number of candidate paths can grow exponentially. The `max_paths_per_entry` and `max_total_paths` limits prevent runaway enumeration but may miss some valid paths.

2. **No shortest-path guarantee**: DFS does not guarantee the shortest path to critical assets is found first. Paths are enumerated in DFS order and sorted by the chosen key.

3. **Synthetic data only**: All topologies are procedurally generated. Real-world path quality depends on real topology data.

4. **GNN probabilities are node-level**: The probability attached to each node is `P(node involved in attack)` from Phase 3. It is NOT a per-edge traversal probability. Phase 5 will combine node probabilities with edge attributes into a path-level risk score.

5. **No path ranking**: Phase 4 outputs an unranked candidate set. Phase 5 produces the Top-K ranked paths.

---

## Files Created

| File | Purpose |
|------|---------|
| `backend/ml/path_models.py` | `PathConfig`, `CandidateAttackPath`, `PathGenerationSummary` |
| `backend/ml/path_generator.py` | `PathGenerator` (DFS), `PathValidator`, `GNNProbabilityProvider` |
| `tests/test_phase4_paths.py` | ~65 tests |
| `run_phase4_paths.py` | Execution script |
| `docs/phase4_candidate_paths.md` | This document |
