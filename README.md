# Graph-Based Attack Path Predictor AI

A cybersecurity research prototype that models computer networks as graphs and uses
Graph Neural Networks (GNNs) to predict likely multi-step attack paths from entry points
toward critical business assets.

## ⚠️ Defensive Research Disclaimer

This project is built for **defensive cybersecurity research only**. All network topologies,
vulnerabilities, and telemetry records are synthetically generated (`SYN-` prefix). No real
systems are targeted or exploited.

---

## Architecture Pipeline

```
Network Topology + Vulnerabilities + Telemetry + Historical Attacks
        ↓
    Graph Construction (Phase 1)
        ↓
    Graph Features (Phase 2: 43 Node / 9 Edge features)
        ↓
    GNN GraphSAGE (Phase 3: Inductive Node Attack Probabilities)
        ↓
    Candidate Attack Paths (Phase 4: Bounded DFS)
        ↓
    Risk Scoring & Top-K Ranking (Phase 5: Likelihood × Impact)
        ↓
    Critical Assets at Risk (Phase 5)
        ↓
    Explainable AI & Defensive Simulator (Phase 6: Factor Breakdown, Chokepoints, Mitigations)
        ↓
    FastAPI Service & Proactive Defense Dashboard (Phase 7 & Phase 8)
```

---

## Technology Stack

| Layer        | Technology                               |
|--------------|------------------------------------------|
| ML/AI        | Python, PyTorch, PyTorch Geometric (PyG) |
| Graph        | NetworkX                                 |
| Data         | NumPy, Pandas, scikit-learn              |
| Backend      | FastAPI, Uvicorn, Pydantic               |
| Frontend     | HTML5, Tailwind CSS, Cytoscape.js        |

---

## Phase Status (All Complete)

- [x] **Phase 1**: Project Foundation & Graph Construction
- [x] **Phase 2**: GNN Input & Graph Data Pipeline
- [x] **Phase 3**: GraphSAGE Model & Training
- [x] **Phase 4**: Attack Path Prediction
- [x] **Phase 5**: Risk Scoring & Top-K Paths
- [x] **Phase 6**: Explainable AI & Mitigation Simulator
- [x] **Phase 7**: FastAPI Backend Service
- [x] **Phase 8**: Interactive Proactive Defense Dashboard

---

## Quickstart

### 1. Run Server (Frontend + Backend in One Command)

```bash
python3 run.py
```

- **Local Access**: [http://localhost:8000](http://localhost:8000)
- **Network / LAN Access**: `http://<machine-ip>:8000` (e.g. `http://10.11.12.189:8000`)
- **Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

Custom host/port:
```bash
HOST=0.0.0.0 PORT=8080 python3 run.py
```

### 2. Run All Tests (395 Passing Tests)

```bash
python3 -m pytest
```

### 3. Run End-to-End Pipeline Validation

```bash
python3 validate_full_pipeline.py
```

---

## Project Structure

```
ai-project/
├── backend/
│   ├── core/
│   │   ├── data_models.py          # Node, Edge, Vulnerability, Network data models
│   │   ├── graph_builder.py        # Directed graph construction from topology
│   │   ├── feature_extractor.py    # Graph feature extraction (43 node features)
│   │   └── synthetic_data.py       # Procedural network generator (SYN- prefix)
│   ├── ml/
│   │   ├── dataset.py              # PyG Data conversion, 9 edge features, split logic
│   │   ├── model.py                # AttackGNN (GraphSAGE / SAGEConv inductive architecture)
│   │   ├── trainer.py              # Reproducible training pipeline with early stopping
│   │   ├── evaluator.py            # Honest metric evaluation (no fabricated AUCs)
│   │   ├── path_models.py          # PathConfig, CandidateAttackPath, PathGenerationSummary
│   │   ├── path_generator.py       # Bounded DFS path enumeration with GNN inference
│   │   ├── risk_scorer.py          # Phase 5: Likelihood x Impact scoring & Top-K ranking
│   │   └── explainer.py            # Phase 6: Kill-chain XAI, chokepoints & defense simulator
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # Phase 7: FastAPI endpoints (/api/analyze, /api/simulate, etc.)
│       └── static/
│           └── index.html          # Phase 8: Cytoscape.js interactive SOC dashboard
├── tests/
│   ├── test_data_models.py
│   ├── test_graph_builder.py
│   ├── test_feature_extractor.py
│   ├── test_synthetic_data.py
│   ├── test_dataset.py
│   ├── test_phase3_gnn.py
│   ├── test_phase4_paths.py
│   ├── test_phase5_risk.py
│   ├── test_phase6_xai.py
│   └── test_api.py
├── docs/
│   ├── phase3_gnn.md
│   ├── phase4_candidate_paths.md
│   ├── phase5_risk_scoring.md
│   └── phase6_explainable_ai.md
├── checkpoints/
│   └── best_attack_gnn.pt          # Trained GraphSAGE model checkpoint
├── run.py                          # Unified launcher for frontend & backend
├── attack-predictor.service        # Systemd service file for Linux deployment
├── validate_full_pipeline.py       # End-to-end integration smoke test
├── instructions.md                 # Detailed deployment and operating instructions
├── project_context.md              # Deep technical context and design decisions
├── requirements.txt
└── README.md
```
