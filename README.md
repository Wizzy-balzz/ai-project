# Graph-Based Attack Path Predictor AI

A cybersecurity research prototype that models computer networks as graphs and uses
Graph Neural Networks (GNNs) to predict likely multi-step attack paths from an entry
point toward critical assets.

## ⚠️ Research Disclaimer

This system is designed for **defensive cybersecurity research only**.
All datasets are synthetic or authorized. No attacks are performed against real systems.
All synthetic data is explicitly labeled as such.

---

## Project Pipeline

```
Network Topology + Vulnerabilities + Network Telemetry + Historical Attacks
        ↓
    Graph Construction
        ↓
    Graph Features
        ↓
    GNN (GraphSAGE)
        ↓
    Attack Probability Prediction
        ↓
    Candidate Attack Paths
        ↓
    Risk Scoring
        ↓
    Top-K Attack Paths
        ↓
    Critical Assets at Risk
        ↓
    Explainable AI
        ↓
    Proactive Defense Dashboard
```

---

## Technology Stack

| Layer        | Technology                              |
|--------------|-----------------------------------------|
| ML/AI        | Python, PyTorch, PyTorch Geometric      |
| Graph        | NetworkX                                |
| Data         | NumPy, Pandas, scikit-learn             |
| Backend      | FastAPI, Uvicorn                        |
| Frontend     | React, Tailwind CSS, Cytoscape.js       |

---

## Phase Status

- [x] **Phase 1**: Project Foundation & Graph Construction
- [x] **Phase 2**: GNN Input & Graph Data Pipeline
- [ ] **Phase 3**: GraphSAGE Model & Training
- [ ] **Phase 4**: Attack Path Prediction
- [ ] **Phase 5**: Risk Scoring & Top-K Paths
- [ ] **Phase 6**: Explainable AI
- [ ] **Phase 7**: FastAPI Backend
- [ ] **Phase 8**: React Dashboard

---

## Local Setup

```bash
cd d:\AI-Project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
d:\AI-Project\
├── backend/
│   ├── core/
│   │   ├── data_models.py        # Node, Edge, Vulnerability, Network data models
│   │   ├── graph_builder.py      # Graph construction from network topology
│   │   ├── feature_extractor.py  # Graph feature extraction (43 features)
│   │   └── synthetic_data.py     # Synthetic network generator (labeled synthetic)
│   ├── ml/
│   │   └── dataset.py            # PyG Data conversion, labels, splitting (Phase 2)
│   ├── gnn/                      # (Phase 3+)
│   ├── prediction/               # (Phase 4+)
│   └── api/                      # (Phase 7+)
├── tests/
│   ├── test_data_models.py
│   ├── test_graph_builder.py
│   ├── test_feature_extractor.py
│   ├── test_synthetic_data.py
│   └── test_dataset.py           # Phase 2 dataset tests
├── data/
│   └── synthetic/                # Generated synthetic datasets
├── validate_phase1.py
├── requirements.txt
└── README.md
```
