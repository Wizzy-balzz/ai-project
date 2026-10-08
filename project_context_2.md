# Project Context: Graph-Based Attack Path Predictor AI (Full-Stack Edition)

## 1. System Summary
The **Graph-Based Attack Path Predictor AI** is an end-to-end cybersecurity intelligence platform designed for defensive research, proactive risk management, and adversary kill-chain modeling.

### Core Architecture
- **Phase 1: Graph Representation**: Directed multigraphs representing enterprise networks (`nx.DiGraph`), with synthetic CVE tags (`SYN-2026-XXXX`).
- **Phase 2: PyTorch Geometric Pipeline**: Inductive Graph Data extraction with 43-dimensional node feature vectors and 9-dimensional edge vectors.
- **Phase 3: GraphSAGE GNN**: 2-layer GraphSAGE architecture with LayerNorm/BatchNorm predicting binary compromise probability $P(\text{attack\_involved}) \in [0.0, 1.0]$.
- **Phase 4: Attack Path Generator**: Deterministic bounded Depth-First Search (DFS) with lexicographical sorting originating exclusively from entry nodes.
- **Phase 5: Deterministic Cybersecurity Risk Scoring**:
  $$\text{Likelihood} = (0.35 \cdot S_{\text{node}} + 0.35 \cdot S_{\text{edge}} + 0.30 \cdot S_{\text{vuln}}) \cdot 0.95^{\text{hops}-1}$$
  $$\text{Impact} = 0.40 + 0.60 \cdot I_{\text{target}}$$
  $$\text{Risk Score} = \text{clamp}(\text{Likelihood} \times \text{Impact}, 0.0, 1.0)$$
- **Phase 6: Explainable AI & Simulator**: Step-by-step kill-chain narratives, factor attribution, graph chokepoint min-cuts, and counterfactual defense simulations.
- **Phase 7: Enterprise REST API**: FastAPI backend under `/api/v1` backed by SQLite ORM with complete cascade deletions and audit logging.
- **Phase 8: Luxury Editorial Frontend**: Built with React, Vite, Tailwind CSS, `@xyflow/react`, and Framer Motion following TasteWinner editorial art direction.

---

## 2. Directory Structure
```
ai-project/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── explain.py
│   │   │   ├── nodes.py
│   │   │   ├── edges.py
│   │   │   ├── paths.py
│   │   │   ├── router.py
│   │   │   ├── scenarios.py
│   │   │   └── simulate.py
│   │   ├── models/
│   │   │   ├── attack_path.py
│   │   │   ├── audit.py
│   │   │   ├── base.py
│   │   │   ├── defense.py
│   │   │   ├── edge.py
│   │   │   ├── node.py
│   │   │   └── scenario.py
│   │   ├── schemas/
│   │   │   ├── graph.py
│   │   │   ├── path.py
│   │   │   ├── scenario.py
│   │   │   └── simulation.py
│   │   ├── services/
│   │   │   └── graph_service.py
│   │   ├── config.py
│   │   ├── database.py
│   │   └── main.py
│   ├── core/
│   │   ├── data_models.py
│   │   ├── feature_extractor.py
│   │   ├── graph_builder.py
│   │   └── synthetic_data.py
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── evaluator.py
│   │   ├── explainer.py
│   │   ├── model.py
│   │   ├── path_generator.py
│   │   ├── path_models.py
│   │   ├── risk_scorer.py
│   │   └── trainer.py
│   ├── requirements.txt
│   └── seed_demo_data.py
├── frontend/
│   ├── src/
│   │   ├── api/client.ts
│   │   ├── components/
│   │   │   ├── collage/TopologyCollage.tsx
│   │   │   ├── graph/
│   │   │   │   ├── CustomEdge.tsx
│   │   │   │   ├── CustomNode.tsx
│   │   │   │   ├── GraphToolbar.tsx
│   │   │   │   └── NetworkGraphCanvas.tsx
│   │   │   ├── home/EditorialHero.tsx
│   │   │   ├── inspector/
│   │   │   │   ├── CharacteristicsMeter.tsx
│   │   │   │   └── NodeInspectorView.tsx
│   │   │   ├── showcase/
│   │   │   │   ├── TrajectoryCard.tsx
│   │   │   │   └── TrajectoryCarousel.tsx
│   │   │   ├── simulation/
│   │   │   │   ├── RiskDropGauge.tsx
│   │   │   │   └── WhatIfSimulatorPanel.tsx
│   │   │   └── xai/
│   │   │       ├── ChokepointCutPanel.tsx
│   │   │       ├── FactorAttributionChart.tsx
│   │   │       └── KillChainNarrativeView.tsx
│   │   ├── store/
│   │   │   ├── graphStore.ts
│   │   │   └── scenarioStore.ts
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── main.tsx
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.ts
│   └── vite.config.ts
├── .env.example
├── .gitignore
├── instructions.md
├── instructions.mf
├── package.json
└── README.md
```

---

## 3. Single Execution Command
```bash
npm run start:all
```
Executes `concurrently -n "BACKEND,FRONTEND" -c "amber,emerald" "npm run start:backend" "npm run start:frontend"`.
