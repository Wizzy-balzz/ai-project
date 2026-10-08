# Graph-Based Attack Path Predictor AI — Operational Manual

## 1. System Overview
The **Graph-Based Attack Path Predictor AI** is an enterprise-grade cybersecurity system designed to model, detect, and proactively mitigate multi-hop network intrusion routes before perimeter breach. It combines:
- **GraphSAGE Inductive Graph Neural Network (GNN)** to compute structural node compromise probabilities.
- **Deterministic Bounded Depth-First Search (DFS)** to enumerate non-cyclical multi-hop lateral movement trajectories.
- **Deterministic Risk Engine**:
  $$\text{Likelihood} = (0.35 \cdot S_{\text{node}} + 0.35 \cdot S_{\text{edge}} + 0.30 \cdot S_{\text{vuln}}) \cdot 0.95^{\text{hops}-1}$$
  $$\text{Impact} = 0.40 + 0.60 \cdot I_{\text{target}}$$
  $$\text{Risk Score} = \text{clamp}(\text{Likelihood} \times \text{Impact}, 0.0, 1.0)$$
- **Explainable AI (XAI)**: Generates human-readable kill-chain narratives, factor attribution weights, and chokepoint graph cuts.
- **Luxury Editorial Frontend**: Powered by React, Vite, Tailwind CSS, `@xyflow/react`, and Framer Motion following TasteWinner editorial art direction.

---

## 2. Prerequisites & Environment Setup

### System Requirements
- **Node.js**: v18.0.0 or higher (v20+ recommended)
- **Python**: 3.10 to 3.14
- **Operating System**: Linux (Ubuntu/Debian, Pop!_OS, RHEL, Arch), macOS, or Windows WSL2

### Quick One-Step Dependency Installation
From the project root:
```bash
npm run install:all
```
This script automatically:
1. Installs root orchestration dependencies (`concurrently`).
2. Installs frontend packages (`@xyflow/react`, `framer-motion`, `zustand`, `@tanstack/react-query`, `lucide-react`, `tailwindcss`, `canvas-confetti`).
3. Installs backend dependencies from `backend/requirements.txt` into the Python environment.

---

## 3. PyTorch & PyTorch Geometric Setup (CPU Optimized)
If configuring PyTorch manually or on low-bandwidth / headless systems without NVIDIA CUDA, install the lightweight CPU wheel:
```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages
python3 -m pip install torch-geometric --break-system-packages
```
For managed Linux distributions enforcing PEP 668:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

---

## 4. Seeding Demo Scenarios
To populate the database with the pre-calculated enterprise topology **"Operation Golden Vineyard"** (12-node multi-tier DMZ, internal routing proxy, core database, backup vault, and pre-scored critical trajectories):
```bash
npm run seed
```
Or directly:
```bash
python3 backend/seed_demo_data.py
```
This creates or updates `attack_predictor.db` and immediately trains/evaluates the candidate attack paths.

---

## 5. Execution & Service Verification

### Full-Stack Single Execution (Recommended)
Run both backend FastAPI service and frontend Vite development server concurrently:
```bash
npm run start:all
```
Output will display:
- `[BACKEND]` running on `http://127.0.0.1:8000` (API & Swagger Docs at `http://127.0.0.1:8000/docs`).
- `[FRONTEND]` running on `http://localhost:5173`.

### Running Components Individually
- **Backend Only**:
  ```bash
  npm run start:backend
  ```
- **Frontend Only**:
  ```bash
  npm run start:frontend
  ```

---

## 6. Service Health Verification
To verify the backend health and GNN model checkpoint:
```bash
curl -s http://127.0.0.1:8000/api/health
```
Expected JSON response:
```json
{
  "status": "online",
  "version": "1.0.0",
  "gnn_checkpoint_present": true,
  "architecture": "GraphSAGE 2-Layer Inductive GNN",
  "scoring_formula": "Risk = Likelihood x Impact",
  "research_disclaimer": "Synthetic defensive research only. No real systems involved."
}
```

---

## 7. Operational Workflow in the UI
1. **Stage 1 (Editorial Hero)**: View panoramic golden-hour dusk banner, explore curated enterprise scenarios, and click **[ Start Attack Path Exploration ]**.
2. **Stage 2 (Trajectories)**: Drag through horizontal bottle-style carousel displaying Top-K ranked attack trajectories with real-time SVG glowing flow paths on the canvas.
3. **Stage 3 (Deep Specification)**: Inspect specific nodes, review "The Story" kill-chain pivot narrative, and analyze the 5-dot characteristics meter matrix.
4. **Stage 4 (Collage)**: Review convergence chokepoints where multi-hop lateral routes intersect, GNN tensor distributions, and crown jewel assets.
5. **Stage 5 (What-If Simulator)**: Isolate compromised nodes or patch CVEs, execute counterfactual simulation, and watch network risk drop with visual progress gauges and confetti.
6. **Canvas Interactions**: Hover over any node to inspect, isolate, or delete with clean database cascade. Click on edge pills to toggle firewall block rules.
