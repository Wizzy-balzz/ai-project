import { create } from 'zustand';
import {
  ScenarioSummary,
  AttackTrajectory,
  SimulationResult,
  fetchScenarios,
  fetchPaths,
  runSimulation,
} from '../api/client';

export type UIStage = 'HERO' | 'SHOWCASE' | 'INSPECTOR' | 'COLLAGE' | 'SIMULATOR';

interface ScenarioState {
  currentStage: UIStage;
  scenarios: ScenarioSummary[];
  selectedScenarioId: string | null;
  selectedScenario: ScenarioSummary | null;
  
  // Attack Paths
  trajectories: AttackTrajectory[];
  activeTrajectoryId: string | null;
  activeTrajectory: AttackTrajectory | null;
  
  // Inspected Node
  inspectedNodeId: string | null;
  
  // What-If Defense State
  isolatedNodeIds: string[];
  blockedEdgeIds: string[];
  patchedCves: string[];
  simulationResult: SimulationResult | null;
  isSimulating: boolean;

  // Actions
  setStage: (stage: UIStage) => void;
  loadScenarios: () => Promise<void>;
  selectScenario: (id: string) => Promise<void>;
  selectTrajectory: (id: string | null) => void;
  setInspectedNodeId: (id: string | null) => void;
  
  // Simulation Actions
  toggleNodeIsolation: (nodeId: string) => void;
  toggleEdgeBlock: (edgeId: string) => void;
  togglePatchCve: (cve: string) => void;
  executeSimulation: () => Promise<void>;
  resetDefenses: () => void;
}

export const useScenarioStore = create<ScenarioState>((set, get) => ({
  currentStage: 'HERO',
  scenarios: [],
  selectedScenarioId: null,
  selectedScenario: null,
  
  trajectories: [],
  activeTrajectoryId: null,
  activeTrajectory: null,
  
  inspectedNodeId: null,
  
  isolatedNodeIds: [],
  blockedEdgeIds: [],
  patchedCves: [],
  simulationResult: null,
  isSimulating: false,

  setStage: (stage) => set({ currentStage: stage }),

  loadScenarios: async () => {
    try {
      const scenarios = await fetchScenarios();
      set({ scenarios });
      if (scenarios.length > 0 && !get().selectedScenarioId) {
        await get().selectScenario(scenarios[0].id);
      }
    } catch (e) {
      console.error('Failed to load scenarios:', e);
    }
  },

  selectScenario: async (id: string) => {
    const sc = get().scenarios.find((s) => s.id === id) || null;
    set({
      selectedScenarioId: id,
      selectedScenario: sc,
      activeTrajectoryId: null,
      activeTrajectory: null,
      simulationResult: null,
      isolatedNodeIds: [],
      blockedEdgeIds: [],
      patchedCves: [],
    });

    try {
      const pathsData = await fetchPaths(id, 8);
      const topPaths = pathsData.paths || [];
      const topOne = topPaths.length > 0 ? topPaths[0] : null;
      set({
        trajectories: topPaths,
        activeTrajectoryId: topOne ? topOne.id : null,
        activeTrajectory: topOne,
        inspectedNodeId: topOne && topOne.node_ids.length > 0 ? topOne.node_ids[0] : null,
      });
    } catch (e) {
      console.error('Failed to load paths for scenario:', e);
    }
  },

  selectTrajectory: (id: string | null) => {
    const traj = get().trajectories.find((t) => t.id === id) || null;
    set({
      activeTrajectoryId: id,
      activeTrajectory: traj,
      inspectedNodeId: traj && traj.node_ids.length > 0 ? traj.node_ids[0] : get().inspectedNodeId,
    });
  },

  setInspectedNodeId: (id: string | null) => set({ inspectedNodeId: id }),

  toggleNodeIsolation: (nodeId: string) => {
    const current = get().isolatedNodeIds;
    const exists = current.includes(nodeId);
    set({
      isolatedNodeIds: exists ? current.filter((id) => id !== nodeId) : [...current, nodeId],
    });
  },

  toggleEdgeBlock: (edgeId: string) => {
    const current = get().blockedEdgeIds;
    const exists = current.includes(edgeId);
    set({
      blockedEdgeIds: exists ? current.filter((id) => id !== edgeId) : [...current, edgeId],
    });
  },

  togglePatchCve: (cve: string) => {
    const current = get().patchedCves;
    const exists = current.includes(cve);
    set({
      patchedCves: exists ? current.filter((c) => c !== cve) : [...current, cve],
    });
  },

  executeSimulation: async () => {
    const scenarioId = get().selectedScenarioId;
    if (!scenarioId) return;

    set({ isSimulating: true });
    try {
      const result = await runSimulation(scenarioId, {
        isolated_node_ids: get().isolatedNodeIds,
        blocked_edge_ids: get().blockedEdgeIds,
        patched_cves: get().patchedCves,
      });
      set({ simulationResult: result, isSimulating: false });
    } catch (e) {
      console.error('Simulation failed:', e);
      set({ isSimulating: false });
    }
  },

  resetDefenses: () => {
    set({
      isolatedNodeIds: [],
      blockedEdgeIds: [],
      patchedCves: [],
      simulationResult: null,
    });
  },
}));
