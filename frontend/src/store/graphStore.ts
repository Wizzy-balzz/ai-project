import { create } from 'zustand';
import {
  applyNodeChanges,
  applyEdgeChanges,
  NodeChange,
  EdgeChange,
} from '@xyflow/react';
import {
  ReactFlowNodeItem,
  ReactFlowEdgeItem,
  fetchTopology,
  deleteNode as apiDeleteNode,
  deleteEdge as apiDeleteEdge,
  toggleFirewallEdge as apiToggleFirewall,
  clearScenarioNodes as apiClearNodes,
} from '../api/client';

interface GraphState {
  nodes: ReactFlowNodeItem[];
  edges: ReactFlowEdgeItem[];
  isLoading: boolean;
  error: string | null;

  // Actions
  setNodes: (nodes: ReactFlowNodeItem[]) => void;
  setEdges: (edges: ReactFlowEdgeItem[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  
  loadGraph: (scenarioId: string, activePathId?: string) => Promise<void>;
  deleteNode: (nodeId: string, scenarioId: string) => Promise<void>;
  deleteEdge: (edgeId: string) => Promise<void>;
  toggleFirewall: (edgeId: string) => Promise<void>;
  clearCanvas: (scenarioId: string) => Promise<void>;
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  isLoading: false,
  error: null,

  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),

  onNodesChange: (changes) => {
    set({
      nodes: applyNodeChanges(changes, get().nodes as any) as any,
    });
  },

  onEdgesChange: (changes) => {
    set({
      edges: applyEdgeChanges(changes, get().edges as any) as any,
    });
  },

  loadGraph: async (scenarioId: string, activePathId?: string) => {
    set({ isLoading: true, error: null });
    try {
      const data = await fetchTopology(scenarioId, activePathId);
      set({
        nodes: data.nodes,
        edges: data.edges,
        isLoading: false,
      });
    } catch (e: any) {
      console.error('Failed to load topology:', e);
      set({ error: e.message || 'Error loading topology', isLoading: false });
    }
  },

  deleteNode: async (nodeId: string, scenarioId: string) => {
    // 1. Optimistically update local nodes and incident edges
    const currentNodes = get().nodes.filter((n) => n.id !== nodeId);
    const currentEdges = get().edges.filter(
      (e) => e.source !== nodeId && e.target !== nodeId
    );
    set({ nodes: currentNodes, edges: currentEdges });

    // 2. Call backend for clean cascade delete and path recalculation
    try {
      await apiDeleteNode(nodeId);
      // Reload topology to sync any fresh layouts and recalculated stats
      const data = await fetchTopology(scenarioId);
      set({ nodes: data.nodes, edges: data.edges });
    } catch (e) {
      console.error('Failed to delete node on backend:', e);
    }
  },

  deleteEdge: async (edgeId: string) => {
    const currentEdges = get().edges.filter((e) => e.id !== edgeId);
    set({ edges: currentEdges });
    try {
      await apiDeleteEdge(edgeId);
    } catch (e) {
      console.error('Failed to delete edge on backend:', e);
    }
  },

  toggleFirewall: async (edgeId: string) => {
    try {
      const res = await apiToggleFirewall(edgeId);
      if (res && res.topology) {
        set({ nodes: res.topology.nodes, edges: res.topology.edges });
      }
    } catch (e) {
      console.error('Failed to toggle firewall:', e);
    }
  },

  clearCanvas: async (scenarioId: string) => {
    set({ nodes: [], edges: [] });
    try {
      await apiClearNodes(scenarioId);
    } catch (e) {
      console.error('Failed to clear nodes:', e);
    }
  },
}));
