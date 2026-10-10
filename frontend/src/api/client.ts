import axios from 'axios';

export const apiClient = axios.create({
  baseURL: (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000/api/v1',
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor for global logging and error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const errorMsg = error.response?.data?.detail || error.message || 'Unknown network error';
    console.error('[API Error]:', errorMsg);
    return Promise.reject(error);
  }
);

// Types
export interface ScenarioSummary {
  id: string;
  name: string;
  description: string;
  topology_type: string;
  num_nodes: number;
  num_edges: number;
  is_synthetic: boolean;
  risk_score: number;
  critical_paths_count: number;
}

export interface NodeData {
  id: string;
  scenario_id: string;
  ip: string;
  role: string;
  os: string;
  cvss: number;
  asset_value: number;
  is_entry: boolean;
  is_target: boolean;
  hostname: string;
  vulnerabilities: string[];
  gnn_probability: number;
  lateral_velocity: number;
  threat_level: string;
  in_critical_path: boolean;
  is_isolated: boolean;
  [key: string]: any;
}

export interface EdgeData {
  id: string;
  scenario_id: string;
  source_id: string;
  target_id: string;
  edge_type: string;
  is_firewalled: boolean;
  is_encrypted: boolean;
  anomaly_score: number;
  port: string;
  in_critical_path: boolean;
  is_blocked: boolean;
  [key: string]: any;
}

export interface ReactFlowNodeItem {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: NodeData;
}

export interface ReactFlowEdgeItem {
  id: string;
  source: string;
  target: string;
  type: string;
  animated: boolean;
  data: EdgeData;
}

export interface TopologyResponse {
  scenario_id: string;
  scenario_name: string;
  nodes: ReactFlowNodeItem[];
  edges: ReactFlowEdgeItem[];
  stats: {
    num_nodes: number;
    num_edges: number;
    entry_points: number;
    critical_targets: number;
  };
}

export interface HopDetail {
  step_number: number;
  source_id: string;
  source_ip: string;
  source_role: string;
  target_id: string;
  target_ip: string;
  target_role: string;
  edge_type: string;
  vulnerability_cve: string;
  cvss_score: number;
  step_likelihood: number;
  action_narrative: string;
}

export interface FactorAttribution {
  gnn_weight: number;
  edge_weight: number;
  cvss_severity: number;
  node_susceptibility: number;
  edge_permeability: number;
  vulnerability_exploitability: number;
}

export interface AttackTrajectory {
  id: string;
  scenario_id: string;
  node_ids: string[];
  node_ips: string[];
  node_roles: string[];
  hops: number;
  likelihood: number;
  impact: number;
  risk_score: number;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  hop_details: HopDetail[];
  factor_attribution: FactorAttribution;
  kill_chain_narrative: string;
  chokepoints: string[];
}

export interface TopKPathsResponse {
  scenario_id: string;
  total_candidates: number;
  top_k_count: number;
  paths: AttackTrajectory[];
  max_risk_score: number;
  average_risk_score: number;
  critical_count: number;
  high_count: number;
}

export interface SimulationRequest {
  isolated_node_ids: string[];
  blocked_edge_ids: string[];
  patched_cves: string[];
}

export interface SimulationResult {
  scenario_id: string;
  original_risk: number;
  simulated_risk: number;
  risk_drop_percentage: number;
  total_original_paths: number;
  severed_paths_count: number;
  remaining_paths_count: number;
  active_defenses: Array<{ type: string; target_id: string; status: string }>;
  recommendation: string;
  is_mitigated: boolean;
}

// API functions
export const fetchScenarios = async (): Promise<ScenarioSummary[]> => {
  const { data } = await apiClient.get<ScenarioSummary[]>('/scenarios');
  return data;
};

export const fetchTopology = async (scenarioId: string, activePath?: string): Promise<TopologyResponse> => {
  const params = activePath ? { active_path: activePath } : {};
  const { data } = await apiClient.get<TopologyResponse>(`/scenarios/${scenarioId}/nodes`, { params });
  return data;
};

export const fetchPaths = async (scenarioId: string, k: number = 5): Promise<TopKPathsResponse> => {
  const { data } = await apiClient.get<TopKPathsResponse>(`/scenarios/${scenarioId}/paths?k=${k}`);
  return data;
};

export const deleteNode = async (nodeId: string): Promise<any> => {
  const { data } = await apiClient.delete(`/nodes/${nodeId}`);
  return data;
};

export const clearScenarioNodes = async (scenarioId: string): Promise<any> => {
  const { data } = await apiClient.post(`/scenarios/${scenarioId}/nodes/clear`);
  return data;
};

export const toggleFirewallEdge = async (edgeId: string): Promise<any> => {
  const { data } = await apiClient.patch(`/edges/${edgeId}/toggle-firewall`);
  return data;
};

export const deleteEdge = async (edgeId: string): Promise<any> => {
  const { data } = await apiClient.delete(`/edges/${edgeId}`);
  return data;
};

export const runSimulation = async (scenarioId: string, req: SimulationRequest): Promise<SimulationResult> => {
  const { data } = await apiClient.post<SimulationResult>(`/scenarios/${scenarioId}/simulate`, req);
  return data;
};

export const createScenario = async (name: string, description: string): Promise<any> => {
  const { data } = await apiClient.post('/scenarios', {
    name,
    description,
    topology_type: 'enterprise',
    is_synthetic: true,
  });
  return data;
};

export interface NodeCreatePayload {
  ip_address: string;
  role?: string;
  os_type?: string;
  cvss_score?: number;
  asset_value?: number;
  is_entry?: boolean;
  is_target?: boolean;
  hostname?: string;
  vulnerabilities?: string[];
  pos_x?: number;
  pos_y?: number;
}

export interface EdgeCreatePayload {
  source_id: string;
  target_id: string;
  edge_type?: string;
  is_firewalled?: boolean;
  is_encrypted?: boolean;
  anomaly_score?: number;
  port?: string;
}

export const createNode = async (scenarioId: string, node: NodeCreatePayload): Promise<TopologyResponse> => {
  const { data } = await apiClient.post<TopologyResponse>(`/scenarios/${scenarioId}/nodes`, node);
  return data;
};

export const createEdge = async (scenarioId: string, edge: EdgeCreatePayload): Promise<TopologyResponse> => {
  const { data } = await apiClient.post<TopologyResponse>(`/scenarios/${scenarioId}/edges`, edge);
  return data;
};

