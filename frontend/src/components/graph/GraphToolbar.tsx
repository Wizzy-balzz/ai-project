import React, { useState } from 'react';
import { useReactFlow } from '@xyflow/react';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
  Trash,
  Plus,
  Link,
  X,
  Server,
  Network,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useGraphStore } from '../../store/graphStore';
import { useScenarioStore } from '../../store/scenarioStore';

export const GraphToolbar: React.FC = () => {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const { selectedScenarioId, selectScenario } = useScenarioStore();
  const { nodes, loadGraph, clearCanvas, addNode, addEdge } = useGraphStore();

  // Modals state
  const [isAddNodeOpen, setIsAddNodeOpen] = useState(false);
  const [isAddEdgeOpen, setIsAddEdgeOpen] = useState(false);

  // Node form
  const [nodeIp, setNodeIp] = useState('10.0.1.50');
  const [nodeHostname, setNodeHostname] = useState('app-server-01');
  const [nodeRole, setNodeRole] = useState('server');
  const [nodeOs, setNodeOs] = useState('linux');
  const [nodeCvss, setNodeCvss] = useState(7.5);
  const [nodeAssetValue, setNodeAssetValue] = useState(0.8);
  const [nodeIsEntry, setNodeIsEntry] = useState(false);
  const [nodeIsTarget, setNodeIsTarget] = useState(false);

  // Edge form
  const [sourceId, setSourceId] = useState('');
  const [targetId, setTargetId] = useState('');
  const [edgePort, setEdgePort] = useState('443');
  const [edgeType, setEdgeType] = useState('network_access');
  const [edgeFirewalled, setEdgeFirewalled] = useState(false);

  const handleReset = () => {
    if (selectedScenarioId) {
      loadGraph(selectedScenarioId);
      setTimeout(() => fitView({ padding: 0.2 }), 200);
    }
  };

  const handleClear = () => {
    if (
      selectedScenarioId &&
      confirm('Are you sure you want to clear all nodes and edges to start on an empty canvas?')
    ) {
      clearCanvas(selectedScenarioId);
    }
  };

  const handleCreateNode = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedScenarioId || !nodeIp.trim()) return;

    try {
      await addNode(selectedScenarioId, {
        ip_address: nodeIp.trim(),
        hostname: nodeHostname.trim() || nodeIp.trim(),
        role: nodeRole,
        os_type: nodeOs,
        cvss_score: Number(nodeCvss),
        asset_value: Number(nodeAssetValue),
        is_entry: Boolean(nodeIsEntry),
        is_target: Boolean(nodeIsTarget),
        vulnerabilities: nodeCvss >= 7.0 ? ['SYN-2026-1042'] : [],
      });
      setIsAddNodeOpen(false);
      // Refresh scenario paths
      await selectScenario(selectedScenarioId);
    } catch (err) {
      console.error('Failed to create node:', err);
    }
  };

  const handleCreateEdge = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedScenarioId || !sourceId || !targetId) return;

    try {
      await addEdge(selectedScenarioId, {
        source_id: sourceId,
        target_id: targetId,
        port: edgePort,
        edge_type: edgeType,
        is_firewalled: edgeFirewalled,
        is_encrypted: edgePort === '443' || edgePort === '22',
      });
      setIsAddEdgeOpen(false);
      // Refresh scenario paths
      await selectScenario(selectedScenarioId);
    } catch (err) {
      console.error('Failed to create edge:', err);
    }
  };

  return (
    <>
      <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-champagne/20 bg-velvetOlive/90 backdrop-blur-2xl shadow-2xl">
        <button
          onClick={() => zoomIn()}
          title="Zoom In"
          className="p-1.5 rounded-full text-champagne/80 hover:text-vintageGold hover:bg-champagne/10 transition-colors"
        >
          <ZoomIn className="w-4 h-4" />
        </button>

        <button
          onClick={() => zoomOut()}
          title="Zoom Out"
          className="p-1.5 rounded-full text-champagne/80 hover:text-vintageGold hover:bg-champagne/10 transition-colors"
        >
          <ZoomOut className="w-4 h-4" />
        </button>

        <div className="w-[1px] h-4 bg-champagne/15 mx-1" />

        <button
          onClick={() => fitView({ padding: 0.2, duration: 400 })}
          title="Fit View"
          className="p-1.5 rounded-full text-champagne/80 hover:text-vintageGold hover:bg-champagne/10 transition-colors"
        >
          <Maximize2 className="w-4 h-4" />
        </button>

        <button
          onClick={handleReset}
          title="Reset & Relayout"
          className="p-1.5 rounded-full text-champagne/80 hover:text-vintageGold hover:bg-champagne/10 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        <div className="w-[1px] h-4 bg-champagne/15 mx-1" />

        {/* Add Node Button */}
        <button
          onClick={() => {
            setNodeIp(`10.0.1.${nodes.length + 10}`);
            setNodeHostname(`host-${nodes.length + 1}`);
            setIsAddNodeOpen(true);
          }}
          title="Add Node to Canvas"
          className="px-2.5 py-1 rounded-full text-xs font-mono font-medium text-vintageGold hover:bg-vintageGold/15 border border-vintageGold/30 transition-all flex items-center gap-1"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Node</span>
        </button>

        {/* Add Edge Button */}
        <button
          onClick={() => {
            if (nodes.length >= 2) {
              setSourceId(nodes[0].id);
              setTargetId(nodes[1].id);
            }
            setIsAddEdgeOpen(true);
          }}
          disabled={nodes.length < 2}
          title={nodes.length < 2 ? 'Need at least 2 nodes to connect' : 'Add Edge between Nodes'}
          className={`px-2.5 py-1 rounded-full text-xs font-mono font-medium border transition-all flex items-center gap-1 ${
            nodes.length < 2
              ? 'opacity-40 cursor-not-allowed border-champagne/10 text-champagne/40'
              : 'text-champagne/90 hover:text-vintageGold hover:bg-champagne/10 border-champagne/25'
          }`}
        >
          <Link className="w-3.5 h-3.5" />
          <span>Edge</span>
        </button>

        <div className="w-[1px] h-4 bg-champagne/15 mx-1" />

        <button
          onClick={handleClear}
          title="Clear Canvas (Empty Blank Scenario)"
          className="p-1.5 rounded-full text-champagne/50 hover:text-crimsonVermillion hover:bg-crimsonVermillion/15 transition-colors"
        >
          <Trash className="w-4 h-4" />
        </button>
      </div>

      {/* Add Node Modal */}
      <AnimatePresence>
        {isAddNodeOpen && (
          <div className="fixed inset-0 z-50 bg-cellarNoir/80 backdrop-blur-md flex items-center justify-center p-4">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-lg rounded-3xl p-6 border border-champagne/20 bg-velvetOlive shadow-2xl text-champagne"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Server className="w-5 h-5 text-vintageGold" />
                  <h3 className="text-xl font-serif">Add Network Node</h3>
                </div>
                <button
                  onClick={() => setIsAddNodeOpen(false)}
                  className="p-1 rounded-full hover:bg-champagne/10 text-champagne/60 hover:text-champagne"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <form onSubmit={handleCreateNode} className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                      IP Address
                    </label>
                    <input
                      type="text"
                      required
                      value={nodeIp}
                      onChange={(e) => setNodeIp(e.target.value)}
                      placeholder="10.0.1.15"
                      className="w-full px-3 py-2 rounded-xl border border-champagne/20 bg-cellarNoir text-champagne text-xs font-mono focus:outline-none focus:border-vintageGold"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                      Hostname / Label
                    </label>
                    <input
                      type="text"
                      value={nodeHostname}
                      onChange={(e) => setNodeHostname(e.target.value)}
                      placeholder="db-primary-01"
                      className="w-full px-3 py-2 rounded-xl border border-champagne/20 bg-cellarNoir text-champagne text-xs font-mono focus:outline-none focus:border-vintageGold"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                      Role / Type
                    </label>
                    <select
                      value={nodeRole}
                      onChange={(e) => setNodeRole(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl border border-champagne/20 bg-cellarNoir text-champagne text-xs font-mono focus:outline-none focus:border-vintageGold"
                    >
                      <option value="workstation">Workstation</option>
                      <option value="server">Server</option>
                      <option value="database">Database</option>
                      <option value="router">Router</option>
                      <option value="firewall">Firewall</option>
                      <option value="critical_asset">Critical Asset</option>
                      <option value="iot_device">IoT Device</option>
                      <option value="internet">Internet Gateway</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                      Operating System
                    </label>
                    <select
                      value={nodeOs}
                      onChange={(e) => setNodeOs(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl border border-champagne/20 bg-cellarNoir text-champagne text-xs font-mono focus:outline-none focus:border-vintageGold"
                    >
                      <option value="linux">Linux</option>
                      <option value="windows">Windows</option>
                      <option value="cisco_ios">Cisco IOS</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                      CVSS Vulnerability ({nodeCvss})
                    </label>
                    <input
                      type="range"
                      min="0.0"
                      max="10.0"
                      step="0.1"
                      value={nodeCvss}
                      onChange={(e) => setNodeCvss(parseFloat(e.target.value))}
                      className="w-full accent-vintageGold cursor-pointer"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                      Asset Value ({nodeAssetValue})
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={nodeAssetValue}
                      onChange={(e) => setNodeAssetValue(parseFloat(e.target.value))}
                      className="w-full accent-vintageGold cursor-pointer"
                    />
                  </div>
                </div>

                <div className="flex items-center gap-6 pt-1">
                  <label className="flex items-center gap-2 cursor-pointer text-xs font-mono text-champagne/90">
                    <input
                      type="checkbox"
                      checked={nodeIsEntry}
                      onChange={(e) => setNodeIsEntry(e.target.checked)}
                      className="rounded border-champagne/30 text-vintageGold focus:ring-vintageGold accent-vintageGold"
                    />
                    <span>External Entry Point</span>
                  </label>

                  <label className="flex items-center gap-2 cursor-pointer text-xs font-mono text-champagne/90">
                    <input
                      type="checkbox"
                      checked={nodeIsTarget}
                      onChange={(e) => setNodeIsTarget(e.target.checked)}
                      className="rounded border-champagne/30 text-crimsonVermillion focus:ring-crimsonVermillion accent-crimsonVermillion"
                    />
                    <span>Crown Jewel Target</span>
                  </label>
                </div>

                <div className="flex items-center justify-end gap-3 pt-3 border-t border-champagne/10">
                  <button
                    type="button"
                    onClick={() => setIsAddNodeOpen(false)}
                    className="px-4 py-2 rounded-full border border-champagne/15 text-xs text-champagne/60 hover:text-champagne"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-6 py-2 rounded-full bg-vintageGold text-cellarNoir text-xs font-semibold hover:bg-goldHoney transition-all shadow-md"
                  >
                    Add Node
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Add Edge Modal */}
      <AnimatePresence>
        {isAddEdgeOpen && (
          <div className="fixed inset-0 z-50 bg-cellarNoir/80 backdrop-blur-md flex items-center justify-center p-4">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="w-full max-w-lg rounded-3xl p-6 border border-champagne/20 bg-velvetOlive shadow-2xl text-champagne"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Network className="w-5 h-5 text-vintageGold" />
                  <h3 className="text-xl font-serif">Connect Nodes (Add Edge)</h3>
                </div>
                <button
                  onClick={() => setIsAddEdgeOpen(false)}
                  className="p-1 rounded-full hover:bg-champagne/10 text-champagne/60 hover:text-champagne"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <form onSubmit={handleCreateEdge} className="space-y-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                      Source Node
                    </label>
                    <select
                      value={sourceId}
                      onChange={(e) => setSourceId(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl border border-champagne/20 bg-cellarNoir text-champagne text-xs font-mono focus:outline-none focus:border-vintageGold"
                    >
                      {nodes.map((n) => (
                        <option key={n.id} value={n.id}>
                          {n.data.ip} ({n.data.role})
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                      Target Node
                    </label>
                    <select
                      value={targetId}
                      onChange={(e) => setTargetId(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl border border-champagne/20 bg-cellarNoir text-champagne text-xs font-mono focus:outline-none focus:border-vintageGold"
                    >
                      {nodes.map((n) => (
                        <option key={n.id} value={n.id} disabled={n.id === sourceId}>
                          {n.data.ip} ({n.data.role})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                      Port / Service
                    </label>
                    <input
                      type="text"
                      value={edgePort}
                      onChange={(e) => setEdgePort(e.target.value)}
                      placeholder="443"
                      className="w-full px-3 py-2 rounded-xl border border-champagne/20 bg-cellarNoir text-champagne text-xs font-mono focus:outline-none focus:border-vintageGold"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                      Connection Type
                    </label>
                    <select
                      value={edgeType}
                      onChange={(e) => setEdgeType(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl border border-champagne/20 bg-cellarNoir text-champagne text-xs font-mono focus:outline-none focus:border-vintageGold"
                    >
                      <option value="network_access">Network Access</option>
                      <option value="ssh">SSH (22)</option>
                      <option value="rdp">RDP (3389)</option>
                      <option value="database_link">Database Link (5432)</option>
                    </select>
                  </div>
                </div>

                <div className="pt-1">
                  <label className="flex items-center gap-2 cursor-pointer text-xs font-mono text-champagne/90">
                    <input
                      type="checkbox"
                      checked={edgeFirewalled}
                      onChange={(e) => setEdgeFirewalled(e.target.checked)}
                      className="rounded border-champagne/30 text-vintageGold focus:ring-vintageGold accent-vintageGold"
                    />
                    <span>Firewalled Edge (Filter Traversal)</span>
                  </label>
                </div>

                <div className="flex items-center justify-end gap-3 pt-3 border-t border-champagne/10">
                  <button
                    type="button"
                    onClick={() => setIsAddEdgeOpen(false)}
                    className="px-4 py-2 rounded-full border border-champagne/15 text-xs text-champagne/60 hover:text-champagne"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-6 py-2 rounded-full bg-vintageGold text-cellarNoir text-xs font-semibold hover:bg-goldHoney transition-all shadow-md"
                  >
                    Connect Edge
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
};
