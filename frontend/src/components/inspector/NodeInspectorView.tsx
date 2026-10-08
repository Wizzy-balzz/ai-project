import React from 'react';
import { motion } from 'framer-motion';
import {
  Server,
  Shield,
  ShieldAlert,
  ArrowRight,
  Layers,
  Cpu,
  Terminal,
  Activity,
  CheckCircle,
  AlertTriangle,
} from 'lucide-react';
import { useScenarioStore } from '../../store/scenarioStore';
import { useGraphStore } from '../../store/graphStore';
import { CharacteristicsMeter } from './CharacteristicsMeter';

export const NodeInspectorView: React.FC = () => {
  const {
    activeTrajectory,
    inspectedNodeId,
    setInspectedNodeId,
    setStage,
    toggleNodeIsolation,
    isolatedNodeIds,
    selectedScenarioId,
  } = useScenarioStore();

  const { nodes, deleteNode } = useGraphStore();

  // Find inspected node object or fallback to first node in active trajectory
  const activeNodeId = inspectedNodeId || (activeTrajectory ? activeTrajectory.node_ids[0] : null);
  const activeNode = nodes.find((n) => n.id === activeNodeId) || nodes[0];

  if (!activeNode) {
    return (
      <div className="py-24 text-center">
        <h3 className="text-2xl font-serif text-champagne mb-2">No Node Selected</h3>
        <p className="text-sm text-champagne/60">Select an attack trajectory or node to inspect specifications.</p>
      </div>
    );
  }

  const isIsolated = isolatedNodeIds.includes(activeNode.id);
  const cvss = activeNode.data.cvss || 7.5;
  const gnnProb = activeNode.data.gnn_probability || 0.85;
  const lateralVel = activeNode.data.lateral_velocity || 0.65;
  const assetVal = activeNode.data.asset_value || 0.8;

  return (
    <div className="relative py-10 px-8">
      {/* Header Breadcrumb */}
      <div className="flex items-center justify-between pb-6 mb-8 border-b border-champagne/10">
        <div>
          <span className="text-xs uppercase font-mono tracking-[0.3em] text-vintageGold block mb-1">
            STAGE 3 • DEEP SPECIFICATION
          </span>
          <h2 className="text-3xl md:text-4xl font-serif text-champagne font-normal">
            Editorial Node & Path <span className="italic text-vintageGold">Inspection.</span>
          </h2>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setStage('COLLAGE')}
            className="px-5 py-2.5 rounded-full border border-champagne/20 bg-velvetOlive/60 hover:bg-velvetOlive text-champagne text-xs font-medium transition-all"
          >
            Chokepoint Collage
          </button>
          <button
            onClick={() => setStage('SIMULATOR')}
            className="px-6 py-2.5 rounded-full bg-vintageGold text-cellarNoir text-xs font-medium tracking-wide hover:bg-goldHoney transition-all shadow-md"
          >
            Open Simulator
          </button>
        </div>
      </div>

      {/* Two-Column Editorial Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Isometric / Clean Vector Topology Preview */}
        <div className="lg:col-span-5 rounded-3xl p-6 border border-champagne/15 bg-gradient-to-b from-velvetOlive/80 to-cellarNoir/80 backdrop-blur-2xl">
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-mono uppercase text-champagne/50">
              Topology Vector Projection
            </span>
            <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono uppercase border ${
              activeNode.data.is_entry
                ? 'text-crimsonVermillion border-crimsonVermillion/40 bg-crimsonVermillion/10'
                : activeNode.data.is_target
                ? 'text-vintageGold border-vintageGold/40 bg-vintageGold/10'
                : 'text-champagne/60 border-champagne/20'
            }`}>
              {activeNode.data.is_entry ? 'Perimeter Entry' : activeNode.data.is_target ? 'Critical Target' : 'Pivot Asset'}
            </span>
          </div>

          {/* Clean Graphic Wireframe Node Visualization */}
          <div className="relative h-64 rounded-2xl bg-cellarNoir/90 border border-champagne/10 flex items-center justify-center overflow-hidden my-4">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(212,175,55,0.08)_0%,transparent_70%)]" />
            
            {/* Glowing Node Core */}
            <motion.div
              animate={{
                scale: [1, 1.05, 1],
                boxShadow: [
                  '0 0 20px rgba(212,175,55,0.2)',
                  '0 0 35px rgba(212,175,55,0.4)',
                  '0 0 20px rgba(212,175,55,0.2)',
                ],
              }}
              transition={{ repeat: Infinity, duration: 4 }}
              className="relative z-10 w-28 h-28 rounded-2xl border-2 border-vintageGold/80 bg-velvetOlive/90 flex flex-col items-center justify-center p-3 text-center"
            >
              <Server className="w-8 h-8 text-vintageGold mb-1.5" />
              <span className="text-[11px] font-mono text-champagne font-semibold truncate max-w-[90px]">
                {activeNode.data.ip}
              </span>
              <span className="text-[9px] uppercase tracking-wider text-champagne/50">
                {activeNode.data.role}
              </span>
            </motion.div>

            {/* Simulated Edge Trajectory Particles */}
            <div className="absolute left-8 top-1/2 -translate-y-1/2 w-16 h-0.5 bg-gradient-to-r from-transparent to-vintageGold/60 border-b border-dashed border-vintageGold/40" />
            <div className="absolute right-8 top-1/2 -translate-y-1/2 w-16 h-0.5 bg-gradient-to-l from-transparent to-vintageGold/60 border-b border-dashed border-vintageGold/40" />
          </div>

          {/* Path Hop Selector Carousel */}
          {activeTrajectory && (
            <div>
              <span className="text-xs uppercase font-mono tracking-wider text-champagne/60 block mb-2">
                Active Trajectory Hop Sequence:
              </span>
              <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-none">
                {activeTrajectory.node_ids.map((nid, idx) => {
                  const isCur = nid === activeNode.id;
                  const nObj = nodes.find((n) => n.id === nid);
                  return (
                    <button
                      key={nid}
                      onClick={() => setInspectedNodeId(nid)}
                      className={`px-3 py-1.5 rounded-xl border text-xs font-mono transition-all flex items-center gap-1.5 whitespace-nowrap ${
                        isCur
                          ? 'border-vintageGold bg-vintageGold/20 text-vintageGold'
                          : 'border-champagne/15 bg-velvetOlive/40 text-champagne/70 hover:border-champagne/30'
                      }`}
                    >
                      <span className="text-[10px] text-champagne/40">#{idx + 1}</span>
                      <span>{nObj ? nObj.data.ip : nid}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: High-contrast Editorial Narrative & Characteristics Meter */}
        <div className="lg:col-span-7 rounded-3xl p-8 border border-champagne/15 bg-gradient-to-b from-velvetOlive/90 to-cellarNoir/90 backdrop-blur-2xl">
          <div className="flex items-start justify-between mb-4">
            <div>
              <span className="text-xs font-mono uppercase tracking-[0.2em] text-champagne/50 block mb-1">
                Asset Identifier
              </span>
              <h3 className="text-3xl font-serif text-champagne font-medium">
                Node: {activeNode.data.hostname || activeNode.data.ip}
              </h3>
            </div>

            <div className="text-right">
              <span className="px-3 py-1 rounded-full text-xs font-mono uppercase tracking-wider border font-semibold inline-block text-crimsonVermillion border-crimsonVermillion/40 bg-crimsonVermillion/10">
                CVSS {cvss.toFixed(1)} / 10
              </span>
              <span className="text-[11px] font-mono text-champagne/50 block mt-1">
                Asset Value {(assetVal * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Editorial Section: "The Story" */}
          <div className="mb-6 p-5 rounded-2xl border border-champagne/10 bg-cellarNoir/60">
            <h4 className="text-xs uppercase font-mono tracking-[0.25em] text-vintageGold mb-2">
              The Story • Kill-Chain Pivot Analysis
            </h4>
            <p className="text-sm text-champagne/80 font-light leading-relaxed">
              {activeTrajectory?.kill_chain_narrative || (
                `The asset ${activeNode.data.ip} executes role ${activeNode.data.role} within the protected perimeter.
                Possessing an exploit susceptibility score of ${cvss.toFixed(1)}, this node functions as an indispensable
                stepping stone, granting attackers access to downstream core databases and confidential vaults.`
              )}
            </p>
          </div>

          {/* 5-Level Filled-Dot Characteristics Meter Matrix */}
          <div className="p-5 rounded-2xl border border-champagne/10 bg-velvetOlive/40 mb-6">
            <h4 className="text-xs uppercase font-mono tracking-[0.25em] text-champagne/60 mb-3">
              Characteristics Dot / Meter Matrix
            </h4>
            <div className="space-y-1">
              <CharacteristicsMeter
                label="Flavor / Exploitability"
                rating={cvss / 2.0}
                description="Ease of arbitrary code execution across network boundary"
              />
              <CharacteristicsMeter
                label="Impact"
                rating={assetVal * 5.0}
                description="Consequence of full machine compromise on core business assets"
              />
              <CharacteristicsMeter
                label="Lateral Velocity"
                rating={lateralVel * 5.0}
                description="Pivot traversal speed to adjacent network segments"
              />
              <CharacteristicsMeter
                label="CVSS Base Vulnerability"
                rating={cvss / 2.0}
                description="Standard Common Vulnerability Scoring System baseline"
              />
              <CharacteristicsMeter
                label="GraphSAGE GNN Risk"
                rating={gnnProb * 5.0}
                description="Inductive Graph Neural Network node involvement probability"
              />
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex items-center gap-4 pt-4 border-t border-champagne/10">
            <button
              onClick={() => toggleNodeIsolation(activeNode.id)}
              className={`px-5 py-2.5 rounded-full text-xs font-medium tracking-wide transition-all border flex items-center gap-2 ${
                isIsolated
                  ? 'border-emerald-500/50 bg-emerald-500/20 text-emerald-400'
                  : 'border-vintageGold/50 bg-vintageGold/10 text-vintageGold hover:bg-vintageGold/20'
              }`}
            >
              <Shield className="w-3.5 h-3.5" />
              <span>{isIsolated ? 'Node Isolated (Active)' : 'Simulate Node Isolation'}</span>
            </button>

            {selectedScenarioId && (
              <button
                onClick={() => deleteNode(activeNode.id, selectedScenarioId)}
                className="px-5 py-2.5 rounded-full text-xs font-medium tracking-wide text-crimsonVermillion/80 border border-crimsonVermillion/30 hover:bg-crimsonVermillion/10 transition-colors"
              >
                Delete Node (Cascade)
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
