import React from 'react';
import { motion } from 'framer-motion';
import {
  Shield,
  ShieldCheck,
  RotateCcw,
  Play,
  Layers,
  Lock,
  Flame,
  CheckCircle2,
  Sparkles,
} from 'lucide-react';
import { useScenarioStore } from '../../store/scenarioStore';
import { useGraphStore } from '../../store/graphStore';
import { RiskDropGauge } from './RiskDropGauge';
import { NetworkGraphCanvas } from '../graph/NetworkGraphCanvas';

export const WhatIfSimulatorPanel: React.FC = () => {
  const {
    selectedScenario,
    isolatedNodeIds,
    toggleNodeIsolation,
    blockedEdgeIds,
    toggleEdgeBlock,
    patchedCves,
    togglePatchCve,
    executeSimulation,
    resetDefenses,
    simulationResult,
    isSimulating,
    setStage,
  } = useScenarioStore();

  const { nodes, edges } = useGraphStore();

  const sampleCves = ['SYN-2026-9910', 'SYN-2026-7811', 'SYN-2026-1042', 'SYN-2026-8801'];

  return (
    <div className="relative py-10 px-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between pb-6 mb-8 border-b border-champagne/10 gap-4">
        <div>
          <span className="text-xs uppercase font-mono tracking-[0.3em] text-vintageGold block mb-1">
            STAGE 5 • PROACTIVE DEFENSE
          </span>
          <h2 className="text-4xl md:text-5xl font-serif text-champagne font-normal">
            What-If Defense <span className="italic text-vintageGold">Simulation.</span>
          </h2>
          <p className="text-sm text-champagne/60 font-light mt-1 max-w-xl">
            Simulate host isolation, edge firewall block rules, and CVE patches to measure counterfactual risk reduction before deploying changes.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={resetDefenses}
            className="px-4 py-2.5 rounded-full border border-champagne/20 bg-velvetOlive/50 hover:bg-velvetOlive text-champagne text-xs font-mono transition-all flex items-center gap-2"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Defenses</span>
          </button>

          <button
            onClick={executeSimulation}
            disabled={isSimulating}
            className="px-7 py-2.5 rounded-full bg-gradient-to-r from-vintageGold to-goldHoney text-cellarNoir text-xs font-semibold tracking-wide hover:opacity-95 transition-all shadow-lg flex items-center gap-2"
          >
            {isSimulating ? (
              <>
                <span className="w-3.5 h-3.5 rounded-full border-2 border-cellarNoir border-t-transparent animate-spin" />
                <span>Simulating...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-cellarNoir" />
                <span>Run Simulation</span>
              </>
            )}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start mb-8">
        {/* Left Column: Interactive Defense Controls */}
        <div className="lg:col-span-5 space-y-6">
          {/* Defense Option 1: Isolate Critical Nodes */}
          <div className="rounded-3xl p-6 border border-champagne/15 bg-velvetOlive/60 backdrop-blur-2xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-vintageGold" />
                <h4 className="text-sm font-serif text-champagne font-medium">
                  Host-Level Containment (Isolate Nodes)
                </h4>
              </div>
              <span className="text-[11px] font-mono text-champagne/50">
                {isolatedNodeIds.length} Active
              </span>
            </div>
            <p className="text-xs text-champagne/60 font-light mb-4">
              Sever all inbound and outbound edges to contain attacker pivots instantly.
            </p>

            <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
              {nodes.map((n) => {
                const isSelected = isolatedNodeIds.includes(n.id);
                return (
                  <button
                    key={n.id}
                    onClick={() => toggleNodeIsolation(n.id)}
                    className={`w-full p-2.5 rounded-xl border text-left text-xs font-mono transition-all flex items-center justify-between ${
                      isSelected
                        ? 'border-emerald-500/50 bg-emerald-500/15 text-emerald-400'
                        : 'border-champagne/10 bg-cellarNoir/60 text-champagne/70 hover:border-champagne/25'
                    }`}
                  >
                    <span>{n.data.ip} ({n.data.role})</span>
                    <span className="text-[10px] uppercase font-semibold">
                      {isSelected ? 'Isolated' : 'Normal'}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Defense Option 2: Patch CVEs */}
          <div className="rounded-3xl p-6 border border-champagne/15 bg-velvetOlive/60 backdrop-blur-2xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Lock className="w-4 h-4 text-vintageGold" />
                <h4 className="text-sm font-serif text-champagne font-medium">
                  Remediate Synthetic CVEs
                </h4>
              </div>
              <span className="text-[11px] font-mono text-champagne/50">
                {patchedCves.length} Patched
              </span>
            </div>
            <p className="text-xs text-champagne/60 font-light mb-4">
              Apply software hotfixes to drop target vulnerability scores to zero.
            </p>

            <div className="space-y-2">
              {sampleCves.map((cve) => {
                const isPatched = patchedCves.includes(cve);
                return (
                  <button
                    key={cve}
                    onClick={() => togglePatchCve(cve)}
                    className={`w-full p-2.5 rounded-xl border text-left text-xs font-mono transition-all flex items-center justify-between ${
                      isPatched
                        ? 'border-emerald-500/50 bg-emerald-500/15 text-emerald-400'
                        : 'border-champagne/10 bg-cellarNoir/60 text-champagne/70 hover:border-champagne/25'
                    }`}
                  >
                    <span>{cve}</span>
                    <span className="text-[10px] uppercase font-semibold">
                      {isPatched ? 'Patched' : 'Vulnerable'}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Simulation Result & Gauge */}
        <div className="lg:col-span-7 space-y-6">
          {simulationResult ? (
            <div>
              <RiskDropGauge
                originalRisk={simulationResult.original_risk}
                simulatedRisk={simulationResult.simulated_risk}
                riskDropPercentage={simulationResult.risk_drop_percentage}
                severedPathsCount={simulationResult.severed_paths_count}
                remainingPathsCount={simulationResult.remaining_paths_count}
              />

              <div className="mt-6 p-5 rounded-3xl border border-champagne/15 bg-velvetOlive/60 text-xs text-champagne/80 font-light leading-relaxed">
                <span className="text-vintageGold font-mono uppercase tracking-wider block mb-1">
                  AI SOC Recommendation:
                </span>
                {simulationResult.recommendation}
              </div>
            </div>
          ) : (
            <div className="py-24 text-center border border-dashed border-champagne/20 rounded-3xl bg-velvetOlive/20 p-8">
              <Sparkles className="w-10 h-10 text-vintageGold mx-auto mb-3" />
              <h4 className="text-xl font-serif text-champagne mb-1">Ready for Simulation</h4>
              <p className="text-xs text-champagne/60 max-w-md mx-auto mb-6">
                Toggle isolated nodes or CVE patches on the left, then click "Run Simulation" to execute the GNN counterfactual inference.
              </p>
              <button
                onClick={executeSimulation}
                className="px-6 py-2.5 rounded-full bg-vintageGold text-cellarNoir text-xs font-medium hover:bg-goldHoney transition-all shadow-md"
              >
                Execute Baseline Simulation
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Network Graph Visualizer for Simulation */}
      <div className="mt-8">
        <h4 className="text-sm font-serif text-champagne font-medium mb-3 uppercase tracking-wider">
          Reactive Network Graph Canvas
        </h4>
        <NetworkGraphCanvas />
      </div>
    </div>
  );
};
