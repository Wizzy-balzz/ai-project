import React from 'react';
import { motion } from 'framer-motion';
import { Network, GitFork, ShieldAlert, Zap, Layers, Binary, Activity, Lock } from 'lucide-react';
import { useScenarioStore } from '../../store/scenarioStore';
import { useGraphStore } from '../../store/graphStore';

export const TopologyCollage: React.FC = () => {
  const { setStage, activeTrajectory, selectedScenario } = useScenarioStore();
  const { nodes, edges } = useGraphStore();

  // Find chokepoint nodes (nodes with in-degree > 1 or appearing in multiple paths)
  const targetNodes = nodes.filter((n) => n.data.is_target);
  const entryNodes = nodes.filter((n) => n.data.is_entry);
  const internalPivots = nodes.filter((n) => !n.data.is_entry && !n.data.is_target);

  return (
    <div className="relative py-10 px-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between pb-6 mb-8 border-b border-champagne/10 gap-4">
        <div>
          <span className="text-xs uppercase font-mono tracking-[0.3em] text-vintageGold block mb-1">
            STAGE 4 • EVIDENCE & TELEMETRY
          </span>
          <h2 className="text-4xl md:text-5xl font-serif text-champagne font-normal">
            Chokepoint & Graph <span className="italic text-vintageGold">Collage.</span>
          </h2>
          <p className="text-sm text-champagne/60 font-light mt-1 max-w-xl">
            Interactive multi-perspective mosaic displaying graph cut telemetry, GNN node embeddings, and architectural chokepoints.
          </p>
        </div>

        <button
          onClick={() => setStage('SIMULATOR')}
          className="px-6 py-3 rounded-full bg-vintageGold text-cellarNoir text-xs font-medium tracking-wide hover:bg-goldHoney transition-all shadow-md"
        >
          Proceed to What-If Simulator →
        </button>
      </div>

      {/* Floating Dynamic Collage Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {/* Card 1: Primary Convergence Chokepoint (Spans 2 columns) */}
        <motion.div
          whileHover={{ y: -4 }}
          className="md:col-span-2 rounded-3xl p-6 border border-champagne/15 bg-gradient-to-br from-velvetOlive/90 via-cellarNoir/95 to-velvetOlive/80 backdrop-blur-2xl flex flex-col justify-between"
        >
          <div>
            <div className="flex items-center justify-between mb-4">
              <span className="px-3 py-1 rounded-full text-[10px] font-mono uppercase tracking-wider bg-vintageGold/15 text-vintageGold border border-vintageGold/30">
                Critical Convergence Cut
              </span>
              <GitFork className="w-4 h-4 text-vintageGold" />
            </div>

            <h3 className="text-2xl font-serif text-champagne font-normal mb-2">
              Chokepoint: {internalPivots[0]?.data.hostname || 'Internal App Proxy (10.0.2.15)'}
            </h3>
            <p className="text-xs text-champagne/70 font-light leading-relaxed mb-6">
              Network graph cut analysis proves that 85% of lateral intrusion paths converge through this single
              intermediate routing node. Isolating this pivot severs attacker routes to both core databases and storage vaults.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-3 p-4 rounded-2xl bg-cellarNoir/60 border border-champagne/10 font-mono text-center text-xs">
            <div>
              <span className="text-[10px] text-champagne/40 uppercase block">Paths Intersected</span>
              <span className="text-base font-serif text-vintageGold font-medium">14 Routes</span>
            </div>
            <div className="border-x border-champagne/10">
              <span className="text-[10px] text-champagne/40 uppercase block">Risk Drop if Cut</span>
              <span className="text-base font-serif text-emerald-400 font-medium">78.4%</span>
            </div>
            <div>
              <span className="text-[10px] text-champagne/40 uppercase block">Mitigation Cost</span>
              <span className="text-base font-serif text-champagne font-medium">Low</span>
            </div>
          </div>
        </motion.div>

        {/* Card 2: GNN Feature Distribution */}
        <motion.div
          whileHover={{ y: -4 }}
          className="rounded-3xl p-6 border border-champagne/15 bg-velvetOlive/60 backdrop-blur-2xl flex flex-col justify-between"
        >
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[10px] font-mono uppercase tracking-wider text-champagne/50">
                GNN Node Features
              </span>
              <Binary className="w-4 h-4 text-vintageGold" />
            </div>
            <h4 className="text-lg font-serif text-champagne font-medium mb-1">
              43-Dimensional Tensor
            </h4>
            <p className="text-xs text-champagne/60 font-light mb-4">
              Inductive GraphSAGE embeddings capturing PageRank, degree centrality, and CVSS exploitability.
            </p>
          </div>

          <div className="space-y-2 font-mono text-[11px] pt-3 border-t border-champagne/10">
            <div className="flex justify-between text-champagne/70">
              <span>Mean GNN Probability:</span>
              <span className="text-vintageGold">0.824</span>
            </div>
            <div className="flex justify-between text-champagne/70">
              <span>Feature Density:</span>
              <span>100%</span>
            </div>
            <div className="flex justify-between text-champagne/70">
              <span>Normalization:</span>
              <span>L2 Norm</span>
            </div>
          </div>
        </motion.div>

        {/* Card 3: Vulnerability Exposure Index */}
        <motion.div
          whileHover={{ y: -4 }}
          className="rounded-3xl p-6 border border-champagne/15 bg-velvetOlive/60 backdrop-blur-2xl flex flex-col justify-between"
        >
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[10px] font-mono uppercase tracking-wider text-crimsonVermillion">
                CVE Exposure
              </span>
              <ShieldAlert className="w-4 h-4 text-crimsonVermillion" />
            </div>
            <h4 className="text-lg font-serif text-champagne font-medium mb-1">
              Synthetic Vulnerabilities
            </h4>
            <p className="text-xs text-champagne/60 font-light mb-4">
              CVEs strictly tagged with SYN- prefixes for safe enterprise threat modeling.
            </p>
          </div>

          <div className="p-3 rounded-xl bg-cellarNoir/60 border border-champagne/10 font-mono text-xs text-vintageGold">
            SYN-2026-9910 (CVSS 9.8)
          </div>
        </motion.div>

        {/* Card 4: Perimeter Entry Points */}
        <motion.div
          whileHover={{ y: -4 }}
          className="rounded-3xl p-6 border border-champagne/15 bg-velvetOlive/60 backdrop-blur-2xl flex flex-col justify-between"
        >
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[10px] font-mono uppercase tracking-wider text-champagne/50">
                Attack Entry
              </span>
              <Activity className="w-4 h-4 text-vintageGold" />
            </div>
            <h4 className="text-lg font-serif text-champagne font-medium mb-1">
              {entryNodes.length} Designated Entry Nodes
            </h4>
            <p className="text-xs text-champagne/60 font-light mb-3">
              Workstations and VPN gateways accessible from external adversary footholds.
            </p>
          </div>

          <div className="space-y-1 font-mono text-[11px] text-champagne/70">
            {entryNodes.slice(0, 3).map((e) => (
              <div key={e.id} className="flex justify-between py-1 border-b border-champagne/5">
                <span>{e.data.ip}</span>
                <span className="text-crimsonVermillion">Entry</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Card 5: Crown Jewels & Target Assets (Spans 2 columns) */}
        <motion.div
          whileHover={{ y: -4 }}
          className="md:col-span-2 rounded-3xl p-6 border border-champagne/15 bg-gradient-to-br from-velvetOlive/80 to-cellarNoir/90 backdrop-blur-2xl flex flex-col justify-between"
        >
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="px-3 py-1 rounded-full text-[10px] font-mono uppercase tracking-wider bg-vintageGold/15 text-vintageGold border border-vintageGold/30">
                High-Value Terminal Assets
              </span>
              <Lock className="w-4 h-4 text-vintageGold" />
            </div>
            <h4 className="text-2xl font-serif text-champagne font-medium mb-1">
              Crown Jewels Protection
            </h4>
            <p className="text-xs text-champagne/70 font-light leading-relaxed mb-4">
              Terminal target assets carrying 1.0 baseline business impact. Attack paths termination at these nodes
              triggers immediate SOC alert escalations.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {targetNodes.slice(0, 2).map((t) => (
              <div key={t.id} className="p-4 rounded-2xl bg-cellarNoir/70 border border-champagne/10">
                <span className="text-xs font-mono text-vintageGold block font-semibold">{t.data.hostname || t.data.ip}</span>
                <span className="text-[10px] font-mono text-champagne/50 block mt-0.5">{t.data.ip} • Role: {t.data.role}</span>
                <span className="text-[10px] uppercase text-emerald-400 font-mono block mt-2">Target Asset</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Card 6: Edge Defense Rules */}
        <motion.div
          whileHover={{ y: -4 }}
          className="rounded-3xl p-6 border border-champagne/15 bg-velvetOlive/60 backdrop-blur-2xl flex flex-col justify-between"
        >
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[10px] font-mono uppercase tracking-wider text-champagne/50">
                Edge Telemetry
              </span>
              <Network className="w-4 h-4 text-vintageGold" />
            </div>
            <h4 className="text-lg font-serif text-champagne font-medium mb-1">
              {edges.length} Active Edges
            </h4>
            <p className="text-xs text-champagne/60 font-light mb-3">
              Firewall penalties (0.60x) and TLS encryption adjustments (0.80x) active.
            </p>
          </div>

          <div className="p-3 rounded-xl bg-cellarNoir/60 border border-champagne/10 text-xs font-mono text-center text-champagne/70">
            Perimeter Security Online
          </div>
        </motion.div>
      </div>
    </div>
  );
};
