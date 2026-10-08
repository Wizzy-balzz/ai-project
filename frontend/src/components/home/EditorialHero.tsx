import React from 'react';
import { motion } from 'framer-motion';
import { Shield, ArrowRight, Activity, Zap, Database, Lock } from 'lucide-react';
import { useScenarioStore } from '../../store/scenarioStore';

export const EditorialHero: React.FC = () => {
  const { setStage, scenarios, selectedScenarioId, selectScenario } = useScenarioStore();

  return (
    <div className="relative min-h-[90vh] flex flex-col justify-between overflow-hidden px-8 py-12">
      {/* Ambient warm golden-hour dusk backlight and velvet noir gradients */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[1000px] h-[500px] bg-gradient-to-b from-vintageGold/15 via-goldHoney/5 to-transparent blur-[120px] rounded-full" />
        <div className="absolute bottom-0 left-0 right-0 h-64 bg-gradient-to-t from-cellarNoir via-cellarNoir/80 to-transparent" />
      </div>

      {/* Top Header Badge */}
      <div className="relative z-10 flex items-center justify-between border-b border-champagne/10 pb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full border border-vintageGold/40 bg-velvetOlive/80 backdrop-blur-md flex items-center justify-center text-vintageGold shadow-[0_0_15px_rgba(212,175,55,0.2)]">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs uppercase tracking-[0.3em] text-champagne/60 font-medium block">
              Autonomous Defensive Cybernetics
            </span>
            <span className="text-sm font-serif italic text-champagne">
              GraphSAGE Inductive Inference Edition
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full border border-champagne/15 bg-velvetOlive/60 backdrop-blur-md text-xs text-champagne/70">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>GNN Active • Latency 14ms</span>
          </div>
        </div>
      </div>

      {/* Central Editorial Hero Typography */}
      <div className="relative z-10 my-auto text-center max-w-5xl mx-auto pt-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        >
          <span className="text-xs font-mono tracking-[0.4em] uppercase text-vintageGold/80 mb-4 inline-block px-4 py-1.5 rounded-full border border-vintageGold/25 bg-vintageGold/5 backdrop-blur-md">
            Phase 1 — 8 Full Pipeline Complete
          </span>

          <h1 className="text-6xl md:text-8xl font-serif font-normal tracking-tight text-champagne leading-[1.08] mb-6 drop-shadow-sm">
            Intrusion. <br />
            <span className="italic text-vintageGold font-serif">Intelligence.</span> Intuition.
          </h1>

          <p className="text-base md:text-lg text-champagne/70 max-w-2xl mx-auto font-light leading-relaxed mb-10">
            Unveiling high-probability multi-hop lateral intrusion trajectories before perimeter compromise.
            Powered by Graph Neural Networks, bounded depth-first enumeration, and plain-language kill-chain narratives.
          </p>

          {/* Centered Magnetic CTA Button */}
          <div className="flex items-center justify-center gap-5">
            <motion.button
              whileHover={{ scale: 1.04, boxShadow: '0 0 30px rgba(212,175,55,0.35)' }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setStage('SHOWCASE')}
              className="px-8 py-4 rounded-full bg-gradient-to-r from-vintageGold to-goldHoney text-cellarNoir font-medium text-sm tracking-wide shadow-lg flex items-center gap-3 transition-all duration-300"
            >
              <span>Start Attack Path Exploration</span>
              <ArrowRight className="w-4 h-4" />
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.02, borderColor: 'rgba(243,238,217,0.4)' }}
              onClick={() => setStage('SIMULATOR')}
              className="px-7 py-4 rounded-full border border-champagne/20 bg-velvetOlive/40 backdrop-blur-md text-champagne text-sm font-medium hover:bg-velvetOlive/70 transition-all duration-300"
            >
              What-If Defense Simulator
            </motion.button>
          </div>
        </motion.div>
      </div>

      {/* Floating Glass Pick Cards: Curated Attack Scenarios */}
      <div className="relative z-10 pt-10">
        <div className="flex items-center justify-between mb-4">
          <div className="text-xs uppercase tracking-[0.25em] text-champagne/50">
            Curated Attack Scenarios: <span className="text-champagne/90">SYNTHETIC ENTERPRISE TOPOLOGY</span>
          </div>
          <span className="text-xs text-vintageGold/80 font-mono">
            {scenarios.length} Scenarios Available
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {scenarios.map((sc) => {
            const isSelected = sc.id === selectedScenarioId;
            return (
              <motion.div
                key={sc.id}
                whileHover={{ y: -4 }}
                onClick={() => selectScenario(sc.id)}
                className={`cursor-pointer rounded-2xl p-5 border transition-all duration-300 backdrop-blur-xl ${
                  isSelected
                    ? 'border-vintageGold/60 bg-gradient-to-b from-velvetOlive/90 to-cellarNoir/90 shadow-[0_8px_30px_rgba(212,175,55,0.15)]'
                    : 'border-champagne/10 bg-velvetOlive/40 hover:border-champagne/25 hover:bg-velvetOlive/60'
                }`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wider bg-vintageGold/15 text-vintageGold border border-vintageGold/30">
                      {sc.topology_type}
                    </span>
                    {sc.is_synthetic && (
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-mono text-mutedSage bg-mutedSage/15 border border-mutedSage/30">
                        SYNTHETIC
                      </span>
                    )}
                  </div>
                  <span className="text-xs font-mono text-champagne/60">
                    Risk {sc.risk_score.toFixed(2)}
                  </span>
                </div>

                <h3 className="text-lg font-serif text-champagne font-medium mb-1">
                  {sc.name}
                </h3>
                <p className="text-xs text-champagne/60 line-clamp-2 mb-4 leading-relaxed font-light">
                  {sc.description || 'Enterprise multi-tier topology with segmented DMZ and databases.'}
                </p>

                <div className="flex items-center justify-between pt-3 border-t border-champagne/10 text-xs text-champagne/50 font-mono">
                  <span className="flex items-center gap-1.5">
                    <Database className="w-3.5 h-3.5 text-vintageGold/70" />
                    {sc.num_nodes} Nodes
                  </span>
                  <span className="flex items-center gap-1.5">
                    <Zap className="w-3.5 h-3.5 text-vintageGold/70" />
                    {sc.num_edges} Edges
                  </span>
                  <span className="flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-crimsonVermillion" />
                    Top-K Critical
                  </span>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
