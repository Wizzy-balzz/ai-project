import React from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, ArrowRight, Gauge, Layers, Terminal } from 'lucide-react';
import { AttackTrajectory } from '../../api/client';

interface TrajectoryCardProps {
  trajectory: AttackTrajectory;
  isActive: boolean;
  onSelect: () => void;
  onInspect: () => void;
}

export const TrajectoryCard: React.FC<TrajectoryCardProps> = ({
  trajectory,
  isActive,
  onSelect,
  onInspect,
}) => {
  const isCritical = trajectory.severity === 'CRITICAL';
  const isHigh = trajectory.severity === 'HIGH';

  const severityColor = isCritical
    ? 'text-crimsonVermillion border-crimsonVermillion/40 bg-crimsonVermillion/10'
    : isHigh
    ? 'text-vintageGold border-vintageGold/40 bg-vintageGold/10'
    : 'text-champagne/70 border-champagne/20 bg-champagne/5';

  return (
    <motion.div
      onClick={onSelect}
      whileHover={{ y: -6, transition: { duration: 0.2 } }}
      animate={{
        scale: isActive ? 1.03 : 0.98,
        opacity: isActive ? 1 : 0.85,
      }}
      className={`min-w-[340px] max-w-[360px] rounded-3xl p-6 border cursor-pointer backdrop-blur-2xl transition-all duration-300 flex flex-col justify-between ${
        isActive
          ? 'border-vintageGold/70 bg-gradient-to-b from-velvetOlive/95 to-cellarNoir/95 shadow-[0_12px_40px_rgba(212,175,55,0.22)]'
          : 'border-champagne/15 bg-velvetOlive/60 hover:border-champagne/30'
      }`}
    >
      <div>
        {/* Top Badges */}
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs font-mono tracking-wider text-champagne/60 uppercase">
            {trajectory.id}
          </span>
          <span className={`px-3 py-1 rounded-full text-[11px] font-mono tracking-wider uppercase border font-semibold ${severityColor}`}>
            {trajectory.severity}
          </span>
        </div>

        {/* Trajectory Title & Hops */}
        <h4 className="text-xl font-serif text-champagne font-medium mb-1">
          Attack Route ({trajectory.hops} Hops)
        </h4>
        <p className="text-xs text-champagne/60 font-mono mb-5 flex items-center gap-2">
          <span>{trajectory.node_ips[0]}</span>
          <ArrowRight className="w-3 h-3 text-vintageGold" />
          <span>{trajectory.node_ips[trajectory.node_ips.length - 1]}</span>
        </p>

        {/* Metrics Pill Grid */}
        <div className="grid grid-cols-3 gap-2 p-3 rounded-2xl bg-cellarNoir/60 border border-champagne/10 mb-5">
          <div className="text-center">
            <span className="text-[10px] uppercase font-mono text-champagne/50 block">Risk Score</span>
            <span className="text-base font-serif font-semibold text-vintageGold">
              {(trajectory.risk_score * 100).toFixed(0)}%
            </span>
          </div>
          <div className="text-center border-x border-champagne/10">
            <span className="text-[10px] uppercase font-mono text-champagne/50 block">Likelihood</span>
            <span className="text-base font-serif text-champagne">
              {(trajectory.likelihood * 100).toFixed(0)}%
            </span>
          </div>
          <div className="text-center">
            <span className="text-[10px] uppercase font-mono text-champagne/50 block">Impact</span>
            <span className="text-base font-serif text-champagne">
              {(trajectory.impact * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Kill Chain Narrative Snippet */}
        <div className="p-3 rounded-xl bg-velvetOlive/50 border border-champagne/10 mb-4 text-xs text-champagne/80 font-light leading-relaxed line-clamp-3">
          "{trajectory.kill_chain_narrative || 'Multi-hop intrusion path targeting database assets.'}"
        </div>
      </div>

      {/* Card Actions */}
      <div className="flex items-center justify-between pt-4 border-t border-champagne/10">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onInspect();
          }}
          className="text-xs text-vintageGold hover:text-goldHoney font-medium flex items-center gap-1.5 transition-colors"
        >
          <span>Deep Dive Specification</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>

        <span className="text-[11px] font-mono text-champagne/50">
          GNN P={((trajectory.factor_attribution?.node_susceptibility || 0.8) * 100).toFixed(0)}%
        </span>
      </div>
    </motion.div>
  );
};
