import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { TrendingDown, ShieldCheck, Zap } from 'lucide-react';
import confetti from 'canvas-confetti';

interface RiskDropGaugeProps {
  originalRisk: number;
  simulatedRisk: number;
  riskDropPercentage: number;
  severedPathsCount: number;
  remainingPathsCount: number;
}

export const RiskDropGauge: React.FC<RiskDropGaugeProps> = ({
  originalRisk,
  simulatedRisk,
  riskDropPercentage,
  severedPathsCount,
  remainingPathsCount,
}) => {
  useEffect(() => {
    if (riskDropPercentage >= 50) {
      try {
        confetti({
          particleCount: 50,
          spread: 60,
          origin: { y: 0.8 },
          colors: ['#D4AF37', '#E5B869', '#6E8472', '#F3EED9'],
        });
      } catch (e) {
        // Safe fallback
      }
    }
  }, [riskDropPercentage]);

  return (
    <div className="p-6 rounded-3xl border border-champagne/15 bg-gradient-to-br from-velvetOlive/90 to-cellarNoir/90 backdrop-blur-2xl">
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs uppercase font-mono tracking-wider text-champagne/50">
          Defense Impact Assessment
        </span>
        <span className="flex items-center gap-1.5 text-xs text-emerald-400 font-mono">
          <ShieldCheck className="w-4 h-4" />
          <span>Simulation Active</span>
        </span>
      </div>

      {/* Main Metric Display */}
      <div className="flex items-baseline justify-between mb-6">
        <div>
          <span className="text-4xl md:text-5xl font-serif text-emerald-400 font-medium">
            -{riskDropPercentage.toFixed(1)}%
          </span>
          <span className="text-xs text-champagne/60 font-mono block mt-1">
            Overall Network Risk Reduction
          </span>
        </div>

        <div className="text-right">
          <div className="text-xs font-mono text-champagne/50">
            Original Risk: <span className="text-crimsonVermillion">{(originalRisk * 100).toFixed(0)}%</span>
          </div>
          <div className="text-xs font-mono text-champagne/50 mt-0.5">
            Residual Risk: <span className="text-emerald-400 font-semibold">{(simulatedRisk * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* Two-Bar Visual Comparison */}
      <div className="space-y-3 mb-6">
        <div>
          <div className="flex justify-between text-[11px] font-mono text-champagne/60 mb-1">
            <span>Pre-Mitigation Baseline Risk</span>
            <span>{(originalRisk * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 rounded-full bg-cellarNoir overflow-hidden border border-champagne/10">
            <div
              style={{ width: `${Math.min(100, originalRisk * 100)}%` }}
              className="h-full rounded-full bg-gradient-to-r from-crimsonVermillion to-red-400"
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-[11px] font-mono text-champagne/60 mb-1">
            <span>Counterfactual Residual Risk</span>
            <span className="text-emerald-400 font-semibold">{(simulatedRisk * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 rounded-full bg-cellarNoir overflow-hidden border border-champagne/10">
            <motion.div
              initial={{ width: `${Math.min(100, originalRisk * 100)}%` }}
              animate={{ width: `${Math.min(100, simulatedRisk * 100)}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-emerald-400"
            />
          </div>
        </div>
      </div>

      {/* Stats Counter Cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 rounded-2xl bg-cellarNoir/70 border border-champagne/10 text-center">
          <span className="text-[10px] uppercase font-mono text-champagne/40 block">Severed Attack Paths</span>
          <span className="text-2xl font-serif text-emerald-400 font-medium">
            {severedPathsCount}
          </span>
          <span className="text-[10px] text-champagne/50 font-mono block mt-0.5">Neutralized</span>
        </div>

        <div className="p-4 rounded-2xl bg-cellarNoir/70 border border-champagne/10 text-center">
          <span className="text-[10px] uppercase font-mono text-champagne/40 block">Remaining Trajectories</span>
          <span className="text-2xl font-serif text-champagne font-medium">
            {remainingPathsCount}
          </span>
          <span className="text-[10px] text-champagne/50 font-mono block mt-0.5">Residual Paths</span>
        </div>
      </div>
    </div>
  );
};
