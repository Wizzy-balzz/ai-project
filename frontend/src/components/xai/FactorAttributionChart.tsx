import React from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Activity } from 'lucide-react';
import { FactorAttribution } from '../../api/client';

interface FactorAttributionChartProps {
  attribution: FactorAttribution;
}

export const FactorAttributionChart: React.FC<FactorAttributionChartProps> = ({ attribution }) => {
  const factors = [
    {
      name: 'GraphSAGE GNN Susceptibility',
      weight: 35,
      score: attribution.node_susceptibility || 0.85,
      color: 'from-vintageGold to-goldHoney',
      desc: 'Inductive node neighborhood embedding and structural role',
    },
    {
      name: 'Edge Perimeter Permeability',
      weight: 35,
      score: attribution.edge_permeability || 0.70,
      color: 'from-mutedSage to-emerald-400',
      desc: 'Firewall rules, TLS encryption, and telemetry anomaly indicators',
    },
    {
      name: 'CVSS Vulnerability Exploitability',
      weight: 30,
      score: attribution.vulnerability_exploitability || 0.88,
      color: 'from-crimsonVermillion to-red-400',
      desc: 'Known CVE exploitability metrics and patch state',
    },
  ];

  return (
    <div className="rounded-3xl p-6 border border-champagne/15 bg-gradient-to-b from-velvetOlive/80 to-cellarNoir/80 backdrop-blur-2xl">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-champagne/10">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-vintageGold" />
          <h4 className="text-sm font-serif text-champagne font-medium uppercase tracking-wider">
            Factor Attribution Weighting
          </h4>
        </div>
        <span className="text-xs font-mono text-champagne/50">
          Formal Risk Formula
        </span>
      </div>

      <p className="text-xs text-champagne/60 font-light mb-6">
        Composite Likelihood = (0.35 × GNN + 0.35 × Edge + 0.30 × CVSS) × 0.95<sup>hops-1</sup>
      </p>

      <div className="space-y-5">
        {factors.map((f) => (
          <div key={f.name}>
            <div className="flex items-center justify-between text-xs font-mono mb-1.5">
              <span className="text-champagne/90">
                {f.name} <strong className="text-vintageGold font-normal">({f.weight}%)</strong>
              </span>
              <span className="text-champagne/70 font-semibold">
                {(f.score * 100).toFixed(0)}% Score
              </span>
            </div>

            {/* Visual Progress Bar */}
            <div className="h-2.5 rounded-full bg-cellarNoir/80 border border-champagne/10 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(100, f.score * 100)}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
                className={`h-full rounded-full bg-gradient-to-r ${f.color}`}
              />
            </div>

            <span className="text-[10px] text-champagne/40 block mt-1 font-light">
              {f.desc}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
