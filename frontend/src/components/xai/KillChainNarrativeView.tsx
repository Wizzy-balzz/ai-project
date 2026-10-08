import React from 'react';
import { ArrowRight, ShieldAlert, Terminal, Layers } from 'lucide-react';
import { AttackTrajectory } from '../../api/client';

interface KillChainNarrativeViewProps {
  trajectory: AttackTrajectory;
}

export const KillChainNarrativeView: React.FC<KillChainNarrativeViewProps> = ({ trajectory }) => {
  return (
    <div className="rounded-3xl p-6 border border-champagne/15 bg-gradient-to-b from-velvetOlive/80 to-cellarNoir/80 backdrop-blur-2xl">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-champagne/10">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-vintageGold" />
          <h4 className="text-sm font-serif text-champagne font-medium uppercase tracking-wider">
            Kill-Chain Plain Narrative
          </h4>
        </div>
        <span className="text-xs font-mono text-champagne/50">
          {trajectory.hops} Hops
        </span>
      </div>

      {/* Narrative Headline */}
      <div className="p-4 rounded-2xl bg-cellarNoir/70 border border-champagne/10 text-xs text-champagne/80 font-light leading-relaxed mb-6">
        "{trajectory.kill_chain_narrative}"
      </div>

      {/* Step-by-Step Hop Breakdown */}
      <div className="space-y-3">
        {trajectory.hop_details && trajectory.hop_details.map((hop) => (
          <div
            key={hop.step_number}
            className="p-3.5 rounded-2xl border border-champagne/10 bg-velvetOlive/40 hover:border-champagne/20 transition-all flex items-start gap-3"
          >
            <div className="w-6 h-6 rounded-full bg-vintageGold/15 border border-vintageGold/30 text-vintageGold flex items-center justify-center text-[10px] font-mono font-bold shrink-0 mt-0.5">
              {hop.step_number}
            </div>

            <div className="flex-1">
              <div className="flex items-center justify-between text-xs font-mono mb-1">
                <span className="text-champagne font-medium">
                  {hop.source_ip} ({hop.source_role})
                </span>
                <ArrowRight className="w-3 h-3 text-vintageGold" />
                <span className="text-vintageGold font-semibold">
                  {hop.target_ip} ({hop.target_role})
                </span>
              </div>

              <div className="text-[11px] text-champagne/70 font-light mb-2">
                {hop.action_narrative || `Attacker pivots to ${hop.target_ip} exploiting vulnerability ${hop.vulnerability_cve}.`}
              </div>

              <div className="flex items-center gap-3 text-[10px] font-mono text-champagne/50">
                <span>CVE: <strong className="text-crimsonVermillion font-normal">{hop.vulnerability_cve}</strong></span>
                <span>CVSS: {hop.cvss_score.toFixed(1)}</span>
                <span>Step Likelihood: {(hop.step_likelihood * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
