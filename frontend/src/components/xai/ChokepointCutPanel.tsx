import React from 'react';
import { GitFork, Shield, ArrowRight, Check } from 'lucide-react';
import { useScenarioStore } from '../../store/scenarioStore';
import { useGraphStore } from '../../store/graphStore';

export const ChokepointCutPanel: React.FC = () => {
  const { trajectories, toggleNodeIsolation, isolatedNodeIds } = useScenarioStore();
  const { nodes } = useGraphStore();

  // Compute node intersection frequency across all top paths
  const frequencyMap: Record<string, number> = {};
  trajectories.forEach((t) => {
    // Check intermediate nodes
    if (t.node_ids.length > 2) {
      t.node_ids.slice(1, -1).forEach((nid) => {
        frequencyMap[nid] = (frequencyMap[nid] || 0) + 1;
      });
    }
  });

  const sortedChokepoints = Object.entries(frequencyMap)
    .sort((a, b) => b[1] - a[1])
    .map(([nid, count]) => {
      const nodeObj = nodes.find((n) => n.id === nid);
      return {
        id: nid,
        count,
        ip: nodeObj?.data.ip || nid,
        hostname: nodeObj?.data.hostname || 'Pivot Node',
        role: nodeObj?.data.role || 'server',
        cvss: nodeObj?.data.cvss || 7.0,
      };
    });

  return (
    <div className="rounded-3xl p-6 border border-champagne/15 bg-gradient-to-b from-velvetOlive/80 to-cellarNoir/80 backdrop-blur-2xl">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-champagne/10">
        <div className="flex items-center gap-2">
          <GitFork className="w-4 h-4 text-vintageGold" />
          <h4 className="text-sm font-serif text-champagne font-medium uppercase tracking-wider">
            Chokepoint Convergence Cuts
          </h4>
        </div>
        <span className="text-xs font-mono text-champagne/50">
          Graph Min-Cut Optimization
        </span>
      </div>

      <p className="text-xs text-champagne/60 font-light mb-5">
        Intersecting transit nodes where applying host-level containment severs the highest number of downstream routes simultaneously.
      </p>

      {sortedChokepoints.length === 0 ? (
        <div className="p-4 text-center rounded-2xl bg-cellarNoir/60 border border-champagne/10 text-xs text-champagne/50">
          No multi-hop intermediate chokepoints found in active scenario.
        </div>
      ) : (
        <div className="space-y-3">
          {sortedChokepoints.slice(0, 4).map((cp) => {
            const isIsolated = isolatedNodeIds.includes(cp.id);
            return (
              <div
                key={cp.id}
                className="p-3.5 rounded-2xl border border-champagne/10 bg-velvetOlive/40 hover:border-champagne/20 transition-all flex items-center justify-between"
              >
                <div>
                  <div className="text-xs font-mono font-medium text-champagne">
                    {cp.ip} <span className="text-champagne/40">({cp.role})</span>
                  </div>
                  <span className="text-[10px] text-vintageGold font-mono block mt-0.5">
                    Intersects {cp.count} / {trajectories.length} Candidate Paths
                  </span>
                </div>

                <button
                  onClick={() => toggleNodeIsolation(cp.id)}
                  className={`px-3 py-1.5 rounded-full text-xs font-mono transition-all flex items-center gap-1.5 ${
                    isIsolated
                      ? 'border border-emerald-500/40 bg-emerald-500/20 text-emerald-400'
                      : 'border border-vintageGold/40 bg-vintageGold/10 text-vintageGold hover:bg-vintageGold/20'
                  }`}
                >
                  {isIsolated ? <Check className="w-3 h-3" /> : <Shield className="w-3 h-3" />}
                  <span>{isIsolated ? 'Cut Active' : 'Sever Node'}</span>
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
