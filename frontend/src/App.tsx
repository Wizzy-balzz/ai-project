import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield,
  Layers,
  Compass,
  GitFork,
  Activity,
  Plus,
  Network,
  ChevronRight,
  Database,
} from 'lucide-react';
import { useScenarioStore, UIStage } from './store/scenarioStore';
import { useGraphStore } from './store/graphStore';
import { createScenario } from './api/client';

import { EditorialHero } from './components/home/EditorialHero';
import { TrajectoryCarousel } from './components/showcase/TrajectoryCarousel';
import { NodeInspectorView } from './components/inspector/NodeInspectorView';
import { TopologyCollage } from './components/collage/TopologyCollage';
import { WhatIfSimulatorPanel } from './components/simulation/WhatIfSimulatorPanel';
import { NetworkGraphCanvas } from './components/graph/NetworkGraphCanvas';
import { KillChainNarrativeView } from './components/xai/KillChainNarrativeView';
import { FactorAttributionChart } from './components/xai/FactorAttributionChart';
import { ChokepointCutPanel } from './components/xai/ChokepointCutPanel';

export const App: React.FC = () => {
  const {
    currentStage,
    setStage,
    loadScenarios,
    scenarios,
    selectedScenarioId,
    selectScenario,
    activeTrajectory,
  } = useScenarioStore();

  const [isCreatingScenario, setIsCreatingScenario] = useState(false);
  const [newScenarioName, setNewScenarioName] = useState('');
  const [newScenarioDesc, setNewScenarioDesc] = useState('');

  useEffect(() => {
    loadScenarios();
  }, [loadScenarios]);

  const stages: Array<{ id: UIStage; label: string; icon: React.ReactNode }> = [
    { id: 'HERO', label: '1. Editorial Hero', icon: <Shield className="w-3.5 h-3.5" /> },
    { id: 'SHOWCASE', label: '2. Trajectories', icon: <Compass className="w-3.5 h-3.5" /> },
    { id: 'INSPECTOR', label: '3. Deep Spec', icon: <Layers className="w-3.5 h-3.5" /> },
    { id: 'COLLAGE', label: '4. Evidence Collage', icon: <GitFork className="w-3.5 h-3.5" /> },
    { id: 'SIMULATOR', label: '5. What-If Defense', icon: <Activity className="w-3.5 h-3.5" /> },
  ];

  const handleCreateBlankScenario = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newScenarioName.trim()) return;

    try {
      const created = await createScenario(newScenarioName, newScenarioDesc || 'Analyst Custom Topology');
      await loadScenarios();
      await selectScenario(created.id);
      setIsCreatingScenario(false);
      setNewScenarioName('');
      setNewScenarioDesc('');
      setStage('SHOWCASE');
    } catch (err) {
      console.error('Failed to create scenario:', err);
    }
  };

  return (
    <div className="min-h-screen bg-cellarNoir text-champagne relative selection:bg-vintageGold/30 selection:text-champagne font-sans overflow-x-hidden p-4 md:p-8">
      {/* Ambient soft blur gradients (Requirement 13) */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-10 left-1/4 w-[600px] h-[600px] bg-vintageGold/5 blur-[160px] rounded-full" />
        <div className="absolute bottom-10 right-1/4 w-[600px] h-[600px] bg-velvetOlive/40 blur-[180px] rounded-full" />
      </div>

      {/* Floating Central Stage Viewport Container (Requirement 13: rounded-[32px]) */}
      <div className="relative z-10 max-w-7xl mx-auto rounded-[32px] border border-champagne/15 bg-cellarNoir/90 backdrop-blur-2xl shadow-[0_20px_70px_rgba(0,0,0,0.8)] overflow-hidden">
        {/* Luxury Editorial Navigation Bar */}
        <header className="px-8 py-5 border-b border-champagne/10 flex flex-wrap items-center justify-between gap-4 bg-velvetOlive/50 backdrop-blur-xl">
          <div className="flex items-center gap-3">
            <span className="w-3 h-3 rounded-full bg-vintageGold animate-pulse" />
            <span className="font-serif italic text-lg tracking-wide text-champagne font-medium">
              Attack Path Predictor AI
            </span>
            <span className="text-xs font-mono text-champagne/40">|</span>
            <span className="text-xs font-mono text-vintageGold/80 tracking-widest uppercase">
              TasteWinner Editorial
            </span>
          </div>

          {/* Sequential Stage Controller Pills (Requirement 14) */}
          <div className="flex items-center gap-1.5 p-1 rounded-full bg-cellarNoir/80 border border-champagne/15 overflow-x-auto max-w-full">
            {stages.map((s) => {
              const isActive = currentStage === s.id;
              return (
                <button
                  key={s.id}
                  onClick={() => setStage(s.id)}
                  className={`px-3.5 py-1.5 rounded-full text-xs font-mono transition-all duration-300 flex items-center gap-1.5 whitespace-nowrap ${
                    isActive
                      ? 'bg-vintageGold text-cellarNoir font-medium shadow-md scale-102'
                      : 'text-champagne/60 hover:text-champagne hover:bg-champagne/5'
                  }`}
                >
                  {s.icon}
                  <span>{s.label}</span>
                </button>
              );
            })}
          </div>

          {/* Prominent Action: [ Create Blank Custom Scenario ] (Requirement 107) */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsCreatingScenario(true)}
              className="px-4 py-2 rounded-full border border-champagne/20 bg-champagne/5 hover:bg-vintageGold hover:text-cellarNoir hover:border-vintageGold text-xs text-champagne font-medium tracking-wide transition-all duration-300 flex items-center gap-1.5"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Create Blank Scenario</span>
            </button>
          </div>
        </header>

        {/* Create Scenario Modal */}
        {isCreatingScenario && (
          <div className="fixed inset-0 z-50 bg-cellarNoir/80 backdrop-blur-md flex items-center justify-center p-4">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="w-full max-w-md rounded-3xl p-6 border border-champagne/20 bg-velvetOlive shadow-2xl"
            >
              <h3 className="text-2xl font-serif text-champagne mb-1">Create Blank Scenario</h3>
              <p className="text-xs text-champagne/60 font-light mb-5">
                Initialize an empty canvas to place and evaluate custom nodes, edges, and perimeters.
              </p>

              <form onSubmit={handleCreateBlankScenario} className="space-y-4">
                <div>
                  <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                    Scenario Name
                  </label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Operation Nightfall DMZ"
                    value={newScenarioName}
                    onChange={(e) => setNewScenarioName(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-xl border border-champagne/20 bg-cellarNoir text-champagne text-xs focus:outline-none focus:border-vintageGold font-mono"
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono uppercase text-champagne/70 block mb-1">
                    Description
                  </label>
                  <textarea
                    rows={2}
                    placeholder="Synthetic topology description..."
                    value={newScenarioDesc}
                    onChange={(e) => setNewScenarioDesc(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-xl border border-champagne/20 bg-cellarNoir text-champagne text-xs focus:outline-none focus:border-vintageGold font-mono"
                  />
                </div>

                <div className="flex items-center justify-end gap-3 pt-3">
                  <button
                    type="button"
                    onClick={() => setIsCreatingScenario(false)}
                    className="px-4 py-2 rounded-full border border-champagne/15 text-xs text-champagne/60 hover:text-champagne"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-6 py-2 rounded-full bg-vintageGold text-cellarNoir text-xs font-semibold hover:bg-goldHoney transition-all"
                  >
                    Create Scenario
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        )}

        {/* Stage Content Choreography (Requirement 14) */}
        <main className="min-h-[80vh]">
          <AnimatePresence mode="wait">
            {currentStage === 'HERO' && (
              <motion.div
                key="HERO"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4 }}
              >
                <EditorialHero />
              </motion.div>
            )}

            {currentStage === 'SHOWCASE' && (
              <motion.div
                key="SHOWCASE"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="space-y-8"
              >
                <TrajectoryCarousel />
                
                <div className="px-8 pb-10">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-serif text-champagne font-normal">
                      Active Trajectory Graph View
                    </h3>
                    <span className="text-xs font-mono text-champagne/50">
                      Glow highlights critical attack route
                    </span>
                  </div>
                  <NetworkGraphCanvas />
                </div>
              </motion.div>
            )}

            {currentStage === 'INSPECTOR' && (
              <motion.div
                key="INSPECTOR"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="space-y-8"
              >
                <NodeInspectorView />

                {activeTrajectory && (
                  <div className="px-8 pb-10 grid grid-cols-1 md:grid-cols-2 gap-6">
                    <KillChainNarrativeView trajectory={activeTrajectory} />
                    <FactorAttributionChart attribution={activeTrajectory.factor_attribution} />
                  </div>
                )}
              </motion.div>
            )}

            {currentStage === 'COLLAGE' && (
              <motion.div
                key="COLLAGE"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="space-y-8"
              >
                <TopologyCollage />

                <div className="px-8 pb-10 grid grid-cols-1 md:grid-cols-2 gap-6">
                  <ChokepointCutPanel />
                  {activeTrajectory && (
                    <FactorAttributionChart attribution={activeTrajectory.factor_attribution} />
                  )}
                </div>
              </motion.div>
            )}

            {currentStage === 'SIMULATOR' && (
              <motion.div
                key="SIMULATOR"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.4 }}
              >
                <WhatIfSimulatorPanel />
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        {/* Luxury Editorial Footer */}
        <footer className="px-8 py-6 border-t border-champagne/10 flex flex-col md:flex-row items-center justify-between gap-4 text-xs font-mono text-champagne/40 bg-velvetOlive/40">
          <div>
            <span>Autonomous Defensive Cybernetics • GraphSAGE GNN Model v1.0</span>
          </div>

          <div className="flex items-center gap-6">
            <span>Synthetic Research Topology Only</span>
            <span className="text-vintageGold/80">Tailwind CSS + @xyflow/react</span>
          </div>
        </footer>
      </div>
    </div>
  );
};
