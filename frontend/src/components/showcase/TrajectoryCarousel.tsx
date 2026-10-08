import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, Compass, ShieldAlert, Sparkles } from 'lucide-react';
import { useScenarioStore } from '../../store/scenarioStore';
import { TrajectoryCard } from './TrajectoryCard';

export const TrajectoryCarousel: React.FC = () => {
  const {
    trajectories,
    activeTrajectoryId,
    selectTrajectory,
    setStage,
    selectedScenario,
  } = useScenarioStore();

  const scrollRef = useRef<HTMLDivElement>(null);

  const scroll = (offset: number) => {
    if (scrollRef.current) {
      scrollRef.current.scrollBy({ left: offset, behavior: 'smooth' });
    }
  };

  return (
    <div className="relative py-10 px-8">
      {/* Editorial Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 pb-6 border-b border-champagne/10 gap-4">
        <div>
          <span className="text-xs uppercase font-mono tracking-[0.3em] text-vintageGold block mb-2">
            STAGE 2 • TRAJECTORY DISCOVERY
          </span>
          <h2 className="text-4xl md:text-5xl font-serif text-champagne font-normal tracking-tight">
            Attack Trajectory <span className="italic text-vintageGold">Showcase.</span>
          </h2>
          <p className="text-sm text-champagne/60 font-light mt-1 max-w-xl">
            Enumerated intrusion vectors discovered by bounded DFS graph traversal and scored via GraphSAGE node inference.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 mr-4">
            <button
              onClick={() => scroll(-380)}
              className="w-10 h-10 rounded-full border border-champagne/20 bg-velvetOlive/80 hover:bg-velvetOlive text-champagne flex items-center justify-center transition-colors"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
            <button
              onClick={() => scroll(380)}
              className="w-10 h-10 rounded-full border border-champagne/20 bg-velvetOlive/80 hover:bg-velvetOlive text-champagne flex items-center justify-center transition-colors"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>

          <button
            onClick={() => setStage('INSPECTOR')}
            className="px-5 py-2.5 rounded-full bg-vintageGold text-cellarNoir text-xs font-medium tracking-wide hover:bg-goldHoney transition-all flex items-center gap-2 shadow-md"
          >
            <Compass className="w-4 h-4" />
            <span>Deep Specification</span>
          </button>
        </div>
      </div>

      {/* Horizontal Carousel Container */}
      {trajectories.length === 0 ? (
        <div className="py-20 text-center border border-dashed border-champagne/20 rounded-3xl bg-velvetOlive/20">
          <ShieldAlert className="w-12 h-12 text-vintageGold/60 mx-auto mb-3" />
          <h4 className="text-xl font-serif text-champagne mb-1">No Paths Discovered</h4>
          <p className="text-sm text-champagne/60 max-w-md mx-auto">
            All attack trajectories are either severed or no designated entry point reaches a critical asset.
          </p>
        </div>
      ) : (
        <div
          ref={scrollRef}
          className="flex items-stretch gap-6 overflow-x-auto pb-6 scrollbar-none snap-x snap-mandatory focus:outline-none"
          style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
        >
          {trajectories.map((traj) => {
            const isActive = traj.id === activeTrajectoryId;
            return (
              <div key={traj.id} className="snap-start">
                <TrajectoryCard
                  trajectory={traj}
                  isActive={isActive}
                  onSelect={() => selectTrajectory(traj.id)}
                  onInspect={() => {
                    selectTrajectory(traj.id);
                    setStage('INSPECTOR');
                  }}
                />
              </div>
            );
          })}
        </div>
      )}

      {/* Trajectory Navigation Footer */}
      <div className="flex items-center justify-between pt-6 border-t border-champagne/10 text-xs text-champagne/50">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-vintageGold" />
          <span>Active Scenario: <strong className="text-champagne font-normal">{selectedScenario?.name}</strong></span>
        </div>
        <span className="font-mono">{trajectories.length} Attack Paths Ranked Descending</span>
      </div>
    </div>
  );
};
