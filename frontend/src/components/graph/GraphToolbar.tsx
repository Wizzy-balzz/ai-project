import React from 'react';
import { useReactFlow } from '@xyflow/react';
import { ZoomIn, ZoomOut, Maximize2, RotateCcw, Trash, Plus } from 'lucide-react';
import { useGraphStore } from '../../store/graphStore';
import { useScenarioStore } from '../../store/scenarioStore';

export const GraphToolbar: React.FC = () => {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const { selectedScenarioId } = useScenarioStore();
  const { loadGraph, clearCanvas } = useGraphStore();

  const handleReset = () => {
    if (selectedScenarioId) {
      loadGraph(selectedScenarioId);
      setTimeout(() => fitView({ padding: 0.2 }), 200);
    }
  };

  const handleClear = () => {
    if (selectedScenarioId && confirm('Are you sure you want to clear all nodes and edges to start on an empty canvas?')) {
      clearCanvas(selectedScenarioId);
    }
  };

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-champagne/20 bg-velvetOlive/80 backdrop-blur-2xl shadow-2xl">
      <button
        onClick={() => zoomIn()}
        title="Zoom In"
        className="p-1.5 rounded-full text-champagne/80 hover:text-vintageGold hover:bg-champagne/10 transition-colors"
      >
        <ZoomIn className="w-4 h-4" />
      </button>

      <button
        onClick={() => zoomOut()}
        title="Zoom Out"
        className="p-1.5 rounded-full text-champagne/80 hover:text-vintageGold hover:bg-champagne/10 transition-colors"
      >
        <ZoomOut className="w-4 h-4" />
      </button>

      <div className="w-[1px] h-4 bg-champagne/15 mx-1" />

      <button
        onClick={() => fitView({ padding: 0.2, duration: 400 })}
        title="Fit View"
        className="p-1.5 rounded-full text-champagne/80 hover:text-vintageGold hover:bg-champagne/10 transition-colors"
      >
        <Maximize2 className="w-4 h-4" />
      </button>

      <button
        onClick={handleReset}
        title="Reset & Relayout"
        className="p-1.5 rounded-full text-champagne/80 hover:text-vintageGold hover:bg-champagne/10 transition-colors"
      >
        <RotateCcw className="w-4 h-4" />
      </button>

      <div className="w-[1px] h-4 bg-champagne/15 mx-1" />

      <button
        onClick={handleClear}
        title="Clear Canvas (Empty Blank Scenario)"
        className="p-1.5 rounded-full text-champagne/50 hover:text-crimsonVermillion hover:bg-crimsonVermillion/15 transition-colors"
      >
        <Trash className="w-4 h-4" />
      </button>
    </div>
  );
};
