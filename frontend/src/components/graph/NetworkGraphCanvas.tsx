import React, { useEffect, useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  ReactFlowProvider,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { CustomNode } from './CustomNode';
import { CustomEdge } from './CustomEdge';
import { GraphToolbar } from './GraphToolbar';
import { useGraphStore } from '../../store/graphStore';
import { useScenarioStore } from '../../store/scenarioStore';

const NetworkGraphCanvasInner: React.FC = () => {
  const { nodes, edges, onNodesChange, onEdgesChange, loadGraph, isLoading } = useGraphStore();
  const { selectedScenarioId, activeTrajectory } = useScenarioStore();

  const nodeTypes = useMemo(() => ({ customNode: CustomNode }), []);
  const edgeTypes = useMemo(() => ({ customEdge: CustomEdge }), []);

  useEffect(() => {
    if (selectedScenarioId) {
      loadGraph(selectedScenarioId, activeTrajectory?.id);
    }
  }, [selectedScenarioId, activeTrajectory?.id, loadGraph]);

  return (
    <div className="relative w-full h-[650px] rounded-3xl border border-champagne/15 bg-cellarNoir overflow-hidden shadow-2xl">
      {/* Dark velvet grid background */}
      <ReactFlow
        nodes={nodes as any}
        edges={edges as any}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes as any}
        edgeTypes={edgeTypes as any}
        fitView
        fitViewOptions={{ padding: 0.25 }}
        minZoom={0.2}
        maxZoom={2.0}
        proOptions={{ hideAttribution: true }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={24}
          size={1.5}
          color="#F3EED912"
        />

        {/* Floating Toolbar */}
        <div className="absolute top-5 right-5 z-20">
          <GraphToolbar />
        </div>

        {/* Minimalist MiniMap */}
        <MiniMap
          nodeColor="#D4AF37"
          maskColor="rgba(8, 9, 10, 0.85)"
          className="!bottom-5 !right-5 !rounded-2xl !border !border-champagne/15 !bg-velvetOlive/80 !overflow-hidden"
          zoomable
          pannable
        />
      </ReactFlow>

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-cellarNoir/60 backdrop-blur-sm flex items-center justify-center z-30">
          <div className="flex items-center gap-3 px-5 py-2.5 rounded-full bg-velvetOlive border border-vintageGold/40 text-vintageGold font-mono text-xs shadow-xl">
            <span className="w-2.5 h-2.5 rounded-full bg-vintageGold animate-ping" />
            <span>Calculating GraphSAGE Topology...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export const NetworkGraphCanvas: React.FC = () => {
  return (
    <ReactFlowProvider>
      <NetworkGraphCanvasInner />
    </ReactFlowProvider>
  );
};
