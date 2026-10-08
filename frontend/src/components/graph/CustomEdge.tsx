import React, { memo } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getSmoothStepPath,
  EdgeProps,
} from '@xyflow/react';
import { Shield, ShieldAlert, Trash2 } from 'lucide-react';
import { useGraphStore } from '../../store/graphStore';
import { useScenarioStore } from '../../store/scenarioStore';

export const CustomEdge = memo(({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
  markerEnd,
}: EdgeProps) => {
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 20,
  });

  const { toggleFirewall, deleteEdge } = useGraphStore();
  const { blockedEdgeIds, toggleEdgeBlock } = useScenarioStore();

  const isFirewalled = data?.is_firewalled || false;
  const isBlocked = blockedEdgeIds.includes(id) || data?.is_blocked || false;
  const inCriticalPath = data?.in_critical_path || false;

  const strokeColor = isBlocked
    ? '#ef4444'
    : inCriticalPath
    ? '#D4AF37'
    : isFirewalled
    ? '#6E8472'
    : '#F3EED940';

  const strokeWidth = inCriticalPath ? 3 : 1.5;

  return (
    <>
      <BaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...style,
          stroke: strokeColor,
          strokeWidth,
          strokeDasharray: isBlocked ? '5,5' : inCriticalPath ? '6,3' : undefined,
          filter: inCriticalPath ? 'drop-shadow(0 0 6px rgba(212,175,55,0.7))' : undefined,
        }}
      />

      {/* Traversal Animation Particle along SVG Path */}
      {inCriticalPath && !isBlocked && (
        <circle r="3.5" fill="#E5B869" filter="drop-shadow(0 0 4px #D4AF37)">
          <animateMotion dur="2.2s" repeatCount="indefinite" path={edgePath} />
        </circle>
      )}

      {/* Interactive Quick-Action Badge on Edge (Requirement 41) */}
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            pointerEvents: 'all',
          }}
          className="group flex items-center gap-1 px-2 py-0.5 rounded-full bg-cellarNoir/90 border border-champagne/20 text-[10px] font-mono backdrop-blur-md shadow-md transition-all hover:scale-105"
        >
          <button
            title="Toggle Firewall Rule"
            onClick={(e) => {
              e.stopPropagation();
              toggleFirewall(id);
              toggleEdgeBlock(id);
            }}
            className={`p-1 rounded-full transition-colors ${
              isBlocked || isFirewalled
                ? 'text-emerald-400 hover:text-emerald-300'
                : 'text-champagne/60 hover:text-vintageGold'
            }`}
          >
            {isBlocked || isFirewalled ? (
              <Shield className="w-3 h-3 text-emerald-400" />
            ) : (
              <ShieldAlert className="w-3 h-3" />
            )}
          </button>

          <span className="text-[9px] text-champagne/60">
            {data?.port ? `:${data.port}` : '443'}
          </span>

          <button
            title="Delete Edge"
            onClick={(e) => {
              e.stopPropagation();
              deleteEdge(id);
            }}
            className="p-1 rounded-full text-champagne/40 hover:text-crimsonVermillion transition-colors"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        </div>
      </EdgeLabelRenderer>
    </>
  );
});

CustomEdge.displayName = 'CustomEdge';
