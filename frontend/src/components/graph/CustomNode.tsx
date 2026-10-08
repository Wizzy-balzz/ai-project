import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import {
  Server,
  Shield,
  Trash2,
  Eye,
  Database,
  Router as RouterIcon,
  Laptop,
  Radio,
  Lock,
} from 'lucide-react';
import { NodeData } from '../../api/client';
import { useScenarioStore } from '../../store/scenarioStore';
import { useGraphStore } from '../../store/graphStore';

interface CustomNodeProps {
  id: string;
  data: NodeData;
  selected?: boolean;
}

const getRoleIcon = (role: string) => {
  const r = (role || '').toLowerCase();
  if (r === 'database') return <Database className="w-4 h-4" />;
  if (r === 'router' || r === 'firewall') return <RouterIcon className="w-4 h-4" />;
  if (r === 'workstation') return <Laptop className="w-4 h-4" />;
  if (r === 'iot_device') return <Radio className="w-4 h-4" />;
  return <Server className="w-4 h-4" />;
};

export const CustomNode = memo(({ id, data, selected }: CustomNodeProps) => {
  const {
    setInspectedNodeId,
    setStage,
    toggleNodeIsolation,
    isolatedNodeIds,
    selectedScenarioId,
  } = useScenarioStore();

  const { deleteNode } = useGraphStore();

  const isIsolated = isolatedNodeIds.includes(id) || data.is_isolated;
  const inCriticalPath = data.in_critical_path;
  const cvss = data.cvss || 0;

  const cvssBadgeColor =
    cvss >= 8.5
      ? 'text-crimsonVermillion border-crimsonVermillion/40 bg-crimsonVermillion/15'
      : cvss >= 6.5
      ? 'text-vintageGold border-vintageGold/40 bg-vintageGold/15'
      : 'text-champagne/60 border-champagne/20 bg-champagne/5';

  return (
    <div
      className={`relative group rounded-2xl p-4 min-w-[210px] max-w-[240px] border backdrop-blur-2xl transition-all duration-300 ${
        isIsolated
          ? 'opacity-40 grayscale border-dashed border-red-500/40 bg-cellarNoir/80'
          : inCriticalPath
          ? 'border-vintageGold bg-gradient-to-b from-velvetOlive/95 to-cellarNoir/95 shadow-[0_0_30px_rgba(212,175,55,0.35)] ring-1 ring-vintageGold/50'
          : selected
          ? 'border-champagne/60 bg-velvetOlive/90 shadow-lg'
          : 'border-champagne/15 bg-velvetOlive/70 hover:border-champagne/40'
      }`}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!w-2.5 !h-2.5 !bg-vintageGold !border-2 !border-cellarNoir -left-1.5"
      />

      {/* Top Header Badge */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5 text-vintageGold">
          {getRoleIcon(data.role)}
          <span className="text-[10px] uppercase font-mono tracking-wider text-champagne/60">
            {data.role}
          </span>
        </div>

        {cvss > 0 && (
          <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono border font-semibold ${cvssBadgeColor}`}>
            CVSS {cvss.toFixed(1)}
          </span>
        )}
      </div>

      {/* Node Identification */}
      <div className="mb-2">
        <div className="text-xs font-mono font-semibold text-champagne truncate">
          {data.ip}
        </div>
        <div className="text-[10px] font-light text-champagne/50 truncate">
          {data.hostname || data.ip}
        </div>
      </div>

      {/* Entry or Target Pill */}
      {data.is_entry && (
        <span className="inline-block px-2 py-0.5 rounded-full text-[9px] font-mono uppercase tracking-wider text-crimsonVermillion border border-crimsonVermillion/40 bg-crimsonVermillion/10 mb-1">
          Perimeter Entry
        </span>
      )}
      {data.is_target && (
        <span className="inline-block px-2 py-0.5 rounded-full text-[9px] font-mono uppercase tracking-wider text-vintageGold border border-vintageGold/40 bg-vintageGold/10 mb-1">
          Target Crown Jewel
        </span>
      )}

      {/* Hover Micro-Menu (Requirement 39) */}
      <div className="absolute -top-10 left-1/2 -translate-x-1/2 hidden group-hover:flex items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-cellarNoir/95 border border-champagne/20 shadow-2xl backdrop-blur-xl z-50">
        <button
          title="Inspect Node"
          onClick={(e) => {
            e.stopPropagation();
            setInspectedNodeId(id);
            setStage('INSPECTOR');
          }}
          className="p-1.5 rounded-full hover:bg-vintageGold/20 text-champagne/80 hover:text-vintageGold transition-colors"
        >
          <Eye className="w-3.5 h-3.5" />
        </button>

        <button
          title="Simulate Isolation"
          onClick={(e) => {
            e.stopPropagation();
            toggleNodeIsolation(id);
          }}
          className={`p-1.5 rounded-full transition-colors ${
            isIsolated
              ? 'text-emerald-400 bg-emerald-500/20'
              : 'hover:bg-vintageGold/20 text-champagne/80 hover:text-vintageGold'
          }`}
        >
          <Shield className="w-3.5 h-3.5" />
        </button>

        {selectedScenarioId && (
          <button
            title="Delete Node (Clean Cascade)"
            onClick={(e) => {
              e.stopPropagation();
              deleteNode(id, selectedScenarioId);
            }}
            className="p-1.5 rounded-full hover:bg-crimsonVermillion/20 text-champagne/80 hover:text-crimsonVermillion transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      <Handle
        type="source"
        position={Position.Right}
        className="!w-2.5 !h-2.5 !bg-vintageGold !border-2 !border-cellarNoir -right-1.5"
      />
    </div>
  );
});

CustomNode.displayName = 'CustomNode';
