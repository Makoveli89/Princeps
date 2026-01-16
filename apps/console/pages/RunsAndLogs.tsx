import React, { useState, useEffect } from 'react';
import useSWR from 'swr';
import { AgentRun, Workspace, RunStatus } from '../types';
import {
  FileText,
  AlertCircle,
  CheckCircle2,
  Clock,
  Terminal,
  ChevronRight,
  X,
} from 'lucide-react';
import { fetcher } from '../lib/fetcher';

export const RunsAndLogs = ({ workspace }: { workspace: Workspace }) => {
  const [selectedRun, setSelectedRun] = useState<AgentRun | null>(null);
  const [filter, setFilter] = useState<'ALL' | 'FAILURE'>('ALL');

  const { data: runs, isLoading: loading } = useSWR<AgentRun[]>(
    workspace ? `/api/runs?workspaceId=${workspace.id}&limit=50` : null,
    fetcher,
    { refreshInterval: 2000 },
  );

  const filteredRuns = (runs || []).filter((run) => {
    if (filter === 'FAILURE') return run.status === RunStatus.FAILURE;
    return true;
  });

  return (
    <div className="flex h-full gap-6">
      {/* List View */}
      <div
        className={`${selectedRun ? 'w-1/2' : 'w-full'} flex flex-col transition-all duration-300`}
      >
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="gothic-font text-2xl tracking-wide text-white">Execution Traces</h2>
            <p className="mono-font text-xs text-gray-500">Immutable Logs • Workspace Scoped</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setFilter('ALL')}
              className={`mono-font border px-3 py-1 text-xs uppercase ${filter === 'ALL' ? 'border-cyan-800 bg-cyan-950/30 text-cyan-400' : 'border-gray-800 bg-[#050505] text-gray-600'}`}
            >
              All Logs
            </button>
            <button
              onClick={() => setFilter('FAILURE')}
              className={`mono-font border px-3 py-1 text-xs uppercase ${filter === 'FAILURE' ? 'border-red-800 bg-red-950/30 text-red-400' : 'border-gray-800 bg-[#050505] text-gray-600'}`}
            >
              Failures
            </button>
          </div>
        </div>

        <div className="relative flex-1 overflow-auto border border-gray-800 bg-[#050505]">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 font-mono text-xs text-cyan-500">
              LOADING TRACES...
            </div>
          )}

          <table className="w-full border-collapse text-left">
            <thead className="sticky top-0 bg-[#0a0a0a] text-[10px] uppercase tracking-widest text-gray-500">
              <tr>
                <th className="border-b border-gray-800 p-3 font-normal">Status</th>
                <th className="border-b border-gray-800 p-3 font-normal">Agent</th>
                <th className="border-b border-gray-800 p-3 font-normal">Preview</th>
                <th className="border-b border-gray-800 p-3 text-right font-normal">Duration</th>
                <th className="border-b border-gray-800 p-3 text-right font-normal">Time</th>
              </tr>
            </thead>
            <tbody className="mono-font text-xs">
              {filteredRuns.length === 0 && !loading && (
                <tr>
                  <td colSpan={5} className="p-10 text-center italic text-gray-600">
                    No execution traces found.
                  </td>
                </tr>
              )}
              {filteredRuns.map((run) => (
                <tr
                  key={run.run_id}
                  onClick={() => setSelectedRun(run)}
                  className={`cursor-pointer border-b border-gray-800/50 transition-colors ${selectedRun?.run_id === run.run_id ? 'bg-cyan-950/20' : 'hover:bg-gray-900/40'} `}
                >
                  <td className="p-3">
                    {run.status === RunStatus.SUCCESS ? (
                      <span className="flex items-center gap-1 text-emerald-500">
                        <CheckCircle2 size={12} /> OK
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-red-500">
                        <AlertCircle size={12} /> ERR
                      </span>
                    )}
                  </td>
                  <td className="p-3 text-cyan-600">{run.agent}</td>
                  <td className="max-w-xs truncate p-3 text-gray-400">{run.input_preview}</td>
                  <td className="p-3 text-right text-gray-600">{run.duration_ms}ms</td>
                  <td className="p-3 text-right text-gray-600">
                    {new Date(run.timestamp).toLocaleTimeString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail Panel */}
      {selectedRun && (
        <div className="animate-in slide-in-from-right-4 relative flex w-1/2 flex-col border border-gray-800 bg-[#020202] duration-300">
          <button
            onClick={() => setSelectedRun(null)}
            className="absolute right-2 top-2 p-1 text-gray-600 hover:text-white"
          >
            <X size={16} />
          </button>

          <div className="border-b border-gray-800 bg-[#080808] p-4">
            <div className="mb-1 flex items-center gap-2">
              <span
                className={`border px-1.5 py-0.5 text-xs ${selectedRun.status === 'SUCCESS' ? 'border-emerald-900 bg-emerald-950/20 text-emerald-500' : 'border-red-900 bg-red-950/20 text-red-500'} mono-font`}
              >
                {selectedRun.status}
              </span>
              <span className="mono-font text-xs text-gray-500">ID: {selectedRun.run_id}</span>
            </div>
            <h3 className="truncate font-mono text-lg text-white">{selectedRun.agent}</h3>
          </div>

          <div className="flex-1 space-y-6 overflow-auto p-4">
            {/* Internal Monologue / Logs */}
            <div>
              <h4 className="mb-2 flex items-center gap-2 text-[10px] uppercase tracking-widest text-gray-500">
                <Terminal size={12} /> System Logs
              </h4>
              <div className="mono-font space-y-1 rounded-sm border border-gray-800 bg-black p-3 text-xs">
                {selectedRun.logs?.map((log, i) => (
                  <div key={i} className="flex gap-2 text-gray-400">
                    <span className="select-none text-gray-700">$</span>
                    <span>{log}</span>
                  </div>
                )) || <span className="italic text-gray-700">No detailed logs stored.</span>}
              </div>
            </div>

            {/* IO Payloads */}
            <div>
              <h4 className="mb-2 text-[10px] uppercase tracking-widest text-gray-500">
                Input Context
              </h4>
              <div className="mono-font overflow-x-auto rounded-sm border border-gray-800 bg-[#050505] p-3 text-xs text-gray-300">
                {selectedRun.input_preview}
              </div>
            </div>

            <div>
              <h4 className="mb-2 text-[10px] uppercase tracking-widest text-gray-500">
                Output Preview
              </h4>
              <pre className="mono-font overflow-x-auto rounded-sm border border-gray-800 bg-[#050505] p-3 text-xs text-cyan-600">
                {selectedRun.output_preview}
              </pre>
            </div>
          </div>

          <div className="mono-font flex justify-between border-t border-gray-800 bg-[#080808] p-2 text-[10px] uppercase text-gray-600">
            <span>Latency: {selectedRun.duration_ms}ms</span>
            <span>Timestamp: {selectedRun.timestamp}</span>
          </div>
        </div>
      )}
    </div>
  );
};
