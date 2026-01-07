import React, { useState, useEffect } from 'react';
import useSWR from 'swr';
import { AgentRun, Workspace, RunStatus } from '../types';
import { FileText, AlertCircle, CheckCircle2, Clock, Terminal, ChevronRight, X } from 'lucide-react';
import { fetcher } from '../lib/fetcher';

export const RunsAndLogs = ({ workspace }: { workspace: Workspace }) => {
    const [selectedRun, setSelectedRun] = useState<AgentRun | null>(null);
    const [filter, setFilter] = useState<'ALL' | 'FAILURE'>('ALL');

    const { data: runs, isLoading: loading } = useSWR<AgentRun[]>(
        workspace ? `/api/runs?workspaceId=${workspace.id}&limit=50` : null,
        fetcher,
        { refreshInterval: 2000 }
    );

    const filteredRuns = (runs || []).filter(run => {
        if (filter === 'FAILURE') return run.status === RunStatus.FAILURE;
        return true;
    });

    return (
        <div className="h-full flex gap-6">
            {/* List View */}
            <div className={`${selectedRun ? 'w-1/2' : 'w-full'} flex flex-col transition-all duration-300`}>
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h2 className="text-2xl text-white gothic-font tracking-wide">Execution Traces</h2>
                        <p className="text-gray-500 text-xs mono-font">Immutable Logs • Workspace Scoped</p>
                    </div>
                    <div className="flex gap-2">
                         <button
                            onClick={() => setFilter('ALL')}
                            className={`px-3 py-1 text-xs uppercase mono-font border ${filter === 'ALL' ? 'bg-cyan-950/30 text-cyan-400 border-cyan-800' : 'bg-[#050505] text-gray-600 border-gray-800'}`}
                         >
                            All Logs
                         </button>
                         <button
                            onClick={() => setFilter('FAILURE')}
                            className={`px-3 py-1 text-xs uppercase mono-font border ${filter === 'FAILURE' ? 'bg-red-950/30 text-red-400 border-red-800' : 'bg-[#050505] text-gray-600 border-gray-800'}`}
                         >
                            Failures
                         </button>
                    </div>
                </div>

                <div className="flex-1 overflow-auto border border-gray-800 bg-[#050505] relative">
                    {loading && <div className="absolute inset-0 bg-black/50 flex items-center justify-center text-cyan-500 font-mono text-xs">LOADING TRACES...</div>}

                    <table className="w-full text-left border-collapse">
                        <thead className="bg-[#0a0a0a] text-gray-500 text-[10px] uppercase tracking-widest sticky top-0">
                            <tr>
                                <th className="p-3 font-normal border-b border-gray-800">Status</th>
                                <th className="p-3 font-normal border-b border-gray-800">Agent</th>
                                <th className="p-3 font-normal border-b border-gray-800">Preview</th>
                                <th className="p-3 font-normal border-b border-gray-800 text-right">Duration</th>
                                <th className="p-3 font-normal border-b border-gray-800 text-right">Time</th>
                            </tr>
                        </thead>
                        <tbody className="mono-font text-xs">
                            {filteredRuns.length === 0 && !loading && (
                                <tr>
                                    <td colSpan={5} className="p-10 text-center text-gray-600 italic">No execution traces found.</td>
                                </tr>
                            )}
                            {filteredRuns.map(run => (
                                <tr
                                    key={run.run_id}
                                    onClick={() => setSelectedRun(run)}
                                    className={`
                                        cursor-pointer transition-colors border-b border-gray-800/50
                                        ${selectedRun?.run_id === run.run_id ? 'bg-cyan-950/20' : 'hover:bg-gray-900/40'}
                                    `}
                                >
                                    <td className="p-3">
                                        {run.status === RunStatus.SUCCESS ?
                                            <span className="text-emerald-500 flex items-center gap-1"><CheckCircle2 size={12}/> OK</span> :
                                            <span className="text-red-500 flex items-center gap-1"><AlertCircle size={12}/> ERR</span>
                                        }
                                    </td>
                                    <td className="p-3 text-cyan-600">{run.agent}</td>
                                    <td className="p-3 text-gray-400 max-w-xs truncate">{run.input_preview}</td>
                                    <td className="p-3 text-right text-gray-600">{run.duration_ms}ms</td>
                                    <td className="p-3 text-right text-gray-600">{new Date(run.timestamp).toLocaleTimeString()}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Detail Panel */}
            {selectedRun && (
                <div className="w-1/2 bg-[#020202] border border-gray-800 flex flex-col relative animate-in slide-in-from-right-4 duration-300">
                    <button
                        onClick={() => setSelectedRun(null)}
                        className="absolute top-2 right-2 p-1 text-gray-600 hover:text-white"
                    >
                        <X size={16} />
                    </button>

                    <div className="p-4 border-b border-gray-800 bg-[#080808]">
                        <div className="flex items-center gap-2 mb-1">
                            <span className={`text-xs px-1.5 py-0.5 border ${selectedRun.status === 'SUCCESS' ? 'border-emerald-900 text-emerald-500 bg-emerald-950/20' : 'border-red-900 text-red-500 bg-red-950/20'} mono-font`}>
                                {selectedRun.status}
                            </span>
                            <span className="text-gray-500 text-xs mono-font">ID: {selectedRun.run_id}</span>
                        </div>
                        <h3 className="text-lg text-white font-mono truncate">{selectedRun.agent}</h3>
                    </div>

                    <div className="flex-1 overflow-auto p-4 space-y-6">
                        {/* Internal Monologue / Logs */}
                        <div>
                             <h4 className="text-[10px] text-gray-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                                <Terminal size={12} /> System Logs
                             </h4>
                             <div className="bg-black border border-gray-800 p-3 rounded-sm mono-font text-xs space-y-1">
                                {selectedRun.logs?.map((log, i) => (
                                    <div key={i} className="flex gap-2 text-gray-400">
                                        <span className="text-gray-700 select-none">$</span>
                                        <span>{log}</span>
                                    </div>
                                )) || <span className="text-gray-700 italic">No detailed logs stored.</span>}
                             </div>
                        </div>

                        {/* IO Payloads */}
                        <div>
                             <h4 className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">Input Context</h4>
                             <div className="bg-[#050505] border border-gray-800 p-3 rounded-sm mono-font text-xs text-gray-300 overflow-x-auto">
                                {selectedRun.input_preview}
                             </div>
                        </div>

                        <div>
                             <h4 className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">Output Preview</h4>
                             <pre className="bg-[#050505] border border-gray-800 p-3 rounded-sm mono-font text-xs text-cyan-600 overflow-x-auto">
                                {selectedRun.output_preview}
                             </pre>
                        </div>
                    </div>

                    <div className="p-2 border-t border-gray-800 bg-[#080808] flex justify-between text-[10px] mono-font text-gray-600 uppercase">
                        <span>Latency: {selectedRun.duration_ms}ms</span>
                        <span>Timestamp: {selectedRun.timestamp}</span>
                    </div>
                </div>
            )}
        </div>
    );
};
