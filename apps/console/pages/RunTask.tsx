import React, { useState } from 'react';
import { Play, Terminal, ChevronRight, Loader2, Cpu, CheckCircle2, Download, FileJson, AlertCircle } from 'lucide-react';
import { Workspace } from '../types';

export const RunTask = ({ workspace }: { workspace: Workspace }) => {
  const [prompt, setPrompt] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [output, setOutput] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    if (!prompt.trim()) return;
    setIsRunning(true);
    setLogs([]);
    setOutput(null);
    setError(null);
    setLogs(prev => [...prev, `[DISPATCHER] Initializing task in workspace: ${workspace?.id || 'unknown'}...`]);

    try {
        const res = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                agentId: 'PlannerAgent', // Default for now
                input: prompt,
                workspaceId: workspace.id
            })
        });

        if (res.ok) {
            const data = await res.json();
            setLogs(prev => [...prev, `[SUCCESS] Task completed. ID: ${data.run_id}`]);
            setOutput(JSON.stringify(data, null, 2));
        } else {
             const errData = await res.json().catch(() => ({}));
             setError(errData.detail || "Unknown error occurred.");
             setLogs(prev => [...prev, `[ERROR] Task failed: ${errData.detail || res.statusText}`]);
        }
    } catch (e) {
        setError(String(e));
        setLogs(prev => [...prev, `[CRITICAL] Network/System failure: ${e}`]);
    } finally {
        setIsRunning(false);
    }
  };

  return (
    <div className="h-full flex flex-col gap-6 animate-in fade-in duration-500">
      <div className="flex items-center justify-between">
         <h2 className="text-2xl text-white gothic-font text-glow">Execute Protocol</h2>
         <div className="text-[10px] mono-font text-cyan-500 border border-cyan-900/50 bg-cyan-950/10 px-3 py-1 rounded-sm flex items-center gap-2 shadow-[0_0_10px_rgba(0,243,255,0.1)]">
            <Cpu size={12} className="animate-pulse" />
            ACTIVE WORKSPACE: <span className="text-cyan-300 font-bold">{workspace?.name.toUpperCase()}</span>
         </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
         {/* Input Column */}
         <div className="flex flex-col gap-4">
            <div className="flex-1 bg-[#030303] border border-gray-800 p-1 relative group focus-within:border-cyan-800 transition-colors duration-300 shadow-2xl">
                {/* Tech Corners */}
                <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-cyan-600 opacity-50 group-hover:opacity-100 transition-opacity"></div>
                <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-cyan-600 opacity-50 group-hover:opacity-100 transition-opacity"></div>
                <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-cyan-600 opacity-50 group-hover:opacity-100 transition-opacity"></div>
                <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-cyan-600 opacity-50 group-hover:opacity-100 transition-opacity"></div>

                <textarea
                    className="w-full h-full bg-[#050505] border-none focus:ring-0 text-gray-300 p-6 mono-font resize-none outline-none text-sm placeholder-gray-700 leading-relaxed custom-scrollbar"
                    placeholder="// Input task instructions or protocol parameters here..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    disabled={isRunning}
                />
            </div>

            <div className="bg-[#030303] border border-gray-800 p-4 relative">
                <h3 className="text-[10px] text-gray-500 uppercase tracking-widest block mb-3 font-bold">Workflow Strategy</h3>
                <div className="flex gap-2">
                    <button className="flex-1 py-3 bg-cyan-950/20 border border-cyan-700/50 text-cyan-400 text-[10px] font-bold uppercase hover:bg-cyan-900/40 transition-colors shadow-[0_0_10px_rgba(0,243,255,0.05)]">
                        Autonomous (Plan+Exec)
                    </button>
                    <button className="flex-1 py-3 bg-[#0a0a0a] border border-gray-800 text-gray-600 text-[10px] font-bold uppercase hover:text-gray-300 transition-colors hover:border-gray-600">
                        Planning Only
                    </button>
                    <button className="flex-1 py-3 bg-[#0a0a0a] border border-gray-800 text-gray-600 text-[10px] font-bold uppercase hover:text-gray-300 transition-colors hover:border-gray-600">
                        RAG Query
                    </button>
                </div>
            </div>

            <button
                onClick={handleRun}
                disabled={isRunning || !prompt}
                className={`w-full py-4 flex items-center justify-center gap-3 text-sm font-bold tracking-[0.2em] transition-all relative overflow-hidden group
                    ${isRunning
                        ? 'bg-gray-900 text-gray-500 cursor-not-allowed border border-gray-800'
                        : 'bg-gradient-to-r from-red-950 to-red-900 hover:from-red-900 hover:to-red-800 text-white border border-red-600/50 shadow-[0_0_20px_rgba(255,0,60,0.2)] hover:shadow-[0_0_30px_rgba(255,0,60,0.4)]'
                    }`}
            >
                {/* Button Glitch Effect Overlay */}
                {!isRunning && <div className="absolute top-0 -left-full w-full h-full bg-gradient-to-r from-transparent via-red-500/20 to-transparent skew-x-12 group-hover:animate-[shimmer_1s_infinite]"></div>}

                {isRunning ? <Loader2 className="animate-spin text-red-500" /> : <Play size={16} className="fill-current" />}
                <span className="relative z-10">{isRunning ? 'PROCESSING STREAM...' : 'INITIALIZE RUN'}</span>
            </button>
         </div>

         {/* Output Column */}
         <div className={`bg-[#030303] border ${output ? 'border-emerald-900/50 shadow-[0_0_20px_rgba(16,185,129,0.1)]' : (error ? 'border-red-900/50' : 'border-gray-800')} flex flex-col relative overflow-hidden transition-all duration-500`}>
            {/* Header */}
            <div className={`h-10 border-b flex items-center px-4 justify-between ${output ? 'bg-emerald-950/20 border-emerald-900/50' : (error ? 'bg-red-950/20 border-red-900/50' : 'bg-[#050505] border-gray-800')}`}>
                <span className={`text-[10px] uppercase tracking-widest flex items-center gap-2 ${output ? 'text-emerald-500' : (error ? 'text-red-500' : 'text-gray-500')}`}>
                    <Terminal size={12} /> {output ? 'Mission Log' : 'Console Output'}
                </span>
                {output && (
                    <span className="text-emerald-400 text-[10px] font-bold tracking-widest flex items-center gap-1 animate-pulse">
                        <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></div> COMPLETE
                    </span>
                )}
                 {error && (
                    <span className="text-red-400 text-[10px] font-bold tracking-widest flex items-center gap-1">
                        <AlertCircle size={12} /> FAILED
                    </span>
                )}
            </div>

            {/* Logs Area */}
            <div className="flex-1 p-4 overflow-auto mono-font text-xs space-y-1.5 custom-scrollbar relative">
                {logs.length === 0 && !output && !error && (
                    <div className="absolute inset-0 flex items-center justify-center flex-col opacity-30 pointer-events-none">
                        <Terminal size={48} className="text-gray-600 mb-4" />
                        <span className="text-gray-500 tracking-widest uppercase text-[10px]">Awaiting Protocol Initialization</span>
                    </div>
                )}

                {logs.map((log, i) => (
                    <div key={i} className="flex gap-3 text-cyan-500/90 animate-in fade-in slide-in-from-left-4 duration-300 border-l border-cyan-900/0 hover:border-cyan-900/50 hover:bg-cyan-950/10 pl-2 transition-colors">
                        <ChevronRight size={14} className="mt-0.5 flex-shrink-0 opacity-50 text-cyan-700" />
                        <span className="tracking-wide text-gray-300">{log}</span>
                    </div>
                ))}

                {/* Success Banner */}
                {output && (
                    <div className="mt-8 mb-4 animate-in zoom-in-95 duration-500 slide-in-from-bottom-4">
                        <div className="bg-gradient-to-r from-emerald-950/40 to-transparent border-l-2 border-emerald-500 p-4 mb-4 relative overflow-hidden group">
                             <div className="absolute inset-0 bg-emerald-500/5 translate-x-[-100%] group-hover:translate-x-0 transition-transform duration-700"></div>
                             <div className="flex items-center gap-3 mb-2 relative z-10">
                                <CheckCircle2 size={18} className="text-emerald-500" />
                                <h3 className="text-emerald-400 font-bold gothic-font tracking-widest">PROTOCOL EXECUTED SUCCESSFULLY</h3>
                             </div>
                             <p className="text-emerald-600/70 text-[10px] mono-font pl-8 relative z-10">
                                All objectives met. Artifacts generated and verified.
                             </p>
                        </div>

                        <div className="pl-4 border-l border-gray-800 ml-2 space-y-4">
                            <div>
                                <div className="text-gray-500 text-[10px] uppercase tracking-widest mb-2">Final Payload</div>
                                <pre className="text-gray-400 bg-[#050505] p-3 rounded-sm border border-gray-800 text-[10px] overflow-x-auto shadow-inner">
                                    {output}
                                </pre>
                            </div>

                            <div className="flex gap-3 pt-2">
                                <button className="flex items-center gap-2 px-4 py-2 bg-gray-900 border border-gray-700 text-gray-300 hover:text-white hover:border-cyan-500 hover:bg-cyan-950/20 transition-all text-[10px] uppercase tracking-widest font-bold group">
                                    <FileJson size={14} className="text-gray-500 group-hover:text-cyan-400 transition-colors" /> View Log
                                </button>
                                <button className="flex items-center gap-2 px-4 py-2 bg-gray-900 border border-gray-700 text-gray-300 hover:text-white hover:border-emerald-500 hover:bg-emerald-950/20 transition-all text-[10px] uppercase tracking-widest font-bold group">
                                    <Download size={14} className="text-gray-500 group-hover:text-emerald-400 transition-colors" /> Download Artifacts
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
         </div>
      </div>
    </div>
  );
};
