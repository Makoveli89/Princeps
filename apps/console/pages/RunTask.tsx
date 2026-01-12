import React, { useState } from 'react';
import {
  Play,
  Terminal,
  ChevronRight,
  Loader2,
  Cpu,
  CheckCircle2,
  Download,
  FileJson,
  AlertCircle,
} from 'lucide-react';
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
    setLogs((prev) => [
      ...prev,
      `[DISPATCHER] Initializing task in workspace: ${workspace?.id || 'unknown'}...`,
    ]);

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agentId: 'PlannerAgent', // Default for now
          input: prompt,
          workspaceId: workspace.id,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setLogs((prev) => [...prev, `[SUCCESS] Task completed. ID: ${data.run_id}`]);
        setOutput(JSON.stringify(data, null, 2));
      } else {
        const errData = await res.json().catch(() => ({}));
        setError(errData.detail || 'Unknown error occurred.');
        setLogs((prev) => [...prev, `[ERROR] Task failed: ${errData.detail || res.statusText}`]);
      }
    } catch (e) {
      setError(String(e));
      setLogs((prev) => [...prev, `[CRITICAL] Network/System failure: ${e}`]);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="animate-in fade-in flex h-full flex-col gap-6 duration-500">
      <div className="flex items-center justify-between">
        <h2 className="gothic-font text-glow text-2xl text-white">Execute Protocol</h2>
        <div className="mono-font flex items-center gap-2 rounded-sm border border-cyan-900/50 bg-cyan-950/10 px-3 py-1 text-[10px] text-cyan-500 shadow-[0_0_10px_rgba(0,243,255,0.1)]">
          <Cpu size={12} className="animate-pulse" />
          ACTIVE WORKSPACE:{' '}
          <span className="font-bold text-cyan-300">{workspace?.name.toUpperCase()}</span>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Input Column */}
        <div className="flex flex-col gap-4">
          <div className="group relative flex-1 border border-gray-800 bg-[#030303] p-1 shadow-2xl transition-colors duration-300 focus-within:border-cyan-800">
            {/* Tech Corners */}
            <div className="absolute left-0 top-0 h-3 w-3 border-l-2 border-t-2 border-cyan-600 opacity-50 transition-opacity group-hover:opacity-100"></div>
            <div className="absolute right-0 top-0 h-3 w-3 border-r-2 border-t-2 border-cyan-600 opacity-50 transition-opacity group-hover:opacity-100"></div>
            <div className="absolute bottom-0 left-0 h-3 w-3 border-b-2 border-l-2 border-cyan-600 opacity-50 transition-opacity group-hover:opacity-100"></div>
            <div className="absolute bottom-0 right-0 h-3 w-3 border-b-2 border-r-2 border-cyan-600 opacity-50 transition-opacity group-hover:opacity-100"></div>

            <textarea
              className="mono-font custom-scrollbar h-full w-full resize-none border-none bg-[#050505] p-6 text-sm leading-relaxed text-gray-300 placeholder-gray-700 outline-none focus:ring-0"
              placeholder="// Input task instructions or protocol parameters here..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={isRunning}
            />
          </div>

          <div className="relative border border-gray-800 bg-[#030303] p-4">
            <label className="mb-3 block text-[10px] font-bold uppercase tracking-widest text-gray-500">
              Workflow Strategy
            </label>
            <div className="flex gap-2">
              <button className="flex-1 border border-cyan-700/50 bg-cyan-950/20 py-3 text-[10px] font-bold uppercase text-cyan-400 shadow-[0_0_10px_rgba(0,243,255,0.05)] transition-colors hover:bg-cyan-900/40">
                Autonomous (Plan+Exec)
              </button>
              <button className="flex-1 border border-gray-800 bg-[#0a0a0a] py-3 text-[10px] font-bold uppercase text-gray-600 transition-colors hover:border-gray-600 hover:text-gray-300">
                Planning Only
              </button>
              <button className="flex-1 border border-gray-800 bg-[#0a0a0a] py-3 text-[10px] font-bold uppercase text-gray-600 transition-colors hover:border-gray-600 hover:text-gray-300">
                RAG Query
              </button>
            </div>
          </div>

          <button
            onClick={handleRun}
            disabled={isRunning || !prompt}
            className={`group relative flex w-full items-center justify-center gap-3 overflow-hidden py-4 text-sm font-bold tracking-[0.2em] transition-all ${
              isRunning
                ? 'cursor-not-allowed border border-gray-800 bg-gray-900 text-gray-500'
                : 'border border-red-600/50 bg-gradient-to-r from-red-950 to-red-900 text-white shadow-[0_0_20px_rgba(255,0,60,0.2)] hover:from-red-900 hover:to-red-800 hover:shadow-[0_0_30px_rgba(255,0,60,0.4)]'
            }`}
          >
            {/* Button Glitch Effect Overlay */}
            {!isRunning && (
              <div className="absolute -left-full top-0 h-full w-full skew-x-12 bg-gradient-to-r from-transparent via-red-500/20 to-transparent group-hover:animate-[shimmer_1s_infinite]"></div>
            )}

            {isRunning ? (
              <Loader2 className="animate-spin text-red-500" />
            ) : (
              <Play size={16} className="fill-current" />
            )}
            <span className="relative z-10">
              {isRunning ? 'PROCESSING STREAM...' : 'INITIALIZE RUN'}
            </span>
          </button>
        </div>

        {/* Output Column */}
        <div
          className={`border bg-[#030303] ${output ? 'border-emerald-900/50 shadow-[0_0_20px_rgba(16,185,129,0.1)]' : error ? 'border-red-900/50' : 'border-gray-800'} relative flex flex-col overflow-hidden transition-all duration-500`}
        >
          {/* Header */}
          <div
            className={`flex h-10 items-center justify-between border-b px-4 ${output ? 'border-emerald-900/50 bg-emerald-950/20' : error ? 'border-red-900/50 bg-red-950/20' : 'border-gray-800 bg-[#050505]'}`}
          >
            <span
              className={`flex items-center gap-2 text-[10px] uppercase tracking-widest ${output ? 'text-emerald-500' : error ? 'text-red-500' : 'text-gray-500'}`}
            >
              <Terminal size={12} /> {output ? 'Mission Log' : 'Console Output'}
            </span>
            {output && (
              <span className="flex animate-pulse items-center gap-1 text-[10px] font-bold tracking-widest text-emerald-400">
                <div className="h-1.5 w-1.5 rounded-full bg-emerald-500"></div> COMPLETE
              </span>
            )}
            {error && (
              <span className="flex items-center gap-1 text-[10px] font-bold tracking-widest text-red-400">
                <AlertCircle size={12} /> FAILED
              </span>
            )}
          </div>

          {/* Logs Area */}
          <div className="mono-font custom-scrollbar relative flex-1 space-y-1.5 overflow-auto p-4 text-xs">
            {logs.length === 0 && !output && !error && (
              <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center opacity-30">
                <Terminal size={48} className="mb-4 text-gray-600" />
                <span className="text-[10px] uppercase tracking-widest text-gray-500">
                  Awaiting Protocol Initialization
                </span>
              </div>
            )}

            {logs.map((log, i) => (
              <div
                key={i}
                className="animate-in fade-in slide-in-from-left-4 flex gap-3 border-l border-cyan-900/0 pl-2 text-cyan-500/90 transition-colors duration-300 hover:border-cyan-900/50 hover:bg-cyan-950/10"
              >
                <ChevronRight size={14} className="mt-0.5 flex-shrink-0 text-cyan-700 opacity-50" />
                <span className="tracking-wide text-gray-300">{log}</span>
              </div>
            ))}

            {/* Success Banner */}
            {output && (
              <div className="animate-in zoom-in-95 slide-in-from-bottom-4 mb-4 mt-8 duration-500">
                <div className="group relative mb-4 overflow-hidden border-l-2 border-emerald-500 bg-gradient-to-r from-emerald-950/40 to-transparent p-4">
                  <div className="absolute inset-0 translate-x-[-100%] bg-emerald-500/5 transition-transform duration-700 group-hover:translate-x-0"></div>
                  <div className="relative z-10 mb-2 flex items-center gap-3">
                    <CheckCircle2 size={18} className="text-emerald-500" />
                    <h3 className="gothic-font font-bold tracking-widest text-emerald-400">
                      PROTOCOL EXECUTED SUCCESSFULLY
                    </h3>
                  </div>
                  <p className="mono-font relative z-10 pl-8 text-[10px] text-emerald-600/70">
                    All objectives met. Artifacts generated and verified.
                  </p>
                </div>

                <div className="ml-2 space-y-4 border-l border-gray-800 pl-4">
                  <div>
                    <div className="mb-2 text-[10px] uppercase tracking-widest text-gray-500">
                      Final Payload
                    </div>
                    <pre className="overflow-x-auto rounded-sm border border-gray-800 bg-[#050505] p-3 text-[10px] text-gray-400 shadow-inner">
                      {output}
                    </pre>
                  </div>

                  <div className="flex gap-3 pt-2">
                    <button className="group flex items-center gap-2 border border-gray-700 bg-gray-900 px-4 py-2 text-[10px] font-bold uppercase tracking-widest text-gray-300 transition-all hover:border-cyan-500 hover:bg-cyan-950/20 hover:text-white">
                      <FileJson
                        size={14}
                        className="text-gray-500 transition-colors group-hover:text-cyan-400"
                      />{' '}
                      View Log
                    </button>
                    <button className="group flex items-center gap-2 border border-gray-700 bg-gray-900 px-4 py-2 text-[10px] font-bold uppercase tracking-widest text-gray-300 transition-all hover:border-emerald-500 hover:bg-emerald-950/20 hover:text-white">
                      <Download
                        size={14}
                        className="text-gray-500 transition-colors group-hover:text-emerald-400"
                      />{' '}
                      Download Artifacts
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
