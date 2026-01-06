import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend
} from 'recharts';
import { Activity, Database, Server, AlertTriangle, ShieldCheck } from 'lucide-react';
import { Workspace, MetricPoint, AgentRun } from '../types';

const StatusCard = ({ title, status, icon: Icon, detail }: { title: string, status: 'ok' | 'err' | 'warn', icon: any, detail: string }) => {
    const color = status === 'ok' ? 'text-emerald-400' : status === 'err' ? 'text-red-500' : 'text-amber-400';
    const border = status === 'ok' ? 'border-emerald-900/30 hover:border-emerald-500/50' : status === 'err' ? 'border-red-900/30 hover:border-red-500/50' : 'border-amber-900/30 hover:border-amber-500/50';
    const bg = status === 'ok' ? 'bg-emerald-950/10' : status === 'err' ? 'bg-red-950/10' : 'bg-amber-950/10';
    const glow = status === 'ok' ? 'shadow-[0_0_10px_rgba(16,185,129,0.1)]' : status === 'err' ? 'shadow-[0_0_10px_rgba(239,68,68,0.1)]' : 'shadow-[0_0_10px_rgba(245,158,11,0.1)]';

    return (
        <div className={`p-4 border ${border} ${bg} ${glow} rounded-sm flex items-start gap-4 relative overflow-hidden group transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_0_20px_rgba(0,0,0,0.5)]`}>
            <div className={`absolute -right-6 -top-6 w-16 h-16 rounded-full ${status === 'ok' ? 'bg-emerald-500' : 'bg-red-500'} blur-[40px] opacity-10 group-hover:opacity-20 transition-opacity duration-500`}></div>
            <div className={`p-2.5 rounded-sm bg-[#050505] border border-gray-800 ${color} shadow-inner`}>
                <Icon size={20} className="drop-shadow-[0_0_5px_currentColor]" />
            </div>
            <div>
                <h3 className="text-gray-500 text-[10px] uppercase tracking-[0.2em] font-mono group-hover:text-gray-300 transition-colors">{title}</h3>
                <div className={`text-lg font-bold ${color} gothic-font mt-1 ${status !== 'ok' ? 'animate-pulse' : ''} drop-shadow-[0_0_3px_currentColor]`}>
                    {status === 'ok' ? 'ONLINE' : status === 'err' ? 'OFFLINE' : 'DEGRADED'}
                </div>
                <p className="text-[10px] text-gray-600 font-mono mt-1 group-hover:text-gray-400 transition-colors">{detail}</p>
            </div>
        </div>
    );
};

export const Dashboard = ({ workspace }: { workspace: Workspace }) => {
    const [metrics, setMetrics] = useState<MetricPoint[]>([]);
    const [recentFailures, setRecentFailures] = useState<AgentRun[]>([]);

    useEffect(() => {
        const fetchData = async () => {
            if (!workspace) return;
            try {
                // Fetch metrics
                const mRes = await fetch('/api/metrics');
                if (mRes.ok) setMetrics(await mRes.json());

                // Fetch recent runs for failures
                const rRes = await fetch(`/api/runs?workspaceId=${workspace.id}&limit=20`);
                if (rRes.ok) {
                    const runs: AgentRun[] = await rRes.json();
                    setRecentFailures(runs.filter(r => r.status === 'FAILURE'));
                }
            } catch (e) {
                console.error("Dashboard fetch failed", e);
            }
        };
        fetchData();
    }, [workspace]);

  return (
    <div className="space-y-6 animate-in fade-in duration-700">
      <div className="flex items-end justify-between border-b border-gray-800 pb-4">
        <div>
            <h2 className="text-3xl text-gray-100 gothic-font tracking-wide text-glow">Command Deck</h2>
            <p className="text-gray-500 text-sm mono-font mt-1">Overview for Workspace: <span className="text-cyan-400 drop-shadow-[0_0_5px_rgba(0,243,255,0.5)]">{workspace?.name}</span></p>
        </div>
        <div className="flex gap-2">
            <span className="px-3 py-1 bg-[#0a0a0a] border border-gray-800 text-[10px] text-gray-500 font-mono flex items-center gap-2 shadow-[inset_0_0_5px_rgba(0,0,0,0.5)]">
                <span className="w-1 h-1 bg-cyan-500 rounded-full animate-pulse"></span>
                LAST SYNC: {new Date().toLocaleTimeString()}
            </span>
        </div>
      </div>

      {/* Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatusCard title="Brain Connection" status="ok" icon={Database} detail="PostgreSQL v15.4" />
        <StatusCard title="Vector Index" status="ok" icon={Server} detail="pgvector extension active" />
        <StatusCard title="Arm Controller" status="ok" icon={Activity} detail="Ready for Tasks" />
        <StatusCard title="Security" status="ok" icon={ShieldCheck} detail="No PII leaks detected" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-96">
         {/* Main Chart */}
         <div className="lg:col-span-2 bg-[#030303] border border-gray-800 p-4 rounded-sm relative shadow-2xl overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/5 blur-[60px] rounded-full pointer-events-none group-hover:bg-cyan-500/10 transition-colors duration-700"></div>
            <h3 className="text-cyan-500 text-xs uppercase tracking-[0.2em] mb-4 mono-font border-b border-gray-900 pb-2 drop-shadow-[0_0_5px_rgba(0,243,255,0.3)]">System Throughput (24h)</h3>
            <ResponsiveContainer width="100%" height="85%">
                <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                    <XAxis dataKey="time" stroke="#444" style={{fontSize: 10, fontFamily: 'Share Tech Mono'}} />
                    <YAxis stroke="#444" style={{fontSize: 10, fontFamily: 'Share Tech Mono'}} />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#000', border: '1px solid #333', boxShadow: '0 0 10px rgba(0,243,255,0.1)' }}
                        itemStyle={{ fontFamily: 'Share Tech Mono', fontSize: '12px' }}
                    />
                    <Legend wrapperStyle={{fontSize: '10px', fontFamily: 'Share Tech Mono', paddingTop: '10px'}} />
                    <Line type="monotone" dataKey="success" stroke="#00f3ff" strokeWidth={2} dot={false} activeDot={{r: 4, fill: "#fff", stroke: "#00f3ff", strokeWidth: 2}} animationDuration={2000} />
                    <Line type="monotone" dataKey="failure" stroke="#ff003c" strokeWidth={2} dot={false} animationDuration={2000} />
                </LineChart>
            </ResponsiveContainer>
         </div>

         {/* Recent Failures / Alerts */}
         <div className="bg-[#030303] border border-gray-800 p-4 rounded-sm flex flex-col shadow-2xl relative overflow-hidden">
            <div className="absolute bottom-0 left-0 w-32 h-32 bg-red-500/5 blur-[60px] rounded-full pointer-events-none"></div>
            <h3 className="text-red-500 text-xs uppercase tracking-[0.2em] mb-4 mono-font border-b border-gray-900 pb-2 flex items-center gap-2 drop-shadow-[0_0_5px_rgba(255,0,60,0.5)]">
                <AlertTriangle size={14} /> Critical Events
            </h3>
            <div className="flex-1 overflow-y-auto space-y-3 custom-scrollbar pr-2">
                {recentFailures.map(run => (
                    <div key={run.run_id} className="p-3 bg-red-950/5 border border-red-900/20 rounded-sm hover:bg-red-950/20 hover:border-red-500/30 transition-all cursor-pointer group hover:shadow-[0_0_10px_rgba(255,0,60,0.1)]">
                        <div className="flex justify-between items-start mb-1">
                            <span className="text-red-400 font-mono text-xs font-bold group-hover:text-red-300">{run.agent}</span>
                            <span className="text-gray-600 text-[10px] mono-font">{new Date(run.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <p className="text-gray-500 text-xs truncate group-hover:text-gray-300">{run.input_preview}</p>
                        <p className="text-red-600/70 text-[10px] mt-1 mono-font">{run.output_preview}</p>
                    </div>
                ))}
                {recentFailures.length === 0 && (
                    <div className="text-gray-700 text-center text-xs italic mt-10">No critical failures logged.</div>
                )}
            </div>
         </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-4">
          <div className="bg-[#030303] border border-gray-800 p-6 flex flex-col items-center justify-center hover:border-gray-700 transition-colors group relative overflow-hidden">
             <div className="absolute inset-0 bg-gradient-to-t from-gray-900/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
             <span className="text-4xl text-gray-200 gothic-font group-hover:text-white transition-colors">{workspace?.docCount || 0}</span>
             <span className="text-[10px] text-gray-600 uppercase tracking-[0.2em] mt-2 group-hover:text-cyan-500 transition-colors">Documents Ingested</span>
          </div>
          <div className="bg-[#030303] border border-gray-800 p-6 flex flex-col items-center justify-center hover:border-gray-700 transition-colors group relative overflow-hidden">
             <div className="absolute inset-0 bg-gradient-to-t from-gray-900/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
             <span className="text-4xl text-gray-200 gothic-font group-hover:text-white transition-colors">{workspace?.chunkCount.toLocaleString() || 0}</span>
             <span className="text-[10px] text-gray-600 uppercase tracking-[0.2em] mt-2 group-hover:text-cyan-500 transition-colors">Knowledge Atoms</span>
          </div>
          <div className="bg-[#030303] border border-gray-800 p-6 flex flex-col items-center justify-center hover:border-cyan-900/30 transition-colors group relative overflow-hidden box-glow">
             <div className="absolute inset-0 bg-gradient-to-t from-cyan-900/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
             <span className="text-4xl text-cyan-400 gothic-font drop-shadow-[0_0_10px_rgba(0,243,255,0.4)]">{workspace?.runCount.toLocaleString() || 0}</span>
             <span className="text-[10px] text-cyan-900 uppercase tracking-[0.2em] mt-2 group-hover:text-cyan-400 transition-colors font-bold">Total Operations</span>
          </div>
      </div>
    </div>
  );
};
