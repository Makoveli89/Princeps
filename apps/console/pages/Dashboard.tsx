import React, { useState, useEffect } from 'react';
import useSWR from 'swr';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Activity, Database, Server, AlertTriangle, ShieldCheck } from 'lucide-react';
import { Workspace, MetricPoint, AgentRun } from '../types';
import { fetcher } from '../lib/fetcher';

const StatusCard = ({
  title,
  status,
  icon: Icon,
  detail,
}: {
  title: string;
  status: 'ok' | 'err' | 'warn';
  icon: any;
  detail: string;
}) => {
  const color =
    status === 'ok' ? 'text-emerald-400' : status === 'err' ? 'text-red-500' : 'text-amber-400';
  const border =
    status === 'ok'
      ? 'border-emerald-900/30 hover:border-emerald-500/50'
      : status === 'err'
        ? 'border-red-900/30 hover:border-red-500/50'
        : 'border-amber-900/30 hover:border-amber-500/50';
  const bg =
    status === 'ok' ? 'bg-emerald-950/10' : status === 'err' ? 'bg-red-950/10' : 'bg-amber-950/10';
  const glow =
    status === 'ok'
      ? 'shadow-[0_0_10px_rgba(16,185,129,0.1)]'
      : status === 'err'
        ? 'shadow-[0_0_10px_rgba(239,68,68,0.1)]'
        : 'shadow-[0_0_10px_rgba(245,158,11,0.1)]';

  return (
    <div
      className={`border p-4 ${border} ${bg} ${glow} group relative flex items-start gap-4 overflow-hidden rounded-sm transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_0_20px_rgba(0,0,0,0.5)]`}
    >
      <div
        className={`absolute -right-6 -top-6 h-16 w-16 rounded-full ${status === 'ok' ? 'bg-emerald-500' : 'bg-red-500'} opacity-10 blur-[40px] transition-opacity duration-500 group-hover:opacity-20`}
      ></div>
      <div className={`rounded-sm border border-gray-800 bg-[#050505] p-2.5 ${color} shadow-inner`}>
        <Icon size={20} className="drop-shadow-[0_0_5px_currentColor]" />
      </div>
      <div>
        <h3 className="font-mono text-[10px] uppercase tracking-[0.2em] text-gray-500 transition-colors group-hover:text-gray-300">
          {title}
        </h3>
        <div
          className={`text-lg font-bold ${color} gothic-font mt-1 ${status !== 'ok' ? 'animate-pulse' : ''} drop-shadow-[0_0_3px_currentColor]`}
        >
          {status === 'ok' ? 'ONLINE' : status === 'err' ? 'OFFLINE' : 'DEGRADED'}
        </div>
        <p className="mt-1 font-mono text-[10px] text-gray-600 transition-colors group-hover:text-gray-400">
          {detail}
        </p>
      </div>
    </div>
  );
};

export const Dashboard = ({ workspace }: { workspace: Workspace }) => {
  const { data: metricsData } = useSWR<MetricPoint[]>('/api/metrics', fetcher);
  const { data: runsData } = useSWR<AgentRun[]>(
    workspace ? `/api/runs?workspaceId=${workspace.id}&limit=20` : null,
    fetcher,
  );

  const metrics = metricsData || [];
  const recentFailures = (runsData || []).filter((r) => r.status === 'FAILURE');

  return (
    <div className="animate-in fade-in space-y-6 duration-700">
      <div className="flex items-end justify-between border-b border-gray-800 pb-4">
        <div>
          <h2 className="gothic-font text-glow text-3xl tracking-wide text-gray-100">
            Command Deck
          </h2>
          <p className="mono-font mt-1 text-sm text-gray-500">
            Overview for Workspace:{' '}
            <span className="text-cyan-400 drop-shadow-[0_0_5px_rgba(0,243,255,0.5)]">
              {workspace?.name}
            </span>
          </p>
        </div>
        <div className="flex gap-2">
          <span className="flex items-center gap-2 border border-gray-800 bg-[#0a0a0a] px-3 py-1 font-mono text-[10px] text-gray-500 shadow-[inset_0_0_5px_rgba(0,0,0,0.5)]">
            <span className="h-1 w-1 animate-pulse rounded-full bg-cyan-500"></span>
            LAST SYNC: {new Date().toLocaleTimeString()}
          </span>
        </div>
      </div>

      {/* Status Grid */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <StatusCard
          title="Brain Connection"
          status="ok"
          icon={Database}
          detail="PostgreSQL v15.4"
        />
        <StatusCard
          title="Vector Index"
          status="ok"
          icon={Server}
          detail="pgvector extension active"
        />
        <StatusCard title="Arm Controller" status="ok" icon={Activity} detail="Ready for Tasks" />
        <StatusCard
          title="Security"
          status="ok"
          icon={ShieldCheck}
          detail="No PII leaks detected"
        />
      </div>

      <div className="grid h-96 grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Main Chart */}
        <div className="group relative overflow-hidden rounded-sm border border-gray-800 bg-[#030303] p-4 shadow-2xl lg:col-span-2">
          <div className="pointer-events-none absolute right-0 top-0 h-32 w-32 rounded-full bg-cyan-500/5 blur-[60px] transition-colors duration-700 group-hover:bg-cyan-500/10"></div>
          <h3 className="mono-font mb-4 border-b border-gray-900 pb-2 text-xs uppercase tracking-[0.2em] text-cyan-500 drop-shadow-[0_0_5px_rgba(0,243,255,0.3)]">
            System Throughput (24h)
          </h3>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
              <XAxis
                dataKey="time"
                stroke="#444"
                style={{ fontSize: 10, fontFamily: 'Share Tech Mono' }}
              />
              <YAxis stroke="#444" style={{ fontSize: 10, fontFamily: 'Share Tech Mono' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#000',
                  border: '1px solid #333',
                  boxShadow: '0 0 10px rgba(0,243,255,0.1)',
                }}
                itemStyle={{ fontFamily: 'Share Tech Mono', fontSize: '12px' }}
              />
              <Legend
                wrapperStyle={{
                  fontSize: '10px',
                  fontFamily: 'Share Tech Mono',
                  paddingTop: '10px',
                }}
              />
              <Line
                type="monotone"
                dataKey="success"
                stroke="#00f3ff"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#fff', stroke: '#00f3ff', strokeWidth: 2 }}
                animationDuration={2000}
              />
              <Line
                type="monotone"
                dataKey="failure"
                stroke="#ff003c"
                strokeWidth={2}
                dot={false}
                animationDuration={2000}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Failures / Alerts */}
        <div className="relative flex flex-col overflow-hidden rounded-sm border border-gray-800 bg-[#030303] p-4 shadow-2xl">
          <div className="pointer-events-none absolute bottom-0 left-0 h-32 w-32 rounded-full bg-red-500/5 blur-[60px]"></div>
          <h3 className="mono-font mb-4 flex items-center gap-2 border-b border-gray-900 pb-2 text-xs uppercase tracking-[0.2em] text-red-500 drop-shadow-[0_0_5px_rgba(255,0,60,0.5)]">
            <AlertTriangle size={14} /> Critical Events
          </h3>
          <div className="custom-scrollbar flex-1 space-y-3 overflow-y-auto pr-2">
            {recentFailures.map((run) => (
              <div
                key={run.run_id}
                className="group cursor-pointer rounded-sm border border-red-900/20 bg-red-950/5 p-3 transition-all hover:border-red-500/30 hover:bg-red-950/20 hover:shadow-[0_0_10px_rgba(255,0,60,0.1)]"
              >
                <div className="mb-1 flex items-start justify-between">
                  <span className="font-mono text-xs font-bold text-red-400 group-hover:text-red-300">
                    {run.agent}
                  </span>
                  <span className="mono-font text-[10px] text-gray-600">
                    {new Date(run.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="truncate text-xs text-gray-500 group-hover:text-gray-300">
                  {run.input_preview}
                </p>
                <p className="mono-font mt-1 text-[10px] text-red-600/70">{run.output_preview}</p>
              </div>
            ))}
            {recentFailures.length === 0 && (
              <div className="mt-10 text-center text-xs italic text-gray-700">
                No critical failures logged.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-4">
        <div className="group relative flex flex-col items-center justify-center overflow-hidden border border-gray-800 bg-[#030303] p-6 transition-colors hover:border-gray-700">
          <div className="absolute inset-0 bg-gradient-to-t from-gray-900/20 to-transparent opacity-0 transition-opacity group-hover:opacity-100"></div>
          <span className="gothic-font text-4xl text-gray-200 transition-colors group-hover:text-white">
            {workspace?.docCount || 0}
          </span>
          <span className="mt-2 text-[10px] uppercase tracking-[0.2em] text-gray-600 transition-colors group-hover:text-cyan-500">
            Documents Ingested
          </span>
        </div>
        <div className="group relative flex flex-col items-center justify-center overflow-hidden border border-gray-800 bg-[#030303] p-6 transition-colors hover:border-gray-700">
          <div className="absolute inset-0 bg-gradient-to-t from-gray-900/20 to-transparent opacity-0 transition-opacity group-hover:opacity-100"></div>
          <span className="gothic-font text-4xl text-gray-200 transition-colors group-hover:text-white">
            {workspace?.chunkCount.toLocaleString() || 0}
          </span>
          <span className="mt-2 text-[10px] uppercase tracking-[0.2em] text-gray-600 transition-colors group-hover:text-cyan-500">
            Knowledge Atoms
          </span>
        </div>
        <div className="box-glow group relative flex flex-col items-center justify-center overflow-hidden border border-gray-800 bg-[#030303] p-6 transition-colors hover:border-cyan-900/30">
          <div className="absolute inset-0 bg-gradient-to-t from-cyan-900/10 to-transparent opacity-0 transition-opacity group-hover:opacity-100"></div>
          <span className="gothic-font text-4xl text-cyan-400 drop-shadow-[0_0_10px_rgba(0,243,255,0.4)]">
            {workspace?.runCount.toLocaleString() || 0}
          </span>
          <span className="mt-2 text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-900 transition-colors group-hover:text-cyan-400">
            Total Operations
          </span>
        </div>
      </div>
    </div>
  );
};
