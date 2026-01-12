import React from 'react';
import { Workspace } from '../types';
import { INITIAL_GYM_RESULTS } from '../constants';
import { CheckCircle2, XCircle, Activity, BarChart2, Zap } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export const ReportsAndGym = ({ workspace }: { workspace: Workspace }) => {
  // For now using initial empty results until API endpoint exists
  const results = INITIAL_GYM_RESULTS;

  const passCount = results.filter((r) => r.status === 'PASS').length;
  const failCount = results.length - passCount;
  const passRate = results.length > 0 ? Math.round((passCount / results.length) * 100) : 0;

  return (
    <div className="flex h-full flex-col gap-8">
      <div className="flex items-end justify-between border-b border-gray-800 pb-4">
        <div>
          <h2 className="gothic-font text-3xl tracking-wide text-white">The Gym</h2>
          <p className="mono-font mt-1 text-sm text-gray-500">Automated Evaluation Suite</p>
        </div>
        <button className="border border-red-900/50 bg-red-950/20 px-6 py-2 text-xs font-bold uppercase tracking-widest text-red-500 shadow-[0_0_10px_rgba(255,0,60,0.1)] transition-all hover:bg-red-900/40 hover:text-red-400">
          Run Full Suite
        </button>
      </div>

      {/* Metrics Overview */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <div className="relative flex flex-col justify-between overflow-hidden border border-gray-800 bg-[#050505] p-6">
          <div className="absolute right-2 top-2 opacity-20">
            <Activity size={40} className="text-gray-700" />
          </div>
          <span className="text-xs uppercase tracking-widest text-gray-500">Pass Rate</span>
          <span
            className={`gothic-font mt-2 text-4xl ${passRate >= 80 ? 'text-emerald-500' : 'text-red-500'}`}
          >
            {passRate}%
          </span>
        </div>
        <div className="flex flex-col justify-between border border-gray-800 bg-[#050505] p-6">
          <span className="text-xs uppercase tracking-widest text-gray-500">Tests Run</span>
          <span className="gothic-font mt-2 text-4xl text-white">{results.length}</span>
        </div>
        <div className="flex flex-col justify-between border border-gray-800 bg-[#050505] p-6">
          <span className="text-xs uppercase tracking-widest text-gray-500">Avg Latency</span>
          <span className="gothic-font mt-2 text-4xl text-cyan-400">
            0<span className="text-lg text-gray-600">ms</span>
          </span>
        </div>
        <div className="flex flex-col justify-between border border-gray-800 bg-[#050505] p-6">
          <span className="text-xs uppercase tracking-widest text-gray-500">Tokens Consumed</span>
          <span className="gothic-font mt-2 text-4xl text-gray-300">
            0<span className="text-lg text-gray-600">k</span>
          </span>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 gap-6">
        {/* Test Results Table */}
        <div className="flex flex-1 flex-col border border-gray-800 bg-[#020202]">
          <div className="border-b border-gray-800 bg-[#080808] px-4 py-3">
            <h3 className="flex items-center gap-2 text-xs uppercase tracking-widest text-gray-400">
              <Zap size={14} className="text-yellow-500" /> Recent Results
            </h3>
          </div>
          <div className="flex-1 overflow-auto p-2">
            <table className="w-full border-collapse text-left">
              <thead className="text-[10px] uppercase tracking-widest text-gray-600">
                <tr>
                  <th className="p-3 font-normal">Status</th>
                  <th className="p-3 font-normal">Test Name</th>
                  <th className="p-3 text-right font-normal">Latency</th>
                  <th className="p-3 text-right font-normal">Tokens</th>
                </tr>
              </thead>
              <tbody className="mono-font text-sm">
                {results.length === 0 && (
                  <tr>
                    <td colSpan={4} className="p-4 text-center text-xs italic text-gray-600">
                      No evaluation results available.
                    </td>
                  </tr>
                )}
                {results.map((res) => (
                  <tr
                    key={res.test_id}
                    className="border-b border-gray-800/30 transition-colors hover:bg-gray-900/30"
                  >
                    <td className="p-3">
                      {res.status === 'PASS' ? (
                        <span className="flex w-fit items-center gap-1 rounded border border-emerald-900/50 bg-emerald-950/20 px-2 py-0.5 text-xs text-emerald-500">
                          <CheckCircle2 size={10} /> PASS
                        </span>
                      ) : (
                        <span className="flex w-fit items-center gap-1 rounded border border-red-900/50 bg-red-950/20 px-2 py-0.5 text-xs text-red-500">
                          <XCircle size={10} /> FAIL
                        </span>
                      )}
                    </td>
                    <td className="p-3 text-gray-300">{res.name}</td>
                    <td className="p-3 text-right text-gray-500">{res.latency_ms}ms</td>
                    <td className="p-3 text-right text-gray-500">{res.tokens}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Chart Area */}
        <div className="flex w-1/3 flex-col border border-gray-800 bg-[#020202]">
          <div className="border-b border-gray-800 bg-[#080808] px-4 py-3">
            <h3 className="flex items-center gap-2 text-xs uppercase tracking-widest text-gray-400">
              <BarChart2 size={14} className="text-cyan-500" /> Latency Distribution
            </h3>
          </div>
          <div className="flex-1 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={results}>
                <XAxis dataKey="test_id" hide />
                <YAxis stroke="#333" fontSize={10} tickFormatter={(val) => `${val}ms`} />
                <Tooltip
                  cursor={{ fill: '#1a1a1a' }}
                  contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }}
                  itemStyle={{ fontFamily: 'monospace', color: '#ccc' }}
                />
                <Bar dataKey="latency_ms">
                  {results.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.status === 'PASS' ? '#0e7490' : '#be123c'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
