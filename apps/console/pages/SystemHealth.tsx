import React from 'react';
import { Workspace } from '../types';
import { Shield, Key, Database, RefreshCw, Power, Activity, AlertTriangle } from 'lucide-react';

const HealthIndicator = ({ label, envKey }: { label: string; envKey: string }) => {
  // In Vite, we check import.meta.env
  const isPresent = !!import.meta.env[envKey];

  return (
    <div className="group flex items-center justify-between border border-gray-800 bg-[#050505] p-4 transition-colors hover:border-gray-700">
      <span className="mono-font text-sm text-gray-400 transition-colors group-hover:text-gray-300">
        {label}
      </span>
      <div
        className={`flex items-center gap-2 rounded-sm border px-3 py-1.5 text-[10px] font-bold tracking-widest ${isPresent ? 'border-emerald-900 bg-emerald-950/20 text-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.1)]' : 'border-red-900 bg-red-950/20 text-red-500 shadow-[0_0_10px_rgba(239,68,68,0.1)]'}`}
      >
        {isPresent ? 'DETECTED' : 'MISSING'}
        <div
          className={`h-1.5 w-1.5 rounded-full ${isPresent ? 'bg-emerald-500' : 'bg-red-500'} animate-pulse`}
        ></div>
      </div>
    </div>
  );
};

export const SystemHealth = ({ workspace }: { workspace: Workspace }) => {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-8 mx-auto max-w-4xl space-y-12 duration-700">
      <div className="border-b border-gray-800 pb-4">
        <h2 className="gothic-font text-glow text-3xl tracking-wide text-gray-100">
          System Health & Vitals
        </h2>
        <p className="mono-font mt-1 text-sm text-gray-500">
          Environment Configuration and Maintenance
        </p>
      </div>

      {/* Environment Variables */}
      <section>
        <h3 className="mb-6 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-cyan-500 drop-shadow-[0_0_5px_rgba(0,243,255,0.4)]">
          <Key size={14} /> Environment Secrets
        </h3>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Note: DATABASE_URL is backend side, cannot check from client easily without API endpoint.
                        Here we check client-side env vars only for demonstration or assume passed via config.
                        For now, leaving DATABASE_URL hardcoded or checking a proxy var if existing.
                    */}
          <HealthIndicator label="VITE_GEMINI_API_KEY" envKey="VITE_GEMINI_API_KEY" />
          <HealthIndicator label="VITE_GOOGLE_API_KEY" envKey="VITE_GOOGLE_API_KEY" />
        </div>
      </section>

      {/* Brain Config */}
      <section>
        <h3 className="mb-6 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-cyan-500 drop-shadow-[0_0_5px_rgba(0,243,255,0.4)]">
          <Database size={14} /> Brain Configuration (Read-Only)
        </h3>
        <div className="relative space-y-4 overflow-hidden border border-gray-800 bg-[#030303] p-8 font-mono text-sm text-gray-400 shadow-xl">
          <div className="pointer-events-none absolute right-0 top-0 h-32 w-32 rounded-full bg-cyan-900/5 blur-[50px]"></div>
          <div className="flex justify-between border-b border-gray-900/50 pb-3 transition-colors hover:text-gray-200">
            <span className="text-gray-600">EMBEDDING_MODEL</span>
            <span className="text-cyan-400">text-embedding-3-small</span>
          </div>
          <div className="flex justify-between border-b border-gray-900/50 pb-3 transition-colors hover:text-gray-200">
            <span className="text-gray-600">CHUNK_SIZE</span>
            <span className="text-cyan-400">512 tokens</span>
          </div>
          <div className="flex justify-between border-b border-gray-900/50 pb-3 transition-colors hover:text-gray-200">
            <span className="text-gray-600">CHUNK_OVERLAP</span>
            <span className="text-cyan-400">50 tokens</span>
          </div>
          <div className="flex justify-between transition-colors hover:text-gray-200">
            <span className="text-gray-600">VECTOR_DIMENSION</span>
            <span className="text-cyan-400">1536</span>
          </div>
        </div>
      </section>

      {/* Danger Zone */}
      <section className="border-t border-gray-900 pt-8">
        <h3 className="mb-6 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-red-600 drop-shadow-[0_0_8px_rgba(255,0,60,0.6)]">
          <Shield size={14} /> Maintenance Protocol (Danger Zone)
        </h3>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
          {/* Card 1 */}
          <div className="group relative flex flex-col justify-between overflow-hidden border border-red-900/40 bg-[#050505] p-6 shadow-[0_0_15px_rgba(255,0,0,0.05)] transition-colors hover:border-red-600/50">
            {/* Hazard Stripes */}
            <div className="absolute left-0 top-0 h-1 w-full bg-[repeating-linear-gradient(45deg,#330000,#330000_10px,#1a0000_10px,#1a0000_20px)] opacity-50"></div>

            <div className="relative z-10">
              <h4 className="mb-2 flex items-center gap-2 font-bold tracking-wide text-red-100">
                <Activity size={16} className="text-red-500" /> Database Smoke Test
              </h4>
              <p className="mb-6 text-xs leading-relaxed text-red-500/60">
                Run a non-destructive read/write cycle to verify persistence layer integrity.
              </p>
            </div>
            <button className="relative z-10 flex items-center justify-center gap-2 border border-gray-700 bg-gray-900 px-4 py-3 text-[10px] font-bold uppercase tracking-[0.2em] text-gray-400 transition-all duration-300 hover:border-red-800 hover:bg-red-950/30 hover:text-red-400">
              Initiate Test
            </button>
          </div>

          {/* Card 2 */}
          <div className="group relative flex flex-col justify-between overflow-hidden border border-red-900/40 bg-[#050505] p-6 shadow-[0_0_15px_rgba(255,0,0,0.05)] transition-colors hover:border-red-600/50">
            {/* Hazard Stripes */}
            <div className="absolute left-0 top-0 h-1 w-full bg-[repeating-linear-gradient(45deg,#330000,#330000_10px,#1a0000_10px,#1a0000_20px)] opacity-50"></div>

            <div className="relative z-10">
              <h4 className="mb-2 flex items-center gap-2 font-bold tracking-wide text-red-100">
                <RefreshCw size={16} className="text-red-500" /> Re-Index Vectors
              </h4>
              <p className="mb-6 text-xs leading-relaxed text-red-500/60">
                Force a complete re-indexing of all chunks.{' '}
                <span className="font-bold text-red-400">Search will be degraded.</span>
              </p>
            </div>
            <button className="relative z-10 flex items-center justify-center gap-2 border border-red-900/40 bg-red-950/10 px-4 py-3 text-[10px] font-bold uppercase tracking-[0.2em] text-red-500 shadow-[0_0_10px_rgba(255,0,0,0.1)] transition-all duration-300 hover:border-red-500/60 hover:bg-red-900/30 hover:shadow-[0_0_20px_rgba(255,0,0,0.2)]">
              <AlertTriangle size={12} /> Rebuild Index
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
