import React from 'react';
import { Workspace } from '../types';
import { Shield, Key, Database, RefreshCw, Power, Activity, AlertTriangle } from 'lucide-react';

const HealthIndicator = ({ label, envKey }: { label: string, envKey: string }) => {
    // In Vite, we check import.meta.env
    const isPresent = !!import.meta.env[envKey];

    return (
    <div className="flex items-center justify-between p-4 bg-[#050505] border border-gray-800 hover:border-gray-700 transition-colors group">
        <span className="text-gray-400 text-sm mono-font group-hover:text-gray-300 transition-colors">{label}</span>
        <div className={`flex items-center gap-2 text-[10px] font-bold tracking-widest px-3 py-1.5 border rounded-sm ${isPresent ? 'border-emerald-900 bg-emerald-950/20 text-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.1)]' : 'border-red-900 bg-red-950/20 text-red-500 shadow-[0_0_10px_rgba(239,68,68,0.1)]'}`}>
            {isPresent ? 'DETECTED' : 'MISSING'}
            <div className={`w-1.5 h-1.5 rounded-full ${isPresent ? 'bg-emerald-500' : 'bg-red-500'} animate-pulse`}></div>
        </div>
    </div>
)};

export const SystemHealth = ({ workspace }: { workspace: Workspace }) => {
    return (
        <div className="max-w-4xl mx-auto space-y-12 animate-in fade-in slide-in-from-bottom-8 duration-700">
             <div className="border-b border-gray-800 pb-4">
                <h2 className="text-3xl text-gray-100 gothic-font tracking-wide text-glow">System Health & Vitals</h2>
                <p className="text-gray-500 text-sm mono-font mt-1">Environment Configuration and Maintenance</p>
            </div>

            {/* Environment Variables */}
            <section>
                <h3 className="text-cyan-500 text-xs uppercase tracking-[0.2em] mb-6 flex items-center gap-2 font-bold drop-shadow-[0_0_5px_rgba(0,243,255,0.4)]">
                    <Key size={14} /> Environment Secrets
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                <h3 className="text-cyan-500 text-xs uppercase tracking-[0.2em] mb-6 flex items-center gap-2 font-bold drop-shadow-[0_0_5px_rgba(0,243,255,0.4)]">
                    <Database size={14} /> Brain Configuration (Read-Only)
                </h3>
                <div className="bg-[#030303] border border-gray-800 p-8 space-y-4 font-mono text-sm text-gray-400 shadow-xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-900/5 blur-[50px] rounded-full pointer-events-none"></div>
                    <div className="flex justify-between border-b border-gray-900/50 pb-3 hover:text-gray-200 transition-colors">
                        <span className="text-gray-600">EMBEDDING_MODEL</span>
                        <span className="text-cyan-400">text-embedding-3-small</span>
                    </div>
                    <div className="flex justify-between border-b border-gray-900/50 pb-3 hover:text-gray-200 transition-colors">
                        <span className="text-gray-600">CHUNK_SIZE</span>
                        <span className="text-cyan-400">512 tokens</span>
                    </div>
                    <div className="flex justify-between border-b border-gray-900/50 pb-3 hover:text-gray-200 transition-colors">
                        <span className="text-gray-600">CHUNK_OVERLAP</span>
                        <span className="text-cyan-400">50 tokens</span>
                    </div>
                    <div className="flex justify-between hover:text-gray-200 transition-colors">
                        <span className="text-gray-600">VECTOR_DIMENSION</span>
                        <span className="text-cyan-400">1536</span>
                    </div>
                </div>
            </section>

            {/* Danger Zone */}
            <section className="pt-8 border-t border-gray-900">
                 <h3 className="text-red-600 text-xs uppercase tracking-[0.2em] mb-6 flex items-center gap-2 font-bold drop-shadow-[0_0_8px_rgba(255,0,60,0.6)]">
                    <Shield size={14} /> Maintenance Protocol (Danger Zone)
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Card 1 */}
                    <div className="border border-red-900/40 bg-[#050505] p-6 flex flex-col justify-between relative overflow-hidden group hover:border-red-600/50 transition-colors shadow-[0_0_15px_rgba(255,0,0,0.05)]">
                        {/* Hazard Stripes */}
                        <div className="absolute top-0 left-0 w-full h-1 bg-[repeating-linear-gradient(45deg,#330000,#330000_10px,#1a0000_10px,#1a0000_20px)] opacity-50"></div>

                        <div className="relative z-10">
                            <h4 className="text-red-100 font-bold mb-2 tracking-wide flex items-center gap-2">
                                <Activity size={16} className="text-red-500" /> Database Smoke Test
                            </h4>
                            <p className="text-red-500/60 text-xs mb-6 leading-relaxed">Run a non-destructive read/write cycle to verify persistence layer integrity.</p>
                        </div>
                        <button className="relative z-10 bg-gray-900 hover:bg-red-950/30 text-gray-400 hover:text-red-400 border border-gray-700 hover:border-red-800 py-3 px-4 text-[10px] font-bold uppercase tracking-[0.2em] flex items-center justify-center gap-2 transition-all duration-300">
                            Initiate Test
                        </button>
                    </div>

                    {/* Card 2 */}
                    <div className="border border-red-900/40 bg-[#050505] p-6 flex flex-col justify-between relative overflow-hidden group hover:border-red-600/50 transition-colors shadow-[0_0_15px_rgba(255,0,0,0.05)]">
                         {/* Hazard Stripes */}
                         <div className="absolute top-0 left-0 w-full h-1 bg-[repeating-linear-gradient(45deg,#330000,#330000_10px,#1a0000_10px,#1a0000_20px)] opacity-50"></div>

                        <div className="relative z-10">
                            <h4 className="text-red-100 font-bold mb-2 tracking-wide flex items-center gap-2">
                                <RefreshCw size={16} className="text-red-500" /> Re-Index Vectors
                            </h4>
                            <p className="text-red-500/60 text-xs mb-6 leading-relaxed">Force a complete re-indexing of all chunks. <span className="text-red-400 font-bold">Search will be degraded.</span></p>
                        </div>
                        <button className="relative z-10 bg-red-950/10 hover:bg-red-900/30 text-red-500 border border-red-900/40 hover:border-red-500/60 py-3 px-4 text-[10px] font-bold uppercase tracking-[0.2em] flex items-center justify-center gap-2 transition-all duration-300 shadow-[0_0_10px_rgba(255,0,0,0.1)] hover:shadow-[0_0_20px_rgba(255,0,0,0.2)]">
                            <AlertTriangle size={12} /> Rebuild Index
                        </button>
                    </div>
                </div>
            </section>
        </div>
    );
};