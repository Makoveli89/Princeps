import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { PrincepsLogo } from './PrincepsLogo';
import { LayoutDashboard, Play, Database, FileText, Search, Activity, Settings, HardDrive, MessageSquare } from 'lucide-react';
import { Workspace } from '../types';
import { COLORS } from '../constants';
import { Toaster } from './ui/toaster';
import { CommandMenu } from './ui/command-menu';

interface LayoutProps {
  children: React.ReactNode;
  workspaces: Workspace[];
  activeWorkspaceId: string;
  onWorkspaceChange: (id: string) => void;
}

const NavItem = ({ to, icon: Icon, label }: { to: string; icon: any; label: string }) => {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-3 px-4 py-3 border-l-2 transition-all duration-300 group relative overflow-hidden ${
          isActive
            ? 'border-red-600 bg-gradient-to-r from-red-950/40 to-transparent text-gray-100 shadow-[inset_10px_0_20px_-10px_rgba(255,0,60,0.3)]'
            : 'border-transparent text-gray-500 hover:text-gray-200 hover:bg-white/5 hover:border-gray-700'
        }`
      }
    >
      <div className={`absolute inset-0 bg-red-600/5 translate-x-[-100%] group-hover:translate-x-0 transition-transform duration-300 ease-out`}></div>
      <Icon size={16} className={`relative z-10 transition-all duration-300 ${window.location.hash.includes(to) ? 'text-red-500 drop-shadow-[0_0_8px_rgba(255,0,60,0.8)]' : 'group-hover:text-cyan-400 group-hover:drop-shadow-[0_0_5px_rgba(0,243,255,0.6)]'}`} />
      <span className="mono-font tracking-[0.15em] text-xs uppercase font-bold relative z-10">{label}</span>
    </NavLink>
  );
};

export const Layout: React.FC<LayoutProps> = ({ children, workspaces, activeWorkspaceId, onWorkspaceChange }) => {
  const activeWorkspace = workspaces.find(w => w.id === activeWorkspaceId) || workspaces[0];

  return (
    <div className="flex h-screen w-full bg-[#020202] overflow-hidden relative">
      {/* CRT Effects */}
      <div className="scanlines"></div>
      <div className="crt-flicker"></div>

      {/* Sidebar - The Crypt Wall */}
      <aside className="w-64 flex-shrink-0 border-r border-gray-900 bg-[#030303] flex flex-col relative z-20 shadow-[5px_0_30px_rgba(0,0,0,0.5)]">
        <div className="p-6 flex flex-col items-center border-b border-gray-900 bg-[#020202] relative overflow-hidden group">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(255,0,60,0.15),transparent_70%)] opacity-50 group-hover:opacity-80 transition-opacity duration-1000"></div>
          <PrincepsLogo className="w-16 h-16 mb-4 drop-shadow-[0_0_20px_rgba(0,243,255,0.2)] relative z-10 group-hover:drop-shadow-[0_0_25px_rgba(0,243,255,0.4)] transition-all duration-500" />
          <h1 className="text-2xl font-bold tracking-[0.2em] text-gray-100 gothic-font relative z-10 text-glow">PRINCEPS</h1>
          <span className="text-[9px] text-red-600 tracking-[0.3em] uppercase mt-1 relative z-10 font-bold drop-shadow-[0_0_2px_rgba(255,0,60,0.8)]">Console v0.9.3</span>
        </div>

        {/* Global Workspace Selector */}
        <div className="px-4 py-6 border-b border-gray-900 bg-[#030303]">
            <label htmlFor="workspace-select" className="text-[9px] uppercase tracking-widest text-gray-500 mb-2 block pl-1 flex items-center gap-2">
                <div className="w-1 h-1 bg-cyan-500 rounded-full shadow-[0_0_5px_#00f3ff]"></div> Active Link
            </label>
            <div className="relative group">
                <select
                    id="workspace-select"
                    value={activeWorkspaceId}
                    onChange={(e) => onWorkspaceChange(e.target.value)}
                    className="w-full bg-[#080808] border border-gray-800 text-cyan-500 font-bold text-xs mono-font py-2 px-3 appearance-none focus:outline-none focus:border-cyan-700/50 transition-colors cursor-pointer hover:border-gray-700 shadow-[inset_0_0_10px_rgba(0,0,0,0.5)]"
                >
                    {workspaces.map(ws => (
                        <option key={ws.id} value={ws.id}>
                            {ws.name}
                        </option>
                    ))}
                </select>
                <div className="absolute right-3 top-2.5 pointer-events-none text-cyan-600">
                    <svg width="10" height="6" viewBox="0 0 10 6" fill="none"><path d="M0 0L5 5L10 0" fill="currentColor"/></svg>
                </div>
            </div>
            <div className="mt-3 flex justify-between items-center px-1">
                <span className="text-[9px] text-gray-600 font-mono">{activeWorkspace?.id.split('-')[2]}</span>
                <span className="text-[9px] text-emerald-500 bg-emerald-950/20 px-2 py-0.5 border border-emerald-900/50 rounded-sm flex items-center gap-1.5 shadow-[0_0_5px_rgba(16,185,129,0.2)]">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div> ONLINE
                </span>
            </div>
        </div>

        <nav className="flex-1 overflow-y-auto py-4 space-y-1">
          <NavItem to="/" icon={LayoutDashboard} label="Dashboard" />
          <NavItem to="/run" icon={Play} label="Run Task" />
          <NavItem to="/chat" icon={MessageSquare} label="Neural Chat" />
          <NavItem to="/ingest" icon={HardDrive} label="Ingest Data" />
          <NavItem to="/search" icon={Search} label="Knowledge" />
          <NavItem to="/logs" icon={FileText} label="Runs & Logs" />
          <NavItem to="/gym" icon={Activity} label="Reports & Gym" />
          <NavItem to="/workspaces" icon={Database} label="Workspaces" />
          <div className="mt-8 pt-4 px-4">
             <div className="h-px bg-gradient-to-r from-transparent via-gray-800 to-transparent mb-4"></div>
             <NavLink
                to="/settings"
                className={({ isActive }) => `flex items-center gap-3 px-2 py-2 text-gray-600 hover:text-red-500 transition-colors group ${isActive ? 'text-red-500' : ''}`}
             >
                <Settings size={16} className="group-hover:rotate-90 transition-transform duration-700 text-gray-500 group-hover:text-red-500" />
                <span className="mono-font tracking-[0.15em] text-xs uppercase font-bold group-hover:drop-shadow-[0_0_3px_red]">System Health</span>
             </NavLink>
          </div>
        </nav>
      </aside>

      {/* Main Content - The Neon Skyline */}
      <main className="flex-1 flex flex-col relative overflow-hidden bg-[#020202]">
        {/* Decorative Grid Background */}
        <div className="absolute inset-0 z-0 pointer-events-none opacity-[0.04]"
             style={{ backgroundImage: 'linear-gradient(#222 1px, transparent 1px), linear-gradient(90deg, #222 1px, transparent 1px)', backgroundSize: '40px 40px' }}>
        </div>
        <div className="absolute inset-0 z-0 pointer-events-none bg-gradient-to-b from-transparent via-transparent to-[#050505]/80"></div>

        {/* Top Header */}
        <header className="h-14 border-b border-gray-900 flex items-center justify-between px-8 bg-[#030303]/80 backdrop-blur-md z-10 shadow-[0_4px_20px_rgba(0,0,0,0.4)]">
           <div className="flex items-center gap-4">
                <div className="flex items-center gap-3 px-3 py-1 bg-[#0a0a0a] border border-gray-800/60 rounded-full shadow-[inset_0_0_10px_rgba(0,0,0,0.5)]">
                    <div className="h-1.5 w-1.5 bg-red-600 rounded-full animate-pulse shadow-[0_0_8px_#ff003c]"></div>
                    <span className="text-gray-400 mono-font text-[10px] uppercase tracking-widest font-semibold">System Nominal</span>
                </div>
           </div>
           <div className="flex items-center gap-6">
               <span className="text-gray-600 mono-font text-[10px] tracking-wider">ENV: <span className="text-gray-300">PRODUCTION</span></span>
               <div className="text-cyan-600/70 mono-font text-[10px] border border-cyan-900/30 px-2 py-1 bg-cyan-950/10 shadow-[0_0_10px_rgba(0,243,255,0.05)]">
                   PID: 8821 // ROOT
               </div>
           </div>
        </header>

        {/* Page Content */}
        <div className="flex-1 overflow-auto p-8 relative z-10 scroll-smooth">
            {children}
        </div>
        <Toaster />
        <CommandMenu />
      </main>
    </div>
  );
};
