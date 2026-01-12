import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { PrincepsLogo } from './PrincepsLogo';
import {
  LayoutDashboard,
  Play,
  Database,
  FileText,
  Search,
  Activity,
  Settings,
  HardDrive,
  MessageSquare,
} from 'lucide-react';
import { Workspace } from '../types';
import { COLORS } from '../constants';
import { Toaster } from './ui/toaster';

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
        `group relative flex items-center gap-3 overflow-hidden border-l-2 px-4 py-3 transition-all duration-300 ${
          isActive
            ? 'border-red-600 bg-gradient-to-r from-red-950/40 to-transparent text-gray-100 shadow-[inset_10px_0_20px_-10px_rgba(255,0,60,0.3)]'
            : 'border-transparent text-gray-500 hover:border-gray-700 hover:bg-white/5 hover:text-gray-200'
        }`
      }
    >
      <div
        className={`absolute inset-0 translate-x-[-100%] bg-red-600/5 transition-transform duration-300 ease-out group-hover:translate-x-0`}
      ></div>
      <Icon
        size={16}
        className={`relative z-10 transition-all duration-300 ${window.location.hash.includes(to) ? 'text-red-500 drop-shadow-[0_0_8px_rgba(255,0,60,0.8)]' : 'group-hover:text-cyan-400 group-hover:drop-shadow-[0_0_5px_rgba(0,243,255,0.6)]'}`}
      />
      <span className="mono-font relative z-10 text-xs font-bold uppercase tracking-[0.15em]">
        {label}
      </span>
    </NavLink>
  );
};

export const Layout: React.FC<LayoutProps> = ({
  children,
  workspaces,
  activeWorkspaceId,
  onWorkspaceChange,
}) => {
  const activeWorkspace = workspaces.find((w) => w.id === activeWorkspaceId) || workspaces[0];

  return (
    <div className="relative flex h-screen w-full overflow-hidden bg-[#020202]">
      {/* CRT Effects */}
      <div className="scanlines"></div>
      <div className="crt-flicker"></div>

      {/* Sidebar - The Crypt Wall */}
      <aside className="relative z-20 flex w-64 flex-shrink-0 flex-col border-r border-gray-900 bg-[#030303] shadow-[5px_0_30px_rgba(0,0,0,0.5)]">
        <div className="group relative flex flex-col items-center overflow-hidden border-b border-gray-900 bg-[#020202] p-6">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(255,0,60,0.15),transparent_70%)] opacity-50 transition-opacity duration-1000 group-hover:opacity-80"></div>
          <PrincepsLogo className="relative z-10 mb-4 h-16 w-16 drop-shadow-[0_0_20px_rgba(0,243,255,0.2)] transition-all duration-500 group-hover:drop-shadow-[0_0_25px_rgba(0,243,255,0.4)]" />
          <h1 className="gothic-font text-glow relative z-10 text-2xl font-bold tracking-[0.2em] text-gray-100">
            PRINCEPS
          </h1>
          <span className="relative z-10 mt-1 text-[9px] font-bold uppercase tracking-[0.3em] text-red-600 drop-shadow-[0_0_2px_rgba(255,0,60,0.8)]">
            Console v0.9.3
          </span>
        </div>

        {/* Global Workspace Selector */}
        <div className="border-b border-gray-900 bg-[#030303] px-4 py-6">
          <label className="mb-2 block flex items-center gap-2 pl-1 text-[9px] uppercase tracking-widest text-gray-500">
            <div className="h-1 w-1 rounded-full bg-cyan-500 shadow-[0_0_5px_#00f3ff]"></div> Active
            Link
          </label>
          <div className="group relative">
            <select
              value={activeWorkspaceId}
              onChange={(e) => onWorkspaceChange(e.target.value)}
              className="mono-font w-full cursor-pointer appearance-none border border-gray-800 bg-[#080808] px-3 py-2 text-xs font-bold text-cyan-500 shadow-[inset_0_0_10px_rgba(0,0,0,0.5)] transition-colors hover:border-gray-700 focus:border-cyan-700/50 focus:outline-none"
            >
              {workspaces.map((ws) => (
                <option key={ws.id} value={ws.id}>
                  {ws.name}
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute right-3 top-2.5 text-cyan-600">
              <svg width="10" height="6" viewBox="0 0 10 6" fill="none">
                <path d="M0 0L5 5L10 0" fill="currentColor" />
              </svg>
            </div>
          </div>
          <div className="mt-3 flex items-center justify-between px-1">
            <span className="font-mono text-[9px] text-gray-600">
              {activeWorkspace?.id.split('-')[2]}
            </span>
            <span className="flex items-center gap-1.5 rounded-sm border border-emerald-900/50 bg-emerald-950/20 px-2 py-0.5 text-[9px] text-emerald-500 shadow-[0_0_5px_rgba(16,185,129,0.2)]">
              <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500"></div> ONLINE
            </span>
          </div>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto py-4">
          <NavItem to="/" icon={LayoutDashboard} label="Dashboard" />
          <NavItem to="/run" icon={Play} label="Run Task" />
          <NavItem to="/chat" icon={MessageSquare} label="Neural Chat" />
          <NavItem to="/ingest" icon={HardDrive} label="Ingest Data" />
          <NavItem to="/search" icon={Search} label="Knowledge" />
          <NavItem to="/logs" icon={FileText} label="Runs & Logs" />
          <NavItem to="/gym" icon={Activity} label="Reports & Gym" />
          <NavItem to="/workspaces" icon={Database} label="Workspaces" />
          <div className="mt-8 px-4 pt-4">
            <div className="mb-4 h-px bg-gradient-to-r from-transparent via-gray-800 to-transparent"></div>
            <NavLink
              to="/settings"
              className={({ isActive }) =>
                `group flex items-center gap-3 px-2 py-2 text-gray-600 transition-colors hover:text-red-500 ${isActive ? 'text-red-500' : ''}`
              }
            >
              <Settings
                size={16}
                className="text-gray-500 transition-transform duration-700 group-hover:rotate-90 group-hover:text-red-500"
              />
              <span className="mono-font text-xs font-bold uppercase tracking-[0.15em] group-hover:drop-shadow-[0_0_3px_red]">
                System Health
              </span>
            </NavLink>
          </div>
        </nav>
      </aside>

      {/* Main Content - The Neon Skyline */}
      <main className="relative flex flex-1 flex-col overflow-hidden bg-[#020202]">
        {/* Decorative Grid Background */}
        <div
          className="pointer-events-none absolute inset-0 z-0 opacity-[0.04]"
          style={{
            backgroundImage:
              'linear-gradient(#222 1px, transparent 1px), linear-gradient(90deg, #222 1px, transparent 1px)',
            backgroundSize: '40px 40px',
          }}
        ></div>
        <div className="pointer-events-none absolute inset-0 z-0 bg-gradient-to-b from-transparent via-transparent to-[#050505]/80"></div>

        {/* Top Header */}
        <header className="z-10 flex h-14 items-center justify-between border-b border-gray-900 bg-[#030303]/80 px-8 shadow-[0_4px_20px_rgba(0,0,0,0.4)] backdrop-blur-md">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3 rounded-full border border-gray-800/60 bg-[#0a0a0a] px-3 py-1 shadow-[inset_0_0_10px_rgba(0,0,0,0.5)]">
              <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-600 shadow-[0_0_8px_#ff003c]"></div>
              <span className="mono-font text-[10px] font-semibold uppercase tracking-widest text-gray-400">
                System Nominal
              </span>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <span className="mono-font text-[10px] tracking-wider text-gray-600">
              ENV: <span className="text-gray-300">PRODUCTION</span>
            </span>
            <div className="mono-font border border-cyan-900/30 bg-cyan-950/10 px-2 py-1 text-[10px] text-cyan-600/70 shadow-[0_0_10px_rgba(0,243,255,0.05)]">
              PID: 8821 // ROOT
            </div>
          </div>
        </header>

        {/* Page Content */}
        <div className="relative z-10 flex-1 overflow-auto scroll-smooth p-8">{children}</div>
        <Toaster />
      </main>
    </div>
  );
};
