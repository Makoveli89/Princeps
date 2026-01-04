import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { LayoutDashboard, Database, Activity, Settings, Crown } from 'lucide-react';
import { clsx } from 'clsx';

export const Layout: React.FC = () => {
  return (
    <div className="flex h-screen w-full overflow-hidden bg-gothic-900 text-gothic-text">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 border-r border-gothic-700 bg-gothic-900 flex flex-col">
        <div className="p-6 flex items-center gap-3">
          <Crown className="w-8 h-8 text-gothic-gold" />
          <span className="text-xl font-serif font-bold tracking-wide text-gothic-text">PRINCEPS</span>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-2">
          <NavItem to="/" icon={<LayoutDashboard size={20} />} label="Overview" />
          <NavItem to="/workspaces" icon={<Database size={20} />} label="Workspaces" />
          <NavItem to="/activity" icon={<Activity size={20} />} label="Activity" />
        </nav>

        <div className="p-4 border-t border-gothic-700">
          <button className="flex items-center gap-3 px-4 py-2 text-gothic-muted hover:text-gothic-text w-full transition-colors">
            <Settings size={20} />
            <span className="text-sm font-medium">Settings</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="container mx-auto p-8 max-w-7xl">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

const NavItem: React.FC<{ to: string; icon: React.ReactNode; label: string }> = ({ to, icon, label }) => {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => clsx(
        "flex items-center gap-3 px-4 py-3 rounded-md transition-all duration-200",
        isActive
          ? "bg-gothic-800 text-gothic-gold border-l-2 border-gothic-gold"
          : "text-gothic-muted hover:bg-gothic-800/50 hover:text-gothic-text"
      )}
    >
      {icon}
      <span className="font-medium">{label}</span>
    </NavLink>
  );
};
