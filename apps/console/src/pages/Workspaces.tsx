import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { Workspace } from '../types';
import { Plus, Server, Activity, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';

export const Workspaces: React.FC = () => {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchWorkspaces = async () => {
    try {
      const response = await fetch('/api/workspaces');
      if (response.ok) {
        const data = await response.json();
        setWorkspaces(data);
      }
    } catch (error) {
      console.error('Failed to fetch workspaces', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWorkspaces();
  }, []);

  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-serif text-gothic-text mb-2">Workspaces</h1>
          <p className="text-gothic-muted">Manage your isolated knowledge environments.</p>
        </div>
        <Button>
          <Plus className="w-4 h-4 mr-2" />
          New Workspace
        </Button>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin w-8 h-8 border-2 border-gothic-gold border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-gothic-muted">Synchronizing with core...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {workspaces.map((ws) => (
            <WorkspaceCard key={ws.id} workspace={ws} />
          ))}
        </div>
      )}
    </div>
  );
};

const WorkspaceCard: React.FC<{ workspace: Workspace }> = ({ workspace }) => {
  const statusColors = {
    active: 'text-green-400',
    archived: 'text-gothic-muted',
    error: 'text-red-400',
  };

  const statusIcon = {
    active: <Activity className="w-4 h-4 text-green-400" />,
    archived: <Server className="w-4 h-4 text-gothic-muted" />,
    error: <AlertCircle className="w-4 h-4 text-red-400" />,
  };

  return (
    <Card className="hover:border-gothic-gold/50 transition-colors group cursor-pointer relative overflow-hidden">
      <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
        <Server className="w-24 h-24 text-gothic-gold" />
      </div>

      <div className="relative z-10">
        <div className="flex justify-between items-start mb-4">
          <div className={`flex items-center gap-2 px-2 py-1 rounded bg-gothic-900/50 text-xs font-mono uppercase tracking-wider border border-gothic-700 ${statusColors[workspace.status]}`}>
            {statusIcon[workspace.status]}
            {workspace.status}
          </div>
          <span className="text-xs text-gothic-muted font-mono">{workspace.id}</span>
        </div>

        <h3 className="text-xl font-serif font-bold text-gothic-text mb-2 group-hover:text-gothic-gold transition-colors">
          {workspace.name}
        </h3>
        <p className="text-gothic-muted text-sm mb-6 line-clamp-2 h-10">
          {workspace.description}
        </p>

        <div className="flex items-center justify-between text-sm pt-4 border-t border-gothic-700/50">
          <div className="flex items-center gap-2 text-gothic-text">
            <span className="font-bold">{workspace.agentCount}</span>
            <span className="text-gothic-muted">Agents</span>
          </div>
          <div className="text-gothic-muted text-xs">
            Last active {new Date(workspace.lastActive).toLocaleDateString()}
          </div>
        </div>
      </div>
    </Card>
  );
};
