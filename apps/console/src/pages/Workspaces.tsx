import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { Workspace } from '../types';
import { Plus, Server, Activity, AlertCircle, X } from 'lucide-react';

export const Workspaces: React.FC = () => {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);

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

  const handleCreate = async (name: string, description: string) => {
    try {
      const res = await fetch('/api/workspaces', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description })
      });
      if (res.ok) {
        fetchWorkspaces();
        setShowCreateModal(false);
      }
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="space-y-8 relative">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-serif text-gothic-text mb-2">Workspaces</h1>
          <p className="text-gothic-muted">Manage your isolated knowledge environments.</p>
        </div>
        <Button onClick={() => setShowCreateModal(true)}>
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
          {workspaces.length === 0 && (
            <div className="col-span-full py-12 text-center bg-gothic-800/30 border border-dashed border-gothic-700 rounded-lg">
              <p className="text-gothic-muted mb-4">No workspaces found. Initialize your first project.</p>
              <Button onClick={() => setShowCreateModal(true)}>Initialize System</Button>
            </div>
          )}
        </div>
      )}

      {showCreateModal && (
        <CreateModal onClose={() => setShowCreateModal(false)} onSubmit={handleCreate} />
      )}
    </div>
  );
};

const WorkspaceCard: React.FC<{ workspace: Workspace }> = ({ workspace }) => {
  const statusColors: any = {
    active: 'text-green-400',
    archived: 'text-gothic-muted',
    error: 'text-red-400',
  };

  const statusIcon: any = {
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
          <div className={`flex items-center gap-2 px-2 py-1 rounded bg-gothic-900/50 text-xs font-mono uppercase tracking-wider border border-gothic-700 ${statusColors[workspace.status] || 'text-gothic-muted'}`}>
            {statusIcon[workspace.status]}
            {workspace.status}
          </div>
          <span className="text-xs text-gothic-muted font-mono">{workspace.id.substring(0, 8)}...</span>
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

const CreateModal: React.FC<{ onClose: () => void, onSubmit: (name: string, desc: string) => void }> = ({ onClose, onSubmit }) => {
  const [name, setName] = useState('');
  const [desc, setDesc] = useState('');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <Card className="w-full max-w-md mx-4 animate-in fade-in zoom-in duration-200">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-serif font-bold text-gothic-text">New Workspace</h2>
          <button onClick={onClose} className="text-gothic-muted hover:text-white"><X size={20}/></button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gothic-muted mb-1">Name</label>
            <input
              className="w-full bg-gothic-900 border border-gothic-700 rounded px-3 py-2 text-gothic-text focus:border-gothic-gold outline-none"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Project Alpha"
              autoFocus
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gothic-muted mb-1">Description</label>
            <textarea
              className="w-full bg-gothic-900 border border-gothic-700 rounded px-3 py-2 text-gothic-text focus:border-gothic-gold outline-none h-24 resize-none"
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="Brief description of the workspace..."
            />
          </div>
          <div className="flex justify-end gap-3 mt-6">
            <Button variant="ghost" onClick={onClose}>Cancel</Button>
            <Button onClick={() => onSubmit(name, desc)} disabled={!name}>Create Workspace</Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
