import React, { useState } from 'react';
import { Plus, Database, Edit2, Check } from 'lucide-react';
import { Workspace } from '../types';

export const Workspaces = ({
  workspaces,
  activeId,
  onChange,
}: {
  workspaces: Workspace[];
  activeId: string | null;
  onChange: (id: string) => void;
}) => {
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');

  const handleCreate = async () => {
    try {
      const res = await fetch('/api/workspaces', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName, description: newDesc }),
      });
      if (res.ok) {
        window.location.reload(); // Simple reload to refresh state for now
      } else {
        alert('Failed to create workspace');
      }
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="gothic-font text-2xl text-white">Workspaces</h2>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="flex items-center gap-2 border border-gray-700 bg-gray-900 px-4 py-2 text-xs uppercase text-gray-300 transition-colors hover:bg-gray-800"
        >
          <Plus size={14} /> Create New
        </button>
      </div>

      {showCreate && (
        <div className="animate-in fade-in slide-in-from-top-4 max-w-lg border border-gray-700 bg-[#080808] p-6">
          <h3 className="mb-4 text-white">New Workspace</h3>
          <input
            className="mb-2 w-full border border-gray-800 bg-black p-2 text-white"
            placeholder="Workspace Name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <input
            className="mb-4 w-full border border-gray-800 bg-black p-2 text-white"
            placeholder="Description"
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
          />
          <button
            onClick={handleCreate}
            className="flex items-center gap-2 border border-cyan-800 bg-cyan-900/50 px-4 py-2 text-sm text-cyan-400 hover:bg-cyan-900/80"
          >
            <Check size={14} /> Confirm
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
        {workspaces.map((ws) => (
          <div
            key={ws.id}
            onClick={() => onChange(ws.id)}
            className={`group relative cursor-pointer border p-6 transition-all duration-300 ${
              ws.id === activeId
                ? 'border-cyan-500/50 bg-cyan-950/10 shadow-[0_0_10px_rgba(0,243,255,0.1)]'
                : 'border-gray-800 bg-[#050505] hover:border-gray-600'
            }`}
          >
            {ws.id === activeId && (
              <div className="absolute right-2 top-2 rounded border border-cyan-800 bg-cyan-900/30 px-2 py-0.5 text-[10px] text-cyan-400">
                ACTIVE
              </div>
            )}

            <div className="mb-4 flex items-center gap-3">
              <div
                className={`rounded bg-gray-900 p-2 ${ws.id === activeId ? 'text-cyan-400' : 'text-gray-600'}`}
              >
                <Database size={20} />
              </div>
              <h3 className="text-lg font-bold tracking-wide text-gray-200">{ws.name}</h3>
            </div>

            <div className="mono-font space-y-2 text-sm">
              <div className="flex justify-between border-b border-gray-800 pb-1">
                <span className="text-gray-600">ID</span>
                <span className="text-xs text-gray-400">{ws.id.substring(0, 8)}...</span>
              </div>
              <div className="flex justify-between border-b border-gray-800 pb-1">
                <span className="text-gray-600">DOCS</span>
                <span className="text-gray-400">{ws.docCount}</span>
              </div>
              <div className="flex justify-between border-b border-gray-800 pb-1">
                <span className="text-gray-600">CHUNKS</span>
                <span className="text-gray-400">{ws.chunkCount.toLocaleString()}</span>
              </div>
              <div className="flex justify-between border-b border-gray-800 pb-1">
                <span className="text-gray-600">RUNS</span>
                <span className="text-gray-400">{ws.runCount.toLocaleString()}</span>
              </div>
            </div>

            <div className="mt-6 flex justify-end opacity-0 transition-opacity group-hover:opacity-100">
              <button className="flex items-center gap-1 text-xs text-gray-500 hover:text-white">
                <Edit2 size={12} /> CONFIG
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
