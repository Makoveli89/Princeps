import React, { useState } from 'react';
import { Plus, Database, Edit2, Check } from 'lucide-react';
import { Workspace } from '../types';

export const Workspaces = ({ workspaces, activeId, onChange }: { workspaces: Workspace[], activeId: string | null, onChange: (id: string) => void }) => {
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');

  const handleCreate = async () => {
      try {
          const res = await fetch('/api/workspaces', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ name: newName, description: newDesc })
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
         <h2 className="text-2xl text-white gothic-font">Workspaces</h2>
         <button
            onClick={() => setShowCreate(!showCreate)}
            className="bg-gray-900 border border-gray-700 text-gray-300 px-4 py-2 flex items-center gap-2 text-xs uppercase hover:bg-gray-800 transition-colors">
            <Plus size={14} /> Create New
         </button>
      </div>

      {showCreate && (
          <div className="bg-[#080808] border border-gray-700 p-6 max-w-lg animate-in fade-in slide-in-from-top-4">
              <h3 className="text-white mb-4">New Workspace</h3>
              <input
                className="w-full bg-black border border-gray-800 text-white p-2 mb-2"
                placeholder="Workspace Name"
                value={newName}
                onChange={e => setNewName(e.target.value)}
              />
               <input
                className="w-full bg-black border border-gray-800 text-white p-2 mb-4"
                placeholder="Description"
                value={newDesc}
                onChange={e => setNewDesc(e.target.value)}
              />
              <button
                onClick={handleCreate}
                className="bg-cyan-900/50 text-cyan-400 border border-cyan-800 px-4 py-2 text-sm flex items-center gap-2 hover:bg-cyan-900/80">
                  <Check size={14} /> Confirm
              </button>
          </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {workspaces.map(ws => (
            <div
                key={ws.id}
                role="button"
                tabIndex={0}
                onClick={() => onChange(ws.id)}
                onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        onChange(ws.id);
                    }
                }}
                className={`p-6 border relative group cursor-pointer transition-all duration-300 outline-none focus:ring-2 focus:ring-cyan-500
                    ${ws.id === activeId
                        ? 'bg-cyan-950/10 border-cyan-500/50 shadow-[0_0_10px_rgba(0,243,255,0.1)]'
                        : 'bg-[#050505] border-gray-800 hover:border-gray-600'
                    }`}
            >
                {ws.id === activeId && (
                    <div className="absolute top-2 right-2 text-[10px] bg-cyan-900/30 text-cyan-400 px-2 py-0.5 border border-cyan-800 rounded">ACTIVE</div>
                )}

                <div className="flex items-center gap-3 mb-4">
                    <div className={`p-2 rounded bg-gray-900 ${ws.id === activeId ? 'text-cyan-400' : 'text-gray-600'}`}>
                        <Database size={20} />
                    </div>
                    <h3 className="text-lg text-gray-200 font-bold tracking-wide">{ws.name}</h3>
                </div>

                <div className="space-y-2 text-sm mono-font">
                    <div className="flex justify-between border-b border-gray-800 pb-1">
                        <span className="text-gray-600">ID</span>
                        <span className="text-gray-400 text-xs">{ws.id.substring(0,8)}...</span>
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

                <div className="mt-6 flex justify-end opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className="text-xs text-gray-500 hover:text-white flex items-center gap-1">
                        <Edit2 size={12} /> CONFIG
                    </button>
                </div>
            </div>
        ))}
      </div>
    </div>
  );
};
