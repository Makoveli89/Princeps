import React, { useState } from 'react';
import { Upload, GitBranch, FileText, CheckCircle2, AlertCircle, HardDrive, Loader2 } from 'lucide-react';
import { Workspace } from '../types';

export const IngestData = ({ workspace }: { workspace: Workspace }) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'repo'>('upload');
  const [isUploading, setIsUploading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ingestStats, setIngestStats] = useState<any>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!e.target.files || e.target.files.length === 0) return;

      const file = e.target.files[0];
      setIsUploading(true);
      setError(null);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('workspace_id', workspace.id);

      try {
          const res = await fetch('/api/ingest', {
              method: 'POST',
              body: formData
          });

          if (res.ok) {
              const data = await res.json();
              if (data.status === 'success' || data.status === 'skipped') {
                  setIngestStats(data);
                  setSuccess(true);
                  setTimeout(() => setSuccess(false), 5000);
              } else {
                  setError("Ingestion reported failure: " + JSON.stringify(data));
              }
          } else {
              setError(`Upload failed: ${res.statusText}`);
          }
      } catch (err) {
          setError(`Error: ${err}`);
      } finally {
          setIsUploading(false);
      }
  };

  return (
    <div className="max-w-4xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="mb-10 text-center">
        <h2 className="text-3xl text-gray-100 gothic-font tracking-wide mb-2 text-glow">Knowledge Assimilation</h2>
        <p className="text-gray-500 mono-font text-sm">Target Workspace: <span className="text-cyan-400 drop-shadow-[0_0_5px_rgba(0,243,255,0.4)]">{workspace?.name}</span></p>
      </div>

      <div className="flex justify-center mb-10">
        <div className="bg-[#050505] border border-gray-800 p-1 flex rounded-sm shadow-lg">
            <button
                onClick={() => setActiveTab('upload')}
                className={`px-8 py-3 text-[10px] font-bold uppercase tracking-[0.2em] flex items-center gap-3 transition-all ${activeTab === 'upload' ? 'bg-cyan-950/20 text-cyan-400 border border-cyan-900/50 shadow-[0_0_15px_rgba(0,243,255,0.1)]' : 'text-gray-600 hover:text-gray-400 border border-transparent'}`}
            >
                <FileText size={14} /> Sacred Texts (Docs)
            </button>
            <button
                disabled
                onClick={() => setActiveTab('repo')}
                className={`opacity-50 cursor-not-allowed px-8 py-3 text-[10px] font-bold uppercase tracking-[0.2em] flex items-center gap-3 transition-all ${activeTab === 'repo' ? 'bg-cyan-950/20 text-cyan-400 border border-cyan-900/50 shadow-[0_0_15px_rgba(0,243,255,0.1)]' : 'text-gray-600 hover:text-gray-400 border border-transparent'}`}
            >
                <GitBranch size={14} /> Cognitive Patterns (Repo)
            </button>
        </div>
      </div>

      <div className="bg-[#030303] border border-gray-800 p-16 relative overflow-hidden group shadow-2xl hover:border-gray-700 transition-colors duration-500">
         {/* Tech Background Lines */}
         <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-900/20 to-transparent"></div>
         <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-900/20 to-transparent"></div>

         {activeTab === 'upload' ? (
             <div
                className="flex flex-col items-center justify-center border-2 border-dashed border-gray-800 hover:border-cyan-500/50 transition-all duration-500 rounded-sm p-12 bg-[#050505] group/drop relative overflow-hidden cursor-pointer"
             >
                 {isUploading && (
                     <div className="absolute inset-0 bg-black/80 z-20 flex flex-col items-center justify-center">
                         <Loader2 className="animate-spin text-cyan-500 mb-4" size={48} />
                         <span className="text-cyan-400 mono-font text-xs tracking-widest animate-pulse">ASSIMILATING KNOWLEDGE...</span>
                     </div>
                 )}

                 <div className="absolute inset-0 bg-cyan-950/5 opacity-0 group-hover/drop:opacity-100 transition-opacity duration-500"></div>
                 <Upload size={48} className="text-gray-700 mb-6 group-hover/drop:text-cyan-400 group-hover/drop:scale-110 transition-all duration-500 drop-shadow-[0_0_10px_rgba(0,243,255,0.2)] relative z-10" />
                 <h3 className="text-gray-300 font-mono text-lg relative z-10 tracking-wide">Upload Documents</h3>
                 <p className="text-gray-600 text-xs mt-2 mb-8 text-center max-w-sm relative z-10 font-mono">
                    Drag and drop PDF, TXT, MD, PY, JSON files here to embed them into the active workspace's vector index.
                 </p>
                 <input
                    type="file"
                    className="hidden"
                    id="file-upload"
                    onChange={handleFileUpload}
                 />
                 <label
                    htmlFor="file-upload"
                    className="relative z-10 cursor-pointer bg-gray-900 hover:bg-cyan-950 hover:text-cyan-400 hover:border-cyan-500/50 text-gray-400 px-8 py-3 border border-gray-700 text-[10px] font-bold uppercase tracking-[0.2em] transition-all shadow-lg"
                 >
                    Select Files
                 </label>
             </div>
         ) : (
            // Repo tab disabled for now
             <div className="text-center text-gray-500 font-mono text-xs">Repository ingestion module offline.</div>
         )}

         {/* Status Overlay */}
         {success && (
            <div className="absolute inset-0 bg-[#020202]/95 backdrop-blur-sm flex flex-col items-center justify-center z-20 animate-in fade-in zoom-in-95 duration-300">
                <div className="relative">
                    <div className="absolute inset-0 bg-emerald-500 blur-[40px] opacity-20 rounded-full animate-pulse"></div>
                    <CheckCircle2 size={64} className="text-emerald-500 mb-6 relative z-10 drop-shadow-[0_0_15px_rgba(16,185,129,0.5)]" />
                </div>
                <h3 className="text-emerald-100 gothic-font text-2xl tracking-widest text-glow">Ingestion Complete</h3>
                <p className="text-emerald-600/80 mono-font text-xs mt-3 uppercase tracking-wider">
                    {ingestStats?.status === 'skipped' ? 'Document already exists.' : `Added ${ingestStats?.chunks || 0} chunks to vector index.`}
                </p>
            </div>
         )}

         {error && (
             <div className="absolute bottom-4 left-4 right-4 bg-red-900/20 border border-red-800 text-red-400 p-4 text-xs mono-font flex items-center gap-3">
                 <AlertCircle size={16} />
                 {error}
                 <button onClick={() => setError(null)} className="ml-auto hover:text-white">DISMISS</button>
             </div>
         )}
      </div>
    </div>
  );
};
