import React, { useState, useRef } from 'react';
import {
  Upload,
  GitBranch,
  FileText,
  CheckCircle2,
  AlertCircle,
  HardDrive,
  Loader2,
} from 'lucide-react';
import { Workspace } from '../types';

export const IngestData = ({ workspace }: { workspace: Workspace }) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'repo'>('upload');
  const [isUploading, setIsUploading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ingestStats, setIngestStats] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        if (data.status === 'success' || data.status === 'skipped') {
          setIngestStats(data);
          setSuccess(true);
          setTimeout(() => setSuccess(false), 5000);
        } else {
          setError('Ingestion reported failure: ' + JSON.stringify(data));
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
    <div className="animate-in fade-in slide-in-from-bottom-4 mx-auto max-w-4xl duration-700">
      <div className="mb-10 text-center">
        <h2 className="gothic-font text-glow mb-2 text-3xl tracking-wide text-gray-100">
          Knowledge Assimilation
        </h2>
        <p className="mono-font text-sm text-gray-500">
          Target Workspace:{' '}
          <span className="text-cyan-400 drop-shadow-[0_0_5px_rgba(0,243,255,0.4)]">
            {workspace?.name}
          </span>
        </p>
      </div>

      <div className="mb-10 flex justify-center">
        <div className="flex rounded-sm border border-gray-800 bg-[#050505] p-1 shadow-lg">
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex items-center gap-3 px-8 py-3 text-[10px] font-bold tracking-[0.2em] uppercase transition-all ${activeTab === 'upload' ? 'border border-cyan-900/50 bg-cyan-950/20 text-cyan-400 shadow-[0_0_15px_rgba(0,243,255,0.1)]' : 'border border-transparent text-gray-600 hover:text-gray-400'}`}
          >
            <FileText size={14} /> Sacred Texts (Docs)
          </button>
          <button
            disabled
            onClick={() => setActiveTab('repo')}
            className={`flex cursor-not-allowed items-center gap-3 px-8 py-3 text-[10px] font-bold tracking-[0.2em] uppercase opacity-50 transition-all ${activeTab === 'repo' ? 'border border-cyan-900/50 bg-cyan-950/20 text-cyan-400 shadow-[0_0_15px_rgba(0,243,255,0.1)]' : 'border border-transparent text-gray-600 hover:text-gray-400'}`}
          >
            <GitBranch size={14} /> Cognitive Patterns (Repo)
          </button>
        </div>
      </div>

      <div className="group relative overflow-hidden border border-gray-800 bg-[#030303] p-16 shadow-2xl transition-colors duration-500 hover:border-gray-700">
        {/* Tech Background Lines */}
        <div className="absolute top-0 left-0 h-1 w-full bg-gradient-to-r from-transparent via-cyan-900/20 to-transparent"></div>
        <div className="absolute bottom-0 left-0 h-1 w-full bg-gradient-to-r from-transparent via-cyan-900/20 to-transparent"></div>

        {activeTab === 'upload' ? (
          <div
            role="button"
            tabIndex={0}
            aria-label="Upload documents drop zone"
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            className="group/drop relative flex cursor-pointer flex-col items-center justify-center overflow-hidden rounded-sm border-2 border-dashed border-gray-800 bg-[#050505] p-12 transition-all duration-500 hover:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#050505] focus:outline-none"
          >
            {isUploading && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/80">
                <Loader2 className="mb-4 animate-spin text-cyan-500" size={48} />
                <span className="mono-font animate-pulse text-xs tracking-widest text-cyan-400">
                  ASSIMILATING KNOWLEDGE...
                </span>
              </div>
            )}

            <div className="absolute inset-0 bg-cyan-950/5 opacity-0 transition-opacity duration-500 group-hover/drop:opacity-100"></div>
            <Upload
              size={48}
              className="relative z-10 mb-6 text-gray-700 drop-shadow-[0_0_10px_rgba(0,243,255,0.2)] transition-all duration-500 group-hover/drop:scale-110 group-hover/drop:text-cyan-400"
            />
            <h3 className="relative z-10 font-mono text-lg tracking-wide text-gray-300">
              Upload Documents
            </h3>
            <p className="relative z-10 mt-2 mb-8 max-w-sm text-center font-mono text-xs text-gray-600">
              Drag and drop PDF, TXT, MD, PY, JSON files here to embed them into the active
              workspace's vector index.
            </p>
            <input
              type="file"
              className="hidden"
              id="file-upload"
              ref={fileInputRef}
              onChange={handleFileUpload}
            />
            <div className="relative z-10 cursor-pointer border border-gray-700 bg-gray-900 px-8 py-3 text-[10px] font-bold tracking-[0.2em] text-gray-400 uppercase shadow-lg transition-all group-hover/drop:border-cyan-500/50 group-hover/drop:bg-cyan-950 group-hover/drop:text-cyan-400">
              Select Files
            </div>
          </div>
        ) : (
          // Repo tab disabled for now
          <div className="text-center font-mono text-xs text-gray-500">
            Repository ingestion module offline.
          </div>
        )}

        {/* Status Overlay */}
        {success && (
          <div className="animate-in fade-in zoom-in-95 absolute inset-0 z-20 flex flex-col items-center justify-center bg-[#020202]/95 backdrop-blur-sm duration-300">
            <div className="relative">
              <div className="absolute inset-0 animate-pulse rounded-full bg-emerald-500 opacity-20 blur-[40px]"></div>
              <CheckCircle2
                size={64}
                className="relative z-10 mb-6 text-emerald-500 drop-shadow-[0_0_15px_rgba(16,185,129,0.5)]"
              />
            </div>
            <h3 className="gothic-font text-glow text-2xl tracking-widest text-emerald-100">
              Ingestion Complete
            </h3>
            <p className="mono-font mt-3 text-xs tracking-wider text-emerald-600/80 uppercase">
              {ingestStats?.status === 'skipped'
                ? 'Document already exists.'
                : `Added ${ingestStats?.chunks || 0} chunks to vector index.`}
            </p>
          </div>
        )}

        {error && (
          <div className="mono-font absolute right-4 bottom-4 left-4 flex items-center gap-3 border border-red-800 bg-red-900/20 p-4 text-xs text-red-400">
            <AlertCircle size={16} />
            {error}
            <button onClick={() => setError(null)} className="ml-auto hover:text-white">
              DISMISS
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
