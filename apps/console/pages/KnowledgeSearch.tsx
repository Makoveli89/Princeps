import React, { useState } from 'react';
import { Workspace, SearchResult } from '../types';
import { Search, Database, FileText, ArrowRight, Sparkles } from 'lucide-react';

export const KnowledgeSearch = ({ workspace }: { workspace: Workspace }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query) return;

    setIsSearching(true);
    setError(null);

    try {
      const res = await fetch(
        `/api/search?q=${encodeURIComponent(query)}&workspaceId=${workspace.id}`,
      );
      if (res.ok) {
        const data = await res.json();
        setResults(data);
        setHasSearched(true);
      } else {
        setError('Search failed.');
      }
    } catch (err) {
      setError(`Network error: ${err}`);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div className="space-y-2 text-center">
        <h2 className="gothic-font text-3xl tracking-wide text-white">Neural Search</h2>
        <p className="mono-font text-sm text-gray-500">
          Direct interface to the Vector Amygdala of{' '}
          <span className="text-cyan-400">{workspace?.name}</span>
        </p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="group relative">
        <div className="absolute -inset-0.5 rounded-sm bg-gradient-to-r from-cyan-900 to-blue-900 opacity-30 blur transition duration-500 group-hover:opacity-100"></div>
        <div className="relative flex rounded-sm border border-gray-800 bg-[#020202] p-1">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Query the knowledge graph (e.g., 'Deployment protocols for cluster alpha')..."
            className="mono-font flex-1 border-none bg-transparent px-4 py-3 text-gray-200 placeholder-gray-700 outline-none"
          />
          <button
            type="submit"
            disabled={isSearching}
            className="border-l border-gray-800 bg-cyan-950 px-6 py-2 text-sm uppercase tracking-widest text-cyan-400 transition-colors hover:bg-cyan-900/40 disabled:opacity-50"
          >
            {isSearching ? 'SCANNING...' : 'SEEK'}
          </button>
        </div>
      </form>

      {/* Results Area */}
      {hasSearched && (
        <div className="animate-in fade-in slide-in-from-bottom-4 space-y-4 duration-500">
          <div className="flex items-center justify-between border-b border-gray-800 pb-2">
            <span className="text-[10px] uppercase tracking-widest text-gray-500">
              Results found: {results.length}
            </span>
            <span className="flex items-center gap-1 text-[10px] uppercase tracking-widest text-cyan-600">
              <Sparkles size={10} /> Hybrid Retrieval Active
            </span>
          </div>

          <div className="grid gap-4">
            {results.length === 0 && (
              <div className="py-10 text-center font-mono text-sm text-gray-500">
                NO MATCHING KNOWLEDGE ATOMS FOUND IN SECTOR.
              </div>
            )}
            {results.map((result) => (
              <div
                key={result.id}
                className="group relative border border-gray-800 bg-[#050505] p-4 transition-colors hover:border-cyan-900/50"
              >
                <div className="absolute left-0 top-0 h-full w-1 bg-gradient-to-b from-transparent via-cyan-900/50 to-transparent opacity-0 transition-opacity group-hover:opacity-100"></div>

                <div className="mb-2 flex items-start justify-between">
                  <div className="mono-font flex items-center gap-2 text-xs text-cyan-500">
                    <FileText size={12} />
                    <span>{result.source}</span>
                    <span className="text-gray-700">::</span>
                    <span>Chunk {result.chunk_index}</span>
                  </div>
                  <div
                    className={`rounded border px-2 py-0.5 text-xs ${result.score > 0.8 ? 'border-emerald-900 bg-emerald-950/20 text-emerald-500' : 'border-cyan-900 bg-cyan-950/10 text-cyan-600'}`}
                  >
                    {(result.score * 100).toFixed(1)}% Match
                  </div>
                </div>

                <p className="border-l border-gray-800 pl-4 font-serif text-sm leading-relaxed text-gray-300 transition-colors group-hover:border-cyan-900/30">
                  "{result.content}"
                </p>

                <div className="mt-4 flex justify-end opacity-0 transition-opacity group-hover:opacity-100">
                  <button className="flex items-center gap-1 text-[10px] uppercase text-gray-500 hover:text-cyan-400">
                    Inspect Source <ArrowRight size={10} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {error && <div className="mt-10 text-center font-mono text-red-500">{error}</div>}

      {!hasSearched && !isSearching && (
        <div className="flex flex-col items-center justify-center py-20 opacity-30">
          <Database size={64} className="mb-4 text-gray-700" />
          <p className="mono-font text-sm uppercase tracking-widest text-gray-600">
            Awaiting Neural Input
          </p>
        </div>
      )}
    </div>
  );
};
