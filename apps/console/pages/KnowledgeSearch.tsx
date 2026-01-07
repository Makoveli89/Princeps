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
        if(!query) return;

        setIsSearching(true);
        setError(null);

        try {
            const res = await fetch(`/api/search?q=${encodeURIComponent(query)}&workspaceId=${workspace.id}`);
            if (res.ok) {
                const data = await res.json();
                setResults(data);
                setHasSearched(true);
            } else {
                setError("Search failed.");
            }
        } catch (err) {
            setError(`Network error: ${err}`);
        } finally {
            setIsSearching(false);
        }
    };

    return (
        <div className="max-w-5xl mx-auto space-y-8">
            <div className="text-center space-y-2">
                <h2 className="text-3xl text-white gothic-font tracking-wide">Neural Search</h2>
                <p className="text-gray-500 mono-font text-sm">Direct interface to the Vector Amygdala of <span className="text-cyan-400">{workspace?.name}</span></p>
            </div>

            {/* Search Bar */}
            <form onSubmit={handleSearch} className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-900 to-blue-900 rounded-sm opacity-30 group-hover:opacity-100 transition duration-500 blur"></div>
                <div className="relative flex bg-[#020202] border border-gray-800 p-1 rounded-sm">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Query the knowledge graph (e.g., 'Deployment protocols for cluster alpha')..."
                        className="flex-1 bg-transparent border-none outline-none text-gray-200 px-4 py-3 mono-font placeholder-gray-700"
                    />
                    <button
                        type="submit"
                        disabled={isSearching}
                        className="bg-cyan-950 border-l border-gray-800 text-cyan-400 px-6 py-2 uppercase text-sm tracking-widest hover:bg-cyan-900/40 transition-colors disabled:opacity-50"
                    >
                        {isSearching ? 'SCANNING...' : 'SEEK'}
                    </button>
                </div>
            </form>

            {/* Results Area */}
            {hasSearched && (
                <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="flex items-center justify-between border-b border-gray-800 pb-2">
                        <span className="text-[10px] uppercase tracking-widest text-gray-500">
                            Results found: {results.length}
                        </span>
                        <span className="text-[10px] uppercase tracking-widest text-cyan-600 flex items-center gap-1">
                            <Sparkles size={10} /> Hybrid Retrieval Active
                        </span>
                    </div>

                    <div className="grid gap-4">
                        {results.length === 0 && (
                            <div className="text-gray-500 text-center py-10 font-mono text-sm">NO MATCHING KNOWLEDGE ATOMS FOUND IN SECTOR.</div>
                        )}
                        {results.map((result) => (
                            <div key={result.id} className="bg-[#050505] border border-gray-800 p-4 relative group hover:border-cyan-900/50 transition-colors">
                                <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-transparent via-cyan-900/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>

                                <div className="flex justify-between items-start mb-2">
                                    <div className="flex items-center gap-2 text-cyan-500 text-xs mono-font">
                                        <FileText size={12} />
                                        <span>{result.source}</span>
                                        <span className="text-gray-700">::</span>
                                        <span>Chunk {result.chunk_index}</span>
                                    </div>
                                    <div className={`text-xs px-2 py-0.5 rounded border ${result.score > 0.8 ? 'border-emerald-900 text-emerald-500 bg-emerald-950/20' : 'border-cyan-900 text-cyan-600 bg-cyan-950/10'}`}>
                                        {(result.score * 100).toFixed(1)}% Match
                                    </div>
                                </div>

                                <p className="text-gray-300 font-serif leading-relaxed text-sm pl-4 border-l border-gray-800 group-hover:border-cyan-900/30 transition-colors">
                                    "{result.content}"
                                </p>

                                <div className="mt-4 flex justify-end opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button className="text-[10px] uppercase text-gray-500 hover:text-cyan-400 flex items-center gap-1">
                                        Inspect Source <ArrowRight size={10} />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {error && <div className="text-red-500 text-center font-mono mt-10">{error}</div>}

            {!hasSearched && !isSearching && (
                <div className="flex flex-col items-center justify-center py-20 opacity-30">
                    <Database size={64} className="text-gray-700 mb-4" />
                    <p className="text-gray-600 mono-font text-sm uppercase tracking-widest">Awaiting Neural Input</p>
                </div>
            )}
        </div>
    );
};
