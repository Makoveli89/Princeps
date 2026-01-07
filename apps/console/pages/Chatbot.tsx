import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI } from "@google/genai";
import { Send, Bot, User, Sparkles, Globe, Terminal, Loader2, AlertTriangle, ExternalLink } from 'lucide-react';
import { Workspace } from '../types';

interface Message {
    id: string;
    role: 'user' | 'model';
    text: string;
    groundingMetadata?: any;
    timestamp: Date;
}

export const Chatbot = ({ workspace }: { workspace: Workspace }) => {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: 'init',
            role: 'model',
            text: `Princeps Neural Link established. Connected to workspace: ${workspace?.name || 'Unknown'}.\n\nReady for input. Toggle "Netrunner Mode" for real-time web access.`,
            timestamp: new Date()
        }
    ]);
    const [input, setInput] = useState('');
    const [isThinking, setIsThinking] = useState(false);
    const [useWebSearch, setUseWebSearch] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isThinking) return;

        const userMsg: Message = {
            id: Date.now().toString(),
            role: 'user',
            text: input,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsThinking(true);

        try {
            // Fix: Use Vite env var
            const apiKey = import.meta.env.VITE_GEMINI_API_KEY || import.meta.env.VITE_GOOGLE_API_KEY;

            if (!apiKey) {
                throw new Error("API_KEY_MISSING");
            }

            const ai = new GoogleGenAI({ apiKey: apiKey });

            // Logic:
            // - Standard Chat: 'gemini-2.0-flash-exp' (Updated to valid model)
            const modelName = 'gemini-2.0-flash-exp';

            const config: any = {};
            if (useWebSearch) {
                config.tools = [{ googleSearch: {} }];
            }

            const response = await ai.models.generateContent({
                model: modelName,
                contents: input,
                config: config
            });

            const text = response.text || "No response received from the neural net.";
            const groundingMetadata = response.candidates?.[0]?.groundingMetadata;

            const aiMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: 'model',
                text: text,
                groundingMetadata: groundingMetadata,
                timestamp: new Date()
            };

            setMessages(prev => [...prev, aiMsg]);

        } catch (error: any) {
            console.error("Gemini Error:", error);

            let errorText = "Neural Link Error: Connection terminated.";
            if (error.message === "API_KEY_MISSING") {
                errorText = "CRITICAL FAILURE: API Key not found in environment variables.\n\nPlease set VITE_GEMINI_API_KEY in .env file.";
            } else {
                // Fallback for demo purposes if API call fails
                errorText = `Error accessing model: ${error.message}.\n\n(Mock Response): This appears to be a simulation. If I were fully connected, I would provide a detailed analysis of "${input}".`;
            }

            const errorMsg: Message = {
                id: (Date.now() + 1).toString(),
                role: 'model',
                text: errorText,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, errorMsg]);
        } finally {
            setIsThinking(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="h-full flex flex-col gap-6 animate-in fade-in duration-500 max-w-6xl mx-auto">

            {/* Header */}
            <div className="flex items-center justify-between border-b border-gray-800 pb-4">
                <div>
                    <h2 className="text-3xl text-white gothic-font tracking-wide text-glow flex items-center gap-3">
                        <Bot size={28} className="text-cyan-400" /> Neural Chat
                    </h2>
                    <p className="text-gray-500 text-sm mono-font mt-1">
                        Direct Interface :: {useWebSearch ? 'WEB ACCESS ENABLED' : 'DEEP REASONING MODE'}
                    </p>
                </div>

                {/* Mode Toggle */}
                <button
                    onClick={() => setUseWebSearch(!useWebSearch)}
                    className={`flex items-center gap-3 px-4 py-2 border transition-all duration-300 ${
                        useWebSearch
                        ? 'bg-cyan-950/30 border-cyan-500 text-cyan-400 shadow-[0_0_15px_rgba(0,243,255,0.2)]'
                        : 'bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-500'
                    }`}
                >
                    {useWebSearch ? <Globe size={16} className="animate-pulse" /> : <Sparkles size={16} />}
                    <span className="text-xs font-bold uppercase tracking-widest">
                        {useWebSearch ? 'Netrunner Mode (Search)' : 'Reasoning Mode (Pro)'}
                    </span>
                </button>
            </div>

            {/* Chat History */}
            <div className="flex-1 bg-[#030303] border border-gray-800 relative overflow-hidden flex flex-col shadow-2xl">
                {/* Background Grid */}
                <div className="absolute inset-0 z-0 opacity-[0.03]" style={{ backgroundImage: 'linear-gradient(#333 1px, transparent 1px), linear-gradient(90deg, #333 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>

                <div className="flex-1 overflow-y-auto p-6 space-y-6 relative z-10 custom-scrollbar">
                    {messages.map((msg) => (
                        <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[80%] flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                                {/* Avatar */}
                                <div className={`w-8 h-8 rounded-sm flex items-center justify-center flex-shrink-0 border ${
                                    msg.role === 'user'
                                    ? 'bg-red-950/20 border-red-800 text-red-500'
                                    : 'bg-cyan-950/20 border-cyan-800 text-cyan-400'
                                }`}>
                                    {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                                </div>

                                {/* Content */}
                                <div className="flex flex-col gap-2">
                                    <div className={`p-4 rounded-sm border backdrop-blur-sm ${
                                        msg.role === 'user'
                                        ? 'bg-red-950/10 border-red-900/50 text-gray-200'
                                        : 'bg-cyan-950/10 border-cyan-900/50 text-cyan-50'
                                    }`}>
                                        <p className="whitespace-pre-wrap mono-font text-sm leading-relaxed">
                                            {msg.text}
                                        </p>
                                    </div>

                                    {/* Grounding / Sources */}
                                    {msg.groundingMetadata?.groundingChunks && (
                                        <div className="bg-[#050505] border border-gray-800 p-3 rounded-sm animate-in slide-in-from-top-2">
                                            <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                                                <Globe size={12} /> Sources Detected
                                            </div>
                                            <div className="flex flex-wrap gap-2">
                                                {msg.groundingMetadata.groundingChunks.map((chunk: any, idx: number) =>
                                                    chunk.web?.uri ? (
                                                        <a
                                                            key={idx}
                                                            href={chunk.web.uri}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            className="flex items-center gap-1 px-2 py-1 bg-gray-900 border border-gray-700 text-[10px] text-cyan-400 hover:border-cyan-500 hover:text-cyan-300 transition-colors"
                                                        >
                                                            <ExternalLink size={10} />
                                                            {chunk.web.title || new URL(chunk.web.uri).hostname}
                                                        </a>
                                                    ) : null
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    <span className="text-[10px] text-gray-600 font-mono">
                                        {msg.timestamp.toLocaleTimeString()}
                                    </span>
                                </div>
                            </div>
                        </div>
                    ))}

                    {isThinking && (
                        <div className="flex justify-start">
                            <div className="max-w-[80%] flex gap-4 flex-row">
                                <div className="w-8 h-8 rounded-sm bg-cyan-950/20 border border-cyan-800 text-cyan-400 flex items-center justify-center flex-shrink-0">
                                    <Bot size={16} />
                                </div>
                                <div className="p-4 rounded-sm border border-cyan-900/30 bg-cyan-950/5 flex items-center gap-2">
                                    <Loader2 size={16} className="animate-spin text-cyan-500" />
                                    <span className="text-xs mono-font text-cyan-600 animate-pulse">
                                        {useWebSearch ? 'SCANNING NETWORK...' : 'PROCESSING THOUGHT...'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 border-t border-gray-800 bg-[#020202]">
                    <div className="relative group focus-within:ring-1 focus-within:ring-cyan-900/50">
                        <div className="absolute inset-0 bg-gradient-to-r from-cyan-900/10 to-transparent pointer-events-none opacity-0 group-focus-within:opacity-100 transition-opacity"></div>
                        <div className="absolute left-4 top-4 text-gray-600 pointer-events-none">
                            <Terminal size={16} />
                        </div>
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={`Enter prompt for ${useWebSearch ? 'Netrunner (Web)' : 'Princeps (Core)'}...`}
                            className="w-full bg-[#050505] border border-gray-800 text-gray-300 pl-12 pr-14 py-4 mono-font resize-none outline-none focus:border-cyan-700/50 transition-colors h-14 min-h-[56px] max-h-32 custom-scrollbar shadow-inner"
                        />
                        <button
                            onClick={handleSend}
                            disabled={!input.trim() || isThinking}
                            className="absolute right-2 top-2 p-2 bg-gray-900 border border-gray-700 text-cyan-500 hover:bg-cyan-950 hover:text-cyan-400 hover:border-cyan-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed group/btn"
                        >
                            <Send size={16} className="group-hover/btn:translate-x-0.5 transition-transform" />
                        </button>
                    </div>
                    <div className="mt-2 flex justify-between items-center text-[10px] text-gray-600 mono-font uppercase tracking-widest">
                        <span>Workspace Context: Active</span>
                        <span>Model: {useWebSearch ? 'Gemini 3 Flash (Grounding)' : 'Gemini 3 Pro (Preview)'}</span>
                    </div>
                </div>
            </div>
        </div>
    );
};
