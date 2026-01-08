import React, { useState, useRef, KeyboardEvent } from 'react';
import { Send, Terminal } from 'lucide-react';

interface ChatInputProps {
    onSend: (text: string) => void;
    disabled: boolean;
    placeholder: string;
}

export const ChatInput = ({ onSend, disabled, placeholder }: ChatInputProps) => {
    const [input, setInput] = useState('');

    const handleSend = () => {
        if (!input.trim() || disabled) return;
        onSend(input);
        setInput('');
    };

    const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
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
                    placeholder={placeholder}
                    className="w-full bg-[#050505] border border-gray-800 text-gray-300 pl-12 pr-14 py-4 mono-font resize-none outline-none focus:border-cyan-700/50 transition-colors h-14 min-h-[56px] max-h-32 custom-scrollbar shadow-inner"
                />
                <button
                    onClick={handleSend}
                    disabled={!input.trim() || disabled}
                    className="absolute right-2 top-2 p-2 bg-gray-900 border border-gray-700 text-cyan-500 hover:bg-cyan-950 hover:text-cyan-400 hover:border-cyan-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed group/btn"
                >
                    <Send size={16} className="group-hover/btn:translate-x-0.5 transition-transform" />
                </button>
            </div>
            <div className="mt-2 flex justify-between items-center text-[10px] text-gray-600 mono-font uppercase tracking-widest">
                <span>Workspace Context: Active</span>
                <span>Model: {placeholder.includes('Web') ? 'Gemini 3 Flash (Grounding)' : 'Gemini 3 Pro (Preview)'}</span>
            </div>
        </div>
    );
};
