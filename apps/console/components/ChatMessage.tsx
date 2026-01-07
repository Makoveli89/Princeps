import React, { memo } from 'react';
import { User, Bot, Globe, ExternalLink } from 'lucide-react';
import { Message } from '../types';

interface ChatMessageProps {
    msg: Message;
}

export const ChatMessage = memo(({ msg }: ChatMessageProps) => {
    return (
        <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
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
    );
}, (prevProps, nextProps) => {
    // Custom comparison to ensure strict equality check if needed, though default shallow compare is usually enough for objects if references are stable.
    // However, since we are iterating over a list where objects might be recreated if not careful, but here `msg` comes from state.
    // In Chatbot.tsx, `messages` state is updated by appending. Existing message objects should preserve referential identity.
    return prevProps.msg === nextProps.msg;
});

ChatMessage.displayName = 'ChatMessage';
