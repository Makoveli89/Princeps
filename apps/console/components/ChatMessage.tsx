import React, { memo } from 'react';
import { User, Bot, Globe, ExternalLink } from 'lucide-react';
import { Message } from '../types';

interface ChatMessageProps {
  msg: Message;
}

export const ChatMessage = memo(
  ({ msg }: ChatMessageProps) => {
    return (
      <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
        <div
          className={`flex max-w-[80%] gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
        >
          {/* Avatar */}
          <div
            className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-sm border ${
              msg.role === 'user'
                ? 'border-red-800 bg-red-950/20 text-red-500'
                : 'border-cyan-800 bg-cyan-950/20 text-cyan-400'
            }`}
          >
            {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
          </div>

          {/* Content */}
          <div className="flex flex-col gap-2">
            <div
              className={`rounded-sm border p-4 backdrop-blur-sm ${
                msg.role === 'user'
                  ? 'border-red-900/50 bg-red-950/10 text-gray-200'
                  : 'border-cyan-900/50 bg-cyan-950/10 text-cyan-50'
              }`}
            >
              <p className="mono-font whitespace-pre-wrap text-sm leading-relaxed">{msg.text}</p>
            </div>

            {/* Grounding / Sources */}
            {msg.groundingMetadata?.groundingChunks && (
              <div className="animate-in slide-in-from-top-2 rounded-sm border border-gray-800 bg-[#050505] p-3">
                <div className="mb-2 flex items-center gap-2 text-[10px] uppercase tracking-widest text-gray-500">
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
                        className="flex items-center gap-1 border border-gray-700 bg-gray-900 px-2 py-1 text-[10px] text-cyan-400 transition-colors hover:border-cyan-500 hover:text-cyan-300"
                      >
                        <ExternalLink size={10} />
                        {chunk.web.title || new URL(chunk.web.uri).hostname}
                      </a>
                    ) : null,
                  )}
                </div>
              </div>
            )}

            <span className="font-mono text-[10px] text-gray-600">
              {msg.timestamp.toLocaleTimeString()}
            </span>
          </div>
        </div>
      </div>
    );
  },
  (prevProps, nextProps) => {
    // Custom comparison to ensure strict equality check if needed, though default shallow compare is usually enough for objects if references are stable.
    // However, since we are iterating over a list where objects might be recreated if not careful, but here `msg` comes from state.
    // In Chatbot.tsx, `messages` state is updated by appending. Existing message objects should preserve referential identity.
    return prevProps.msg === nextProps.msg;
  },
);

ChatMessage.displayName = 'ChatMessage';
