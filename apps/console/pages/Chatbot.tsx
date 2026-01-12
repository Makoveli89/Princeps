import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI } from '@google/genai';
import { Send, Bot, Sparkles, Globe, Terminal, Loader2 } from 'lucide-react';
import { Workspace, Message } from '../types';
import { ChatMessage } from '../components/ChatMessage';

export const Chatbot = ({ workspace }: { workspace: Workspace }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'init',
      role: 'model',
      text: `Princeps Neural Link established. Connected to workspace: ${workspace?.name || 'Unknown'}.\n\nReady for input. Toggle "Netrunner Mode" for real-time web access.`,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [useWebSearch, setUseWebSearch] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
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
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsThinking(true);

    try {
      // Fix: Use Vite env var
      const apiKey = import.meta.env.VITE_GEMINI_API_KEY || import.meta.env.VITE_GOOGLE_API_KEY;

      if (!apiKey) {
        throw new Error('API_KEY_MISSING');
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
        config: config,
      });

      const text = response.text || 'No response received from the neural net.';
      const groundingMetadata = response.candidates?.[0]?.groundingMetadata;

      const aiMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'model',
        text: text,
        groundingMetadata: groundingMetadata,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch (error: any) {
      console.error('Gemini Error:', error);

      let errorText = 'Neural Link Error: Connection terminated.';
      if (error.message === 'API_KEY_MISSING') {
        errorText =
          'CRITICAL FAILURE: API Key not found in environment variables.\n\nPlease set VITE_GEMINI_API_KEY in .env file.';
      } else {
        // Fallback for demo purposes if API call fails
        errorText = `Error accessing model: ${error.message}.\n\n(Mock Response): This appears to be a simulation. If I were fully connected, I would provide a detailed analysis of "${input}".`;
      }

      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'model',
        text: errorText,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
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
    <div className="animate-in fade-in mx-auto flex h-full max-w-6xl flex-col gap-6 duration-500">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-800 pb-4">
        <div>
          <h2 className="gothic-font text-glow flex items-center gap-3 text-3xl tracking-wide text-white">
            <Bot size={28} className="text-cyan-400" /> Neural Chat
          </h2>
          <p className="mono-font mt-1 text-sm text-gray-500">
            Direct Interface :: {useWebSearch ? 'WEB ACCESS ENABLED' : 'DEEP REASONING MODE'}
          </p>
        </div>

        {/* Mode Toggle */}
        <button
          onClick={() => setUseWebSearch(!useWebSearch)}
          className={`flex items-center gap-3 border px-4 py-2 transition-all duration-300 ${
            useWebSearch
              ? 'border-cyan-500 bg-cyan-950/30 text-cyan-400 shadow-[0_0_15px_rgba(0,243,255,0.2)]'
              : 'border-gray-700 bg-gray-900 text-gray-400 hover:border-gray-500'
          }`}
        >
          {useWebSearch ? <Globe size={16} className="animate-pulse" /> : <Sparkles size={16} />}
          <span className="text-xs font-bold uppercase tracking-widest">
            {useWebSearch ? 'Netrunner Mode (Search)' : 'Reasoning Mode (Pro)'}
          </span>
        </button>
      </div>

      {/* Chat History */}
      <div className="relative flex flex-1 flex-col overflow-hidden border border-gray-800 bg-[#030303] shadow-2xl">
        {/* Background Grid */}
        <div
          className="absolute inset-0 z-0 opacity-[0.03]"
          style={{
            backgroundImage:
              'linear-gradient(#333 1px, transparent 1px), linear-gradient(90deg, #333 1px, transparent 1px)',
            backgroundSize: '20px 20px',
          }}
        ></div>

        <div className="custom-scrollbar relative z-10 flex-1 space-y-6 overflow-y-auto p-6">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} msg={msg} />
          ))}

          {isThinking && (
            <div className="flex justify-start">
              <div className="flex max-w-[80%] flex-row gap-4">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-sm border border-cyan-800 bg-cyan-950/20 text-cyan-400">
                  <Bot size={16} />
                </div>
                <div className="flex items-center gap-2 rounded-sm border border-cyan-900/30 bg-cyan-950/5 p-4">
                  <Loader2 size={16} className="animate-spin text-cyan-500" />
                  <span className="mono-font animate-pulse text-xs text-cyan-600">
                    {useWebSearch ? 'SCANNING NETWORK...' : 'PROCESSING THOUGHT...'}
                  </span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-800 bg-[#020202] p-4">
          <div className="group relative focus-within:ring-1 focus-within:ring-cyan-900/50">
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-cyan-900/10 to-transparent opacity-0 transition-opacity group-focus-within:opacity-100"></div>
            <div className="pointer-events-none absolute left-4 top-4 text-gray-600">
              <Terminal size={16} />
            </div>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Enter prompt for ${useWebSearch ? 'Netrunner (Web)' : 'Princeps (Core)'}...`}
              className="mono-font custom-scrollbar h-14 max-h-32 min-h-[56px] w-full resize-none border border-gray-800 bg-[#050505] py-4 pl-12 pr-14 text-gray-300 shadow-inner outline-none transition-colors focus:border-cyan-700/50"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isThinking}
              className="group/btn absolute right-2 top-2 border border-gray-700 bg-gray-900 p-2 text-cyan-500 transition-all hover:border-cyan-600 hover:bg-cyan-950 hover:text-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Send size={16} className="transition-transform group-hover/btn:translate-x-0.5" />
            </button>
          </div>
          <div className="mono-font mt-2 flex items-center justify-between text-[10px] uppercase tracking-widest text-gray-600">
            <span>Workspace Context: Active</span>
            <span>
              Model: {useWebSearch ? 'Gemini 3 Flash (Grounding)' : 'Gemini 3 Pro (Preview)'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
