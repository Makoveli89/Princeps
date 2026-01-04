import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { Agent, Workspace } from '../types';
import { Play, Terminal, Cpu } from 'lucide-react';

export const AgentConsole: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);

  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [selectedWorkspace, setSelectedWorkspace] = useState<string>('');
  const [prompt, setPrompt] = useState('');
  const [output, setOutput] = useState('');
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetch('/api/agents').then(r => r.json()).then(setAgents);
    fetch('/api/workspaces').then(r => r.json()).then(setWorkspaces);
  }, []);

  // Select first available options
  useEffect(() => {
    if (!selectedAgent && agents.length > 0) setSelectedAgent(agents[0].id);
    if (!selectedWorkspace && workspaces.length > 0) setSelectedWorkspace(workspaces[0].id);
  }, [agents, workspaces]);

  const handleRun = async () => {
    if (!prompt || !selectedAgent || !selectedWorkspace) return;

    setRunning(true);
    setOutput(''); // Clear previous output

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agentId: selectedAgent,
          workspaceId: selectedWorkspace,
          input: prompt
        })
      });

      const data = await res.json();
      if (data.status === 'success') {
         setOutput(data.output || "Task completed successfully, but returned no text.");
      } else {
         setOutput(`Error: ${JSON.stringify(data)}`);
      }
    } catch (e: any) {
      setOutput(`Failed to execute: ${e.message}`);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-140px)]">
      {/* Left Panel: Configuration */}
      <Card className="flex flex-col gap-6 lg:col-span-1 h-full">
        <div>
          <h2 className="text-xl font-serif font-bold text-gothic-text mb-4">Mission Control</h2>
          <p className="text-gothic-muted text-sm">Configure your agent run parameters.</p>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gothic-muted mb-1">Target Workspace</label>
            <select
              className="w-full bg-gothic-900 border border-gothic-700 rounded px-3 py-2 text-gothic-text focus:border-gothic-gold outline-none"
              value={selectedWorkspace}
              onChange={e => setSelectedWorkspace(e.target.value)}
            >
              {workspaces.map(ws => (
                <option key={ws.id} value={ws.id}>{ws.name}</option>
              ))}
              {workspaces.length === 0 && <option value="">No workspaces found</option>}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gothic-muted mb-1">Select Agent</label>
            <div className="space-y-2">
              {agents.map(agent => (
                <div
                  key={agent.id}
                  onClick={() => setSelectedAgent(agent.id)}
                  className={`p-3 rounded border cursor-pointer transition-colors flex items-center gap-3 ${
                    selectedAgent === agent.id
                      ? 'bg-gothic-800 border-gothic-gold'
                      : 'bg-gothic-900 border-gothic-700 hover:border-gothic-600'
                  }`}
                >
                  <Cpu className={`w-5 h-5 ${selectedAgent === agent.id ? 'text-gothic-gold' : 'text-gothic-muted'}`} />
                  <div>
                    <div className="font-medium text-sm">{agent.name}</div>
                    <div className="text-xs text-gothic-muted">{agent.role}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Right Panel: Execution & Output */}
      <div className="lg:col-span-2 flex flex-col gap-6 h-full">
        <Card className="flex-1 flex flex-col min-h-0">
          <div className="flex items-center gap-2 mb-2 text-gothic-muted text-sm font-mono uppercase">
            <Terminal size={14} /> Input
          </div>
          <textarea
            className="w-full flex-1 bg-gothic-900 border border-gothic-700 rounded p-4 text-gothic-text font-mono text-sm focus:border-gothic-gold outline-none resize-none mb-4"
            placeholder="Enter your instructions for the agent..."
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
          />
          <div className="flex justify-end">
            <Button onClick={handleRun} disabled={running || !prompt}>
              {running ? (
                <span className="flex items-center gap-2">Processing...</span>
              ) : (
                <span className="flex items-center gap-2"><Play size={16} /> Execute</span>
              )}
            </Button>
          </div>
        </Card>

        <Card className="flex-[2] flex flex-col min-h-0 bg-black/40">
          <div className="flex items-center gap-2 mb-2 text-gothic-muted text-sm font-mono uppercase">
             Output Stream
          </div>
          <div className="flex-1 bg-gothic-900/50 border border-gothic-700/50 rounded p-4 font-mono text-sm text-gothic-text overflow-auto whitespace-pre-wrap">
            {output || <span className="text-gothic-muted opacity-50">// Agent output will appear here...</span>}
          </div>
        </Card>
      </div>
    </div>
  );
};
