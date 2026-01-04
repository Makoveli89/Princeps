export interface Workspace {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'archived' | 'error';
  agentCount: number;
  lastActive: string;
}

export interface Agent {
  id: string;
  name: string;
  role: string;
  status: 'idle' | 'running' | 'failed';
  capabilities: string[];
}

export interface Run {
  id: string;
  agentId: string;
  taskId: string;
  status: 'pending' | 'running' | 'success' | 'failed';
  startedAt: string;
  completedAt?: string;
  input: string;
  output?: string;
}
