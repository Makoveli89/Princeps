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

export interface RunRequest {
  agentId: string;
  input: string;
  workspaceId: string;
}

export interface RunResponse {
  status: string;
  run_id: string;
  output: string;
  full_result: any;
}

export interface Stats {
  activeAgents: number;
  tasksCompleted: number;
  uptime: string;
  knowledgeNodes: number;
}
