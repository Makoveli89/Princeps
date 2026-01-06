export interface Workspace {
  id: string;
  name: string;
  description?: string;
  status?: string;
  docCount: number;
  chunkCount: number;
  runCount: number;
  agentCount?: number;
  lastActive?: string;
}

export enum AgentType {
  PLANNER = 'PlannerAgent',
  EXECUTOR = 'ExecutorAgent',
  RETRIEVER = 'RetrieverAgent',
  CRITIC = 'CriticAgent',
  SCRIBE = 'Scribe', // Added based on backend
  TRANSIENT = 'TransientScribe'
}

export enum RunStatus {
  SUCCESS = 'SUCCESS',
  FAILURE = 'FAILURE',
  RUNNING = 'RUNNING'
}

export interface AgentRun {
  run_id: string;
  agent: string; // Changed to string to accommodate dynamic backend agents
  timestamp: string;
  status: string; // Changed to string to match backend DTO
  input_preview: string;
  output_preview: string;
  duration_ms: number;
  workspace_id: string;
  logs?: string[];
  payload?: any;
}

export interface MetricPoint {
  time: string;
  success: number;
  failure: number;
}

export interface SearchResult {
  id: string;
  score: number;
  content: string;
  source: string;
  chunk_index: number;
}

export interface GymResult {
  test_id: string;
  name: string;
  status: 'PASS' | 'FAIL';
  latency_ms: number;
  tokens: number;
  timestamp: string;
}

export interface SystemHealth {
  dbConnected: boolean;
  pgVectorEnabled: boolean;
  activeWorkspaceId: string | null;
  envVars: {
    DATABASE_URL: boolean;
    OPENAI_API_KEY: boolean;
    ANTHROPIC_API_KEY: boolean;
    GEMINI_API_KEY: boolean;
  };
}

export interface Message {
    id: string;
    role: 'user' | 'model';
    text: string;
    groundingMetadata?: any;
    timestamp: Date;
}
