import { Workspace, Agent, Run } from './types';

export const MOCK_WORKSPACES: Workspace[] = [
  {
    id: 'ws-001',
    name: 'Sanctum Archivum',
    description: 'Knowledge base for ancient artifacts and lore processing.',
    status: 'active',
    agentCount: 3,
    lastActive: '2024-03-10T14:30:00Z',
  },
  {
    id: 'ws-002',
    name: 'Obsidian Strategos',
    description: 'High-frequency trading and market strategy synthesis.',
    status: 'active',
    agentCount: 5,
    lastActive: '2024-03-11T09:15:00Z',
  },
  {
    id: 'ws-003',
    name: 'Aetherial Nexus',
    description: 'Experimental communication protocols and linguistic analysis.',
    status: 'error',
    agentCount: 1,
    lastActive: '2024-03-08T18:45:00Z',
  },
];

export const MOCK_AGENTS: Agent[] = [
  {
    id: 'ag-001',
    name: 'Scribe-Alpha',
    role: 'Archivist',
    status: 'idle',
    capabilities: ['OCR', 'Translation', 'Summarization'],
  },
  {
    id: 'ag-002',
    name: 'Strategos-Prime',
    role: 'Planner',
    status: 'running',
    capabilities: ['Reasoning', 'Strategy', 'Optimization'],
  },
];
