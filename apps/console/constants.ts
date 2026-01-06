import { Workspace, AgentRun, MetricPoint, SearchResult, GymResult } from './types';

// Aesthetic Colors
export const COLORS = {
  void: '#020202', // Darker void
  neonBlue: '#00f3ff',
  bloodRed: '#ff003c',
  grimGrey: '#1a1a1a',
  textDim: '#6b7280',
  textBright: '#f3f4f6',
};

// Initial State (Empty) - Data should be fetched from API
export const INITIAL_WORKSPACES: Workspace[] = [];
export const INITIAL_METRICS: MetricPoint[] = [];
export const INITIAL_RUNS: AgentRun[] = [];
export const INITIAL_SEARCH_RESULTS: SearchResult[] = [];
export const INITIAL_GYM_RESULTS: GymResult[] = [];
