import React, { useState, useEffect } from 'react';
import useSWR from 'swr';
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { RunTask } from './pages/RunTask';
import { IngestData } from './pages/IngestData';
import { Workspaces } from './pages/Workspaces';
import { RunsAndLogs } from './pages/RunsAndLogs';
import { KnowledgeSearch } from './pages/KnowledgeSearch';
import { ReportsAndGym } from './pages/ReportsAndGym';
import { SystemHealth } from './pages/SystemHealth';
import { Chatbot } from './pages/Chatbot';
import { INITIAL_WORKSPACES } from './constants';
import { Workspace } from './types';
import { fetcher } from './lib/fetcher';

const App = () => {
  const { data: workspacesData, error, isLoading } = useSWR<Workspace[]>('/api/workspaces', fetcher);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | null>(null);

  const workspaces = workspacesData || INITIAL_WORKSPACES;

  useEffect(() => {
    if (workspaces && workspaces.length > 0 && !activeWorkspaceId) {
      setActiveWorkspaceId(workspaces[0].id);
    }
  }, [workspaces, activeWorkspaceId]);

  const activeWorkspace = workspaces.find(w => w.id === activeWorkspaceId) || workspaces[0];

  if (isLoading) {
     return <div className="min-h-screen bg-[#050505] flex items-center justify-center text-gray-500 font-mono">INITIALIZING SYSTEM...</div>;
  }

  // Handling case where no workspace exists yet
  if (!activeWorkspace && !isLoading && workspaces.length === 0) {
      // Create default if none? Or show empty state?
      // For now let's redirect to workspace creation or show a simplified layout
      // Or just render Workspaces page forced
      return (
        <HashRouter>
            <div className="min-h-screen bg-[#050505] text-white p-10">
                <Workspaces workspaces={[]} activeId={null} onChange={() => {}} />
            </div>
        </HashRouter>
      )
  }

  return (
    <HashRouter>
      <Layout
        workspaces={workspaces}
        activeWorkspaceId={activeWorkspaceId || ''}
        onWorkspaceChange={setActiveWorkspaceId}
      >
        <Routes>
          <Route path="/" element={<Dashboard workspace={activeWorkspace} />} />
          <Route path="/run" element={<RunTask workspace={activeWorkspace} />} />
          <Route path="/chat" element={<Chatbot workspace={activeWorkspace} />} />
          <Route path="/ingest" element={<IngestData workspace={activeWorkspace} />} />
          <Route path="/search" element={<KnowledgeSearch workspace={activeWorkspace} />} />
          <Route path="/logs" element={<RunsAndLogs workspace={activeWorkspace} />} />
          <Route path="/gym" element={<ReportsAndGym workspace={activeWorkspace} />} />
          <Route path="/workspaces" element={<Workspaces workspaces={workspaces} activeId={activeWorkspaceId} onChange={setActiveWorkspaceId} />} />
          <Route path="/settings" element={<SystemHealth workspace={activeWorkspace} />} />
          <Route path="*" element={<div className="p-10 text-gray-500 font-mono text-center">404 - SECTOR NOT FOUND</div>} />
        </Routes>
      </Layout>
    </HashRouter>
  );
};

export default App;