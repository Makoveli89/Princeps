import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { BarChart3, Cpu, Zap, ScrollText } from 'lucide-react';

export const Dashboard: React.FC = () => {
  const [stats, setStats] = useState({
    activeAgents: 0,
    tasksCompleted: 0,
    uptime: '0h',
    knowledgeNodes: 0
  });

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      setStats({
        activeAgents: 8,
        tasksCompleted: 142,
        uptime: '42h',
        knowledgeNodes: 8391
      });
    }, 500);
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-serif text-gothic-text mb-2">Command Center</h1>
        <p className="text-gothic-muted">System status and operational overview.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          label="Active Agents"
          value={stats.activeAgents.toString()}
          icon={<Cpu className="text-gothic-purple" />}
        />
        <StatCard
          label="Tasks Resolved"
          value={stats.tasksCompleted.toString()}
          icon={<Zap className="text-gothic-gold" />}
        />
        <StatCard
          label="Knowledge Nodes"
          value={stats.knowledgeNodes.toLocaleString()}
          icon={<ScrollText className="text-blue-400" />}
        />
        <StatCard
          label="System Uptime"
          value={stats.uptime}
          icon={<BarChart3 className="text-green-400" />}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <Card className="lg:col-span-2" glow>
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-serif font-semibold">Live Operations</h3>
            <Button size="sm" variant="secondary">View Log</Button>
          </div>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-gothic-900/50 rounded border border-gothic-700/50">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-gothic-gold animate-pulse" />
                  <span className="font-mono text-sm text-gothic-text">OP-{1000 + i}: Knowledge distillation in sector 7</span>
                </div>
                <span className="text-xs text-gothic-muted font-mono">00:0{i}:23</span>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-serif font-semibold mb-4">System Health</h3>
          <div className="space-y-6">
            <HealthBar label="Memory Usage" percent={45} color="bg-gothic-purple" />
            <HealthBar label="CPU Load" percent={72} color="bg-gothic-gold" />
            <HealthBar label="Network IO" percent={28} color="bg-blue-500" />
            <HealthBar label="Database" percent={12} color="bg-green-500" />
          </div>
        </Card>
      </div>
    </div>
  );
};

const StatCard: React.FC<{ label: string; value: string; icon: React.ReactNode }> = ({ label, value, icon }) => (
  <Card className="flex items-center justify-between">
    <div>
      <p className="text-gothic-muted text-sm font-medium mb-1">{label}</p>
      <p className="text-2xl font-serif font-bold text-gothic-text">{value}</p>
    </div>
    <div className="p-3 bg-gothic-900 rounded-lg border border-gothic-700">
      {icon}
    </div>
  </Card>
);

const HealthBar: React.FC<{ label: string; percent: number; color: string }> = ({ label, percent, color }) => (
  <div>
    <div className="flex justify-between text-sm mb-1">
      <span className="text-gothic-muted">{label}</span>
      <span className="text-gothic-text font-mono">{percent}%</span>
    </div>
    <div className="h-2 w-full bg-gothic-900 rounded-full overflow-hidden">
      <div
        className={`h-full ${color} transition-all duration-500`}
        style={{ width: `${percent}%` }}
      />
    </div>
  </div>
);
