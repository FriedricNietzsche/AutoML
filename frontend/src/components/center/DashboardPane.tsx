import AIBuilderDashboard from './AIBuilderDashboard';
import type { FileSystemNode } from '../../lib/types';
import { BarChart2 } from 'lucide-react';

interface DashboardPaneProps {
  files: FileSystemNode[];
}

export default function DashboardPane({ files }: DashboardPaneProps) {
  // Use a key to force re-render if needed or just let the dashboard handle it
  // The Dashboard reads file content, so it should be reactive if files prop changes.
  
  return (
    <div className="h-full flex flex-col bg-replit-bg/30 backdrop-blur-xl">
       <div className="h-10 bg-replit-surface/60 backdrop-blur border-b border-replit-border/70 flex items-center px-4 justify-between shrink-0">
          <div className="flex items-center gap-2 text-replit-text font-medium text-sm">
             <BarChart2 size={16} />
             <span>Training Metrics</span>
          </div>
          <div className="text-xs text-replit-textMuted">
             Real-time Monitoring
          </div>
       </div>
       <div className="flex-1 overflow-hidden">
          <AIBuilderDashboard files={files} />
       </div>
    </div>
  );
}
