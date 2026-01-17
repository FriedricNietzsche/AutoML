import { X } from 'lucide-react';

export type TabType = 'preview' | 'dashboard' | 'publishing' | 'console' | 'file';

export interface Tab {
  id: string;
  type: TabType;
  title: string;
  filePath?: string;
  closable: boolean;
}

interface WorkspaceTabsProps {
  tabs: Tab[];
  activeTabId: string;
  onTabClick: (tabId: string) => void;
  onTabClose: (tabId: string) => void;
}

export default function WorkspaceTabs({
  tabs,
  activeTabId,
  onTabClick,
  onTabClose,
}: WorkspaceTabsProps) {
  return (
    <div className="h-10 bg-replit-surface/60 backdrop-blur-xl border-b border-replit-border/70 flex items-center px-2 gap-1 overflow-x-auto shrink-0">
      {tabs.map((tab) => {
        const isActive = tab.id === activeTabId;
        
        return (
          <div
            key={tab.id}
            onClick={() => onTabClick(tab.id)}
            className={`
              group flex items-center gap-2 px-3 py-1.5 rounded-md text-sm cursor-pointer transition-colors shrink-0
              ${isActive 
                ? 'bg-replit-surface/85 text-replit-text border border-replit-border/60 shadow-sm' 
                : 'text-replit-textMuted hover:text-replit-text hover:bg-replit-surface/70'
              }
            `}
          >
            <span>{tab.title}</span>
            
            {tab.closable && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onTabClose(tab.id);
                }}
                className="opacity-0 group-hover:opacity-100 hover:bg-replit-surface/70 rounded p-0.5 transition-opacity"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
