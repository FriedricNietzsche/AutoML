import { useState } from 'react';
import { ChevronRight, ChevronDown, Folder, FileCode, FileJson, FileText } from 'lucide-react';
import type { FileSystemNode } from '../../lib/types';

interface FilesPanelProps {
  onFileSelect: (node: FileSystemNode) => void;
  files: FileSystemNode[];
}

function FileIcon({ name }: { name: string }) {
  if (name.endsWith('.ts') || name.endsWith('.tsx')) {
    return <FileCode className="w-4 h-4 text-replit-accent" />;
  }
  if (name.endsWith('.js') || name.endsWith('.jsx')) {
    return <FileCode className="w-4 h-4 text-replit-warning" />;
  }
  if (name.endsWith('.json')) {
    return <FileJson className="w-4 h-4 text-replit-warning" />;
  }
  if (name.endsWith('.md')) {
    return <FileText className="w-4 h-4 text-replit-textMuted" />;
  }
  if (name.endsWith('.css')) {
    return <FileCode className="w-4 h-4 text-replit-accent" />;
  }
  return <FileText className="w-4 h-4 text-replit-textMuted" />;
}

interface FileTreeItemProps {
  node: FileSystemNode;
  depth: number;
  onToggle: (id: string) => void;
  onSelect: (node: FileSystemNode) => void;
}

function FileTreeItem({ node, depth, onToggle, onSelect }: FileTreeItemProps) {
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (node.type === 'folder') {
      onToggle(node.id);
    } else {
      onSelect(node);
    }
  };

  return (
    <div>
      <div
        onClick={handleClick}
        className="flex items-center gap-1.5 px-3 py-1 hover:bg-replit-surfaceHover cursor-pointer transition-colors group select-none border-l-2 border-transparent hover:border-replit-border"
        style={{ paddingLeft: `${depth * 12 + 12}px` }}
      >
        {node.type === 'folder' ? (
          <>
            <div className="text-replit-textMuted transition-transform duration-200">
               {node.isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            </div>
            <Folder size={14} className={`${node.isOpen ? 'text-blue-400 fill-blue-400/20' : 'text-slate-400'}`} />
          </>
        ) : (
          <>
            <div className="w-3" /> {/* Spacer for alignment */}
            <FileIcon name={node.name} />
          </>
        )}
        <span className="text-sm text-replit-text truncate leading-relaxed">{node.name}</span>
      </div>

      {node.type === 'folder' && node.isOpen && node.children && (
        <div className="animate-in slide-in-from-top-1 duration-200 fade-in">
          {node.children.map((child) => (
            <FileTreeItem
              key={child.id}
              node={child}
              depth={depth + 1}
              onToggle={onToggle}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function FilesPanel({ onFileSelect, files }: FilesPanelProps) {
  // Local state for expanded folders - ideally this should also sync up, but for now local is fine
  // Actually, 'files' prop has 'isOpen' property, but AppShell owns the state.
  // We need to support toggling folders.
  // BUT: The updated 'files' state in AppShell is "complex".
  // For this demo, let's keep it simple: we assume 'files' prop is the source of truth for structure,
  // but we might need toggle logic.
  // Wait, if node.isOpen comes from props, we need a callback to toggle it in AppShell.
  // However, I didn't verify if AppShell implements `onToggle`.
  // Looking at AppShell, it only has `updateFileContent`. It doesn't seem to have `toggleFolder`.
  // So: I will maintain a local set of open folder IDs for the UI state of the tree.
  
  const [openFolders, setOpenFolders] = useState<Set<string>>(new Set(['root', 'src', 'config', 'artifacts']));

  const handleToggle = (id: string) => {
    const newOpen = new Set(openFolders);
    if (newOpen.has(id)) {
      newOpen.delete(id);
    } else {
      newOpen.add(id);
    }
    setOpenFolders(newOpen);
  };

  // Helper to merge prop structure with local toggle state
  // We render the prop 'files', but override 'isOpen' with local state
  const renderTree = (nodes: FileSystemNode[], depth = 0) => {
    return nodes.map(node => (
      <FileTreeItem
        key={node.id}
        node={{ ...node, isOpen: openFolders.has(node.id) }} 
        depth={depth}
        onToggle={handleToggle}
        onSelect={onFileSelect}
      />
    ));
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto py-2">
        {files && files.length > 0 ? renderTree(files) : (
            <div className="p-4 text-xs text-replit-textMuted text-center">No files open</div>
        )}
      </div>
    </div>
  );
}
