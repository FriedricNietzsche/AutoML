import React from 'react';
import { 
  Folder, 
  FileCode, 
  FileJson, 
  ChevronRight, 
  ChevronDown, 
  FileText 
} from 'lucide-react';
import type { FileNode } from '../utils/fileSystem';
import clsx from 'clsx';

interface FileIconProps {
  name: string;
  type: 'file' | 'folder';
  isOpen?: boolean;
}

const FileIcon: React.FC<FileIconProps> = ({ name, type, isOpen }) => {
  if (type === 'folder') {
    return isOpen ? (
      <Folder className="w-4 h-4 text-sand-600 dark:text-sand-400 fill-sand-600 dark:fill-sand-400" />
    ) : (
      <Folder className="w-4 h-4 text-sand-600 dark:text-sand-400" />
    );
  }

  if (name.endsWith('.tsx') || name.endsWith('.ts')) {
    return <FileCode className="w-4 h-4 text-blue-500" />;
  }
  if (name.endsWith('.css')) {
    return <FileCode className="w-4 h-4 text-sky-400" />;
  }
  if (name.endsWith('.json')) {
    return <FileJson className="w-4 h-4 text-yellow-500" />;
  }
  return <FileText className="w-4 h-4 text-slate-500" />;
};

interface FileExplorerProps {
  nodes: FileNode[];
  activeFileId: string | null;
  onSelectFile: (node: FileNode) => void;
  onToggleFolder: (id: string) => void;
  depth?: number;
}

const FileExplorer: React.FC<FileExplorerProps> = ({ 
  nodes, 
  activeFileId, 
  onSelectFile, 
  onToggleFolder,
  depth = 0
}) => {
  return (
    <div className="select-none">
      {nodes.map((node) => (
        <div key={node.id}>
          <div
            className={clsx(
              "flex items-center gap-1.5 py-1 px-2 cursor-pointer hover:bg-sand-200 dark:hover:bg-midnight-800 transition-colors",
              activeFileId === node.id && "bg-sand-200 dark:bg-midnight-800 border-l-2 border-sand-500"
            )}
            style={{ paddingLeft: `${depth * 12 + 8}px` }}
            onClick={(e) => {
              e.stopPropagation();
              if (node.type === 'folder') {
                onToggleFolder(node.id);
              } else {
                onSelectFile(node);
              }
            }}
          >
            <span className="opacity-70">
              {node.type === 'folder' && (
                node.isOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />
              )}
              {node.type === 'file' && <div className="w-3" />}
            </span>
            
            <FileIcon name={node.name} type={node.type} isOpen={node.isOpen} />
            
            <span className="text-sm text-slate-700 dark:text-slate-300 truncate">
              {node.name}
            </span>
          </div>

          {node.type === 'folder' && node.isOpen && node.children && (
            <FileExplorer
              nodes={node.children}
              activeFileId={activeFileId}
              onSelectFile={onSelectFile}
              onToggleFolder={onToggleFolder}
              depth={depth + 1}
            />
          )}
        </div>
      ))}
    </div>
  );
};

export default FileExplorer;