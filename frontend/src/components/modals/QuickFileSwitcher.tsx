import { useCallback, useEffect, useRef, useState } from 'react';
import { Search, File, X } from 'lucide-react';
import { mockFileTree, type FileTreeNode } from '../../lib/mockData';

interface QuickFileSwitcherProps {
  isOpen: boolean;
  onClose: () => void;
  onFileSelect: (node: FileTreeNode) => void;
}

function getAllFiles(nodes: FileTreeNode[], files: FileTreeNode[] = []): FileTreeNode[] {
  nodes.forEach((node) => {
    if (node.type === 'file') {
      files.push(node);
    }
    if (node.children) {
      getAllFiles(node.children, files);
    }
  });
  return files;
}

export default function QuickFileSwitcher({ isOpen, onClose, onFileSelect }: QuickFileSwitcherProps) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleClose = useCallback(() => {
    setQuery('');
    setSelectedIndex(0);
    onClose();
  }, [onClose]);

  const allFiles = getAllFiles(mockFileTree);
  const filteredFiles = allFiles.filter((file) =>
    file.name.toLowerCase().includes(query.toLowerCase()) ||
    file.path.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'Escape') {
        handleClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => Math.min(prev + 1, filteredFiles.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter' && filteredFiles[selectedIndex]) {
        e.preventDefault();
        onFileSelect(filteredFiles[selectedIndex]);
        handleClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredFiles, selectedIndex, handleClose, onFileSelect]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-start justify-center pt-32 z-50" onClick={handleClose}>
      <div
        className="w-full max-w-2xl bg-replit-surface rounded-xl border border-replit-border shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-replit-border">
          <Search className="w-5 h-5 text-replit-textMuted" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            placeholder="Search files..."
            className="flex-1 bg-transparent text-replit-text outline-none placeholder:text-replit-textMuted"
          />
          <button onClick={handleClose} className="p-1 hover:bg-replit-surfaceHover rounded transition-colors">
            <X className="w-4 h-4 text-replit-textMuted" />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-96 overflow-y-auto">
          {filteredFiles.length === 0 ? (
            <div className="px-4 py-8 text-center text-replit-textMuted text-sm">
              No files found
            </div>
          ) : (
            filteredFiles.map((file, index) => (
              <div
                key={file.id}
                onClick={() => {
                  onFileSelect(file);
                  handleClose();
                }}
                className={`
                  flex items-center gap-3 px-4 py-3 cursor-pointer transition-colors
                  ${index === selectedIndex ? 'bg-replit-accent/10' : 'hover:bg-replit-surfaceHover'}
                `}
              >
                <File className="w-4 h-4 text-replit-textMuted shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-replit-text truncate">{file.name}</div>
                  <div className="text-xs text-replit-textMuted truncate">{file.path}</div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-replit-border bg-replit-bg text-xs text-replit-textMuted flex items-center gap-4">
          <span>↑↓ Navigate</span>
          <span>Enter Select</span>
          <span>Esc Close</span>
        </div>
      </div>
    </div>
  );
}
