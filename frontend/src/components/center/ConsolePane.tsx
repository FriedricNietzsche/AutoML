import { Terminal, Trash2 } from 'lucide-react';
import { useMemo, useState } from 'react';

interface ConsolePaneProps {
  logsText?: string;
}

export default function ConsolePane({ logsText }: ConsolePaneProps) {
  const [input, setInput] = useState('');

  const lines = useMemo(() => {
    const raw = (logsText || '').trimEnd();
    if (!raw) return [] as string[];
    return raw.split('\n').slice(-500);
  }, [logsText]);

  return (
    <div className="h-full flex flex-col bg-replit-bg">
      {/* Console Header */}
      <div className="h-10 bg-replit-surface border-b border-replit-border flex items-center justify-between px-3 shrink-0">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-replit-textMuted" />
          <span className="text-sm text-replit-text font-medium">Console</span>
        </div>
        <button
          onClick={() => setInput('')}
          className="p-1.5 hover:bg-replit-surfaceHover rounded transition-colors"
          title="Clear input"
        >
          <Trash2 className="w-4 h-4 text-replit-textMuted" />
        </button>
      </div>

      {/* Console Output */}
      <div className="flex-1 overflow-y-auto font-mono text-xs p-4 space-y-1">
        {lines.length > 0 ? (
          lines.map((line, idx) => (
            <div key={idx} className="text-replit-text whitespace-pre-wrap break-words">
              {line}
            </div>
          ))
        ) : (
          <div className="text-replit-textMuted">No logs yet.</div>
        )}
        
        {/* Input Line */}
        <div className="flex gap-3 mt-4">
          <span className="text-replit-textMuted shrink-0">{'>'}</span>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 bg-transparent outline-none text-replit-text"
            placeholder="Type a command..."
          />
        </div>
      </div>
    </div>
  );
}
