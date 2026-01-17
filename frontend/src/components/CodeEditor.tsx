import React from 'react';
import Editor from '@monaco-editor/react';
import type { FileNode } from '../utils/fileSystem';

interface CodeEditorProps {
  activeFile: FileNode | null;
  onChange: (value: string | undefined) => void;
  theme: 'light' | 'dark';
}

const CodeEditor: React.FC<CodeEditorProps> = ({ activeFile, onChange, theme }) => {
  if (!activeFile) {
    return (
      <div className="h-full flex items-center justify-center text-slate-400 bg-sand-50 dark:bg-midnight-950">
        <div className="text-center">
          <p>Select a file to start editing</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full overflow-hidden">
      <Editor
        height="100%"
        language={activeFile.language || 'typescript'}
        value={activeFile.content}
        theme={theme === 'dark' ? 'vs-dark' : 'light'}
        onChange={onChange}
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          wordWrap: 'on',
          padding: { top: 16 },
          fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
        }}
      />
    </div>
  );
};

export default CodeEditor;