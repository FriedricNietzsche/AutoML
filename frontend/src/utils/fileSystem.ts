
export type FileType = 'file' | 'folder';

export interface FileNode {
  id: string;
  name: string;
  type: FileType;
  content?: string; // Only for files
  language?: string; // For syntax highlighting
  children?: FileNode[]; // Only for folders
  isOpen?: boolean; // For folder UI state
}

export const initialFileSystem: FileNode[] = [
  {
    id: '1',
    name: 'src',
    type: 'folder',
    isOpen: true,
    children: [
      {
        id: '2',
        name: 'App.tsx',
        type: 'file',
        language: 'typescript',
        content: `import React from 'react';

function App() {
  return (
    <div className="p-4">
      <h1>Hello Replit Clone!</h1>
    </div>
  );
}

export default App;`
      },
      {
        id: '3',
        name: 'index.css',
        type: 'file',
        language: 'css',
        content: `body {
  background: #fff;
  color: #333;
}`
      },
      {
        id: '4',
        name: 'utils',
        type: 'folder',
        isOpen: false,
        children: [
           {
             id: '5',
             name: 'helpers.ts',
             type: 'file',
             language: 'typescript',
             content: '// Helper functions go here'
           }
        ]
      }
    ]
  },
  {
    id: '6',
    name: 'package.json',
    type: 'file',
    language: 'json',
    content: `{
  "name": "demo-project",
  "version": "1.0.0"
}`
  },
  {
    id: '7',
    name: 'readme.md',
    type: 'file',
    language: 'markdown',
    content: '# Demo Project\n\nThis is a cool project.'
  }
];
