export interface PipelineNode {
  id: string;
  label: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number; // 0-100
  logs: string[];
}

export interface PipelineState {
  nodes: PipelineNode[];
  isRunning: boolean;
  startTime?: number;
}

export type FileType = 'file' | 'folder';

export interface FileSystemNode {
  id: string;
  name: string;
  type: FileType;
  content?: string;
  updatedAt: number;
  children?: FileSystemNode[];
  path: string;
  isOpen?: boolean; // For folders
}
