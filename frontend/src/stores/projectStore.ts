import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface ProjectState {
  projectId: string;
  prompt: string;
  taskType: string | null;
  datasetInfo: {
    name: string;
    rows: number;
    cols: number;
    source: string;
  } | null;
  setProjectId: (id: string) => void;
  setPrompt: (prompt: string) => void;
  setTaskType: (type: string) => void;
  setDatasetInfo: (info: ProjectState['datasetInfo']) => void;
  generateNewSession: () => string;
  reset: () => void;
}

// Generate a random session ID
const generateSessionId = (): string => {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
};

// Get initial project ID from env or generate new
const getInitialProjectId = (): string => {
  const envId = import.meta.env.VITE_PROJECT_ID;
  if (envId && envId !== 'demo') {
    return envId;
  }
  // Check localStorage for existing session
  const stored = localStorage.getItem('automl-project');
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      if (parsed.state?.projectId) {
        return parsed.state.projectId;
      }
    } catch {
      // ignore parse errors
    }
  }
  return generateSessionId();
};

export const useProjectStore = create<ProjectState>()(
  persist(
    (set, get) => ({
      projectId: getInitialProjectId(),
      prompt: '',
      taskType: null,
      datasetInfo: null,

      setProjectId: (id) => set({ projectId: id }),
      
      setPrompt: (prompt) => set({ prompt }),
      
      setTaskType: (type) => set({ taskType: type }),
      
      setDatasetInfo: (info) => set({ datasetInfo: info }),

      generateNewSession: () => {
        const newId = generateSessionId();
        set({ projectId: newId, prompt: '', taskType: null, datasetInfo: null });
        return newId;
      },

      reset: () => {
        const newId = generateSessionId();
        set({
          projectId: newId,
          prompt: '',
          taskType: null,
          datasetInfo: null,
        });
      },
    }),
    {
      name: 'automl-project',
      partialize: (state) => ({
        projectId: state.projectId,
        prompt: state.prompt,
        taskType: state.taskType,
      }),
    }
  )
);
