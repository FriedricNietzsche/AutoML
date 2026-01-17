export type BuildStatus = 'building' | 'ready';

export type ChatRole = 'user' | 'ai';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  at: string;
}

export interface BuildSession {
  id: string;
  modelName: string;
  goalPrompt: string;
  kaggleLink?: string;
  datasetLinks?: string[];
  lastUserMessage?: string;
  lastUserMessageAt?: string;
  chatHistory?: ChatMessage[];
  aiThinking?: boolean;
  createdAt: string;
  status: BuildStatus;
}

const STORAGE_KEY = 'autoai.buildSession.current';

export function createBuildSession(input: {
  modelName: string;
  goalPrompt: string;
  kaggleLink?: string;
  datasetLinks?: string[];
}): BuildSession {
  const normalizedDatasets = (input.datasetLinks ?? [])
    .map((s) => s.trim())
    .filter(Boolean);

  const kaggle = input.kaggleLink?.trim() || undefined;
  const datasetLinks = normalizedDatasets.length > 0 ? normalizedDatasets : kaggle ? [kaggle] : undefined;

  return {
    id: `sess_${Math.random().toString(16).slice(2)}_${Date.now()}`,
    modelName: input.modelName.trim() || 'Untitled Model',
    goalPrompt: input.goalPrompt.trim(),
    kaggleLink: datasetLinks?.[0] ?? kaggle,
    datasetLinks,
    chatHistory: [],
    aiThinking: false,
    createdAt: new Date().toISOString(),
    status: 'building',
  };
}

export function getCurrentSession(): BuildSession | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as BuildSession;
  } catch {
    return null;
  }
}

export function setCurrentSession(session: BuildSession | null) {
  try {
    if (!session) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  } catch {
    // ignore
  }
}

export function updateCurrentSession(patch: Partial<BuildSession>) {
  const prev = getCurrentSession();
  if (!prev) return;
  setCurrentSession({ ...prev, ...patch });
}

export function isValidKaggleDatasetLink(link: string): boolean {
  const trimmed = link.trim();
  if (!trimmed) return false;
  return /(^|https?:\/\/)(www\.)?kaggle\.com\/datasets\/.+/i.test(trimmed);
}
