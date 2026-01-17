export const resolveHttpBase = (wsBase?: string) => {
  const envBase =
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_BASE) ||
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_BACKEND_HTTP_BASE);
  if (envBase) return envBase as string;

  if (wsBase?.startsWith('wss://')) return wsBase.replace(/^wss/, 'https');
  if (wsBase?.startsWith('ws://')) return wsBase.replace(/^ws/, 'http');

  if (typeof window !== 'undefined') {
    const proto = window.location.protocol === 'https:' ? 'https' : 'http';
    return `${proto}://${window.location.hostname}:8000`;
  }
  return 'http://localhost:8000';
};

export const joinUrl = (base: string, path: string) => {
  const trimmedBase = base.endsWith('/') ? base.slice(0, -1) : base;
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${trimmedBase}${normalizedPath}`;
};
