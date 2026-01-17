/* eslint-disable react-refresh/only-export-components */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

export type RoutePath = '/' | '/workspace';

interface RouterContextValue {
  pathname: string;
  navigate: (to: RoutePath, options?: { replace?: boolean }) => void;
}

const RouterContext = createContext<RouterContextValue | null>(null);

function getPathname(): string {
  // Keep it extremely simple: only path routing.
  return window.location.pathname || '/';
}

export function RouterProvider({ children }: { children: React.ReactNode }) {
  const [pathname, setPathname] = useState<string>(() => getPathname());

  useEffect(() => {
    const onPopState = () => setPathname(getPathname());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const navigate = useCallback((to: RoutePath, options?: { replace?: boolean }) => {
    const next = to;
    if (options?.replace) {
      window.history.replaceState({}, '', next);
    } else {
      window.history.pushState({}, '', next);
    }
    setPathname(next);
  }, []);

  const value = useMemo(() => ({ pathname, navigate }), [pathname, navigate]);

  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>;
}

export function useRouter() {
  const ctx = useContext(RouterContext);
  if (!ctx) throw new Error('useRouter must be used within RouterProvider');
  return ctx;
}

export function Route({ path, children }: { path: RoutePath; children: React.ReactNode }) {
  const { pathname } = useRouter();
  if (pathname !== path) return null;
  return <>{children}</>;
}

export function Redirect({ to }: { to: RoutePath }) {
  const { navigate } = useRouter();
  useEffect(() => {
    navigate(to, { replace: true });
  }, [navigate, to]);
  return null;
}
