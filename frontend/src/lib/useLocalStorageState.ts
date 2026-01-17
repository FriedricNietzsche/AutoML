import { useEffect, useRef, useState } from 'react';

type LocalStorageStateOptions = {
  /**
   * If true, initialize from defaultValue first (fast) and hydrate from localStorage in an effect.
   * This avoids blocking the first paint on large JSON payloads.
   */
  defer?: boolean;
};

export function useLocalStorageState<T>(
  key: string,
  defaultValue: T,
  options?: LocalStorageStateOptions
): [T, (value: T | ((prev: T) => T)) => void] {
  const defer = !!options?.defer;
  const hasHydratedFromStorageRef = useRef(!defer);

  const [state, setState] = useState<T>(() => {
    try {
      if (defer) return defaultValue;
      const item = window.localStorage.getItem(key);
      return item ? (JSON.parse(item) as T) : defaultValue;
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    if (!defer) return;
    try {
      const item = window.localStorage.getItem(key);
      if (item != null) {
        setState(JSON.parse(item) as T);
      }
    } catch {
      // ignore
    } finally {
      hasHydratedFromStorageRef.current = true;
    }
  }, [defer, key]);

  useEffect(() => {
    if (defer && !hasHydratedFromStorageRef.current) return;
    try {
      window.localStorage.setItem(key, JSON.stringify(state));
    } catch (error) {
      console.error('Failed to save to localStorage:', error);
    }
  }, [defer, key, state]);

  return [state, setState];
}
