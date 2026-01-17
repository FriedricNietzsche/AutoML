import { useEffect } from 'react';
import { useLocalStorageState } from './useLocalStorageState';

export type ThemeMode = 'light' | 'midnight';
type StoredThemeMode = ThemeMode | 'warm-sand';

const STORAGE_KEY = 'autoai.theme';

export function applyThemeToDocument(theme: ThemeMode) {
  const root = document.documentElement;
  const isDark = theme === 'midnight';
  root.classList.toggle('dark', isDark);
  root.dataset.theme = theme;
  root.style.colorScheme = isDark ? 'dark' : 'light';
  root.style.backgroundColor = isDark ? '#0a0f1c' : '#f8fafc';
}

function normalizeTheme(value: StoredThemeMode | string): ThemeMode {
  if (value === 'midnight') return 'midnight';
  return 'light';
}

export function useTheme() {
  const [storedTheme, setStoredTheme] = useLocalStorageState<StoredThemeMode>(STORAGE_KEY, 'light');
  const theme = normalizeTheme(storedTheme);

  useEffect(() => {
    applyThemeToDocument(theme);
  }, [theme]);

  // Legacy migration: older sessions stored the light theme as "warm-sand".
  useEffect(() => {
    if (storedTheme === 'warm-sand') {
      setStoredTheme('light');
    }
  }, [storedTheme, setStoredTheme]);

  const toggleTheme = () => {
    setStoredTheme((prev) => (normalizeTheme(prev) === 'midnight' ? 'light' : 'midnight'));
  };

  const setTheme = (next: ThemeMode | ((prev: ThemeMode) => ThemeMode)) => {
    if (typeof next === 'function') {
      setStoredTheme((prev) => next(normalizeTheme(prev)));
      return;
    }
    setStoredTheme(next);
  };

  return { theme, setTheme, toggleTheme };
}
