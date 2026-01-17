import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { applyThemeToDocument } from './lib/theme'

// Prevent a flash of incorrect theme on hard refresh.
try {
  const raw = window.localStorage.getItem('autoai.theme');
  const parsed = raw ? (JSON.parse(raw) as unknown) : null;
  const theme = parsed === 'midnight' ? 'midnight' : 'light';
  applyThemeToDocument(theme);
} catch {
  applyThemeToDocument('light');
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
