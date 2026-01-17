/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Tokenized palette (CSS variables) so we can theme-switch without rewriting components.
        replit: {
          bg: 'rgb(var(--replit-bg) / <alpha-value>)',
          surface: 'rgb(var(--replit-surface) / <alpha-value>)',
          surfaceHover: 'rgb(var(--replit-surface-hover) / <alpha-value>)',
          border: 'rgb(var(--replit-border) / <alpha-value>)',
          borderSubtle: 'rgb(var(--replit-border-subtle) / <alpha-value>)',
          text: 'rgb(var(--replit-text) / <alpha-value>)',
          textMuted: 'rgb(var(--replit-text-muted) / <alpha-value>)',
          accent: 'rgb(var(--replit-accent) / <alpha-value>)',
          accentHover: 'rgb(var(--replit-accent-hover) / <alpha-value>)',
          success: 'rgb(var(--replit-success) / <alpha-value>)',
          warning: 'rgb(var(--replit-warning) / <alpha-value>)',
        }
      },
    },
  },
  plugins: [],
}