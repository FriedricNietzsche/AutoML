'use client';

import * as React from 'react';
import { useTheme } from 'next-themes';
import { Moon, Sun } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

export default function ThemeToggle({ className }: { className?: string }) {
  const { theme, setTheme, systemTheme } = useTheme();
  void systemTheme;
  const isDark = theme === 'dark';

  return (
    <motion.button
      type="button"
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      whileTap={{ scale: 0.96 }}
      className={cn(
        'glass rounded-xl px-2.5 py-2 text-sm',
        'grid place-items-center border border-border/60',
        'hover:border-border transition-colors',
        className
      )}
      aria-label="Toggle theme"
    >
      <span className="relative h-5 w-5">
        <motion.span
          initial={false}
          animate={{ opacity: isDark ? 0 : 1, rotate: isDark ? -40 : 0, scale: isDark ? 0.8 : 1 }}
          transition={{ type: 'spring', stiffness: 260, damping: 18 }}
          className="absolute inset-0 grid place-items-center"
        >
          <Sun className="h-4 w-4" />
        </motion.span>
        <motion.span
          initial={false}
          animate={{ opacity: isDark ? 1 : 0, rotate: isDark ? 0 : 40, scale: isDark ? 1 : 0.8 }}
          transition={{ type: 'spring', stiffness: 260, damping: 18 }}
          className="absolute inset-0 grid place-items-center"
        >
          <Moon className="h-4 w-4" />
        </motion.span>
      </span>
    </motion.button>
  );
}
