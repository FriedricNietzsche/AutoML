import { useEffect, useState } from 'react';

interface ThemedBackgroundProps {
  isDark: boolean;
  interactive?: boolean;
}

export default function ThemedBackground({ isDark, interactive = true }: ThemedBackgroundProps) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!interactive) return;

    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, [interactive]);

  const alpha = isDark ? 0.22 : 0.16;

  return (
    <div className="fixed inset-0 pointer-events-none -z-10">
      {/* Mouse-follow accent glow */}
      {interactive ? (
        <div
          className={`absolute inset-0 transition-opacity duration-500 ${isDark ? 'opacity-35' : 'opacity-25'}`}
          style={{
            background: `radial-gradient(circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(var(--replit-accent-rgb), ${alpha}), transparent 55%)`,
          }}
        />
      ) : (
        <div
          className={`absolute inset-0 transition-opacity duration-500 ${isDark ? 'opacity-30' : 'opacity-20'}`}
          style={{
            background: `radial-gradient(circle at 40% 20%, rgba(var(--replit-accent-rgb), ${alpha}), transparent 55%)`,
          }}
        />
      )}

      {/* Floating orbs */}
      <div
        className={`absolute top-16 left-16 w-72 h-72 rounded-full mix-blend-multiply filter blur-3xl animate-pulse ${
          isDark ? 'opacity-25' : 'opacity-12'
        }`}
        style={{ backgroundColor: 'rgba(var(--replit-accent-rgb), 0.55)' }}
      />
      <div
        className={`absolute top-32 right-10 w-[28rem] h-[28rem] rounded-full mix-blend-multiply filter blur-3xl animate-pulse ${
          isDark ? 'opacity-22' : 'opacity-10'
        }`}
        style={{ backgroundColor: 'rgba(14, 165, 233, 0.55)', animationDelay: '1s' }}
      />
      <div
        className={`absolute -bottom-10 left-1/2 w-96 h-96 rounded-full mix-blend-multiply filter blur-3xl animate-pulse ${
          isDark ? 'opacity-22' : 'opacity-10'
        }`}
        style={{ backgroundColor: 'rgba(99, 102, 241, 0.5)', animationDelay: '2s' }}
      />

      {/* Subtle vignette */}
      <div
        className={`absolute inset-0 ${isDark ? 'opacity-80' : 'opacity-35'}`}
        style={{
          background:
            'radial-gradient(1200px 800px at 50% 20%, rgba(255,255,255,0.06), transparent 55%), radial-gradient(900px 700px at 50% 120%, rgba(0,0,0,0.18), transparent 55%)',
        }}
      />
    </div>
  );
}
