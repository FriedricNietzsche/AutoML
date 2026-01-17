import { useReducedMotion } from 'framer-motion';

export default function MatrixScreenLoader({
  label,
  variant = 'fullscreen',
  className,
}: {
  label?: string;
  variant?: 'fullscreen' | 'inline';
  className?: string;
}) {
  const reduceMotion = useReducedMotion();

  const cells = Array.from({ length: 9 }).map((_, i) => i);
  const colors = ['bg-replit-accent/60', 'bg-replit-success/50', 'bg-replit-warning/50'];

  const grid = (
    <div className={"flex flex-col items-center gap-2"}>
      <div
        className={
          reduceMotion
            ? 'grid grid-cols-3 gap-1 opacity-80'
            : 'grid grid-cols-3 gap-1 opacity-90'
        }
        aria-hidden
      >
        {cells.map((i) => (
          <div
            key={i}
            className={
              `h-2.5 w-2.5 rounded-[3px] ${colors[i % colors.length]} ` +
              (reduceMotion ? '' : 'animate-[matrixPulse_900ms_ease-in-out_infinite]')
            }
            style={!reduceMotion ? { animationDelay: `${i * 70}ms` } : undefined}
          />
        ))}
      </div>
      {label ? <div className="text-xs text-replit-textMuted">{label}</div> : null}
    </div>
  );

  if (variant === 'inline') {
    return <div className={className}>{grid}</div>;
  }

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-replit-bg">
      <div className={className}>{grid}</div>
    </div>
  );
}
