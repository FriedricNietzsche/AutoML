import type { ReactNode } from 'react';

interface ResizablePanelProps {
  children: ReactNode;
  width?: number;
  minWidth?: number;
  maxWidth?: number;
  side: 'left' | 'right';
  isCollapsed?: boolean;
  onResize?: (side: 'left' | 'right', ev: PointerEvent) => void;
  collapsedContent?: ReactNode;
}

export default function ResizablePanel({
  children,
  width,
  side,
  isCollapsed = false,
  onResize,
  collapsedContent,
}: ResizablePanelProps) {
  if (isCollapsed) {
    return (
      <div
        className={
          'w-12 bg-replit-surface shrink-0 flex flex-col items-center py-2 gap-2 ' +
          (side === 'left' ? 'border-r border-replit-border' : 'border-l border-replit-border')
        }
      >
        {collapsedContent}
      </div>
    );
  }

  return (
    <div
      className={
        'bg-replit-surface shrink-0 relative ' +
        (side === 'left' ? 'border-r border-replit-border' : 'border-l border-replit-border')
      }
      style={{ width: width ? `${width}px` : undefined }}
    >
      {children}
      
      {/* Resize Handle */}
      <div
        className={
          `absolute top-0 bottom-0 w-1 cursor-col-resize transition-all ` +
          (side === 'left' ? 'right-0' : 'left-0')
        }
        onPointerDown={(e) => {
          e.preventDefault();
          e.stopPropagation();
          e.currentTarget.setPointerCapture(e.pointerId);
          onResize?.(side, e.nativeEvent);
        }}
      />

      {/* Visual divider line (so the separator doesn't disappear mid-drag) */}
      <div
        className={
          `absolute top-0 bottom-0 w-px bg-replit-border pointer-events-none ` +
          (side === 'left' ? 'right-0' : 'left-0')
        }
      />
    </div>
  );
}
