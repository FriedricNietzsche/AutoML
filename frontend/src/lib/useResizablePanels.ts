import { useState, useCallback, useRef, useEffect } from 'react';

export interface PanelSizes {
  left: number;
  right: number;
}

const MIN_PANEL_WIDTH = 240;
const MAX_PANEL_WIDTH = 760;
const MIN_CENTER_WIDTH = 420;

export function useResizablePanels(initialSizes: PanelSizes) {
  const [sizes, setSizes] = useState<PanelSizes>(initialSizes);
  const [isDragging, setIsDragging] = useState<'left' | 'right' | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number | null>(null);
  const latestMouseRef = useRef<{ x: number } | null>(null);
  const activePointerIdRef = useRef<number | null>(null);
  const prevBodyUserSelectRef = useRef<string>('');
  const prevBodyCursorRef = useRef<string>('');

  const handlePointerDown = useCallback((panel: 'left' | 'right', ev: PointerEvent) => {
    // Only primary button.
    if (typeof ev.button === 'number' && ev.button !== 0) return;
    activePointerIdRef.current = ev.pointerId;
    latestMouseRef.current = { x: ev.clientX };

    prevBodyUserSelectRef.current = document.body.style.userSelect;
    prevBodyCursorRef.current = document.body.style.cursor;
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';

    setIsDragging(panel);
  }, []);

  useEffect(() => {
    if (!isDragging) return;

    const applyResize = () => {
      rafRef.current = null;
      if (!containerRef.current) return;
      if (!latestMouseRef.current) return;

      const containerRect = containerRef.current.getBoundingClientRect();
      const containerWidth = containerRect.width;
      const mouseX = latestMouseRef.current.x - containerRect.left;

      const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

      if (isDragging === 'left') {
        setSizes((prev) => {
          const maxLeft = Math.max(
            MIN_PANEL_WIDTH,
            Math.min(MAX_PANEL_WIDTH, containerWidth - prev.right - MIN_CENTER_WIDTH)
          );
          const desired = mouseX;
          const nextLeft = clamp(desired, MIN_PANEL_WIDTH, maxLeft);
          return { ...prev, left: nextLeft };
        });
      } else if (isDragging === 'right') {
        setSizes((prev) => {
          const maxRight = Math.max(
            MIN_PANEL_WIDTH,
            Math.min(MAX_PANEL_WIDTH, containerWidth - prev.left - MIN_CENTER_WIDTH)
          );
          const desired = containerWidth - mouseX;
          const nextRight = clamp(desired, MIN_PANEL_WIDTH, maxRight);
          return { ...prev, right: nextRight };
        });
      }
    };

    const handlePointerMove = (e: PointerEvent) => {
      if (activePointerIdRef.current != null && e.pointerId !== activePointerIdRef.current) return;
      latestMouseRef.current = { x: e.clientX };
      if (rafRef.current) return;
      rafRef.current = window.requestAnimationFrame(applyResize);
    };

    const endDrag = () => {
      setIsDragging(null);
      activePointerIdRef.current = null;
      document.body.style.userSelect = prevBodyUserSelectRef.current;
      document.body.style.cursor = prevBodyCursorRef.current;
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', endDrag);
    window.addEventListener('pointercancel', endDrag);

    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', endDrag);
      window.removeEventListener('pointercancel', endDrag);
      if (rafRef.current) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      latestMouseRef.current = null;
      activePointerIdRef.current = null;
      document.body.style.userSelect = prevBodyUserSelectRef.current;
      document.body.style.cursor = prevBodyCursorRef.current;
    };
  }, [isDragging]);

  return {
    sizes,
    setSizes,
    isDragging,
    handlePointerDown,
    containerRef,
  };
}
