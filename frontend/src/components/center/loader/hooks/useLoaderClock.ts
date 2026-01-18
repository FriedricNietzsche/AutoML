import { useState, useEffect, useRef } from 'react';

/**
 * Custom hook that provides a requestAnimationFrame-based clock
 * Returns the current timestamp and a ref to access it synchronously
 */
export function useLoaderClock() {
  const [now, setNow] = useState(0);
  const nowRef = useRef(0);
  const clockInitRef = useRef(false);

  useEffect(() => {
    let raf = 0;
    let last = 0;

    const tick = (t: number) => {
      raf = requestAnimationFrame(tick);
      nowRef.current = t;

      if (!clockInitRef.current) {
        clockInitRef.current = true;
        last = t;
        setNow(t);
        return;
      }

      // Update state every 50ms to avoid too frequent renders
      if (t - last > 50) {
        setNow(t);
        last = t;
      }
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return { now, nowRef };
}
