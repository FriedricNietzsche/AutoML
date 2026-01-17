import { useState, useCallback } from 'react';
import type { FileSystemNode } from './types';

export function usePipelineRunner(
  _files: FileSystemNode[],
  updateFileContent: (path: string, content: string | ((prev: string) => string)) => void
) {
  const [isRunning, setIsRunning] = useState(false);

  const runPipeline = useCallback(async () => {
    if (isRunning) return;
    setIsRunning(true);
    
    // Attempt to clear logs if possible, handled mostly by visualizer
    try {
       updateFileContent('/logs/training.log', '');
    } catch (e) { console.warn(e) }

  }, [isRunning, updateFileContent]);

  const completePipeline = useCallback(() => {
    setIsRunning(false);
  }, []);

  return { 
    isRunning, 
    runPipeline,
    completePipeline 
  };
}
