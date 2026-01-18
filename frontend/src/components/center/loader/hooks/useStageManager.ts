import { useState, useEffect, useRef, useCallback } from 'react';
import type { StageID, StageStatus } from '../../../../lib/contract';
import type { StepDef } from '../types';
import { STAGE_DEFINITIONS } from '../utils/stageDefinitions';
import { writeJson, appendLog } from '../utils/loaderHelpers';

interface UseStageManagerProps {
  onComplete: () => void;
  updateFileContent: (path: string, content: string | ((prev: string) => string)) => void;
  nowRef: React.MutableRefObject<number>;
  backendStageId?: StageID;
  backendStageStatus?: StageStatus;
  useMockStream?: boolean;
  onStart?: () => void | Promise<void>;
}

const BACKEND_TO_FRONTEND_MAP: Record<StageID, number> = {
  // Hold at 0 during PARSE_INTENT to avoid starting visuals until user confirms
  PARSE_INTENT: 0,
  DATA_SOURCE: 1,
  PROFILE_DATA: 2,
  PREPROCESS: 2,
  MODEL_SELECT: 3,
  TRAIN: 3,
  REVIEW_EDIT: 4,
  EXPORT: 5,
};

/**
 * Custom hook that manages stage progression and state
 */
export function useStageManager({ 
  onComplete, 
  updateFileContent, 
  nowRef, 
  backendStageId, 
  backendStageStatus,
  useMockStream = true,
  onStart 
}: UseStageManagerProps) {
  const [currentStage, setCurrentStage] = useState<number>(0);
  const [isStageRunning, setIsStageRunning] = useState<boolean>(false);
  const [stageCompleted, setStageCompleted] = useState<boolean>(false);
  const [showChangeOption, setShowChangeOption] = useState<boolean>(false);
  const [changeRequest, setChangeRequest] = useState<string>('');

  const [stepIndex, setStepIndex] = useState(0);
  const [stepStartedAt, setStepStartedAt] = useState(0);
  const stepIndexRef = useRef(0);
  const updateFileContentRef = useRef(updateFileContent);
  const lastBackendStageRef = useRef<StageID | undefined>(undefined);

  useEffect(() => {
    updateFileContentRef.current = updateFileContent;
  }, [updateFileContent]);

  useEffect(() => {
    stepIndexRef.current = stepIndex;
  }, [stepIndex]);

  // Sync with Backend (Live Mode)
  useEffect(() => {
    if (useMockStream) return;
    if (!backendStageId) return;

    // React if stage ID or status changes
    if (backendStageId === lastBackendStageRef.current && !backendStageStatus) return;
    lastBackendStageRef.current = backendStageId;

    const frontendStage = BACKEND_TO_FRONTEND_MAP[backendStageId] ?? 0;
    // Do not start visuals during PARSE_INTENT (mapped to 0)
    if (frontendStage === 0) {
      setCurrentStage(0);
      setIsStageRunning(false);
      setStageCompleted(false);
      return;
    }

    // Start/stop based on backend status
    if (backendStageStatus === 'IN_PROGRESS') {
      if (frontendStage !== currentStage || !isStageRunning) {
        setCurrentStage(frontendStage);
        setIsStageRunning(true);
        setStageCompleted(false);
        setStepIndex(0);
        setStepStartedAt(nowRef.current || 0);
      }
    } else if (backendStageStatus === 'WAITING_CONFIRMATION') {
      setCurrentStage(frontendStage);
      setIsStageRunning(false);
      setStageCompleted(true); // show completed overlay until user confirms
    } else {
      // Other states: pause animations
      setCurrentStage(frontendStage);
      setIsStageRunning(false);
    }
  }, [backendStageId, backendStageStatus, useMockStream, currentStage, isStageRunning, nowRef]);

  // Handle stage progression (Mock Mode or Manual)
  const handleProceed = useCallback(() => {
    if (!useMockStream) {
      // In live mode, trigger backend start
      if (currentStage === 0 && onStart) {
        onStart();
      }
      return;
    }

    if (currentStage === 0) {
      // Start stage 1
      setCurrentStage(1);
      setIsStageRunning(true);
      setStageCompleted(false);
      setStepIndex(0);
      setStepStartedAt(nowRef.current || 0);
    } else if (stageCompleted && !isStageRunning) {
      // Move to next stage
      const nextStage = currentStage + 1;
      if (nextStage <= 5) {
        setCurrentStage(nextStage);
        setIsStageRunning(true);
        setStageCompleted(false);
        setShowChangeOption(false);
        setStepIndex(0);
        setStepStartedAt(nowRef.current || 0);
      }
    }
  }, [currentStage, stageCompleted, isStageRunning, nowRef, useMockStream, onStart]);

  const handleMakeChanges = useCallback(() => {
    if (changeRequest.trim()) {
      console.log('Change requested:', changeRequest);
      appendLog(updateFileContentRef.current, `User requested changes: ${changeRequest}`);
    }
    // Only allow manual backtrack in mock or if specific backend logic allows
    if (useMockStream) {
        setCurrentStage(3);
        setIsStageRunning(true);
        setStageCompleted(false);
        setShowChangeOption(false);
        setChangeRequest('');
        setStepIndex(0);
        setStepStartedAt(nowRef.current || 0);
    }
  }, [changeRequest, nowRef, useMockStream]);

  const handleDeployment = useCallback(() => {
    if (useMockStream) {
        setCurrentStage(5);
        setIsStageRunning(true);
        setStageCompleted(false);
        setShowChangeOption(false);
        setStepIndex(0);
        setStepStartedAt(nowRef.current || 0);
    }
  }, [nowRef, useMockStream]);

  // Log step start
  const logStepStart = useCallback((step: StepDef) => {
    if (currentStage === 0 || !isStageRunning) return;
    appendLog(updateFileContentRef.current, `${step.title} — ${step.subtitle}`);
    writeJson(updateFileContentRef.current, '/artifacts/progress.json', {
      step: step.id,
      stage: currentStage,
      startedAt: new Date().toISOString(),
    });
  }, [currentStage, isStageRunning]);

  // Check when stage animation is complete (Mock Mode only)
  useEffect(() => {
    if (!useMockStream) return; // Disable timer-based progression in live mode
    if (!isStageRunning || currentStage === 0) return;
    
    const stageDef = STAGE_DEFINITIONS[currentStage - 1];
    if (!stageDef) return;
    
    // In live mode, we don't timeout, we wait for backend status to change
    
    const timeout = setTimeout(() => {
      setIsStageRunning(false);
      setStageCompleted(true);
      if (currentStage === 4) {
        setShowChangeOption(true);
      } else if (currentStage === 5) {
        // After deployment, navigate to tester page
        setTimeout(() => {
          onComplete();
        }, 1500);
      }
    }, stageDef.durationMs);
    
    return () => clearTimeout(timeout);
  }, [isStageRunning, currentStage, onComplete, useMockStream]);

  return {
    currentStage,
    isStageRunning,
    stageCompleted,
    showChangeOption,
    changeRequest,
    stepIndex,
    stepStartedAt,
    stepIndexRef,
    setCurrentStage,
    setIsStageRunning,
    setStageCompleted,
    setShowChangeOption,
    setChangeRequest,
    setStepIndex,
    setStepStartedAt,
    handleProceed,
    handleMakeChanges,
    handleDeployment,
    logStepStart,
  };
}
