import { motion } from 'framer-motion';
import type { StepDef } from '../types';

import type { MetricsState } from '../../../../lib/metricsReducer';

interface Stage2PreprocessingProps {
  metricsState: MetricsState;
  step: StepDef;
  now: number;
  imageAnimStartedAt: number | null;
  imageCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  reducedMotion: boolean;
  stepSeed: number;
}

/**
 * Stage 2: Data Preprocessing Component
 * Shows either image vectorization animation or tabular data preview
 */
export function Stage2Preprocessing({
  metricsState,
  step,
  now,
  imageAnimStartedAt,
  imageCanvasRef,
  reducedMotion,
  stepSeed,
}: Stage2PreprocessingProps) {
  const datasetPreview = metricsState.datasetPreview;
  const isImageData = datasetPreview?.dataType === 'image';
  const loadedImagePixels = datasetPreview?.imageData?.pixels ?? [];

  if (isImageData) {
    return <ImageVectorizationView 
      step={step}
      now={now}
      imageAnimStartedAt={imageAnimStartedAt}
      loadedImagePixels={loadedImagePixels}
      imageCanvasRef={imageCanvasRef}
      reducedMotion={reducedMotion}
    />;
  }

  return <TabularDataView step={step} stepSeed={stepSeed} reducedMotion={reducedMotion} datasetPreview={datasetPreview} />;
}

// Image Vectorization Sub-component
function ImageVectorizationView({
  step,
  now,
  imageAnimStartedAt,
  loadedImagePixels,
  imageCanvasRef,
  reducedMotion,
}: {
  step: StepDef;
  now: number;
  imageAnimStartedAt: number | null;
  loadedImagePixels: Array<{ r: number; g: number; b: number }> | null;
  imageCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  reducedMotion: boolean;
}) {
  const totalDuration = step.durationMs;
  const imageElapsed = imageAnimStartedAt === null ? 0 : Math.max(0, now - imageAnimStartedAt);
  
  // Stage progression: 0 = hidden, 1 = pixels appear, 2 = grid formation, 3 = vector columns
  const animStage = imageElapsed < totalDuration * 0.2 ? 0
    : imageElapsed < totalDuration * 0.45 ? 1
    : imageElapsed < totalDuration * 0.7 ? 2
    : 3;

  const gridSize = 20;
  const canvasSize = 280;
  const canvasLeft = 40;
  const canvasTop = 56;
  const cellSize = canvasSize / gridSize;
  const particleSize = 14;

  const pixels = loadedImagePixels ?? [];
  const hasPixels = pixels.length >= gridSize * gridSize;

  const particles = hasPixels
    ? pixels.slice(0, gridSize * gridSize).map((pixel, idx) => {
        const gridX = idx % gridSize;
        const gridY = Math.floor(idx / gridSize);
        const startX = canvasLeft + gridX * cellSize + cellSize / 2 - particleSize / 2;
        const startY = canvasTop + gridY * cellSize + cellSize / 2 - particleSize / 2;
        return { ...pixel, gridX, gridY, startX, startY, idx };
      })
    : [];

  const totalParticles = 400;
  const displayCount = 100;
  const displayStart = Math.floor((totalParticles - displayCount) / 2);
  const displayEnd = displayStart + displayCount;

  // Debug logging
  console.log('ImageVectorization Debug:', {
    hasPixels,
    pixelCount: pixels.length,
    animStage,
    imageElapsed,
    totalDuration,
    imageAnimStartedAt,
    now,
  });

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4 px-6 pt-4">
        <div className="text-base font-semibold text-replit-text">Image Vectorization</div>
        <div className="text-xs text-replit-textMuted">20×20 pixels → 400-d vector</div>
      </div>

      <div className="flex-1 relative" style={{ minHeight: '500px' }}>
        {/* Original Canvas */}
        <div
          className="absolute transition-all duration-[2000ms] ease-out"
          style={{
            left: `${canvasLeft}px`,
            top: `${canvasTop}px`,
            opacity: !hasPixels ? 0 : animStage >= 1 ? 0 : 1,
            transform: animStage >= 1 ? 'scale(0.9)' : 'scale(1)',
            filter: animStage >= 1 ? 'blur(8px)' : 'blur(0px)',
          }}
        >
          <canvas
            ref={imageCanvasRef}
            width={canvasSize}
            height={canvasSize}
            className="border border-replit-border/70 rounded-xl bg-white/95 shadow-xl"
          />
          <div className="text-center mt-3 text-replit-textMuted font-medium text-xs">
            Original Image
          </div>
        </div>

        {!hasPixels && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-sm text-replit-textMuted">Loading image…</div>
          </div>
        )}

        {/* Stage Labels */}
        {animStage === 1 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute left-1/2 top-4 transform -translate-x-1/2 text-replit-text text-sm font-semibold"
          >
            Breaking into pixels...
          </motion.div>
        )}
        {animStage === 2 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute left-64 top-64 text-replit-text text-xs font-medium bg-replit-surface/70 px-2 py-1 rounded"
          >
            Pixel Grid
          </motion.div>
        )}
        {animStage === 3 && (
          <>
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute top-4 left-1/2 transform -translate-x-1/2 text-center"
            >
              <div className="text-replit-text text-sm font-semibold">Vector Representation</div>
              <div className="text-replit-textMuted text-xs">RGB color vectors in columns</div>
            </motion.div>
            <div className="absolute text-replit-text text-6xl font-bold left-8 top-1/2 transform -translate-y-1/2">⟨</div>
            <div className="absolute text-replit-text text-6xl font-bold right-8 top-1/2 transform -translate-y-1/2">⟩</div>
          </>
        )}

        {/* Particles */}
        {hasPixels &&
          particles.map((particle) => {
          const getStyle = () => {
            const color = `rgb(${particle.r}, ${particle.g}, ${particle.b})`;

            // Stage 0: Hidden
            if (animStage === 0) {
              return {
                left: `${particle.startX}px`,
                top: `${particle.startY}px`,
                width: '14px',
                height: '14px',
                opacity: 0,
                transform: 'scale(0) rotate(0deg)',
                transition: reducedMotion ? 'none' : 'all 3s cubic-bezier(0.25, 1, 0.5, 1)',
                filter: 'blur(4px)',
              };
            }

            // Stage 1: Pixels appear at original positions
            if (animStage === 1) {
              return {
                left: `${particle.startX}px`,
                top: `${particle.startY}px`,
                width: '14px',
                height: '14px',
                opacity: 1,
                transform: 'scale(1) rotate(0deg)',
                backgroundColor: color,
                transition: reducedMotion ? 'none' : 'all 3s cubic-bezier(0.25, 1, 0.5, 1)',
                transitionDelay: reducedMotion ? '0s' : `${particle.idx * 0.0012}s`,
                filter: 'blur(0px)',
              };
            }

            // Stage 2: Form tight grid
            if (animStage === 2) {
              const gridCellSize = 12;
              return {
                left: `${canvasLeft + 40 + particle.gridX * gridCellSize}px`,
                top: `${canvasTop + 60 + particle.gridY * gridCellSize}px`,
                width: `${gridCellSize - 2}px`,
                height: `${gridCellSize - 2}px`,
                opacity: 1,
                transform: 'scale(1) rotate(0deg)',
                backgroundColor: color,
                transition: reducedMotion ? 'none' : 'all 3s cubic-bezier(0.25, 1, 0.5, 1)',
                transitionDelay: reducedMotion ? '0s' : `${(particle.gridY * gridSize + particle.gridX) * 0.0008}s`,
                filter: 'blur(0px)',
              };
            }

            // Stage 3: Vector columns
            if (animStage === 3) {
              const isVisible = particle.idx >= displayStart && particle.idx < displayEnd;
              if (!isVisible) {
                return {
                  opacity: 0,
                  transform: 'scale(0) rotate(45deg)',
                  transition: reducedMotion ? 'none' : 'all 3s cubic-bezier(0.25, 1, 0.5, 1)',
                  filter: 'blur(8px)',
                };
              }

              const relativeIdx = particle.idx - displayStart;
              const itemsPerColumn = 20;
              const columnIdx = Math.floor(relativeIdx / itemsPerColumn);
              const rowIdx = relativeIdx % itemsPerColumn;
              
              // Center the columns by calculating from the middle
              // 5 columns total, each 220px wide
              // Position them symmetrically around the center, shifted right to align between brackets
              const centerX = 650; // Shifted right to center between the angle brackets
              const columnWidth = 220;
              const totalColumns = 5;
              
              // Offset from center: -2, -1, 0, 1, 2 for columns 0-4
              const columnOffset = columnIdx - Math.floor(totalColumns / 2);
              const vectorX = centerX + columnOffset * columnWidth;
              const vectorY = 80 + rowIdx * 26;

              return {
                left: `${vectorX}px`,
                top: `${vectorY}px`,
                width: 'auto',
                height: 'auto',
                opacity: 1,
                transform: 'scale(1) rotate(0deg)',
                transition: reducedMotion ? 'none' : 'all 3s cubic-bezier(0.25, 1, 0.5, 1)', // Ultra smooth easing
                transitionDelay: reducedMotion ? '0s' : `${relativeIdx * 0.005}s`, // Smooth reveal
                filter: 'blur(0px)',
              };
            }
          };

          const style = getStyle();
          const isVectorStage = animStage === 3;
          const isVisible = particle.idx >= displayStart && particle.idx < displayEnd;

          return (
            <div
              key={particle.idx}
              className="absolute will-change-transform"
              style={{
                ...style,
                borderRadius: animStage === 2 ? '2px' : '4px',
                boxShadow: animStage === 2 
                  ? '0 0 4px rgba(100, 116, 139, 0.4)' 
                  : animStage >= 3 
                    ? '0 2px 8px rgba(0,0,0,0.3)' 
                    : 'none'
              }}
            >
              {isVectorStage && isVisible && (
                <div className="flex items-center gap-2 px-2.5 py-1.5 bg-replit-surface/95 rounded border border-replit-border/60 shadow-lg">
                  <span className="text-replit-textMuted font-mono text-[10px] font-semibold min-w-[45px]">
                    v[{particle.idx}]
                  </span>
                  <span className="text-replit-text font-mono text-[10px] font-bold">
                    ({particle.r},{particle.g},{particle.b})
                  </span>
                  <div
                    className="w-7 h-3 rounded"
                    style={{
                      backgroundColor: `rgb(${particle.r}, ${particle.g}, ${particle.b})`,
                      border: '1px solid rgba(255,255,255,0.3)',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
                    }}
                  />
                </div>
              )}
            </div>
          );
        })}

        <div className="absolute bottom-4 left-4 text-xs text-replit-textMuted">
          {imageElapsed < totalDuration * 0.2
            ? 'Loading image...'
            : imageElapsed < totalDuration * 0.45
              ? 'Breaking into pixels...'
              : imageElapsed < totalDuration * 0.7
                ? 'Forming pixel grid...'
                : 'Generating feature vectors...'}
        </div>
      </div>
    </div>
  );
}

// Tabular Data Preview Sub-component  
function TabularDataView({
  step,
  stepSeed,
  reducedMotion,
  datasetPreview
}: {
  step: StepDef;
  stepSeed: number;
  reducedMotion: boolean;
  datasetPreview: MetricsState['datasetPreview'];
}) {
  const seeded = (n: number) => {
    const x = Math.sin((stepSeed + n) * 12345.6789) * 10000;
    return x - Math.floor(x);
  };
  
  const rows = datasetPreview?.rows ?? [];
  const cols = datasetPreview?.columns ?? [];
  const matrixRows = step.matrixRows ?? 8;
  const matrixCols = step.matrixCols ?? 6;
  
  const totalCells = matrixRows * matrixCols;
  const cells = Array.from({ length: totalCells }).map((_, idx) => {
    const r = Math.floor(idx / matrixCols);
    const c = idx % matrixCols;
    
    // If we have real data, try to use it
    if (r < rows.length && c < rows[r].length) {
      return { val: rows[r][c], r, c, isReal: true };
    }
    
    // Otherwise fallback to simulated data
    return { val: seeded(r * 100 + c * 10), r, c, isReal: false };
  });

  return (
    <div className="flex-1 rounded-xl bg-replit-surface/35 p-6 min-h-0 flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="text-base font-semibold text-replit-text">Data Preview</div>
        <div className="text-xs text-replit-textMuted">
            {rows.length > 0 ? `${rows.length} rows × ${cols.length} columns` : 'Loading schema...'}
        </div>
      </div>

      <div className="flex-1 overflow-hidden rounded-lg bg-replit-surface/40 p-4 md:p-6 flex flex-col min-h-0">
        <div className="text-xs text-replit-textMuted mb-3">
             {rows.length > 0 ? 'Dataset loaded & validated' : 'Profiling schema and computing statistics'}
        </div>
        
        {/* Matrix Grid with Progressive Row Animation */}
        <div className="flex-1 min-h-0 flex items-center justify-center">
          <div
            className="grid gap-px rounded-lg border border-replit-border/60 bg-replit-border/60 p-px overflow-hidden w-full h-full"
            style={{ 
              gridTemplateColumns: `repeat(${matrixCols}, minmax(0, 1fr))`,
              // Force rows to fill available space
              gridAutoRows: 'minmax(0, 1fr)',
            }}
          >
            {cells.map((cell, idx) => {
              const { val, r, isReal } = cell;
              const hue = isReal ? 200 + (val % 1) * 20 : 205 + val * 30; // Slightly different hue for real data
              const sat = isReal ? 60 : 55 + val * 25;
              const light = isReal ? 65 : 70 - val * 20;

              return (
                <div
                  key={idx}
                  className="relative min-h-0 bg-replit-surface transition-all"
                  style={{
                    animation: reducedMotion ? 'none' : `cellReveal 0.4s ease-out forwards ${r * 0.08}s`,
                    opacity: reducedMotion ? 1 : 0,
                  }}
                >
                  <div
                    className="absolute inset-0 flex items-center justify-center text-[9px] font-mono font-semibold"
                    style={{
                      color: `hsl(${hue}, ${sat}%, ${light}%)`,
                    }}
                  >
                    {typeof val === 'number' ? val.toFixed(2) : String(val).slice(0, 4)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="mt-3 text-xs text-replit-textMuted">
          {step.matrixLabel ?? 'Data Matrix'}
        </div>
      </div>
    </div>
  );
}
