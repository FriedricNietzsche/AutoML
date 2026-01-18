import { useState, useEffect, useRef } from 'react';

interface ImagePixel {
  r: number;
  g: number;
  b: number;
}

interface UseImageVectorizationProps {
  currentStage: number;
  dataType?: string;
  needsClientLoad?: boolean;
  nowRef: React.MutableRefObject<number>;
}

/**
 * Custom hook that handles image loading and pixel extraction for vectorization animation
 */
export function useImageVectorization({
  currentStage,
  dataType,
  needsClientLoad,
  nowRef,
}: UseImageVectorizationProps) {
  const [loadedImagePixels, setLoadedImagePixels] = useState<ImagePixel[] | null>(null);
  const [imageAnimStartedAt, setImageAnimStartedAt] = useState<number | null>(null);
  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageOffscreenRef = useRef<HTMLCanvasElement | null>(null);

  // Load image and extract pixels
  useEffect(() => {
    const isImage = dataType === 'image';
    const needsLoad = !!needsClientLoad;

    if (!isImage || !needsLoad) {
      setLoadedImagePixels(null);
      setImageAnimStartedAt(null);
      imageOffscreenRef.current = null;
      return;
    }

    // Only do DOM image extraction when Stage 2 is actually mounted.
    if (currentStage !== 2) return;

    const canvasSize = 280;
    const gridSize = 20;
    let cancelled = false;

    // Use an offscreen canvas for sampling so we don't depend on the on-screen canvas being present.
    const offscreen = document.createElement('canvas');
    offscreen.width = canvasSize;
    offscreen.height = canvasSize;
    const sampleCtx = offscreen.getContext('2d');
    if (!sampleCtx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    let retried = false;

    img.onload = () => {
      if (cancelled) return;

      const scale = Math.min(canvasSize / img.width, canvasSize / img.height);
      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;
      const x = (canvasSize - scaledWidth) / 2;
      const y = (canvasSize - scaledHeight) / 2;

      // Draw to offscreen canvas
      sampleCtx.fillStyle = '#ffffff';
      sampleCtx.fillRect(0, 0, canvasSize, canvasSize);
      sampleCtx.drawImage(img, x, y, scaledWidth, scaledHeight);

      // Keep a handle so we can paint it onto the on-screen canvas later.
      imageOffscreenRef.current = offscreen;

      // Also draw to on-screen canvas if present
      const onCanvas = imageCanvasRef.current;
      const onCtx = onCanvas?.getContext('2d');
      if (onCtx) {
        onCtx.clearRect(0, 0, canvasSize, canvasSize);
        onCtx.drawImage(offscreen, 0, 0);
      }

      // Sample centers of a 20x20 grid from the 280x280 imageData
      const imgData = sampleCtx.getImageData(0, 0, canvasSize, canvasSize);
      const pixels: ImagePixel[] = [];
      const cellSize = canvasSize / gridSize;

      for (let gy = 0; gy < gridSize; gy++) {
        for (let gx = 0; gx < gridSize; gx++) {
          const centerX = gx * cellSize + cellSize / 2;
          const centerY = gy * cellSize + cellSize / 2;
          const idx = (Math.floor(centerY) * canvasSize + Math.floor(centerX)) * 4;
          pixels.push({ r: imgData.data[idx], g: imgData.data[idx + 1], b: imgData.data[idx + 2] });
        }
      }

      setLoadedImagePixels(pixels);
      setImageAnimStartedAt(nowRef.current || performance.now());
    };

    img.onerror = () => {
      if (cancelled) return;
      if (!retried) {
        retried = true;
        img.src = '/src/assets/image.jpg';
        return;
      }
      console.error('Failed to load image');
    };

    // Use the correct path to the image
    img.src = new URL('../../../../assets/image.jpg', import.meta.url).href;
    return () => {
      cancelled = true;
    };
  }, [currentStage, dataType, needsClientLoad, nowRef]);

  // If the image finished loading before the on-screen canvas ref was mounted, paint it now.
  useEffect(() => {
    const isImage = dataType === 'image';
    const needsLoad = !!needsClientLoad;
    if (currentStage !== 2 || !isImage || !needsLoad) return;
    if (!loadedImagePixels || loadedImagePixels.length === 0) return;

    const canvas = imageCanvasRef.current;
    const offscreen = imageOffscreenRef.current;
    if (!canvas || !offscreen) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreen, 0, 0);
  }, [currentStage, loadedImagePixels, dataType, needsClientLoad]);

  return {
    loadedImagePixels,
    imageAnimStartedAt,
    imageCanvasRef,
    imageOffscreenRef,
  };
}
