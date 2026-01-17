import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import type { VisualProps } from '../types';
import type { LossSurfaceSpec } from '../../../../mock/backendEventTypes';
import { clamp01, lerp } from '../types';

type RGB = { r: number; g: number; b: number };

function getCssVarRgb(varName: string, fallback: RGB): RGB {
  if (typeof window === 'undefined') return fallback;
  const raw = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  // Expected format in this app: "R G B" (space-separated)
  const parts = raw
    .split(/\s+/)
    .map((n) => Number(n))
    .filter((n) => Number.isFinite(n));
  if (parts.length >= 3) return { r: parts[0], g: parts[1], b: parts[2] };
  return fallback;
}

function toThreeColor(rgb: RGB) {
  return new THREE.Color(rgb.r / 255, rgb.g / 255, rgb.b / 255);
}

/*
function heightFn(x: number, y: number) {
  // Match the user's example surface.
  // z = sin(x) * cos(y) * exp(-(x^2+y^2)/20) * 8 + (x^2+y^2)/10 - 3
  return Math.sin(x) * Math.cos(y) * Math.exp(-(x * x + y * y) / 20) * 8 + (x * x + y * y) / 10 - 3;
}
*/

const defaultHeightFn = (x: number, y: number) =>
  Math.sin(x) * Math.cos(y) * Math.exp(-(x * x + y * y) / 20) * 8 + (x * x + y * y) / 10 - 3;

function makeHeightFn(spec?: LossSurfaceSpec | null) {
  if (!spec) return defaultHeightFn;
  switch (spec.kind) {
    case 'fixed_example':
      return defaultHeightFn;
    case 'bowl': {
      const params = spec.params;
      const a = params?.a ?? 0.18;
      const b = params?.b ?? 0.18;
      const tiltX = params?.tiltX ?? 0;
      const tiltY = params?.tiltY ?? 0;
      const offset = params?.offset ?? 0;
      return (x: number, y: number) => a * x * x + b * y * y + tiltX * x + tiltY * y + offset;
    }
    case 'multi_hill': {
      const params = spec.params;
      const bowlStrength = params?.bowlStrength ?? 0.06;
      const offset = params?.offset ?? 0;
      const hills = params?.hills ?? [];
      return (x: number, y: number) => {
        const r2 = x * x + y * y;
        let z = bowlStrength * r2 + offset;
        for (const hill of hills) {
          const dx = x - hill.x;
          const dy = y - hill.y;
          const s2 = Math.max(1e-4, hill.sigma * hill.sigma);
          z += hill.amp * Math.exp(-(dx * dx + dy * dy) / (2 * s2));
        }
        return z;
      };
    }
    case 'ripples': {
      const params = spec.params;
      const amp = params?.amp ?? 2.5;
      const freq = params?.freq ?? 2;
      const decay = params?.decay ?? 0.15;
      const bowlStrength = params?.bowlStrength ?? 0.06;
      const offset = params?.offset ?? 0;
      return (x: number, y: number) => {
        const r2 = x * x + y * y;
        const ripples = amp * Math.sin(freq * x) * Math.cos(freq * y) * Math.exp(-decay * r2);
        return ripples + bowlStrength * r2 + offset;
      };
    }
    default:
      return defaultHeightFn;
  }
}

function smoothstep(t: number) {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

function colormap(t: number) {
  // Blue -> Cyan -> Green -> Yellow -> Red
  const x = clamp01(t);
  const c = new THREE.Color();
  if (x < 0.25) {
    // Blue to Cyan
    c.setRGB(0, x * 4, 1);
  } else if (x < 0.5) {
    // Cyan to Green
    c.setRGB(0, 1, 1 - (x - 0.25) * 4);
  } else if (x < 0.75) {
    // Green to Yellow
    c.setRGB((x - 0.5) * 4, 1, 0);
  } else {
    // Yellow to Red
    c.setRGB(1, 1 - (x - 0.75) * 4, 0);
  }
  return c;
}

function niceStep(range: number, targetTicks: number) {
  const safe = Math.max(1e-9, range);
  const rough = safe / Math.max(1, targetTicks);
  const pow10 = Math.pow(10, Math.floor(Math.log10(rough)));
  const x = rough / pow10;
  let base = 1;
  if (x <= 1) base = 1;
  else if (x <= 2) base = 2;
  else if (x <= 5) base = 5;
  else base = 10;
  return base * pow10;
}

export default function GradientDescentViz({ timeMs, phaseProgress, seed, reducedMotion, path, surfaceSpec }: VisualProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasHostRef = useRef<HTMLDivElement>(null);
  const lockRef = useRef<{ locked: boolean; x: number; y: number }>({
    locked: false,
    x: 0,
    y: 0,
  });
  

  // Keep prop for API compatibility; this visualization is deterministic.
  void seed;

  const [themeMode, setThemeMode] = useState<'dark' | 'light'>(() => {
    if (typeof document === 'undefined') return 'dark';
    return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
  });

  useEffect(() => {
    if (typeof document === 'undefined') return;
    const root = document.documentElement;
    const update = () => setThemeMode(root.classList.contains('dark') ? 'dark' : 'light');
    update();

    const mo = new MutationObserver(() => update());
    mo.observe(root, { attributes: true, attributeFilter: ['class'] });
    return () => mo.disconnect();
  }, []);

  const heightFn = useMemo(() => makeHeightFn(surfaceSpec), [surfaceSpec]);

  const zScale = surfaceSpec?.zScale ?? 0.3;
  const resolution = 50;
  const size = 6;
  const steps = 100;
  const eta = 0.05;

  const { dfx, dfy } = useMemo(() => {
    const h = 0.02;
    return {
      dfx: (x: number, y: number) => (heightFn(x + h, y) - heightFn(x - h, y)) / (2 * h),
      dfy: (x: number, y: number) => (heightFn(x, y + h) - heightFn(x, y - h)) / (2 * h),
    };
  }, [heightFn]);

  const computedDomainHalf = useMemo(() => {
    // Shrink the *numeric* domain to cover the interesting region (global extrema + padding),
    // while keeping the rendered surface the same world size.
    let mn = Infinity;
    let mx = -Infinity;
    let minX = 0;
    let minY = 0;
    let maxX = 0;
    let maxY = 0;

    for (let i = 0; i <= resolution; i += 1) {
      for (let j = 0; j <= resolution; j += 1) {
        const x = (i / resolution) * size * 2 - size;
        const y = (j / resolution) * size * 2 - size;
        const z = heightFn(x, y);
        if (z < mn) {
          mn = z;
          minX = x;
          minY = y;
        }
        if (z > mx) {
          mx = z;
          maxX = x;
          maxY = y;
        }
      }
    }

    const needed = Math.max(Math.abs(minX), Math.abs(minY), Math.abs(maxX), Math.abs(maxY));
    const padded = needed + 1.2;
    // Clamp to sensible bounds so axes never disappear or blow up.
    return clamp01(padded / size) * size + 2.5 * (1 - clamp01(padded / size));
  }, [heightFn, resolution, size]);

  const domainHalf = surfaceSpec?.domainHalf ?? computedDomainHalf;

  const { minZ, maxZ } = useMemo(() => {
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i <= resolution; i += 1) {
      for (let j = 0; j <= resolution; j += 1) {
        const x = (i / resolution) * domainHalf * 2 - domainHalf;
        const y = (j / resolution) * domainHalf * 2 - domainHalf;
        const z = heightFn(x, y);
        mn = Math.min(mn, z);
        mx = Math.max(mx, z);
      }
    }
    return { minZ: mn, maxZ: mx };
  }, [domainHalf, heightFn, resolution]);

  const p = reducedMotion ? 1 : clamp01(phaseProgress);

  // Speed up the surface generation a bit.
  const surfaceSplit = 0.45;
  const surfaceProgress = reducedMotion ? 1 : clamp01(p / surfaceSplit);
  const gdProgress = reducedMotion ? 1 : clamp01((p - surfaceSplit) / (1 - surfaceSplit));

  const effectiveSteps = path && path.length > 1 ? path.length - 1 : steps;
  const currentIter = reducedMotion ? effectiveSteps : Math.floor(lerp(0, effectiveSteps, gdProgress));

  const latestRef = useRef({
    timeMs,
    reducedMotion,
    surfaceProgress,
    gdProgress,
  });

  useEffect(() => {
    latestRef.current = { timeMs, reducedMotion, surfaceProgress, gdProgress };
  }, [gdProgress, reducedMotion, surfaceProgress, timeMs]);

  useEffect(() => {
    const host = canvasHostRef.current;
    if (!host) return;

    // Clear any previous canvas (hot reload safety).
    host.innerHTML = '';
    // ✅ reset convergence lock whenever the scene re-mounts
    lockRef.current = { locked: false, x: 0, y: 0 };


    const fallbackAccent = { r: 99, g: 102, b: 241 };
    const fallbackSuccess = { r: 34, g: 197, b: 94 };

    const isDark = themeMode === 'dark';
    const bgRgb = isDark ? { r: 0, g: 0, b: 0 } : { r: 255, g: 255, b: 255 };
    const accentRgb = getCssVarRgb('--replit-accent', fallbackAccent);
    const successRgb = getCssVarRgb('--replit-success', fallbackSuccess);

    const axisRgb = isDark ? { r: 255, g: 255, b: 255 } : { r: 0, g: 0, b: 0 };

    const scene = new THREE.Scene();
    scene.background = toThreeColor(bgRgb);

    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    host.appendChild(renderer.domElement);

    // Keep lighting bright even in light theme
    const ambient = new THREE.AmbientLight(new THREE.Color(1, 1, 1), 0.7);
    scene.add(ambient);

    const dir = new THREE.DirectionalLight(new THREE.Color(1, 1, 1), 0.85);
    dir.position.set(6, 10, 6);
    scene.add(dir);

    const geo = new THREE.PlaneGeometry(size * 2, size * 2, resolution, resolution);
    // Rotate so Y is up.
    geo.rotateX(-Math.PI / 2);

    const posAttr = geo.getAttribute('position') as THREE.BufferAttribute;
    const vertexCount = posAttr.count;

    const heightsRaw = new Float32Array(vertexCount);
    const heightsScaled = new Float32Array(vertexCount);
    const distances = new Float32Array(vertexCount);
    let mnRaw = Infinity;
    let mxRaw = -Infinity;
    let minIdx = 0;
    let maxIdx = 0;

    for (let i = 0; i < vertexCount; i += 1) {
      const x = posAttr.getX(i);
      const z = posAttr.getZ(i);
      const fx = (x / size) * domainHalf;
      const fy = (z / size) * domainHalf;
      const h = heightFn(fx, fy);
      heightsRaw[i] = h;
      heightsScaled[i] = h * zScale;
      if (h < mnRaw) {
        mnRaw = h;
        minIdx = i;
      }
      if (h > mxRaw) {
        mxRaw = h;
        maxIdx = i;
      }
      distances[i] = Math.hypot(x, z);
    }

    const colorAttr = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
    geo.setAttribute('color', colorAttr);

    for (let i = 0; i < vertexCount; i += 1) {
      const t = (heightsRaw[i] - mnRaw) / Math.max(1e-9, mxRaw - mnRaw);
      const c = colormap(t);
      colorAttr.setXYZ(i, c.r, c.g, c.b);
    }
    colorAttr.needsUpdate = true;

    // Initialize all vertices at min height.
    for (let i = 0; i < vertexCount; i += 1) {
      posAttr.setY(i, mnRaw * zScale);
    }
    posAttr.needsUpdate = true;
    geo.computeVertexNormals();

    const mat = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 30,
      transparent: true,
      opacity: 0.96,
    });

    // (Optional but recommended) prevent z-fighting shimmer under the wireframe
    mat.polygonOffset = true;
    mat.polygonOffsetFactor = 1;
    mat.polygonOffsetUnits = 1;

    const surface = new THREE.Mesh(geo, mat);
    surface.scale.set(1.08, 1, 1.08);
    scene.add(surface);

    // ✅ PERFECTLY STICKING WIREFRAME:
    // Render a wireframe Mesh using the SAME geometry, so it always matches the surface exactly.
    const wireMat = new THREE.MeshBasicMaterial({
      color: toThreeColor(axisRgb),
      wireframe: true,
      transparent: true,
      opacity: 0,
    });

    const wireframe = new THREE.Mesh(geo, wireMat);
    wireframe.scale.copy(surface.scale);

    // Render on top for consistent visibility
    wireMat.depthTest = false;
    wireMat.depthWrite = false;
    wireframe.renderOrder = 10;
    scene.add(wireframe);

    // Axes + ground grid (match user's style)
    const axisOffset = mnRaw * zScale - 2;
    const targetY = mnRaw * zScale + 2.2;

    const axisMaterial = new THREE.MeshBasicMaterial({ color: toThreeColor(axisRgb) });
    const axisRadius = 0.035;
    const createAxisCylinder = (start: THREE.Vector3, end: THREE.Vector3) => {
      const dirVec = new THREE.Vector3().subVectors(end, start);
      const len = dirVec.length();
      const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      const g = new THREE.CylinderGeometry(axisRadius, axisRadius, len, 12);
      const m = axisMaterial.clone();
      const mesh = new THREE.Mesh(g, m);
      mesh.position.copy(mid);
      // Default cylinder is Y-up.
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dirVec.normalize());
      return mesh;
    };

    const axisWorldHalf = size * 1.02;
    const xAxis = createAxisCylinder(
      new THREE.Vector3(-axisWorldHalf, axisOffset, 0),
      new THREE.Vector3(axisWorldHalf, axisOffset, 0)
    );
    const yAxis = createAxisCylinder(
      new THREE.Vector3(0, axisOffset, -axisWorldHalf),
      new THREE.Vector3(0, axisOffset, axisWorldHalf)
    );
    const zAxis = createAxisCylinder(
      new THREE.Vector3(-axisWorldHalf, axisOffset, -axisWorldHalf),
      new THREE.Vector3(-axisWorldHalf, mxRaw * zScale + 4, -axisWorldHalf)
    );
    scene.add(xAxis);
    scene.add(yAxis);
    scene.add(zAxis);

    const gridMaterial = new THREE.LineBasicMaterial({
      color: toThreeColor(axisRgb),
      transparent: true,
      opacity: isDark ? 0.28 : 0.22,
    });
    const tickStep = niceStep(domainHalf * 2, 6);
    const tickMin = Math.ceil(-domainHalf / tickStep) * tickStep;
    const tickMax = Math.floor(domainHalf / tickStep) * tickStep;

    for (let v = tickMin; v <= tickMax + 1e-6; v += tickStep) {
      const w = (v / domainHalf) * size;
      // Lines parallel to X (varying Y)
      {
        const pts = [new THREE.Vector3(-size, axisOffset, w), new THREE.Vector3(size, axisOffset, w)];
        const g = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.Line(g, gridMaterial);
        scene.add(line);
      }
      // Lines parallel to Y (varying X)
      {
        const pts = [new THREE.Vector3(w, axisOffset, -size), new THREE.Vector3(w, axisOffset, size)];
        const g = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.Line(g, gridMaterial);
        scene.add(line);
      }
    }

    const canvasFill = isDark ? 'rgb(255 255 255)' : 'rgb(0 0 0)';
    const createTextSprite = (message: string, x: number, y: number, z: number) => {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      if (!context) return null;

      canvas.width = 256;
      canvas.height = 128;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.font =
        'Bold 86px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace';
      context.fillStyle = canvasFill;
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText(message, canvas.width / 2, canvas.height / 2);

      const texture = new THREE.CanvasTexture(canvas);
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;

      const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.position.set(x, y, z);
      sprite.scale.set(2.2, 1.1, 1);
      return sprite;
    };

    for (let v = tickMin; v <= tickMax + 1e-6; v += tickStep) {
      const w = (v / domainHalf) * size;
      const msg = Math.abs(v) < 1e-6 ? '0' : Number.isInteger(v) ? v.toString() : v.toFixed(1);
      const xLabelTick = createTextSprite(msg, w, axisOffset - 1, 0);
      if (xLabelTick) scene.add(xLabelTick);
      const yLabelTick = createTextSprite(msg, 0, axisOffset - 1, w);
      if (yLabelTick) scene.add(yLabelTick);
    }

    const zRange = mxRaw - mnRaw;
    const zStep = niceStep(zRange, 6);
    const zMin = Math.ceil(mnRaw / zStep) * zStep;
    const zMax = Math.floor(mxRaw / zStep) * zStep;
    for (let v = zMin; v <= zMax + 1e-6; v += zStep) {
      const msg = Number.isInteger(v) ? v.toString() : v.toFixed(1);
      const label = createTextSprite(msg, -axisWorldHalf - 1.2, v * zScale, -axisWorldHalf);
      if (label) scene.add(label);
    }

    const xLabel = createTextSprite('X', axisWorldHalf + 1.5, axisOffset, 0);
    const yLabel = createTextSprite('Y', 0, axisOffset, axisWorldHalf + 1.5);
    const zLabel = createTextSprite('Z', -axisWorldHalf - 1.5, mxRaw * zScale + 2, -axisWorldHalf);
    if (xLabel) scene.add(xLabel);
    if (yLabel) scene.add(yLabel);
    if (zLabel) scene.add(zLabel);

    // Markers for absolute min/max.
    const minPos = new THREE.Vector3(posAttr.getX(minIdx), heightsScaled[minIdx], posAttr.getZ(minIdx));
    const maxPos = new THREE.Vector3(posAttr.getX(maxIdx), heightsScaled[maxIdx], posAttr.getZ(maxIdx));

    const markerGeo = new THREE.SphereGeometry(0.14, 18, 18);
    const minMarkerMat = new THREE.MeshPhongMaterial({
      color: toThreeColor(successRgb),
      emissive: toThreeColor(successRgb),
      emissiveIntensity: 0.45,
    });
    const maxMarkerMat = new THREE.MeshPhongMaterial({
      color: toThreeColor(accentRgb),
      emissive: toThreeColor(accentRgb),
      emissiveIntensity: 0.35,
    });
    const minMarker = new THREE.Mesh(markerGeo, minMarkerMat);
    const maxMarker = new THREE.Mesh(markerGeo, maxMarkerMat);
    minMarker.position.copy(minPos);
    maxMarker.position.copy(maxPos);
    scene.add(minMarker);
    scene.add(maxMarker);

    const ringGeo = new THREE.RingGeometry(0.18, 0.26, 48);
    const minRingMat = new THREE.MeshBasicMaterial({
      color: toThreeColor(successRgb),
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
    });
    const maxRingMat = new THREE.MeshBasicMaterial({
      color: toThreeColor(accentRgb),
      transparent: true,
      opacity: 0.4,
      side: THREE.DoubleSide,
    });
    const minRing = new THREE.Mesh(ringGeo, minRingMat);
    const maxRing = new THREE.Mesh(ringGeo, maxRingMat);
    minRing.rotation.x = -Math.PI / 2;
    maxRing.rotation.x = -Math.PI / 2;
    minRing.position.set(minPos.x, mnRaw * zScale + 0.01, minPos.z);
    maxRing.position.set(maxPos.x, mnRaw * zScale + 0.01, maxPos.z);
    scene.add(minRing);
    scene.add(maxRing);

    // Path line (preallocate max points).
    const pathPositions = new Float32Array((effectiveSteps + 1) * 3);
    const pathGeo = new THREE.BufferGeometry();
    const pathAttr = new THREE.BufferAttribute(pathPositions, 3);
    pathGeo.setAttribute('position', pathAttr);
    pathGeo.setDrawRange(0, 0);
    
    const markerColor = new THREE.Color(1, 0.0, 0.0); // 🔥 pure red
    
    const pathMat = new THREE.LineBasicMaterial({ color: markerColor });
const pathLine = new THREE.Line(pathGeo, pathMat);
scene.add(pathLine);

// ✅ Thick trail mesh (tube)
const trailMat = new THREE.MeshBasicMaterial({
  color: markerColor,
  transparent: true,
  opacity: 0.85,
  depthTest: false,
  depthWrite: false,
});

const trailCurve = new THREE.CatmullRomCurve3([
  new THREE.Vector3(0, 0, 0),
  new THREE.Vector3(0.01, 0.01, 0.01),
]);

const trailMesh = new THREE.Mesh(new THREE.TubeGeometry(trailCurve, 64, 0.07, 12, false), trailMat);
trailMesh.renderOrder = 25;
scene.add(trailMesh);


    // ✅ Bigger + brighter moving marker

const pointGeo = new THREE.SphereGeometry(0.20, 24, 24);
const pointMat = new THREE.MeshPhongMaterial({
  color: markerColor,
  emissive: markerColor,
  emissiveIntensity: 1.7,
  shininess: 120,
});

const point = new THREE.Mesh(pointGeo, pointMat);
scene.add(point);

// ✅ Halo ring to keep marker visible
const haloGeo = new THREE.RingGeometry(0.22, 0.34, 64);
const haloMat = new THREE.MeshBasicMaterial({
  color: markerColor,
  transparent: true,
  opacity: 0.55,
  side: THREE.DoubleSide,
  depthTest: false,
  depthWrite: false,
});
const halo = new THREE.Mesh(haloGeo, haloMat);
halo.renderOrder = 30;
scene.add(halo);


    // Start point: mimic user's "70% high" start by sampling the existing grid.
    const sortedIdx = Array.from({ length: vertexCount }, (_, i) => i).sort((a, b) => heightsRaw[a] - heightsRaw[b]);
    const startIdx = sortedIdx[Math.floor(sortedIdx.length * 0.7)] ?? 0;
    const startXw = posAttr.getX(startIdx);
    const startYw = posAttr.getZ(startIdx);
    const gdX = (startXw / size) * domainHalf;
    const gdY = (startYw / size) * domainHalf;

    let raf = 0;
    let lastTrailUpdate = 0;
    let lastNormalsAt = 0;

    const resize = () => {
      const rect = host.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };

    resize();
    const ro = new ResizeObserver(() => resize());
    ro.observe(host);

    const render = () => {
      raf = requestAnimationFrame(render);

      const { timeMs: nowMs, reducedMotion: rm, surfaceProgress: sp, gdProgress: gp } = latestRef.current;
      const t = nowMs / 1000;
      const orbit = rm ? 0 : 0.25 * Math.sin(t * 0.55);

      // Bring the camera closer so the surface fills more of the canvas.
      camera.position.set(11.5 * Math.cos(orbit), 7.2, 11.5 * Math.sin(orbit));
      camera.lookAt(0, targetY, 0);

      // Surface reveal driven by progress.
      const reveal = rm ? 1 : sp;
      const radius = size * 1.45 * reveal;
      const band = size * 0.35;

      for (let i = 0; i < vertexCount; i += 1) {
        const d = distances[i];
        if (d <= radius) {
          const v = clamp01((radius - d) / Math.max(1e-6, band));
          const e = smoothstep(v);
          posAttr.setY(i, mnRaw * zScale + (heightsScaled[i] - mnRaw * zScale) * e);
        } else {
          posAttr.setY(i, mnRaw * zScale);
        }
      }
      posAttr.needsUpdate = true;

      // Avoid recomputing normals every frame.
      if (rm || nowMs - lastNormalsAt > 150) {
        geo.computeVertexNormals();
        lastNormalsAt = nowMs;
      }

      // Fade wireframe in during reveal
      wireMat.opacity = Math.min(0.4, reveal * 0.4);

      // Gradient descent path driven by progress.
      // Gradient descent path driven by progress.
const iter = rm ? effectiveSteps : Math.floor(lerp(0, effectiveSteps, gp));

// ✅ if we already converged, freeze forever
let x = lockRef.current.locked ? lockRef.current.x : gdX;
let y = lockRef.current.locked ? lockRef.current.y : gdY;

const tolGrad = 1e-3; // ✅ gradient magnitude threshold
const tolStep = 5e-4; // ✅ step size threshold

if (!lockRef.current.locked) {
  if (path && path.length > 1) {
    // ✅ STREAMED PATH CASE: lock when we reach the last path point
    const lastIdx = path.length - 1;
    const clampedIter = Math.min(iter, lastIdx);

    for (let k = 0; k <= clampedIter; k += 1) {
      const pp = path[k];
      const px = pp.x * domainHalf;
      const py = pp.y * domainHalf;

      const z = heightFn(px, py);
      const wx = (px / domainHalf) * size;
      const wy = (py / domainHalf) * size;

      pathAttr.setXYZ(k, wx, z * zScale, wy);

      x = px;
      y = py;
    }

    // ✅ lock at the final streamed point
    if (clampedIter >= lastIdx) {
      lockRef.current = { locked: true, x, y };
    }
  } else {
    // ✅ SIMULATED GD CASE: stop once we converge
    for (let k = 0; k <= iter; k += 1) {
      const z = heightFn(x, y);
      const wx = (x / domainHalf) * size;
      const wy = (y / domainHalf) * size;

      pathAttr.setXYZ(k, wx, z * zScale, wy);

      if (k < iter) {
        const gx = dfx(x, y);
        const gy = dfy(x, y);

        const gnorm = Math.hypot(gx, gy);
        const nx = x - eta * gx;
        const ny = y - eta * gy;
        const stepDist = Math.hypot(nx - x, ny - y);

        // ✅ CONVERGENCE CHECK
        if (gnorm < tolGrad || stepDist < tolStep) {
          lockRef.current = { locked: true, x, y };
          break; // ✅ stop simulating immediately
        }

        // update normally
        x = nx * 0.995;
        y = ny * 0.995;
      }
    }
  }
}

      pathAttr.needsUpdate = true;
      pathGeo.setDrawRange(0, Math.max(0, iter + 1));
      point.position.set((x / domainHalf) * size, heightFn(x, y) * zScale, (y / domainHalf) * size);
      // ✅ Keep halo on top of marker
halo.position.copy(point.position);
halo.rotation.x = -Math.PI / 2;

if (!rm) {
  const pulse = 0.6 + 0.35 * Math.sin(t * 4.5);
  haloMat.opacity = 0.25 + 0.35 * pulse;
}

      // ✅ Update thick trail tube geometry from current path points
      if (iter < 1) {
        trailMesh.visible = false;
      } else {
        trailMesh.visible = true;
        if (nowMs - lastTrailUpdate > 80) {
          lastTrailUpdate = nowMs;
          const trailPoints: THREE.Vector3[] = [];
          const count = iter + 1;
          for (let k = 0; k < count; k += 1) {
            const vx = pathAttr.getX(k);
            const vy = pathAttr.getY(k);
            const vz = pathAttr.getZ(k);
            trailPoints.push(new THREE.Vector3(vx, vy, vz));
          }

          if (trailPoints.length >= 2) {
            const newCurve = new THREE.CatmullRomCurve3(trailPoints);
            const newGeo = new THREE.TubeGeometry(newCurve, 40, 0.05, 8, false);
            trailMesh.geometry.dispose();
            trailMesh.geometry = newGeo;
          }
        }
      }

      if (!rm) {
        const pulse = 0.55 + 0.35 * Math.sin(t * 2.4);
        (minRing.material as THREE.MeshBasicMaterial).opacity = 0.25 + 0.25 * pulse;
        (maxRing.material as THREE.MeshBasicMaterial).opacity = 0.2 + 0.2 * pulse;
      }

      renderer.render(scene, camera);
    };

    render();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();

      // Remove objects
      scene.remove(surface);
      scene.remove(wireframe);
      scene.remove(pathLine);
      scene.remove(point);
      scene.remove(minMarker);
      scene.remove(maxMarker);
      scene.remove(minRing);
      scene.remove(maxRing);
      scene.remove(xAxis);
      scene.remove(yAxis);
      scene.remove(zAxis);
      scene.remove(halo);
haloGeo.dispose();
haloMat.dispose();

scene.remove(trailMesh);
trailMat.dispose();
trailMesh.geometry.dispose();


      // Dispose (IMPORTANT: geo is shared by surface + wireframe -> dispose ONCE)
      geo.dispose();
      mat.dispose();
      wireMat.dispose();

      pathGeo.dispose();
      pathMat.dispose();
      pointGeo.dispose();
      pointMat.dispose();

      markerGeo.dispose();
      minMarkerMat.dispose();
      maxMarkerMat.dispose();

      ringGeo.dispose();
      minRingMat.dispose();
      maxRingMat.dispose();

      gridMaterial.dispose();
      axisMaterial.dispose();

      (xAxis.geometry as THREE.BufferGeometry).dispose();
      (yAxis.geometry as THREE.BufferGeometry).dispose();
      (zAxis.geometry as THREE.BufferGeometry).dispose();
      (xAxis.material as THREE.Material).dispose();
      (yAxis.material as THREE.Material).dispose();
      (zAxis.material as THREE.Material).dispose();

      renderer.dispose();
      host.innerHTML = '';
    };
  }, [dfx, dfy, domainHalf, effectiveSteps, heightFn, path, resolution, size, themeMode, zScale]);

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-replit-border/60 bg-replit-surface/35 backdrop-blur p-6 pb-5">
        <div ref={containerRef} className="relative rounded-xl border border-replit-border/60 overflow-hidden bg-white dark:bg-black">
          <div ref={canvasHostRef} className="h-[760px] w-full" />

          {/* Keep only min/max + iter badges; axes are now 3D inside the scene. */}
          <div className="absolute right-4 top-4 pointer-events-none flex flex-col gap-1 text-[11px] font-mono text-replit-text">
            <div className="px-2 py-1 rounded border border-replit-border/60 bg-replit-surface/40">
              <span className="text-replit-success">min</span> {minZ.toFixed(2)}
            </div>
            <div className="px-2 py-1 rounded border border-replit-border/60 bg-replit-surface/40">
              <span className="text-replit-accent">max</span> {maxZ.toFixed(2)}
            </div>
            <div className="px-2 py-1 rounded border border-replit-border/60 bg-replit-surface/40 text-replit-textMuted">
              iter {currentIter}/{steps}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
