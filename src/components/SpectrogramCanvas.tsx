import { useEffect, useRef, useState } from 'react';
import { buildColorMapLut } from '../audio/dspSpectrogram';
import type { ColorMapId, PredictionResult, SpectrogramData } from '../types/bioacoustics';

interface SpectrogramCanvasProps {
  spectrogram: SpectrogramData;
  prediction: PredictionResult | null;
  colorMap: ColorMapId;
  showGradCam: boolean;
  gradCamOpacity: number;
  playbackProgress: number | null;
}

interface HoverProbe {
  timeSec: number;
  freqHz: number;
  db: number;
  camWeight: number | null;
}

export function SpectrogramCanvas({
  spectrogram,
  prediction,
  colorMap,
  showGradCam,
  gradCamOpacity,
  playbackProgress,
}: SpectrogramCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const legendCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [probe, setProbe] = useState<HoverProbe | null>(null);

  // Render main spectrogram + optional Grad-CAM overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height, normalized } = spectrogram;

    // Offscreen 64x64 pixel buffer for crisp bilinear scaling
    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;

    const imgData = offCtx.createImageData(width, height);
    const lut = buildColorMapLut(colorMap);

    for (let y = 0; y < height; y++) {
      const melIdx = height - 1 - y;
      for (let x = 0; x < width; x++) {
        const idx = melIdx * width + x;
        const val = Math.max(0, Math.min(1, normalized[idx]));
        const lutOffset = Math.min(255, Math.floor(val * 255)) * 3;
        let r = lut[lutOffset];
        let g = lut[lutOffset + 1];
        let b = lut[lutOffset + 2];

        if (showGradCam && prediction && prediction.gradCamMap.length === width * height) {
          const cam = Math.max(0, Math.min(1, prediction.gradCamMap[idx]));
          if (cam > 0.18) {
            const alpha = Math.min(0.88, (cam - 0.18) * 1.2 * gradCamOpacity);
            // High-contrast scarlet-amber saliency highlight
            const hr = 244;
            const hg = Math.round(63 + (1 - cam) * 120);
            const hb = 48;
            r = Math.round(r * (1 - alpha) + hr * alpha);
            g = Math.round(g * (1 - alpha) + hg * alpha);
            b = Math.round(b * (1 - alpha) + hb * alpha);
          }
        }

        const px = (y * width + x) * 4;
        imgData.data[px] = r;
        imgData.data[px + 1] = g;
        imgData.data[px + 2] = b;
        imgData.data[px + 3] = 255;
      }
    }

    offCtx.putImageData(imgData, 0, 0);

    const displayW = canvas.width;
    const displayH = canvas.height;
    ctx.clearRect(0, 0, displayW, displayH);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(offscreen, 0, 0, displayW, displayH);

    // Subtle scientific coordinate gridlines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    const gridRows = 4;
    for (let r = 1; r < gridRows; r++) {
      const y = Math.round((r / gridRows) * displayH) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(displayW, y);
      ctx.stroke();
    }
    const gridCols = 6;
    for (let c = 1; c < gridCols; c++) {
      const x = Math.round((c / gridCols) * displayW) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, displayH);
      ctx.stroke();
    }

    // Active audio playback cursor
    if (playbackProgress !== null && playbackProgress >= 0 && playbackProgress <= 1) {
      const px = playbackProgress * displayW;
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, displayH);
      ctx.stroke();
    }
  }, [spectrogram, prediction, colorMap, showGradCam, gradCamOpacity, playbackProgress]);

  // Render vertical dB colorbar legend
  useEffect(() => {
    const canvas = legendCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const lut = buildColorMapLut(colorMap);
    const h = canvas.height;
    const w = canvas.width;
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      const norm = 1 - y / Math.max(1, h - 1);
      const lutIdx = Math.min(255, Math.floor(norm * 255)) * 3;
      for (let x = 0; x < w; x++) {
        const px = (y * w + x) * 4;
        img.data[px] = lut[lutIdx];
        img.data[px + 1] = lut[lutIdx + 1];
        img.data[px + 2] = lut[lutIdx + 2];
        img.data[px + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [colorMap]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const normX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const normY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

    const timeIdx = Math.min(spectrogram.width - 1, Math.floor(normX * spectrogram.width));
    const melIdx = Math.min(
      spectrogram.height - 1,
      Math.max(0, spectrogram.height - 1 - Math.floor(normY * spectrogram.height))
    );

    const idx = melIdx * spectrogram.width + timeIdx;
    const camWeight =
      showGradCam && prediction && prediction.gradCamMap.length > idx
        ? prediction.gradCamMap[idx]
        : null;

    setProbe({
      timeSec: normX * spectrogram.durationSec,
      freqHz: spectrogram.melFrequencies[melIdx] || 0,
      db: spectrogram.dbMatrix[idx] ?? spectrogram.minDb,
      camWeight,
    });
  };

  // Compute y-axis Mel frequency labels at 5 even vertical steps
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((frac) => {
    const melIdx = Math.min(
      spectrogram.height - 1,
      Math.round(frac * (spectrogram.height - 1))
    );
    const hz = spectrogram.melFrequencies[melIdx] || 0;
    return hz >= 1000 ? `${(hz / 1000).toFixed(1)} kHz` : `${Math.round(hz)} Hz`;
  });

  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((frac) =>
    `${(frac * spectrogram.durationSec).toFixed(2)}s`
  );

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-stretch gap-3">
        {/* Y-Axis Mel Frequency Ticks */}
        <div className="flex flex-col justify-between items-end py-0.5 text-[11px] font-mono tabular-nums text-slate-500 w-14 shrink-0">
          <span>{yTicks[4]}</span>
          <span>{yTicks[3]}</span>
          <span>{yTicks[2]}</span>
          <span>{yTicks[1]}</span>
          <span>{yTicks[0]}</span>
        </div>

        {/* Main Spectrogram Viewport */}
        <div className="relative flex-1 bg-slate-950 border border-slate-200 rounded-md overflow-hidden">
          <canvas
            ref={canvasRef}
            width={768}
            height={320}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setProbe(null)}
            className="w-full h-72 md:h-80 block cursor-crosshair"
          />

          {/* Crosshair Coordinate HUD */}
          <div className="pointer-events-none absolute top-2.5 left-2.5 bg-slate-950/85 border border-slate-700/80 text-slate-100 px-2.5 py-1 rounded text-[11px] font-mono tabular-nums flex items-center gap-3">
            {probe ? (
              <>
                <span>t: {probe.timeSec.toFixed(2)}s</span>
                <span className="text-slate-500">|</span>
                <span>f: {Math.round(probe.freqHz)} Hz</span>
                <span className="text-slate-500">|</span>
                <span>{probe.db.toFixed(1)} dB</span>
                {probe.camWeight !== null && (
                  <>
                    <span className="text-slate-500">|</span>
                    <span className="text-amber-300">
                      CAM: {(probe.camWeight * 100).toFixed(0)}%
                    </span>
                  </>
                )}
              </>
            ) : (
              <span className="text-slate-300">
                Mel-Spectrogram Tensor 64 × 64 · Hover to inspect time-frequency power
              </span>
            )}
          </div>
        </div>

        {/* dB Colorbar Legend */}
        <div className="flex items-stretch gap-1.5 shrink-0">
          <canvas
            ref={legendCanvasRef}
            width={14}
            height={256}
            className="w-3.5 h-full rounded-sm border border-slate-200"
          />
          <div className="flex flex-col justify-between text-[11px] font-mono tabular-nums text-slate-500 py-0.5 w-11">
            <span>{spectrogram.maxDb} dB</span>
            <span>{Math.round((spectrogram.maxDb + spectrogram.minDb) / 2)} dB</span>
            <span>{spectrogram.minDb} dB</span>
          </div>
        </div>
      </div>

      {/* X-Axis Time Ticks */}
      <div className="flex items-center gap-3">
        <div className="w-14 shrink-0" />
        <div className="flex-1 flex justify-between text-[11px] font-mono tabular-nums text-slate-500 pr-14">
          {xTicks.map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
