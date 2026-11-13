import { useEffect, useRef } from 'react';

interface WaveformCanvasProps {
  samples: Float32Array;
  sampleRate: number;
  envelope: Float32Array;
  playbackProgress: number | null;
  onSeek?: (progress: number) => void;
}

export function WaveformCanvas({
  samples,
  sampleRate,
  envelope,
  playbackProgress,
  onSeek,
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const midY = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Subtle clinical grid
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    // Draw min/max waveform peaks per pixel column
    const step = Math.max(1, Math.floor(samples.length / w));
    ctx.fillStyle = '#0369a1';
    for (let x = 0; x < w; x++) {
      const start = Math.floor((x / w) * samples.length);
      let min = 0;
      let max = 0;
      for (let j = 0; j < step && start + j < samples.length; j++) {
        const s = samples[start + j];
        if (s < min) min = s;
        if (s > max) max = s;
      }
      const yTop = midY - max * (midY - 6);
      const yBot = midY - min * (midY - 6);
      ctx.fillRect(x, yTop, 1, Math.max(1, yBot - yTop));
    }

    // Draw RMS envelope contour
    if (envelope.length > 1) {
      let maxEnv = 1e-6;
      for (let i = 0; i < envelope.length; i++) {
        if (envelope[i] > maxEnv) maxEnv = envelope[i];
      }
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < envelope.length; i++) {
        const x = (i / (envelope.length - 1)) * w;
        const y = midY - (envelope[i] / maxEnv) * (midY - 8);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Playhead cursor
    if (playbackProgress !== null && playbackProgress >= 0 && playbackProgress <= 1) {
      const px = playbackProgress * w;
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, h);
      ctx.stroke();
    }
  }, [samples, sampleRate, envelope, playbackProgress]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const progress = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(progress);
  };

  return (
    <div className="border border-slate-200 rounded-md overflow-hidden bg-white">
      <div className="flex items-center justify-between px-3 py-1.5 bg-slate-50 border-b border-slate-200 text-xs text-slate-600">
        <span className="font-medium text-slate-700">Time-Domain Oscillogram &amp; RMS Envelope</span>
        <span className="font-mono tabular-nums text-slate-500">
          {samples.length.toLocaleString()} samples · {(samples.length / sampleRate).toFixed(2)}s @{' '}
          {(sampleRate / 1000).toFixed(0)} kHz
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={768}
        height={88}
        onClick={handleClick}
        className="w-full h-22 block cursor-pointer"
      />
    </div>
  );
}
