import { useEffect, useRef } from 'react';
import { buildColorMapLut } from '../audio/dspSpectrogram';
import type { ColorMapId, LayerFeatureMap, PredictionResult } from '../types/bioacoustics';

interface CnnArchitectureViewProps {
  prediction: PredictionResult | null;
  totalParams: number;
  colorMap: ColorMapId;
}

function FeatureMapCanvas({
  data,
  width,
  height,
  colorMap,
}: {
  data: Float32Array;
  width: number;
  height: number;
  colorMap: ColorMapId;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const off = document.createElement('canvas');
    off.width = width;
    off.height = height;
    const offCtx = off.getContext('2d');
    if (!offCtx) return;

    const img = offCtx.createImageData(width, height);
    const lut = buildColorMapLut(colorMap);
    for (let y = 0; y < height; y++) {
      const melIdx = height - 1 - y;
      for (let x = 0; x < width; x++) {
        const val = Math.max(0, Math.min(1, data[melIdx * width + x] ?? 0));
        const lutIdx = Math.min(255, Math.floor(val * 255)) * 3;
        const px = (y * width + x) * 4;
        img.data[px] = lut[lutIdx];
        img.data[px + 1] = lut[lutIdx + 1];
        img.data[px + 2] = lut[lutIdx + 2];
        img.data[px + 3] = 255;
      }
    }
    offCtx.putImageData(img, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }, [data, width, height, colorMap]);

  return (
    <canvas
      ref={ref}
      width={64}
      height={64}
      className="w-full aspect-square rounded-sm border border-slate-200 bg-slate-950"
    />
  );
}

const STATIC_LAYERS = [
  { name: 'mel_spectrogram_input', type: 'Input Tensor', shape: '64 × 64 × 1', params: 0 },
  { name: 'conv2d_1', type: 'Conv2D (3×3, ReLU, same)', shape: '64 × 64 × 16', params: 160 },
  { name: 'maxpool_1', type: 'MaxPooling2D (2×2)', shape: '32 × 32 × 16', params: 0 },
  { name: 'conv2d_2', type: 'Conv2D (3×3, ReLU, same)', shape: '32 × 32 × 24', params: 3480 },
  { name: 'maxpool_2', type: 'MaxPooling2D (2×2)', shape: '16 × 16 × 24', params: 0 },
  { name: 'conv2d_3', type: 'Conv2D (3×3, ReLU, same)', shape: '16 × 16 × 32', params: 6944 },
  { name: 'maxpool_3', type: 'MaxPooling2D (2×2)', shape: '8 × 8 × 32', params: 0 },
  { name: 'flatten', type: 'Flatten', shape: '2048', params: 0 },
  { name: 'dropout', type: 'Dropout (p = 0.20)', shape: '2048', params: 0 },
  { name: 'dense_latent', type: 'Dense (ReLU)', shape: '48', params: 98352 },
  { name: 'output_probs', type: 'Dense (Softmax)', shape: '4', params: 196 },
];

export function CnnArchitectureView({
  prediction,
  totalParams,
  colorMap,
}: CnnArchitectureViewProps) {
  return (
    <div className="flex flex-col gap-6">
      <div className="bg-white border border-slate-200 rounded-md p-6">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
          <div>
            <h2 className="font-display text-lg font-bold text-slate-900">
              2D Convolutional Neural Network Topology
            </h2>
            <p className="text-xs text-slate-600 mt-1">
              Three spatial convolutional blocks extract harmonic formants, paroxysmal burst edges, and high-frequency wheeze/crackle textures from 64×64 Mel-spectrogram tensors.
            </p>
          </div>
          <div className="text-right font-mono tabular-nums">
            <div className="text-xs text-slate-500">Trainable Parameters</div>
            <div className="text-xl font-bold text-slate-900">
              {(totalParams || 109132).toLocaleString()}
            </div>
          </div>
        </div>

        {/* Layer Specification Table */}
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500 uppercase tracking-wider text-[11px]">
                <th className="py-2 pr-4 font-semibold">Layer Name</th>
                <th className="py-2 pr-4 font-semibold">Operation</th>
                <th className="py-2 pr-4 font-semibold">Output Tensor</th>
                <th className="py-2 text-right font-semibold">Parameters</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {STATIC_LAYERS.map((layer) => (
                <tr key={layer.name} className="hover:bg-slate-50">
                  <td className="py-2 pr-4 font-mono text-slate-900 font-medium">{layer.name}</td>
                  <td className="py-2 pr-4 text-slate-700">{layer.type}</td>
                  <td className="py-2 pr-4 font-mono tabular-nums text-sky-700">{layer.shape}</td>
                  <td className="py-2 text-right font-mono tabular-nums text-slate-600">
                    {layer.params.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Live Convolutional Feature Maps */}
      <div className="bg-white border border-slate-200 rounded-md p-6">
        <div className="pb-4 border-b border-slate-200">
          <h3 className="font-display text-base font-bold text-slate-900">
            Live Intermediate Activation Feature Maps
          </h3>
          <p className="text-xs text-slate-600 mt-1">
            Top 8 highest-energy convolutional filter responses extracted during the active spectrogram forward pass.
          </p>
        </div>

        {prediction && prediction.featureMaps.length > 0 ? (
          <div className="mt-5 flex flex-col gap-6">
            {prediction.featureMaps.map((block: LayerFeatureMap) => (
              <div key={block.layerName} className="flex flex-col gap-3">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <span className="font-semibold text-slate-800">{block.label}</span>
                  <span className="font-mono tabular-nums text-slate-500">
                    {block.layerName} · Tensor {block.shapeLabel} · {block.paramsCount.toLocaleString()} params
                  </span>
                </div>
                <div className="grid grid-cols-4 sm:grid-cols-8 gap-2.5">
                  {block.maps.map((mapData, idx) => (
                    <div key={`${block.layerName}-${idx}`} className="flex flex-col gap-1">
                      <FeatureMapCanvas
                        data={mapData}
                        width={block.mapWidth}
                        height={block.mapHeight}
                        colorMap={colorMap}
                      />
                      <span className="text-[10px] font-mono tabular-nums text-slate-500 text-center">
                        Filter #{idx + 1}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="py-8 text-center text-xs text-slate-500">
            Classify a cough spectrogram in Spectrogram Lab to inspect live convolutional filter activations.
          </div>
        )}
      </div>
    </div>
  );
}
