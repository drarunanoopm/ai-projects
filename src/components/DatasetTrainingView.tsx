import { useMemo, useState } from 'react';
import { DISEASE_CLASSES, DISEASE_PROFILES } from '../audio/cattleSynthesizer';
import { renderSpectrogramThumbnail } from '../audio/dspSpectrogram';
import type { DatasetSamplePreview } from '../ml/cattleCnn';
import type {
  ColorMapId,
  ModelEvaluation,
  TrainingConfig,
  TrainingMetricsPoint,
} from '../types/bioacoustics';

interface DatasetTrainingViewProps {
  trainingConfig: TrainingConfig;
  onChangeTrainingConfig: (next: TrainingConfig) => void;
  onRetrain: () => void;
  isTraining: boolean;
  history: TrainingMetricsPoint[];
  evaluation: ModelEvaluation | null;
  previews: DatasetSamplePreview[];
  colorMap: ColorMapId;
  onLoadDatasetSample: (sample: DatasetSamplePreview) => void;
}

export function DatasetTrainingView({
  trainingConfig,
  onChangeTrainingConfig,
  onRetrain,
  isTraining,
  history,
  evaluation,
  previews,
  colorMap,
  onLoadDatasetSample,
}: DatasetTrainingViewProps) {
  const [hoveredEpoch, setHoveredEpoch] = useState<number | null>(null);

  const previewThumbnails = useMemo(() => {
    return previews.map((sample) => ({
      sample,
      dataUrl: renderSpectrogramThumbnail(
        sample.spectrogram.normalized,
        sample.spectrogram.width,
        sample.spectrogram.height,
        colorMap
      ),
    }));
  }, [previews, colorMap]);

  // Chart geometry for SVG Dual-Axis Training Curves
  const chartW = 640;
  const chartH = 220;
  const padL = 42;
  const padR = 42;
  const padT = 18;
  const padB = 30;
  const plotW = chartW - padL - padR;
  const plotH = chartH - padT - padB;

  const maxEpoch = Math.max(2, trainingConfig.epochs, history.length);
  const maxLoss = Math.max(1.6, ...history.map((p) => Math.max(p.loss, p.valLoss)));

  const toX = (epoch: number) =>
    padL + ((epoch - 1) / Math.max(1, maxEpoch - 1)) * plotW;
  const toYAcc = (acc: number) => padT + (1 - Math.max(0, Math.min(1, acc))) * plotH;
  const toYLoss = (loss: number) =>
    padT + (1 - Math.max(0, Math.min(1, loss / maxLoss))) * plotH;

  const buildPath = (pts: TrainingMetricsPoint[], yFn: (p: TrainingMetricsPoint) => number) =>
    pts
      .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${toX(p.epoch).toFixed(1)} ${yFn(p).toFixed(1)}`)
      .join(' ');

  const hoveredPoint =
    hoveredEpoch !== null ? history.find((p) => p.epoch === hoveredEpoch) ?? null : null;

  return (
    <div className="flex flex-col gap-6">
      {/* Row 1: Hyperparameters + Training Curves */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-md p-5 flex flex-col gap-4">
          <div className="pb-3 border-b border-slate-200">
            <h2 className="font-display text-base font-bold text-slate-900">
              Dataset &amp; Training Controls
            </h2>
            <p className="text-xs text-slate-600 mt-1">
              Synthesizes balanced Mel-spectrogram tensors and trains the 2D CNN weights in browser memory.
            </p>
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-slate-700">Samples per Class</span>
              <span className="font-mono tabular-nums text-slate-600">
                {trainingConfig.samplesPerClass} ({trainingConfig.samplesPerClass * 4} total)
              </span>
            </div>
            <input
              type="range"
              min={20}
              max={64}
              step={4}
              value={trainingConfig.samplesPerClass}
              onChange={(e) =>
                onChangeTrainingConfig({
                  ...trainingConfig,
                  samplesPerClass: Number(e.target.value),
                })
              }
              className="w-full accent-sky-700 cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-slate-700">Training Epochs</span>
              <span className="font-mono tabular-nums text-slate-600">{trainingConfig.epochs}</span>
            </div>
            <input
              type="range"
              min={8}
              max={30}
              step={2}
              value={trainingConfig.epochs}
              onChange={(e) =>
                onChangeTrainingConfig({
                  ...trainingConfig,
                  epochs: Number(e.target.value),
                })
              }
              className="w-full accent-sky-700 cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-slate-700">Barn Noise Augmentation</span>
              <span className="font-mono tabular-nums text-slate-600">
                {trainingConfig.noiseAugmentation.toFixed(3)}
              </span>
            </div>
            <input
              type="range"
              min={0.005}
              max={0.05}
              step={0.005}
              value={trainingConfig.noiseAugmentation}
              onChange={(e) =>
                onChangeTrainingConfig({
                  ...trainingConfig,
                  noiseAugmentation: Number(e.target.value),
                })
              }
              className="w-full accent-sky-700 cursor-pointer"
            />
          </div>

          <div className="flex items-center justify-between gap-2">
            <label htmlFor="lr-select" className="text-xs font-medium text-slate-700">
              Adam Learning Rate
            </label>
            <select
              id="lr-select"
              value={trainingConfig.learningRate}
              onChange={(e) =>
                onChangeTrainingConfig({
                  ...trainingConfig,
                  learningRate: Number(e.target.value),
                })
              }
              className="text-xs font-mono bg-slate-50 border border-slate-200 rounded px-2 py-1 text-slate-800"
            >
              <option value={0.001}>0.0010</option>
              <option value={0.0018}>0.0018</option>
              <option value={0.003}>0.0030</option>
            </select>
          </div>

          <button
            type="button"
            onClick={onRetrain}
            disabled={isTraining}
            className="mt-auto w-full py-2.5 px-4 rounded-md bg-sky-700 hover:bg-sky-800 disabled:bg-slate-300 text-white text-xs font-semibold transition-colors cursor-pointer"
          >
            {isTraining ? 'Training CNN Weights...' : 'Retrain CNN Model'}
          </button>
        </div>

        {/* Right 8 cols: Dual-Axis Epoch Loss & Accuracy Chart */}
        <div className="lg:col-span-8 bg-white border border-slate-200 rounded-md p-5 flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
            <div>
              <h3 className="font-display text-base font-bold text-slate-900">
                Training &amp; Validation Convergence
              </h3>
              <p className="text-xs text-slate-500">
                Solid lines denote validation set · Dashed lines denote training set
              </p>
            </div>
            <div className="flex items-center gap-4 text-xs font-mono">
              <span className="text-sky-700 font-medium">Accuracy (0–100%)</span>
              <span className="text-amber-600 font-medium">Cross-Entropy Loss</span>
            </div>
          </div>

          <div className="relative">
            <svg
              viewBox={`0 0 ${chartW} ${chartH}`}
              className="w-full h-56 overflow-visible"
              onMouseLeave={() => setHoveredEpoch(null)}
            >
              {/* Horizontal Gridlines */}
              {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
                const y = padT + (1 - frac) * plotH;
                return (
                  <g key={frac}>
                    <line
                      x1={padL}
                      y1={y}
                      x2={chartW - padR}
                      y2={y}
                      stroke="#e2e8f0"
                      strokeWidth={1}
                    />
                    <text
                      x={padL - 6}
                      y={y + 3}
                      textAnchor="end"
                      className="fill-slate-400 text-[10px] font-mono"
                    >
                      {(frac * 100).toFixed(0)}%
                    </text>
                    <text
                      x={chartW - padR + 6}
                      y={y + 3}
                      textAnchor="start"
                      className="fill-slate-400 text-[10px] font-mono"
                    >
                      {(frac * maxLoss).toFixed(2)}
                    </text>
                  </g>
                );
              })}

              {history.length > 0 && (
                <>
                  {/* Train Accuracy (dashed sky) */}
                  <path
                    d={buildPath(history, (p) => toYAcc(p.accuracy))}
                    fill="none"
                    stroke="#0284c7"
                    strokeWidth={1.75}
                    strokeDasharray="4 3"
                  />
                  {/* Val Accuracy (solid sky) */}
                  <path
                    d={buildPath(history, (p) => toYAcc(p.valAccuracy))}
                    fill="none"
                    stroke="#0369a1"
                    strokeWidth={2.25}
                  />
                  {/* Train Loss (dashed amber) */}
                  <path
                    d={buildPath(history, (p) => toYLoss(p.loss))}
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                  />
                  {/* Val Loss (solid amber) */}
                  <path
                    d={buildPath(history, (p) => toYLoss(p.valLoss))}
                    fill="none"
                    stroke="#d97706"
                    strokeWidth={2}
                  />

                  {/* Interactive Hover Targets */}
                  {history.map((pt) => (
                    <g key={pt.epoch}>
                      <line
                        x1={toX(pt.epoch)}
                        y1={padT}
                        x2={toX(pt.epoch)}
                        y2={padT + plotH}
                        stroke={hoveredEpoch === pt.epoch ? '#94a3b8' : 'transparent'}
                        strokeWidth={1}
                      />
                      <rect
                        x={toX(pt.epoch) - plotW / (maxEpoch * 2)}
                        y={padT}
                        width={Math.max(8, plotW / maxEpoch)}
                        height={plotH}
                        fill="transparent"
                        onMouseEnter={() => setHoveredEpoch(pt.epoch)}
                      />
                    </g>
                  ))}
                </>
              )}

              {/* X-axis Epoch labels */}
              {Array.from({ length: 5 }, (_, i) => {
                const ep = Math.max(1, Math.round(1 + (i / 4) * (maxEpoch - 1)));
                return (
                  <text
                    key={ep}
                    x={toX(ep)}
                    y={chartH - 8}
                    textAnchor="middle"
                    className="fill-slate-500 text-[10px] font-mono"
                  >
                    Ep {ep}
                  </text>
                );
              })}
            </svg>

            <div className="mt-1 flex items-center justify-between text-xs font-mono tabular-nums text-slate-600 bg-slate-50 border border-slate-200 rounded px-3 py-1.5">
              {hoveredPoint ? (
                <>
                  <span>Epoch {hoveredPoint.epoch}</span>
                  <span>
                    Val Acc: {(hoveredPoint.valAccuracy * 100).toFixed(1)}% · Train Acc:{' '}
                    {(hoveredPoint.accuracy * 100).toFixed(1)}%
                  </span>
                  <span>
                    Val Loss: {hoveredPoint.valLoss.toFixed(3)} · Train Loss:{' '}
                    {hoveredPoint.loss.toFixed(3)}
                  </span>
                </>
              ) : history.length > 0 ? (
                <>
                  <span>Latest Epoch {history[history.length - 1].epoch}</span>
                  <span>
                    Val Acc: {(history[history.length - 1].valAccuracy * 100).toFixed(1)}%
                  </span>
                  <span>Val Loss: {history[history.length - 1].valLoss.toFixed(3)}</span>
                </>
              ) : (
                <span>Waiting for training epochs...</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Row 2: Validation Confusion Matrix & Per-Class Metrics */}
      {evaluation && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-6 bg-white border border-slate-200 rounded-md p-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-200">
              <h3 className="font-display text-base font-bold text-slate-900">
                Held-Out Validation Confusion Matrix
              </h3>
              <span className="text-xs font-mono tabular-nums text-slate-600">
                n = {evaluation.validationCount} · Acc {(evaluation.accuracy * 100).toFixed(1)}%
              </span>
            </div>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr>
                    <th className="p-2 text-left text-[11px] font-semibold text-slate-500 border-b border-slate-200">
                      Actual \ Predicted
                    </th>
                    {DISEASE_CLASSES.map((c) => (
                      <th
                        key={c}
                        className="p-2 text-center text-[11px] font-semibold text-slate-600 border-b border-slate-200"
                      >
                        {DISEASE_PROFILES[c].shortName}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {DISEASE_CLASSES.map((rowCls, rIdx) => (
                    <tr key={rowCls} className="border-b border-slate-100">
                      <td className="p-2 font-medium text-slate-800">
                        {DISEASE_PROFILES[rowCls].shortName}
                      </td>
                      {DISEASE_CLASSES.map((colCls, cIdx) => {
                        const count = evaluation.confusionMatrix[rIdx]?.[cIdx] ?? 0;
                        const isDiag = rIdx === cIdx;
                        return (
                          <td
                            key={colCls}
                            className={`p-2 text-center font-mono tabular-nums rounded-sm ${
                              isDiag
                                ? 'bg-sky-50 text-sky-900 font-semibold'
                                : count > 0
                                  ? 'bg-rose-50 text-rose-800'
                                  : 'text-slate-400'
                            }`}
                          >
                            {count}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="lg:col-span-6 bg-white border border-slate-200 rounded-md p-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-200">
              <h3 className="font-display text-base font-bold text-slate-900">
                Per-Class Classification Benchmarks
              </h3>
              <span className="text-xs font-mono tabular-nums text-slate-600">
                Macro F1: {(evaluation.macroF1 * 100).toFixed(1)}%
              </span>
            </div>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-[11px] text-slate-500 uppercase tracking-wider">
                    <th className="py-2 text-left font-semibold">Pathology Class</th>
                    <th className="py-2 text-right font-semibold">Precision</th>
                    <th className="py-2 text-right font-semibold">Recall</th>
                    <th className="py-2 text-right font-semibold">F1 Score</th>
                    <th className="py-2 text-right font-semibold">Support</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {evaluation.perClass.map((m) => (
                    <tr key={m.classId}>
                      <td className="py-2.5 font-medium text-slate-800">
                        {DISEASE_PROFILES[m.classId].shortName}
                      </td>
                      <td className="py-2.5 text-right font-mono tabular-nums text-slate-700">
                        {(m.precision * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 text-right font-mono tabular-nums text-slate-700">
                        {(m.recall * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 text-right font-mono tabular-nums font-semibold text-slate-900">
                        {(m.f1 * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 text-right font-mono tabular-nums text-slate-500">
                        {m.support}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Row 3: Spectrogram Training Dataset Gallery */}
      <div className="bg-white border border-slate-200 rounded-md p-5">
        <div className="pb-3 border-b border-slate-200">
          <h3 className="font-display text-base font-bold text-slate-900">
            Mel-Spectrogram Training Dataset Samples
          </h3>
          <p className="text-xs text-slate-600 mt-1">
            Representative 64×64 audio-derived Mel-spectrograms across all four classes. Click any sample to inspect and classify it in Spectrogram Lab.
          </p>
        </div>

        <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
          {previewThumbnails.map(({ sample, dataUrl }) => (
            <button
              key={sample.id}
              type="button"
              onClick={() => onLoadDatasetSample(sample)}
              className="text-left border border-slate-200 hover:border-sky-400 rounded-md p-2 transition-colors bg-white cursor-pointer"
            >
              <img
                src={dataUrl}
                alt={`${DISEASE_PROFILES[sample.classId].shortName} spectrogram`}
                className="w-full aspect-square object-cover rounded-sm bg-slate-950"
              />
              <div className="mt-1.5 text-xs font-semibold text-slate-900 truncate">
                {DISEASE_PROFILES[sample.classId].shortName}
              </div>
              <div className="text-[11px] font-mono tabular-nums text-slate-500">
                Peak {sample.spectrogram.biomarkers.peakFreqHz} Hz
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
