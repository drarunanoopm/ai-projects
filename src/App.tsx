import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CLINICAL_SAMPLE_PRESETS,
  encodeWavBlob,
  synthesizeCattleAudio,
} from './audio/cattleSynthesizer';
import {
  computeMelSpectrogram,
  DEFAULT_DSP_CONFIG,
  prepareCoughWindow,
  renderSpectrogramThumbnail,
} from './audio/dspSpectrogram';
import { AboutView } from './components/AboutView';
import { AcousticControlsPanel } from './components/AcousticControlsPanel';
import { CnnArchitectureView } from './components/CnnArchitectureView';
import { DatasetTrainingView } from './components/DatasetTrainingView';
import { DiagnosticOutputPanel } from './components/DiagnosticOutputPanel';
import { HerdLogView } from './components/HerdLogView';
import { SpectrogramCanvas } from './components/SpectrogramCanvas';
import { TopBar } from './components/TopBar';
import { WaveformCanvas } from './components/WaveformCanvas';
import {
  classifySpectrogram,
  type DatasetSamplePreview,
  DEFAULT_TRAINING_CONFIG,
  isCnnReady,
  trainCattleCnn,
} from './ml/cattleCnn';
import type {
  AudioSamplePreset,
  ColorMapId,
  DspConfig,
  HerdScreeningRecord,
  ModelEvaluation,
  PredictionResult,
  TrainingConfig,
  TrainingMetricsPoint,
  WorkspaceTab,
} from './types/bioacoustics';

export default function App() {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('lab');
  const [selectedPreset, setSelectedPreset] = useState<AudioSamplePreset>(CLINICAL_SAMPLE_PRESETS[0]);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [sourceTag, setSourceTag] = useState<string>(CLINICAL_SAMPLE_PRESETS[0].cowTag);
  const [sourceZone, setSourceZone] = useState<string>(CLINICAL_SAMPLE_PRESETS[0].barnZone);
  const [sourceName, setSourceName] = useState<string>(CLINICAL_SAMPLE_PRESETS[0].label);

  const [rawAudio, setRawAudio] = useState<Float32Array>(() =>
    synthesizeCattleAudio(CLINICAL_SAMPLE_PRESETS[0].trueClass, CLINICAL_SAMPLE_PRESETS[0].seed)
  );
  const [rawSampleRate, setRawSampleRate] = useState<number>(16000);
  const [dspConfig, setDspConfig] = useState<DspConfig>(DEFAULT_DSP_CONFIG);

  const [colorMap, setColorMap] = useState<ColorMapId>('viridis');
  const [showGradCam, setShowGradCam] = useState<boolean>(true);
  const [gradCamOpacity, setGradCamOpacity] = useState<number>(0.72);

  // CNN training & inference states
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);
  const [isTraining, setIsTraining] = useState<boolean>(true);
  const [trainingHistory, setTrainingHistory] = useState<TrainingMetricsPoint[]>([]);
  const [evaluation, setEvaluation] = useState<ModelEvaluation | null>(null);
  const [datasetPreviews, setDatasetPreviews] = useState<DatasetSamplePreview[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isClassifying, setIsClassifying] = useState<boolean>(false);

  // Herd screening records
  const [herdRecords, setHerdRecords] = useState<HerdScreeningRecord[]>([]);

  // Audio playback state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackProgress, setPlaybackProgress] = useState<number | null>(null);
  const playCtxRef = useRef<AudioContext | null>(null);
  const playSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const playRafRef = useRef<number | null>(null);
  const triggerUploadRef = useRef<(() => void) | null>(null);

  // Compute normalized 1.2s cough window & Mel-spectrogram deterministically from audio + DSP config
  const activeSamples = useMemo(
    () => prepareCoughWindow(rawAudio, rawSampleRate, dspConfig.sampleRate, 1.2),
    [rawAudio, rawSampleRate, dspConfig.sampleRate]
  );

  const spectrogram = useMemo(
    () => computeMelSpectrogram(activeSamples, dspConfig.sampleRate, dspConfig),
    [activeSamples, dspConfig]
  );

  const stopAudio = useCallback(() => {
    if (playRafRef.current !== null) {
      cancelAnimationFrame(playRafRef.current);
      playRafRef.current = null;
    }
    if (playSourceRef.current) {
      try {
        playSourceRef.current.stop();
      } catch {
        // Ignore already-stopped source
      }
      playSourceRef.current.disconnect();
      playSourceRef.current = null;
    }
    if (playCtxRef.current && playCtxRef.current.state !== 'closed') {
      playCtxRef.current.close().catch(() => {});
      playCtxRef.current = null;
    }
    setIsPlaying(false);
    setPlaybackProgress(null);
  }, []);

  const playAudioFromProgress = useCallback(
    (startProgress = 0) => {
      stopAudio();
      const ctx = new AudioContext();
      playCtxRef.current = ctx;

      const buffer = ctx.createBuffer(1, activeSamples.length, dspConfig.sampleRate);
      buffer.getChannelData(0).set(activeSamples);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      playSourceRef.current = source;

      const duration = buffer.duration;
      const offsetSec = Math.max(0, Math.min(duration - 0.02, startProgress * duration));
      const startedAt = ctx.currentTime;

      source.onended = () => {
        stopAudio();
      };
      source.start(0, offsetSec);
      setIsPlaying(true);

      const tick = () => {
        if (!playCtxRef.current) return;
        const elapsed = playCtxRef.current.currentTime - startedAt + offsetSec;
        const prog = elapsed / duration;
        if (prog >= 1) {
          setPlaybackProgress(null);
          return;
        }
        setPlaybackProgress(prog);
        playRafRef.current = requestAnimationFrame(tick);
      };
      playRafRef.current = requestAnimationFrame(tick);
    },
    [activeSamples, dspConfig.sampleRate, stopAudio]
  );

  useEffect(() => {
    return () => {
      stopAudio();
    };
  }, [stopAudio]);

  const runClassification = useCallback(
    async (specOverride?: typeof spectrogram) => {
      if (!isCnnReady()) return;
      setIsClassifying(true);
      try {
        const result = await classifySpectrogram(specOverride ?? spectrogram);
        setPrediction(result);
      } finally {
        setIsClassifying(false);
      }
    },
    [spectrogram]
  );

  const runTrainingPipeline = useCallback(
    async (cfg: TrainingConfig, seedInitialHerdLog = false) => {
      setIsTraining(true);
      setTrainingHistory([]);
      try {
        const res = await trainCattleCnn(cfg, (pt) => {
          setTrainingHistory((prev) => [...prev, pt]);
        });
        setEvaluation(res.evaluation);
        setDatasetPreviews(res.previews);

        // Classify currently active spectrogram immediately after training completes
        const initialPred = await classifySpectrogram(spectrogram);
        setPrediction(initialPred);

        // Populate initial herd screening ledger with real CNN inference on 3 baseline field samples
        if (seedInitialHerdLog) {
          const baselinePresets = [
            CLINICAL_SAMPLE_PRESETS[0],
            CLINICAL_SAMPLE_PRESETS[1],
            CLINICAL_SAMPLE_PRESETS[2],
            CLINICAL_SAMPLE_PRESETS[3],
          ];
          const entries: HerdScreeningRecord[] = [];
          for (const preset of baselinePresets) {
            const audio = synthesizeCattleAudio(preset.trueClass, preset.seed);
            const spec = computeMelSpectrogram(audio, 16000, DEFAULT_DSP_CONFIG);
            const pred = await classifySpectrogram(spec);
            entries.push({
              id: `rec-${preset.cowTag}`,
              cowTag: preset.cowTag,
              barnZone: preset.barnZone,
              timestamp: new Date().toLocaleTimeString(),
              sourceName: preset.label,
              predictedClass: pred.predictedClass,
              confidence: pred.confidence,
              biomarkers: spec.biomarkers,
              spectrogramDataUrl: renderSpectrogramThumbnail(
                spec.normalized,
                spec.width,
                spec.height,
                'viridis'
              ),
              notes: preset.description,
            });
          }
          setHerdRecords(entries);
        }
      } finally {
        setIsTraining(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  // Train CNN automatically on mount
  const didInitRef = useRef(false);
  useEffect(() => {
    if (didInitRef.current) return;
    didInitRef.current = true;
    runTrainingPipeline(DEFAULT_TRAINING_CONFIG, true);
  }, [runTrainingPipeline]);

  // Re-classify automatically whenever user switches sample or tweaks DSP config (after model is ready)
  useEffect(() => {
    if (isTraining || !isCnnReady()) return;
    let cancelled = false;
    (async () => {
      setIsClassifying(true);
      try {
        const res = await classifySpectrogram(spectrogram);
        if (!cancelled) setPrediction(res);
      } finally {
        if (!cancelled) setIsClassifying(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [spectrogram, isTraining]);

  const handleSelectPreset = (preset: AudioSamplePreset) => {
    stopAudio();
    setSelectedPreset(preset);
    setUploadedFileName(null);
    setSourceTag(preset.cowTag);
    setSourceZone(preset.barnZone);
    setSourceName(preset.label);
    setRawSampleRate(16000);
    setRawAudio(synthesizeCattleAudio(preset.trueClass, preset.seed));
  };

  const handleAudioLoaded = (samples: Float32Array, sampleRate: number, name: string) => {
    stopAudio();
    setUploadedFileName(name);
    setSourceName(name);
    setSourceTag('CUSTOM-01');
    setSourceZone('Field Upload');
    setRawSampleRate(sampleRate);
    setRawAudio(samples);
  };

  const handleDownloadWav = () => {
    const blob = encodeWavBlob(activeSamples, dspConfig.sampleRate);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${sourceTag.toLowerCase().replace(/\s+/g, '-')}-cough.wav`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSaveToHerdLog = (cowTag: string, barnZone: string, notes: string) => {
    if (!prediction) return;
    const record: HerdScreeningRecord = {
      id: `rec-${Date.now()}`,
      cowTag,
      barnZone,
      timestamp: new Date().toLocaleTimeString(),
      sourceName,
      predictedClass: prediction.predictedClass,
      confidence: prediction.confidence,
      biomarkers: spectrogram.biomarkers,
      spectrogramDataUrl: renderSpectrogramThumbnail(
        spectrogram.normalized,
        spectrogram.width,
        spectrogram.height,
        colorMap
      ),
      notes: notes || 'Manual diagnostic screening logged.',
    };
    setHerdRecords((prev) => [record, ...prev]);
  };

  const handleLoadDatasetSample = (sample: DatasetSamplePreview) => {
    stopAudio();
    setUploadedFileName(`Dataset (${sample.id})`);
    setSourceName(`Dataset Sample · ${sample.id}`);
    setSourceTag(sample.id.toUpperCase());
    setSourceZone('Training Corpus');
    setRawSampleRate(16000);
    setRawAudio(sample.audio);
    setActiveTab('lab');
  };

  const trainingProgressText =
    trainingHistory.length > 0
      ? `Epoch ${trainingHistory[trainingHistory.length - 1].epoch} / ${trainingConfig.epochs}`
      : 'Preparing Mel-Spectrogram Tensors...';

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <TopBar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onTriggerUpload={() => {
          setActiveTab('lab');
          window.setTimeout(() => triggerUploadRef.current?.(), 40);
        }}
      />

      {/* Precision Telemetry & Viewport Control Ribbon */}
      <div className="bg-white border-b border-slate-200 px-6 py-2 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex flex-wrap items-center gap-3 text-slate-600">
          <span className="font-semibold text-slate-900">{sourceName}</span>
          <span aria-hidden="true">·</span>
          <span className="font-mono tabular-nums">
            {dspConfig.sampleRate.toLocaleString()} Hz
          </span>
          <span aria-hidden="true">·</span>
          <span className="font-mono tabular-nums">Tensor 64×64×1</span>
          <span aria-hidden="true">·</span>
          {isTraining ? (
            <span className="font-mono text-amber-700 font-medium">
              ▲ Training CNN ({trainingProgressText})
            </span>
          ) : evaluation ? (
            <span className="font-mono tabular-nums text-emerald-700 font-medium">
              ● CNN Ready · Val Acc {(evaluation.accuracy * 100).toFixed(1)}%
            </span>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={showGradCam}
              onChange={(e) => setShowGradCam(e.target.checked)}
              className="accent-sky-700 rounded"
            />
            <span className="font-medium">Grad-CAM Saliency</span>
          </label>

          {showGradCam && (
            <div className="flex items-center gap-2">
              <span className="text-slate-500">Overlay</span>
              <input
                type="range"
                min={0.25}
                max={1}
                step={0.05}
                value={gradCamOpacity}
                onChange={(e) => setGradCamOpacity(Number(e.target.value))}
                aria-label="Grad-CAM overlay intensity"
                className="w-20 accent-sky-700 cursor-pointer"
              />
            </div>
          )}

          <div className="flex items-center gap-2">
            <label htmlFor="colormap-select" className="text-slate-500">
              Palette
            </label>
            <select
              id="colormap-select"
              value={colorMap}
              onChange={(e) => setColorMap(e.target.value as ColorMapId)}
              className="font-mono text-xs bg-slate-50 border border-slate-200 rounded px-2 py-1 text-slate-800"
            >
              <option value="viridis">Viridis</option>
              <option value="inferno">Inferno</option>
              <option value="cyanAmber">Cyan-Amber</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Content Stage */}
      <main className="flex-1 w-full max-w-[1440px] mx-auto px-6 py-6">
        {activeTab === 'lab' && (
          <div className="flex flex-col lg:flex-row items-start gap-6">
            <AcousticControlsPanel
              selectedPresetId={selectedPreset.id}
              onSelectPreset={handleSelectPreset}
              onAudioLoaded={handleAudioLoaded}
              onPlayAudio={() => playAudioFromProgress(0)}
              onStopAudio={stopAudio}
              isPlaying={isPlaying}
              onDownloadWav={handleDownloadWav}
              dspConfig={dspConfig}
              onChangeDspConfig={setDspConfig}
              onRunClassification={() => runClassification()}
              isClassifying={isClassifying}
              isModelReady={!isTraining && isCnnReady()}
              uploadedFileName={uploadedFileName}
              triggerUploadRef={triggerUploadRef}
            />

            <div className="flex-1 min-w-0 w-full flex flex-col gap-5">
              {/* Primary Spectrogram & Oscillogram Visualization Card */}
              <section className="bg-white border border-slate-200 rounded-md p-5 flex flex-col gap-4">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
                  <div>
                    <h1 className="font-display text-lg font-bold text-slate-900 text-balance">
                      Log-Mel Acoustic Spectrogram &amp; CNN Attention
                    </h1>
                    <p className="text-xs text-slate-500 mt-0.5">
                      64 Mel-frequency bands ({dspConfig.fMin}–{dspConfig.fMax} Hz) across 1.20 s window · Window: Hann ({dspConfig.fftSize} pts)
                    </p>
                  </div>
                  <div className="text-xs font-mono tabular-nums text-slate-500">
                    Tag: <span className="text-slate-900 font-semibold">{sourceTag}</span> · {sourceZone}
                  </div>
                </div>

                <WaveformCanvas
                  samples={activeSamples}
                  sampleRate={dspConfig.sampleRate}
                  envelope={spectrogram.envelope}
                  playbackProgress={playbackProgress}
                  onSeek={(prog) => playAudioFromProgress(prog)}
                />

                <SpectrogramCanvas
                  spectrogram={spectrogram}
                  prediction={prediction}
                  colorMap={colorMap}
                  showGradCam={showGradCam}
                  gradCamOpacity={gradCamOpacity}
                  playbackProgress={playbackProgress}
                />
              </section>

              {/* CNN Prediction & Biomarkers */}
              <DiagnosticOutputPanel
                prediction={prediction}
                biomarkers={spectrogram.biomarkers}
                defaultCowTag={sourceTag}
                defaultBarnZone={sourceZone}
                isTraining={isTraining}
                trainingEpochProgress={trainingProgressText}
                onSaveToHerdLog={handleSaveToHerdLog}
              />
            </div>
          </div>
        )}

        {activeTab === 'architecture' && (
          <CnnArchitectureView
            prediction={prediction}
            totalParams={evaluation?.totalParams ?? 0}
            colorMap={colorMap}
          />
        )}

        {activeTab === 'dataset' && (
          <DatasetTrainingView
            trainingConfig={trainingConfig}
            onChangeTrainingConfig={setTrainingConfig}
            onRetrain={() => runTrainingPipeline(trainingConfig, false)}
            isTraining={isTraining}
            history={trainingHistory}
            evaluation={evaluation}
            previews={datasetPreviews}
            colorMap={colorMap}
            onLoadDatasetSample={handleLoadDatasetSample}
          />
        )}

        {activeTab === 'herd' && (
          <HerdLogView
            records={herdRecords}
            onDeleteRecord={(id) =>
              setHerdRecords((prev) => prev.filter((item) => item.id !== id))
            }
          />
        )}

        {activeTab === 'about' && <AboutView onNavigateTab={setActiveTab} />}
      </main>

      <footer className="border-t border-slate-200 bg-white px-6 py-3">
        <div className="max-w-[1440px] mx-auto flex flex-wrap items-center justify-between gap-4 text-xs text-slate-500">
          <span>BovAcoustic · Bovine Respiratory Bioacoustics &amp; CNN Diagnostics</span>
          <div className="flex items-center gap-4">
            <span className="text-slate-700 font-medium">Dr.Arun A M</span>
            <button
              type="button"
              onClick={() => setActiveTab('about')}
              className="text-sky-700 hover:underline cursor-pointer"
            >
              About Me
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
