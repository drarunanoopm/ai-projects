import { Download, Mic, Play, Square, Upload } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { CLINICAL_SAMPLE_PRESETS, DISEASE_PROFILES } from '../audio/cattleSynthesizer';
import type { AudioSamplePreset, DspConfig } from '../types/bioacoustics';

type InputSourceMode = 'library' | 'upload' | 'microphone';

interface AcousticControlsPanelProps {
  selectedPresetId: string;
  onSelectPreset: (preset: AudioSamplePreset) => void;
  onAudioLoaded: (samples: Float32Array, sampleRate: number, sourceName: string) => void;
  onPlayAudio: () => void;
  onStopAudio: () => void;
  isPlaying: boolean;
  onDownloadWav: () => void;
  dspConfig: DspConfig;
  onChangeDspConfig: (next: DspConfig) => void;
  onRunClassification: () => void;
  isClassifying: boolean;
  isModelReady: boolean;
  uploadedFileName: string | null;
  triggerUploadRef: React.MutableRefObject<(() => void) | null>;
}

export function AcousticControlsPanel({
  selectedPresetId,
  onSelectPreset,
  onAudioLoaded,
  onPlayAudio,
  onStopAudio,
  isPlaying,
  onDownloadWav,
  dspConfig,
  onChangeDspConfig,
  onRunClassification,
  isClassifying,
  isModelReady,
  uploadedFileName,
  triggerUploadRef,
}: AcousticControlsPanelProps) {
  const [sourceMode, setSourceMode] = useState<InputSourceMode>('library');
  const [isRecording, setIsRecording] = useState(false);
  const [micLevel, setMicLevel] = useState(0);
  const [micError, setMicError] = useState<string | null>(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const animFrameRef = useRef<number | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    triggerUploadRef.current = () => {
      setSourceMode('upload');
      fileInputRef.current?.click();
    };
  }, [triggerUploadRef]);

  useEffect(() => {
    return () => {
      stopMicCaptureCleanup();
    };
  }, []);

  const stopMicCaptureCleanup = () => {
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setMicError(null);
    try {
      const arrayBuffer = await file.arrayBuffer();
      const ctx = new AudioContext();
      const decoded = await ctx.decodeAudioData(arrayBuffer);
      const channelData = decoded.getChannelData(0);
      const copy = new Float32Array(channelData.length);
      copy.set(channelData);
      await ctx.close();
      setSourceMode('upload');
      onAudioLoaded(copy, decoded.sampleRate, file.name);
    } catch {
      setMicError('Could not decode audio file. Please use a standard WAV, MP3, or OGG file.');
    } finally {
      e.target.value = '';
    }
  };

  const startRecording = async () => {
    setMicError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const updateMeter = () => {
        analyser.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const v = (dataArray[i] - 128) / 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / dataArray.length);
        setMicLevel(Math.min(1, rms * 3.5));
        animFrameRef.current = requestAnimationFrame(updateMeter);
      };
      updateMeter();

      chunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;

      recorder.ondataavailable = (ev) => {
        if (ev.data.size > 0) chunksRef.current.push(ev.data);
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        stopMicCaptureCleanup();
        setIsRecording(false);
        setMicLevel(0);
        setRecordingSeconds(0);

        try {
          const buf = await blob.arrayBuffer();
          const decodeCtx = new AudioContext();
          const decoded = await decodeCtx.decodeAudioData(buf);
          const channelData = decoded.getChannelData(0);
          const copy = new Float32Array(channelData.length);
          copy.set(channelData);
          await decodeCtx.close();
          onAudioLoaded(copy, decoded.sampleRate, `Mic Capture (${new Date().toLocaleTimeString()})`);
        } catch {
          setMicError('Microphone buffer could not be decoded. Try recording for at least 1 second.');
        }
      };

      recorder.start();
      setIsRecording(true);
      setRecordingSeconds(0);
      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => prev + 0.1);
      }, 100);
    } catch {
      setMicError('Microphone access denied or unavailable in this browser.');
      stopMicCaptureCleanup();
    }
  };

  const stopRecording = () => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
    }
  };

  return (
    <aside className="w-full lg:w-[340px] shrink-0 bg-white border border-slate-200 rounded-md flex flex-col">
      {/* Hidden global file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Section 1: Acquisition Source */}
      <div className="p-4 border-b border-slate-200">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
          01. Acoustic Signal Source
        </h2>

        <div className="grid grid-cols-3 gap-1 p-1 bg-slate-100 rounded-md mb-4">
          <button
            type="button"
            onClick={() => setSourceMode('library')}
            className={`px-2 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              sourceMode === 'library'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Field Library
          </button>
          <button
            type="button"
            onClick={() => setSourceMode('upload')}
            className={`px-2 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              sourceMode === 'upload'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Upload File
          </button>
          <button
            type="button"
            onClick={() => setSourceMode('microphone')}
            className={`px-2 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              sourceMode === 'microphone'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Microphone
          </button>
        </div>

        {sourceMode === 'library' && (
          <div className="flex flex-col gap-1.5 max-h-64 overflow-y-auto pr-1">
            {CLINICAL_SAMPLE_PRESETS.map((preset) => {
              const selected = preset.id === selectedPresetId && !uploadedFileName;
              const profile = DISEASE_PROFILES[preset.trueClass];
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => onSelectPreset(preset)}
                  className={`text-left p-2.5 rounded-md border transition-colors cursor-pointer ${
                    selected
                      ? 'bg-sky-50/70 border-sky-300 text-slate-900'
                      : 'bg-white border-slate-200 hover:bg-slate-50 text-slate-700'
                  }`}
                >
                  <div className="text-xs font-semibold text-slate-900 truncate">
                    {preset.label}
                  </div>
                  <div className="text-[11px] text-slate-500 mt-0.5 flex items-center gap-1.5">
                    <span>{profile.shortName}</span>
                    <span aria-hidden="true">·</span>
                    <span className="truncate">{preset.barnZone}</span>
                  </div>
                </button>
              );
            })}
          </div>
        )}

        {sourceMode === 'upload' && (
          <div className="flex flex-col gap-3">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="border border-dashed border-slate-300 hover:border-sky-500 hover:bg-sky-50/40 rounded-md p-5 flex flex-col items-center justify-center gap-2 text-center transition-colors cursor-pointer"
            >
              <Upload className="w-5 h-5 text-sky-700" />
              <span className="text-xs font-medium text-slate-800">
                Select Cattle Cough Recording
              </span>
              <span className="text-[11px] text-slate-500">
                WAV, MP3, OGG, or M4A · Auto-resampled to 16 kHz
              </span>
            </button>
            {uploadedFileName && (
              <div className="text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded px-3 py-2 font-mono truncate">
                Loaded: {uploadedFileName}
              </div>
            )}
          </div>
        )}

        {sourceMode === 'microphone' && (
          <div className="flex flex-col gap-3">
            <div className="bg-slate-50 border border-slate-200 rounded-md p-3">
              <div className="flex items-center justify-between text-xs text-slate-600 mb-2">
                <span>Input Gain Level</span>
                <span className="font-mono tabular-nums">
                  {isRecording ? `${recordingSeconds.toFixed(1)}s` : 'Standby'}
                </span>
              </div>
              <div className="w-full h-2 bg-slate-200 rounded-sm overflow-hidden">
                <div
                  className="h-full bg-sky-600 transition-transform duration-75 origin-left"
                  style={{ transform: `scaleX(${micLevel})` }}
                />
              </div>
            </div>

            {!isRecording ? (
              <button
                type="button"
                onClick={startRecording}
                className="w-full py-2 px-4 rounded-md bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium flex items-center justify-center gap-2 transition-colors cursor-pointer"
              >
                <Mic className="w-3.5 h-3.5" />
                Start Recording
              </button>
            ) : (
              <button
                type="button"
                onClick={stopRecording}
                className="w-full py-2 px-4 rounded-md bg-rose-600 hover:bg-rose-700 text-white text-xs font-medium flex items-center justify-center gap-2 transition-colors cursor-pointer"
              >
                <Square className="w-3.5 h-3.5" />
                Stop &amp; Analyze Cough
              </button>
            )}
          </div>
        )}

        {micError && (
          <p className="mt-2.5 text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded px-2.5 py-2">
            {micError}
          </p>
        )}

        {/* Playback & WAV Export */}
        <div className="mt-4 pt-3 border-t border-slate-200 flex items-center gap-2">
          <button
            type="button"
            onClick={isPlaying ? onStopAudio : onPlayAudio}
            className="flex-1 py-2 px-3 border border-slate-300 hover:bg-slate-50 text-slate-800 rounded-md text-xs font-medium flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            {isPlaying ? <Square className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            {isPlaying ? 'Stop Audio' : 'Play Audio'}
          </button>
          <button
            type="button"
            onClick={onDownloadWav}
            className="py-2 px-3 border border-slate-300 hover:bg-slate-50 text-slate-700 rounded-md text-xs font-medium flex items-center justify-center gap-1.5 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            title="Download 16kHz PCM WAV"
          >
            <Download className="w-3.5 h-3.5" />
            WAV
          </button>
        </div>
      </div>

      {/* Section 2: DSP Spectrogram Parameters */}
      <div className="p-4 border-b border-slate-200 flex flex-col gap-3.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          02. STFT &amp; Mel-Filterbank DSP
        </h2>

        <div className="flex items-center justify-between gap-3">
          <label htmlFor="fft-size-select" className="text-xs text-slate-700 font-medium">
            FFT Window Size
          </label>
          <select
            id="fft-size-select"
            value={dspConfig.fftSize}
            onChange={(e) =>
              onChangeDspConfig({
                ...dspConfig,
                fftSize: Number(e.target.value) as 512 | 1024,
              })
            }
            className="text-xs font-mono bg-slate-50 border border-slate-200 rounded px-2 py-1 text-slate-800"
          >
            <option value={512}>512 pts (32 ms)</option>
            <option value={1024}>1024 pts (64 ms)</option>
          </select>
        </div>

        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-slate-700 font-medium">Frequency Ceiling</span>
            <span className="font-mono tabular-nums text-slate-600">{dspConfig.fMax} Hz</span>
          </div>
          <input
            type="range"
            min={4000}
            max={8000}
            step={500}
            value={dspConfig.fMax}
            onChange={(e) =>
              onChangeDspConfig({ ...dspConfig, fMax: Number(e.target.value) })
            }
            className="w-full accent-sky-700 cursor-pointer"
          />
        </div>

        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-slate-700 font-medium">Noise Floor Threshold</span>
            <span className="font-mono tabular-nums text-slate-600">{dspConfig.minDb} dB</span>
          </div>
          <input
            type="range"
            min={-90}
            max={-45}
            step={5}
            value={dspConfig.minDb}
            onChange={(e) =>
              onChangeDspConfig({ ...dspConfig, minDb: Number(e.target.value) })
            }
            className="w-full accent-sky-700 cursor-pointer"
          />
        </div>
      </div>

      {/* Section 3: Inference Trigger */}
      <div className="p-4 mt-auto">
        <button
          type="button"
          onClick={onRunClassification}
          disabled={!isModelReady || isClassifying}
          className="w-full py-2.5 px-4 rounded-md bg-sky-700 hover:bg-sky-800 disabled:bg-slate-300 disabled:text-slate-500 text-white text-xs font-semibold transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          {!isModelReady
            ? 'Training CNN Model...'
            : isClassifying
              ? 'Running Conv2D Inference...'
              : 'Classify Spectrogram'}
        </button>
        <p className="mt-2 text-[11px] text-slate-500 leading-relaxed">
          Computes 64×64 Log-Mel tensor, runs 3-block Conv2D inference, and generates Grad-CAM spatial saliency.
        </p>
      </div>
    </aside>
  );
}
