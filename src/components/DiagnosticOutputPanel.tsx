import { useEffect, useState } from 'react';
import { DISEASE_CLASSES, DISEASE_PROFILES } from '../audio/cattleSynthesizer';
import type { AcousticBiomarkers, DiseaseClassId, PredictionResult } from '../types/bioacoustics';

interface DiagnosticOutputPanelProps {
  prediction: PredictionResult | null;
  biomarkers: AcousticBiomarkers;
  defaultCowTag: string;
  defaultBarnZone: string;
  isTraining: boolean;
  trainingEpochProgress: string;
  onSaveToHerdLog: (cowTag: string, barnZone: string, notes: string) => void;
}

function getRiskToneClasses(riskLevel: string): string {
  switch (riskLevel) {
    case 'normal':
      return 'text-emerald-700 bg-emerald-50 border-emerald-200';
    case 'moderate':
      return 'text-amber-700 bg-amber-50 border-amber-200';
    case 'high':
      return 'text-orange-700 bg-orange-50 border-orange-200';
    case 'critical':
      return 'text-rose-700 bg-rose-50 border-rose-200';
    default:
      return 'text-slate-700 bg-slate-50 border-slate-200';
  }
}

function getBarColorClass(classId: DiseaseClassId, isWinner: boolean): string {
  if (!isWinner) return 'bg-slate-300';
  switch (classId) {
    case 'healthy':
      return 'bg-emerald-600';
    case 'brd':
      return 'bg-amber-600';
    case 'lungworm':
      return 'bg-orange-600';
    case 'pneumonia':
      return 'bg-rose-600';
  }
}

export function DiagnosticOutputPanel({
  prediction,
  biomarkers,
  defaultCowTag,
  defaultBarnZone,
  isTraining,
  trainingEpochProgress,
  onSaveToHerdLog,
}: DiagnosticOutputPanelProps) {
  const [cowTag, setCowTag] = useState(defaultCowTag);
  const [barnZone, setBarnZone] = useState(defaultBarnZone);
  const [notes, setNotes] = useState('');
  const [savedNotice, setSavedNotice] = useState(false);

  useEffect(() => {
    setCowTag(defaultCowTag);
    setBarnZone(defaultBarnZone);
  }, [defaultCowTag, defaultBarnZone]);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    onSaveToHerdLog(cowTag.trim() || defaultCowTag, barnZone.trim() || defaultBarnZone, notes.trim());
    setNotes('');
    setSavedNotice(true);
    window.setTimeout(() => setSavedNotice(false), 2500);
  };

  if (isTraining && !prediction) {
    return (
      <section className="bg-white border border-slate-200 rounded-md p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              Initializing 2D Convolutional Neural Network
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Training Conv2D filters on 4-class bovine Mel-spectrogram dataset ({trainingEpochProgress}).
            </p>
          </div>
          <span className="text-xs font-mono tabular-nums text-sky-700 font-medium">
            {trainingEpochProgress}
          </span>
        </div>
      </section>
    );
  }

  const predictedProfile = prediction ? DISEASE_PROFILES[prediction.predictedClass] : null;

  return (
    <section className="grid grid-cols-1 xl:grid-cols-12 gap-5">
      {/* Left 7 cols: CNN Classification & Softmax Distribution */}
      <div className="xl:col-span-7 bg-white border border-slate-200 rounded-md p-5 flex flex-col gap-5">
        {prediction && predictedProfile ? (
          <>
            <div className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-slate-200">
              <div>
                <div className="text-xs font-medium text-slate-500 mb-1">
                  Primary CNN Classification · Inference {prediction.inferenceTimeMs} ms
                </div>
                <h3 className="font-display text-xl font-bold text-slate-900 text-balance">
                  {predictedProfile.name}
                </h3>
                <div className="text-xs text-slate-600 mt-1">
                  Suspected Etiology: {predictedProfile.pathogen}
                </div>
              </div>

              <div className="flex flex-col items-end gap-1.5">
                <div className="text-2xl font-mono font-bold tabular-nums text-slate-900">
                  {(prediction.confidence * 100).toFixed(1)}
                  <span className="text-xs font-normal text-slate-500 ml-1">%</span>
                </div>
                <span
                  className={`text-xs font-medium px-2.5 py-0.5 rounded border ${getRiskToneClasses(
                    predictedProfile.riskLevel
                  )}`}
                >
                  {predictedProfile.riskGlyph} {predictedProfile.riskLabel}
                </span>
              </div>
            </div>

            {/* 4-Class Softmax Probability Distribution */}
            <div className="flex flex-col gap-2.5">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Softmax Class Probabilities
              </div>
              {DISEASE_CLASSES.map((cls) => {
                const prob = prediction.probabilities[cls] ?? 0;
                const prof = DISEASE_PROFILES[cls];
                const isWinner = cls === prediction.predictedClass;
                return (
                  <div key={cls} className="grid grid-cols-12 items-center gap-3 text-xs">
                    <div
                      className={`col-span-4 truncate ${
                        isWinner ? 'font-semibold text-slate-900' : 'text-slate-600'
                      }`}
                    >
                      {prof.shortName}
                    </div>
                    <div className="col-span-6 h-2.5 bg-slate-100 rounded-sm overflow-hidden">
                      <div
                        className={`h-full transition-all duration-300 ${getBarColorClass(
                          cls,
                          isWinner
                        )}`}
                        style={{ width: `${Math.max(1, prob * 100)}%` }}
                      />
                    </div>
                    <div className="col-span-2 text-right font-mono tabular-nums text-slate-700">
                      {(prob * 100).toFixed(1)}%
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Clinical Guidance & Acoustic Signature */}
            <div className="pt-4 border-t border-slate-200 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div>
                <div className="font-semibold text-slate-800 mb-1">Acoustic Morphology</div>
                <p className="text-slate-600 leading-relaxed">
                  {predictedProfile.acousticSignature}
                </p>
              </div>
              <div>
                <div className="font-semibold text-slate-800 mb-1">Recommended Herd Action</div>
                <p className="text-slate-600 leading-relaxed">
                  {predictedProfile.clinicalAction}
                </p>
              </div>
            </div>
          </>
        ) : (
          <div className="py-10 text-center text-xs text-slate-500">
            Select or record a cough audio sample and click Classify Spectrogram.
          </div>
        )}
      </div>

      {/* Right 5 cols: Quantitative Bioacoustic Biomarkers & Herd Log Action */}
      <div className="xl:col-span-5 bg-white border border-slate-200 rounded-md p-5 flex flex-col justify-between gap-5">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
            Extracted Bioacoustic Biomarkers
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-3 pb-4 border-b border-slate-200">
            <div>
              <div className="text-[11px] text-slate-500">Peak Frequency</div>
              <div className="text-base font-mono font-semibold tabular-nums text-slate-900">
                {biomarkers.peakFreqHz}
                <span className="text-[11px] font-normal text-slate-500 ml-1">Hz</span>
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-500">Spectral Centroid</div>
              <div className="text-base font-mono font-semibold tabular-nums text-slate-900">
                {biomarkers.spectralCentroidHz}
                <span className="text-[11px] font-normal text-slate-500 ml-1">Hz</span>
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-500">Active Cough Duration</div>
              <div className="text-base font-mono font-semibold tabular-nums text-slate-900">
                {biomarkers.coughDurationMs}
                <span className="text-[11px] font-normal text-slate-500 ml-1">ms</span>
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-500">Harmonic-to-Noise</div>
              <div className="text-base font-mono font-semibold tabular-nums text-slate-900">
                {biomarkers.hnrDb.toFixed(1)}
                <span className="text-[11px] font-normal text-slate-500 ml-1">dB</span>
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-500">Zero-Crossing Rate</div>
              <div className="text-base font-mono font-semibold tabular-nums text-slate-900">
                {biomarkers.zeroCrossingRate.toFixed(4)}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-500">Detected Bursts</div>
              <div className="text-base font-mono font-semibold tabular-nums text-slate-900">
                {biomarkers.burstCount}
                <span className="text-[11px] font-normal text-slate-500 ml-1">pulses</span>
              </div>
            </div>
          </div>
        </div>

        {/* Log to Herd Registry */}
        <form onSubmit={handleSave} className="flex flex-col gap-2.5">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Record to Herd Screening Log
          </div>
          <div className="grid grid-cols-2 gap-2">
            <input
              type="text"
              value={cowTag}
              onChange={(e) => setCowTag(e.target.value)}
              placeholder="Ear Tag (e.g. HF-408)"
              aria-label="Cow ear tag"
              className="text-xs bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 text-slate-900 font-mono"
            />
            <input
              type="text"
              value={barnZone}
              onChange={(e) => setBarnZone(e.target.value)}
              placeholder="Barn / Pen Zone"
              aria-label="Barn zone"
              className="text-xs bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 text-slate-900"
            />
          </div>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Clinical observation notes (optional)"
              aria-label="Clinical observation notes"
              className="flex-1 text-xs bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 text-slate-900"
            />
            <button
              type="submit"
              disabled={!prediction}
              className="px-3 py-1.5 rounded bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 text-white text-xs font-medium whitespace-nowrap shrink-0 cursor-pointer"
            >
              {savedNotice ? 'Saved' : 'Log Screening'}
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}
