import { Mail, Stethoscope } from 'lucide-react';
import { DISEASE_CLASSES, DISEASE_PROFILES } from '../audio/cattleSynthesizer';
import type { WorkspaceTab } from '../types/bioacoustics';

interface AboutViewProps {
  onNavigateTab: (tab: WorkspaceTab) => void;
}

export function AboutView({ onNavigateTab }: AboutViewProps) {
  return (
    <div className="flex flex-col gap-6">
      {/* Researcher Profile & Project Overview */}
      <section className="bg-white border border-slate-200 rounded-md p-6">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 pb-6 border-b border-slate-200">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-md bg-sky-50 border border-sky-200 flex items-center justify-center shrink-0">
              <Stethoscope className="w-6 h-6 text-sky-700" />
            </div>
            <div>
              <div className="text-xs font-medium text-slate-500">
                Principal Researcher &amp; Author
              </div>
              <h1 className="font-display text-2xl font-bold text-slate-900 mt-0.5">
                Dr.Arun A M
              </h1>
              <p className="text-sm text-slate-600 mt-1 max-w-2xl leading-relaxed">
                Computer Science &amp; Engineering researcher developing deep learning bioacoustic diagnostics and automated spectrogram classification pipelines for precision veterinary medicine and livestock respiratory screening.
              </p>
            </div>
          </div>

          <div className="flex flex-col items-start md:items-end gap-2 shrink-0">
            <a
              href="mailto:dr.aam.cse@gmail.com"
              className="inline-flex items-center gap-2 px-3 py-2 text-xs font-mono text-slate-700 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md transition-colors whitespace-nowrap shrink-0"
            >
              <Mail className="w-3.5 h-3.5 text-sky-700" />
              dr.aam.cse@gmail.com
            </a>
            <span className="text-xs text-slate-500">
              BovAcoustic Research Workbench
            </span>
          </div>
        </div>

        {/* Research Objectives & Pipeline Architecture */}
        <div className="pt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h2 className="text-sm font-semibold text-slate-900 mb-1.5">
              01. Bioacoustic Signal Processing
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              Raw cattle cough recordings are resampled to 16 kHz, centered around peak acoustic energy, and transformed using a Radix-2 Cooley-Tukey FFT with Hann windowing into 64-band Log-Mel spectrogram tensors spanning 60 Hz to 8,000 Hz.
            </p>
          </div>

          <div>
            <h2 className="text-sm font-semibold text-slate-900 mb-1.5">
              02. 2D CNN &amp; Grad-CAM Explainability
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              A three-block 2D Convolutional Neural Network classifies time-frequency cough morphology in-browser while computing Grad-CAM spatial attention maps so veterinarians can verify the exact harmonic bands driving each prediction.
            </p>
          </div>

          <div>
            <h2 className="text-sm font-semibold text-slate-900 mb-1.5">
              03. Herd Respiratory Triage
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              Quantitative acoustic biomarkers (peak frequency, spectral centroid, active cough duration, harmonic-to-noise ratio, and burst count) are paired with ear-tag logging and CSV export for rapid barn-level screening.
            </p>
          </div>
        </div>

        <div className="mt-6 pt-5 border-t border-slate-200 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => onNavigateTab('lab')}
            className="px-4 py-2 text-xs font-medium text-white bg-sky-700 hover:bg-sky-800 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Open Spectrogram Lab
          </button>
          <button
            type="button"
            onClick={() => onNavigateTab('dataset')}
            className="px-4 py-2 text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-300 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Explore Dataset &amp; Training
          </button>
        </div>
      </section>

      {/* Clinical Pathology Acoustic Reference */}
      <section className="bg-white border border-slate-200 rounded-md p-6">
        <div className="pb-3 border-b border-slate-200">
          <h2 className="font-display text-base font-bold text-slate-900">
            Bovine Respiratory Acoustic Reference Matrix
          </h2>
          <p className="text-xs text-slate-600 mt-0.5">
            Target respiratory classifications modeled and evaluated within the BovAcoustic diagnostic pipeline.
          </p>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-[11px] uppercase tracking-wider text-slate-500">
                <th className="py-2.5 pr-4 font-semibold">Classification</th>
                <th className="py-2.5 pr-4 font-semibold">Primary Etiology</th>
                <th className="py-2.5 pr-4 font-semibold">Dominant Band</th>
                <th className="py-2.5 pr-4 font-semibold">Typical Duration</th>
                <th className="py-2.5 font-semibold">Acoustic Characteristics</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {DISEASE_CLASSES.map((cls) => {
                const prof = DISEASE_PROFILES[cls];
                return (
                  <tr key={cls} className="align-top">
                    <td className="py-3 pr-4 font-semibold text-slate-900 whitespace-nowrap">
                      {prof.name}
                    </td>
                    <td className="py-3 pr-4 text-slate-600">{prof.pathogen}</td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-sky-700 whitespace-nowrap">
                      {prof.frequencyBandLabel}
                    </td>
                    <td className="py-3 pr-4 font-mono tabular-nums text-slate-700 whitespace-nowrap">
                      {prof.typicalDurationLabel}
                    </td>
                    <td className="py-3 text-slate-600 leading-relaxed">
                      {prof.acousticSignature}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
