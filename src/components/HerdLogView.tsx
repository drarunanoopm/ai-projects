import { Download, Trash2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { DISEASE_CLASSES, DISEASE_PROFILES } from '../audio/cattleSynthesizer';
import type { DiseaseClassId, HerdScreeningRecord } from '../types/bioacoustics';

interface HerdLogViewProps {
  records: HerdScreeningRecord[];
  onDeleteRecord: (id: string) => void;
}

export function HerdLogView({ records, onDeleteRecord }: HerdLogViewProps) {
  const [filterClass, setFilterClass] = useState<'all' | DiseaseClassId>('all');
  const [searchTerm, setSearchTerm] = useState('');

  const filtered = useMemo(() => {
    const q = searchTerm.trim().toLowerCase();
    return records.filter((rec) => {
      if (filterClass !== 'all' && rec.predictedClass !== filterClass) return false;
      if (
        q &&
        !rec.cowTag.toLowerCase().includes(q) &&
        !rec.barnZone.toLowerCase().includes(q) &&
        !rec.sourceName.toLowerCase().includes(q)
      ) {
        return false;
      }
      return true;
    });
  }, [records, filterClass, searchTerm]);

  const exportCsv = () => {
    const headers = [
      'ID',
      'Timestamp',
      'CowTag',
      'BarnZone',
      'Source',
      'PredictedClass',
      'ConfidencePct',
      'PeakFreqHz',
      'SpectralCentroidHz',
      'DurationMs',
      'HNR_dB',
      'Notes',
    ];
    const rows = filtered.map((r) => [
      r.id,
      r.timestamp,
      r.cowTag,
      r.barnZone,
      r.sourceName,
      DISEASE_PROFILES[r.predictedClass].name,
      (r.confidence * 100).toFixed(1),
      r.biomarkers.peakFreqHz,
      r.biomarkers.spectralCentroidHz,
      r.biomarkers.coughDurationMs,
      r.biomarkers.hnrDb.toFixed(1),
      r.notes.replace(/,/g, ';'),
    ]);

    const csv = [headers.join(','), ...rows.map((row) => row.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bovacoustic-herd-log-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-md p-5 flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <h2 className="font-display text-lg font-bold text-slate-900">
            Herd Respiratory Acoustic Screening Ledger
          </h2>
          <p className="text-xs text-slate-600 mt-0.5">
            Logged cattle cough classifications, Mel-spectrogram thumbnails, and quantitative biomarkers.
          </p>
        </div>

        <button
          type="button"
          onClick={exportCsv}
          disabled={filtered.length === 0}
          className="px-3 py-2 rounded-md border border-slate-300 hover:bg-slate-50 disabled:opacity-40 text-xs font-medium text-slate-800 flex items-center gap-1.5 whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Download className="w-3.5 h-3.5" />
          Export CSV
        </button>
      </div>

      {/* Filter & Search Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md overflow-x-auto">
          <button
            type="button"
            onClick={() => setFilterClass('all')}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              filterClass === 'all'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            All ({records.length})
          </button>
          {DISEASE_CLASSES.map((cls) => (
            <button
              key={cls}
              type="button"
              onClick={() => setFilterClass(cls)}
              className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                filterClass === cls
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {DISEASE_PROFILES[cls].shortName}
            </button>
          ))}
        </div>

        <input
          type="search"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Filter by ear tag or barn zone..."
          aria-label="Filter herd log"
          className="w-full sm:w-64 text-xs bg-slate-50 border border-slate-200 rounded-md px-3 py-1.5 text-slate-900"
        />
      </div>

      {filtered.length === 0 ? (
        <div className="py-12 text-center text-xs text-slate-500 border border-dashed border-slate-200 rounded-md">
          No acoustic screening records match the active filter.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-[11px] uppercase tracking-wider text-slate-500">
                <th className="py-2.5 pr-3 font-semibold">Spectrogram</th>
                <th className="py-2.5 pr-3 font-semibold">Cow Tag / Zone</th>
                <th className="py-2.5 pr-3 font-semibold">Predicted Pathology</th>
                <th className="py-2.5 pr-3 font-semibold text-right">Confidence</th>
                <th className="py-2.5 pr-3 font-semibold text-right">Peak Freq</th>
                <th className="py-2.5 pr-3 font-semibold text-right">Duration</th>
                <th className="py-2.5 pr-3 font-semibold">Recorded</th>
                <th className="py-2.5 text-right font-semibold">Remove</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filtered.map((rec) => {
                const prof = DISEASE_PROFILES[rec.predictedClass];
                const riskTextColor =
                  prof.riskLevel === 'normal'
                    ? 'text-emerald-700'
                    : prof.riskLevel === 'moderate'
                      ? 'text-amber-700'
                      : prof.riskLevel === 'high'
                        ? 'text-orange-700'
                        : 'text-rose-700';

                return (
                  <tr key={rec.id} className="hover:bg-slate-50">
                    <td className="py-2.5 pr-3">
                      <img
                        src={rec.spectrogramDataUrl}
                        alt={`${rec.cowTag} spectrogram`}
                        className="w-11 h-11 rounded-sm border border-slate-200 object-cover bg-slate-950"
                      />
                    </td>
                    <td className="py-2.5 pr-3">
                      <div className="font-mono font-semibold text-slate-900">{rec.cowTag}</div>
                      <div className="text-[11px] text-slate-500">{rec.barnZone}</div>
                    </td>
                    <td className="py-2.5 pr-3">
                      <div className={`font-medium ${riskTextColor}`}>
                        {prof.riskGlyph} {prof.name}
                      </div>
                      <div className="text-[11px] text-slate-500 truncate max-w-xs">
                        {rec.notes || rec.sourceName}
                      </div>
                    </td>
                    <td className="py-2.5 pr-3 text-right font-mono tabular-nums font-semibold text-slate-900">
                      {(rec.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 pr-3 text-right font-mono tabular-nums text-slate-700">
                      {rec.biomarkers.peakFreqHz} Hz
                    </td>
                    <td className="py-2.5 pr-3 text-right font-mono tabular-nums text-slate-700">
                      {rec.biomarkers.coughDurationMs} ms
                    </td>
                    <td className="py-2.5 pr-3 font-mono tabular-nums text-slate-500">
                      {rec.timestamp}
                    </td>
                    <td className="py-2.5 text-right">
                      <button
                        type="button"
                        onClick={() => onDeleteRecord(rec.id)}
                        aria-label={`Delete record ${rec.cowTag}`}
                        className="p-1.5 text-slate-400 hover:text-rose-600 rounded transition-colors cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
