import type { AudioSamplePreset, DiseaseClassId, DiseaseProfile } from '../types/bioacoustics';

export const DISEASE_CLASSES: DiseaseClassId[] = ['healthy', 'brd', 'pneumonia', 'lungworm'];

export const DISEASE_PROFILES: Record<DiseaseClassId, DiseaseProfile> = {
  healthy: {
    id: 'healthy',
    name: 'Healthy / Normal Clearance',
    shortName: 'Healthy',
    pathogen: 'Non-infectious / Physiological',
    riskLevel: 'normal',
    riskGlyph: '●',
    riskLabel: 'Normal',
    frequencyBandLabel: '110 – 480 Hz',
    typicalDurationLabel: '140 – 210 ms',
    acousticSignature:
      'Brief, low-turbulence throat clearance with clean harmonic fundamental (110–145 Hz), rapid exponential decay, and minimal high-frequency broadband noise.',
    clinicalAction:
      'No respiratory intervention indicated. Continue routine barn acoustic monitoring during morning feeding intervals.',
  },
  brd: {
    id: 'brd',
    name: 'Bovine Respiratory Disease (BRD)',
    shortName: 'BRD Complex',
    pathogen: 'Mannheimia haemolytica / BHV-1 / BRSV',
    riskLevel: 'moderate',
    riskGlyph: '▲',
    riskLabel: 'Moderate Risk',
    frequencyBandLabel: '380 – 1,150 Hz',
    typicalDurationLabel: '280 – 460 ms',
    acousticSignature:
      'Dry, forceful paroxysmal double-burst hacking cough with sharp glottal transient onsets and concentrated mid-low tracheal formant energy.',
    clinicalAction:
      'Isolate tagged animal for rectal temperature check (>39.7°C), thoracic auscultation, and early-stage antimicrobial/anti-inflammatory protocol.',
  },
  pneumonia: {
    id: 'pneumonia',
    name: 'Infectious Bovine Pneumonia',
    shortName: 'Pneumonia',
    pathogen: 'Pasteurella multocida / Mycoplasma bovis',
    riskLevel: 'critical',
    riskGlyph: '✖',
    riskLabel: 'High Risk',
    frequencyBandLabel: '950 – 3,900 Hz',
    typicalDurationLabel: '420 – 680 ms',
    acousticSignature:
      'Wet, productive, congested cough featuring prolonged turbulent broadband crackles, irregular mucus-stridor modulation, and elevated upper-band spectral centroid.',
    clinicalAction:
      'Immediate veterinary clinical exam required. Assess nasal discharge, dyspnea, and lung consolidation; initiate targeted herd quarantine.',
  },
  lungworm: {
    id: 'lungworm',
    name: 'Parasitic Bronchitis (Lungworm)',
    shortName: 'Lungworm',
    pathogen: 'Dictyocaulus viviparus',
    riskLevel: 'high',
    riskGlyph: '▲',
    riskLabel: 'Elevated Risk',
    frequencyBandLabel: '1,450 – 3,100 Hz',
    typicalDurationLabel: '320 – 540 ms',
    acousticSignature:
      'Spasmodic husky cough accompanied by sustained high-pitched bronchial wheeze harmonics (1.6–2.8 kHz chirp) and audible inspiratory stridor.',
    clinicalAction:
      'Collect fecal Baermann sample to confirm Dictyocaulus larvae; schedule anthelmintic drench across pasture cohort and rotate grazing paddocks.',
  },
};

export const CLINICAL_SAMPLE_PRESETS: AudioSamplePreset[] = [
  {
    id: 'sample-brd-01',
    label: 'BRD Dry Double-Cough (#HF-408)',
    cowTag: 'HF-408',
    barnZone: 'Pen B-02 (Weaning)',
    trueClass: 'brd',
    description: 'Paroxysmal dry hacking double cough recorded 1.8m from feed bunk.',
    seed: 4081,
  },
  {
    id: 'sample-pneu-01',
    label: 'Wet Productive Pneumonia (#HF-219)',
    cowTag: 'HF-219',
    barnZone: 'Hospital Bay 1',
    trueClass: 'pneumonia',
    description: 'Congested wet cough with high-frequency alveolar crackles and prolonged decay.',
    seed: 2194,
  },
  {
    id: 'sample-lung-01',
    label: 'Parasitic Husk Wheeze (#JS-114)',
    cowTag: 'JS-114',
    barnZone: 'North Pasture Lot',
    trueClass: 'lungworm',
    description: 'High-pitched bronchial wheeze with narrowband resonant stridor around 2.1 kHz.',
    seed: 1147,
  },
  {
    id: 'sample-hlth-01',
    label: 'Normal Feed-Dust Clearing (#HF-082)',
    cowTag: 'HF-082',
    barnZone: 'Milking Parlor A',
    trueClass: 'healthy',
    description: 'Single brief physiological throat clear with clean low-frequency vocal fold fundamental.',
    seed: 823,
  },
  {
    id: 'sample-brd-02',
    label: 'BRD Triple-Burst Cough (#HF-512)',
    cowTag: 'HF-512',
    barnZone: 'Pen B-04 (Transit)',
    trueClass: 'brd',
    description: 'Rapid triple-pulse dry tracheal cough typical of acute shipping fever onset.',
    seed: 5129,
  },
  {
    id: 'sample-pneu-02',
    label: 'Severe Suppurative Cough (#AN-331)',
    cowTag: 'AN-331',
    barnZone: 'Hospital Bay 2',
    trueClass: 'pneumonia',
    description: 'Heavy broadband exudative turbulence spanning 1.2–4.0 kHz with wet burst tail.',
    seed: 3318,
  },
  {
    id: 'sample-lung-02',
    label: 'Bronchial Spasm & Stridor (#JS-207)',
    cowTag: 'JS-207',
    barnZone: 'South Pasture Lot',
    trueClass: 'lungworm',
    description: 'Spasmodic husky expiration followed by descending harmonic bronchial whistle.',
    seed: 2075,
  },
  {
    id: 'sample-hlth-02',
    label: 'Normal Calm Exhalation (#HF-105)',
    cowTag: 'HF-105',
    barnZone: 'Dry Cow Barn C',
    trueClass: 'healthy',
    description: 'Gentle low-energy breath clearing without airway obstruction or crackles.',
    seed: 1056,
  },
];

function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRandom(rng: () => number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Synthesizes a physically modeled 16kHz cattle acoustic recording (1.20s)
 * with realistic glottal harmonics, tracheal formants, turbulent crackles,
 * bronchial wheeze chirps, and barn ambient noise floor.
 */
export function synthesizeCattleAudio(
  classId: DiseaseClassId,
  seed: number,
  sampleRate = 16000,
  durationSec = 1.2,
  extraNoiseLevel = 0.015
): Float32Array {
  const rng = createSeededRng(seed);
  const length = Math.floor(sampleRate * durationSec);
  const out = new Float32Array(length);

  // Pink-ish barn ambient noise floor (ventilation + distant movement)
  let b0 = 0;
  let b1 = 0;
  let b2 = 0;
  const ambientGain = 0.008 + extraNoiseLevel * (0.5 + rng() * 0.8);
  for (let i = 0; i < length; i++) {
    const white = gaussianRandom(rng);
    b0 = 0.99765 * b0 + white * 0.099046;
    b1 = 0.963 * b1 + white * 0.2965164;
    b2 = 0.57 * b2 + white * 1.0526913;
    out[i] = (b0 + b1 + b2 + white * 0.1848) * ambientGain;
  }

  // Resonant biquad filter helper for tracheal/bronchial formants
  const applyBandpassBurst = (
    excitation: Float32Array,
    centerFreq: number,
    qFactor: number,
    gain: number,
    startSample: number
  ) => {
    const omega = (2 * Math.PI * centerFreq) / sampleRate;
    const sinW = Math.sin(omega);
    const cosW = Math.cos(omega);
    const alpha = sinW / (2 * qFactor);

    const a0 = 1 + alpha;
    const b0c = (sinW / 2) / a0;
    const b2c = -(sinW / 2) / a0;
    const a1c = (-2 * cosW) / a0;
    const a2c = (1 - alpha) / a0;

    let x1 = 0;
    let x2 = 0;
    let y1 = 0;
    let y2 = 0;

    for (let i = 0; i < excitation.length; i++) {
      const idx = startSample + i;
      if (idx < 0 || idx >= length) continue;
      const x0 = excitation[i];
      const y0 = b0c * x0 + b2c * x2 - a1c * y1 - a2c * y2;
      x2 = x1;
      x1 = x0;
      y2 = y1;
      y1 = y0;
      out[idx] += y0 * gain;
    }
  };

  if (classId === 'healthy') {
    // Brief physiological throat clear: clean low fundamental (115-145 Hz), short 140-200ms
    const onset = Math.floor((0.28 + rng() * 0.25) * sampleRate);
    const burstLen = Math.floor((0.15 + rng() * 0.06) * sampleRate);
    const f0 = 115 + rng() * 32;
    const exc = new Float32Array(burstLen);
    for (let i = 0; i < burstLen; i++) {
      const t = i / sampleRate;
      const env = Math.sin(Math.PI * (i / burstLen)) ** 1.4 * Math.exp(-t * 9.5);
      const glottal =
        Math.sin(2 * Math.PI * f0 * t) +
        0.55 * Math.sin(2 * Math.PI * f0 * 2 * t) +
        0.25 * Math.sin(2 * Math.PI * f0 * 3 * t) +
        0.12 * gaussianRandom(rng);
      exc[i] = glottal * env;
    }
    applyBandpassBurst(exc, 240 + rng() * 60, 2.2, 0.72, onset);
    applyBandpassBurst(exc, 430 + rng() * 70, 3.1, 0.28, onset);
  } else if (classId === 'brd') {
    // Dry, forceful double or triple hacking cough bursts in 380 - 1,100 Hz band
    const numBursts = rng() > 0.45 ? 2 : 3;
    let cursor = Math.floor((0.12 + rng() * 0.14) * sampleRate);
    const fBase = 410 + rng() * 140;

    for (let b = 0; b < numBursts; b++) {
      const burstLen = Math.floor((0.14 + rng() * 0.05) * sampleRate);
      const exc = new Float32Array(burstLen);
      for (let i = 0; i < burstLen; i++) {
        const t = i / sampleRate;
        // Sharp percussive attack, dry fast exponential decay
        const attack = Math.min(1, i / (0.008 * sampleRate));
        const env = attack * Math.exp(-t * 15.5);
        const pulse =
          Math.sin(2 * Math.PI * (150 + rng() * 20) * t) * 0.45 +
          gaussianRandom(rng) * 0.85;
        exc[i] = pulse * env;
      }
      applyBandpassBurst(exc, fBase + b * 45, 2.8, 0.95, cursor);
      applyBandpassBurst(exc, fBase * 1.85, 3.4, 0.65, cursor);
      cursor += burstLen + Math.floor((0.08 + rng() * 0.07) * sampleRate);
    }
  } else if (classId === 'pneumonia') {
    // Wet, productive, congested cough: broad high-freq turbulent crackles (1.1 - 3.8 kHz) & long decay
    const onset = Math.floor((0.14 + rng() * 0.16) * sampleRate);
    const burstLen = Math.floor((0.46 + rng() * 0.16) * sampleRate);
    const exc = new Float32Array(burstLen);

    // Pre-generate discrete alveolar crackle impulses
    const crackleEnv = new Float32Array(burstLen);
    const numCrackles = 18 + Math.floor(rng() * 14);
    for (let c = 0; c < numCrackles; c++) {
      const pos = Math.floor(rng() * burstLen * 0.85);
      const decaySamples = Math.floor(0.012 * sampleRate);
      for (let k = 0; k < decaySamples && pos + k < burstLen; k++) {
        crackleEnv[pos + k] += Math.exp(-k / (0.003 * sampleRate)) * (0.7 + rng() * 0.6);
      }
    }

    for (let i = 0; i < burstLen; i++) {
      const t = i / sampleRate;
      const attack = Math.min(1, i / (0.018 * sampleRate));
      // Wet bubbling modulation at ~28 Hz
      const mucusMod = 0.65 + 0.35 * Math.sin(2 * Math.PI * (24 + rng() * 2) * t);
      const env = attack * Math.exp(-t * 4.2) * mucusMod;
      const broadband = gaussianRandom(rng) * (0.9 + 0.9 * crackleEnv[i]);
      exc[i] = broadband * env;
    }

    applyBandpassBurst(exc, 1350 + rng() * 250, 1.4, 0.85, onset);
    applyBandpassBurst(exc, 2450 + rng() * 350, 1.6, 0.82, onset);
    applyBandpassBurst(exc, 3450 + rng() * 300, 1.8, 0.55, onset);
    applyBandpassBurst(exc, 520 + rng() * 120, 2.0, 0.45, onset);
  } else if (classId === 'lungworm') {
    // Spasmodic husky cough + prominent high-pitched bronchial wheeze chirp (1.6 - 2.9 kHz)
    const onset = Math.floor((0.15 + rng() * 0.16) * sampleRate);
    const coughLen = Math.floor((0.14 + rng() * 0.04) * sampleRate);
    const wheezeLen = Math.floor((0.28 + rng() * 0.09) * sampleRate);

    // Phase 1: husky spasmodic bark
    const coughExc = new Float32Array(coughLen);
    for (let i = 0; i < coughLen; i++) {
      const t = i / sampleRate;
      const env = Math.min(1, i / (0.01 * sampleRate)) * Math.exp(-t * 11);
      coughExc[i] = (gaussianRandom(rng) * 0.75 + Math.sin(2 * Math.PI * 220 * t) * 0.4) * env;
    }
    applyBandpassBurst(coughExc, 780 + rng() * 120, 2.5, 0.62, onset);

    // Phase 2: high-pitched descending bronchial wheeze / inspiratory stridor
    const wheezeStart = onset + Math.floor(coughLen * 0.55);
    const fStart = 2050 + rng() * 480;
    const fEnd = fStart - (320 + rng() * 240);
    let phase = 0;
    for (let i = 0; i < wheezeLen; i++) {
      const idx = wheezeStart + i;
      if (idx >= length) break;
      const progress = i / wheezeLen;
      const instFreq = fStart + (fEnd - fStart) * progress;
      phase += (2 * Math.PI * instFreq) / sampleRate;
      const env = Math.sin(Math.PI * progress) ** 0.85 * Math.exp(-progress * 0.9);
      const harmonic =
        Math.sin(phase) * 0.58 +
        Math.sin(phase * 1.5) * 0.16 +
        gaussianRandom(rng) * 0.09;
      out[idx] += harmonic * env * 0.48;
    }
  }

  // Peak normalize safely to [-0.88, 0.88] so audio playback is clear and consistent
  let maxAbs = 1e-6;
  for (let i = 0; i < length; i++) {
    const abs = Math.abs(out[i]);
    if (abs > maxAbs) maxAbs = abs;
  }
  const scale = 0.86 / maxAbs;
  for (let i = 0; i < length; i++) {
    out[i] *= scale;
  }

  return out;
}

/**
 * Encodes mono Float32Array PCM samples into a standard 16-bit PCM WAV Blob.
 */
export function encodeWavBlob(samples: Float32Array, sampleRate = 16000): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}
