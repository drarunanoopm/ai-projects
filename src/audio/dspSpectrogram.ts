import type { AcousticBiomarkers, ColorMapId, DspConfig, SpectrogramData } from '../types/bioacoustics';

export const DEFAULT_DSP_CONFIG: DspConfig = {
  sampleRate: 16000,
  fftSize: 512,
  melBins: 64,
  timeFrames: 64,
  fMin: 60,
  fMax: 8000,
  minDb: -75,
  maxDb: 0,
};

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

/**
 * In-place iterative Cooley-Tukey Radix-2 FFT.
 * `real` and `imag` must have length N where N is a power of 2.
 */
function fftRadix2(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (i < j) {
      const tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      const ti = imag[i];
      imag[i] = imag[j];
      imag[j] = ti;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = (-2 * Math.PI) / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const angle = k * step;
        const wr = Math.cos(angle);
        const wi = Math.sin(angle);
        const evenIdx = start + k;
        const oddIdx = evenIdx + half;

        const tr = wr * real[oddIdx] - wi * imag[oddIdx];
        const ti = wr * imag[oddIdx] + wi * real[oddIdx];

        real[oddIdx] = real[evenIdx] - tr;
        imag[oddIdx] = imag[evenIdx] - ti;
        real[evenIdx] += tr;
        imag[evenIdx] += ti;
      }
    }
  }
}

/**
 * Resamples mono audio to target sample rate and extracts a 1.2s active window
 * centered around the peak cough energy if the uploaded audio is longer.
 */
export function prepareCoughWindow(
  input: Float32Array,
  inputSampleRate: number,
  targetSampleRate = 16000,
  targetDurationSec = 1.2
): Float32Array {
  let resampled: Float32Array;
  if (inputSampleRate === targetSampleRate) {
    resampled = input;
  } else {
    const ratio = inputSampleRate / targetSampleRate;
    const newLen = Math.max(1, Math.floor(input.length / ratio));
    resampled = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const srcIdx = i * ratio;
      const idx0 = Math.floor(srcIdx);
      const idx1 = Math.min(input.length - 1, idx0 + 1);
      const frac = srcIdx - idx0;
      resampled[i] = input[idx0] * (1 - frac) + input[idx1] * frac;
    }
  }

  const targetLen = Math.floor(targetSampleRate * targetDurationSec);
  const out = new Float32Array(targetLen);

  if (resampled.length <= targetLen) {
    // Center shorter clips or copy from start
    out.set(resampled, 0);
  } else {
    // Slide 0.3s energy window to locate primary cough event and extract 1.2s around it
    const win = Math.floor(targetSampleRate * 0.25);
    let bestStart = 0;
    let maxEnergy = -1;
    let running = 0;
    for (let i = 0; i < resampled.length; i++) {
      running += resampled[i] * resampled[i];
      if (i >= win) {
        running -= resampled[i - win] * resampled[i - win];
      }
      if (i >= win - 1 && running > maxEnergy) {
        maxEnergy = running;
        bestStart = Math.max(0, i - win - Math.floor(targetLen * 0.25));
      }
    }
    const clampedStart = Math.min(bestStart, resampled.length - targetLen);
    out.set(resampled.subarray(clampedStart, clampedStart + targetLen), 0);
  }

  // Peak normalize safely
  let peak = 1e-6;
  for (let i = 0; i < out.length; i++) {
    const a = Math.abs(out[i]);
    if (a > peak) peak = a;
  }
  if (peak > 0.01) {
    const gain = 0.88 / peak;
    for (let i = 0; i < out.length; i++) {
      out[i] *= gain;
    }
  }

  return out;
}

function buildMelFilterbank(
  melBins: number,
  fftSize: number,
  sampleRate: number,
  fMin: number,
  fMax: number
): { filters: Float32Array[]; melFreqs: Float32Array } {
  const numFreqBins = fftSize / 2 + 1;
  const clampedMax = Math.min(fMax, sampleRate / 2);
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(clampedMax);

  const melPoints = new Float32Array(melBins + 2);
  const hzPoints = new Float32Array(melBins + 2);
  const binPoints = new Int32Array(melBins + 2);

  for (let i = 0; i < melBins + 2; i++) {
    melPoints[i] = melMin + (i / (melBins + 1)) * (melMax - melMin);
    hzPoints[i] = melToHz(melPoints[i]);
    binPoints[i] = Math.min(
      numFreqBins - 1,
      Math.max(0, Math.floor(((fftSize + 1) * hzPoints[i]) / sampleRate))
    );
  }

  const filters: Float32Array[] = [];
  const melFreqs = new Float32Array(melBins);

  for (let m = 1; m <= melBins; m++) {
    const f = new Float32Array(numFreqBins);
    const left = binPoints[m - 1];
    const center = binPoints[m];
    const right = Math.max(center + 1, binPoints[m + 1]);

    for (let k = left; k < center; k++) {
      f[k] = (k - left) / Math.max(1, center - left);
    }
    for (let k = center; k <= right && k < numFreqBins; k++) {
      f[k] = (right - k) / Math.max(1, right - center);
    }
    // Slaney area normalization
    const bandwidth = hzPoints[m + 1] - hzPoints[m - 1];
    const norm = bandwidth > 0 ? 2 / bandwidth : 1;
    for (let k = 0; k < numFreqBins; k++) {
      f[k] *= norm * 140;
    }

    filters.push(f);
    melFreqs[m - 1] = hzPoints[m];
  }

  return { filters, melFreqs };
}

/**
 * Computes a 64x64 Log-Mel Spectrogram and quantitative bioacoustic biomarkers.
 */
export function computeMelSpectrogram(
  rawSamples: Float32Array,
  sampleRate: number,
  config: DspConfig = DEFAULT_DSP_CONFIG
): SpectrogramData {
  const samples = prepareCoughWindow(rawSamples, sampleRate, config.sampleRate, 1.2);
  const { fftSize, melBins, timeFrames, fMin, fMax, minDb, maxDb } = config;
  const numFreqBins = fftSize / 2 + 1;

  // Precompute Hann window
  const hann = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
  }

  const { filters, melFreqs } = buildMelFilterbank(melBins, fftSize, config.sampleRate, fMin, fMax);

  const dbMatrix = new Float32Array(melBins * timeFrames);
  const normalized = new Float32Array(melBins * timeFrames);
  const real = new Float32Array(fftSize);
  const imag = new Float32Array(fftSize);
  const powerSpectrum = new Float32Array(numFreqBins);

  const maxStart = Math.max(1, samples.length - fftSize);
  const frameEnergies = new Float32Array(timeFrames);
  const melTotalEnergy = new Float32Array(melBins);

  let peakFrameDb = -120;

  for (let t = 0; t < timeFrames; t++) {
    const startIdx = Math.floor((t / Math.max(1, timeFrames - 1)) * maxStart);
    let frameSumSq = 0;
    for (let i = 0; i < fftSize; i++) {
      const s = samples[startIdx + i] || 0;
      real[i] = s * hann[i];
      imag[i] = 0;
      frameSumSq += s * s;
    }
    frameEnergies[t] = Math.sqrt(frameSumSq / fftSize);

    fftRadix2(real, imag);

    for (let k = 0; k < numFreqBins; k++) {
      powerSpectrum[k] = (real[k] * real[k] + imag[k] * imag[k]) / fftSize;
    }

    for (let m = 0; m < melBins; m++) {
      const filter = filters[m];
      let melPower = 0;
      for (let k = 0; k < numFreqBins; k++) {
        if (filter[k] > 0) {
          melPower += powerSpectrum[k] * filter[k];
        }
      }
      const db = 10 * Math.log10(melPower + 1e-9);
      if (db > peakFrameDb) peakFrameDb = db;
      dbMatrix[m * timeFrames + t] = db;
      melTotalEnergy[m] += melPower;
    }
  }

  // Relative dB normalization anchored to peak energy per recording
  const refPeak = Math.max(-20, peakFrameDb);
  const dbSpan = Math.max(10, maxDb - minDb);

  for (let i = 0; i < dbMatrix.length; i++) {
    const relDb = dbMatrix[i] - refPeak;
    const clamped = Math.max(minDb, Math.min(maxDb, relDb));
    dbMatrix[i] = clamped;
    normalized[i] = (clamped - minDb) / dbSpan;
  }

  const biomarkers = extractBiomarkers(
    samples,
    config.sampleRate,
    melTotalEnergy,
    melFreqs,
    frameEnergies
  );

  return {
    normalized,
    dbMatrix,
    width: timeFrames,
    height: melBins,
    durationSec: samples.length / config.sampleRate,
    sampleRate: config.sampleRate,
    melFrequencies: melFreqs,
    minDb,
    maxDb,
    biomarkers,
    envelope: frameEnergies,
  };
}

function extractBiomarkers(
  samples: Float32Array,
  sampleRate: number,
  melTotalEnergy: Float32Array,
  melFreqs: Float32Array,
  frameEnergies: Float32Array
): AcousticBiomarkers {
  // Peak Frequency & Spectral Centroid
  let maxEnergy = -1;
  let peakBin = 0;
  let weightedFreqSum = 0;
  let totalMelEnergy = 1e-12;

  for (let m = 0; m < melFreqs.length; m++) {
    const e = melTotalEnergy[m];
    if (e > maxEnergy) {
      maxEnergy = e;
      peakBin = m;
    }
    weightedFreqSum += melFreqs[m] * e;
    totalMelEnergy += e;
  }

  const peakFreqHz = Math.round(melFreqs[peakBin] || 0);
  const spectralCentroidHz = Math.round(weightedFreqSum / totalMelEnergy);

  // Active Cough Duration & Burst Count
  let maxEnv = 1e-6;
  for (let i = 0; i < frameEnergies.length; i++) {
    if (frameEnergies[i] > maxEnv) maxEnv = frameEnergies[i];
  }
  const activeThreshold = maxEnv * 0.22;
  const burstThreshold = maxEnv * 0.32;
  let activeFrames = 0;
  let burstCount = 0;
  let inBurst = false;

  for (let i = 0; i < frameEnergies.length; i++) {
    if (frameEnergies[i] >= activeThreshold) activeFrames++;
    if (!inBurst && frameEnergies[i] >= burstThreshold) {
      burstCount++;
      inBurst = true;
    } else if (inBurst && frameEnergies[i] < burstThreshold * 0.55) {
      inBurst = false;
    }
  }

  const frameStepMs = ((samples.length / sampleRate) * 1000) / frameEnergies.length;
  const coughDurationMs = Math.round(activeFrames * frameStepMs);

  // Zero-Crossing Rate & RMS
  let zeroCrossings = 0;
  let sumSq = 0;
  for (let i = 1; i < samples.length; i++) {
    if ((samples[i] >= 0 && samples[i - 1] < 0) || (samples[i] < 0 && samples[i - 1] >= 0)) {
      zeroCrossings++;
    }
    sumSq += samples[i] * samples[i];
  }
  const zeroCrossingRate = Number((zeroCrossings / samples.length).toFixed(4));
  const rmsEnergy = Number(Math.sqrt(sumSq / samples.length).toFixed(4));

  // Harmonic-to-Noise Ratio proxy from short-window normalized autocorrelation
  const peakIdx = frameEnergies.indexOf(maxEnv);
  const centerSample = Math.floor((peakIdx / frameEnergies.length) * samples.length);
  const winLen = Math.min(1024, samples.length - centerSample);
  let r0 = 1e-9;
  let maxLagCorr = 0;
  if (winLen > 256) {
    for (let i = 0; i < winLen; i++) {
      r0 += samples[centerSample + i] * samples[centerSample + i];
    }
    const minLag = Math.floor(sampleRate / 600);
    const maxLag = Math.floor(sampleRate / 70);
    for (let lag = minLag; lag <= maxLag && lag < winLen / 2; lag++) {
      let r = 0;
      for (let i = 0; i < winLen - lag; i++) {
        r += samples[centerSample + i] * samples[centerSample + i + lag];
      }
      const normR = r / r0;
      if (normR > maxLagCorr) maxLagCorr = normR;
    }
  }
  const clampedCorr = Math.max(0.04, Math.min(0.96, maxLagCorr));
  const hnrDb = Number((10 * Math.log10(clampedCorr / (1 - clampedCorr))).toFixed(1));

  return {
    peakFreqHz,
    spectralCentroidHz,
    coughDurationMs,
    zeroCrossingRate,
    hnrDb,
    rmsEnergy,
    burstCount: Math.max(1, burstCount),
  };
}

/**
 * Generates a 256-entry RGB lookup table for scientific spectrogram colormaps.
 */
export function buildColorMapLut(mapId: ColorMapId): Uint8Array {
  const lut = new Uint8Array(256 * 3);

  const stops: Record<ColorMapId, Array<[number, [number, number, number]]>> = {
    viridis: [
      [0.0, [68, 1, 84]],
      [0.25, [59, 82, 139]],
      [0.5, [33, 145, 140]],
      [0.75, [94, 201, 98]],
      [1.0, [253, 231, 37]],
    ],
    inferno: [
      [0.0, [8, 6, 20]],
      [0.25, [87, 16, 110]],
      [0.5, [188, 55, 84]],
      [0.75, [249, 142, 9]],
      [1.0, [252, 255, 164]],
    ],
    cyanAmber: [
      [0.0, [10, 15, 28]],
      [0.3, [14, 72, 110]],
      [0.55, [6, 182, 212]],
      [0.8, [245, 158, 11]],
      [1.0, [254, 243, 199]],
    ],
  };

  const palette = stops[mapId];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let left = palette[0];
    let right = palette[palette.length - 1];
    for (let s = 0; s < palette.length - 1; s++) {
      if (t >= palette[s][0] && t <= palette[s + 1][0]) {
        left = palette[s];
        right = palette[s + 1];
        break;
      }
    }
    const span = Math.max(1e-6, right[0] - left[0]);
    const f = (t - left[0]) / span;
    lut[i * 3] = Math.round(left[1][0] + (right[1][0] - left[1][0]) * f);
    lut[i * 3 + 1] = Math.round(left[1][1] + (right[1][1] - left[1][1]) * f);
    lut[i * 3 + 2] = Math.round(left[1][2] + (right[1][2] - left[1][2]) * f);
  }

  return lut;
}

/**
 * Renders a compact 64x64 spectrogram thumbnail to a Data URL for herd logs and dataset galleries.
 */
export function renderSpectrogramThumbnail(
  normalized: Float32Array,
  width: number,
  height: number,
  colorMap: ColorMapId = 'viridis'
): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const img = ctx.createImageData(width, height);
  const lut = buildColorMapLut(colorMap);

  for (let y = 0; y < height; y++) {
    // Invert vertical axis so low frequencies sit at bottom
    const melIdx = height - 1 - y;
    for (let x = 0; x < width; x++) {
      const val = Math.max(0, Math.min(1, normalized[melIdx * width + x]));
      const lutIdx = Math.min(255, Math.floor(val * 255)) * 3;
      const px = (y * width + x) * 4;
      img.data[px] = lut[lutIdx];
      img.data[px + 1] = lut[lutIdx + 1];
      img.data[px + 2] = lut[lutIdx + 2];
      img.data[px + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL('image/png');
}
