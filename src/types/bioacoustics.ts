export type DiseaseClassId = 'healthy' | 'brd' | 'pneumonia' | 'lungworm';

export type RiskLevel = 'normal' | 'moderate' | 'high' | 'critical';

export type ColorMapId = 'viridis' | 'inferno' | 'cyanAmber';

export type WorkspaceTab = 'lab' | 'architecture' | 'dataset' | 'herd' | 'about';

export interface DiseaseProfile {
  id: DiseaseClassId;
  name: string;
  shortName: string;
  pathogen: string;
  riskLevel: RiskLevel;
  riskGlyph: string;
  riskLabel: string;
  frequencyBandLabel: string;
  typicalDurationLabel: string;
  acousticSignature: string;
  clinicalAction: string;
}

export interface DspConfig {
  sampleRate: number;
  fftSize: 512 | 1024;
  melBins: 64;
  timeFrames: 64;
  fMin: number;
  fMax: number;
  minDb: number;
  maxDb: number;
}

export interface AcousticBiomarkers {
  peakFreqHz: number;
  spectralCentroidHz: number;
  coughDurationMs: number;
  zeroCrossingRate: number;
  hnrDb: number;
  rmsEnergy: number;
  burstCount: number;
}

export interface SpectrogramData {
  /** Normalized [0, 1] values in row-major order: [melBin (0..height-1, low->high), timeFrame (0..width-1)] */
  normalized: Float32Array;
  /** Raw log-power dB values in row-major order [melBin, timeFrame] */
  dbMatrix: Float32Array;
  width: number;
  height: number;
  durationSec: number;
  sampleRate: number;
  melFrequencies: Float32Array;
  minDb: number;
  maxDb: number;
  biomarkers: AcousticBiomarkers;
  envelope: Float32Array;
}

export interface LayerFeatureMap {
  layerName: string;
  label: string;
  shapeLabel: string;
  paramsCount: number;
  channelsShown: number;
  mapWidth: number;
  mapHeight: number;
  /** Flattened array of [channelsShown * mapHeight * mapWidth] normalized [0,1] */
  maps: Float32Array[];
}

export interface PredictionResult {
  predictedClass: DiseaseClassId;
  confidence: number;
  probabilities: Record<DiseaseClassId, number>;
  gradCamMap: Float32Array;
  gradCamWidth: number;
  gradCamHeight: number;
  featureMaps: LayerFeatureMap[];
  inferenceTimeMs: number;
}

export interface TrainingMetricsPoint {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
}

export interface TrainingConfig {
  samplesPerClass: number;
  epochs: number;
  learningRate: number;
  noiseAugmentation: number;
}

export interface ClassMetrics {
  classId: DiseaseClassId;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelEvaluation {
  accuracy: number;
  macroF1: number;
  confusionMatrix: number[][];
  perClass: ClassMetrics[];
  validationCount: number;
  trainedAt: string;
  totalParams: number;
}

export interface AudioSamplePreset {
  id: string;
  label: string;
  cowTag: string;
  barnZone: string;
  trueClass: DiseaseClassId;
  description: string;
  seed: number;
}

export interface HerdScreeningRecord {
  id: string;
  cowTag: string;
  barnZone: string;
  timestamp: string;
  sourceName: string;
  predictedClass: DiseaseClassId;
  confidence: number;
  biomarkers: AcousticBiomarkers;
  spectrogramDataUrl: string;
  notes: string;
}
