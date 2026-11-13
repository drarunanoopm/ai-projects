import * as tf from '@tensorflow/tfjs';
import { DISEASE_CLASSES, synthesizeCattleAudio } from '../audio/cattleSynthesizer';
import { computeMelSpectrogram, DEFAULT_DSP_CONFIG } from '../audio/dspSpectrogram';
import type {
  ClassMetrics,
  DiseaseClassId,
  LayerFeatureMap,
  ModelEvaluation,
  PredictionResult,
  SpectrogramData,
  TrainingConfig,
  TrainingMetricsPoint,
} from '../types/bioacoustics';

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  samplesPerClass: 36,
  epochs: 16,
  learningRate: 0.0018,
  noiseAugmentation: 0.02,
};

export interface DatasetSamplePreview {
  id: string;
  classId: DiseaseClassId;
  seed: number;
  spectrogram: SpectrogramData;
  audio: Float32Array;
}

interface BuiltModelBundle {
  model: tf.LayersModel;
  activationsModel: tf.LayersModel;
  conv1: tf.layers.Layer;
  conv2: tf.layers.Layer;
  conv3: tf.layers.Layer;
  pool3: tf.layers.Layer;
  flatten: tf.layers.Layer;
  denseLatent: tf.layers.Layer;
  outputProbs: tf.layers.Layer;
  totalParams: number;
}

let activeBundle: BuiltModelBundle | null = null;

function buildCnnBundle(learningRate: number): BuiltModelBundle {
  const input = tf.input({ shape: [64, 64, 1], name: 'mel_spectrogram_input' });

  const conv1 = tf.layers.conv2d({
    filters: 16,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu',
    kernelInitializer: 'heNormal',
    name: 'conv2d_1',
  });
  const c1 = conv1.apply(input) as tf.SymbolicTensor;
  const p1 = tf.layers
    .maxPooling2d({ poolSize: [2, 2], name: 'maxpool_1' })
    .apply(c1) as tf.SymbolicTensor;

  const conv2 = tf.layers.conv2d({
    filters: 24,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu',
    kernelInitializer: 'heNormal',
    name: 'conv2d_2',
  });
  const c2 = conv2.apply(p1) as tf.SymbolicTensor;
  const p2 = tf.layers
    .maxPooling2d({ poolSize: [2, 2], name: 'maxpool_2' })
    .apply(c2) as tf.SymbolicTensor;

  const conv3 = tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu',
    kernelInitializer: 'heNormal',
    name: 'conv2d_3',
  });
  const c3 = conv3.apply(p2) as tf.SymbolicTensor;

  const pool3 = tf.layers.maxPooling2d({ poolSize: [2, 2], name: 'maxpool_3' });
  const p3 = pool3.apply(c3) as tf.SymbolicTensor;

  const flatten = tf.layers.flatten({ name: 'flatten' });
  const flat = flatten.apply(p3) as tf.SymbolicTensor;

  const dropout = tf.layers.dropout({ rate: 0.2, name: 'dropout' });
  const drop = dropout.apply(flat) as tf.SymbolicTensor;

  const denseLatent = tf.layers.dense({
    units: 48,
    activation: 'relu',
    kernelInitializer: 'heNormal',
    name: 'dense_latent',
  });
  const d1 = denseLatent.apply(drop) as tf.SymbolicTensor;

  const outputProbs = tf.layers.dense({
    units: DISEASE_CLASSES.length,
    activation: 'softmax',
    name: 'output_probs',
  });
  const out = outputProbs.apply(d1) as tf.SymbolicTensor;

  const model = tf.model({ inputs: input, outputs: out, name: 'BovAcousticCNN' });
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const activationsModel = tf.model({
    inputs: input,
    outputs: [c1, c2, c3, out],
    name: 'BovAcousticActivations',
  });

  return {
    model,
    activationsModel,
    conv1,
    conv2,
    conv3,
    pool3,
    flatten,
    denseLatent,
    outputProbs,
    totalParams: model.countParams(),
  };
}

/**
 * Generates balanced Mel-spectrogram datasets from synthesized bioacoustic recordings.
 */
export function buildSpectrogramDataset(
  samplesPerClass: number,
  noiseAugmentation: number,
  seedOffset = 10000
): {
  xs: Float32Array;
  ys: Float32Array;
  labels: number[];
  totalSamples: number;
  previews: DatasetSamplePreview[];
} {
  const totalSamples = samplesPerClass * DISEASE_CLASSES.length;
  const xs = new Float32Array(totalSamples * 64 * 64);
  const ys = new Float32Array(totalSamples * DISEASE_CLASSES.length);
  const labels: number[] = [];
  const previews: DatasetSamplePreview[] = [];

  // Interleave classes so train/val splits remain balanced
  let cursor = 0;
  for (let i = 0; i < samplesPerClass; i++) {
    for (let c = 0; c < DISEASE_CLASSES.length; c++) {
      const classId = DISEASE_CLASSES[c];
      const seed = seedOffset + c * 997 + i * 37;
      const audio = synthesizeCattleAudio(classId, seed, 16000, 1.2, noiseAugmentation);
      const spec = computeMelSpectrogram(audio, 16000, DEFAULT_DSP_CONFIG);

      // Copy spectrogram into [64, 64, 1] with melBins along vertical axis
      xs.set(spec.normalized, cursor * 64 * 64);
      ys[cursor * DISEASE_CLASSES.length + c] = 1;
      labels.push(c);

      if (i < 3) {
        previews.push({
          id: `${classId}-ds-${i + 1}`,
          classId,
          seed,
          spectrogram: spec,
          audio,
        });
      }
      cursor++;
    }
  }

  return { xs, ys, labels, totalSamples, previews };
}

export async function trainCattleCnn(
  config: TrainingConfig,
  onEpoch?: (point: TrainingMetricsPoint) => void
): Promise<{
  evaluation: ModelEvaluation;
  history: TrainingMetricsPoint[];
  previews: DatasetSamplePreview[];
}> {
  if (activeBundle) {
    activeBundle.activationsModel.dispose();
    activeBundle.model.dispose();
    activeBundle = null;
  }

  // Ensure TF backend is initialized
  await tf.ready();

  const bundle = buildCnnBundle(config.learningRate);
  activeBundle = bundle;

  const dataset = buildSpectrogramDataset(config.samplesPerClass, config.noiseAugmentation, 10400);
  const splitIdx = Math.floor(dataset.totalSamples * 0.78);
  // Align split to multiple of 4 for balanced validation classes
  const trainCount = Math.max(8, splitIdx - (splitIdx % 4));
  const valCount = dataset.totalSamples - trainCount;

  const xAll = tf.tensor4d(dataset.xs, [dataset.totalSamples, 64, 64, 1]);
  const yAll = tf.tensor2d(dataset.ys, [dataset.totalSamples, DISEASE_CLASSES.length]);

  const xTrain = xAll.slice([0, 0, 0, 0], [trainCount, 64, 64, 1]);
  const yTrain = yAll.slice([0, 0], [trainCount, DISEASE_CLASSES.length]);
  const xVal = xAll.slice([trainCount, 0, 0, 0], [valCount, 64, 64, 1]);
  const yVal = yAll.slice([trainCount, 0], [valCount, DISEASE_CLASSES.length]);

  const history: TrainingMetricsPoint[] = [];

  await bundle.model.fit(xTrain, yTrain, {
    epochs: config.epochs,
    batchSize: 16,
    shuffle: true,
    validationData: [xVal, yVal],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const pt: TrainingMetricsPoint = {
          epoch: epoch + 1,
          loss: Number((logs?.loss ?? 0).toFixed(4)),
          accuracy: Number((logs?.acc ?? logs?.accuracy ?? 0).toFixed(4)),
          valLoss: Number((logs?.val_loss ?? 0).toFixed(4)),
          valAccuracy: Number((logs?.val_acc ?? logs?.val_accuracy ?? 0).toFixed(4)),
        };
        history.push(pt);
        onEpoch?.(pt);
        await tf.nextFrame();
      },
    },
  });

  // Compute held-out validation confusion matrix & per-class metrics
  const predsTensor = bundle.model.predict(xVal) as tf.Tensor;
  const predData = await predsTensor.data();
  predsTensor.dispose();

  const cm = Array.from({ length: 4 }, () => [0, 0, 0, 0]);
  let correct = 0;

  for (let i = 0; i < valCount; i++) {
    const trueCls = dataset.labels[trainCount + i];
    let bestCls = 0;
    let bestProb = -1;
    for (let c = 0; c < 4; c++) {
      const p = predData[i * 4 + c];
      if (p > bestProb) {
        bestProb = p;
        bestCls = c;
      }
    }
    cm[trueCls][bestCls]++;
    if (trueCls === bestCls) correct++;
  }

  const perClass: ClassMetrics[] = DISEASE_CLASSES.map((classId, cIdx) => {
    const tp = cm[cIdx][cIdx];
    let fp = 0;
    let fn = 0;
    let support = 0;
    for (let i = 0; i < 4; i++) {
      support += cm[cIdx][i];
      if (i !== cIdx) {
        fp += cm[i][cIdx];
        fn += cm[cIdx][i];
      }
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { classId, precision, recall, f1, support };
  });

  const macroF1 = perClass.reduce((acc, m) => acc + m.f1, 0) / perClass.length;
  const accuracy = valCount > 0 ? correct / valCount : 0;

  xAll.dispose();
  yAll.dispose();
  xTrain.dispose();
  yTrain.dispose();
  xVal.dispose();
  yVal.dispose();

  return {
    evaluation: {
      accuracy,
      macroF1,
      confusionMatrix: cm,
      perClass,
      validationCount: valCount,
      trainedAt: new Date().toLocaleTimeString(),
      totalParams: bundle.totalParams,
    },
    history,
    previews: dataset.previews,
  };
}

function extractTopFeatureMaps(
  tensorData: Float32Array,
  height: number,
  width: number,
  channels: number,
  channelsToShow: number
): Float32Array[] {
  // Rank channels by L2 energy so the inspector displays the most active convolutional filters
  const scores: Array<{ ch: number; energy: number }> = [];
  for (let c = 0; c < channels; c++) {
    let e = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const v = tensorData[(y * width + x) * channels + c];
        e += v * v;
      }
    }
    scores.push({ ch: c, energy: e });
  }
  scores.sort((a, b) => b.energy - a.energy);

  const selected = scores.slice(0, channelsToShow);
  return selected.map(({ ch }) => {
    const map = new Float32Array(height * width);
    let maxVal = 1e-6;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const v = Math.max(0, tensorData[(y * width + x) * channels + ch]);
        map[y * width + x] = v;
        if (v > maxVal) maxVal = v;
      }
    }
    for (let i = 0; i < map.length; i++) {
      map[i] /= maxVal;
    }
    return map;
  });
}

function bilinearUpsample(
  src: Float32Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Float32Array {
  const out = new Float32Array(dstW * dstH);
  for (let y = 0; y < dstH; y++) {
    const sy = (y / Math.max(1, dstH - 1)) * (srcH - 1);
    const y0 = Math.floor(sy);
    const y1 = Math.min(srcH - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < dstW; x++) {
      const sx = (x / Math.max(1, dstW - 1)) * (srcW - 1);
      const x0 = Math.floor(sx);
      const x1 = Math.min(srcW - 1, x0 + 1);
      const fx = sx - x0;

      const v00 = src[y0 * srcW + x0];
      const v10 = src[y0 * srcW + x1];
      const v01 = src[y1 * srcW + x0];
      const v11 = src[y1 * srcW + x1];
      const top = v00 * (1 - fx) + v10 * fx;
      const bot = v01 * (1 - fx) + v11 * fx;
      out[y * dstW + x] = top * (1 - fy) + bot * fy;
    }
  }
  return out;
}

export function isCnnReady(): boolean {
  return activeBundle !== null;
}

/**
 * Runs 2D CNN inference on a 64x64 Mel-spectrogram, extracts intermediate Conv2D feature maps,
 * and computes Grad-CAM spatial attention over time-frequency bins.
 */
export async function classifySpectrogram(spec: SpectrogramData): Promise<PredictionResult> {
  if (!activeBundle) {
    throw new Error('CNN model weights have not been trained yet.');
  }

  const startMs = performance.now();
  const bundle = activeBundle;

  const inputTensor = tf.tensor4d(spec.normalized, [1, 64, 64, 1]);
  const [c1Tensor, c2Tensor, c3Tensor, probTensor] = bundle.activationsModel.predict(
    inputTensor
  ) as tf.Tensor[];

  const probData = await probTensor.data();
  const c1Data = (await c1Tensor.data()) as Float32Array;
  const c2Data = (await c2Tensor.data()) as Float32Array;
  const c3Data = (await c3Tensor.data()) as Float32Array;

  let bestIdx = 0;
  let bestProb = -1;
  const probabilities: Record<DiseaseClassId, number> = {
    healthy: 0,
    brd: 0,
    pneumonia: 0,
    lungworm: 0,
  };
  DISEASE_CLASSES.forEach((cls, idx) => {
    const p = probData[idx];
    probabilities[cls] = p;
    if (p > bestProb) {
      bestProb = p;
      bestIdx = idx;
    }
  });

  // Compute Grad-CAM with respect to conv2d_3 activation tensor
  const gradCam16 = tf.tidy(() => {
    const gradFn = tf.grad((conv3Out: tf.Tensor) => {
      const p3 = bundle.pool3.apply(conv3Out) as tf.Tensor;
      const fl = bundle.flatten.apply(p3) as tf.Tensor;
      const d1 = bundle.denseLatent.apply(fl) as tf.Tensor;
      const logits = bundle.outputProbs.apply(d1) as tf.Tensor;
      return logits.slice([0, bestIdx], [1, 1]).squeeze();
    });

    const grads = gradFn(c3Tensor); // [1, 16, 16, 32]
    const weights = grads.mean([0, 1, 2]); // [32]
    const conv3Squeezed = c3Tensor.squeeze([0]); // [16, 16, 32]
    const cam = conv3Squeezed.mul(weights).sum(-1); // [16, 16]
    const reluCam = tf.relu(cam);
    const maxVal = reluCam.max();
    return reluCam.div(maxVal.add(1e-6));
  });

  const gradCamRaw = (await gradCam16.data()) as Float32Array;
  const gradCamMap = bilinearUpsample(gradCamRaw, 16, 16, 64, 64);

  const featureMaps: LayerFeatureMap[] = [
    {
      layerName: 'conv2d_1',
      label: 'Conv2D Block 1 (Low-Level Edges & Formants)',
      shapeLabel: '64 × 64 × 16',
      paramsCount: bundle.conv1.countParams(),
      channelsShown: 8,
      mapWidth: 64,
      mapHeight: 64,
      maps: extractTopFeatureMaps(c1Data, 64, 64, 16, 8),
    },
    {
      layerName: 'conv2d_2',
      label: 'Conv2D Block 2 (Harmonic & Burst Motifs)',
      shapeLabel: '32 × 32 × 24',
      paramsCount: bundle.conv2.countParams(),
      channelsShown: 8,
      mapWidth: 32,
      mapHeight: 32,
      maps: extractTopFeatureMaps(c2Data, 32, 32, 24, 8),
    },
    {
      layerName: 'conv2d_3',
      label: 'Conv2D Block 3 (Pathology Saliency Features)',
      shapeLabel: '16 × 16 × 32',
      paramsCount: bundle.conv3.countParams(),
      channelsShown: 8,
      mapWidth: 16,
      mapHeight: 16,
      maps: extractTopFeatureMaps(c3Data, 16, 16, 32, 8),
    },
  ];

  gradCam16.dispose();
  inputTensor.dispose();
  c1Tensor.dispose();
  c2Tensor.dispose();
  c3Tensor.dispose();
  probTensor.dispose();

  const inferenceTimeMs = Math.max(1, Math.round(performance.now() - startMs));

  return {
    predictedClass: DISEASE_CLASSES[bestIdx],
    confidence: bestProb,
    probabilities,
    gradCamMap,
    gradCamWidth: 64,
    gradCamHeight: 64,
    featureMaps,
    inferenceTimeMs,
  };
}
