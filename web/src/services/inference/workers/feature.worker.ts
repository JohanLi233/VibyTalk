/**
 * @file feature.worker.ts
 * @author JohanLi233
 */
import FFT from "fft.js";
import { create, ConverterType } from "@alexanderolsen/libsamplerate-js";
import {
  SAMPLE_RATE,
  FRAME_LENGTH_MS,
  FRAME_SHIFT_MS,
  NUM_MEL_BINS,
  NUM_SEQUENCE_FRAMES,
  ENERGY_FLOOR,
  PREEMPH_COEFF,
  DEFAULT_LOW_FREQ,
  REMOVE_DC_OFFSET,
  ROUND_TO_POWER_OF_TWO,
} from "../../core";

const FRAME_LENGTH = Math.round((FRAME_LENGTH_MS * SAMPLE_RATE) / 1000);
const FRAME_SHIFT = Math.round((FRAME_SHIFT_MS * SAMPLE_RATE) / 1000);

const roundUpToNearestPowerOfTwo = (n: number): number => {
  if (n <= 0) return 1;
  let power = 1;
  while (power < n) {
    power *= 2;
  }
  return power;
};

const FFT_SIZE = ROUND_TO_POWER_OF_TWO
  ? roundUpToNearestPowerOfTwo(FRAME_LENGTH)
  : FRAME_LENGTH;

export function createKaldiMelFilterBank(
  numFilters: number,
  fftSize: number,
  sampleRate: number,
  lowFreq: number = 20,
  highFreq?: number
): Float32Array[] {
  highFreq = highFreq || sampleRate / 2;

  const melToFreq = (mel: number) => 700.0 * (Math.exp(mel / 1127.0) - 1.0);
  const freqToMel = (freq: number) => 1127.0 * Math.log(1.0 + freq / 700.0);

  const lowMel = freqToMel(lowFreq);
  const highMel = freqToMel(highFreq);

  const melPoints = new Float32Array(numFilters + 2);
  for (let i = 0; i < numFilters + 2; i++) {
    melPoints[i] = lowMel + ((highMel - lowMel) * i) / (numFilters + 1);
  }

  const freqPoints = melPoints.map(melToFreq);
  const binPoints = freqPoints.map((freq) => (freq * fftSize) / sampleRate);

  const filterBank: Float32Array[] = [];
  const numFftBins = Math.floor(fftSize / 2) + 1;

  for (let m = 1; m <= numFilters; m++) {
    const filter = new Float32Array(numFftBins);
    const leftBin = binPoints[m - 1];
    const centerBin = binPoints[m];
    const rightBin = binPoints[m + 1];

    for (let k = 0; k < numFftBins; k++) {
      if (k >= leftBin && k <= rightBin) {
        if (k <= centerBin) {
          filter[k] =
            centerBin > leftBin ? (k - leftBin) / (centerBin - leftBin) : 0.0;
        } else {
          filter[k] =
            rightBin > centerBin
              ? (rightBin - k) / (rightBin - centerBin)
              : 0.0;
        }
      } else {
        filter[k] = 0.0;
      }
    }
    filterBank.push(filter);
  }

  return filterBank;
}

/**
 * Precomputed Kaldi-compatible Mel filter bank for improved runtime performance.
 */
const MEL_FILTER_BANK = createKaldiMelFilterBank(
  NUM_MEL_BINS,
  FFT_SIZE,
  SAMPLE_RATE,
  DEFAULT_LOW_FREQ
);

// =============================================================================
// --- Audio processing utility functions ---
// =============================================================================

/**
 * Apply Hanning window (in-place) to the signal.
 */
function applyHanningWindowInPlace(signal: Float32Array): void {
  const N = signal.length;
  if (N <= 1) return;
  const factor = (2.0 * Math.PI) / (N - 1);
  for (let i = 0; i < N; i++) {
    const window = 0.5 - 0.5 * Math.cos(i * factor);
    signal[i] *= window;
  }
}

/**
 * Apply pre-emphasis filter (in-place) to the signal.
 */
function applyPreemphasisInPlace(signal: Float32Array, coeff: number): void {
  if (coeff === 0.0) return;
  for (let i = signal.length - 1; i > 0; i--) {
    signal[i] -= coeff * signal[i - 1];
  }
  if (signal.length > 0) {
    signal[0] -= coeff * signal[0];
  }
}

/**
 * Remove the DC component (in-place) from the signal.
 */
function removeDCOffsetInPlace(signal: Float32Array): void {
  if (signal.length === 0) return;
  const mean = signal.reduce((sum, val) => sum + val, 0) / signal.length;
  for (let i = 0; i < signal.length; i++) {
    signal[i] -= mean;
  }
}

/**
 * Compute the power spectrum of the signal.
 */
function computePowerSpectrum(
  outPowerSpectrum: Float32Array,
  fftResult: Float32Array | number[]
): void {
  const numBins = outPowerSpectrum.length;
  for (let i = 0; i < numBins; i++) {
    const re = fftResult[i * 2];
    const im = fftResult[i * 2 + 1];
    outPowerSpectrum[i] = re * re + im * im;
  }
}

/**
 * Perform instance-level normalization on the features.
 */
function perInstanceNormalize(features: Float32Array): Float32Array {
  const len = features.length;
  if (len < 2) {
    if (len === 1) return new Float32Array([0.0]);
    return features;
  }

  const mean = features.reduce((sum, val) => sum + val, 0) / len;
  const variance =
    features.reduce((sum, val) => sum + (val - mean) ** 2, 0) / (len - 1);
  const std = Math.sqrt(variance);

  const normalized = new Float32Array(len);
  if (std > 1e-8) {
    for (let i = 0; i < len; i++) {
      normalized[i] = (features[i] - mean) / std;
    }
  } else {
    for (let i = 0; i < len; i++) {
      normalized[i] = features[i] - mean;
    }
  }
  return normalized;
}

// =============================================================================
// --- Worker core logic ---
// =============================================================================

/**
 * @interface WorkerInputData
 * Define the data structure sent from the main thread to the Worker.
 */
interface WorkerInputData {
  leftChannel: Float32Array;
  rightChannel?: Float32Array;
  sampleRate: number;
}

async function extractMelFeatures(
  audioData: WorkerInputData
): Promise<{ features: Float32Array; dimensions: number[] }> {
  let pcmData = audioData.leftChannel;
  const originalSampleRate = audioData.sampleRate;

  if (audioData.rightChannel) {
    const mixedData = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      mixedData[i] = (pcmData[i] + audioData.rightChannel[i]) / 2;
    }
    pcmData = mixedData;
  }

  if (originalSampleRate !== SAMPLE_RATE) {
    const resampler = await create(1, originalSampleRate, SAMPLE_RATE, {
      converterType: ConverterType.SRC_SINC_MEDIUM_QUALITY,
    });
    pcmData = resampler.simple(pcmData) as Float32Array;
    resampler.destroy();
  }

  const numFrames =
    Math.floor((pcmData.length - FRAME_LENGTH) / FRAME_SHIFT) + 1;
  if (numFrames <= 0) {
    throw new Error("Feature extraction failed: audio duration too short, cannot generate at least one frame.");
  }

  const fft = new FFT(FFT_SIZE);
  const combinedFeatures = new Float32Array(numFrames * NUM_MEL_BINS);
  const frameData = new Float32Array(FRAME_LENGTH);
  const fftInput = fft.createComplexArray();
  const fftOutput = fft.createComplexArray();
  const powerSpectrum = new Float32Array(Math.floor(FFT_SIZE / 2) + 1);

  for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
    const startSample = frameIdx * FRAME_SHIFT;
    const frameView = pcmData.subarray(startSample, startSample + FRAME_LENGTH);
    frameData.set(frameView);

    if (REMOVE_DC_OFFSET) removeDCOffsetInPlace(frameData);
    applyPreemphasisInPlace(frameData, PREEMPH_COEFF);
    applyHanningWindowInPlace(frameData);

    fftInput.fill(0);
    for (let i = 0; i < FRAME_LENGTH; i++) {
      fftInput[i * 2] = frameData[i];
    }

    fft.transform(fftOutput, fftInput);
    computePowerSpectrum(powerSpectrum, fftOutput);

    const melFeatureOffset = frameIdx * NUM_MEL_BINS;
    for (let m = 0; m < NUM_MEL_BINS; m++) {
      let energy = 0.0;
      const currentFilter = MEL_FILTER_BANK[m];
      for (let k = 0; k < powerSpectrum.length; k++) {
        energy += powerSpectrum[k] * currentFilter[k];
      }
      energy = Math.max(energy, ENERGY_FLOOR);
      combinedFeatures[melFeatureOffset + m] = Math.log(energy);
    }
  }

  const numMelFrames = numFrames;
  const paddedLength =
    Math.ceil(numMelFrames / NUM_SEQUENCE_FRAMES) * NUM_SEQUENCE_FRAMES;
  const paddedFeatures = new Float32Array(paddedLength * NUM_MEL_BINS);
  paddedFeatures.set(combinedFeatures);

  if (paddedLength > numMelFrames) {
    const lastFrame = combinedFeatures.subarray(
      (numMelFrames - 1) * NUM_MEL_BINS
    );
    for (let i = numMelFrames; i < paddedLength; i++) {
      paddedFeatures.set(lastFrame, i * NUM_MEL_BINS);
    }
  }

  const normalizedFeatures = perInstanceNormalize(paddedFeatures);

  const T = paddedLength;
  const newT = T / NUM_SEQUENCE_FRAMES;

  return {
    features: normalizedFeatures,
    dimensions: [newT, NUM_SEQUENCE_FRAMES, NUM_MEL_BINS],
  };
}

self.addEventListener(
  "message",
  async (event: MessageEvent<WorkerInputData>) => {
    try {
      const audioData = event.data;
      const result = await extractMelFeatures(audioData);

      self.postMessage(
        {
          status: "success",
          payload: result,
        },
        { transfer: [result.features.buffer] }
      );
    } catch (e) {
      self.postMessage({
        status: "error",
        error: (e as Error).message,
      });
    }
  }
);
