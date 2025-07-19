/**
 * @file AudioService.ts
 * @author JohanLi233
 */

import {
  NUM_MEL_BINS,
  NUM_SEQUENCE_FRAMES,
  WINDOW_NUM_CHUNKS,
  WINDOW_NUM_CHUNKS_HALF,
  createErrorHandler,
} from "./core";

export { NUM_MEL_BINS, NUM_SEQUENCE_FRAMES };

/**
 * Extract a fixed-size context window from the complete audio features
 * @param allAudioFeatures flattened feature array returned by `FeatureExtractorService`
 * @param allAudioDims dimensions array returned by `FeatureExtractorService`, e.g. `[T_chunks, 4, 80]`
 * @param centerChunkIndex index of the chunk in the center of the window
 * @returns a new flattened Float32Array representing the (32, 4, 80) window data
 */
export const getAudioWindow = (
  allAudioFeatures: Float32Array,
  allAudioDims: number[],
  centerChunkIndex: number
): Float32Array => {
  if (allAudioDims.length !== 3) {
    throw new Error(
      "Invalid dimensions array provided to getAudioWindow. Expected length 3."
    );
  }
  const [numChunks, chunkSize, melBins] = allAudioDims;

  const finalWindowData = new Float32Array(
    WINDOW_NUM_CHUNKS * chunkSize * melBins
  );

  const chunkTotalSize = chunkSize * melBins;

  for (let j = 0; j < WINDOW_NUM_CHUNKS; j++) {
    const sourceChunkIndex = centerChunkIndex - WINDOW_NUM_CHUNKS_HALF + j;

    const clampedSourceIndex = Math.max(
      0,
      Math.min(sourceChunkIndex, numChunks - 1)
    );

    const sourceChunkStart = clampedSourceIndex * chunkTotalSize;
    const sourceChunkEnd = sourceChunkStart + chunkTotalSize;

    const chunkData = allAudioFeatures.subarray(
      sourceChunkStart,
      sourceChunkEnd
    );

    const destChunkStart = j * chunkTotalSize;
    finalWindowData.set(chunkData, destChunkStart);
  }

  return finalWindowData;
};

/**
 * Feature extraction result interface
 */
export interface FeatureExtractionResult {
  features: Float32Array;
  dimensions: number[];
}

/**
 * Audio feature extraction service class
 * Wraps the interaction with the feature extraction Web Worker, providing a simple Promise-based API
 */
export class FeatureExtractorService {
  private errorHandler = createErrorHandler("FeatureExtractorService");
  private worker: Worker;

  constructor() {
    this.worker = new Worker(
      new URL("./inference/workers/feature.worker.ts", import.meta.url),
      { type: "module" }
    );
  }

  /**
   * Process AudioBuffer, extract features in the background thread
   * @param audioBuffer AudioBuffer object from Web Audio API
   * @returns a Promise, resolves to feature result on success, rejects with error message on failure
   */
  public async process(
    audioBuffer: AudioBuffer
  ): Promise<FeatureExtractionResult> {
    try {
      const leftChannel = audioBuffer.getChannelData(0);
      const rightChannel =
        audioBuffer.numberOfChannels > 1
          ? audioBuffer.getChannelData(1)
          : undefined;

      const message = {
        leftChannel,
        rightChannel,
        sampleRate: audioBuffer.sampleRate,
      };

      const transferList = [leftChannel.buffer, rightChannel?.buffer].filter(
        Boolean
      ) as ArrayBuffer[];

      const result = await this.createWorkerPromise<FeatureExtractionResult>(
        message,
        transferList
      );

      return result;
    } catch (error) {
      throw this.errorHandler(error);
    }
  }

  private createWorkerPromise<T>(
    message: any,
    transferList: ArrayBuffer[]
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const handleMessage = (event: MessageEvent) => {
        this.worker.removeEventListener("message", handleMessage);
        this.worker.removeEventListener("error", handleError);

        if (event.data.status === "success") {
          resolve(event.data.payload);
        } else {
          reject(new Error(event.data.error || "Unknown worker error"));
        }
      };

      const handleError = (error: ErrorEvent) => {
        this.worker.removeEventListener("message", handleMessage);
        this.worker.removeEventListener("error", handleError);
        reject(error);
      };

      this.worker.addEventListener("message", handleMessage);
      this.worker.addEventListener("error", handleError);

      this.worker.postMessage(message, transferList);
    });
  }

  public terminate() {
    this.worker.terminate();
  }
}
