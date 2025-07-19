/**
 * @file StreamingInferenceService.ts
 *
 *
 * @author JohanLi233
 */

import type {
  MainThreadMessage,
  StreamingWorkerInitMessage as WorkerInitMessage,
} from "./workers/streaming.inference.worker";
import type { ImageDataResponse } from "../media/DataLoaderService";
import type { ChunkFeatureResult } from "../StreamingFeatureExtractorService";

export interface StreamingChunkData {
  chunkIndex: number;
  audioFeatures: Float32Array;
  audioDimensions: number[];
  startFrame: number;
  endFrame: number;
}

/**
 * Streaming inference message type sent to worker
 */
export type StreamingWorkerRunMessage = {
  type: "streaming_run";
  chunkData: StreamingChunkData;
  dataset: ImageDataResponse;
  zipBlob: Blob;
  blendingMaskBitmap: ImageBitmap;
  startImageIndex: number;
};

/**
 * Streaming inference callback interface definition
 */
export interface StreamingInferenceCallbacks {
  /**
   * Progress callback
   * @param processed processed frames
   * @param total total frames
   */
  onProgress?: (processed: number, total: number) => void;
  /**
   * Single frame inference completion callback
   * @param frame generated video frame
   * @param frameIndex frame index
   */
  onFrame?: (frame: ImageBitmap, frameIndex: number) => void;
  /**
   * Single chunk inference completion callback
   * @param chunkIndex chunk index
   * @param timings chunk timing
   */
  onChunkComplete?: (
    chunkIndex: number,
    timings: Record<string, number>
  ) => void;
  onAllComplete?: (totalTimings: Record<string, number>) => void;
  onError?: (message: string) => void;
}

export class StreamingInferenceService {
  private worker: Worker | null = null;
  private callbacks: StreamingInferenceCallbacks = {};
  private isInitialized = false;
  private pendingChunks = new Map<number, StreamingChunkData>();
  private completedChunks = new Set<number>();
  private totalFrames = 0;
  private isProcessing = false;

  constructor(modelPath: string, onReady?: () => void) {
    this.worker = new Worker(
      new URL("./workers/streaming.inference.worker.ts", import.meta.url),
      {
        type: "module",
      }
    );

    this.worker.onmessage = (event: MessageEvent<MainThreadMessage>) => {
      const { type } = event.data;
      switch (type) {
        case "ready":
          this.isInitialized = true;
          onReady?.();
          break;
        case "progress":
          this.callbacks.onProgress?.(
            event.data.payload.processed,
            event.data.payload.total
          );
          break;
        case "frame":
          const frameData = event.data.payload as {
            frame: ImageBitmap;
            frameIndex: number;
          };
          this.callbacks.onFrame?.(frameData.frame, frameData.frameIndex);
          break;
        case "chunk_complete":
          const chunkData = event.data.payload as {
            chunkIndex: number;
            timings: Record<string, number>;
          };
          this.completedChunks.add(chunkData.chunkIndex);
          this.callbacks.onChunkComplete?.(
            chunkData.chunkIndex,
            chunkData.timings
          );
          if (this.completedChunks.size === this.pendingChunks.size) {
            this.callbacks.onAllComplete?.({});
            this.cleanup();
          }
          break;
        case "error":
          this.callbacks.onError?.(event.data.payload);
          this.cleanup();
          break;
      }
    };

    this.worker.onerror = (error) => {
      this.callbacks.onError?.(`Worker error: ${error.message}`);
      this.cleanup();
    };

    this.worker.postMessage({ type: "init", modelPath } as WorkerInitMessage);
  }

  /**
   * Check if the inference service is ready
   * @returns true if initialized
   */
  public isReady(): boolean {
    return this.isInitialized;
  }

  /**
   * Start streaming inference process, send shared data to worker
   * @param sharedData shared data (image, config, etc.)
   * @param callbacks inference callbacks
   * @param startImageIndex start image frame index, default is 0
   */
  public async startStreaming(
    sharedData: {
      dataset: ImageDataResponse;
      zipBlob: Blob;
      blendingMaskBitmap: ImageBitmap;
    },
    callbacks: StreamingInferenceCallbacks,
    startImageIndex = 0
  ): Promise<void> {
    if (!this.worker || !this.isInitialized) {
      throw new Error("Inference service not initialized");
    }

    this.callbacks = callbacks;
    this.pendingChunks.clear();
    this.completedChunks.clear();
    this.totalFrames = 0;
    this.isProcessing = true;

    const zipBuffer = await sharedData.zipBlob.arrayBuffer();
    const message = {
      type: "init_streaming",
      dataset: sharedData.dataset,
      zipBuffer,
      blendingMaskBitmap: sharedData.blendingMaskBitmap,
      startImageIndex,
    };

    this.worker.postMessage(message, [
      sharedData.blendingMaskBitmap,
      zipBuffer,
    ]);
  }

  /**
   * Add a chunk for inference
   * @param chunkResult chunk result
   */
  public addChunk(chunkResult: ChunkFeatureResult): void {
    if (!this.worker || !this.isProcessing) {
      console.warn(
        "Cannot add chunk: inference service not ready or not processing"
      );
      return;
    }

    const chunkFrames = chunkResult.dimensions[0];
    const startFrame = this.totalFrames;
    const endFrame = startFrame + chunkFrames;
    this.totalFrames = endFrame;

    const chunkData: StreamingChunkData = {
      chunkIndex: chunkResult.chunkIndex,
      audioFeatures: chunkResult.features,
      audioDimensions: chunkResult.dimensions,
      startFrame,
      endFrame,
    };

    this.pendingChunks.set(chunkResult.chunkIndex, chunkData);

    console.log(
      `Add chunk ${chunkResult.chunkIndex} for inference, frame range: ${startFrame}-${endFrame}`
    );

    const message: StreamingWorkerRunMessage = {
      type: "streaming_run",
      chunkData,
      dataset: {} as ImageDataResponse,
      zipBlob: new Blob(),
      blendingMaskBitmap: {} as ImageBitmap,
      startImageIndex: 0,
    };

    this.worker.postMessage(message, [chunkData.audioFeatures.buffer]);
  }

  /**
   * Notify all chunks are added, send total frames to worker
   */
  public finishAddingChunks(): void {
    if (!this.worker) return;

    this.worker.postMessage({
      type: "finish_chunks",
      totalFrames: this.totalFrames,
    });
  }

  /**
   * Clean up internal state and resources
   */
  private cleanup(): void {
    this.isProcessing = false;
    this.callbacks = {};
    this.pendingChunks.clear();
    this.completedChunks.clear();
    this.totalFrames = 0;
  }

  /**
   * Stop inference and clean up resources, notify worker to stop
   */
  public stop(): void {
    this.cleanup();
    if (this.worker) {
      this.worker.postMessage({ type: "stop" });
    }
  }

  /**
   * Terminate worker and clean up all resources
   */
  public terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      this.isInitialized = false;
    }
    this.cleanup();
  }
}
