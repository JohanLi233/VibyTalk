/**
 * @file StreamingFeatureExtractorService.ts
 *
 * @author JohanLi233
 */

import {
  FeatureExtractorService,
  NUM_MEL_BINS,
  NUM_SEQUENCE_FRAMES,
} from "./AudioService";
import { CHUNK_DURATION_SECONDS, PerformanceTimer } from "./core";

export interface ChunkFeatureResult {
  chunkIndex: number;
  features: Float32Array;
  dimensions: number[];
  startTimeSeconds: number;
  endTimeSeconds: number;
}

export interface StreamingCallbacks {
  /**
   * Single chunk processing complete callback
   * @param result current chunk feature result
   */
  onChunkComplete?: (result: ChunkFeatureResult) => void;
  /**
   * Progress update callback
   * @param completed number of completed chunks
   * @param total total number of chunks
   */
  onProgress?: (completed: number, total: number) => void;
  /**
   * All chunks processing complete callback
   * @param totalDimensions merged feature dimensions
   */
  onComplete?: (totalDimensions: number[]) => void;
  /**
   * Error callback
   * @param error error message
   */
  onError?: (error: string) => void;
}

/**
 * Streaming audio feature extraction service class
 * Supports parallel processing of long audio in chunks, real-time callback of chunk results, and final merging of all features
 */
export class StreamingFeatureExtractorService {
  /** callback function collection */
  private callbacks: StreamingCallbacks = {};
  /** performance timer */
  private timer = new PerformanceTimer();
  /** store the processing result of each chunk, used for final merging */
  private chunkResults: (ChunkFeatureResult | null)[] = [];
  /** total number of audio chunks */
  private totalChunks = 0;
  /** number of completed chunks */
  private completedChunks = 0;
  /** total number of processed frames */
  private totalFramesProcessed = 0;
  /** whether there is a task running */
  private isRunning = false;
  /** current audio buffer being processed */
  private audioBuffer: AudioBuffer | null = null;
  /** audio context instance, used to create AudioBuffer */
  private audioContext: AudioContext | null = null;
  /** reusable feature extractor instance, to improve performance */
  private featureExtractor: FeatureExtractorService | null = null;

  /**
   * Create AudioBuffer compatible method, handling Safari's limitations
   * @param numberOfChannels number of channels
   * @param length buffer length
   * @param sampleRate sample rate
   * @returns AudioBuffer instance
   */
  private async createAudioBuffer(
    numberOfChannels: number,
    length: number,
    sampleRate: number
  ): Promise<AudioBuffer> {
    // try using AudioBuffer constructor (modern browsers)
    try {
      return new AudioBuffer({
        numberOfChannels,
        length,
        sampleRate,
      });
    } catch (error) {
      // if failed, use AudioContext.createBuffer (Safari compatible)
      if (!this.audioContext) {
        const AudioContextClass =
          window.AudioContext || (window as any).webkitAudioContext;
        this.audioContext = new AudioContextClass();
      }

      return this.audioContext.createBuffer(
        numberOfChannels,
        length,
        sampleRate
      );
    }
  }

  /**
   * Start streaming audio feature extraction
   * @param audioBuffer input audio buffer
   * @param callbacks callback function collection
   */
  public async processStreaming(
    audioBuffer: AudioBuffer,
    callbacks: StreamingCallbacks
  ): Promise<void> {
    if (this.isRunning) {
      throw new Error("Another audio is being processed, please wait.");
    }

    this.isRunning = true;
    this.audioBuffer = audioBuffer;
    this.callbacks = callbacks;
    this.chunkResults = [];
    this.completedChunks = 0;
    this.totalFramesProcessed = 0;
    this.featureExtractor = new FeatureExtractorService(); // reusable instance

    const durationSeconds = audioBuffer.duration;
    this.totalChunks = Math.ceil(durationSeconds / CHUNK_DURATION_SECONDS);

    console.log(
      `Start streaming audio: ${durationSeconds.toFixed(2)}s, divided into ${
        this.totalChunks
      } 个chunks`
    );

    try {
      this.timer.start("totalProcessing");

      // optimization: process chunks one by one to reduce memory accumulation and allow immediate callback
      for (let chunkIndex = 0; chunkIndex < this.totalChunks; chunkIndex++) {
        if (!this.isRunning) break; // check if it has stopped
        this.timer.start(`chunk_${chunkIndex}`);
        await this.processChunk(chunkIndex);
        this.timer.end(`chunk_${chunkIndex}`);
      }

      if (this.isRunning) {
        await this.finalize();
      }

      this.timer.end("totalProcessing");
      console.log(
        "Streaming processing performance statistics:",
        this.timer.getTimings()
      );
    } catch (error) {
      // handle exception, notify upper layer
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      this.callbacks.onError?.(errorMessage);
    } finally {
      // 清理资源
      this.cleanup();
    }
  }

  /**
   * Process a single audio chunk, extract features and callback
   * @param chunkIndex chunk index (from 0)
   */
  private async processChunk(chunkIndex: number): Promise<void> {
    if (!this.audioBuffer || !this.featureExtractor) return;

    // 1. calculate the start and end time of the chunk (seconds)
    const startTimeSeconds = chunkIndex * CHUNK_DURATION_SECONDS;
    const endTimeSeconds = Math.min(
      (chunkIndex + 1) * CHUNK_DURATION_SECONDS,
      this.audioBuffer.duration
    );

    // 2. calculate the sample point range
    const startSample = Math.floor(
      startTimeSeconds * this.audioBuffer.sampleRate
    );
    const endSample = Math.floor(endTimeSeconds * this.audioBuffer.sampleRate);
    const chunkLength = endSample - startSample;

    console.log(
      `Processing chunk ${chunkIndex}: ${startTimeSeconds.toFixed(
        2
      )}s - ${endTimeSeconds.toFixed(2)}s`
    );

    // 3. create the AudioBuffer for the chunk (Safari compatible)
    const chunkBuffer = await this.createAudioBuffer(
      this.audioBuffer.numberOfChannels,
      chunkLength,
      this.audioBuffer.sampleRate
    );

    // 4. copy audio data to chunk buffer
    for (
      let channel = 0;
      channel < this.audioBuffer.numberOfChannels;
      channel++
    ) {
      const sourceData = this.audioBuffer.getChannelData(channel);
      const chunkData = chunkBuffer.getChannelData(channel);
      chunkData.set(sourceData.subarray(startSample, endSample));
    }

    // 5. use the reusable FeatureExtractorService instance to process the chunk
    try {
      this.timer.start(`extract_chunk_${chunkIndex}`);
      const result = await this.featureExtractor.process(chunkBuffer);
      this.timer.end(`extract_chunk_${chunkIndex}`);
      const chunkResult: ChunkFeatureResult = {
        chunkIndex,
        features: result.features,
        dimensions: result.dimensions,
        startTimeSeconds,
        endTimeSeconds,
      };

      // accumulate the total number of processed frames
      this.totalFramesProcessed += chunkResult.dimensions[0];

      this.chunkResults[chunkIndex] = chunkResult;
      this.completedChunks++;

      console.log(
        `Chunk ${chunkIndex} completed, feature dimensions: [${result.dimensions.join(
          ", "
        )}]`
      );

      // 7. callback to notify the upper layer (immediately transfer data)
      this.callbacks.onChunkComplete?.(chunkResult);
      this.callbacks.onProgress?.(this.completedChunks, this.totalChunks);

      // optimization: immediately release the reference to the chunk result, because it has been passed
      // and its ArrayBuffer will be transferred soon, reducing memory usage.
      this.chunkResults[chunkIndex] = null as any;
    } catch (error) {
      console.error(`处理chunk ${chunkIndex} 时发生错误:`, error);
      throw error;
    }
  }

  private async finalize(): Promise<void> {
    console.log("Finalizing all chunks...");

    if (this.completedChunks === 0) {
      this.callbacks.onError?.("No chunks were processed.");
      return;
    }

    // construct dimension information from constants. These values are fixed throughout the process
    const totalDimensions = [
      this.totalFramesProcessed,
      NUM_SEQUENCE_FRAMES,
      NUM_MEL_BINS,
    ];

    console.log(
      `Feature processing completed, total dimensions: [${totalDimensions.join(
        ", "
      )}]`
    );

    // 4. callback to notify all complete
    this.callbacks.onComplete?.(totalDimensions);
  }

  /**
   * Clean up internal state and resources
   */
  private cleanup(): void {
    this.isRunning = false;
    this.audioBuffer = null;

    // terminate the reusable Worker
    if (this.featureExtractor) {
      this.featureExtractor.terminate();
      this.featureExtractor = null;
    }

    // clean up the chunk result array
    this.chunkResults.forEach((chunk) => {
      if (chunk && chunk.features) {
        // release Float32Array memory (if possible)
        (chunk.features as any) = null;
      }
    });
    this.chunkResults = [];

    this.callbacks = {};
    this.completedChunks = 0;
    this.totalChunks = 0;
    this.totalFramesProcessed = 0;

    // reset the performance timer
    this.timer.reset();
  }

  /**
   * Stop the current streaming processing
   */
  public stop(): void {
    if (this.isRunning) {
      console.log("Stopping streaming audio processing");
      this.isRunning = false;

      // terminate the reusable Worker
      if (this.featureExtractor) {
        this.featureExtractor.terminate();
        this.featureExtractor = null;
      }

      this.callbacks = {};
      this.chunkResults = [];
      this.totalChunks = 0;
      this.completedChunks = 0;
      this.totalFramesProcessed = 0; // reset the accumulator
    }
  }
}
