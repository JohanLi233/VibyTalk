/**
 * @file streaming.inference.worker.ts
 * @author JohanLi233
 */

import { InferenceSession, Tensor, env } from "onnxruntime-web";
import JSZip from "jszip";
import { getAudioWindow } from "../../AudioService";
import {
  convertTensorToImageData,
  loadTensorFromZip,
  calculatePingPongState,
} from "../../media/ImageProcessingService";
import type {
  DatasetInfo,
  ImageDataResponse,
  ImageMetadata,
} from "../../media/DataLoaderService";
import type { StreamingChunkData } from "../StreamingInferenceService";

env.wasm.wasmPaths = "/";

declare interface GPUAdapter {}

export type StreamingWorkerInitMessage = {
  type: "init";
  modelPath: string;
};

export type StreamingWorkerInitStreamingMessage = {
  type: "init_streaming";
  dataset: ImageDataResponse;
  zipBuffer: ArrayBuffer;
  blendingMaskBitmap: ImageBitmap;
  startImageIndex: number;
};

export type StreamingWorkerRunMessage = {
  type: "streaming_run";
  chunkData: StreamingChunkData;
  dataset: ImageDataResponse;
  imageBlobs: Map<string, Blob>;
  blendingMaskBitmap: ImageBitmap;
};

export type StreamingWorkerFinishMessage = {
  type: "finish_chunks";
  totalFrames: number;
};

export type StreamingWorkerStopMessage = {
  type: "stop";
};

export type StreamingWorkerMessage =
  | StreamingWorkerInitMessage
  | StreamingWorkerInitStreamingMessage
  | StreamingWorkerRunMessage
  | StreamingWorkerFinishMessage
  | StreamingWorkerStopMessage;

export type MainThreadFrameMessage = {
  type: "frame";
  payload: {
    frame: ImageBitmap;
    frameIndex: number;
  };
};

export type MainThreadChunkCompleteMessage = {
  type: "chunk_complete";
  payload: {
    chunkIndex: number;
    timings: Record<string, number>;
  };
};

export type MainThreadProgressMessage = {
  type: "progress";
  payload: {
    processed: number;
    total: number;
  };
};

export type MainThreadAllCompleteMessage = {
  type: "all_complete";
  payload: {
    totalTimings: Record<string, number>;
  };
};

export type MainThreadErrorMessage = {
  type: "error";
  payload: string;
};

export type MainThreadReadyMessage = {
  type: "ready";
};

export type MainThreadMessage =
  | MainThreadFrameMessage
  | MainThreadChunkCompleteMessage
  | MainThreadProgressMessage
  | MainThreadAllCompleteMessage
  | MainThreadErrorMessage
  | MainThreadReadyMessage;

interface NavigatorWithGPU extends Navigator {
  gpu?: {
    requestAdapter(): Promise<GPUAdapter | null>;
  };
}

interface WasmSessionOptions {
  wasm?: {
    numThreads?: number;
  };
}

const detectBestExecutionProvider = async (): Promise<{
  providers: string[];
  sessionOptions: WasmSessionOptions;
}> => {
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await (
        navigator as NavigatorWithGPU
      ).gpu?.requestAdapter();
      if (adapter) {
        console.log("WebGPU is available, using 'webgpu' execution provider.");
        return {
          providers: ["webgpu"],
          sessionOptions: {
            wasm: {
              numThreads: navigator.hardwareConcurrency || 4,
            },
          },
        };
      }
    } catch (e) {
      console.log("WebGPU detection failed, falling back to WASM.", e);
    }
  }

  console.log("Using 'wasm' execution provider.");
  return {
    providers: ["wasm"],
    sessionOptions: {
      wasm: {
        numThreads: navigator.hardwareConcurrency || 4,
      },
    },
  };
};

class StreamingONNXRunner {
  private session: InferenceSession | null = null;

  async initialize(modelPath: string): Promise<void> {
    const response = await fetch(modelPath);
    const modelBuffer = await response.arrayBuffer();

    const { providers, sessionOptions } = await detectBestExecutionProvider();
    console.log(
      `Initializing streaming ONNX session with providers: ${providers.join(
        ", "
      )}`
    );

    this.session = await InferenceSession.create(new Uint8Array(modelBuffer), {
      executionProviders: providers,
      graphOptimizationLevel: "all",
      ...sessionOptions,
    } as InferenceSession.SessionOptions);

    console.log("Streaming ONNX session created successfully");
  }

  /**
   * Run inference once
   * @param imageTensor Input image tensor
   * @param audioTensor Input audio tensor
   * @returns Inference output tensor
   */
  async runInference(
    imageTensor: Tensor,
    audioTensor: Tensor
  ): Promise<Tensor> {
    if (!this.session) {
      throw new Error("ONNX session not initialized.");
    }
    const feeds = {
      [this.session.inputNames[0]]: imageTensor,
      [this.session.inputNames[1]]: audioTensor,
    };
    const results = await this.session.run(feeds);
    return results[this.session.outputNames[0]];
  }
}

/**
 * Composite final frame image
 * @param predTensor 推理输出张量
 * @param frameMeta 当前帧元数据
 * @param datasetInfo 数据集配置信息
 * @param bitmaps 当前帧所需的ImageBitmap集合
 * @param maskImage 融合掩码ImageBitmap
 * @returns [final frame ImageBitmap, timings]
 */
async function compositeFrame(
  predTensor: Tensor,
  frameMeta: ImageMetadata,
  datasetInfo: DatasetInfo,
  bitmaps: Map<string, ImageBitmap>,
  maskImage: ImageBitmap
): Promise<[ImageBitmap, Record<string, number>]> {
  const t: Record<string, number> = {
    convertTensor: 0,
    createPredCanvas: 0,
    createPastedCanvas: 0,
    blendOps: 0,
    compositeFinal: 0,
    createImageBitmap: 0,
  };
  let t0 = performance.now();

  const fullImage = bitmaps.get(frameMeta.full_image);
  const faceImage = bitmaps.get(frameMeta.face_image);

  if (!fullImage || !faceImage) {
    throw new Error(`Missing image data for frame ${frameMeta.frame_id}`);
  }

  if (
    !predCanvas ||
    !predCtx ||
    !pastedPredCanvas ||
    !pastedPredCtx ||
    !blendedFaceCanvas ||
    !blendedFaceCtx ||
    !finalCanvas ||
    !finalCtx
  ) {
    throw new Error("Reusable Canvas instances not initialized");
  }

  t0 = performance.now();
  const predImageData = convertTensorToImageData(predTensor);
  t.convertTensor = performance.now() - t0;
  if (!predImageData) {
    throw new Error(
      `Failed to convert tensor to image for frame ${frameMeta.frame_id}`
    );
  }

  const cropSize = datasetInfo.config.crop_size;
  let border: number;
  if (cropSize === 252) {
    border = 6;
  } else if (cropSize === 192) {
    border = 6;
  } else if (cropSize === 128) {
    border = 4;
  } else {
    // cropSize === 96 (nano/tiny)
    border = 3;
  }

  t0 = performance.now();
  predCtx.clearRect(0, 0, predCanvas.width, predCanvas.height);
  predCtx.putImageData(predImageData, 0, 0);
  t.createPredCanvas = performance.now() - t0;

  t0 = performance.now();
  pastedPredCtx.clearRect(
    0,
    0,
    pastedPredCanvas.width,
    pastedPredCanvas.height
  );
  pastedPredCtx.drawImage(faceImage, 0, 0);
  pastedPredCtx.drawImage(predCanvas, border, border);
  t.createPastedCanvas = performance.now() - t0;

  t0 = performance.now();
  blendedFaceCtx.clearRect(
    0,
    0,
    blendedFaceCanvas.width,
    blendedFaceCanvas.height
  );
  blendedFaceCtx.drawImage(pastedPredCanvas, 0, 0);
  blendedFaceCtx.globalCompositeOperation = "destination-in";
  blendedFaceCtx.drawImage(maskImage, 0, 0, cropSize, cropSize);
  blendedFaceCtx.globalCompositeOperation = "destination-over";
  blendedFaceCtx.drawImage(faceImage, 0, 0);
  blendedFaceCtx.globalCompositeOperation = "source-over";
  t.blendOps = performance.now() - t0;

  t0 = performance.now();
  finalCtx.clearRect(0, 0, finalCanvas.width, finalCanvas.height);
  finalCtx.drawImage(fullImage, 0, 0);
  const { xmin, ymin, width } = frameMeta.crop_info;
  const cropHeight = frameMeta.crop_info.ymax - ymin;
  finalCtx.drawImage(blendedFaceCanvas, xmin, ymin, width, cropHeight);
  t.compositeFinal = performance.now() - t0;

  t0 = performance.now();
  const finalBitmap = await createImageBitmap(finalCanvas);
  t.createImageBitmap = performance.now() - t0;

  return [finalBitmap, t];
}

const onnxRunner = new StreamingONNXRunner();
let sharedDataset: ImageDataResponse | null = null;
let sharedZip: JSZip | null = null;
let sharedBlendingMask: ImageBitmap | null = null;
let totalExpectedFrames = 0;
let processedFrames = 0;
let imageIndex = 0;
let imageStep = 1;

let predCanvas: OffscreenCanvas | null = null;
let predCtx: OffscreenCanvasRenderingContext2D | null = null;
let pastedPredCanvas: OffscreenCanvas | null = null;
let pastedPredCtx: OffscreenCanvasRenderingContext2D | null = null;
let blendedFaceCanvas: OffscreenCanvas | null = null;
let blendedFaceCtx: OffscreenCanvasRenderingContext2D | null = null;
let finalCanvas: OffscreenCanvas | null = null;
let finalCtx: OffscreenCanvasRenderingContext2D | null = null;

const chunkQueue = new Map<number, StreamingChunkData>();
let isProcessingQueue = false;
let nextChunkToProcess = 0;

function initializeReusableCanvases(
  cropSize: number,
  border: number,
  fullImageWidth: number,
  fullImageHeight: number
): void {
  const predSize = cropSize - 2 * border;
  predCanvas = new OffscreenCanvas(predSize, predSize);
  predCtx = predCanvas.getContext("2d")!;

  pastedPredCanvas = new OffscreenCanvas(cropSize, cropSize);
  pastedPredCtx = pastedPredCanvas.getContext("2d")!;

  blendedFaceCanvas = new OffscreenCanvas(cropSize, cropSize);
  blendedFaceCtx = blendedFaceCanvas.getContext("2d")!;

  finalCanvas = new OffscreenCanvas(fullImageWidth, fullImageHeight);
  finalCtx = finalCanvas.getContext("2d")!;

  console.log(
    `Initialized reusable Canvas instances, crop size: ${cropSize}, pred size: ${predSize}, full image size: ${fullImageWidth}x${fullImageHeight}`
  );
}

/**
 * Clean up reusable Canvas instances
 */
function cleanupReusableCanvases(): void {
  predCanvas = null;
  predCtx = null;
  pastedPredCanvas = null;
  pastedPredCtx = null;
  blendedFaceCanvas = null;
  blendedFaceCtx = null;
  finalCanvas = null;
  finalCtx = null;
}

async function loadImageBitmapFromZip(
  imagePath: string,
  zip: JSZip
): Promise<ImageBitmap> {
  const file = zip.file(imagePath);
  if (!file) {
    throw new Error(`Image file not found in ZIP: ${imagePath}`);
  }

  const blob = await file.async("blob");
  return await createImageBitmap(blob);
}

/**
 * Process a single chunk, frame by frame inference and composition
 * @param chunkData chunk data
 */
async function processChunk(chunkData: StreamingChunkData): Promise<void> {
  if (!sharedDataset || !sharedZip || !sharedBlendingMask) {
    throw new Error("Shared data not initialized, cannot process chunk");
  }

  const { chunkIndex, audioFeatures, audioDimensions, startFrame, endFrame } =
    chunkData;

  console.log(`Processing chunk ${chunkIndex}, frame range: ${startFrame}-${endFrame}`);

  const timings = {
    loadTensor: 0,
    getAudio: 0,
    onnxRun: 0,
    composite: 0,
    totalFrames: 0,
    t_convertTensor: 0,
    t_createPredCanvas: 0,
    t_createPastedCanvas: 0,
    t_blendOps: 0,
    t_compositeFinal: 0,
    t_createImageBitmap: 0,
  };

  const imageFrames = sharedDataset.images;
  const numImageFrames = imageFrames.length;
  if (numImageFrames === 0) {
    throw new Error("Image data is empty.");
  }

  const [, chunk_size, mel_bins] = audioDimensions;
  const audioWindowSize = 32;
  const numFramesToProcess = audioDimensions[0];

  for (let i = 0; i < numFramesToProcess; i++) {
    const globalFrameIndex = startFrame + i;
    const imageFrameMeta = imageFrames[imageIndex];

    let t0 = performance.now();
    const cropSize = sharedDataset.dataset_info.config.crop_size;
    let dynamicBorder: number;
    if (cropSize === 252) {
      dynamicBorder = 6;
    } else if (cropSize === 192) {
      dynamicBorder = 6;
    } else if (cropSize === 128) {
      dynamicBorder = 4;
    } else {
      // cropSize === 96 (nano/tiny)
      dynamicBorder = 3;
    }

    const imageTensor = await loadTensorFromZip(
      imageFrameMeta.tensor_file,
      sharedZip,
      cropSize,
      [
        dynamicBorder,
        dynamicBorder,
        cropSize - dynamicBorder,
        cropSize - dynamicBorder,
      ] as [number, number, number, number]
    );
    timings.loadTensor += performance.now() - t0;

    t0 = performance.now();
    const audioWindowData = getAudioWindow(audioFeatures, audioDimensions, i);
    const audioTensor = new Tensor("float32", audioWindowData, [
      1,
      audioWindowSize,
      chunk_size,
      mel_bins,
    ]);
    timings.getAudio += performance.now() - t0;

    t0 = performance.now();
    const outputTensor = await onnxRunner.runInference(
      imageTensor,
      audioTensor
    );
    timings.onnxRun += performance.now() - t0;

    try {
      const fullImageBitmap = await loadImageBitmapFromZip(
        imageFrameMeta.full_image,
        sharedZip
      );
      const faceImageBitmap = await loadImageBitmapFromZip(
        imageFrameMeta.face_image,
        sharedZip
      );

      const tempBitmaps = new Map<string, ImageBitmap>();
      tempBitmaps.set(imageFrameMeta.full_image, fullImageBitmap);
      tempBitmaps.set(imageFrameMeta.face_image, faceImageBitmap);

      t0 = performance.now();
      const [finalFrameBitmap, frameTimings] = await compositeFrame(
        outputTensor,
        imageFrameMeta,
        sharedDataset.dataset_info,
        tempBitmaps,
        sharedBlendingMask
      );
      timings.composite += performance.now() - t0;

      self.postMessage(
        {
          type: "frame",
          payload: {
            frame: finalFrameBitmap,
            frameIndex: globalFrameIndex,
          },
        } as MainThreadFrameMessage,
        {
          transfer: [finalFrameBitmap],
        }
      );

      processedFrames++;
      self.postMessage({
        type: "progress",
        payload: { processed: processedFrames, total: totalExpectedFrames },
      } as MainThreadProgressMessage);

      timings.t_convertTensor += frameTimings.convertTensor;
      timings.t_createPredCanvas += frameTimings.createPredCanvas;
      timings.t_createPastedCanvas += frameTimings.createPastedCanvas;
      timings.t_blendOps += frameTimings.blendOps;
      timings.t_compositeFinal += frameTimings.compositeFinal;
      timings.t_createImageBitmap += frameTimings.createImageBitmap;
    } catch (error) {
      console.error(`Error processing frame ${globalFrameIndex}:`, error);
      throw error;
    }

    const { nextIndex, nextDirection } = calculatePingPongState(
      imageIndex,
      numImageFrames,
      imageStep
    );
    imageIndex = nextIndex;
    imageStep = nextDirection;
    timings.totalFrames++;

    imageTensor.dispose();
    audioTensor.dispose();
    outputTensor.dispose();
  }

  console.log(`Chunk ${chunkIndex} processed`);

  self.postMessage({
    type: "chunk_complete",
    payload: {
      chunkIndex,
      timings,
    },
  } as MainThreadChunkCompleteMessage);
}

/**
 * Process chunk queue, ensure sequential processing
 */
async function processQueue() {
  if (isProcessingQueue) return;
  if (chunkQueue.has(nextChunkToProcess)) {
    isProcessingQueue = true;

    const chunkToProcess = chunkQueue.get(nextChunkToProcess)!;
    chunkQueue.delete(nextChunkToProcess);

    console.log(`Processing chunk ${chunkToProcess.chunkIndex}`);

    try {
      await processChunk(chunkToProcess);
    } catch (e: unknown) {
      self.postMessage({
        type: "error",
        payload: (e as Error).message || "Unknown error occurred while processing chunk",
      } as MainThreadErrorMessage);
    } finally {
      nextChunkToProcess++;
      isProcessingQueue = false;
      queueMicrotask(processQueue);
    }
  }
}

/**
 * Worker main message processing entry
 * Execute different actions based on message type
 */
self.onmessage = async (event: MessageEvent<StreamingWorkerMessage>) => {
  try {
    const { type } = event.data;

    if (type === "init") {
      await onnxRunner.initialize(event.data.modelPath);
      self.postMessage({ type: "ready" } as MainThreadReadyMessage);
    } else if (type === "init_streaming") {
      sharedDataset = event.data.dataset;
      sharedZip = await JSZip.loadAsync(event.data.zipBuffer);
      sharedBlendingMask = event.data.blendingMaskBitmap;

      cleanupReusableCanvases();

      const datasetInfo = sharedDataset.dataset_info;
      const cropSize = datasetInfo.config.crop_size;
      let border: number;
      if (cropSize === 252) {
        border = 6;
      } else if (cropSize === 192) {
        border = 6;
      } else if (cropSize === 128) {
        border = 4;
      } else {
        // cropSize === 96 (nano/tiny)
        border = 3;
      }
      const sourceDims = datasetInfo.source_image_dimensions;

      if (!sourceDims) {
        throw new Error("Source image dimensions not found in dataset");
      }

      initializeReusableCanvases(
        cropSize,
        border,
        sourceDims.width,
        sourceDims.height
      );

      processedFrames = 0;
      totalExpectedFrames = 0;

      const numImages = sharedDataset.images.length;
      if (numImages > 0) {
        const resumeFromIndex = event.data.startImageIndex || 0;
        if (numImages > 1) {
          const cycleLen = (numImages - 1) * 2;
          const effectiveIndex = cycleLen > 0 ? resumeFromIndex % cycleLen : 0;

          if (effectiveIndex < numImages) {
            imageIndex = effectiveIndex;
            imageStep = 1;
          } else {
            imageIndex = cycleLen - effectiveIndex;
            imageStep = -1;
          }
        } else {
          imageIndex = 0;
          imageStep = 1;
        }
      } else {
        imageIndex = 0;
        imageStep = 1;
      }

      chunkQueue.clear();
      isProcessingQueue = false;
      nextChunkToProcess = 0;
    } else if (type === "streaming_run") {
      const chunkData = event.data.chunkData;
      chunkQueue.set(chunkData.chunkIndex, chunkData);
      processQueue();
    } else if (type === "finish_chunks") {
      totalExpectedFrames = event.data.totalFrames;
    } else if (type === "stop") {
      sharedDataset = null;
      sharedZip = null;
      if (sharedBlendingMask) {
        sharedBlendingMask.close();
        sharedBlendingMask = null;
      }

      cleanupReusableCanvases();

      processedFrames = 0;
      totalExpectedFrames = 0;
      imageIndex = 0;
      imageStep = 1;
      chunkQueue.clear();
      isProcessingQueue = false;
      nextChunkToProcess = 0;
    }
  } catch (e: unknown) {
    const error = e as Error;
    console.error("Streaming inference worker error:", error);
    self.postMessage({
      type: "error",
      payload:
        error.message ||
        "An unknown error occurred in the streaming inference worker.",
    } as MainThreadErrorMessage);
  }
};
