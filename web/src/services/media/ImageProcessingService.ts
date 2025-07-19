/**
 * @file ImageProcessingService.ts
 * @author JohanLi233
 */

import { Tensor } from "onnxruntime-web";
import type JSZip from "jszip";

let sharedCanvas: HTMLCanvasElement | OffscreenCanvas | null = null;
let sharedCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;

export function getSharedCanvasContext(width: number, height: number) {
  if (!sharedCanvas) {
    sharedCanvas = typeof OffscreenCanvas !== "undefined" && typeof document === "undefined" 
      ? new OffscreenCanvas(width, height) 
      : document.createElement("canvas");
    sharedCtx = sharedCanvas.getContext("2d", { willReadFrequently: true });
    if (!sharedCtx) throw new Error("Failed to create canvas context");
  }
  if (sharedCanvas.width !== width || sharedCanvas.height !== height) {
    sharedCanvas.width = width;
    sharedCanvas.height = height;
  }
  return sharedCtx!;
}

export function calculatePingPongState(currentIndex: number, numFrames: number, direction: number) {
  if (numFrames <= 1) return { nextIndex: 0, nextDirection: 1 };
  let nextIndex = currentIndex + direction;
  let nextDirection = direction;
  if (nextIndex >= numFrames) {
    nextIndex = numFrames - 2;
    nextDirection = -1;
  } else if (nextIndex < 0) {
    nextIndex = 1;
    nextDirection = 1;
  }
  return { nextIndex, nextDirection };
}

export async function loadTensorFromZip(
  tensorFileName: string, zip: JSZip, cropSize: number, maskRegion: [number, number, number, number]
): Promise<Tensor> {
  const zipFile = zip.file(tensorFileName);
  if (!zipFile) throw new Error(`Tensor file not found: ${tensorFileName}`);
  const border = maskRegion[0];
  const innerSize = cropSize - 2 * border;
  const buffer = await zipFile.async("arraybuffer");
  const tensorData = new Float32Array(buffer);
  return new Tensor("float32", tensorData, [1, 6, innerSize, innerSize]);
}

export const convertTensorToImageData = (tensor: Tensor): ImageData | null => {
  const [, channels, height, width] = tensor.dims;
  if (tensor.dims.length !== 4 || (channels !== 3 && channels !== 1)) return null;
  
  const ctx = getSharedCanvasContext(width, height);
  const imageData = ctx.createImageData(width, height);
  const data = tensor.data as Float32Array;
  const pixels = imageData.data;
  const channelSize = height * width;
  
  for (let i = 0; i < height * width; i++) {
    const pixelIndex = i * 4;
    const clamp = (v: number) => Math.min(Math.max(v, 0), 1) * 255;
    
    const r_val = data[i];
    const g_val = data[i + channelSize];
    const b_val = data[i + 2 * channelSize];
    
    pixels[pixelIndex] = clamp(r_val);             // R
    pixels[pixelIndex + 1] = clamp(g_val);         // G  
    pixels[pixelIndex + 2] = clamp(b_val);         // B
    pixels[pixelIndex + 3] = 255;                  // A
  }
  
  return imageData;
};

export class ImageProcessingService {
  private bitmapPool = new Map<string, ImageBitmap>();
  private maxPoolSize = 8;
  
  async processImage(buffer: ArrayBuffer, options?: { width?: number; height?: number }): Promise<ImageBitmap> {
    const blob = new Blob([buffer]);
    let bitmap = await createImageBitmap(blob);
    
    if (options?.width || options?.height) {
      const canvas = getSharedCanvasContext(options.width || bitmap.width, options.height || bitmap.height);
      canvas.canvas.width = options.width || bitmap.width;
      canvas.canvas.height = options.height || bitmap.height;
      canvas.drawImage(bitmap, 0, 0, canvas.canvas.width, canvas.canvas.height);
      bitmap.close();
      bitmap = await createImageBitmap(canvas.canvas);
    }
    
    return bitmap;
  }
  
  async processImageBatch(buffers: ArrayBuffer[]): Promise<ImageBitmap[]> {
    return Promise.all(buffers.map(buffer => this.processImage(buffer)));
  }
  
  cleanup(): void {
    this.bitmapPool.forEach(bitmap => bitmap.close());
    this.bitmapPool.clear();
  }
}

export const globalImageProcessingService = new ImageProcessingService();