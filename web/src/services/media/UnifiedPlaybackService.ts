/**
 * @file UnifiedPlaybackService.ts
 *
 * @author JohanLi233
 */

import { ref, shallowRef } from "vue";
import { FRAME_INTERVAL_MS } from "../core";
import JSZip from "jszip";
import type { ImageDataResponse } from "./DataLoaderService";

class WebGLRenderer {
  private canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext | WebGL2RenderingContext;
  private program: WebGLProgram | null = null;
  private textureCache = new Map<ImageBitmap, WebGLTexture>();

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) throw new Error("WebGL not supported");
    this.gl = gl;
  }

  async initialize(): Promise<void> {
    const vs = this.gl.createShader(this.gl.VERTEX_SHADER)!;
    this.gl.shaderSource(
      vs,
      `
      attribute vec2 a_pos;
      attribute vec2 a_tex;
      varying vec2 v_tex;
      void main() {
        gl_Position = vec4(a_pos, 0.0, 1.0);
        v_tex = a_tex;
      }
    `
    );
    this.gl.compileShader(vs);

    const fs = this.gl.createShader(this.gl.FRAGMENT_SHADER)!;
    this.gl.shaderSource(
      fs,
      `
      precision mediump float;
      uniform sampler2D u_tex;
      varying vec2 v_tex;
      void main() {
        gl_FragColor = texture2D(u_tex, v_tex);
      }
    `
    );
    this.gl.compileShader(fs);

    this.program = this.gl.createProgram()!;
    this.gl.attachShader(this.program, vs);
    this.gl.attachShader(this.program, fs);
    this.gl.linkProgram(this.program);
    this.gl.useProgram(this.program);

    const vb = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, vb);
    this.gl.bufferData(
      this.gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 0, 1, 1, -1, 1, 1, -1, 1, 0, 0, 1, 1, 1, 0]),
      this.gl.STATIC_DRAW
    );

    const pos = this.gl.getAttribLocation(this.program, "a_pos");
    const tex = this.gl.getAttribLocation(this.program, "a_tex");
    this.gl.enableVertexAttribArray(pos);
    this.gl.enableVertexAttribArray(tex);
    this.gl.vertexAttribPointer(pos, 2, this.gl.FLOAT, false, 16, 0);
    this.gl.vertexAttribPointer(tex, 2, this.gl.FLOAT, false, 16, 8);
  }

  render(bitmap: ImageBitmap): void {
    let texture = this.textureCache.get(bitmap);
    if (!texture) {
      texture = this.gl.createTexture()!;
      this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
      this.gl.texImage2D(
        this.gl.TEXTURE_2D,
        0,
        this.gl.RGBA,
        this.gl.RGBA,
        this.gl.UNSIGNED_BYTE,
        bitmap
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_MIN_FILTER,
        this.gl.LINEAR
      );
      this.gl.texParameteri(
        this.gl.TEXTURE_2D,
        this.gl.TEXTURE_MAG_FILTER,
        this.gl.LINEAR
      );
      this.textureCache.set(bitmap, texture);
    }
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
  }

  clearTexture(bitmap: ImageBitmap): void {
    const texture = this.textureCache.get(bitmap);
    if (texture) {
      this.gl.deleteTexture(texture);
      this.textureCache.delete(bitmap);
    }
  }

  cleanup(): void {
    this.textureCache.forEach((t) => this.gl.deleteTexture(t));
    this.textureCache.clear();
  }

  static isSupported(): boolean {
    try {
      const canvas = document.createElement("canvas");
      return !!(canvas.getContext("webgl") || canvas.getContext("webgl2"));
    } catch {
      return false;
    }
  }
}

export type RenderMode = "webgl" | "canvas2d";

export class UnifiedPlaybackService {
  private renderer: WebGLRenderer | CanvasRenderingContext2D;
  private renderMode: RenderMode;
  private canvas: HTMLCanvasElement;
  private audioPlayer: HTMLAudioElement;
  private inferenceRunning = true;
  private animationFrameId: number | null = null;
  private lastFrameTimestamp = 0;
  private onPlaybackEndCallback: (() => void | Promise<void>) | null = null;

  private dataset?: ImageDataResponse;
  private zipInstance?: JSZip;

  public readonly isPlaying = ref(false);
  public readonly currentFrameIndex = ref(0);
  public readonly generatedFrames = shallowRef<ImageBitmap[]>([]);

  constructor(
    canvas: HTMLCanvasElement,
    audioPlayer: HTMLAudioElement,
    onPlaybackEndCallback: (() => void | Promise<void>) | null = null,
    preferWebGL: boolean = true
  ) {
    this.canvas = canvas;
    this.audioPlayer = audioPlayer;
    this.onPlaybackEndCallback = onPlaybackEndCallback;

    if (preferWebGL && WebGLRenderer.isSupported()) {
      this.renderMode = "webgl";
      this.renderer = new WebGLRenderer(canvas);
    } else {
      this.renderMode = "canvas2d";
      this.renderer = canvas.getContext("2d")!;
    }
  }

  async initialize(): Promise<void> {
    if (this.renderMode === "webgl") {
      await (this.renderer as WebGLRenderer).initialize();
    }
  }

  setFrameData(dataset: ImageDataResponse, zipInstance: JSZip): void {
    this.dataset = dataset;
    this.zipInstance = zipInstance;
  }

  setInferenceRunning(status: boolean): void {
    this.inferenceRunning = status;
    if (
      !status &&
      this.isPlaying.value &&
      this.currentFrameIndex.value >= this.generatedFrames.value.length
    ) {
      this.stop();
    }
  }

  addFrame(frameBitmap: ImageBitmap): void {
    this.generatedFrames.value.push(frameBitmap);
    if (this.generatedFrames.value.length === 1 && !this.isPlaying.value) {
      this.start();
    }
  }

  start(): void {
    if (this.isPlaying.value) return;
    this.isPlaying.value = true;
    this.lastFrameTimestamp = performance.now();

    if (this.audioPlayer) {
      this.audioPlayer.currentTime =
        this.currentFrameIndex.value * (FRAME_INTERVAL_MS / 1000);
      this.audioPlayer.play().catch(() => {
        this.audioPlayer.muted = true;
        this.audioPlayer.play().catch(() => {});
      });
    }

    this.animationFrameId = requestAnimationFrame(this.playbackLoop.bind(this));
  }

  stop(): void {
    this.isPlaying.value = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    this.audioPlayer?.pause();
  }

  cleanup(): void {
    this.stop();
    this.generatedFrames.value.forEach((frame) => {
      if (frame) {
        if (this.renderMode === "webgl") {
          (this.renderer as WebGLRenderer).clearTexture(frame);
        }
        frame.close();
      }
    });
    this.generatedFrames.value = [];
    if (this.renderMode === "webgl") {
      (this.renderer as WebGLRenderer).cleanup();
    }
    this.currentFrameIndex.value = 0;
    this.inferenceRunning = true;
  }

  async renderSilentFrame(frame: ImageBitmap): Promise<void> {
    if (this.renderMode === "webgl") {
      (this.renderer as WebGLRenderer).render(frame);
      (this.renderer as WebGLRenderer).clearTexture(frame);
    } else {
      (this.renderer as CanvasRenderingContext2D).drawImage(
        frame,
        0,
        0,
        this.canvas.width,
        this.canvas.height
      );
    }
  }

  private playbackLoop(timestamp: number): void {
    if (!this.isPlaying.value) return;

    const elapsed = timestamp - this.lastFrameTimestamp;
    if (elapsed >= FRAME_INTERVAL_MS) {
      this.lastFrameTimestamp = timestamp - (elapsed % FRAME_INTERVAL_MS);

      if (this.generatedFrames.value.length > 0) {
        const frame = this.generatedFrames.value.shift()!;

        if (frame) {
          if (this.renderMode === "webgl") {
            (this.renderer as WebGLRenderer).render(frame);
            (this.renderer as WebGLRenderer).clearTexture(frame);
          } else {
            (this.renderer as CanvasRenderingContext2D).drawImage(
              frame,
              0,
              0,
              this.canvas.width,
              this.canvas.height
            );
          }

          frame.close();
        }

        this.currentFrameIndex.value++;
      }

      if (!this.inferenceRunning && this.generatedFrames.value.length === 0) {
        this.stop();
        this.cleanup();
        this.onPlaybackEndCallback?.();
        return;
      }
    }

    this.animationFrameId = requestAnimationFrame(this.playbackLoop.bind(this));
  }

  static isWebGLSupported(): boolean {
    return WebGLRenderer.isSupported();
  }

  enableAudio(): void {
    if (this.audioPlayer) {
      this.audioPlayer.muted = false;
      if (this.isPlaying.value) {
        this.audioPlayer.play().catch(() => {});
      }
    }
  }

  isAudioEnabled(): boolean {
    return !this.audioPlayer?.muted;
  }
}

export {
  UnifiedPlaybackService as PlaybackService,
  UnifiedPlaybackService as WebGLPlaybackService,
};
