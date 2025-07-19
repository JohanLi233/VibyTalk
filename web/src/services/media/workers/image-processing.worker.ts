/**
 * @file image-processing.worker.ts
 * @author JohanLi233
 */

interface Task {
  id: string;
  type: "process" | "resize";
  data: ArrayBuffer | ArrayBuffer[];
  options?: { width?: number; height?: number };
}

interface Result {
  id: string;
  success: boolean;
  data?: ImageBitmap | ImageBitmap[];
  error?: string;
}

class Worker {
  private canvas = new OffscreenCanvas(1024, 1024);
  private ctx = this.canvas.getContext("2d")!;

  async processTask(task: Task): Promise<Result> {
    try {
      const data = Array.isArray(task.data)
        ? await Promise.all(
            task.data.map((b) => this.processBuffer(b, task.options))
          )
        : await this.processBuffer(task.data, task.options);

      return { id: task.id, success: true, data };
    } catch (error) {
      return {
        id: task.id,
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  private async processBuffer(
    buffer: ArrayBuffer,
    options?: { width?: number; height?: number }
  ): Promise<ImageBitmap> {
    const blob = new Blob([buffer]);
    let bitmap = await createImageBitmap(blob);

    if (options?.width || options?.height) {
      const w = options.width || bitmap.width;
      const h = options.height || bitmap.height;
      this.canvas.width = w;
      this.canvas.height = h;
      this.ctx.drawImage(bitmap, 0, 0, w, h);
      bitmap.close();
      bitmap = await createImageBitmap(this.canvas);
    }

    return bitmap;
  }
}

const worker = new Worker();

self.onmessage = async (event: MessageEvent<Task>) => {
  const result = await worker.processTask(event.data);
  self.postMessage(result, {
    transfer: result.data
      ? Array.isArray(result.data)
        ? result.data
        : [result.data]
      : [],
  });
};
