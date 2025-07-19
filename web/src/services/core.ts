/**
 * @file core.ts
 * @author JohanLi233
 */

export const NUM_MEL_BINS = 80;
export const NUM_SEQUENCE_FRAMES = 4;

export const CHUNK_DURATION_SECONDS = 5;
export const TARGET_FPS = 25;

export const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;

export const SAMPLE_RATE = 16000;
export const FRAME_LENGTH_MS = 25;

export const FRAME_SHIFT_MS = 10;
export const ENERGY_FLOOR = 1e-10;

export const PREEMPH_COEFF = 0.97;

export const WINDOW_NUM_CHUNKS = 32;
export const WINDOW_NUM_CHUNKS_HALF = WINDOW_NUM_CHUNKS / 2;

export const DEFAULT_LOW_FREQ = 20;
export const REMOVE_DC_OFFSET = true;

export const ROUND_TO_POWER_OF_TWO = true;

/**
 * Performance timer class
 * Used to measure the execution time of code segments
 */
export class PerformanceTimer {
  private timings: Record<string, number> = {};
  private startTimes: Record<string, number> = {};

  /**
   * Start timing
   * @param name timer name
   */
  start(name: string): void {
    this.startTimes[name] = performance.now();
  }

  /**
   * End timing and record
   * @param name timer name
   * @returns elapsed time (milliseconds)
   */
  end(name: string): number {
    const startTime = this.startTimes[name];
    if (startTime === undefined) {
      console.warn(`Timer "${name}" was not started`);
      return 0;
    }

    const elapsed = performance.now() - startTime;
    this.timings[name] = elapsed;
    delete this.startTimes[name];

    return elapsed;
  }

  /**
   * Get all timing results
   * @returns timing result object
   */
  getTimings(): Record<string, number> {
    return { ...this.timings };
  }

  /**
   * Reset all timers
   */
  reset(): void {
    this.timings = {};
    this.startTimes = {};
  }

  /**
   * Accumulate timing results
   * @param name timer name
   * @param value value to accumulate
   */
  accumulate(name: string, value: number): void {
    this.timings[name] = (this.timings[name] || 0) + value;
  }

  /**
   * Merge timing results from other timers
   * @param other other timer instance
   */
  merge(other: PerformanceTimer): void {
    const otherTimings = other.getTimings();
    for (const [name, value] of Object.entries(otherTimings)) {
      this.accumulate(name, value);
    }
  }
}

/**
 * Create performance measurement decorator
 * @param timerName timer name
 * @param timer timer instance
 * @returns decorator function
 */
export function measurePerformance(timerName: string, timer: PerformanceTimer) {
  return function (
    _target: any,
    _propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;

    descriptor.value = function (...args: any[]) {
      timer.start(timerName);
      const result = originalMethod.apply(this, args);

      if (result instanceof Promise) {
        return result.finally(() => {
          timer.end(timerName);
        });
      } else {
        timer.end(timerName);
        return result;
      }
    };

    return descriptor;
  };
}

/**
 * Standardize error object
 * @param error original error
 * @returns standardized error message
 */
export function normalizeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === "string") {
    return error;
  }

  return "Unknown error occurred";
}

/**
 * Create error handling wrapper
 * @param context error context
 * @param fallbackMessage fallback error message
 * @returns error handling function
 */
export function createErrorHandler(
  context: string,
  _fallbackMessage: string = "Operation failed"
) {
  return function (error: unknown): never {
    const errorMessage = normalizeError(error);
    const fullMessage = `${context}: ${errorMessage}`;

    console.error(fullMessage);
    throw new Error(fullMessage);
  };
}

/**
 * Async operation with retry
 * @param operation operation to execute
 * @param maxRetries maximum retry count
 * @param delay retry interval (milliseconds)
 * @returns Promise result
 */
export async function withRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> {
  let lastError: unknown;

  for (let i = 0; i <= maxRetries; i++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;

      if (i === maxRetries) {
        break;
      }

      console.warn(
        `Operation failed, retrying in ${delay}ms... (${i + 1}/${maxRetries})`
      );
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw new Error(
    `Operation failed after ${maxRetries} retries: ${normalizeError(lastError)}`
  );
}

// =============================================================================


/**
 * Deep clone object
 * @param obj object to clone
 * @returns cloned object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== "object") {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime()) as unknown as T;
  }

  if (obj instanceof Array) {
    return obj.map((item) => deepClone(item)) as unknown as T;
  }

  if (typeof obj === "object") {
    const cloned = {} as T;
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        cloned[key] = deepClone(obj[key]);
      }
    }
    return cloned;
  }

  return obj;
}

/**
 * Debounce function
 * @param func function to debounce
 * @param wait wait time (milliseconds)
 * @returns debounced function
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return function (this: any, ...args: Parameters<T>) {
    if (timeout) {
      clearTimeout(timeout);
    }

    timeout = setTimeout(() => {
      func.apply(this, args);
    }, wait);
  };
}

/**
 * Throttle function
 * @param func function to throttle
 * @param wait throttle interval (milliseconds)
 * @returns throttled function
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let lastTime = 0;

  return function (this: any, ...args: Parameters<T>) {
    const now = Date.now();

    if (now - lastTime >= wait) {
      lastTime = now;
      func.apply(this, args);
    }
  };
}

// =============================================================================

/**
 * Check if value is non-null
 * @param value value to check
 * @returns type guard result
 */
export function isNonNull<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

/**
 * Check if value is a valid number
 * @param value value to check
 * @returns type guard result
 */
export function isValidNumber(value: unknown): value is number {
  return typeof value === "number" && !isNaN(value) && isFinite(value);
}

/**
 * Check if value is a valid ArrayBuffer
 * @param value value to check
 * @returns type guard result
 */
export function isValidArrayBuffer(value: unknown): value is ArrayBuffer {
  return value instanceof ArrayBuffer && value.byteLength > 0;
}
