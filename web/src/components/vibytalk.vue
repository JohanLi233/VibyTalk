<script setup lang="ts">
import { onMounted, ref, shallowRef, nextTick, onUnmounted, computed } from 'vue';
import {
    loadImageData,
    StreamingFeatureExtractorService,
    StreamingInferenceService,
    UnifiedPlaybackService,
    WebGLPlaybackService,
    globalImageProcessingService,
    type ImageDataResponse,
    type ChunkFeatureResult,
    calculatePingPongState,
} from '../services';
import JSZip from 'jszip';

let streamingInferenceService: StreamingInferenceService | null = null;
const streamingExtractorService = new StreamingFeatureExtractorService();
const playbackService = shallowRef<UnifiedPlaybackService | WebGLPlaybackService | null>(null);
const useWebGL = ref(true);

const isServiceReady = ref(false);
const inferenceRunning = ref(false);
const inferenceProgress = ref(0);
const inferenceTotal = ref(0);
const inferenceError = ref<string | null>(null);

const generatedFrames = computed(() => playbackService.value?.generatedFrames.value ?? []);

const mainThreadZip = shallowRef<JSZip | null>(null);
const zipFile = shallowRef<Blob | null>(null);
const isPreparingData = ref(true);
const dataset = shallowRef<ImageDataResponse | null>(null);
const blendingMaskBitmap = shallowRef<ImageBitmap | null>(null);
const silentModeFrames = shallowRef<ImageBitmap[]>([]);


const selectedAudioFile = ref<File | null>(null);
const isProcessingAudio = ref(false);
const audioErrorMessage = ref<string>('');
const audioUrl = ref<string>('');
const streamingProgress = ref({ completed: 0, total: 0 });
const firstFrameGenerated = ref(false);
const streamingStartTime = ref(0);

const uploadId = ref(0);

const isSilentModeActive = ref(false);
const silentModeFrameId = ref<number | null>(null);
const currentSourceImageIndex = ref(0);
const pingPongDirection = ref(1);
const lastSilentFrameTime = ref(0);
const SILENT_FRAME_INTERVAL = 40;
const isTransitioningToTalk = ref(false);
const transitionTargetFrameIndex = ref(0);
const TRANSITION_BUFFER_FRAMES = 5;
const audioPlayer = ref<HTMLAudioElement | null>(null);
const playbackCanvas = ref<HTMLCanvasElement | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const createAudioContext = async (): Promise<AudioContext> => {
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;

    if (!AudioContextClass) {
        throw new Error('Your browser does not support Web Audio API');
    }

    const audioContext = new AudioContextClass();

    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }

    return audioContext;
};

const getSilentModeFrame = async (frameIndex: number): Promise<ImageBitmap> => {
    if (!dataset.value || !mainThreadZip.value) {
        throw new Error('Image data is not ready');
    }

    const imageMeta = dataset.value.images[frameIndex];
    if (!imageMeta) {
        throw new Error(`Frame index ${frameIndex} out of range`);
    }

    const file = mainThreadZip.value.file(imageMeta.full_image);
    if (!file) {
        throw new Error(`Image file not found: ${imageMeta.full_image}`);
    }

    const blob = await file.async("blob");
    return await createImageBitmap(blob);
};


const silentLoop = (timestamp: number) => {
    if (!isSilentModeActive.value ||
        (inferenceRunning.value && !isTransitioningToTalk.value) ||
        (generatedFrames.value.length > 0)) {
        console.log('Silent loop running unexpectedly, stopping immediately:', {
            isSilentModeActive: isSilentModeActive.value,
            inferenceRunning: inferenceRunning.value,
            hasGeneratedFrames: generatedFrames.value.length > 0
        });
        stopSilentLoop();
        return;
    }

    if (isTransitioningToTalk.value) {
        if (currentSourceImageIndex.value >= transitionTargetFrameIndex.value) {
            console.log(`Silent mode transition complete, switching to playback service, target frame: ${transitionTargetFrameIndex.value}`);
            stopSilentLoop();
            playbackService.value?.start();
            isTransitioningToTalk.value = false;
            return;
        }
    }

    const elapsed = timestamp - lastSilentFrameTime.value;

    if (elapsed >= SILENT_FRAME_INTERVAL) {
        lastSilentFrameTime.value = timestamp;

        renderCurrentSilentFrame().catch(error => {
            console.error('Error rendering silent mode frame:', error);
        });

        if (dataset.value) {
            const { nextIndex, nextDirection } = calculatePingPongState(
                currentSourceImageIndex.value,
                dataset.value.images.length,
                pingPongDirection.value
            );
            currentSourceImageIndex.value = nextIndex;
            pingPongDirection.value = nextDirection;
        }
    }

    silentModeFrameId.value = requestAnimationFrame(silentLoop);
};

const renderCurrentSilentFrame = async (): Promise<void> => {
    if (!playbackCanvas.value || !dataset.value || !playbackService.value) return;

    try {
        const currentFrame = await getSilentModeFrame(currentSourceImageIndex.value);

        await playbackService.value.renderSilentFrame(currentFrame);
        currentFrame.close();

    } catch (error) {
        console.error('Error loading silent mode frame:', error);
    }
};

const resizeCanvas = () => {
    if (!dataset.value?.dataset_info?.source_image_dimensions || !playbackCanvas.value) {
        return;
    }

    const dims = dataset.value.dataset_info.source_image_dimensions;
    const maxWidth = window.innerWidth * 0.8;
    const maxHeight = window.innerHeight * 0.8;

    const scaleX = maxWidth / dims.width;
    const scaleY = maxHeight / dims.height;
    const scale = Math.min(scaleX, scaleY, 1);

    const displayWidth = dims.width * scale;
    const displayHeight = dims.height * scale;

    playbackCanvas.value.style.width = `${displayWidth}px`;
    playbackCanvas.value.style.height = `${displayHeight}px`;

    console.log(`Canvas size recalculated: ${displayWidth}x${displayHeight}, scale: ${scale}`);
};

let resizeTimeout: number | null = null;
const debouncedResize = () => {
    if (resizeTimeout) {
        clearTimeout(resizeTimeout);
    }
    resizeTimeout = window.setTimeout(() => {
        resizeCanvas();
        resizeTimeout = null;
    }, 100);
};

const startSilentLoop = () => {
    if (inferenceRunning.value ||
        (generatedFrames.value.length > 0) ||
        isSilentModeActive.value ||
        silentModeFrameId.value !== null) {
        console.log('Silent mode start blocked:', {
            inferenceRunning: inferenceRunning.value,
            hasGeneratedFrames: generatedFrames.value.length > 0,
            isSilentModeActive: isSilentModeActive.value,
            silentModeFrameId: silentModeFrameId.value
        });
        return;
    }
    console.log('Starting silent mode loop');
    isTransitioningToTalk.value = false;
    isSilentModeActive.value = true;
    lastSilentFrameTime.value = performance.now();
    silentModeFrameId.value = requestAnimationFrame(silentLoop);
};

const stopSilentLoop = () => {
    if (!isSilentModeActive.value) return;
    console.log('stop silent mode loop');
    isSilentModeActive.value = false;
    if (silentModeFrameId.value) {
        cancelAnimationFrame(silentModeFrameId.value);
        silentModeFrameId.value = null;
    }

};

const loadBlendingMask = async () => {
    try {
        const img = new Image();
        img.src = '/blending_mask.png';
        await img.decode();
        blendingMaskBitmap.value = await createImageBitmap(img);
    } catch (e) {
        console.error("Failed to load blending mask:", e);
        const errorMessage = "Failed to load blending mask 'blending_mask.png'. Please ensure you have run the latest extract_dataset_data.py script.";
        inferenceError.value = errorMessage;
        throw new Error(errorMessage);
    }
};

const resetForNewUpload = async () => {
    stopSilentLoop();

    console.log('Audio playback ended, resetting state for next upload.');

    const lastInferenceTotal = inferenceTotal.value;

    const resetId = uploadId.value;

    streamingInferenceService?.stop();
    streamingExtractorService.stop();

    playbackService.value?.cleanup();

    selectedAudioFile.value = null;
    if (audioUrl.value) {
        URL.revokeObjectURL(audioUrl.value);
        audioUrl.value = '';
    }
    audioErrorMessage.value = '';
    inferenceError.value = null;
    firstFrameGenerated.value = false;
    streamingProgress.value = { completed: 0, total: 0 };
    inferenceProgress.value = 0;
    inferenceTotal.value = 0;
    inferenceRunning.value = false;

    if (audioPlayer.value) {
        audioPlayer.value.src = '';
    }

    if (fileInput.value) {
        fileInput.value.value = '';
    }

    await nextTick();

    if (resetId !== uploadId.value) {
        console.log(`Task ID changed from ${resetId} to ${uploadId.value}, canceling outdated reset callback.`);
        return;
    }

    if (isSilentModeActive.value || silentModeFrameId.value !== null) {
        console.log('Silent mode still running, forcing stop');
        stopSilentLoop();
    }

    if (dataset.value && dataset.value.images.length > 0) {
        isTransitioningToTalk.value = false;

        const resumeFromIndex = transitionTargetFrameIndex.value + lastInferenceTotal;
        console.log(`Inference ended, resuming silent mode: transition start frame=${transitionTargetFrameIndex.value} + inference total frames=${lastInferenceTotal} = resume frame=${resumeFromIndex}`);

        const numImages = dataset.value.images.length;
        if (numImages > 1) {
            const cycleLen = (numImages - 1) * 2;
            const effectiveIndex = cycleLen > 0 ? resumeFromIndex % cycleLen : 0;

            if (effectiveIndex < numImages) {
                currentSourceImageIndex.value = effectiveIndex;
                pingPongDirection.value = 1;
            } else {
                currentSourceImageIndex.value = cycleLen - effectiveIndex;
                pingPongDirection.value = -1;
            }
        } else {
            currentSourceImageIndex.value = 0;
        }
    }

    inferenceRunning.value = false;

    startSilentLoop();
};

const prepareData = async () => {
    isPreparingData.value = true;
    try {
        const { imageData, zip, zipBlob } = await loadImageData();
        dataset.value = imageData;
        mainThreadZip.value = zip;
        zipFile.value = zipBlob;

        console.log(`Image data loaded, ${imageData.images.length} frames, using lazy loading mode to control memory usage`);

        if (silentModeFrames.value.length > 0) {
            silentModeFrames.value.forEach(frame => frame.close());
            silentModeFrames.value = [];
        }

        if (playbackService.value && mainThreadZip.value) {
            playbackService.value.setFrameData(imageData, mainThreadZip.value);
        }

        const dims = imageData.dataset_info?.source_image_dimensions;
        if (dims && playbackCanvas.value) {
            console.log(`Source image dimensions detected, setting Canvas size to: ${dims.width}x${dims.height}`);

            playbackCanvas.value.width = dims.width;
            playbackCanvas.value.height = dims.height;

            const maxWidth = window.innerWidth * 0.8;
            const maxHeight = window.innerHeight * 0.8;

            const scaleX = maxWidth / dims.width;
            const scaleY = maxHeight / dims.height;
            const scale = Math.min(scaleX, scaleY, 1);

            const displayWidth = dims.width * scale;
            const displayHeight = dims.height * scale;

            playbackCanvas.value.style.width = `${displayWidth}px`;
            playbackCanvas.value.style.height = `${displayHeight}px`;

            console.log(`Canvas internal size: ${dims.width}x${dims.height}, display size: ${displayWidth}x${displayHeight}, scale: ${scale}`);
        }

        startSilentLoop();

    } catch (e: any) {
        console.error("Data preparation error:", e);
        inferenceError.value = "无法加载推理所需的图像数据。请确保已运行最新的 `extract_dataset_data.py` 脚本，并检查 `public` 目录下的 `complete_dataset.json` 和 `processed_images.zip` 文件是否存在。";
    } finally {
        isPreparingData.value = false;
        if (selectedAudioFile.value && isServiceReady.value && !inferenceRunning.value) {
            nextTick().then(() => {
                handleAudioFileChange({ target: { files: [selectedAudioFile.value] } } as unknown as Event);
            });
        }
    }
};

onMounted(async () => {
    if (playbackCanvas.value && audioPlayer.value) {
        if (useWebGL.value && WebGLPlaybackService.isWebGLSupported()) {
            try {
                const webglService = new WebGLPlaybackService(
                    playbackCanvas.value,
                    audioPlayer.value,
                    resetForNewUpload
                );
                await webglService.initialize();
                playbackService.value = webglService;
                console.log('Using WebGL playback service');
            } catch (error) {
                console.warn('WebGL playback service initialization failed, falling back to Canvas 2D:', error);
                playbackService.value = new UnifiedPlaybackService(
                    playbackCanvas.value,
                    audioPlayer.value,
                    resetForNewUpload
                );
                useWebGL.value = false;
            }
        } else {
            playbackService.value = new UnifiedPlaybackService(
                playbackCanvas.value,
                audioPlayer.value,
                resetForNewUpload
            );
            console.log('Using Canvas 2D playback service');
        }
    }

    streamingInferenceService = new StreamingInferenceService('/model.onnx', () => {
        isServiceReady.value = true;
        console.log("Streaming inference service is ready.");
    });

    window.addEventListener('resize', debouncedResize);

    await loadBlendingMask();
    prepareData();
});

onUnmounted(() => {
    streamingInferenceService?.terminate();
    streamingExtractorService.stop();
    playbackService.value?.cleanup();
    silentModeFrames.value.forEach(frame => frame.close());
    silentModeFrames.value = [];
    globalImageProcessingService.cleanup();
    window.removeEventListener('resize', debouncedResize);
    if (resizeTimeout) {
        clearTimeout(resizeTimeout);
        resizeTimeout = null;
    }
});

const handleAudioFileChange = async (event: Event) => {
    const target = event.target as HTMLInputElement;
    if (!target.files || target.files.length === 0) return;

    uploadId.value++;
    console.log(`New task created, ID: ${uploadId.value}`);

    isTransitioningToTalk.value = true;
    transitionTargetFrameIndex.value = currentSourceImageIndex.value + TRANSITION_BUFFER_FRAMES;
    console.log(`Starting silent mode transition, current frame: ${currentSourceImageIndex.value}, target frame: ${transitionTargetFrameIndex.value}`);

    playbackService.value?.cleanup();
    streamingInferenceService?.stop();
    streamingExtractorService.stop();

    const file = target.files[0];
    selectedAudioFile.value = file;
    isProcessingAudio.value = true;
    audioErrorMessage.value = '';
    firstFrameGenerated.value = false;
    streamingProgress.value = { completed: 0, total: 0 };

    if (audioUrl.value) URL.revokeObjectURL(audioUrl.value);
    audioUrl.value = URL.createObjectURL(file);

    try {
        await loadBlendingMask();

        const audioContext = await createAudioContext();
        const arrayBuffer = await file.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        streamingStartTime.value = performance.now();

        const inferenceStartImageIndex = transitionTargetFrameIndex.value;
        await handleStreamingAudioProcessing(audioBuffer, inferenceStartImageIndex);

    } catch (e: any) {
        console.error('Audio processing error:', e);
        audioErrorMessage.value = e.message || 'Unknown error occurred while processing audio.';
    } finally {
        isProcessingAudio.value = false;
    }
};

const handleStreamingAudioProcessing = async (
    audioBuffer: AudioBuffer,
    inferenceStartImageIndex: number
) => {
    if (!streamingInferenceService || !streamingInferenceService.isReady()) {
        audioErrorMessage.value = 'Streaming inference service is not ready, please wait...';
        console.warn('Streaming inference service is not ready');
        return;
    }

    if (!dataset.value || !blendingMaskBitmap.value || !zipFile.value) {
        audioErrorMessage.value = 'Image data is not ready, please wait...';
        console.warn('Image data is not ready');
        return;
    }

    audioErrorMessage.value = '';
    inferenceError.value = null;

    const sharedData = {
        dataset: structuredClone(dataset.value),
        zipBlob: zipFile.value,
        blendingMaskBitmap: blendingMaskBitmap.value,
    };

    await streamingInferenceService.startStreaming(sharedData, {
        onProgress: (processed, total) => {
            inferenceProgress.value = processed;
            inferenceTotal.value = total;
        },
        onFrame: (frameBitmap) => {
            playbackService.value?.addFrame(frameBitmap);

            if (!firstFrameGenerated.value) {
                firstFrameGenerated.value = true;
            }
        },
        onChunkComplete: (chunkIndex, timings) => {
            console.log(`Chunk ${chunkIndex} complete, timing:`, timings);
        },
        onAllComplete: () => {
            playbackService.value?.setInferenceRunning(false);
        },
        onError: (errorMessage) => {
            console.error('error:', errorMessage);
            inferenceError.value = errorMessage;
            inferenceRunning.value = false;
            playbackService.value?.setInferenceRunning(false);
            resetForNewUpload().catch(error => {
                console.error('Error resetting state:', error);
            });
        },
    }, inferenceStartImageIndex);

    inferenceRunning.value = true;
    playbackService.value?.setInferenceRunning(true);
    inferenceProgress.value = 0;
    inferenceTotal.value = 0;

    await streamingExtractorService.processStreaming(audioBuffer, {
        onChunkComplete: (chunkResult: ChunkFeatureResult) => {
            streamingInferenceService?.addChunk(chunkResult);
        },
        onProgress: (completed, total) => {
            streamingProgress.value = { completed, total };
        },
        onComplete: (totalDimensions) => {
            inferenceTotal.value = totalDimensions[0];
            streamingInferenceService?.finishAddingChunks();
        },
        onError: (error) => {
            console.error('error:', error);
            inferenceError.value = error;
            inferenceRunning.value = false;
            playbackService.value?.setInferenceRunning(false);
            resetForNewUpload().catch(error => {
                console.error('error:', error);
            });
        },
    });
};
</script>

<template>
    <div class="onnx-test-container">
        <audio ref="audioPlayer" :src="audioUrl" style="display: none;" preload="auto" controls="false">
        </audio>

        <div class="main-workflow">
            <div class="workflow-step">
                <div class="step-header">
                    <span class="step-number">1</span>
                    <h4>Select Audio File</h4>
                </div>
                <div class="step-content">
                    <input type="file" ref="fileInput" @change="handleAudioFileChange"
                        accept="audio/*,.mp3,.wav,.ogg,.m4a,.aac,.flac,.wma" :disabled="inferenceRunning"
                        class="file-input" />
                    <div v-if="audioErrorMessage" class="error-message">{{ audioErrorMessage }}</div>
                </div>
            </div>
        </div>

        <div class="results-section-container">
            <div v-if="inferenceError" class="error-message">
                {{ inferenceError }}
            </div>

            <div v-if="isPreparingData || (selectedAudioFile && !isServiceReady && !isTransitioningToTalk)"
                class="status-info">
                <p v-if="isPreparingData">Loading...</p>
                <div class="spinner"></div>
            </div>

            <div v-show="!isPreparingData" class="playback-container">
                <div class="player">
                    <canvas ref="playbackCanvas" class="playback-image"></canvas>
                </div>
            </div>
        </div>
    </div>
</template>
