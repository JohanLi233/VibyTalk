import argparse
import os
import cv2
import torch
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import time
import onnxruntime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sounddevice as sd

from config import get_config, list_available_models, get_border_from_crop_size
from data_utils.mel_extractor import get_mel_from_16k_speech
from data_utils.face_utils import get_face_mask_poly_fixed


def main():
    command_parser = argparse.ArgumentParser(
        description="Real-time lip sync generation and playback (ONNX only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    command_parser.add_argument(
        "--dataset", required=True, type=str, help="Directory containing source images and landmarks"
    )
    command_parser.add_argument(
        "--wav_path", required=True, type=str, help="Input WAV audio file path"
    )
    command_parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=list_available_models(),
        help="Model size to use",
    )
    command_parser.add_argument(
        "--onnx_model",
        type=str,
        default="model.onnx",
        help="ONNX model file path",
    )

    command_args = command_parser.parse_args()

    network_config = get_config(command_args.model_size)
    print(f"Using model configuration: {command_args.model_size}")

    if not os.path.exists(command_args.onnx_model):
        raise FileNotFoundError(f"ONNX model not found: {command_args.onnx_model}")
    inference_session = onnxruntime.InferenceSession(
        command_args.onnx_model, providers=["CPUExecutionProvider"]
    )
    print(f"Loaded ONNX model: {command_args.onnx_model}")

    print(f"Processing audio file: {command_args.wav_path}")
    audio_data, sample_rate = sf.read(command_args.wav_path)
    if sample_rate != 16000:
        audio_data = librosa.resample(
            audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=16000
        )

    audio_data = audio_data.astype(np.float32)

    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    if np.max(np.abs(audio_data)) < 0.1:
        audio_data = audio_data * (0.3 / np.max(np.abs(audio_data)))

    print(f"Audio preprocessing complete. Range: [{np.min(audio_data):.3f}, {np.max(audio_data):.3f}]")

    print(f"Using mel features, mel_bins={network_config.spectral_bins}")

    print("Extracting Mel spectrogram features...")
    audio_features = get_mel_from_16k_speech(audio_data, num_mel_bins=network_config.spectral_bins)
    audio_features = audio_features.numpy()
    print(f"Audio feature extraction complete. Shape: {audio_features.shape}")

    print("Preloading images and landmarks data...")
    images_directory = os.path.join(command_args.dataset, "full_body_img/")
    landmarks_directory = os.path.join(command_args.dataset, "landmarks/")
    if not os.path.exists(images_directory) or not os.path.exists(landmarks_directory):
        raise FileNotFoundError(f"Dataset directory '{command_args.dataset}' is incomplete or not found")

    image_file_list = [f for f in os.listdir(images_directory) if f.endswith(".jpg")]
    landmark_file_list = [f for f in os.listdir(landmarks_directory) if f.endswith(".lms")]

    image_indices = {int(f.split(".")[0]) for f in image_file_list}
    landmark_indices = {int(f.split(".")[0]) for f in landmark_file_list}
    valid_indices = sorted(image_indices & landmark_indices)

    if not valid_indices:
        raise ValueError(
            "Could not find any matching image/LMS pairs from dataset. Please check file naming and paths."
        )

    print(f"Found {len(valid_indices)} valid image/landmarks file pairs")

    def load_single_frame_data(frame_index):
        image_path = os.path.join(images_directory, f"{frame_index}.jpg")
        landmarks_path = os.path.join(landmarks_directory, f"{frame_index}.lms")

        source_image = cv2.imread(image_path)
        if source_image is None:
            return None

        try:
            landmarks_data = np.loadtxt(landmarks_path, dtype=np.int32)
        except ValueError:
            with open(landmarks_path, "r") as file_handle:
                text_lines = file_handle.read().splitlines()
                landmarks_data = np.array(
                    [np.fromstring(line, dtype=np.int32, sep=" ") for line in text_lines]
                )

        return frame_index, source_image, landmarks_data

    loaded_frame_data = []
    max_worker_count = min(8, os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=max_worker_count) as executor:
        future_tasks = {executor.submit(load_single_frame_data, idx) for idx in valid_indices}

        for future_task in tqdm(
            as_completed(future_tasks), total=len(valid_indices), desc="Parallel data loading"
        ):
            task_result = future_task.result()
            if task_result is not None:
                loaded_frame_data.append(task_result)

    loaded_frame_data.sort(key=lambda x: x[0])

    all_source_images = [item[1] for item in loaded_frame_data]
    all_facial_landmarks = [item[2] for item in loaded_frame_data]

    if not all_source_images:
        raise ValueError(
            "Could not load any valid image/LMS pairs from dataset. Please check file content and format."
        )

    print(f"Data preloading complete, successfully loaded {len(all_source_images)} data pairs.")

    input_resolution = network_config.input_resolution
    border_size = get_border_from_crop_size(input_resolution)
    effective_dimension = input_resolution - 2 * border_size

    print("Precomputing blending mask...")
    kernel_size = 2 * border_size + 1
    blending_mask = np.zeros((network_config.input_resolution, network_config.input_resolution, 1), dtype=np.float32)
    blending_mask[border_size : border_size + effective_dimension, border_size : border_size + effective_dimension] = 1.0
    blending_mask = cv2.GaussianBlur(blending_mask, (kernel_size, kernel_size), 0)[
        :, :, np.newaxis
    ]
    print("Blending mask computation complete.")

    print("\n--- Press 'q' key to exit playback ---")

    current_image_index = 0
    frame_direction = 1
    total_background_frames = len(all_source_images)

    def extract_audio_feature_window(features, frame_index):
        half_window_size = network_config.temporal_window_length // 2
        start_index, end_index = frame_index - half_window_size, frame_index + half_window_size
        left_padding, right_padding = 0, 0
        if start_index < 0:
            left_padding = -start_index
            start_index = 0
        if end_index > features.shape[0]:
            right_padding = end_index - features.shape[0]
            end_index = features.shape[0]
        audio_window = features[start_index:end_index]
        if left_padding > 0:
            audio_window = np.concatenate(
                [np.repeat(audio_window[0:1], left_padding, axis=0), audio_window], axis=0
            )
        if right_padding > 0:
            audio_window = np.concatenate(
                [audio_window, np.repeat(audio_window[-1:], right_padding, axis=0)], axis=0
            )
        return torch.from_numpy(audio_window.astype(np.float32))

    try:
        sd.default.blocksize = 4096
        sd.play(audio_data, samplerate=16000, blocking=False)
        print("Audio playback started...")
    except Exception as audio_error:
        print(f"Audio playback setup failed, using default settings: {audio_error}")
        sd.play(audio_data, 16000)

    playback_start_time = time.time()

    for frame_idx in tqdm(range(audio_features.shape[0]), desc="Real-time synthesis and playback"):
        current_image = all_source_images[current_image_index].copy()
        current_landmarks = all_facial_landmarks[current_image_index]

        if current_image_index >= total_background_frames - 1:
            frame_direction = -1
        elif current_image_index <= 0:
            frame_direction = 1

        current_image_index += frame_direction

        left_boundary, top_boundary = current_landmarks[1][0], current_landmarks[52][1]
        right_boundary = current_landmarks[31][0]
        face_width = right_boundary - left_boundary
        bottom_boundary = top_boundary + face_width

        cropped_face = current_image[top_boundary:bottom_boundary, left_boundary:right_boundary]
        crop_height, crop_width = cropped_face.shape[:2]
        if crop_height == 0 or crop_width == 0:
            continue

        resized_face = cv2.resize(
            cropped_face, (network_config.input_resolution, network_config.input_resolution), interpolation=cv2.INTER_AREA
        )
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        original_rgb_face = rgb_face.copy()

        face_inner_region = rgb_face[
            border_size : border_size + effective_dimension, border_size : border_size + effective_dimension
        ].copy()

        transformed_landmarks = current_landmarks.copy().astype(np.float32)
        if face_width > 0:
            scale_factor = network_config.input_resolution / float(face_width)
            transformed_landmarks[:, 0] = (transformed_landmarks[:, 0] - left_boundary) * scale_factor - border_size
            transformed_landmarks[:, 1] = (transformed_landmarks[:, 1] - top_boundary) * scale_factor - border_size

        face_polygon_mask = get_face_mask_poly_fixed(
            (effective_dimension, effective_dimension), transformed_landmarks
        )
        masked_face_region = face_inner_region.copy()
        masked_face_region[face_polygon_mask == 255] = 0

        masked_face_normalized = masked_face_region.transpose(2, 0, 1).astype(np.float32) / 255.0
        reference_face_normalized = face_inner_region.transpose(2, 0, 1).astype(np.float32) / 255.0
        concatenated_input_tensor = torch.cat(
            [torch.from_numpy(reference_face_normalized), torch.from_numpy(masked_face_normalized)], dim=0
        ).unsqueeze(0)

        audio_feature_window = extract_audio_feature_window(audio_features, frame_idx).unsqueeze(0)

        onnx_input_dict = {
            inference_session.get_inputs()[0].name: concatenated_input_tensor.numpy(),
            inference_session.get_inputs()[1].name: audio_feature_window.numpy(),
        }
        prediction_result = inference_session.run(None, onnx_input_dict)[0][0] # type: ignore

        prediction_image = (prediction_result.transpose(1, 2, 0) * 255).astype(np.uint8)

        composite_face = original_rgb_face.copy()
        composite_face[border_size : border_size + effective_dimension, border_size : border_size + effective_dimension] = prediction_image

        composite_face = (
            composite_face.astype(np.float32) * blending_mask
            + original_rgb_face.astype(np.float32) * (1 - blending_mask)
        ).astype(np.uint8)

        final_composite = cv2.cvtColor(composite_face, cv2.COLOR_RGB2BGR)
        final_output_frame = current_image.copy()
        final_output_frame[top_boundary:bottom_boundary, left_boundary:right_boundary] = cv2.resize(final_composite, (crop_width, crop_height))

        cv2.imshow("Real-time Talking Head", final_output_frame)

        elapsed_milliseconds = (time.time() - playback_start_time) * 1000
        target_frame_time = (frame_idx + 1) * 40
        wait_duration = max(1, int(target_frame_time - elapsed_milliseconds))
        if cv2.waitKey(wait_duration) & 0xFF == ord("q"):
            break

    try:
        sd.stop()
        sd.wait()
    except Exception as cleanup_error:
        print(f"Warning during audio cleanup: {cleanup_error}")

    cv2.destroyAllWindows()
    print("\nPlayback finished.")


if __name__ == "__main__":
    main()