import argparse
import os
import cv2
import numpy as np
import json
import zipfile
import tempfile
from PIL import Image

from config import get_config, list_available_models, get_border_from_crop_size


def extract_complete_dataset_data():
    command_parser = argparse.ArgumentParser(
        description="Extract complete image sequence from dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    command_parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Directory containing source images and landmarks (e.g. 'lrs3_hdtf/train/5535415893579437641')",
    )
    command_parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=list_available_models(),
        help="Model size to use",
    )
    command_parser.add_argument(
        "--output_path",
        type=str,
        default="web/public/complete_dataset.json",
        help="Output JSON file path",
    )
    command_parser.add_argument(
        "--images_zip_path",
        type=str,
        default="web/public/processed_images.zip",
        help="Output processed images zip archive path",
    )
    command_parser.add_argument(
        "--webp_quality",
        type=int,
        default=45,
        choices=range(0, 101),
        metavar="[0-100]",
        help="WebP image compression quality (0-100, lower values mean higher compression)",
    )

    command_args = command_parser.parse_args()

    network_configuration = get_config(command_args.model_size)
    print(f"Using model configuration: {command_args.model_size}")
    print(f"Using WebP compression quality: {command_args.webp_quality}")

    print("Generating blending mask...")
    json_output_directory = os.path.dirname(command_args.output_path)
    if json_output_directory and not os.path.exists(json_output_directory):
        os.makedirs(json_output_directory)

    border_width = get_border_from_crop_size(network_configuration.input_resolution)
    effective_size = network_configuration.input_resolution - 2 * border_width
    kernel_dimension = 2 * border_width + 1

    blending_mask = np.zeros((network_configuration.input_resolution, network_configuration.input_resolution), dtype=np.float32)
    blending_mask[border_width : border_width + effective_size, border_width : border_width + effective_size] = 1.0
    blending_mask = cv2.GaussianBlur(blending_mask, (kernel_dimension, kernel_dimension), 0)

    mask_image_data = (blending_mask * 255).astype(np.uint8)
    mask_output_path = os.path.join(json_output_directory, "blending_mask.png")
    cv2.imwrite(mask_output_path, mask_image_data)
    print(f"Blending mask saved to: {mask_output_path}")

    print(f"Processing dataset: {command_args.dataset}")

    images_directory = os.path.join(command_args.dataset, "full_body_img/")
    landmarks_directory = os.path.join(command_args.dataset, "landmarks/")
    if not os.path.exists(images_directory) or not os.path.exists(landmarks_directory):
        raise FileNotFoundError(
            f"Dataset directory '{command_args.dataset}' is incomplete or missing 'full_body_img' or 'landmarks' subdirectories."
        )

    image_file_list = [f for f in os.listdir(images_directory) if f.endswith(".jpg")]
    landmark_file_list = [f for f in os.listdir(landmarks_directory) if f.endswith(".lms")]

    image_indices = {int(f.split(".")[0]) for f in image_file_list}
    landmark_indices = {int(f.split(".")[0]) for f in landmark_file_list}
    valid_frame_indices = sorted(image_indices & landmark_indices)

    if not valid_frame_indices:
        raise ValueError(
            "Could not find any matching image/LMS pairs from dataset. Please check file naming and paths."
        )

    print(f"Found {len(valid_frame_indices)} valid image/landmarks file pairs, will process all")

    with tempfile.TemporaryDirectory() as temporary_directory:
        processed_frame_data = []
        saved_file_paths = []
        source_dimensions = None

        for iteration, frame_index in enumerate(valid_frame_indices):
            image_file_path = os.path.join(images_directory, f"{frame_index}.jpg")
            landmarks_file_path = os.path.join(landmarks_directory, f"{frame_index}.lms")

            if (iteration + 1) % 50 == 0:
                print(f"Processing progress: {iteration + 1}/{len(valid_frame_indices)}")

            try:
                source_image = cv2.imread(image_file_path)
                if source_image is None:
                    print(f"Warning: Skipping image {frame_index}.jpg, cannot read")
                    continue

                if source_dimensions is None:
                    height, width = source_image.shape[:2]
                    source_dimensions = {"width": width, "height": height}

                try:
                    facial_landmarks = np.loadtxt(landmarks_file_path, dtype=np.int32)
                except ValueError:
                    with open(landmarks_file_path, "r") as file_handle:
                        text_lines = file_handle.read().splitlines()
                        facial_landmarks = np.array(
                            [
                                np.fromstring(line, dtype=np.int32, sep=" ")
                                for line in text_lines
                            ]
                        )

                left_bound, top_bound = facial_landmarks[1][0], facial_landmarks[52][1]
                right_bound = facial_landmarks[31][0]
                face_width = right_bound - left_bound
                bottom_bound = top_bound + face_width

                cropped_region = source_image[top_bound:bottom_bound, left_bound:right_bound]
                crop_height, crop_width = cropped_region.shape[:2]
                if crop_height == 0 or crop_width == 0:
                    print(f"Warning: Skipping image {frame_index}.jpg, invalid crop dimensions")
                    continue

                resized_crop = cv2.resize(
                    cropped_region,
                    (network_configuration.input_resolution, network_configuration.input_resolution),
                    interpolation=cv2.INTER_AREA,
                )

                crop_dimension = network_configuration.input_resolution
                border_offset = get_border_from_crop_size(crop_dimension)
                inner_dimension = crop_dimension - 2 * border_offset

                # Convert resized crop to RGB for proper processing
                rgb_face = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                face_inner_region_rgb = rgb_face[
                    border_offset : border_offset + inner_dimension, border_offset : border_offset + inner_dimension
                ].copy()

                # Transform landmarks to match the resized face coordinates
                transformed_landmarks = facial_landmarks.copy().astype(np.float32)
                if face_width > 0:
                    scale_factor = network_configuration.input_resolution / float(face_width)
                    transformed_landmarks[:, 0] = (transformed_landmarks[:, 0] - left_bound) * scale_factor - border_offset
                    transformed_landmarks[:, 1] = (transformed_landmarks[:, 1] - top_bound) * scale_factor - border_offset

                # Import face mask utility
                from data_utils.face_utils import get_face_mask_poly_fixed
                
                # Generate proper face polygon mask
                face_polygon_mask = get_face_mask_poly_fixed(
                    (inner_dimension, inner_dimension), transformed_landmarks
                )
                masked_face_region_rgb = face_inner_region_rgb.copy()
                masked_face_region_rgb[face_polygon_mask == 255] = 0

                real_normalized = face_inner_region_rgb.astype(np.float32) / 255.0
                masked_normalized = masked_face_region_rgb.astype(np.float32) / 255.0
                real_channels_first = np.transpose(real_normalized, (2, 0, 1))
                masked_channels_first = np.transpose(masked_normalized, (2, 0, 1))
                concatenated_tensor = np.concatenate((real_channels_first, masked_channels_first), axis=0)

                full_image_webp_path = os.path.join(temporary_directory, f"{frame_index}_full.webp")
                face_image_webp_path = os.path.join(temporary_directory, f"{frame_index}_face.webp")
                tensor_binary_path = os.path.join(temporary_directory, f"{frame_index}_tensor.bin")

                full_image_pil = Image.fromarray(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
                full_image_pil.save(
                    full_image_webp_path, format="WEBP", quality=command_args.webp_quality, method=6
                )

                face_image_pil = Image.fromarray(
                    cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                )
                face_image_pil.save(
                    face_image_webp_path, format="WEBP", quality=command_args.webp_quality, method=6
                )

                concatenated_tensor.tofile(tensor_binary_path)

                saved_file_paths.extend([full_image_webp_path, face_image_webp_path, tensor_binary_path])

                processed_frame_data.append(
                    {
                        "frame_id": str(frame_index),
                        "full_image": f"{frame_index}_full.webp",
                        "face_image": f"{frame_index}_face.webp",
                        "tensor_file": f"{frame_index}_tensor.bin",
                        "crop_info": {
                            "xmin": int(left_bound),
                            "ymin": int(top_bound),
                            "xmax": int(right_bound),
                            "ymax": int(bottom_bound),
                            "width": int(face_width),
                        },
                    }
                )

            except Exception as processing_error:
                print(f"Warning: Error processing image {frame_index}.jpg: {processing_error}")
                continue

        print(f"Successfully processed {len(processed_frame_data)} images")

        print(f"Creating image archive: {command_args.images_zip_path}")

        zip_output_directory = os.path.dirname(command_args.images_zip_path)
        if zip_output_directory and not os.path.exists(zip_output_directory):
            os.makedirs(zip_output_directory)

        with zipfile.ZipFile(command_args.images_zip_path, "w", zipfile.ZIP_DEFLATED) as zip_archive:
            for file_path in saved_file_paths:
                archive_name = os.path.basename(file_path)
                zip_archive.write(file_path, archive_name)

        complete_dataset_info = {
            "dataset_info": {
                "source_image_dimensions": source_dimensions,
                "config": {
                    "crop_size": network_configuration.input_resolution,
                    "mask_region": [border_width, border_width, border_width + effective_size, border_width + effective_size],
                },
            },
            "images": processed_frame_data,
        }

        print(f"Saving complete data to: {command_args.output_path}")

        json_output_directory = os.path.dirname(command_args.output_path)
        if json_output_directory and not os.path.exists(json_output_directory):
            os.makedirs(json_output_directory)

        with open(command_args.output_path, "w") as json_file:
            json.dump(complete_dataset_info, json_file, indent=2)

    print("\n--- ✅ Complete dataset extraction finished ---")
    print(f"Dataset path: {command_args.dataset}")
    print(f"Processed frames: {len(processed_frame_data)}")
    print(f"JSON data file: {command_args.output_path}")
    print(f"JSON file size: {os.path.getsize(command_args.output_path) / (1024 * 1024):.2f} MB")
    print(f"Image archive: {command_args.images_zip_path}")
    print(f"Archive size: {os.path.getsize(command_args.images_zip_path) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    extract_complete_dataset_data()