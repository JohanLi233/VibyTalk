import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
import PIL.Image
from torchvision import transforms
from torchvision.transforms import functional as F

from torch.utils.data import Dataset
from data_utils.face_utils import get_face_mask_poly_fixed
from config import get_config, get_border_from_crop_size


def _pass_through_transform(x):
    return x


class MultiPersonDataset(Dataset):
    def __init__(
        self,
        data_root,
        network_variant="nano",
        identity_swap_probability=0.8,
        enable_data_augmentation=True,
    ):
        self.network_config = get_config(network_variant)
        self.identity_swap_probability = identity_swap_probability

        if enable_data_augmentation:
            print("Data augmentation enabled")
            self.color_transform = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
            self.horizontal_flip_probability = 0.5
        else:
            print("Data augmentation disabled")
            self.color_transform = transforms.Lambda(_pass_through_transform)
            self.horizontal_flip_probability = 0.0

        self.tensor_converter = transforms.ToTensor()

        image_directory = os.path.join(data_root, "full_body_img")
        keypoints_directory = os.path.join(data_root, "landmarks")
        self.frame_metadata = {}
        valid_frame_list = []

        print(f"Scanning frames in {data_root}...")
        image_files = sorted(
            [f for f in os.listdir(image_directory) if f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

        for image_file in tqdm(image_files, desc="Validating data integrity"):
            filename_base = os.path.splitext(image_file)[0]
            try:
                frame_number = int(filename_base)
                keypoint_file = os.path.join(keypoints_directory, f"{filename_base}.lms")
                if os.path.exists(keypoint_file) and os.path.getsize(keypoint_file) > 0:
                    self.frame_metadata[frame_number] = {"lms_path": keypoint_file}
                    valid_frame_list.append(frame_number)
            except ValueError:
                continue

        self.person_groups = []
        self.training_samples = []
        self.frame_person_mapping = {}
        identity_mapping_file = os.path.join(data_root, "identity_map.txt")

        if os.path.exists(identity_mapping_file):
            print("Found identity mapping file, loading multi-person data.")
            with open(identity_mapping_file, "r") as f:
                person_data = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        person_name, start_frame, end_frame = parts
                        person_data.append(
                            {
                                "name": person_name,
                                "frame_range": range(int(start_frame), int(end_frame) + 1),
                            }
                        )
            if not person_data:
                raise ValueError("Identity mapping file is empty or malformed.")

            for person_info in person_data:
                person_valid_frames = [
                    f for f in person_info["frame_range"] if f in self.frame_metadata
                ]
                if person_valid_frames:
                    current_person_idx = len(self.person_groups)
                    self.person_groups.append(
                        {**person_info, "valid_frames": person_valid_frames}
                    )
                    self.training_samples.extend(person_valid_frames)
                    for frame_num in person_valid_frames:
                        self.frame_person_mapping[frame_num] = current_person_idx

            print(f"Successfully loaded identity mapping with {len(self.person_groups)} valid persons.")
        else:
            print("No identity mapping found, treating all frames as single person.")
            self.training_samples = valid_frame_list
            if valid_frame_list:
                self.person_groups.append(
                    {
                        "name": "default_person",
                        "valid_frames": valid_frame_list,
                    }
                )
                for frame_num in valid_frame_list:
                    self.frame_person_mapping[frame_num] = 0

        audio_features_path = os.path.join(
            data_root, f"aud_mel_{self.network_config.spectral_bins}.npy"
        )
        if not os.path.exists(audio_features_path):
            raise FileNotFoundError(f"Audio features file not found: {audio_features_path}")
        self.audio_features = np.load(audio_features_path).astype(np.float32)
        self.landmarks_cache = {}
        self.image_root_directory = image_directory
        print(f"Scanning complete! Found {len(self.training_samples)} valid frames for training.")

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        try:
            motion_frame_number = self.training_samples[idx]
            motion_person_idx = self.frame_person_mapping[motion_frame_number]
            motion_image_path = os.path.join(
                self.image_root_directory, f"{motion_frame_number}.jpg"
            )
            motion_landmarks_path = self.frame_metadata[motion_frame_number]["lms_path"]

            if random.random() < self.identity_swap_probability and len(self.person_groups) > 1:
                candidate_person_indices = [
                    i
                    for i, person in enumerate(self.person_groups)
                    if i != motion_person_idx and person["valid_frames"]
                ]
                if not candidate_person_indices:
                    reference_person_idx = motion_person_idx
                else:
                    reference_person_idx = random.choice(candidate_person_indices)
            else:
                reference_person_idx = motion_person_idx

            reference_frame_number = random.choice(
                self.person_groups[reference_person_idx]["valid_frames"]
            )
            reference_image_path = os.path.join(self.image_root_directory, f"{reference_frame_number}.jpg")
            reference_landmarks_path = self.frame_metadata[reference_frame_number]["lms_path"]

            motion_image = cv2.imread(motion_image_path)
            reference_image = cv2.imread(reference_image_path)
            if motion_image is None or reference_image is None:
                raise RuntimeError(
                    f"Failed to load images: {motion_image_path} or {reference_image_path}"
                )

            motion_landmarks = self._load_landmarks(motion_landmarks_path)
            reference_landmarks = self._load_landmarks(reference_landmarks_path)
            if motion_landmarks is None or reference_landmarks is None:
                raise RuntimeError("Failed to load landmarks")

            concatenated_input, ground_truth_output = self.process_image_pair(
                motion_image, motion_landmarks, reference_image, reference_landmarks
            )

            audio_features = self.extract_audio_window(self.audio_features, motion_frame_number)

            return (
                concatenated_input.contiguous(),
                ground_truth_output.contiguous(),
                audio_features.contiguous(),
            )

        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _load_landmarks(self, landmarks_path):
        if landmarks_path in self.landmarks_cache:
            return self.landmarks_cache[landmarks_path]
        if not os.path.exists(landmarks_path):
            return None
        try:
            with open(landmarks_path, "r") as f:
                lines = f.read().splitlines()
            if len(lines) < 68:
                return None
            landmarks = np.array(
                [np.array(line.split(" "), dtype=np.float32) for line in lines],
                dtype=np.int32,
            )
            self.landmarks_cache[landmarks_path] = landmarks
            return landmarks
        except (ValueError, IndexError):
            return None

    def _extract_face_region(self, image, landmarks):
        if len(landmarks) < 68:
            raise RuntimeError(f"Insufficient landmarks: {len(landmarks)}, expected 68")
        left_boundary = landmarks[1][0]
        top_boundary = landmarks[52][1]
        right_boundary = landmarks[31][0]
        face_width = right_boundary - left_boundary
        bottom_boundary = top_boundary + face_width
        if face_width <= 0 or bottom_boundary <= top_boundary:
            raise RuntimeError("Invalid face bounding box")
        image_height, image_width = image.shape[:2]
        left_boundary = max(0, min(left_boundary, image_width - 1))
        top_boundary = max(0, min(top_boundary, image_height - 1))
        right_boundary = max(left_boundary + 1, min(right_boundary, image_width))
        bottom_boundary = max(top_boundary + 1, min(bottom_boundary, image_height))
        cropped_face = image[top_boundary:bottom_boundary, left_boundary:right_boundary]
        if cropped_face.size == 0:
            raise RuntimeError("Cropped image is empty")
        target_size = self.network_config.input_resolution
        return cv2.resize(
            cropped_face, (target_size, target_size), interpolation=cv2.INTER_AREA
        )

    def _generate_mask(self, inner_size: int, transformed_landmarks: np.ndarray) -> np.ndarray:
        if random.random() < self.network_config.PROBABILISTIC_MASKING:
            mask = np.zeros((inner_size, inner_size), dtype=np.uint8)
            min_scale, max_scale = self.network_config.MASK_DIMENSION_RANGE
            scale = random.uniform(min_scale, max_scale)

            mask_height = int(inner_size * scale)
            mask_width = int(inner_size * scale)

            x_start = random.randint(0, inner_size - mask_width)
            y_start = random.randint(0, inner_size - mask_height)
            x_end = x_start + mask_width
            y_end = y_start + mask_height

            mask[y_start:y_end, x_start:x_end] = 255
            return mask

        return get_face_mask_poly_fixed((inner_size, inner_size), transformed_landmarks)

    def process_image_pair(self, motion_img, motion_lms, reference_img, reference_lms):
        motion_cropped = self._extract_face_region(motion_img, motion_lms)
        reference_cropped = self._extract_face_region(reference_img, reference_lms)
        crop_size = self.network_config.input_resolution
        border = get_border_from_crop_size(crop_size)
        inner_size = crop_size - 2 * border
        motion_inner = motion_cropped[
            border : border + inner_size, border : border + inner_size
        ]
        reference_inner = reference_cropped[
            border : border + inner_size, border : border + inner_size
        ]
        motion_pil = PIL.Image.fromarray(cv2.cvtColor(motion_inner, cv2.COLOR_BGR2RGB))
        reference_pil = PIL.Image.fromarray(cv2.cvtColor(reference_inner, cv2.COLOR_BGR2RGB))
        apply_horizontal_flip = random.random() < self.horizontal_flip_probability
        reference_augmented = self.color_transform(reference_pil)
        if random.random() < self.horizontal_flip_probability:
            reference_augmented = F.hflip(reference_augmented)
        reference_tensor = self.tensor_converter(reference_augmented)
        motion_geometric_aug = motion_pil
        if apply_horizontal_flip:
            motion_geometric_aug = F.hflip(motion_geometric_aug)
        motion_target_tensor = self.tensor_converter(motion_geometric_aug)
        motion_color_augmented = self.color_transform(motion_geometric_aug)

        landmarks_transformed = motion_lms.copy().astype(np.float32)
        x_min, y_min, x_max = motion_lms[1][0], motion_lms[52][1], motion_lms[31][0]
        face_width = x_max - x_min
        if face_width > 0:
            scaling_factor = crop_size / face_width
            landmarks_transformed[:, 0] = (landmarks_transformed[:, 0] - x_min) * scaling_factor - border
            landmarks_transformed[:, 1] = (landmarks_transformed[:, 1] - y_min) * scaling_factor - border
        if apply_horizontal_flip:
            landmarks_transformed[:, 0] = inner_size - 1 - landmarks_transformed[:, 0]

        face_mask_array = self._generate_mask(inner_size, landmarks_transformed)
        if apply_horizontal_flip:
            face_mask_array = cv2.flip(face_mask_array, 1)

        motion_input_tensor = self.tensor_converter(motion_color_augmented)
        mask_tensor = self.tensor_converter(PIL.Image.fromarray(face_mask_array))
        masked_input_tensor = motion_input_tensor * (1.0 - mask_tensor)
        concatenated_input = torch.cat([reference_tensor, masked_input_tensor], dim=0)
        return concatenated_input, motion_target_tensor

    def extract_audio_window(self, audio_features, frame_index):
        max_frame_index = audio_features.shape[0] - 1
        if frame_index >= audio_features.shape[0]:
            frame_index = max_frame_index
        half_window = self.network_config.temporal_window_length // 2
        start_idx = frame_index - half_window
        end_idx = frame_index + half_window
        left_padding = max(0, -start_idx)
        right_padding = max(0, end_idx - audio_features.shape[0])
        start_idx = max(0, start_idx)
        end_idx = min(audio_features.shape[0], end_idx)
        audio_window = audio_features[start_idx:end_idx]
        if left_padding > 0:
            left_repeat = np.repeat(audio_window[0:1], left_padding, axis=0)
            audio_window = np.concatenate([left_repeat, audio_window], axis=0)
        if right_padding > 0:
            right_repeat = np.repeat(audio_window[-1:], right_padding, axis=0)
            audio_window = np.concatenate([audio_window, right_repeat], axis=0)
        target_length = self.network_config.temporal_window_length
        if audio_window.shape[0] != target_length:
            if audio_window.shape[0] < target_length:
                additional_padding = target_length - audio_window.shape[0]
                padding_frames = np.repeat(audio_window[-1:], additional_padding, axis=0)
                audio_window = np.concatenate([audio_window, padding_frames], axis=0)
            else:
                audio_window = audio_window[:target_length]
        return torch.from_numpy(np.ascontiguousarray(audio_window.astype(np.float32)))