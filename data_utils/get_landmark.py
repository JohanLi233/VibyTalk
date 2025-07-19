#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landmark detection with optional Apple Silicon (MPS) acceleration.
"""

import math
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from detect_face import SCRFD
from pfld_mobileone import PFLD_GhostOne as PFLDInference


# ───────────────────────────────────────── face detection helper ─────────────────────────────────────────
def face_det(img, model):
    """
    Detect first face in `img` using SCRFD and return cropped image + geometry info.
    """
    cropped_imgs, boxes_list, center_list, alpha_list = [], [], [], []
    height, width = img.shape[:2]

    # SCRFD forward pass
    bboxes, indices, kps = model.detect(img)

    for i in indices:
        x1, y1, x2, y2 = (
            int(bboxes[i, 0]),
            int(bboxes[i, 1]),
            int(bboxes[i, 0] + bboxes[i, 2]),
            int(bboxes[i, 1] + bboxes[i, 3]),
        )

        p1, p2 = kps[i, 0], kps[i, 1]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x2 + x1) // 2, (y2 + y1) // 2
        boxsize = int(max(w, h) * 1.05)

        # square crop box
        size = boxsize
        xy = np.asarray((cx - size // 2, cy - size // 2), dtype=np.int32)
        x1, y1 = xy
        x2, y2 = xy + size

        # padding if crop goes outside
        dx, dy = max(0, -x1), max(0, -y1)
        x1, y1 = max(0, x1), max(0, y1)
        edx, edy = max(0, x2 - width), max(0, y2 - height)
        x2, y2 = min(width, x2), min(height, y2)

        cropped = img[y1:y2, x1:x2]
        if dx or dy or edx or edy:
            cropped = cv2.copyMakeBorder(
                cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            y1, x1 = y1 - dy, x1 - dx

        center = (int((x2 - x1) // 2), int((y2 - y1) // 2))
        alpha = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / math.pi

        cropped_imgs.append(cropped)
        boxes_list.append([x1, y1, x2, y2])
        center_list.append(center)
        alpha_list.append(alpha)
        break

    return cropped_imgs, boxes_list, center_list, alpha_list


def batch_face_det(imgs, model):
    all_cropped_imgs = []
    all_boxes_list = []
    all_center_list = []
    all_alpha_list = []

    for img in imgs:
        cropped_imgs, boxes_list, center_list, alpha_list = face_det(img, model)
        all_cropped_imgs.append(cropped_imgs)
        all_boxes_list.append(boxes_list)
        all_center_list.append(center_list)
        all_alpha_list.append(alpha_list)

    return all_cropped_imgs, all_boxes_list, all_center_list, all_alpha_list


# ────────────────────────────────────── main landmark class ─────────────────────────────────────────────
class Landmark:
    """
    Wrapper around SCRFD (face detection) + PFLD (landmark regression).
    Automatically uses Apple Silicon MPS if可用.
    """

    def __init__(
        self,
        mean_face_path: str | None = None,
        onnx_model_path: str | None = None,
        checkpoint_path: str | None = None,
        batch_size: int = 32,
    ):
        # ---------- 路径处理 ----------
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        mean_face_path = mean_face_path or os.path.join(cur_dir, "mean_face.txt")
        onnx_model_path = onnx_model_path or os.path.join(
            cur_dir, "scrfd_2.5g_kps.onnx"
        )
        checkpoint_path = checkpoint_path or os.path.join(
            cur_dir, "checkpoint_epoch_335.pth.tar"
        )

        self.device = (
            torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
        )
        self.batch_size = batch_size
        print(f"[INFO] Using device: {self.device}, batch_size: {batch_size}")

        with open(mean_face_path, "r") as f_mean_face:
            mean_face = f_mean_face.read()
        self.mean_face = np.asarray(mean_face.split(" "), dtype=np.float32)

        self.det_net = SCRFD(onnx_model_path, confThreshold=0.1, nmsThreshold=0.5)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.pfld_backbone = PFLDInference().to(self.device)
        self.pfld_backbone.load_state_dict(checkpoint["pfld_backbone"])
        self.pfld_backbone.eval()

    def detect(self, img_path: str):
        """
        Returns:
            pre_landmark (np.ndarray) – (98, 2) landmark coordinates in original image space
            x1, y1 – top-left corner of the face crop in original image
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img_ori = img.copy()

        cropped_imgs, boxes_list, _, _ = face_det(img_ori, self.det_net)
        if not cropped_imgs:
            raise RuntimeError("No face detected!")

        cropped = cropped_imgs[0]
        crop_h, crop_w = cropped.shape[:2]
        x1, y1, _, _ = boxes_list[0]

        input_tensor = cv2.resize(cropped, (192, 192))
        input_tensor = input_tensor.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)  # HWC → CHW
        input_tensor = torch.from_numpy(input_tensor)[None].to(self.device)

        with torch.no_grad():
            landmarks = self.pfld_backbone(input_tensor)

        pre_landmark = landmarks[0].cpu().numpy() + self.mean_face
        pre_landmark = pre_landmark.reshape(-1, 2)
        pre_landmark[:, 0] *= crop_w
        pre_landmark[:, 1] *= crop_h
        pre_landmark = pre_landmark.astype(np.int32)

        return pre_landmark, x1, y1

    def batch_detect(self, img_paths: list):
        results = []
        total_batches = (len(img_paths) + self.batch_size - 1) // self.batch_size

        pbar = tqdm(total=len(img_paths), desc="progress", unit="张")

        for batch_start in range(0, len(img_paths), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(img_paths))
            batch_paths = img_paths[batch_start:batch_end]

            batch_imgs = []
            valid_indices = []

            for i, img_path in enumerate(batch_paths):
                img = cv2.imread(img_path)
                if img is not None:
                    batch_imgs.append(img)
                    valid_indices.append(i)
                else:
                    tqdm.write(f"[WARNING] Cannot read image: {img_path}")

            if not batch_imgs:
                results.extend([None] * len(batch_paths))
                pbar.update(len(batch_paths))
                continue

            all_cropped_imgs, all_boxes_list, _, _ = batch_face_det(
                batch_imgs, self.det_net
            )

            batch_tensors = []
            batch_metadata = []

            for img_idx, (cropped_imgs, boxes_list) in enumerate(
                zip(all_cropped_imgs, all_boxes_list)
            ):
                if cropped_imgs and boxes_list:
                    cropped = cropped_imgs[0]
                    crop_h, crop_w = cropped.shape[:2]
                    x1, y1, _, _ = boxes_list[0]

                    input_tensor = cv2.resize(cropped, (192, 192))
                    input_tensor = input_tensor.astype(np.float32) / 255.0
                    input_tensor = input_tensor.transpose(2, 0, 1)  # HWC → CHW

                    batch_tensors.append(input_tensor)
                    batch_metadata.append(
                        (crop_w, crop_h, x1, y1, valid_indices[img_idx])
                    )
                else:
                    batch_metadata.append(None)

            if batch_tensors:
                batch_input = torch.from_numpy(np.stack(batch_tensors)).to(self.device)

                with torch.no_grad():
                    batch_landmarks = self.pfld_backbone(batch_input)

                landmark_idx = 0
                batch_results = [None] * len(batch_paths)

                for meta in batch_metadata:
                    if meta is not None:
                        crop_w, crop_h, x1, y1, orig_idx = meta
                        landmarks = (
                            batch_landmarks[landmark_idx].cpu().numpy() + self.mean_face
                        )
                        pre_landmark = landmarks.reshape(-1, 2)
                        pre_landmark[:, 0] *= crop_w
                        pre_landmark[:, 1] *= crop_h
                        pre_landmark = pre_landmark.astype(np.int32)

                        batch_results[orig_idx] = (pre_landmark, x1, y1)
                        landmark_idx += 1

                results.extend(batch_results)
            else:
                results.extend([None] * len(batch_paths))

            pbar.update(len(batch_paths))

            success_count = sum(1 for r in results if r is not None)
            pbar.set_postfix(
                {
                    "成功": f"{success_count}/{len(results)}",
                    "批次": f"{(batch_start // self.batch_size) + 1}/{total_batches}",
                }
            )

        pbar.close()
        return results

    def batch_detect_from_dir(self, img_dir: str, extensions=(".jpg", ".jpeg", ".png")):
        img_paths = []
        img_names = []

        for filename in sorted(os.listdir(img_dir)):
            if filename.lower().endswith(extensions):
                img_paths.append(os.path.join(img_dir, filename))
                img_names.append(filename)

        print(f"[INFO] Found {len(img_paths)} images in {img_dir}")

        if not img_paths:
            print("[WARNING] No valid images found!")
            return {}

        results = self.batch_detect(img_paths)

        result_dict = {}
        for name, result in zip(img_names, results):
            result_dict[name] = result

        return result_dict
