import cv2
import numpy as np


def get_face_mask_poly_fixed(img_shape, landmarks):
    """
    Generate lower face mask based on 98-point model landmarks (fixed version).

    This function forms the upper boundary of the mask by selecting keypoints
    along the upper lip contour, avoiding the nose cutting issue from the original version.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Jaw line contour points (0-32)
    jaw_line_indices = list(range(0, 33))
    upper_line_indices = list(range(52, 68))

    # 3. Combine into a closed polygon point set
    polygon_indices = jaw_line_indices + upper_line_indices[::-1]

    if max(polygon_indices) >= len(landmarks):
        # If insufficient keypoints, return a black mask to avoid crashes
        print(
            f"[WARNING] Keypoint index out of range. Need {max(polygon_indices) + 1} keypoints, "
            f"but only {len(landmarks)} provided. Returning empty mask."
        )
        return mask

    points = landmarks[polygon_indices].astype(np.int32)

    # 4. Fill polygon
    cv2.fillPoly(mask, [points], (255, 255, 255))

    return mask 