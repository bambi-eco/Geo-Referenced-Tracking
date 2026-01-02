"""
Geo-Referenced Tracking Utilities

Common utility functions for geo-referenced multi-object tracking.
"""

from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Geo-Space Coordinate Transformations
# =============================================================================


def xyxy2xysr_geo(bbox: np.ndarray) -> np.ndarray:
    """
    Convert [x1, y1, x2, y2] to [cx, cy, area, aspect_ratio] for geo coordinates.

    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format (geo coordinates)

    Returns:
        Array of [cx, cy, area, aspect_ratio] shaped (4, 1)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    area = w * h
    aspect_ratio = w / h if h > 1e-6 else 1.0
    return np.array([cx, cy, area, aspect_ratio]).reshape((4, 1))


def convert_x_to_bbox_geo(x: np.ndarray, score: Optional[float] = None) -> np.ndarray:
    """
    Convert Kalman state [cx, cy, area, aspect_ratio, ...] to [x1, y1, x2, y2].

    Args:
        x: Kalman filter state vector
        score: Optional confidence score to append

    Returns:
        Bounding box array [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w if w > 1e-6 else 0
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                         x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                     x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


# =============================================================================
# Velocity and Direction Utilities
# =============================================================================


def speed_direction_geo(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """
    Compute normalized speed direction between two geo bounding boxes.

    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]

    Returns:
        Unit vector in direction of motion [dy, dx] normalized
    """
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def k_previous_obs(
    observations: dict,
    cur_age: int,
    k: int
) -> List[float]:
    """
    Get observation from k frames ago.

    Args:
        observations: Dictionary mapping age -> observation
        cur_age: Current track age
        k: Number of frames to look back

    Returns:
        Previous observation or [-1, -1, -1, -1, -1] if not found
    """
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


# =============================================================================
# IoU Computations for Geo Coordinates
# =============================================================================


def geo_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes in geo coordinates.

    Args:
        boxes_a: (M, 4+) array of boxes [x1, y1, x2, y2, ...]
        boxes_b: (N, 4+) array of boxes [x1, y1, x2, y2, ...]

    Returns:
        (M, N) IoU matrix
    """
    M = boxes_a.shape[0]
    N = boxes_b.shape[0]

    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=float)

    # Expand dimensions for broadcasting
    a = boxes_a[:, :4]  # (M, 4)
    b = boxes_b[:, :4]  # (N, 4)

    # Intersection
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)  # (M, N)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Areas
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # (M,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # (N,)

    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter

    iou = np.where(union > 0, inter / union, 0)
    return iou


def geo_giou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute Generalized IoU between two sets of boxes.

    GIoU handles non-overlapping boxes better than IoU.

    Args:
        boxes_a: (M, 4+) array of boxes [x1, y1, x2, y2, ...]
        boxes_b: (N, 4+) array of boxes [x1, y1, x2, y2, ...]

    Returns:
        (M, N) GIoU matrix with values in [-1, 1]
    """
    M = boxes_a.shape[0]
    N = boxes_b.shape[0]

    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=float)

    a = boxes_a[:, :4]
    b = boxes_b[:, :4]

    # Intersection
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Areas
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter

    iou = np.where(union > 0, inter / union, 0)

    # Enclosing box
    enc_x1 = np.minimum(a[:, 0:1], b[:, 0:1].T)
    enc_y1 = np.minimum(a[:, 1:2], b[:, 1:2].T)
    enc_x2 = np.maximum(a[:, 2:3], b[:, 2:3].T)
    enc_y2 = np.maximum(a[:, 3:4], b[:, 3:4].T)

    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou = iou - np.where(enc_area > 0, (enc_area - union) / enc_area, 0)
    return giou


def compute_geo_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute IoU between two [x1, y1, x2, y2] boxes in geo coordinates.

    Args:
        box_a: First bounding box [x1, y1, x2, y2]
        box_b: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU value in [0, 1]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_geo_giou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Generalized IoU between two [x1, y1, x2, y2] boxes.

    GIoU handles non-overlapping boxes better than standard IoU.

    Args:
        box_a: First bounding box [x1, y1, x2, y2]
        box_b: Second bounding box [x1, y1, x2, y2]

    Returns:
        GIoU value in [-1, 1] where 1 is perfect overlap
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    iou = inter / union if union > 0 else 0.0

    # Enclosing box
    enc_x1 = min(box_a[0], box_b[0])
    enc_y1 = min(box_a[1], box_b[1])
    enc_x2 = max(box_a[2], box_b[2])
    enc_y2 = max(box_a[3], box_b[3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    if enc_area <= 0:
        return iou

    giou = iou - (enc_area - union) / enc_area
    return giou


def compute_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Euclidean distance between box centers in geo coordinates.

    Args:
        box_a: First bounding box [x1, y1, x2, y2]
        box_b: Second bounding box [x1, y1, x2, y2]

    Returns:
        Euclidean distance in geo units (typically meters)
    """
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2
    return np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)


# =============================================================================
# Geo Detection Parsing
# =============================================================================


def parse_geodets_to_boxes(geodets: np.ndarray) -> np.ndarray:
    """
    Convert geodets format to simple [x1, y1, x2, y2, score] array.

    Input format: [source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls]
    Output format: [x1, y1, x2, y2, score]

    Args:
        geodets: Array of geo detections in full format

    Returns:
        Array of [x1, y1, x2, y2, score] boxes
    """
    if geodets is None or len(geodets) == 0:
        return None

    geo_boxes = np.column_stack([
        geodets[:, 2],  # x1
        geodets[:, 3],  # y1
        geodets[:, 5],  # x2
        geodets[:, 6],  # y2
        geodets[:, 8],  # conf
    ]).astype(float)

    return geo_boxes


def extract_geo_box_from_detection(geo_det: np.ndarray) -> np.ndarray:
    """
    Extract [x1, y1, x2, y2] from a single geo detection.

    Input format: [source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls]
    Output format: [x1, y1, x2, y2]

    Args:
        geo_det: Single geo detection

    Returns:
        Array [x1, y1, x2, y2]
    """
    return np.array([
        geo_det[2],  # x1
        geo_det[3],  # y1
        geo_det[5],  # x2
        geo_det[6],  # y2
    ], dtype=float)
