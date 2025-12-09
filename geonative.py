# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
# Modified to operate in geo-referenced coordinates

"""
DeepOCSort Geo-Native

This version operates the ENTIRE tracking pipeline in geo-referenced coordinates
instead of pixel coordinates. Key differences from standard DeepOCSort:

1. Kalman filter operates in geo-space (meters, real-world velocities)
2. All IoU calculations use geo bounding boxes
3. OCR (Observation-Centric Recovery) uses geo observations
4. Velocity estimation is in real-world units (more predictable)
5. CMC is disabled (geo coordinates are camera-motion invariant)
6. Output is converted back to pixel coordinates

This approach should provide:
- More stable tracking (geo coords don't change with camera motion)
- Better velocity prediction (real-world units are more consistent)
- Improved association in crowded scenes (geo separates overlapping objects)
"""

from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils.association import associate, linear_assignment
from boxmot.utils.ops import xyxy2xysr

from boxmot.utils import logger as LOGGER


# =============================================================================
# Geo-Space Utilities
# =============================================================================

def xyxy2xysr_geo(bbox: np.ndarray) -> np.ndarray:
    """
    Convert [x1, y1, x2, y2] to [cx, cy, area, aspect_ratio] for geo coordinates.
    Same as pixel version but operates on geo coords.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    area = w * h
    aspect_ratio = w / h if h > 1e-6 else 1.0
    return np.array([cx, cy, area, aspect_ratio]).reshape((4, 1))


def convert_x_to_bbox_geo(x, score=None):
    """
    Convert Kalman state [cx, cy, area, aspect_ratio, ...] to [x1, y1, x2, y2].
    Works for both geo and pixel coordinates.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w if w > 1e-6 else 0
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                         x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                     x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction_geo(bbox1, bbox2):
    """
    Compute normalized speed direction between two geo bounding boxes.
    Returns unit vector in direction of motion.
    """
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def k_previous_obs(observations, cur_age, k):
    """Get observation from k frames ago."""
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


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


# =============================================================================
# Geo-Native Kalman Box Tracker
# =============================================================================

class KalmanBoxTrackerGeoNative:
    """
    Kalman Box Tracker that operates entirely in geo-referenced coordinates.

    Key differences from standard KalmanBoxTracker:
    - State is in geo coordinates (meters)
    - Velocity is in real-world units (m/frame)
    - Observations and history are in geo space
    - Pixel coordinates are only stored for output
    """

    count = 0

    def __init__(
            self,
            geo_det: np.ndarray,
            pixel_det: np.ndarray,
            delta_t: int = 3,
            emb: Optional[np.ndarray] = None,
            alpha: float = 0,
            max_obs: int = 50,
            Q_xy_scaling: float = 0.01,
            Q_s_scaling: float = 0.0001,
    ):
        """
        Initialize tracker with geo and pixel detections.

        Args:
            geo_det: [x1, y1, x2, y2, score] in geo coordinates
            pixel_det: [x1, y1, x2, y2, score, cls, det_ind] in pixel coordinates
            delta_t: Time window for velocity estimation
            emb: Appearance embedding
            alpha: Embedding update rate
            max_obs: Maximum observations to store
            Q_xy_scaling: Kalman filter position noise scaling
            Q_s_scaling: Kalman filter scale noise scaling
        """
        self.max_obs = max_obs

        # Store pixel info for output
        self.pixel_box = pixel_det[0:4].copy()
        self.conf = pixel_det[4]
        self.cls = pixel_det[5]
        self.det_ind = pixel_det[6]

        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling

        # === Kalman Filter in GEO space ===
        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            # cx cy  s  r cx' cy' s'
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling
        self.kf.Q[-1, -1] *= self.Q_s_scaling

        self.bbox_to_z_func = xyxy2xysr_geo
        self.x_to_bbox_func = convert_x_to_bbox_geo

        # Initialize state with GEO coordinates
        geo_bbox = geo_det[0:4]
        self.kf.x[:4] = self.bbox_to_z_func(geo_bbox)

        # Track state
        self.time_since_update = 0
        self.id = KalmanBoxTrackerGeoNative.count
        KalmanBoxTrackerGeoNative.count += 1

        self.history = deque([], maxlen=self.max_obs)  # Geo predictions
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Observation tracking (all in GEO space)
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # [x1,y1,x2,y2,score] geo
        self.observations = dict()  # age -> geo observation
        self.history_observations = deque([], maxlen=self.max_obs)

        # Velocity in GEO space (more stable than pixel velocity)
        self.velocity = None
        self.delta_t = delta_t

        # Pixel history (for output only)
        self.pixel_history = deque([], maxlen=self.max_obs)
        self.last_pixel_box = pixel_det[0:4].copy()

        # Appearance
        self.emb = emb
        self.features = deque([], maxlen=self.max_obs)

        self.frozen = False

    def update(
            self,
            geo_det: Optional[np.ndarray],
            pixel_det: Optional[np.ndarray] = None
    ):
        """
        Update state with new observation.

        Args:
            geo_det: [x1, y1, x2, y2, score] in geo coordinates, or None
            pixel_det: [x1, y1, x2, y2, score, cls, det_ind] in pixel coordinates
        """
        if geo_det is not None:
            geo_bbox = geo_det[0:5]  # [x1, y1, x2, y2, score]
            self.conf = geo_bbox[4]
            self.frozen = False

            # Update pixel info if provided
            if pixel_det is not None:
                self.pixel_box = pixel_det[0:4].copy()
                self.last_pixel_box = pixel_det[0:4].copy()
                self.cls = pixel_det[5]
                self.det_ind = pixel_det[6]
                self.pixel_history.append(pixel_det[0:4].copy())

            # Compute velocity from GEO observations
            if self.last_observation[0] >= 0:  # Has previous observation
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                # Velocity in GEO space
                self.velocity = speed_direction_geo(previous_box, geo_bbox)

            # Store GEO observation
            self.last_observation = geo_bbox
            self.observations[self.age] = geo_bbox
            self.history_observations.append(geo_bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1

            # Kalman update with GEO measurement
            self.kf.update(self.bbox_to_z_func(geo_bbox))
        else:
            self.kf.update(None)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):
        """Update appearance embedding with exponential moving average."""
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def apply_affine_correction(self, affine):
        """
        Apply camera motion compensation.

        NOTE: In geo-native mode, we DON'T apply CMC to the Kalman state
        because geo coordinates are camera-motion invariant!
        We only update pixel-space data for output purposes.
        """
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)

        # Update pixel box for output
        if self.last_pixel_box is not None:
            ps = self.last_pixel_box.reshape(2, 2).T
            ps = m @ ps + t
            self.last_pixel_box = ps.T.reshape(-1)

        # NOTE: We do NOT update the Kalman filter state (self.kf.x)
        # because it operates in geo coordinates which are invariant to camera motion!
        # This is a key advantage of geo-native tracking.

    def predict(self):
        """
        Advance state and return predicted GEO bounding box.
        """
        # Don't allow negative bounding boxes
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict(Q=None)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # Store GEO prediction
        geo_pred = self.x_to_bbox_func(self.kf.x)
        self.history.append(geo_pred)
        return geo_pred

    def get_state(self):
        """Return current GEO bounding box estimate."""
        return self.x_to_bbox_func(self.kf.x)

    def get_pixel_state(self):
        """Return last known pixel bounding box (for output)."""
        return self.last_pixel_box.reshape(1, 4) if self.last_pixel_box is not None else None

    def mahalanobis(self, bbox):
        """Compute Mahalanobis distance for a GEO bbox measurement."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


# =============================================================================
# Geo-Native Association Function
# =============================================================================

def associate_geo(
        detections: np.ndarray,
        trackers: np.ndarray,
        iou_func,
        iou_threshold: float,
        velocities: np.ndarray,
        previous_obs: np.ndarray,
        vdc_weight: float,
        emb_cost: Optional[np.ndarray] = None,
        w_emb: float = 0.5,
        aw_off: bool = False,
        aw_param: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Associate detections to trackers using geo coordinates.

    This is adapted from the standard associate() function but uses
    geo-space IoU and velocity calculations.

    Args:
        detections: (M, 5) geo detections [x1, y1, x2, y2, score]
        trackers: (N, 5) geo tracker predictions [x1, y1, x2, y2, 0]
        iou_func: IoU function (geo_iou_batch or geo_giou_batch)
        iou_threshold: Minimum IoU for valid match
        velocities: (N, 2) tracker velocities
        previous_obs: (N, 5) previous observations for OCR
        vdc_weight: Velocity direction consistency weight (inertia)
        emb_cost: (M, N) embedding similarity matrix
        w_emb: Weight for embedding cost
        aw_off: Disable adaptive weighting
        aw_param: Adaptive weighting parameter

    Returns:
        matches: (K, 2) array of matched (det_idx, trk_idx) pairs
        unmatched_dets: (M-K,) array of unmatched detection indices
        unmatched_trks: (N-K,) array of unmatched tracker indices
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))

    # Compute IoU matrix in geo space
    iou_matrix = iou_func(detections, trackers)

    # Velocity Direction Consistency (VDC) - OCM from OC-SORT
    # This penalizes matches where detection movement contradicts track velocity
    if velocities is not None and len(velocities) > 0:
        det_centers = (detections[:, :2] + detections[:, 2:4]) / 2  # (M, 2)
        trk_centers = (trackers[:, :2] + trackers[:, 2:4]) / 2  # (N, 2)
        prev_centers = (previous_obs[:, :2] + previous_obs[:, 2:4]) / 2  # (N, 2)

        # Valid previous observations
        valid_prev = previous_obs[:, 0] >= 0

        if np.any(valid_prev):
            # Compute angle consistency
            Y = det_centers[:, 0:1] - prev_centers[np.newaxis, :, 0]  # (M, N)
            X = det_centers[:, 1:2] - prev_centers[np.newaxis, :, 1]  # (M, N)

            det_angles = np.arctan2(Y, X)  # (M, N)
            trk_angles = np.arctan2(velocities[:, 0], velocities[:, 1])  # (N,)

            angle_diff = np.abs(det_angles - trk_angles[np.newaxis, :])
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

            # Convert to cost (0 = same direction, 1 = opposite)
            vdc_cost = angle_diff / np.pi

            # Apply only to tracks with valid history
            vdc_cost[:, ~valid_prev] = 0

            # Combine with IoU (reduce IoU for velocity-inconsistent matches)
            iou_matrix = iou_matrix - vdc_weight * vdc_cost

    # Embedding cost
    if emb_cost is not None and not aw_off:
        # Adaptive weighting based on IoU
        if aw_param > 0:
            iou_weight = np.clip(iou_matrix, 0, 1)
            emb_weight = 1 - iou_weight ** aw_param
        else:
            emb_weight = w_emb

        # Combine IoU and embedding (higher is better)
        combined = iou_matrix + emb_weight * emb_cost
    else:
        combined = iou_matrix

    # Hungarian assignment (minimize negative combined score)
    if min(combined.shape) > 0:
        matched_indices = linear_assignment(-combined)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    # Filter matches by threshold
    unmatched_dets = []
    unmatched_trks = []
    matches = []

    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trks.append(t)

    for m in matched_indices:
        if combined[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)


# =============================================================================
# Main Tracker Class
# =============================================================================

class GeoNativeDeepOcSort(BaseTracker):
    """
    DeepOCSort operating entirely in geo-referenced coordinates.

    The entire tracking pipeline (Kalman filter, IoU, velocity estimation)
    operates in geo-space. Pixel coordinates are only used for:
    1. Appearance embedding extraction (requires pixel crops)
    2. Final output formatting

    Advantages:
    - Camera motion invariance (no CMC needed)
    - Real-world velocity prediction
    - Better handling of perspective distortion
    - More consistent IoU across different image regions
    """

    def __init__(
            self,
            reid_weights: Path,
            device: torch.device,
            half: bool,
            # BaseTracker parameters
            det_thresh: float = 0.3,
            max_age: int = 50,
            max_obs: int = 50,
            min_hits: int = 3,
            iou_threshold: float = 0.2,
            per_class: bool = False,
            nr_classes: int = 80,
            asso_func: str = "giou",  # Default to GIoU for geo space
            is_obb: bool = False,
            # DeepOcSort-specific parameters
            delta_t: int = 3,
            inertia: float = 0.25,
            w_association_emb: float = 0.5,
            alpha_fixed_emb: float = 0.95,
            aw_param: float = 0.5,
            use_embs: bool = False,
            cmc_off: bool = True,  # CMC is off by default in geo mode
            aw_off: bool = False,
            Q_xy_scaling: float = 0.01,
            Q_s_scaling: float = 0.0001,
            # Geo-specific parameters
            use_giou: bool = True,  # Use GIoU instead of IoU
            ocr_iou_threshold: float = 0.15,  # Separate threshold for OCR stage
            geo_referenced: bool = True,
            **kwargs
    ):
        super().__init__(
            det_thresh=det_thresh,
            max_age=max_age,
            max_obs=max_obs,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            per_class=per_class,
            nr_classes=nr_classes,
            asso_func=asso_func,
            is_obb=is_obb,
            **kwargs
        )

        # Standard parameters
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        self.per_class = per_class
        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling

        # OCR threshold (can be lower than main threshold)
        self.ocr_iou_threshold = ocr_iou_threshold if ocr_iou_threshold is not None else iou_threshold

        KalmanBoxTrackerGeoNative.count = 1

        # ReID model for appearance
        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model

        # CMC - disabled by default for geo-native (geo coords are stable)
        self.cmc_off = cmc_off
        if not cmc_off:
            self.cmc = get_cmc_method("sof")()
        else:
            self.cmc = None

        self.use_embs = use_embs
        self.aw_off = aw_off

        # Geo-specific
        self.use_giou = use_giou
        self.iou_func = geo_giou_batch if use_giou else geo_iou_batch

        LOGGER.success(f"Initialized DeepOcSortGeoNative (giou={use_giou}, cmc_off={cmc_off})")

    def _parse_geodets(
            self,
            geodets: np.ndarray
    ) -> np.ndarray:
        """
        Convert geodets format to simple [x1, y1, x2, y2, score] array.

        Input format: [source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls]
        Output format: [x1, y1, x2, y2, score]
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

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
            self,
            dets: np.ndarray,
            img: np.ndarray,
            index: int,
            embs: np.ndarray = None,
            geodets: np.ndarray = None,
    ) -> np.ndarray:
        """
        Process one frame with geo-native tracking.

        Args:
            dets: (N, >=5) Pixel detections [x1, y1, x2, y2, score, (cls)]
            img: Current frame image (for embedding extraction)
            index: Frame index
            embs: Optional pre-computed embeddings
            geodets: (N, 10) Geo detections
                     [source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls]

        Returns:
            (M, 8) array: [x1, y1, x2, y2, track_id, conf, cls, det_ind] in PIXEL coords
        """
        self.check_inputs(dets, img)
        self.frame_count += 1
        self.height, self.width = img.shape[:2]

        # === Validate geo detections ===
        if geodets is None or len(geodets) == 0:
            LOGGER.warning(f"Frame {index}: No geo detections provided, skipping")
            # Age all tracks
            for trk in self.active_tracks:
                trk.predict()
                trk.update(None)
            # Remove dead tracks
            self.active_tracks = [t for t in self.active_tracks if t.time_since_update <= self.max_age]
            return np.array([])

        # === Filter by confidence ===
        scores = dets[:, 4]
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])  # Add det_ind
        assert dets.shape[1] == 7

        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        geodets = geodets[remain_inds]

        # Parse geo detections to [x1, y1, x2, y2, score]
        geo_boxes = self._parse_geodets(geodets)

        if len(dets) == 0:
            # No detections after filtering
            for trk in self.active_tracks:
                trk.predict()
                trk.update(None)
            self.active_tracks = [t for t in self.active_tracks if t.time_since_update <= self.max_age]
            return np.array([])

        # === Extract appearance embeddings (using PIXEL boxes) ===
        if not self.use_embs or dets.shape[0] == 0:
            dets_embs = np.ones((dets.shape[0], 1))
        elif embs is not None:
            dets_embs = embs[remain_inds]
        else:
            dets_embs = self.model.get_features(dets[:, 0:4], img)

        # === CMC (optional - usually off for geo-native) ===
        if not self.cmc_off and self.cmc is not None:
            transform = self.cmc.apply(img, dets[:, :4])
            for trk in self.active_tracks:
                trk.apply_affine_correction(transform)

        # Embedding update rates
        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        dets_alpha = af + (1 - af) * (1 - trust)

        # === Predict existing tracks (in GEO space) ===
        trks = np.zeros((len(self.active_tracks), 5))
        trk_embs = []
        to_del = []

        for t, trk in enumerate(self.active_tracks):
            pos = trk.predict()[0]  # GEO prediction
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(trk.get_emb())

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        if len(trk_embs) > 0:
            trk_embs = np.vstack(trk_embs)
        else:
            trk_embs = np.array(trk_embs)

        for t in reversed(to_del):
            self.active_tracks.pop(t)

        # Get velocities and previous observations (all in GEO space)
        velocities = np.array([
            trk.velocity if trk.velocity is not None else np.array((0, 0))
            for trk in self.active_tracks
        ])
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])
        k_observations = np.array([
            k_previous_obs(trk.observations, trk.age, self.delta_t)
            for trk in self.active_tracks
        ])

        # =====================================================================
        # Stage 1: Primary association (GEO IoU + Embedding + Velocity)
        # =====================================================================
        if not self.use_embs or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
            stage1_emb_cost = None
        else:
            stage1_emb_cost = dets_embs @ trk_embs.T

        matched, unmatched_dets, unmatched_trks = associate_geo(
            detections=geo_boxes,
            trackers=trks,
            iou_func=self.iou_func,
            iou_threshold=self.iou_threshold,
            velocities=velocities,
            previous_obs=k_observations,
            vdc_weight=self.inertia,
            emb_cost=stage1_emb_cost,
            w_emb=self.w_association_emb,
            aw_off=self.aw_off,
            aw_param=self.aw_param,
        )

        for m in matched:
            self.active_tracks[m[1]].update(geo_boxes[m[0]], dets[m[0], :])
            self.active_tracks[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        # =====================================================================
        # Stage 2: OCR (Observation-Centric Recovery) - using GEO last observations
        # =====================================================================
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = geo_boxes[unmatched_dets]
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks] if len(trk_embs) > 0 else np.array([])

            # Filter out invalid last observations
            valid_mask = left_trks[:, 0] >= 0

            if np.any(valid_mask):
                valid_trk_indices = np.where(valid_mask)[0]
                valid_left_trks = left_trks[valid_mask]

                # Compute IoU in GEO space
                iou_left = self.iou_func(left_dets, valid_left_trks)

                # Embedding cost
                if self.use_embs and len(left_trks_embs) > 0:
                    valid_embs = left_trks_embs[valid_mask]
                    emb_cost_left = left_dets_embs @ valid_embs.T
                else:
                    emb_cost_left = np.zeros_like(iou_left)

                if iou_left.max() > self.ocr_iou_threshold:
                    rematched_indices = linear_assignment(-iou_left)
                    to_remove_det_indices = []
                    to_remove_trk_indices = []

                    for m in rematched_indices:
                        if iou_left[m[0], m[1]] < self.ocr_iou_threshold:
                            continue

                        det_ind = unmatched_dets[m[0]]
                        trk_ind = unmatched_trks[valid_trk_indices[m[1]]]

                        self.active_tracks[trk_ind].update(geo_boxes[det_ind], dets[det_ind, :])
                        self.active_tracks[trk_ind].update_emb(
                            dets_embs[det_ind], alpha=dets_alpha[det_ind]
                        )
                        to_remove_det_indices.append(det_ind)
                        to_remove_trk_indices.append(trk_ind)

                    unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                    unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # =====================================================================
        # Update unmatched tracks
        # =====================================================================
        for m in unmatched_trks:
            self.active_tracks[m].update(None)

        # =====================================================================
        # Create new tracks for unmatched detections
        # =====================================================================
        for i in unmatched_dets:
            trk = KalmanBoxTrackerGeoNative(
                geo_det=geo_boxes[i],
                pixel_det=dets[i],
                delta_t=self.delta_t,
                emb=dets_embs[i],
                alpha=dets_alpha[i],
                Q_xy_scaling=self.Q_xy_scaling,
                Q_s_scaling=self.Q_s_scaling,
                max_obs=self.max_obs,
            )
            self.active_tracks.append(trk)

        # =====================================================================
        # Build output (convert to PIXEL coordinates)
        # =====================================================================
        ret = []
        i = len(self.active_tracks)

        for trk in reversed(self.active_tracks):
            # Get PIXEL box for output
            if trk.last_observation[0] >= 0:
                # Use last pixel observation
                d = trk.last_pixel_box
            else:
                # Fallback to stored pixel box
                d = trk.pixel_box

            if (trk.time_since_update < 1) and (
                    trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate(
                        (d, [trk.id], [trk.conf], [trk.cls], [trk.det_ind])
                    ).reshape(1, -1)
                )

            i -= 1

            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])