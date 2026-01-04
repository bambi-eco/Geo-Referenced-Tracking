"""
GeoHybrid DeepOCSort Tracker

This extends the standard DeepOCSort with an additional geo-referenced
association stage as a final fallback. The geo stage helps recover
associations that fail in pixel-space due to:
- Severe occlusion
- Similar appearance (embedding confusion)
- Camera motion artifacts
- Large displacement between frames

Association cascade:
1. Stage 1: IoU + Embedding + Velocity (standard DeepOCSort)
2. Stage 2: OCR - Observation-Centric Recovery (last observations)
3. Stage 3: Geo-referenced matching (NEW - world coordinates)

Based on the BoxMot Framework by Mikel Broström.
Extended with geo-referenced association for wildlife tracking.
"""

from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils.association import associate, linear_assignment
from boxmot.utils.ops import xyxy2xysr

try:
    from boxmot.utils import logger as LOGGER
except ImportError:
    import logging
    LOGGER = logging.getLogger(__name__)

from .utils import (
    compute_geo_giou,
    compute_geo_iou,
    compute_center_distance,
    extract_geo_box_from_detection,
)


# =============================================================================
# Helper Functions
# =============================================================================


def k_previous_obs(observations: dict, cur_age: int, k: int) -> List[float]:
    """Get observation from k frames ago."""
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_x_to_bbox(x: np.ndarray, score: Optional[float] = None) -> np.ndarray:
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    return np.array(
        [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
    ).reshape((1, 5))


def speed_direction(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute normalized speed direction between two bounding boxes."""
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


# =============================================================================
# Geo-Referenced Association
# =============================================================================


def geo_association(
    det_geo_boxes: np.ndarray,
    trk_geo_boxes: np.ndarray,
    unmatched_dets: np.ndarray,
    unmatched_trks: np.ndarray,
    geo_iou_threshold: float = 0.2,
    geo_center_dist_threshold: float = 3.0,
    use_giou: bool = True,
    class_aware: bool = False,
    det_classes: Optional[np.ndarray] = None,
    trk_classes: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Perform geo-referenced association on unmatched detections and tracks.

    Two-stage approach:
    1. GIoU/IoU-based matching with threshold
    2. Center-distance fallback for remaining (useful for very lost tracks)

    Args:
        det_geo_boxes: (len(unmatched_dets), 4) geo boxes for unmatched detections
                       IMPORTANT: This should be pre-filtered to only contain
                       boxes for unmatched_dets, indexed 0..len(unmatched_dets)-1
        trk_geo_boxes: (len(unmatched_trks), 4) geo boxes for unmatched tracks
                       IMPORTANT: This should be pre-filtered to only contain
                       boxes for unmatched_trks, indexed 0..len(unmatched_trks)-1
        unmatched_dets: Original indices of unmatched detections (for output mapping)
        unmatched_trks: Original indices of unmatched tracks (for output mapping)
        geo_iou_threshold: Minimum IoU/GIoU for geo matching
        geo_center_dist_threshold: Max center distance (meters) for fallback
        use_giou: Use GIoU instead of IoU
        class_aware: Enforce class matching
        det_classes: Class labels indexed same as det_geo_boxes (len = len(unmatched_dets))
        trk_classes: Class labels indexed same as trk_geo_boxes (len = len(unmatched_trks))

    Returns:
        matches: List of (det_idx, trk_idx) pairs using ORIGINAL indices
        unmatched_dets: Remaining unmatched detection indices (original)
        unmatched_trks: Remaining unmatched track indices (original)
    """
    if len(unmatched_dets) == 0 or len(unmatched_trks) == 0:
        return [], unmatched_dets, unmatched_trks

    # Build cost matrix for unmatched pairs
    # M = number of unmatched detections, N = number of unmatched tracks
    M = len(unmatched_dets)
    N = len(unmatched_trks)

    # Validate input dimensions
    assert len(det_geo_boxes) == M, (
        f"det_geo_boxes length ({len(det_geo_boxes)}) must match unmatched_dets ({M})"
    )
    assert len(trk_geo_boxes) == N, (
        f"trk_geo_boxes length ({len(trk_geo_boxes)}) must match unmatched_trks ({N})"
    )

    iou_matrix = np.zeros((M, N), dtype=float)
    dist_matrix = np.zeros((M, N), dtype=float)

    metric_fn = compute_geo_giou if use_giou else compute_geo_iou

    # Use LOCAL indices (i, j) to index into the pre-filtered arrays
    for i in range(M):
        det_box = det_geo_boxes[i]  # Local index
        det_cls = det_classes[i] if det_classes is not None else None

        for j in range(N):
            trk_box = trk_geo_boxes[j]  # Local index
            trk_cls = trk_classes[j] if trk_classes is not None else None

            # Class mismatch -> invalid
            if class_aware and det_cls is not None and trk_cls is not None:
                if det_cls != trk_cls:
                    iou_matrix[i, j] = -1e6  # Large negative for GIoU
                    dist_matrix[i, j] = 1e6
                    continue

            iou_matrix[i, j] = metric_fn(det_box, trk_box)
            dist_matrix[i, j] = compute_center_distance(det_box, trk_box)

    matches = []
    remaining_det_mask = np.ones(M, dtype=bool)
    remaining_trk_mask = np.ones(N, dtype=bool)

    # === Stage 3a: GIoU/IoU-based matching ===
    if iou_matrix.max() > geo_iou_threshold:
        # Hungarian assignment on IoU (maximize IoU = minimize -IoU)
        cost_matrix = -iou_matrix.copy()

        # Mask out invalid entries
        cost_matrix[iou_matrix < geo_iou_threshold] = 1e6

        matched_indices = linear_assignment(cost_matrix)

        for m in matched_indices:
            i, j = m[0], m[1]  # Local indices
            if iou_matrix[i, j] >= geo_iou_threshold:
                # Map back to ORIGINAL indices for output
                det_idx = unmatched_dets[i]
                trk_idx = unmatched_trks[j]
                matches.append((det_idx, trk_idx))
                remaining_det_mask[i] = False
                remaining_trk_mask[j] = False

    # === Stage 3b: Center distance fallback for remaining ===
    remaining_dets_local = np.where(remaining_det_mask)[0]
    remaining_trks_local = np.where(remaining_trk_mask)[0]

    if len(remaining_dets_local) > 0 and len(remaining_trks_local) > 0:
        # Build reduced distance matrix
        sub_dist = dist_matrix[np.ix_(remaining_dets_local, remaining_trks_local)]

        if sub_dist.min() < geo_center_dist_threshold:
            matched_indices = linear_assignment(sub_dist)

            for m in matched_indices:
                i_sub, j_sub = m[0], m[1]  # Indices into sub_dist
                if sub_dist[i_sub, j_sub] < geo_center_dist_threshold:
                    # Map sub_dist indices -> local indices -> original indices
                    i_local = remaining_dets_local[i_sub]
                    j_local = remaining_trks_local[j_sub]
                    det_idx = unmatched_dets[i_local]
                    trk_idx = unmatched_trks[j_local]
                    matches.append((det_idx, trk_idx))
                    remaining_det_mask[i_local] = False
                    remaining_trk_mask[j_local] = False

    # Compute final unmatched (return ORIGINAL indices)
    final_unmatched_dets = unmatched_dets[remaining_det_mask]
    final_unmatched_trks = unmatched_trks[remaining_trk_mask]

    return matches, final_unmatched_dets, final_unmatched_trks


# =============================================================================
# Extended Kalman Box Tracker with Geo Support
# =============================================================================


class KalmanBoxTrackerGeo:
    """
    Extended KalmanBoxTracker with geo-referenced coordinate tracking.
    Maintains both pixel-space and geo-space state.
    """

    count = 0

    def __init__(
        self,
        det: np.ndarray,
        geo_det: Optional[np.ndarray] = None,
        delta_t: int = 3,
        emb: Optional[np.ndarray] = None,
        alpha: float = 0,
        max_obs: int = 50,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,
    ):
        """
        Initialize tracker with detection and optional geo detection.

        Args:
            det: [x1, y1, x2, y2, score, cls, det_ind] pixel detection
            geo_det: [source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls] geo detection
            delta_t: Time window for velocity estimation
            emb: Appearance embedding
            alpha: Embedding update rate
            max_obs: Maximum observations to store
            Q_xy_scaling: Kalman filter position noise scaling
            Q_s_scaling: Kalman filter scale noise scaling
        """
        self.max_obs = max_obs
        bbox = det[0:5]
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]

        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling

        # Pixel-space Kalman filter (standard DeepOCSort)
        self.kf = KalmanFilterXYSR(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling
        self.kf.Q[-1, -1] *= self.Q_s_scaling

        self.bbox_to_z_func = xyxy2xysr
        self.x_to_bbox_func = convert_x_to_bbox

        self.kf.x[:4] = self.bbox_to_z_func(bbox)

        # === Geo-referenced state ===
        self.geo_box = None  # [x1, y1, x2, y2] in geo coordinates
        self.geo_history = deque([], maxlen=max_obs)
        self.geo_velocity = None  # [vx, vy] in geo coordinates

        if geo_det is not None:
            self._update_geo(geo_det)

        # Standard tracking state
        self.time_since_update = 0
        self.id = KalmanBoxTrackerGeo.count
        KalmanBoxTrackerGeo.count += 1
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.features = deque([], maxlen=self.max_obs)
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t
        self.history_observations = deque([], maxlen=self.max_obs)

        self.emb = emb
        self.frozen = False

    def _update_geo(self, geo_det: np.ndarray):
        """Update geo state from geo detection."""
        new_geo_box = extract_geo_box_from_detection(geo_det)

        # Compute geo velocity if we have history
        if self.geo_box is not None:
            old_center = np.array(
                [
                    (self.geo_box[0] + self.geo_box[2]) / 2,
                    (self.geo_box[1] + self.geo_box[3]) / 2,
                ]
            )
            new_center = np.array(
                [
                    (new_geo_box[0] + new_geo_box[2]) / 2,
                    (new_geo_box[1] + new_geo_box[3]) / 2,
                ]
            )
            self.geo_velocity = new_center - old_center

        self.geo_box = new_geo_box
        self.geo_history.append(new_geo_box.copy())

    def get_geo_box(self) -> Optional[np.ndarray]:
        """Get current geo box, with simple velocity prediction if lost."""
        if self.geo_box is None:
            return None

        if self.time_since_update > 0 and self.geo_velocity is not None:
            # Predict geo position using velocity
            predicted = self.geo_box.copy()
            predicted[0] += self.geo_velocity[0] * self.time_since_update
            predicted[1] += self.geo_velocity[1] * self.time_since_update
            predicted[2] += self.geo_velocity[0] * self.time_since_update
            predicted[3] += self.geo_velocity[1] * self.time_since_update
            return predicted

        return self.geo_box.copy()

    def update(self, det: Optional[np.ndarray], geo_det: Optional[np.ndarray] = None):
        """
        Updates the state vector with observed bbox.

        Args:
            det: [x1, y1, x2, y2, score, cls, det_ind] pixel detection, or None
            geo_det: Optional geo detection
        """
        if det is not None:
            bbox = det[0:5]
            self.conf = det[4]
            self.cls = det[5]
            self.det_ind = det[6]
            self.frozen = False

            if self.last_observation.sum() >= 0:
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)

            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1

            self.kf.update(self.bbox_to_z_func(bbox))

            # Update geo state
            if geo_det is not None:
                self._update_geo(geo_det)
        else:
            self.kf.update(det)
            self.frozen = True

    def update_emb(self, emb: np.ndarray, alpha: float = 0.9):
        """Update appearance embedding with exponential moving average."""
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self) -> np.ndarray:
        """Get current appearance embedding."""
        return self.emb

    def apply_affine_correction(self, affine: np.ndarray):
        """Apply camera motion compensation."""
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)

        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        self.kf.apply_affine_correction(m, t)

        # Note: geo coordinates are NOT affected by camera motion
        # This is a key advantage of geo-referenced tracking!

    def predict(self) -> np.ndarray:
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict(Q=None)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """Returns the current bounding box estimate."""
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox: np.ndarray) -> float:
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


# =============================================================================
# Main Tracker Class
# =============================================================================


class GeoHybridDeepOcSort(BaseTracker):
    """
    DeepOCSort with Geo-Referenced Association.

    Extends standard DeepOCSort with geo-referenced matching as a final
    fallback stage. The association cascade is:

    1. Stage 1: IoU + Embedding + Velocity (standard DeepOCSort)
       - Best for: Normal tracking with good visibility

    2. Stage 2: OCR (Observation-Centric Recovery)
       - Best for: Brief occlusions, uses last known positions

    3. Stage 3: Geo-referenced matching (NEW)
       - Best for: Long occlusions, camera motion, similar appearances
       - Sub-stages:
         a) GIoU-based matching in world coordinates
         b) Center-distance fallback for very lost tracks

    The geo stage is particularly effective because:
    - World coordinates are invariant to camera motion
    - Objects that look similar in pixels may be far apart in world space
    - Geo velocity is in real units (m/s), more predictable than pixels
    """

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,
        # BaseTracker parameters
        det_thresh: float = 0.3,
        max_age: int = 30,
        max_obs: int = 50,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_class: bool = False,
        nr_classes: int = 80,
        asso_func: str = "iou",
        is_obb: bool = False,
        # DeepOcSort-specific parameters
        delta_t: int = 3,
        inertia: float = 0.2,
        w_association_emb: float = 0.5,
        alpha_fixed_emb: float = 0.95,
        aw_param: float = 0.5,
        use_embs: bool = False,
        cmc_off: bool = False,
        aw_off: bool = False,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,
        # Geo-referenced parameters (NEW)
        geo_referenced: bool = True,
        geo_iou_threshold: float = 0.12,
        geo_center_dist_threshold: float = 5.0,
        geo_use_giou: bool = True,
        geo_only_for_lost: bool = False,  # Only use geo for tracks lost > N frames
        geo_lost_threshold: int = 1,  # Frames lost before geo kicks in
        **kwargs,
    ):
        """
        Initialize the GeoHybrid DeepOCSort tracker.

        Args:
            reid_weights: Path to ReID model weights
            device: Torch device for inference
            half: Use half precision for ReID model
            det_thresh: Detection confidence threshold
            max_age: Maximum frames to keep track alive without detection
            max_obs: Maximum observations to store per track
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for association
            per_class: Track objects per class
            nr_classes: Number of classes
            asso_func: Association function name
            is_obb: Use oriented bounding boxes
            delta_t: Time window for velocity estimation
            inertia: Velocity direction consistency weight
            w_association_emb: Embedding weight in association
            alpha_fixed_emb: Fixed component of embedding update rate
            aw_param: Adaptive weighting parameter
            use_embs: Use appearance embeddings
            cmc_off: Disable camera motion compensation
            aw_off: Disable adaptive weighting
            Q_xy_scaling: Kalman filter position noise scaling
            Q_s_scaling: Kalman filter scale noise scaling
            geo_referenced: Enable geo-referenced association
            geo_iou_threshold: IoU threshold for geo association
            geo_center_dist_threshold: Max center distance for geo fallback
            geo_use_giou: Use GIoU for geo association
            geo_only_for_lost: Only use geo for lost tracks
            geo_lost_threshold: Frames lost before geo kicks in
        """
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
            **kwargs,
        )

        # Standard DeepOCSort parameters
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = asso_func
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        self.per_class = per_class
        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling
        KalmanBoxTrackerGeo.count = 1

        self.model = ReidAutoBackend(weights=reid_weights, device=device, half=half).model
        self.cmc = get_cmc_method("sof")()
        self.cmc_off = cmc_off
        self.aw_off = aw_off

        # Geo-referenced parameters
        self.geo_referenced = geo_referenced
        self.geo_iou_threshold = geo_iou_threshold
        self.geo_center_dist_threshold = geo_center_dist_threshold
        self.geo_use_giou = geo_use_giou
        self.geo_only_for_lost = geo_only_for_lost
        self.geo_lost_threshold = geo_lost_threshold
        self.use_embs = use_embs

        if hasattr(LOGGER, "success"):
            LOGGER.success(f"Initialized GeoHybridDeepOcSort (geo_enabled={geo_referenced})")

    def _parse_geodets(
        self, geodets: Optional[np.ndarray], n_dets: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse geo detections into boxes and full records.

        Returns:
            geo_boxes: (N, 4) array of [x1, y1, x2, y2] in geo coords, or None
            geodets: Original geodets array, or None
        """
        if geodets is None or len(geodets) == 0:
            return None, None

        assert len(geodets) == n_dets, f"geodets ({len(geodets)}) must match dets ({n_dets})"

        # Extract [x1, y1, x2, y2] from geodets
        # Format: [source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls]
        geo_boxes = np.column_stack(
            [
                geodets[:, 2],  # x1
                geodets[:, 3],  # y1
                geodets[:, 5],  # x2
                geodets[:, 6],  # y2
            ]
        ).astype(float)

        return geo_boxes, geodets

    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        index: int = None,
        embs: np.ndarray = None,
        geodets: np.ndarray = None,
    ) -> np.ndarray:
        """
        Process one frame of detections.

        Args:
            dets: (N, >=5) Pixel detections [x1, y1, x2, y2, score, (cls)]
            img: Current frame image
            index: Frame index (optional - falls back to self._current_frame_index or frame_count)
            embs: Optional pre-computed embeddings
            geodets: (N, 10) Geo detections (optional - falls back to self._current_geodets)
                     [source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls]

        Returns:
            (M, 8) array: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
        """
        # Support both direct parameters and instance attributes (for BoxMOT v16+ compatibility)
        if index is None:
            index = getattr(self, '_current_frame_index', self.frame_count)
        if geodets is None:
            geodets = getattr(self, '_current_geodets', None)
        
        # Handle empty detections (replaces @setup_decorator functionality)
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 6))
        
        self.check_inputs(dets, img)

        self.frame_count += 1
        self.height, self.width = img.shape[:2]

        scores = dets[:, 4]
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        assert dets.shape[1] == 7

        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # Parse geo detections
        if geodets is not None and len(geodets) > 0:
            geodets = geodets[remain_inds]
        geo_boxes, geodets_filtered = self._parse_geodets(
            geodets, len(dets) if geodets is not None else 0
        )

        # Appearance descriptor extraction
        if not self.use_embs or dets.shape[0] == 0:
            dets_embs = np.ones((dets.shape[0], 1))
        elif embs is not None:
            dets_embs = embs[remain_inds]
        else:
            dets_embs = self.model.get_features(dets[:, 0:4], img)

        # CMC (Camera Motion Compensation)
        if not self.cmc_off:
            transform = self.cmc.apply(img, dets[:, :4])
            for trk in self.active_tracks:
                trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        dets_alpha = af + (1 - af) * (1 - trust)

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.active_tracks), 5))
        trk_embs = []
        trk_geo_boxes = []
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.active_tracks[t].get_emb())
                trk_geo_boxes.append(self.active_tracks[t].get_geo_box())

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        if len(trk_embs) > 0:
            trk_embs = np.vstack(trk_embs)
        else:
            trk_embs = np.array(trk_embs)

        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.active_tracks
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])
        k_observations = np.array(
            [
                k_previous_obs(trk.observations, trk.age, self.delta_t)
                for trk in self.active_tracks
            ]
        )

        # =====================================================================
        # Stage 1: First round of association (IoU + Embedding + Velocity)
        # =====================================================================
        if not self.use_embs or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
            stage1_emb_cost = None
        else:
            stage1_emb_cost = dets_embs @ trk_embs.T

        matched, unmatched_dets, unmatched_trks = associate(
            dets[:, 0:5],
            trks,
            self.asso_func,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
            img.shape[1],
            img.shape[0],
            stage1_emb_cost,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
        )

        for m in matched:
            geo_det = geodets_filtered[m[0]] if geodets_filtered is not None else None
            self.active_tracks[m[1]].update(dets[m[0], :], geo_det=geo_det)
            self.active_tracks[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        # =====================================================================
        # Stage 2: Second round of association by OCR (last observations)
        # =====================================================================
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_trks_embs = trk_embs[unmatched_trks]

            iou_left = self.asso_func(left_dets, left_trks)
            emb_cost_left = left_dets_embs @ left_trks_embs.T
            if not self.use_embs:
                emb_cost_left = np.zeros_like(emb_cost_left)
            iou_left = np.array(iou_left)

            if iou_left.max() > self.iou_threshold:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []

                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    geo_det = geodets_filtered[det_ind] if geodets_filtered is not None else None
                    self.active_tracks[trk_ind].update(dets[det_ind, :], geo_det=geo_det)
                    self.active_tracks[trk_ind].update_emb(
                        dets_embs[det_ind], alpha=dets_alpha[det_ind]
                    )
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)

                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # =====================================================================
        # Stage 3: Geo-referenced association (NEW)
        # =====================================================================
        if (
            self.geo_referenced
            and geo_boxes is not None
            and unmatched_dets.shape[0] > 0
            and unmatched_trks.shape[0] > 0
        ):
            # Filter tracks for geo matching
            if self.geo_only_for_lost:
                # Only consider tracks that have been lost for a while
                geo_eligible_trks = np.array(
                    [
                        tj
                        for tj in unmatched_trks
                        if (
                            self.active_tracks[tj].time_since_update >= self.geo_lost_threshold
                            and self.active_tracks[tj].geo_box is not None
                        )
                    ]
                )
            else:
                geo_eligible_trks = np.array(
                    [
                        tj
                        for tj in unmatched_trks
                        if self.active_tracks[tj].geo_box is not None
                    ]
                )

            if len(geo_eligible_trks) > 0:
                # Build FILTERED detection geo boxes (only unmatched dets)
                det_geo_boxes_filtered = geo_boxes[unmatched_dets]

                # Build FILTERED track geo boxes (only geo-eligible tracks)
                trk_geo_boxes_filtered = np.array(
                    [self.active_tracks[tj].get_geo_box() for tj in geo_eligible_trks]
                )

                # Get class information if per_class tracking (also filtered)
                det_classes_filtered = (
                    dets[unmatched_dets, 5].astype(int) if self.per_class else None
                )
                trk_classes_filtered = (
                    np.array([self.active_tracks[tj].cls for tj in geo_eligible_trks])
                    if self.per_class
                    else None
                )

                # Perform geo association
                # NOTE: geo_association expects pre-filtered arrays indexed 0..N-1
                # and returns matches using the ORIGINAL indices
                geo_matches, remaining_dets, remaining_trks = geo_association(
                    det_geo_boxes=det_geo_boxes_filtered,
                    trk_geo_boxes=trk_geo_boxes_filtered,
                    unmatched_dets=unmatched_dets,
                    unmatched_trks=geo_eligible_trks,
                    geo_iou_threshold=self.geo_iou_threshold,
                    geo_center_dist_threshold=self.geo_center_dist_threshold,
                    use_giou=self.geo_use_giou,
                    class_aware=self.per_class,
                    det_classes=det_classes_filtered,
                    trk_classes=trk_classes_filtered,
                )

                # Apply geo matches (det_idx and trk_idx are ORIGINAL indices)
                for det_idx, trk_idx in geo_matches:
                    geo_det = geodets_filtered[det_idx] if geodets_filtered is not None else None
                    self.active_tracks[trk_idx].update(dets[det_idx, :], geo_det=geo_det)
                    self.active_tracks[trk_idx].update_emb(
                        dets_embs[det_idx], alpha=dets_alpha[det_idx]
                    )

                # Update unmatched lists
                matched_det_set = set(m[0] for m in geo_matches)
                matched_trk_set = set(m[1] for m in geo_matches)
                unmatched_dets = np.array(
                    [d for d in unmatched_dets if d not in matched_det_set]
                )
                unmatched_trks = np.array(
                    [t for t in unmatched_trks if t not in matched_trk_set]
                )

                if len(geo_matches) > 0 and hasattr(LOGGER, "debug"):
                    LOGGER.debug(
                        f"Frame {index}: Geo stage recovered {len(geo_matches)} associations"
                    )

        # =====================================================================
        # Update unmatched tracks (no detection)
        # =====================================================================
        for m in unmatched_trks:
            self.active_tracks[m].update(None)

        # =====================================================================
        # Create new tracks for unmatched detections
        # =====================================================================
        for i in unmatched_dets:
            geo_det = geodets_filtered[i] if geodets_filtered is not None else None
            trk = KalmanBoxTrackerGeo(
                dets[i],
                geo_det=geo_det,
                delta_t=self.delta_t,
                emb=dets_embs[i],
                alpha=dets_alpha[i],
                Q_xy_scaling=self.Q_xy_scaling,
                Q_s_scaling=self.Q_s_scaling,
                max_obs=self.max_obs,
            )
            self.active_tracks.append(trk)

        # =====================================================================
        # Build output and cleanup
        # =====================================================================
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]

            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id], [trk.conf], [trk.cls], [trk.det_ind])).reshape(
                        1, -1
                    )
                )
            i -= 1

            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.array([])
