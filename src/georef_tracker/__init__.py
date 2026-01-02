"""
Geo-Referenced Tracking

A framework for geo-referenced multi-object tracking in thermal drone videos.
Built on top of the BoxMot Framework, this package provides trackers that
operate in geo-referenced (world) coordinates for improved wildlife tracking.

Features:
- Geo-Native Tracker: Performs tracking entirely in world coordinates
- Hybrid Tracker: Augments standard tracking with geo-referenced matching

Example:
    >>> from georef_tracker import GeoNativeDeepOcSort, GeoHybridDeepOcSort
    >>> import torch
    >>>
    >>> # Initialize a geo-native tracker
    >>> tracker = GeoNativeDeepOcSort(
    ...     reid_weights="osnet_x0_25_msmt17.pt",
    ...     device=torch.device("cuda:0"),
    ...     half=False,
    ... )
    >>>
    >>> # Process a frame
    >>> tracks = tracker.update(
    ...     dets=pixel_detections,  # [x1, y1, x2, y2, score, cls]
    ...     img=frame,
    ...     index=frame_idx,
    ...     geodets=geo_detections,  # [source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls]
    ... )
"""

__version__ = "0.1.0"
__author__ = "Christoph Praschl"
__email__ = "christoph.praschl@fh-hagenberg.at"

from .geonative import (
    GeoNativeDeepOcSort,
    KalmanBoxTrackerGeoNative,
    associate_geo,
)

from .hybrid import (
    GeoHybridDeepOcSort,
    KalmanBoxTrackerGeo,
    geo_association,
)

from .utils import (
    # Coordinate transformations
    xyxy2xysr_geo,
    convert_x_to_bbox_geo,
    # Velocity utilities
    speed_direction_geo,
    k_previous_obs,
    # IoU computations
    geo_iou_batch,
    geo_giou_batch,
    compute_geo_iou,
    compute_geo_giou,
    compute_center_distance,
    # Detection parsing
    parse_geodets_to_boxes,
    extract_geo_box_from_detection,
)

__all__ = [
    # Main tracker classes
    "GeoNativeDeepOcSort",
    "GeoHybridDeepOcSort",
    # Kalman trackers
    "KalmanBoxTrackerGeoNative",
    "KalmanBoxTrackerGeo",
    # Association functions
    "associate_geo",
    "geo_association",
    # Utilities
    "xyxy2xysr_geo",
    "convert_x_to_bbox_geo",
    "speed_direction_geo",
    "k_previous_obs",
    "geo_iou_batch",
    "geo_giou_batch",
    "compute_geo_iou",
    "compute_geo_giou",
    "compute_center_distance",
    "parse_geodets_to_boxes",
    "extract_geo_box_from_detection",
]
