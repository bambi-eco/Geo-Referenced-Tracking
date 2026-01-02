# Geo-Referenced Tracker

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A Python framework for **geo-referenced multi-object tracking** in thermal drone videos. This package extends the [BoxMot Framework](https://github.com/mikel-brostrom/boxmot) with geo-referenced tracking capabilities specifically designed for wildlife monitoring applications.

## Overview

Traditional tracking methods operate purely in image space and often fail in challenging environments due to:
- Occlusions from forest canopy
- Low texture in thermal imagery
- Changing viewpoints from drone movement
- Ambiguous thermal signatures of similar animals

This framework addresses these limitations by projecting detections into **global geographic coordinates** using RTK-GNSS drone localization, camera pose data, and digital elevation models.

### Two Tracking Strategies

1. **Geo-Native Tracker** (`GeoNativeDeepOcSort`): Performs tracking entirely in world coordinates, enabling physically meaningful motion modeling and removing the need for camera-motion compensation.

2. **Hybrid Tracker** (`GeoHybridDeepOcSort`): Augments standard image-space association with an additional geo-referenced matching stage to recover identities during severe occlusions or ambiguous visual conditions.

## Installation

```bash
git clone https://github.com/cpraschl/georef-tracker.git
cd georef-tracker
pip install -e .
```

## Requirements

- Python 3.8+
- PyTorch >= 1.9.0
- BoxMot >= 11.0.0
- NumPy >= 1.21.0
- OpenCV >= 4.5.0

## Re-ID Model

You can use the standard BoxMot osnet Re-ID model or us our custom one trained on thermal wildlife data: [Download here](https://huggingface.co/cpraschl/bambi-thermal-omni)

## Quick Start

### Basic Usage

```python
import numpy as np
import cv2
import torch
from pathlib import Path

from georef_tracker import GeoHybridDeepOcSort, GeoNativeDeepOcSort

# Initialize tracker
tracker = GeoHybridDeepOcSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    half=False,
    geo_referenced=True,
)

# Or use the geo-native tracker
# tracker = GeoNativeDeepOcSort(
#     reid_weights=Path("osnet_x0_25_msmt17.pt"),
#     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#     half=False,
# )

# Load your data
img = cv2.imread("frame.jpg")

# Pixel detections: [x1, y1, x2, y2, score, class_id]
dets = np.array([
    [100, 150, 200, 250, 0.95, 0],
    [300, 350, 400, 450, 0.87, 0],
])

# Geo detections: [source_id, frame_id, geo_x1, geo_y1, geo_z1, geo_x2, geo_y2, geo_z2, conf, cls]
geodets = np.array([
    [0 21 438970.718620 5291560.590827 503.802050 438972.105714 5291564.108368 505.893069 0.7970 0],
    [1 23 438870.921288 5291604.090894 487.355146 438874.372865 5291606.145935 488.605293 0.6427 0],
])

# Run tracking
tracks = tracker.update(
    dets=dets,
    img=img,
    index=1,  # frame index
    geodets=geodets,
)

# Output: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
for track in tracks:
    x1, y1, x2, y2, track_id, conf, cls, det_ind = track
    print(f"Track {int(track_id)}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={conf:.2f}")
```

### Complete Example with File Loading

```python
import numpy as np
import cv2
from pathlib import Path
import torch

from georef_tracker import GeoHybridDeepOcSort

# Paths to data files
dets_path = Path("/path/to/sequence/dets/seq01.txt")
geo_dets_path = Path("/path/to/sequence/geodets/seq01.txt")
img_dir = Path("/path/to/sequence/images/")
embs_path = Path("/path/to/sequence/embs/seq01.txt")  # optional

# Load detections
# Format: frame_id, x, y, w, h, score, class_id
dets = np.atleast_2d(np.loadtxt(dets_path, skiprows=1))

# Load geo detections
# Format: line_id, frame_id, min_x, min_y, min_z, max_x, max_y, max_z, score, class_id
geo_dets = np.atleast_2d(np.loadtxt(geo_dets_path, skiprows=1))

# Load embeddings (optional)
embs = np.atleast_2d(np.loadtxt(embs_path)) if embs_path.exists() else None

# Combine arrays for easy iteration
if embs is not None:
    dets_n_geo_n_embs = np.concatenate((dets, geo_dets, embs), axis=1)
else:
    dets_n_geo_n_embs = np.concatenate((dets, geo_dets), axis=1)

# Initialize tracker
tracker = GeoHybridDeepOcSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device=torch.device("cuda:0"),
    half=False,
)

# Process frames
for frame_num in range(1, 100):
    frame_num_padded = f"{frame_num:08d}"
    
    # Load image
    img_path = img_dir / f"{frame_num_padded}.jpg"
    img = cv2.imread(str(img_path))
    
    # Get detections for this frame
    frame_rows = dets_n_geo_n_embs[dets_n_geo_n_embs[:, 0] == int(frame_num_padded)]
    
    if len(frame_rows) == 0:
        continue
    
    # Parse columns
    dets_frame = frame_rows[:, 1:7]      # [x, y, w, h, score, class_id]
    geo_dets_frame = frame_rows[:, 7:17]  # geo detection columns
    embs_frame = frame_rows[:, 17:] if embs is not None else None
    
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    dets_xyxy = np.column_stack([
        dets_frame[:, 0],                    # x1
        dets_frame[:, 1],                    # y1
        dets_frame[:, 0] + dets_frame[:, 2], # x2
        dets_frame[:, 1] + dets_frame[:, 3], # y2
        dets_frame[:, 4],                    # score
        dets_frame[:, 5],                    # class_id
    ])
    
    # Run tracker
    tracks = tracker.update(
        dets=dets_xyxy,
        img=img,
        index=frame_num,
        embs=embs_frame,
        geodets=geo_dets_frame,
    )
    
    # Process results
    for track in tracks:
        x1, y1, x2, y2, track_id, conf, cls, det_ind = track
        print(f"Frame {frame_num}, Track {int(track_id)}: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
```

## Tracker Parameters

### GeoNativeDeepOcSort

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reid_weights` | Path | - | Path to ReID model weights |
| `device` | torch.device | - | Device for inference |
| `half` | bool | False | Use FP16 for ReID |
| `det_thresh` | float | 0.3 | Detection confidence threshold |
| `max_age` | int | 50 | Max frames to keep lost track |
| `min_hits` | int | 3 | Min hits before track confirmation |
| `iou_threshold` | float | 0.2 | IoU threshold for association |
| `use_giou` | bool | True | Use GIoU instead of IoU |
| `use_embs` | bool | False | Enable appearance embeddings |
| `cmc_off` | bool | True | Disable camera motion compensation |

### GeoHybridDeepOcSort

Includes all parameters from GeoNativeDeepOcSort plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geo_referenced` | bool | True | Enable geo association stage |
| `geo_iou_threshold` | float | 0.12 | IoU threshold for geo matching |
| `geo_center_dist_threshold` | float | 5.0 | Max center distance (meters) for fallback |
| `geo_use_giou` | bool | True | Use GIoU for geo matching |
| `geo_only_for_lost` | bool | False | Only use geo for lost tracks |
| `geo_lost_threshold` | int | 1 | Frames lost before geo kicks in |

## Data Formats

### Pixel Detections

Input format: `[x1, y1, x2, y2, score, class_id]`

- `x1, y1`: Top-left corner (pixels)
- `x2, y2`: Bottom-right corner (pixels)
- `score`: Detection confidence [0, 1]
- `class_id`: Object class identifier

### Geo Detections

Input format: `[source_id, frame_id, x1, y1, z1, x2, y2, z2, conf, cls]`

- `source_id`: Detection source identifier
- `frame_id`: Frame number
- `x1, y1, z1`: Min corner in UTM coordinates
- `x2, y2, z2`: Max corner in UTM coordinates 
- `conf`: Detection confidence
- `cls`: Class identifier

### Track Output

Output format: `[x1, y1, x2, y2, track_id, conf, cls, det_ind]`

- `x1, y1, x2, y2`: Bounding box in pixel coordinates
- `track_id`: Unique track identifier
- `conf`: Detection confidence
- `cls`: Class identifier
- `det_ind`: Index into input detections

## Architecture

```
georef_tracker/
├── __init__.py          # Package exports
├── utils.py             # Shared utilities (IoU, transformations)
├── geonative.py         # GeoNativeDeepOcSort tracker
└── hybrid.py            # GeoHybridDeepOcSort tracker
```

### Association Cascade (Hybrid Tracker)

1. **Stage 1**: IoU + Embedding + Velocity (standard DeepOCSort)
2. **Stage 2**: OCR - Observation-Centric Recovery
3. **Stage 3**: Geo-referenced matching with GIoU and center distance fallback

## Citation

If you use this framework in your research, please cite:

```bibtex
@Article{praschlGeoReferencedTracking,
    author  = {Praschl, Christoph and Coucke, Vincent and Maschek, Anna and Schedl, David},
    title   = {Stay Local or Go Global: Geo-Referenced Bounding Boxes for
               Tracking Wildlife in Thermal Drone Videos},
}
```

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [BoxMot Framework](https://github.com/mikel-brostrom/boxmot) by Mikel Broström
- Part of the [BAMBI Project](https://github.com/bambi-eco) for wildlife detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
