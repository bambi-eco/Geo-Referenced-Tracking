# Geo-referenced Tracking

This replication package accompanies the paper “Stay Local or Go Global: Geo-Referenced Bounding Boxes for Tracking Wildlife in Thermal Drone Videos.” The work introduces a geo-referenced multi-object tracking framework that improves wildlife tracking in challenging thermal drone imagery.

Traditional tracking methods operate purely in image space and often fail in forested environments due to occlusions, low texture, changing viewpoints, and ambiguous thermal signatures. To address these limitations, our methodology projects each detection into global geographic coordinates using precise RTK-GNSS drone localization, camera pose data, and a digital elevation model. 

Building upon the DeepOcSort algorithm implemented as part of the [BoxMot Framework](https://github.com/mikel-brostrom/boxmot), two new tracking strategies are proposed
- Geo-Native Tracker: performs tracking entirely in world coordinates, enabling physically meaningful motion modeling and removing the need for camera-motion compensation.
- Hybrid Tracker: augments standard image-space association with an additional geo-referenced matching stage to recover identities during severe occlusions or ambiguous visual conditions.

## Usage

For replication of our results: Clone the BoxMot Framework and include the two implementations (geonative.py and hybrid.py) to a new Python directory in "boxmot/trackers/geo".

Use our [conversion script](https://github.com/bambi-eco/bambi_detection/blob/main/src/bambi/georeference_deepsort_mot.py) to convert local bounding boxes to global geo-referenced bounding boxes or get examples from our [replication package]() and use them as shown below:

```python
import numpy as np
import cv2
from pathlib import Path

# Example tracker imports (adapt to your project)
from boxmot.trackers.geo.geonative import GeoHybridDeepOcSort
from boxmot.trackers.geo.geonative import GeoNativeDeepOcSort

# --- paths to one set of files ----------------------------------------------------------------------------------
dets_path     = Path("/path/to/one/sequence/dets/seq01.txt")
geo_dets_path = Path("/path/to/one/sequence/geodets_new/seq01.txt")
img_path = Path("/path/to/one/sequence/images/00000001.jpg")
# optionally also use custom embeddings
embs_path     = Path("/path/to/one/sequence/embs/seq01.txt")

# --- load files -------------------------------------------------------------------------------------------------
# dets: frame_id, x, y, w, h, score, class_id
dets = np.atleast_2d(np.loadtxt(dets_path, skiprows=1))
# geo_dets: line_id_of_local_bounding_box, frame_id, min_x, min_y, min_z, max_x, max_y, max_z, score, class_id ... 
geo_dets = np.atleast_2d(np.loadtxt(geo_dets_path, skiprows=1))

# embs: embedding per detection
if embs_path.exists():
    embs = np.atleast_2d(np.loadtxt(embs_path))
else:
    embs = None

# --- prepare combined array so you can iterate frame by frame with all associated data ---------------------------
# order: [dets | geo_dets | embs]
dets_n_geo_n_embs = np.concatenate((dets, geo_dets, embs), axis=1)


# --- pick one frame and slice its data --------------------------------------------------------------------------
frame_num = 1
frame_num_padded = f"{frame_num:08d}"

# select rows belonging to this frame
frame_rows = dets_n_geo_n_embs[dets_n_geo_n_embs[:, 0] == int(frame_num_padded)]

# same column layout as in generate_mot_results()
dets_frame     = frame_rows[:, 1:7]    # [x, y, w, h, score, class_id]
geo_dets_frame = frame_rows[:, 7:17]   # [frame_idx, geo_x, geo_y, ...] (10 cols)
embs_frame     = frame_rows[:, 17:]    # embeddings

# --- load image ---------------------------------------------------------------------------------------------------
if img_path.exists():
    img = cv2.imread(str(img_path))
else:
    img = None


# --- init tracker -------------------------------------------------------------------------------------------------
# choose ONE of the two:
tracker = GeoHybridDeepOcSort()
# tracker = GeoNativeDeepOcSort()


# --- call tracker.update with all parameters ----------------------------------------------------------------------
tracks = tracker.update(
    dets=dets_frame,
    img=img,
    index=frame_num,   
    embs=embs_frame,
    geodets=geo_dets_frame
)
```

## Cite

This package is part of a publication that is currently submitted for publication:

```latex
@Article{praschlGeoReferencedTracking,
      author       = {Praschl, Christoph and Coucke, Vincent and Maschek, Anna and Schedl, David},
      title        = {Stay Local or Go Global: Geo‐Referenced Bounding Boxes for
Tracking Wildlife in Thermal Drone Videos},
}
```
