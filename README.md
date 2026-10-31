# alpineRoadspy

used to detect road features semi-automatically from video feeds of alpine roads

## Goal

From a continuous video drive along a mountain road, produce an inventory of
roadside features — traffic signs, delineator posts (Leitpfosten), guardrails
(Leitschienen) and similar — each with a position along the road
(chainage/station), side (left/right) and map coordinate.

Constraints that shape the approach:

- **No GNSS track** in the video. Only sparse manual position fixes
  (start/end, roughly every 60 s, ±5 m) are available.
- **Road geometry is a weak prior**: the reference centerline of mountain
  roads is often inaccurate or incomplete, so it cannot be used as a hard
  constraint.
- **Everything runs locally**; no footage is uploaded to third-party services.

See [docs/concept.md](docs/concept.md) for the approach.

## Status

Concept phase. Available so far: a browser tool for setting position fixes.

## Setting position fixes

`tools/fix_picker.html` shows the video next to a map. Pause at a
recognisable spot (hairpin, bridge, km post), click the map, and a fix with
the current video time is stored.

Each fix is snapped to the nearest road within the snap radius. Between two
fixes the shortest path through the road network is used — also across road
changes (car park → ramp → B road → L road) — and the position is
interpolated along it at constant speed, including the lateral offset of the
clicked points. After the last fix the position keeps moving along the road
at the last measured speed (orange); as long as it stays in sync with the
video, no further fix is needed. Implausible speeds are flagged in the table.

### Road network

| Source | Use |
|---|---|
| **GIP** (local file, default) | routable link network of Austria, split at every junction |
| Tirol Verkehrswege (WFS, Land Tirol) | fallback without local file, Tyrol only |
| swissTLM3D (swisstopo) | Switzerland |

Prepare the GIP once: download `B_gip_network_ogd.zip` (about 4 GB) from
<https://open.gip.gv.at/ogd/>, unzip it and extract the link network, e.g.
clipped to Tyrol:

```bash
python tools/prepare_gip.py path/to/gip_network_ogd.gpkg --bbox 10.0,46.6,13.1,47.8
```

This writes `data/roads/gip_linknetz.gpkg`, which `serve.py` picks up
automatically (any other line file: `--roads path`). Needs `geopandas`.
Both the documented GIP schema and the renamed one of the 2026 exports
(layer `LINK`, `node_from_short_id`, ...) are understood, see `tools/gip_schema.py`.

### Running

```bash
python tools/serve.py videos/brenner.mp4
```

Then open the printed URL (`http://localhost:8000/tools/fix_picker.html?video=/videos/brenner.mp4`).
The server only binds to localhost and supports seeking in the video; the
video never leaves the machine. A video can also be opened via the file
dialog or drag & drop. Map tiles and the WFS fallback need internet access.

### Exports (WGS84)

- `<video>_fixes.csv` — the fixes:
  `video_time_s,frame,lon,lat,sigma_m,note,station_m,offset_m,road`
- `<video>_fixes.geojson` — fixes plus the routed path between them
- `<video>_trajectory.csv` — position for every frame:
  `frame,video_time_s,lon,lat,station_m,heading_deg,speed_kmh,mode,offset_m`
  (`offset_m`: camera position relative to the road axis, left positive)

## Detecting roadside features (first pass)

`tools/detect.py` runs a Mapillary Vistas segmentation model (Mask2Former,
Swin-L) on every n-th frame and turns traffic signs, poles (delineator posts
show up as "Pole"), street lights and guardrails into regions with bounding
boxes. With the per-frame trajectory from the fix picker every region gets
station and position.

Setup on a machine with an NVIDIA GPU (pick the CUDA build matching the
driver on <https://pytorch.org>):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers opencv-python
```

```bash
python tools/detect.py videos/brenner.mp4 --step 3 --trajectory data/brenner_trajectory.csv
```

Results in `data/detect/<video>/`: `detections.csv`, `guardrails.csv`
(guardrail pixels left/right per frame), overlay images and `index.html`, a
contact sheet that can be filtered by class; via `serve.py`
(`http://localhost:8000/data/detect/brenner/index.html`) a click opens the
frame in the fix picker. The model (about 0.9 GB) is downloaded once from
Hugging Face; Mapillary Vistas models are licensed for non-commercial use.

### Objects

`tools/track.py` links the per-frame regions into individual objects and
places each one at the moment the camera passes it (apparent size grows
inversely with distance: the zero of a line through 1/size over time). Station
and coordinates come from the trajectory, the side from the direction in which
the object leaves the image; guardrails become sections (station from–to,
side).

```bash
python tools/track.py videos/brenner.mp4 --trajectory data/brenner_trajectory.csv
```

Results in `data/objects/<video>/`: `objects.csv`, `objects.geojson` (load it
in the fix picker via "Referenz/Objekte…"; a click on an object jumps to the
video shortly before passing it) and `index.html` with a thumbnail per object.

Objects are placed beside the road axis: half the carriageway (`--road-width`,
default 6.5 m) plus an assumed distance from the edge per object type, on both
sides alike. The camera's offset from the axis comes from the trajectory
(`offset_m`); older trajectory exports without it assume the middle of the
right lane.

Not done yet: sign type classification, telling delineator posts apart from
other posts, compensating camera roll, measuring the lateral distance per
object instead of assuming it.

## Layout

```
docs/               concept and design notes
tools/              browser tools and local server
src/alpineroadspy/  Python package
videos/, frames/,   local footage, extracted images, results —
data/               git-ignored, never committed (public repo)
```
