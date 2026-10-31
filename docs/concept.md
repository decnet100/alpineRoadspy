# Concept

## Problem

Two sub-problems:

1. **Where is the camera?** — position (and heading) of the camera for every
   frame, without GNSS.
2. **What is beside the road?** — detect and track roadside features and
   place them relative to the camera trajectory.

The vehicle stays on the road, so (1) reduces mostly to a one-dimensional
question: *how far along the road am I at time t?* plus a small lateral
offset (lane, ±2–3 m).

## Inputs

| Input | Format | Notes |
|---|---|---|
| Video | MP4/MOV | single forward-facing camera, fixed mount |
| Position fixes | CSV: `video_time_s, x, y, sigma_m` | manual, sparse (start/end, ~60 s), ±5 m |
| Reference centerline | GeoPackage/GeoJSON/GPX line | optional, may be inaccurate or have gaps |
| Terrain model | GeoTIFF (ALS DTM, e.g. 1 m Austria) | optional, for height profile and centerline refinement |
| Camera calibration | YAML (intrinsics, distortion) | one-off checkerboard calibration |

## Part 1 — camera localisation

### Visual odometry as distance meter

Monocular visual odometry (VO) provides relative motion up to an unknown
scale. The position fixes supply that scale: the VO path length between two
consecutive fixes is rescaled to the distance between them. Over ~60 s the
scale drift is small, so the error between fixes is expected in the range of
a few metres up to ~10 m — compared to 50–150 m for plain linear
interpolation in time, which breaks down at stops, slow sections and
hairpins.

Candidates: ORB-SLAM3 / stella_vslam (monocular), or a lighter custom
frame-to-frame VO with OpenCV; learned matchers (SuperPoint + LightGlue)
for robustness on low-texture rock and vegetation.

### Shape matching against the road (mountain-road specific)

Mountain roads are curvy, which is an advantage: the heading change from VO
(hairpins, curve sequences) is a strong signature that can be aligned to the
curvature profile of the centerline (DTW-style alignment). Likewise the VO
pitch / grade profile can be aligned to the height profile from the DTM.

Because the reference centerline is unreliable, it enters only as a soft
constraint. The fused trajectory can in turn **improve the centerline** — a
useful by-product. The ALS terrain model is a better geometric reference
than many vector datasets: road cuts and embankments are clearly visible
in the DTM even under forest cover.

### Fusion

All observations go into one least-squares / factor-graph problem
(GTSAM or Ceres, or scipy for a first version):

- VO relative motion (between frames)
- sparse position fixes (absolute, ±5 m)
- centerline proximity (soft, down-weighted where geometry is poor)
- DTM height profile (soft)
- re-observed anchors: km posts, bridges, junctions, tunnel portals

Delineator posts are placed at regular spacing on many roads and can act as
an additional odometer, but spacing varies in curves and posts may be
missing — use with care, mainly as a plausibility check.

## Part 2 — roadside features

| Feature | Method | Off-the-shelf? |
|---|---|---|
| Traffic signs | detection + type classifier (YOLO) | yes (MTSD / GTSRB-trained models) |
| Guardrails | segmentation, output as intervals (from–to station, side) | largely (Mapillary Vistas class "Guard Rail") |
| Delineator posts | detection | no — needs a few hundred labelled frames |
| Snow poles, rockfall nets, retaining walls | detection / segmentation | later, as needed |

Detections are tracked across frames (ByteTrack) so each object is counted
once. Each object is placed by its bearing from the camera plus a range
estimate — from known object size (signs and delineators are standardised)
or by triangulation over the frames in which it was tracked.

## Output

GeoPackage / CSV, one row per object:

`object_class, subtype, station_m, side, lateral_offset_m, x, y, confidence, first_frame, last_frame, thumbnail`

Guardrails as line features with `station_from_m, station_to_m, side`.
Additionally: the estimated camera trajectory per frame.

## Expected difficulties

- tunnels and galleries (darkness, no texture)
- strong light/shadow changes in forest, backlight, wet road
- low-texture rock walls, snow
- steep grades and tight hairpins (large rotation, little translation)
- rolling shutter on consumer cameras

## Next steps

1. Get a short sample drive (a few minutes) with 3–5 manual fixes.
2. Prototype VO + rescaling between fixes; check plausibility against
   hand-picked control points.
3. Run off-the-shelf sign and guardrail models on the same clip.
4. Label delineator posts and train a first detector.
