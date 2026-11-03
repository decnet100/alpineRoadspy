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

## Preparing a video

```bash
python tools/prepare_video.py videos/samnaun.webm
```

Transcodes to H.264 with constant frame rate and no audio:
`<name>_work.mp4` (1080 lines) for detection, tracking and the fix picker,
and – if the source is larger – `<name>_full.mp4` at full resolution, from
which `track.py` and `signs.py` cut their crops automatically. Both copies
come from one ffmpeg run, so frame numbers match. H.264 rather than
H.265/VP9/AV1: it decodes fastest and plays in every browser. Uses NVENC if
available. Run all further steps on the `_work.mp4`.

It also looks for cuts (the video jumps, e.g. a skipped tunnel) and writes
`<name>_work_cuts.csv`; for an already prepared video use
`python tools/prepare_video.py videos/samnaun_work.mp4 --cuts-only`. Cuts are
found by a brightness-independent comparison of consecutive frames (tunnel
portals and exposure changes do not count, fades through black do). The fix
picker marks them on the time bar, does not interpolate or extrapolate across
them and asks for a fix shortly before and after each cut; track.py does not
track across them. Cuts the detection missed (soft cross-fades) or detected wrongly
can be set or removed in the fix picker with `X` at the current position;
those edits are kept in `data/cuts/<video>.json` and used by track.py too.

`prepare_video.py` also writes `<name>_work_motion.csv`: camera yaw, pitch
and roll per frame from the image motion (corners above the bonnet tracked
into the next frame, rotation + shift fitted robustly). The fix picker shows
the measured yaw rate together with the heading rate expected from the
routed road between the fixes (constant speed per section) in a chart below
the video; the scale image width -> degrees is fitted from the data. A click
in the chart jumps in the video. On the Samnaun drive both curves agree
hairpin by hairpin (r = 0.75); the remaining differences – sharper expected
peaks in hairpins where the car really slows down – are what later steps can
use to estimate the speed. The camera's lateral position can be taken from
the clicks or fixed to the right/left lane or the centre (overtaking, closed
lanes).

With "Geschwindigkeit aus Kurven" (on by default when motion data exist)
the position between two fixes no longer assumes constant speed:
`tools/speedfit.js` aligns the road heading along the section with the
heading the camera turned through (dynamic programming over time x station,
speed 0 to 110 km/h (adjustable "max. km/h", raised per section if its mean
speed needs more), penalty on deviating from the mean speed, so straights stay
even). Cross-validated on the Samnaun drive by leaving out fixes: with a fix
every ~25 s the position error at the left-out fixes drops from median 10 m /
90 % 27 m to 8 m / 20 m, with a fix every ~37 s from 14 m / 38 m to 9 m /
18 m.

For helmet-camera videos the camera's turning is spoiled by head movements.
If `data/audio/<video>/track.csv` (engine-sound speed, `audio_speed.py
--fixes`) and `<video>_cockpit.csv` (`cockpit.py`) exist, the fix picker loads
them: "Drehung aus: Schräglage + Ton" uses the vehicle's turn rate g x
tan(lean) / speed instead (in degrees, no scale to fit), and "Geschwindigkeit
aus Ton" makes the engine-sound speed the reference between the fixes (and
moves the position on after the last fix with it). A second chart shows the
engine-sound speed next to the trajectory's, and the cockpit parts found are
outlined over the video (red where uncertain) with the lean angle.
Cross-validated on the Samnaun 2 ride by leaving out fixes, with the
engine-sound gears fitted without them: position error at a left-out fix
(fixes every ~20-40 s) median 25 m -> 5 m (90 %: 75 -> 22 m); with three
fixes left out 34 -> 10 m (90 %: 124 -> 22 m).

## Speed from the engine sound

```bash
python tools/audio_speed.py videos/samnaun2.mp4 --cylinders 3 --fixes data/samnaun2_fixes.csv
```

Uses the original video (the `_work.mp4` copies have no audio). It tracks the
engine's firing frequency f0 (rpm x cylinders / 120 for a four-stroke) and a
load indicator (share of the upper harmonics, which drops when the throttle
closes), splits the drive into steady segments (no pitch jumps, no abrupt load
changes) and lists what happens in the gaps between them: upshifts (pitch falls
by the gear step), downshifts (rises by it; while braking by less, because the
speed falls at the same time) and coasting with closed throttle.

The speed is k_gear x f0 with one constant per gear. With the fixes
(`--fixes`, the fix picker's CSV export) the gears are fitted without any
speed reading: they form a ladder with the step measured at the upshifts
(six gears around the typical one); the gear per moment is chosen so that
the speed stays continuous – a pitch jump by the gear step forces a gear
change at that moment, a gear change without one is expensive. The fixes
anchor it by distance, not by mean speed: per fix interval (of 10 s or more)
a cost per metre is raised or lowered until the integrated speed matches the
distance along the route, so a stretch ridden at 41-92 km/h keeps its
course; the base constant is chosen the same way over all intervals.
Stretches tracked at twice the pitch (slow hairpins) are halved, short pitch
and speed outliers bridged; speed jumps that are left (more than 8 km/h in
0.5 s) are reported. Alternatively `--readings` (CSV `time_s,speed_kmh`,
e.g. read off the speedometer; `--lag` for its delay, 0.7 s) fits the gears
from readings. Results in `data/audio/<video>/` (`track.csv`,
`segments.csv`, `events.csv`, `gears.csv`, `plot.png`); the fix picker
loads `track.csv` automatically.

Where the sound alone does not tell the gear – shifts with the clutch pulled
while braking leave no jump – the fixes decide: within each fix interval the
sure stretches (steady segments) are summed up, and the rest of the distance
has to come from the unclear stretches between them. Their gear and speed at
both ends are known; within, the sound is followed with the gear before, then
ignored for a while (clutch pulled; the speed joined smoothly within
plausible braking/acceleration), then followed with the gear after – after a
coasting phase that gear is open too, for the sure stretch that follows.
Time and length of the clutch window and the gear are chosen so that the
interval's distance is met (column `clutch` in `track.csv`).

Shifts are often audible as a short metallic click (the lever, with the
clutch pulled). The tool reads the sound at 44.1 kHz for that and looks for
abrupt broadband rises between 1.5 and 11 kHz, where the engine is low and
steady (above ~12 kHz the compressed sound tracks of online videos are cut
off). A strong click opens a stretch for the check above even where the
pitch is steady, and a gear change chosen there must happen at the click.
A weak click is used only where the speed jumps next to it, so that a shift
there could remove the jump (also accepting a pitch change of about a gear
step across it picked up smooth acceleration out of hairpins and made the
cross-validation worse).
On the Samnaun 2 ride all 9 strong clicks were shifts, including the one
hidden by the clutch while braking in a tunnel; upshifts in the loud tunnel
were not heard.

On the first 6½ minutes of the Samnaun 2 ride (Triumph Tiger 1050) the step
is 0.784 and the speed from the fixes agrees with 63 speedometer readings
(corrected by the speedometer's 6 % over-reading) within median 0.8 km/h,
90 % 4.9 km/h, max 11.5 km/h – including braking in a tunnel, where the
hidden downshifts (gear 5 -> 4 -> 3, each at its click) are found from the
distance. The tool also counts the speed jumps it produces (a motorcycle does
not do them): here 2 of more than 8 km/h in 0.5 s (the two braking clutch
windows), none beyond 8 m/s² braking or 5 m/s² accelerating – the first
version of this analysis had 18 and 16.

## Helmet cameras: cockpit and lean angle

```bash
python tools/cockpit.py videos/samnaun2.mp4 --grid 130
python tools/cockpit.py videos/samnaun2.mp4
```

Work in progress. With a helmet camera the image turns with the head too.
Parts of the cockpit that are rigid with the frame (instrument cluster,
fairing, windscreen and its clamps – not mirrors, phone holder or switches on
the handlebar, which steer) are noted once per video as boxes/polygons in
the pixels of a reference frame: `--grid T` writes that frame with a pixel
grid to `data/cockpit/<video>_ref.jpg`, the parts go into
`data/cockpit/<video>.json` (example: `docs/cockpit_example.json`). The
tool then finds the cockpit in 10 frames per second by matching the edge
directions of those parts (robust to shade, backlight and tunnels, also when
partly out of the image; only where the cockpit can plausibly be, with a
fresh search every second so that a lock on rock or guardrail texture does
not persist) and writes `<video>_cockpit.csv`: cockpit position and rotation in
the image, and the lean angle of the vehicle = roll of the scene above the
cockpit minus roll of the cockpit.

On the Samnaun 2 ride the lean agrees with the lean expected from the
trajectory (v x turn rate / g) with r = 0.80 in the first three minutes and
0.70 over the 6½ minutes with fixes (where fixes are sparse the expected lean
itself is less certain), with the right magnitude and without calibration. Speed from lean and road curvature is right on average but
scatters by about ±9 km/h, far more than the engine sound; the lean is more
useful for timing curves. Subtracting the head's turning (from the cockpit's
sideways shift) did not improve the comparison of camera turning with the
road – leaning and head movements also shift the near cockpit – so it is not
applied. About 1 s per video second and CPU core for 4K input; the video is
split into one-minute parts processed in parallel (`--workers`, default: all
cores but one).

## Reviewing results

```bash
python tools/serve.py
```

Open `http://localhost:8000/tools/review.html?video=/videos/samnaun_work.mp4`
and click through everything found for that video: tunnels (lining: concrete,
rock, shotcrete, mixed), guardrail and barrier sections, poles, lights and
signs (code from the catalog suggestions, value, text exactly as printed,
names per language, destinations). The video jumps to each item; sample
frames or crops are shown next to it. Keyboard: digits choose, Enter confirms
and moves on, arrow keys navigate, space plays. Decisions are saved
automatically to `data/review/<video>.json`.

Tunnel sections come from the segmentation (class "Tunnel", detect.py column
`tunnel_px`) and are listed by track.py together with guardrails/barriers in
`sections.csv`, with four sample frames each.

## Traffic signs

`catalog/signs.csv` lists the signs of the Austrian StVO (§§ 50, 52–54) and
the Swiss SSV (appendix 2) with OpenStreetMap-style codes (`AT:52.10a`,
`CH:2.30`), official name, legal reference and a public-domain reference
drawing; rebuild it with `tools/build_sign_catalog.py`.

`tools/signs.py` cuts the best full-resolution views of every sign object and
ranks catalog signs by image similarity; with `--vlm <model>` a local
vision-language model also reads value, text lines and destinations.
Work in progress: the catalog matching is not reliable yet.

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
| AT + CH (border crossing) | GIP (or Tirol WFS) and swissTLM3D together; loose road ends of the two datasets are joined within 25 m |

With several roads within the snap radius (parallel roads, overlapping
datasets at the border) the routing tries the nearest three per fix and takes
the shortest plausible drive; a road chosen by hand (key `R`) is kept.
On the Swiss side use the swisstopo or Esri imagery – basemap.at ends at the
border.

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

Frames are scaled to `--work-height` lines (default 720, aspect ratio kept)
before segmentation; the model's own default of 384 x 384 misses distant
delineator posts. 4K footage is reduced right after decoding, so it costs
little more than 720p; a higher `--work-height` (e.g. 1080) can find even
smaller objects at roughly proportional extra time. Coordinates in the
results are always in pixels of the original video.

Long videos can be processed in parts, e.g. Swiss and Austrian section:

```bash
python tools/detect.py videos/long.mp4 --step 3 --end 41:30
python tools/detect.py videos/long.mp4 --step 3 --start 41:30
```

Times as seconds, `mm:ss` or `h:mm:ss`. `--index-only` rebuilds the overview page
of an existing run without detecting again. Each part goes to its own folder
`data/detect/<video>/part_<start>-<end>/`; `track.py` combines all parts
automatically (overlapping frames are counted once).

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
side). Guardrails and barriers (walls, concrete barriers, rock faces) are kept
apart: a barrier stretch counts as guardrail only if it mostly overlaps a
guardrail section or fills a gap of up to 30 m between two (the model
sometimes calls guardrails in curves "Barrier"). Sections are written to
`sections.csv`; barriers are brown on the map.

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

Pole candidates get a plausibility check: delineator posts are slender and
are passed closely, so candidates with a rotation-invariant elongation below
`--pole-min-elongation` (default 3) or never taller than `--pole-min-height`
(default 6 % of the image height) are kept but marked `uncertain` (column
`status`, reason in `reasons`; hollow on the map, dashed in `index.html`).
Detections from older detect.py runs only have bounding boxes, so the
elongation falls back to the box aspect (threshold 1.5).

Not done yet: sign type classification, separating delineator posts from other
slender posts (tree trunks, guardrail posts in shade – brightness did not
separate them reliably; a trained classifier is the next step), compensating
camera roll, measuring the lateral distance per object instead of assuming it.

## Layout

```
docs/               concept and design notes
tools/              browser tools and local server
src/alpineroadspy/  Python package
videos/, frames/,   local footage, extracted images, results —
data/               git-ignored, never committed (public repo)
```
