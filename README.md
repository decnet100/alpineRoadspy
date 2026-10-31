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
the current video time is stored. Export as CSV or GeoJSON.

```bash
python tools/serve.py videos/brenner.mp4
```

Then open the printed URL (`http://localhost:8000/tools/fix_picker.html?video=/videos/brenner.mp4`).
The server only binds to localhost and supports seeking in the video; the
video never leaves the machine. A video can also be opened via the file
dialog or drag & drop. The map tiles (basemap.at, Tirol orthophoto,
swisstopo, OSM, OpenTopoMap, Esri imagery) need internet access.

Export format (`<video>_fixes.csv`, WGS84):

```
video_time_s,frame,lon,lat,sigma_m,note
12.345,370,11.3912345,47.2201234,5,Kehre 1
```

## Layout

```
docs/               concept and design notes
tools/              browser tools and local server
src/alpineroadspy/  Python package
videos/, frames/,   local footage, extracted images, results —
data/               git-ignored, never committed (public repo)
```
