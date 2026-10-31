"""Turn per-frame detections into individual roadside objects with position.

Links the regions from tools/detect.py across frames (constant-velocity
prediction + Hungarian assignment), then places each object at the moment the
camera passes it: the apparent size of an object grows inversely with its
distance, so a straight line through 1/size over time hits zero when the
camera is level with it. Station and coordinates at that moment come from the
per-frame trajectory of the fix picker; the side from the direction in which
the object leaves the image. Objects are placed beside the road axis (half the
carriageway width plus an assumed distance from the edge per object type).

    python tools/track.py videos/brenner.mp4 --trajectory data/brenner_trajectory.csv

Reads data/detect/<video>/detections.csv and guardrails.csv (override with
--detections / --guardrails) and writes to data/objects/<video>/:
    objects.csv       one row per object
    objects.geojson   objects (points) and guardrail sections (lines), WGS84;
                      load it in the fix picker as reference layer to review
    index.html        one thumbnail per object, click opens the fix picker there
"""

import argparse
import csv
import html
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent

# detect.py class -> object group tracked together
GROUPS = {
    "pole": "pole", "utility_pole": "pole",
    "traffic_sign": "sign", "traffic_sign_back": "sign", "sign_frame": "sign",
    "street_light": "light",
}
# assumed distance from the edge of the carriageway; objects are placed from the road axis
EDGE_M = {"pole": 1.0, "sign": 1.5, "light": 1.5, "guardrail": 0.5}
MERGE_S = 0.6  # same group and side passed within this time: one object


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Trajectory:
    """Per-frame trajectory from fix_picker.html, interpolated by video time."""

    def __init__(self, path):
        rows = read_csv(path)
        self.t = np.array([float(r["video_time_s"]) for r in rows])
        self.lon = np.array([float(r["lon"]) for r in rows])
        self.lat = np.array([float(r["lat"]) for r in rows])
        self.station = np.array([float(r["station_m"] or "nan") for r in rows])
        self.heading = np.array([float(r["heading_deg"] or "nan") for r in rows])
        # camera offset from the road axis (left positive); older exports lack the column
        self.offset = np.array([float(r.get("offset_m") or "nan") for r in rows])
        self.has_offset = bool(np.isfinite(self.offset).any())

    def covers(self, t):
        return self.t[0] - 0.5 <= t <= self.t[-1] + 0.5

    def at(self, t):
        i = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        w = float(np.clip((t - self.t[i - 1]) / max(self.t[i] - self.t[i - 1], 1e-9), 0, 1))
        lerp = lambda a: a[i - 1] + w * (a[i] - a[i - 1])
        return {"lon": lerp(self.lon), "lat": lerp(self.lat), "station": lerp(self.station),
                "heading": self.heading[i if w > 0.5 else i - 1], "offset": lerp(self.offset)}


def offset(lon, lat, heading_deg, side, metres):
    """Point `metres` to the left/right of the driving direction."""
    ang = math.radians(heading_deg + (90 if side == "right" else -90))
    dlat = metres * math.cos(ang) / 111320
    dlon = metres * math.sin(ang) / (111320 * math.cos(math.radians(lat)))
    return lon + dlon, lat + dlat


def place(p, side, edge_m, road_width):
    """Coordinates of an object beside the road axis: half the carriageway plus `edge_m`.

    The trajectory point is the camera; its offset from the axis comes from the
    trajectory or, if unknown, is assumed to be the middle of the right lane.
    Returns lon, lat, distance from the axis.
    """
    from_axis = road_width / 2 + edge_m
    target = from_axis if side == "left" else -from_axis  # left of the axis positive
    camera = p["offset"] if np.isfinite(p["offset"]) else -road_width / 4
    shift = target - camera
    lon, lat = offset(p["lon"], p["lat"], p["heading"], "left" if shift > 0 else "right", abs(shift))
    return lon, lat, from_axis


def track(dets, step, max_gap):
    """Greedy-optimal frame-to-frame association within one group. Returns lists of detections."""
    by_frame = {}
    for d in dets:
        by_frame.setdefault(d["frame"], []).append(d)
    active, done = [], []
    for frame in sorted(by_frame):
        cur = by_frame[frame]
        alive = [tr for tr in active if frame - tr[-1]["frame"] <= max_gap * step]
        done += [tr for tr in active if tr not in alive]
        active = alive
        cost = np.full((len(active), len(cur)), 1e6)
        for i, tr in enumerate(active):
            last = tr[-1]
            dt = (frame - last["frame"]) / step
            vx, vy = (last["cx"] - tr[-2]["cx"], last["cy"] - tr[-2]["cy"]) if len(tr) > 1 else (0.0, 0.0)
            if len(tr) > 1:
                gap = (last["frame"] - tr[-2]["frame"]) / step
                vx, vy = vx / gap, vy / gap
            px, py = last["cx"] + vx * dt, last["cy"] + vy * dt
            gate = (20 + 1.5 * last["size"] + 0.5 * math.hypot(vx, vy)) * dt + (40 if len(tr) == 1 else 0)
            for j, d in enumerate(cur):
                dist = math.hypot(d["cx"] - px, d["cy"] - py)
                growth = d["size"] / max(last["size"], 1)
                if dist <= gate and 0.5 <= growth <= 3:  # approaching objects grow, never shrink much
                    cost[i, j] = dist / gate
        matched = set()
        if active and cur:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1e6:
                    active[i].append(cur[j])
                    matched.add(j)
        active += [[d] for j, d in enumerate(cur) if j not in matched]
    return done + active


def pass_time(tr):
    """Time the camera passes the object: zero of a line through 1/size over time."""
    t = np.array([d["t"] for d in tr])
    inv = 1 / np.array([d["size"] for d in tr])
    t_last = t[-1]
    if len(tr) >= 3 and np.ptp(t) > 0:
        b, a = np.polyfit(t, inv, 1)
        if b < 0:
            t0 = -a / b
            return float(np.clip(t0, t_last, t_last + 3.0)), True
    return t_last + 0.3, False


def side_of(tr, width):
    dx = tr[-1]["cx"] - tr[0]["cx"]
    if abs(dx) > 20:
        return "right" if dx > 0 else "left"
    return "right" if tr[-1]["cx"] > width / 2 else "left"


def guardrail_sections(rows, traj, min_px, min_len, max_gap_m):
    """Sections (station from/to, side) where a guardrail is seen in the near field."""
    near = "left_near_px" in rows[0]
    sections = []
    for side in ("left", "right"):
        col = f"{side}_near_px" if near else f"{side}_px"
        pts = []
        for r in rows:
            t = float(r["video_time_s"])
            if traj.covers(t):
                pts.append((traj.at(t)["station"], float(r[col]) >= min_px))
        pts.sort()
        flags = np.array([f for _, f in pts], dtype=float)
        if len(flags) >= 5:  # median filter against single-frame dropouts
            flags = np.array([np.median(flags[max(0, i - 2): i + 3]) for i in range(len(flags))])
        start = end = None
        for (st, _), f in zip(pts, flags):
            if f >= 0.5:
                if start is not None and st - end > max_gap_m:
                    sections.append((start, end, side))
                    start = None
                start = st if start is None else start
                end = st
        if start is not None:
            sections.append((start, end, side))
    return [s for s in sections if s[1] - s[0] >= min_len], near


def thumbnails(video, objects, out_dir):
    import cv2

    cap = cv2.VideoCapture(str(video))
    (out_dir / "thumbs").mkdir(exist_ok=True)
    for o in sorted(objects, key=lambda o: o["best"]["frame"]):
        d = o["best"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, d["frame"])
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        bw, bh = d["x2"] - d["x1"], d["y2"] - d["y1"]
        pad = max(bw, bh, 40)
        x1, y1 = max(0, int(d["x1"] - pad)), max(0, int(d["y1"] - pad))
        x2, y2 = min(w, int(d["x2"] + pad)), min(h, int(d["y2"] + pad))
        crop = img[y1:y2, x1:x2].copy()
        cv2.rectangle(crop, (int(d["x1"] - x1), int(d["y1"] - y1)), (int(d["x2"] - x1), int(d["y2"] - y1)), (255, 0, 255), 1)
        scale = 160 / crop.shape[0]
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 160))
        cv2.imwrite(str(out_dir / "thumbs" / f"{o['id']}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])


REASON_TEXT = {"not_slender": "nicht schlank", "never_near": "nie nah passiert"}


def image_height(video):
    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        return int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    except Exception:
        return None


def write_index(out_dir, video_url, objects, sections, has_thumbs):
    picker = Path(os.path.relpath(ROOT / "tools" / "fix_picker.html", out_dir)).as_posix()

    def cell(o):
        link = f"{picker}?video={html.escape(video_url)}&amp;t={max(0, o['t_pass'] - 1.5):.2f}" if video_url else "#"
        img = f'<img loading="lazy" src="thumbs/{o["id"]}.jpg">' if has_thumbs else ""
        why = f'<br><i>unsicher: {", ".join(REASON_TEXT[r] for r in o["reasons"])}</i>' if o["reasons"] else ""
        return (f'<a class="cell g-{o["group"]} s-{o["status"]}" href="{link}" target="_blank">{img}<div><b>{o["id"]}</b> {o["class"]}<br>'
                f'Station {o["station"]:.0f} m · {o["side"]} · {o["hits"]}× · {o["t_pass"]:.1f} s{why}</div></a>')

    rails = "".join(f"<li>{s:.0f}–{e:.0f} m {side}</li>" for s, e, side in sections)
    page = f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><title>Objekte</title>
<style>
body {{ font: 13px system-ui, sans-serif; margin: 10px; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.cell {{ color: inherit; text-decoration: none; border: 1px solid #ddd; width: 170px; }}
.cell img {{ display: block; max-width: 170px; height: 120px; object-fit: contain; background: #000; margin: auto; }}
.cell div {{ padding: 3px 5px; font-size: 12px; }}
.g-sign {{ border-color: #e6b800; }} .g-light {{ border-color: #00b3b3; }}
.s-uncertain {{ opacity: .5; border-style: dashed; }}
.hide {{ display: none; }}
</style></head><body>
<h3>{len(objects)} Objekte, davon {sum(o["status"] == "uncertain" for o in objects)} unsicher</h3>
<p>Zeigen: <label><input type="checkbox" value="pole" checked> Pfosten</label>
<label><input type="checkbox" value="sign" checked> Schilder</label>
<label><input type="checkbox" value="light" checked> Laternen</label>
· <label><input type="checkbox" value="uncertain" checked> unsichere</label>
· Klick öffnet die Stelle 1,5 s vor dem Passieren im Fixpunkt-Tool.</p>
<div class="grid">{''.join(cell(o) for o in objects)}</div>
<h4>Leitschienen</h4><ul>{rails or '<li>keine</li>'}</ul>
<script>
const boxes = [...document.querySelectorAll('input')];
boxes.forEach((b) => b.onchange = () => {{
  const on = Object.fromEntries(boxes.map((x) => [x.value, x.checked]));
  document.querySelectorAll('.cell').forEach((c) => {{
    const group = [...c.classList].find((k) => k.startsWith('g-')).slice(2);
    c.classList.toggle('hide', !on[group] || (c.classList.contains('s-uncertain') && !on.uncertain));
  }});
}});
</script></body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", help="the video (for thumbnails and links)")
    parser.add_argument("--trajectory", required=True, help="per-frame trajectory CSV from fix_picker.html")
    parser.add_argument("--detections", type=Path, help="default data/detect/<video>/detections.csv")
    parser.add_argument("--guardrails", type=Path, help="default data/detect/<video>/guardrails.csv")
    parser.add_argument("--out", type=Path, help="default data/objects/<video>")
    parser.add_argument("--min-hits", type=int, default=3, help="minimum detections per object (default 3)")
    parser.add_argument("--max-gap", type=int, default=2, help="evaluated frames an object may be missed (default 2)")
    parser.add_argument("--rail-px", type=int, default=1500, help="guardrail pixels per image side counted as present (near field: a third)")
    parser.add_argument("--road-width", type=float, default=6.5, help="carriageway width in m (default 6.5)")
    parser.add_argument("--pole-min-elongation", type=float, default=3.0,
                        help="poles less elongated than this (major/minor axis) are uncertain (default 3)")
    parser.add_argument("--pole-min-height", type=float, default=0.06,
                        help="poles never taller than this share of the image height are uncertain (default 0.06)")
    parser.add_argument("--no-thumbs", action="store_true")
    args = parser.parse_args()

    video = Path(args.video)
    det_path = args.detections or ROOT / "data" / "detect" / video.stem / "detections.csv"
    rail_path = args.guardrails or det_path.with_name("guardrails.csv")
    out_dir = args.out or ROOT / "data" / "objects" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = Trajectory(args.trajectory)

    dets = []
    for r in read_csv(det_path):
        if r["class"] not in GROUPS:
            continue
        x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        dets.append({"frame": int(r["frame"]), "t": float(r["video_time_s"]), "class": r["class"],
                     "group": GROUPS[r["class"]], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "cx": float(r["cx"]), "cy": float(r["cy"]), "size": math.hypot(x2 - x1 + 1, y2 - y1 + 1),
                     # rotation-invariant elongation from detect.py; older files: bounding box aspect
                     "elongation": float(r["elongation"]) if r.get("elongation") else
                                   max(x2 - x1 + 1, y2 - y1 + 1) / min(x2 - x1 + 1, y2 - y1 + 1)})
    frames = sorted({d["frame"] for d in dets})
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
    width = max(d["x2"] for d in dets) + 1
    height = image_height(video) or max(d["y2"] for d in dets) + 1
    has_elongation = "elongation" in read_csv(det_path)[0]
    min_elongation = args.pole_min_elongation if has_elongation else 1.5

    objects = []
    for group in sorted({d["group"] for d in dets}):
        for tr in track([d for d in dets if d["group"] == group], step, args.max_gap):
            if len(tr) < args.min_hits:
                continue
            t_pass, fitted = pass_time(tr)
            if not traj.covers(t_pass):
                continue
            largest = sorted(tr, key=lambda d: d["size"])[-3:]
            objects.append({"group": group, "class": Counter(d["class"] for d in tr).most_common(1)[0][0], "hits": len(tr),
                            "t_first": tr[0]["t"], "t_last": tr[-1]["t"], "t_pass": t_pass, "fitted": fitted,
                            "side": side_of(tr, width), "best": largest[-1],
                            "max_height": max(d["y2"] - d["y1"] + 1 for d in tr),
                            "elongation": float(np.median([d["elongation"] for d in largest]))})

    # fragments of one object (detection dropouts) pass at nearly the same time on the same side
    objects.sort(key=lambda o: (o["group"], o["side"], o["t_pass"]))
    merged = []
    for o in objects:
        prev = merged[-1] if merged else None
        if prev and prev["group"] == o["group"] and prev["side"] == o["side"] and o["t_pass"] - prev["t_pass"] < MERGE_S:
            keep = prev if prev["hits"] >= o["hits"] else o
            keep["hits"] = prev["hits"] + o["hits"]
            keep["best"] = max(prev["best"], o["best"], key=lambda d: d["size"])
            keep["max_height"] = max(prev["max_height"], o["max_height"])
            merged[-1] = keep
        else:
            merged.append(o)
    objects = sorted(merged, key=lambda o: o["t_pass"])

    # plausibility of pole candidates: delineator posts are slender and are passed closely
    for o in objects:
        reasons = []
        if o["group"] == "pole":
            if o["elongation"] < min_elongation:
                reasons.append("not_slender")
            if o["max_height"] < args.pole_min_height * height:
                reasons.append("never_near")
        o["reasons"] = reasons
        o["status"] = "uncertain" if reasons else "ok"

    for i, o in enumerate(objects, 1):
        p = traj.at(o["t_pass"])
        o["id"] = f"{o['group'][0].upper()}{i:04d}"
        o["station"] = p["station"]
        o["lon"], o["lat"], o["axis_dist"] = place(p, o["side"], EDGE_M[o["group"]], args.road_width)

    if args.guardrails and not rail_path.exists():
        raise SystemExit(f"Guardrail file not found: {rail_path}")
    rail_rows = read_csv(rail_path) if rail_path.exists() else []
    near = bool(rail_rows) and "left_near_px" in rail_rows[0]
    rail_px = args.rail_px // 3 if near else args.rail_px
    sections, near = guardrail_sections(rail_rows, traj, rail_px, min_len=20, max_gap_m=15) if rail_rows else ([], False)

    with open(out_dir / "objects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "class", "status", "reasons", "station_m", "side", "lon", "lat", "axis_dist_m", "pass_time_s",
                    "first_seen_s", "last_seen_s", "hits", "pass_from_size", "max_height_px", "elongation"])
        for o in objects:
            w.writerow([o["id"], o["group"], o["class"], o["status"], " ".join(o["reasons"]), f"{o['station']:.1f}", o["side"], f"{o['lon']:.7f}",
                        f"{o['lat']:.7f}", f"{o['axis_dist']:.2f}", f"{o['t_pass']:.2f}", f"{o['t_first']:.2f}",
                        f"{o['t_last']:.2f}", o["hits"], int(o["fitted"]), o["max_height"], f"{o['elongation']:.1f}"])

    features = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(o["lon"], 7), round(o["lat"], 7)]},
                 "properties": {"id": o["id"], "kind": o["group"], "class": o["class"], "station_m": round(o["station"], 1),
                                "side": o["side"], "video_time_s": round(o["t_pass"], 2), "hits": o["hits"],
                                "status": o["status"], "reasons": " ".join(o["reasons"])}}
                for o in objects]
    for start, end, side in sections:
        sts = np.linspace(start, end, max(2, int((end - start) / 5) + 1))
        idx = np.interp(sts, traj.station[~np.isnan(traj.station)], np.flatnonzero(~np.isnan(traj.station)))
        coords = []
        for k in idx:
            p = traj.at(traj.t[int(round(k))])
            coords.append([round(c, 7) for c in place(p, side, EDGE_M["guardrail"], args.road_width)[:2]])
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                         "properties": {"kind": "guardrail", "side": side, "station_from_m": round(start, 1),
                                        "station_to_m": round(end, 1)}})
    (out_dir / "objects.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")

    has_thumbs = not args.no_thumbs and video.exists()
    if has_thumbs:
        thumbnails(video, objects, out_dir)
    try:
        video_url = "/" + video.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        video_url = None
    write_index(out_dir, video_url, objects, sections, has_thumbs)

    counts = {}
    for o in objects:
        counts[o["class"]] = counts.get(o["class"], 0) + 1
    print(f"{len(dets)} detections (step {step}) -> {len(objects)} objects: {counts}")
    flagged = Counter(r for o in objects for r in o["reasons"])
    print(f"Uncertain poles: {sum(o['status'] == 'uncertain' for o in objects)} of {sum(o['group'] == 'pole' for o in objects)} "
          f"({', '.join(f'{k}: {v}' for k, v in flagged.items())})"
          + ("" if has_elongation else " – shape from bounding boxes; rerun detect.py for rotation-invariant values"))
    if not rail_rows:
        print(f"No guardrails: {rail_path} not found")
    else:
        print(f"Guardrail sections from {rail_path} ({'near field' if near else 'whole image half – rerun detect.py for near-field values'}): "
              + (", ".join(f"{s:.0f}-{e:.0f} m {side}" for s, e, side in sections) or "none"))
    if not traj.has_offset:
        print("Note: trajectory has no offset_m column (export it again from the fix picker); "
              "camera assumed in the middle of the right lane")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
