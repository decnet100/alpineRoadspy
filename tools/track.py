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

Reads data/detect/<video>/detections.csv and guardrails.csv, or those of all
partial runs in data/detect/<video>/part_*/ (override with --detections /
--guardrails, several files allowed) and writes to data/objects/<video>/:
    objects.csv       one row per object
    tracks.json       all detections (frame, box) per object
    sections.csv      guardrail and barrier sections (station from/to, side)
    objects.geojson   objects (points), guardrail and barrier sections (lines), WGS84;
                      load it in the fix picker as reference layer to review
    index.html        one thumbnail per object, click opens the fix picker there
"""

import argparse
import csv
import html
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from videoio import full_resolution

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


def default_detections(base):
    """detections.csv of a full run, else those of all partial runs (part_* folders)."""
    if (base / "detections.csv").exists():
        return [base / "detections.csv"]
    parts = sorted(base.glob("part_*/detections.csv"))
    if not parts:
        raise SystemExit(f"No detections in {base} (run tools/detect.py first)")
    return parts


def combine(paths, key):
    """Rows of several CSV files in frame order; rows of overlapping parts only once."""
    seen, rows = set(), []
    for path in paths:
        for r in read_csv(path):
            k = tuple(r.get(c) for c in key)
            if k not in seen:
                seen.add(k)
                rows.append(r)
    return sorted(rows, key=lambda r: int(r["frame"]))


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


def same_object(a, b, group):
    """Whether two regions of one frame are parts of one object (see merge_split_regions)."""
    ix = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    iy = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])
    if group == "pole":
        narrower = min(a["x2"] - a["x1"], b["x2"] - b["x1"]) + 1
        return ix >= 0.5 * narrower and -iy <= 0.25 * max(a["size"], b["size"])
    smaller = min((a["x2"] - a["x1"]) * (a["y2"] - a["y1"]), (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return ix > 0 and iy > 0 and ix * iy >= 0.6 * max(smaller, 1)


def merge_split_regions(dets):
    """Join regions of one group in the same frame that belong to one object.

    The segmentation sometimes splits an object (sign with a sticker or glare, a
    post behind a twig) into several regions, which then became separate objects.
    Neighbouring objects must stay apart, so the rules are narrow:
      - poles: parts one above the other (horizontal overlap of at least half
        the narrower part, small vertical gap) – posts side by side stay apart;
      - signs: one part lies mostly (>= 60 %) inside the other – signs that
        overlap in perspective, stacked signs and additional panels stay apart.
    """
    by_key = {}
    for d in dets:
        by_key.setdefault((d["frame"], d["group"]), []).append(d)
    out = []
    for (frame, group), boxes in by_key.items():
        boxes = sorted(boxes, key=lambda d: -d["size"])
        merged = True
        while merged and len(boxes) > 1:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    if not same_object(a, b, group):
                        continue
                    if True:
                        x1, y1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
                        x2, y2 = max(a["x2"], b["x2"]), max(a["y2"], b["y2"])
                        boxes[i] = dict(a, x1=x1, y1=y1, x2=x2, y2=y2, cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                                        size=math.hypot(x2 - x1 + 1, y2 - y1 + 1))   # class/shape of the larger part
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break
        out += boxes
    return out


def stitch(tracks, step, max_steps=5):
    """Join a track that broke off (blur, short occlusion) with the one continuing it.

    Strict on purpose – successive delineator posts look alike: the continuation
    must start within `max_steps` evaluated frames, close to the position
    extrapolated from the last motion, keep moving in the same direction and be
    about as large (0.8-2.5x) as the broken track at its end. Each track is
    joined at most once per side, no chains of guesses.
    """
    tracks = sorted(tracks, key=lambda t: t[0]["frame"])
    links = []
    for i, a in enumerate(tracks):
        if len(a) < 2:
            continue
        end, prev = a[-1], a[-min(len(a), 4)]
        span = max(1, (end["frame"] - prev["frame"]) / step)
        vx, vy = (end["cx"] - prev["cx"]) / span, (end["cy"] - prev["cy"]) / span
        for j, b in enumerate(tracks):
            start = b[0]
            gap = (start["frame"] - end["frame"]) / step
            if j == i or not 0 < gap <= max_steps or not 0.8 * end["size"] <= start["size"] <= 2.5 * end["size"]:
                continue
            dx, dy = start["cx"] - end["cx"], start["cy"] - end["cy"]
            if len(b) >= 2:   # the continuation must move on in the same direction
                bx, by = b[1]["cx"] - start["cx"], b[1]["cy"] - start["cy"]
                if bx * vx + by * vy < 0:
                    continue
            dist = math.hypot(dx - vx * gap, dy - vy * gap)
            if dist <= end["size"] + 5 * gap:
                links.append((dist, i, j))
    head, tail = set(), set()
    nxt = {}
    for dist, i, j in sorted(links):
        if i not in head and j not in tail:
            head.add(i); tail.add(j); nxt[i] = j
    out = []
    for i, t in enumerate(tracks):
        if i in tail:
            continue
        chain, k = list(t), i
        while k in nxt:
            k = nxt[k]
            chain += tracks[k]
        out.append(chain)
    return out


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


def guardrail_sections(rows, traj, min_px, min_len, max_gap_m, lead_m, prefix=""):
    """Sections (station from/to, side) with a guardrail (prefix "barrier_": with a barrier).

    A section lasts as long as a guardrail is seen anywhere on that image side;
    the near part next to the vehicle is often hidden behind the cockpit. It
    starts where the guardrail first shows up in the near field (detect.py
    *_near_px), because further ahead it is visible long before it is reached;
    without near-field values the start is shifted by `lead_m` instead.
    """
    near = f"{prefix}left_near_px" in rows[0]
    smooth = lambda f: np.array([np.median(f[max(0, i - 2): i + 3]) for i in range(len(f))]) if len(f) >= 5 else f
    sections = []
    for side in ("left", "right"):
        pts = sorted((traj.at(float(r["video_time_s"]))["station"], float(r[f"{prefix}{side}_px"]) >= min_px,
                      near and float(r[f"{prefix}{side}_near_px"]) >= min_px / 3)
                     for r in rows if traj.covers(float(r["video_time_s"])))
        if not pts:
            continue
        stations = [p[0] for p in pts]
        seen = smooth(np.array([p[1] for p in pts], dtype=float)) >= 0.5   # median filter against dropouts
        close = smooth(np.array([p[2] for p in pts], dtype=float)) >= 0.5
        runs, first = [], None
        for i, st in enumerate(stations):
            if seen[i]:
                if first is not None and st - stations[last] > max_gap_m:
                    runs.append((first, last))
                    first = None
                first = i if first is None else first
                last = i
        if first is not None:
            runs.append((first, last))
        for a, b in runs:
            close_idx = [i for i in range(a, b + 1) if close[i]]
            begin = stations[close_idx[0]] if close_idx else min(stations[a] + lead_m, stations[b])
            if stations[b] - begin >= min_len:
                sections.append((begin, stations[b], side))
    return sections, near


def restraint_sections(rows, traj, args, join_m=30):
    """Guardrail and barrier sections: [(from, to, side, kind)], kind "guardrail" or "barrier".

    The model sometimes calls guardrails in curves or shade "Barrier". A barrier
    section therefore counts as guardrail if it mostly overlaps a guardrail section
    or fills a gap of at most `join_m` between two; otherwise (walls, concrete
    barriers, rock faces) it stays a barrier. Older guardrails.csv files without
    barrier columns contain both classes in the guardrail columns.
    """
    opts = dict(min_px=args.rail_px, min_len=20, max_gap_m=15, lead_m=args.rail_lead)
    rails, near = guardrail_sections(rows, traj, **opts)
    barriers = guardrail_sections(rows, traj, prefix="barrier_", **opts)[0] if "barrier_left_px" in rows[0] else []
    out = []
    for side in ("left", "right"):
        g = sorted([s, e] for s, e, sd in rails if sd == side)
        rest = []
        for s, e, sd in barriers:
            if sd != side:
                continue
            overlap = sum(max(0, min(e, ge) - max(s, gs)) for gs, ge in g)
            before = [ge for gs, ge in g if gs < s]   # guardrail starting before the barrier (may overlap it)
            after = [gs for gs, ge in g if ge > e]    # guardrail ending after it
            fills_gap = before and after and s - max(before) <= join_m and min(after) - e <= join_m
            if overlap >= 0.5 * (e - s) or fills_gap:
                g.append([s, e])
            else:
                rest.append((s, e))
        merged = []   # guardrail parts that now touch become one section
        for s, e in sorted(g):
            if merged and s - merged[-1][1] <= join_m:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out += [(s, e, side, "guardrail") for s, e in merged] + [(s, e, side, "barrier") for s, e in rest]
    return sorted(out), near, bool(barriers) or "barrier_left_px" in rows[0]


def tunnel_sections(rows, traj, min_px, min_len=20, max_gap_m=20):
    """Sections where much of the image is tunnel structure (detect.py tunnel_px)."""
    pts = sorted((traj.at(float(r["video_time_s"]))["station"], float(r["tunnel_px"]) >= min_px)
                 for r in rows if traj.covers(float(r["video_time_s"])))
    if not pts:
        return []
    stations = [p[0] for p in pts]
    flags = np.array([p[1] for p in pts], dtype=float)
    if len(flags) >= 5:
        flags = np.array([np.median(flags[max(0, i - 2): i + 3]) for i in range(len(flags))])
    out, first, last = [], None, None
    for i, st in enumerate(stations):
        if flags[i] >= 0.5:
            if first is not None and st - stations[last] > max_gap_m:
                out.append((stations[first], stations[last]))
                first = None
            first = i if first is None else first
            last = i
    if first is not None:
        out.append((stations[first], stations[last]))
    return [(s, e, "both", "tunnel") for s, e in out if e - s >= min_len]


def previous_run(out_dir):
    """tracks.json and sections.csv of the previous run, read before they are replaced."""
    tracks = json.loads((out_dir / "tracks.json").read_text(encoding="utf-8")) if (out_dir / "tracks.json").exists() else {}
    sections = {r["id"]: r for r in read_csv(out_dir / "sections.csv") if "id" in r} if (out_dir / "sections.csv").exists() else {}
    return tracks, sections


def carry_over_review(review_path, old, objects, section_info):
    """Keep review decisions when object ids change (new tracking, new detection run).

    Old ids are matched to new ones by shared detections (same frame, overlapping
    box) for objects and by kind, side and overlapping stations for sections.
    The previous file is kept as *.bak; decisions without a match go to
    "unmatched" instead of being dropped.
    """
    old_tracks, old_sections = old
    if not review_path.exists() or not (old_tracks or old_sections):
        return None
    review = json.loads(review_path.read_text(encoding="utf-8"))
    by_frame = {}
    for o in objects:
        for d in o["track"]:
            by_frame.setdefault(d["frame"], []).append((o["id"], d))
    items, unmatched, moved, lost = {}, dict(review.get("unmatched", {})), 0, 0
    for old_id, entry in review.get("items", {}).items():
        new_id = None
        if old_id in old_tracks:
            # strict: same kind of object, clearly overlapping boxes, and at least half of the
            # old detections at the new object – rather unmatched than attached to a neighbour
            votes = {}
            for f, x1, y1, x2, y2 in old_tracks[old_id]:
                for oid, d in by_frame.get(f, []):
                    if oid[0] != old_id[0]:
                        continue
                    iw = max(0, min(x2, d["x2"]) - max(x1, d["x1"]))
                    ih = max(0, min(y2, d["y2"]) - max(y1, d["y1"]))
                    union = (x2 - x1) * (y2 - y1) + (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]) - iw * ih
                    if union > 0 and iw * ih / union >= 0.3:
                        votes[oid] = votes.get(oid, 0) + 1
            best = max(votes, key=votes.get) if votes else None
            new_id = best if best and votes[best] >= 0.5 * len(old_tracks[old_id]) else None
        elif old_id in old_sections:
            o = old_sections[old_id]
            s0, s1 = float(o["station_from_m"]), float(o["station_to_m"])
            best = max((min(s1, x["to"]) - max(s0, x["from"]), x["id"]) for x in section_info
                       if x["kind"] == o["kind"] and x["side"] == o["side"]) if section_info else (0, None)
            new_id = best[1] if best[0] > 0 else None
        if new_id and new_id not in items:
            items[new_id] = dict(entry, previous_id=entry.get("previous_id", old_id)) if new_id != old_id else entry
            moved += new_id != old_id
        else:
            unmatched[old_id] = entry
            lost += 1
    review_path.with_name(review_path.name + ".bak").write_text(review_path.read_text(encoding="utf-8"), encoding="utf-8")
    review["items"], review["unmatched"] = items, unmatched
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(items), moved, lost


def thumbnails(video, objects, out_dir, scale=1.0, frames=None):
    """One crop per object from its largest detection, plus whole frames
    `frames` ({frame: [file name]}, e.g. sample views of tunnel sections).

    Reads the video once from front to back: seeking for every object is very
    slow, especially for long, 4K or VP9/WebM videos. `scale` converts boxes to
    the pixels of `video` (full-resolution copy).
    """
    import cv2

    (out_dir / "thumbs").mkdir(exist_ok=True)
    wanted, whole = {}, frames or {}
    for o in objects:
        wanted.setdefault(o["best"]["frame"], []).append(o)
    targets = sorted(set(wanted) | set(whole))
    if not targets:
        return
    cap = cv2.VideoCapture(str(video))
    frame = 0
    if targets[0] > 0:  # one jump to the first object; if it overshoots, start over from the beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, targets[0])
        frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame > targets[0]:
            cap = cv2.VideoCapture(str(video))
            frame = 0
    started, report = time.time(), max(1, len(targets) // 10)
    for n, target in enumerate(targets, 1):
        while frame < target and cap.grab():
            frame += 1
        ok, img = cap.read()
        frame += 1
        if not ok:
            break
        h, w = img.shape[:2]
        for name in whole.get(target, []):
            fit = 640 / w
            cv2.imwrite(str(out_dir / "thumbs" / name), cv2.resize(img, (640, int(h * fit)), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
        for o in wanted.get(target, []):
            d = {k: o["best"][k] * scale for k in ("x1", "y1", "x2", "y2")}
            bw, bh = d["x2"] - d["x1"], d["y2"] - d["y1"]
            pad = max(bw, bh, 40 * h / 720)
            x1, y1 = max(0, int(d["x1"] - pad)), max(0, int(d["y1"] - pad))
            x2, y2 = min(w, int(d["x2"] + pad)), min(h, int(d["y2"] + pad))
            crop = img[y1:y2, x1:x2].copy()
            cv2.rectangle(crop, (int(d["x1"] - x1), int(d["y1"] - y1)), (int(d["x2"] - x1), int(d["y2"] - y1)),
                          (255, 0, 255), max(1, h // 720))
            fit = 160 / crop.shape[0]   # (not `scale`: that converts the boxes of every object)
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * fit)), 160), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / "thumbs" / f"{o['id']}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if n % report == 0:
            print(f"  thumbnails: {n}/{len(targets)} frames, {time.time() - started:.0f} s", flush=True)


REASON_TEXT = {"not_slender": "nicht schlank", "never_near": "nie nah passiert"}


def video_fps(video):
    try:
        import cv2
        return cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30.0
    except Exception:
        return 30.0


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

    rails = "".join(f"<li>{s:.0f}–{e:.0f} m {side}</li>" for s, e, side, kind in sections if kind == "guardrail")
    walls = "".join(f"<li>{s:.0f}–{e:.0f} m {side}</li>" for s, e, side, kind in sections if kind == "barrier")
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
<h4>Barrieren / Wände (Barrier)</h4><ul>{walls or '<li>keine</li>'}</ul>
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
    parser.add_argument("--detections", type=Path, nargs="+",
                        help="default data/detect/<video>/detections.csv, or all part_*/detections.csv of partial runs")
    parser.add_argument("--guardrails", type=Path, nargs="+", help="default: guardrails.csv next to each detections file")
    parser.add_argument("--out", type=Path, help="default data/objects/<video>")
    parser.add_argument("--min-hits", type=int, default=3, help="minimum detections per object (default 3)")
    parser.add_argument("--max-gap", type=int, default=2, help="evaluated frames an object may be missed (default 2)")
    parser.add_argument("--rail-px", type=int, default=1500, help="guardrail pixels per image side counted as present (near field: a third)")
    parser.add_argument("--rail-lead", type=float, default=20,
                        help="without near-field values: metres a guardrail is seen before it is reached (default 20)")
    parser.add_argument("--road-width", type=float, default=6.5, help="carriageway width in m (default 6.5)")
    parser.add_argument("--pole-min-elongation", type=float, default=3.0,
                        help="poles less elongated than this (major/minor axis) are uncertain (default 3)")
    parser.add_argument("--pole-min-height", type=float, default=0.06,
                        help="poles never taller than this share of the image height are uncertain (default 0.06)")
    parser.add_argument("--tunnel-px", type=int, default=100000,
                        help="tunnel pixels (720-line image) for a frame to count as inside a tunnel (default 100000, ~11 %%)")
    parser.add_argument("--no-thumbs", action="store_true")
    parser.add_argument("--full-res", help="video to cut thumbnails from (default: <name>_full.mp4 next to a <name>_work.mp4)")
    args = parser.parse_args()

    video = Path(args.video)
    det_paths = args.detections or default_detections(ROOT / "data" / "detect" / video.stem)
    rail_paths = args.guardrails or [d.with_name("guardrails.csv") for d in det_paths]
    for path in det_paths + (args.guardrails or []):
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
    det_rows = combine(det_paths, key=("frame", "class", "x1", "y1"))
    rail_rows = combine([r for r in rail_paths if r.exists()], key=("frame",))
    rail_path = ", ".join(str(r) for r in rail_paths)
    out_dir = args.out or ROOT / "data" / "objects" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = Trajectory(args.trajectory)
    old_run = previous_run(out_dir)   # for carrying review decisions over to new ids

    dets = []
    for r in det_rows:
        if r["class"] not in GROUPS:
            continue
        x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        dets.append({"frame": int(r["frame"]), "t": float(r["video_time_s"]), "class": r["class"],
                     "group": GROUPS[r["class"]], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "cx": float(r["cx"]), "cy": float(r["cy"]), "size": math.hypot(x2 - x1 + 1, y2 - y1 + 1),
                     # rotation-invariant elongation from detect.py; older files: bounding box aspect
                     "elongation": float(r["elongation"]) if r.get("elongation") else
                                   max(x2 - x1 + 1, y2 - y1 + 1) / min(x2 - x1 + 1, y2 - y1 + 1)})
    before = len(dets)
    dets = merge_split_regions(dets)
    print(f"{before} detections, {before - len(dets)} joined with another region of the same object", flush=True)
    frames = sorted({d["frame"] for d in dets})
    step = int(np.median(np.diff(frames))) if len(frames) > 1 else 1
    width = max(d["x2"] for d in dets) + 1
    height = image_height(video) or max(d["y2"] for d in dets) + 1
    has_elongation = "elongation" in det_rows[0]
    min_elongation = args.pole_min_elongation if has_elongation else 1.5

    objects = []
    for group in sorted({d["group"] for d in dets}):
        raw = track([d for d in dets if d["group"] == group], step, args.max_gap)
        tracks = stitch(raw, step)
        print(f"  {group}: {len(raw)} tracks, {len(raw) - len(tracks)} broken ones joined", flush=True)
        for tr in tracks:
            if len(tr) < args.min_hits:
                continue
            t_pass, fitted = pass_time(tr)
            if not traj.covers(t_pass):
                continue
            largest = sorted(tr, key=lambda d: d["size"])[-3:]
            objects.append({"group": group, "class": Counter(d["class"] for d in tr).most_common(1)[0][0], "hits": len(tr),
                            "t_first": tr[0]["t"], "t_last": tr[-1]["t"], "t_pass": t_pass, "fitted": fitted,
                            "side": side_of(tr, width), "best": largest[-1], "track": tr,
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
            keep["track"] = sorted(prev["track"] + o["track"], key=lambda d: d["frame"])
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

    sections, near, separate = restraint_sections(rail_rows, traj, args) if rail_rows else ([], False, False)
    has_tunnels = bool(rail_rows) and "tunnel_px" in rail_rows[0]
    if has_tunnels:
        sections += tunnel_sections(rail_rows, traj, args.tunnel_px)
    # ids, video times and sample frames (for the review page) per section
    fps = video_fps(video)
    valid = ~np.isnan(traj.station)
    to_time = lambda st: float(np.interp(st, traj.station[valid], traj.t[valid]))
    section_info, counters, sample_frames = [], {}, {}
    for s, e, side, kind in sorted(sections, key=lambda x: (x[0], x[3])):
        prefix = {"guardrail": "G", "barrier": "B", "tunnel": "T"}[kind]
        counters[prefix] = counters.get(prefix, 0) + 1
        sid = f"{prefix}{counters[prefix]:03d}"
        t0, t1 = to_time(s), to_time(e)
        thumbs = []
        for k, q in enumerate((0.1, 0.4, 0.7, 0.95)):
            name = f"{sid}_{k}.jpg"
            sample_frames.setdefault(int(round((t0 + q * (t1 - t0)) * fps)), []).append(name)
            thumbs.append(name)
        section_info.append({"id": sid, "kind": kind, "side": side, "from": s, "to": e, "t0": t0, "t1": t1, "thumbs": thumbs})
    with open(out_dir / "sections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "side", "station_from_m", "station_to_m", "length_m", "time_from_s", "time_to_s"])
        for x in section_info:
            w.writerow([x["id"], x["kind"], x["side"], f"{x['from']:.1f}", f"{x['to']:.1f}", f"{x['to'] - x['from']:.1f}",
                        f"{x['t0']:.2f}", f"{x['t1']:.2f}"])

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
    for x in section_info:
        start, end, side, kind = x["from"], x["to"], x["side"], x["kind"]
        sts = np.linspace(start, end, max(2, int((end - start) / 5) + 1))
        idx = np.interp(sts, traj.station[~np.isnan(traj.station)], np.flatnonzero(~np.isnan(traj.station)))
        coords = []
        for k in idx:
            p = traj.at(traj.t[int(round(k))])
            if side == "both":   # tunnel: on the camera track
                coords.append([round(p["lon"], 7), round(p["lat"], 7)])
            else:
                coords.append([round(c, 7) for c in place(p, side, EDGE_M["guardrail"], args.road_width)[:2]])
        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                         "properties": {"id": x["id"], "kind": kind, "side": side, "video_time_s": round(x["t0"], 2),
                                        "time_to_s": round(x["t1"], 2), "thumbs": x["thumbs"], "station_from_m": round(start, 1),
                                        "station_to_m": round(end, 1)}})
    (out_dir / "objects.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    # every detection of every object, e.g. for tools/signs.py to pick the best views
    tracks = {o["id"]: [[d["frame"], round(d["x1"], 1), round(d["y1"], 1), round(d["x2"], 1), round(d["y2"], 1)]
                        for d in o["track"]] for o in objects}
    review_path = ROOT / "data" / "review" / f"{video.stem}.json"
    carried = carry_over_review(review_path, old_run, objects, section_info)
    (out_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    has_thumbs = not args.no_thumbs and video.exists()
    if has_thumbs:
        crop_video, scale = full_resolution(video, args.full_res)
        print(f"Writing thumbnails for {len(objects)} objects from {crop_video.name} "
              f"(one pass through the video; --no-thumbs skips this) ...", flush=True)
        thumbnails(crop_video, objects, out_dir, scale, sample_frames)
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
        fmt = lambda k: ", ".join(f"{s:.0f}-{e:.0f} m {side}" for s, e, side, kind in sections if kind == k) or "none"
        print(f"Sections from {rail_path} ({'start from near field' if near else 'whole image half – rerun detect.py for near-field values'}):")
        print(f"  guardrails: {fmt('guardrail')}")
        print(f"  barriers:   {fmt('barrier')}" if separate else
              "  barriers:   not separated in this file (older detect.py) – rerun detect.py to tell walls from guardrails")
        print(f"  tunnels:    {fmt('tunnel').replace(' both', '')}" if has_tunnels else
              "  tunnels:    no tunnel values in this file (older detect.py)")
    if carried:
        print(f"Review decisions carried over: {carried[0]} kept ({carried[1]} with a new id), {carried[2]} without "
              f"a matching object, kept under 'unmatched' in {review_path.name} (backup: {review_path.name}.bak)")
    if not traj.has_offset:
        print("Note: trajectory has no offset_m column (export it again from the fix picker); "
              "camera assumed in the middle of the right lane")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
