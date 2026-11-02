"""Locate the full-resolution copy of a working video (see prepare_video.py)."""

from pathlib import Path

import cv2


def full_resolution(video, explicit=None):
    """(path, scale): the video to cut crops from and the factor from pixels of
    `video` (in which detections are given) to pixels of that video.

    For <name>_work.mp4 the sibling <name>_full.mp4 is used if it exists.
    """
    video = Path(video)
    path = Path(explicit) if explicit else None
    if path is None and video.stem.endswith("_work"):
        sibling = video.with_name(video.stem[:-len("_work")] + "_full.mp4")
        if sibling.exists():
            path = sibling
    if path is None:
        return video, 1.0
    width = lambda p: cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_WIDTH) or 1
    return path, width(path) / width(video)


# ---------- cuts ----------

def cuts_path(video):
    """<name>_cuts.csv next to the video (written by prepare_video.py)."""
    video = Path(video)
    return video.with_name(video.stem + "_cuts.csv")


def manual_cuts_path(video):
    """data/cuts/<name>.json: cuts added or rejected by hand in the fix picker."""
    return Path(__file__).resolve().parent.parent / "data" / "cuts" / f"{Path(video).stem}.json"


def read_cuts(video):
    """Frame numbers where a new shot starts: the detected cuts (<name>_cuts.csv)
    plus those added by hand, minus those rejected by hand (data/cuts/<name>.json,
    times in seconds, matched within 0.5 s)."""
    import csv
    import json

    path, manual = cuts_path(video), manual_cuts_path(video)
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30
    times = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            times = [float(r["time_s"]) for r in csv.DictReader(f)]
    if manual.exists():
        edits = json.loads(manual.read_text(encoding="utf-8"))
        times = [t for t in times if not any(abs(t - r) < 0.5 for r in edits.get("removed", []))]
        times += [t for t in edits.get("added", []) if not any(abs(t - x) < 0.5 for x in times)]
    return sorted(round(t * fps) for t in times)


def detect_cuts(video, min_score=0.25, quiet=0.3, max_gap_s=0.5, progress=True):
    """Hard cuts: frames that differ abruptly from the previous one while the
    frames around them change smoothly.

    The difference is 1 - correlation of small, normalised greyscale images, so
    brightness changes (tunnel portals, shade, exposure) hardly count, while a
    jump to another scene does. A frame is a cut if its difference is at least
    `min_score` and the nearest real neighbours on both sides differ less than
    `quiet` times as much (fast pans change many frames in a row). Exact
    duplicate frames, common in re-encoded online videos, are skipped when
    looking for neighbours. Black, uniform and very dark low-contrast frames
    (unlit tunnels, where a passing lamp would otherwise look like a new scene)
    are skipped: a fade is no cut, but across a short black gap (<= `max_gap_s`,
    a cut through black) the frames before and after it are compared – not
    across dark tunnel frames or a long gap, or a tunnel's entry and exit would
    count as a cut.
    Returns [(frame, time_s, score)].
    """
    import time

    import numpy as np

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    diffs, last, gap_start, gap_black, frame, started = [], None, None, True, 0, time.time()
    max_gap = max(1, round(max_gap_s * fps))
    while True:
        ok, img = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        std, mean = small.std(), small.mean()
        if std <= 2 or (mean < 40 and std < 8):   # black, uniform or dark low-contrast: part of a gap
            if gap_start is None:
                gap_start, gap_black = frame, True
            gap_black = gap_black and std <= 2      # only a truly black gap may hide a cut
            diffs.append(0.0)
        else:
            norm = (small - small.mean()) / std
            d = float(1 - (norm * last).mean()) if last is not None else 0.0
            if gap_start is not None and gap_start > 0 and last is not None:
                if gap_black and frame - gap_start <= max_gap:
                    diffs[gap_start] = d   # cut through a short black gap: counted at its first frame
                diffs.append(0.0)
            else:
                diffs.append(d)
            last, gap_start = norm, None
        frame += 1
        if progress and frame % 3000 == 0:
            print(f"  cuts: {frame}/{total} frames, {time.time() - started:.0f} s", flush=True)
    diffs = np.array(diffs)

    def neighbour(t, direction):
        """Difference of the nearest frame that is not an exact duplicate (within 5 frames)."""
        for k in range(1, 6):
            i = t + direction * k
            if 0 < i < len(diffs) and diffs[i] > 0.002:
                return diffs[i]
        return 0.0

    cuts = []
    for t in range(1, len(diffs)):
        d = diffs[t]
        if d >= min_score and neighbour(t, -1) < quiet * d and neighbour(t, 1) < quiet * d:
            if not cuts or t - cuts[-1][0] > 5:
                cuts.append((t, t / fps, round(float(d), 3)))
    return cuts


def write_cuts(video, cuts):
    import csv

    with open(cuts_path(video), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "score"])
        for frame, t, score in cuts:
            w.writerow([frame, f"{t:.3f}", score])


# ---------- camera motion ----------

def motion_path(video):
    """<name>_motion.csv next to the video (written by prepare_video.py)."""
    video = Path(video)
    return video.with_name(video.stem + "_motion.csv")


def estimate_motion(video, width=640, roi=(0.08, 0.68), cuts=(), progress=True):
    """Camera rotation between consecutive frames from the image motion.

    Corners in the part of the image above the bonnet / dashboard (`roi` as
    fractions of the height) are tracked into the next frame (Lucas-Kanade);
    a rotation + shift is fitted robustly (RANSAC), so cars and close objects
    moving differently do not count. Per frame:
        yaw   horizontal shift as a fraction of the image width (+ = scene moves
              left, i.e. the camera turns right); x horizontal field of view
              = degrees
        pitch vertical shift as a fraction of the image height
        roll  image rotation in degrees (+ = clockwise)
    No values at cuts and where too few points could be tracked (dark tunnels).
    Returns [(frame, time_s, yaw, pitch, roll, inliers)].
    """
    import time

    import numpy as np

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cut_set, out, prev, frame, started = set(cuts), [], None, 0, time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        h0, w0 = img.shape[:2]
        gray = cv2.cvtColor(cv2.resize(img, (width, round(width * h0 / w0)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        row = (frame, frame / fps, None, None, None, 0)
        if prev is not None and frame not in cut_set:
            mask = np.zeros_like(prev)
            mask[int(roi[0] * h):int(roi[1] * h), :] = 255
            pts = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01, minDistance=8, mask=mask)
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(21, 21), maxLevel=3)
                good = st.ravel() == 1
                if good.sum() >= 12:
                    m, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if m is not None and inl is not None and inl.sum() >= 10:
                        # rotation about the image centre: shift of the centre point
                        cx, cy = width / 2, h / 2
                        dx = m[0, 0] * cx + m[0, 1] * cy + m[0, 2] - cx
                        dy = m[1, 0] * cx + m[1, 1] * cy + m[1, 2] - cy
                        roll = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
                        row = (frame, frame / fps, -dx / width, dy / h, roll, int(inl.sum()))
        out.append(row)
        prev = gray
        frame += 1
        if progress and frame % 3000 == 0:
            print(f"  motion: {frame}/{total} frames, {time.time() - started:.0f} s", flush=True)
    return out


def write_motion(video, rows):
    import csv

    with open(motion_path(video), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "yaw", "pitch", "roll_deg", "inliers"])
        for frame, t, yaw, pitch, roll, n in rows:
            fmt = lambda v, d: "" if v is None else f"{v:.{d}f}"
            w.writerow([frame, f"{t:.3f}", fmt(yaw, 5), fmt(pitch, 5), fmt(roll, 3), n])
