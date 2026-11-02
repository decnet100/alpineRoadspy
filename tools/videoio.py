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


def read_cuts(video):
    """Frame numbers where a new shot starts (empty if no cut list exists)."""
    import csv

    path = cuts_path(video)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return sorted(int(r["frame"]) for r in csv.DictReader(f))


def detect_cuts(video, min_score=0.25, quiet=0.3, progress=True):
    """Hard cuts: frames that differ abruptly from the previous one while the
    frames around them change smoothly.

    The difference is 1 - correlation of small, normalised greyscale images, so
    brightness changes (tunnel portals, shade, exposure) hardly count, while a
    jump to another scene does. A frame is a cut if its difference is at least
    `min_score` and the nearest real neighbours on both sides differ less than
    `quiet` times as much (fast pans change many frames in a row). Exact
    duplicate frames, common in re-encoded online videos, are skipped when
    looking for neighbours. Black or uniform frames are skipped altogether: a
    fade is no cut, but the frames before and after a black gap are compared.
    Returns [(frame, time_s, score)].
    """
    import time

    import numpy as np

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    diffs, last, gap_start, frame, started = [], None, None, 0, time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        std = small.std()
        if std <= 2:                       # black / uniform frame: remember where the gap began
            gap_start = frame if gap_start is None else gap_start
            diffs.append(0.0)
        else:
            norm = (small - small.mean()) / std
            d = float(1 - (norm * last).mean()) if last is not None else 0.0
            if gap_start is not None and gap_start > 0 and last is not None:
                diffs[gap_start] = d       # cut across a black gap: counted at its first frame
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
