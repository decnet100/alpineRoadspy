"""Head-camera videos: head pose relative to the vehicle and lean angle, from the cockpit.

    python tools/cockpit.py videos/samnaun2_work.mp4 --grid 130
    python tools/cockpit.py videos/samnaun2_work.mp4 --config data/cockpit/samnaun2.json

With a helmet camera the image turns with the head as well as with the vehicle.
Parts of the cockpit that are rigid with the frame (instrument cluster, fairing,
windscreen and its clamps - not the mirrors, phone holder or switches on the
handlebar, which steer) show where the camera points relative to the vehicle.

1. `--grid T` writes the frame at T seconds with a pixel grid to
   data/cockpit/<video>_ref.jpg. Pick a frame with the whole cockpit in view and
   note the rigid parts as boxes [x0, y0, x1, y1] and/or polygons [[x, y], ...]
   in that frame's pixels (see docs/cockpit_example.json).
2. With the config the cockpit is found in every sampled frame: the edges of
   those parts (gradient image, robust to lighting) are searched at a few angles,
   allowing the cockpit to be partly outside the image, and refined by
   ECC alignment. Jumps are filtered out afterwards.
3. The camera's roll in the world comes from the scene above the cockpit (frame
   to frame, integrated; slow drift removed over 30 s). Lean of the vehicle =
   camera roll in the world - camera roll relative to the cockpit; positive =
   leaning right. No calibration needed: both are image rotations in degrees.

Writes <video>_cockpit.csv next to the video:
    frame, time_s, x, y      cockpit reference point (centre of the parts) in pixels
    rot_deg                  cockpit rotation in the image (camera roll relative to the vehicle)
    match                    template match score (0-1), ok (0/1: not an outlier)
    scene_roll_deg           camera roll change since the previous sample (scene)
    lean_deg                 lean of the vehicle

On the Samnaun 2 ride (Tiger 1050) the lean agrees with v * omega / g from the
trajectory with r = 0.71 and slope 0.96. Head yaw from the cockpit's shift did
not improve the comparison of camera turning with the road: leaning and head
movements shift the near cockpit as well.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from detect import parse_time

WIDTH = 960           # processing width
ANGLES = range(-15, 16, 3)


def cockpit_path(video):
    video = Path(video)
    return video.with_name(video.stem + "_cockpit.csv")


def default_config(video):
    return Path(__file__).resolve().parent.parent / "data" / "cockpit" / f"{Path(video).stem.removesuffix('_work')}.json"


def read_frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    if not ok:
        raise SystemExit(f"No frame at {t} s")
    return img


def write_grid(video, t, out):
    img = read_frame(cv2.VideoCapture(str(video)), t)
    h, w = img.shape[:2]
    step = 10 ** int(np.log10(w / 8))
    step = step * (2 if w / step < 12 else 1)
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), max(1, w // 1920))
        cv2.putText(img, str(x), (x + 3, h // 2), 0, w / 3000, (0, 255, 255), max(1, w // 1280))
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), max(1, w // 1920))
        cv2.putText(img, str(y), (3, y - 3), 0, w / 3000, (0, 255, 255), max(1, w // 1280))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"{w}x{h} frame at {t} s with grid -> {out}")


def gradient(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    return np.sqrt(mag / (np.percentile(mag, 99) + 1e-6))


class Cockpit:
    """Finds the rigid cockpit parts of a reference frame in other frames."""

    def __init__(self, ref_img, config):
        h0, w0 = ref_img.shape[:2]
        self.s = WIDTH / w0
        ref = cv2.resize(ref_img, (WIDTH, round(h0 * self.s)), interpolation=cv2.INTER_AREA)
        self.mask = np.zeros(ref.shape[:2], np.uint8)
        pts = []
        for x0, y0, x1, y1 in config.get("boxes", []):
            p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
            cv2.fillPoly(self.mask, [(p * self.s).astype(np.int32)], 255)
            pts.append(p)
        for poly in config.get("polygons", []):
            p = np.array(poly, float)
            cv2.fillPoly(self.mask, [(p * self.s).astype(np.int32)], 255)
            pts.append(p)
        if not pts:
            raise SystemExit("Config has no boxes or polygons")
        allp = np.vstack(pts) * self.s
        self.x0, self.y0 = np.floor(allp.min(0)).astype(int)
        self.x1, self.y1 = np.ceil(allp.max(0)).astype(int)
        self.top = allp[:, 1].min()                      # upper edge of the cockpit parts
        self.centre = ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
        self.pad = (self.y1 - self.y0) // 2
        self.ref = gradient(ref)
        self.templates = []
        for a in ANGLES:
            rot = cv2.getRotationMatrix2D(self.centre, a, 1.0)
            size = (self.ref.shape[1], self.ref.shape[0])
            tg = cv2.warpAffine(self.ref * (self.mask > 0), rot, size)[self.y0:self.y1, self.x0:self.x1]
            tm = cv2.warpAffine(self.mask, rot, size, flags=cv2.INTER_NEAREST)[self.y0:self.y1, self.x0:self.x1]
            self.templates.append((a, tg, tm))

    def locate(self, small):
        """(2x3 transform reference -> frame at processing scale, match score)."""
        grad = gradient(small)
        p = self.pad
        padded = cv2.copyMakeBorder(grad, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        best = None
        for a, tg, tm in self.templates:
            res = cv2.matchTemplate(padded, tg, cv2.TM_CCORR_NORMED, mask=tm)
            res[~np.isfinite(res)] = 0
            _, score, _, loc = cv2.minMaxLoc(res)
            if best is None or score > best[0]:
                best = (score, a, loc)
        score, a, (lx, ly) = best
        warp = cv2.getRotationMatrix2D(self.centre, a, 1.0).astype(np.float32)
        warp[0, 2] += lx - p - self.x0
        warp[1, 2] += ly - p - self.y0
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
            _, fine = cv2.findTransformECC(self.ref, grad, warp.copy(), cv2.MOTION_EUCLIDEAN, crit, self.mask, 5)
            if abs(fine[0, 2] - warp[0, 2]) < 15 and abs(fine[1, 2] - warp[1, 2]) < 15:
                warp = fine
        except cv2.error:
            pass
        return warp, score


def scene_roll(prev, gray, top):
    """Image rotation (degrees) of the scene above row `top` between two frames."""
    mask = np.zeros_like(gray)
    mask[int(0.03 * gray.shape[0]):max(0, int(top) - 10)] = 255
    pts = cv2.goodFeaturesToTrack(prev, 200, 0.01, 6, mask=mask)
    if pts is None or len(pts) < 12:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(21, 21), maxLevel=3)
    good = st.ravel() == 1
    if good.sum() < 12:
        return None
    m, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=2)
    if m is None or inl.sum() < 10:
        return None
    return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))


def track(video, config, start=0.0, end=None, rate=10.0, progress=True):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cockpit = Cockpit(read_frame(cap, config["ref_time"]), config)
    step = max(1, round(fps / rate))
    first = round(start * fps)
    last = round(end * fps) if end is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    rows, prev, started, w0 = [], None, time.time(), 1
    for frame in range(first, last, step):
        if frame > first:
            for _ in range(step - 1):
                cap.grab()
        ok, img = cap.read()
        if not ok:
            break
        h0, w0 = img.shape[:2]
        small = cv2.resize(img, (WIDTH, round(h0 * WIDTH / w0)), interpolation=cv2.INTER_AREA)
        warp, score = cockpit.locate(small)
        top = cv2.transform(np.float32([[[cockpit.x0, cockpit.top], [cockpit.x1, cockpit.top]]]), warp)[0][:, 1].min()
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        roll = scene_roll(prev, gray, top) if prev is not None else None
        prev = gray
        cx, cy = cv2.transform(np.float32([[cockpit.centre]]), warp)[0][0] * (w0 / WIDTH)
        rows.append([frame, frame / fps, float(cx), float(cy), float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))),
                     float(score), roll])
        if progress and len(rows) % 300 == 0:
            print(f"  cockpit: {frame / fps:.0f} s, {time.time() - started:.0f} s", flush=True)
    return rows, step / fps, w0


def lean(rows, dt, width, min_score=0.75):
    """Mark outliers and compute the lean angle (degrees, + = right) per row."""
    from scipy.ndimage import median_filter

    rot = np.array([r[4] for r in rows])
    score = np.array([r[5] for r in rows])
    x = np.array([r[2] for r in rows])
    ok = (np.abs(rot - median_filter(rot, 9, mode="nearest")) <= 3) & \
         (np.abs(x - median_filter(x, 7, mode="nearest")) <= 0.015 * width) & (score >= min_score)
    idx = np.arange(len(rows))
    rel = np.interp(idx, idx[ok], rot[ok]) if ok.any() else rot
    rel = median_filter(rel, 5, mode="nearest")
    rel -= np.median(rel)
    cam = np.cumsum([r[6] or 0.0 for r in rows])
    cam -= median_filter(cam, max(3, int(30 / dt)), mode="nearest")
    return ok, -(cam - rel)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--grid", type=float, metavar="T", help="write the frame at T s with a pixel grid and stop")
    parser.add_argument("--config", help="cockpit parts (default data/cockpit/<video>.json)")
    parser.add_argument("--start", default="0")
    parser.add_argument("--end")
    parser.add_argument("--rate", type=float, default=10, help="samples per second (default 10)")
    args = parser.parse_args()

    video = Path(args.video)
    config_path = Path(args.config) if args.config else default_config(video)
    if args.grid is not None:
        write_grid(video, args.grid, config_path.with_name(config_path.stem + "_ref.jpg"))
        print(f"Note the rigid cockpit parts in {config_path} (see docs/cockpit_example.json)")
        return
    if not config_path.exists():
        raise SystemExit(f"No cockpit config {config_path}: start with --grid T")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"Tracking the cockpit in {video.name} ...", flush=True)
    rows, dt, width = track(video, config, parse_time(args.start), parse_time(args.end) if args.end else None, args.rate)
    ok, lean_deg = lean(rows, dt, width)
    with open(cockpit_path(video), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "x", "y", "rot_deg", "match", "ok", "scene_roll_deg", "lean_deg"])
        for r, good, phi in zip(rows, ok, lean_deg):
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.2f}", f"{r[5]:.3f}", int(good),
                        "" if r[6] is None else f"{r[6]:.3f}", f"{phi:.1f}"])
    print(f"  {len(rows)} samples, {ok.mean():.0%} without outliers -> {cockpit_path(video).name}")


if __name__ == "__main__":
    main()
