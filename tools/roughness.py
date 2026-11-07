"""Head-camera videos: how bumpy the road is, from the cockpit's shaking and low rumble.

    python tools/roughness.py videos/samnaun2.mp4
    python tools/roughness.py videos/samnaun2_work.mp4 --start 1:00 --end 2:00

A helmet camera hardly shakes on a bumpy road - neck and helmet damp the jolts -
but the motorcycle does: the cockpit parts rigid with the frame shake relative to
the camera. That shaking is measured frame to frame (every frame, phase
correlation of the largest cockpit box of the cockpit.py config); the shift of
the cockpit within 0.2 s (head movements, looking around) is removed, what is
left is the jitter.

The cockpit position for each frame comes from <video>_cockpit.csv (cockpit.py),
so that the measured window follows the cockpit when the head turns; without it
the window stays where the box is in the config.

Second, independent sign: low rumble in the sound - the 20-60 Hz band relative to
the engine band (120-400 Hz), without the bins at the engine's firing frequency
and half of it (from data/audio/<video>/track.csv if audio_speed.py has run),
relative to the median of the surrounding 30 s. Rumble alone also comes from
wind; where both rise together it is the road.

Writes data/roughness/<video>.csv (10 rows per second):
    time_s
    jitter_px      vertical cockpit jitter, RMS over 0.25 s, pixels of the video per frame
    jitter_x_px    the same sideways
    quality        0-1: how well the cockpit could be followed (low: out of view, light change)
    rumble_db      low rumble relative to the surrounding 30 s (dB)
    speed_kmh      from the engine sound, if available
    station_m, lat, lon   from data/<video>_trajectory.csv (fix picker), if available
and <video>.png.

The values are relative - bumpier here than there - for this motorcycle, rider
and camera; faster driving over the same bump shakes more.
"""

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from audio_speed import source_video
from cockpit import default_config
from detect import parse_time
from prepare_video import find_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
SCALE = 0.5          # processing scale of the cockpit window
DETREND_S = 0.2      # shifts slower than this are head movements
WINDOW_S = 0.25      # RMS window of the jitter
OUT_RATE = 10        # output rows per second
MIN_RESPONSE = 0.5   # phase correlation peak below this: frame pair not used
SCENE_W = 960        # processing width of the scene for the pitch


def stem_of(video):
    return Path(video).stem.removesuffix("_work").removesuffix("_full")


def video_info(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps, w, h


def cockpit_window(config):
    """Largest box of the config (or bounding box of the largest polygon) and the
    centre of all parts, which cockpit.py reports as the cockpit position."""
    boxes = [tuple(b) for b in config.get("boxes", [])]
    boxes += [(*np.min(p, 0), *np.max(p, 0)) for p in map(np.array, config.get("polygons", []))]
    if not boxes:
        raise SystemExit("Cockpit config has no boxes or polygons")
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    allp = np.array([[b[0], b[1]] for b in boxes] + [[b[2], b[3]] for b in boxes], float)
    centre = (allp.min(0) + allp.max(0)) / 2
    return np.array(box, float), centre, float(allp[:, 1].min())


def read_cockpit_track(video):
    """(path, t, x, y, lean) of the cockpit centre and the lean from cockpit.py, outliers dropped; None if missing."""
    video = Path(video)
    stem = stem_of(video)
    for cand in (video.with_name(video.stem + "_cockpit.csv"), video.with_name(stem + "_cockpit.csv"),
                 video.with_name(stem + "_full_cockpit.csv")):
        if cand.exists():
            with open(cand, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r.get("ok", "1") in ("1", "1.0", "True")]
            if rows:
                lean = lambda r: float(r["lean_deg"]) if r.get("lean_deg") not in (None, "") else np.nan
                arr = np.array([[float(r["time_s"]), float(r["x"]), float(r["y"]), lean(r)] for r in rows])
                return cand, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    return None


def measure_chunk(job):
    """Frame-to-frame shift of the cockpit (dx, dy in video pixels, response) and of the scene
    above it (dy, response) for [start, end)."""
    video, ffmpeg, start, end, fps, size, box, top, offset_t, offset_xy = job
    cv2.setNumThreads(1)
    W, H = size
    t_frames = np.arange(round(start * fps), round(end * fps) + 1) / fps
    if offset_t is not None:
        ox = np.interp(t_frames, offset_t, offset_xy[0])
        oy = np.interp(t_frames, offset_t, offset_xy[1])
    else:
        ox = oy = np.zeros(len(t_frames))
    bw, bh = box[2] - box[0], box[3] - box[1]
    # decode only the band the window can be in during this chunk
    x0 = int(max(0, min(box[0] + ox.min(), W - bw) - 8)) // 2 * 2
    y0 = int(max(0, min(box[1] + oy.min(), H - bh) - 8)) // 2 * 2
    x1 = int(min(W, max(box[2] + ox.max(), bw) + 8)) // 2 * 2
    y1 = int(min(H, max(box[3] + oy.max(), bh) + 8)) // 2 * 2
    cw, ch = round((x1 - x0) * SCALE) // 2 * 2, round((y1 - y0) * SCALE) // 2 * 2
    # the whole picture small for the scene, stacked below the cockpit band in one stream
    sw, sh = SCENE_W, round(SCENE_W * H / W) // 2 * 2
    pw = max(cw, sw)
    graph = (f"[0:v]format=gray,split[a][b];[a]crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={cw}:{ch},pad={pw}:{ch}[c];"
             f"[b]scale={sw}:{sh},pad={pw}:{sh}[s];[c][s]vstack[v]")
    # few threads per ffmpeg: several of them run side by side
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "2", "-ss", f"{start:.4f}", "-t", f"{end - start + 1.5 / fps:.4f}",
           "-i", str(video), "-an", "-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1",
           "-filter_complex", graph, "-map", "[v]", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = pw * (ch + sh)
    ss = sw / W
    # scene window: middle 60 % of the width, 28 % of the height, just above the cockpit
    sx0, swin_w, swin_h = round(0.2 * sw), round(0.6 * sw), round(0.28 * sh)
    shann = cv2.createHanningWindow((swin_w, swin_h), cv2.CV_32F)
    s = (cw / (x1 - x0), ch / (y1 - y0))
    win_w, win_h = round(bw * s[0]), round(bh * s[1])
    hann = cv2.createHanningWindow((win_w, win_h), cv2.CV_32F)
    out, prev, i = [], None, 0
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n or i >= len(t_frames):
            break
        full = np.frombuffer(buf, np.uint8).reshape(ch + sh, pw).astype(np.float32)
        img, scene = full[:ch, :cw], full[ch:, :sw]
        if prev is not None:
            # same window in both frames: where the cockpit was in the earlier one
            wx = int(round((box[0] + ox[i - 1] - x0) * s[0]))
            wy = int(round((box[1] + oy[i - 1] - y0) * s[1]))
            wx = min(max(wx, 0), cw - win_w)
            wy = min(max(wy, 0), ch - win_h)
            # share of the box that is in the picture
            vis_h = (min(H, box[3] + oy[i - 1]) - max(0, box[1] + oy[i - 1])) / bh
            vis_w = (min(W, box[2] + ox[i - 1]) - max(0, box[0] + ox[i - 1])) / bw
            a = prev[wy:wy + win_h, wx:wx + win_w]
            b = img[wy:wy + win_h, wx:wx + win_w]
            (dx, dy), resp = cv2.phaseCorrelate(a, b, hann)
            resp *= float(np.clip(vis_h, 0, 1) * np.clip(vis_w, 0, 1))
            bottom = int(np.clip((top + oy[i - 1]) * ss - 0.03 * sh, swin_h + 0.05 * sh, sh))
            (_, sdy), sresp = cv2.phaseCorrelate(prev_scene[bottom - swin_h:bottom, sx0:sx0 + swin_w],
                                                  scene[bottom - swin_h:bottom, sx0:sx0 + swin_w], shann)
            out.append((t_frames[i], dx / s[0], dy / s[1], resp, sdy / ss, sresp))
        prev, prev_scene, i = img, scene, i + 1
    proc.stdout.close()
    proc.wait()
    if i < 0.95 * len(t_frames) and t_frames[-1] < duration_hint(video, fps) - 1:
        raise RuntimeError(f"{start:.0f}-{end:.0f} s: only {i} of {len(t_frames)} frames decoded (ffmpeg exit {proc.returncode})")
    return out


def duration_hint(video, fps):
    cap = cv2.VideoCapture(str(video))
    return cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps


def measure(video, ffmpeg, start, end, fps, size, box, centre, top, track, workers, chunk_s=60):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    offset_t = offset_xy = None
    if track is not None:
        _, t, x, y, _ = track
        offset_t, offset_xy = t, (x - centre[0], y - centre[1])
    borders = list(np.arange(start, end, chunk_s)) + [end]
    jobs = [(video, ffmpeg, a, b, fps, size, box, top, offset_t, offset_xy) for a, b in zip(borders, borders[1:]) if b - a > 2 / fps]
    rows, started, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(measure_chunk, j) for j in jobs]):
            rows += fut.result()
            done += 1
            print(f"  cockpit jitter: {done}/{len(jobs)} parts, {time.time() - started:.0f} s", flush=True)
    rows.sort()
    return np.array(rows, float).reshape(-1, 6)


def jitter(shifts, fps, t_out):
    """RMS of the frame-to-frame shift minus its 0.2 s mean, over 0.25 s around each output time."""
    t, dx, dy, resp = shifts[:, :4].T
    good = resp >= MIN_RESPONSE
    k = max(3, round(DETREND_S * fps))
    ker = np.ones(k) / k

    def detrend(v):
        v = np.where(good, v, 0.0)
        w = np.convolve(good.astype(float), ker, "same")
        return v - np.convolve(v, ker, "same") / np.maximum(w, 1e-9)

    hx, hy = detrend(dx), detrend(dy)
    half = WINDOW_S / 2
    jy, jx, q = [], [], []
    lo = np.searchsorted(t, t_out - half)
    hi = np.searchsorted(t, t_out + half)
    for a, b in zip(lo, hi):
        g = good[a:b]
        if b - a == 0 or g.sum() < 0.5 * (b - a):
            jy.append(np.nan); jx.append(np.nan)
        else:
            jy.append(np.sqrt(np.mean(hy[a:b][g] ** 2)))
            jx.append(np.sqrt(np.mean(hx[a:b][g] ** 2)))
        q.append(float(np.mean(np.clip(resp[a:b], 0, 1))) if b > a else 0.0)
    return np.array(jy), np.array(jx), np.array(q)


def pitch(shifts, fps, t_out, speed, lean):
    """Pitching of the motorcycle at 0.7-4 Hz (waves, crests, dips), RMS over 1 s, in video pixels,
    and whether the driving there is steady.

    Scene shift minus cockpit shift: turning the head moves both alike and drops out, what is
    left is the vehicle's pitch - and the rider's body moving up and down relative to it (1 cm
    shifts the near cockpit by about 20 px), which is a response to the road as well. Braking and
    accelerating pitch the vehicle too, and in curves turning at a lean shows up as pitch in the
    image: only steady stretches (|acceleration| < 0.5 m/s^2, lean < 8 deg) count."""
    t, _, cdy, cr, sdy, sr = shifts.T
    good = (cr >= MIN_RESPONSE) & (sr >= MIN_RESPONSE)
    cum = np.cumsum(np.where(good, sdy - cdy, 0.0))

    def smooth(x, sec):
        k = max(1, round(sec * fps))
        return np.convolve(x, np.ones(k) / k, "same")

    band = smooth(cum, 0.125) - smooth(cum, 1.4)
    steady = good.copy()
    if speed is not None:
        ok = np.isfinite(speed[1])
        vs = np.interp(speed[0], speed[0][ok], speed[1][ok]) / 3.6
        k = max(1, round(1.0 / np.median(np.diff(speed[0]))))       # 1 s
        vs = np.convolve(vs, np.ones(k) / k, "same")
        acc = np.gradient(vs, speed[0])
        steady &= np.abs(np.interp(t, speed[0], acc)) < 0.5
        steady &= np.interp(t, speed[0], vs) > 5 / 3.6
    if lean is not None:
        ok = np.isfinite(lean[1])
        steady &= np.abs(np.interp(t, lean[0][ok], lean[1][ok], left=np.inf, right=np.inf)) < 8
    lo, hi = np.searchsorted(t, t_out - 0.5), np.searchsorted(t, t_out + 0.5)
    rms = np.array([np.sqrt(np.mean(band[a:b] ** 2)) if b > a else np.nan for a, b in zip(lo, hi)])
    st = np.array([steady[a:b].mean() > 0.8 if b > a else False for a, b in zip(lo, hi)])
    return rms, st


def rumble(video, ffmpeg, start, end, t_out, engine):
    """Low band (20-60 Hz) over engine band (120-400 Hz) in dB, relative to the surrounding 30 s."""
    from scipy.ndimage import median_filter

    sr, hop = 4000, 0.05
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
           "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)
    if len(x) < sr:
        return np.full(len(t_out), np.nan)
    win = int(hop * sr)
    nfft = 4 * win
    f = np.fft.rfftfreq(nfft, 1 / sr)
    low, high = (f >= 20) & (f < 60), (f >= 120) & (f < 400)
    n = len(x) // win
    frames = x[:n * win].reshape(n, win) * np.hanning(win)
    S = np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2
    ts = start + (np.arange(n) + 0.5) * hop
    mask = np.tile(low, (n, 1))
    if engine is not None:
        f0 = np.interp(ts, engine[0], engine[1], left=np.nan, right=np.nan)
        for m in (1.0, 0.5):
            near = np.abs(f[None, :] - m * f0[:, None]) < 0.1 * m * np.nan_to_num(f0, nan=-1)[:, None]
            mask &= ~near
    lowp = (S * mask).sum(1) / np.maximum(mask.sum(1), 1) * low.sum()
    ratio = lowp / (S[:, high].sum(1) + 1e-12)
    k = round(0.5 / hop)
    ratio = np.convolve(ratio, np.ones(k) / k, "same")
    db = 10 * np.log10(ratio + 1e-12)
    db -= median_filter(db, size=round(30 / hop), mode="nearest")
    return np.interp(t_out, ts, db, left=np.nan, right=np.nan)


def read_audio_track(stem):
    path = ROOT / "data" / "audio" / stem / "track.csv"
    if not path.exists():
        return None, None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["time_s"]) for r in rows])
    num = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])
    return (t, num("f0_hz")), (t, num("speed_kmh"))


def read_trajectory(stem):
    path = ROOT / "data" / f"{stem}_trajectory.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])
    return path, col("video_time_s"), col("station_m"), col("lat"), col("lon")


def plot(path, t, jy, q, rb, pr, speed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    ax[0].plot(t, np.where(q >= 0.5, jy, np.nan), lw=0.8, color="#c62828")
    ax[0].set_ylabel("cockpit jitter [px]")
    ax[1].plot(t, rb, lw=0.8, color="#1565c0")
    ax[1].axhline(0, color="#999", lw=0.5)
    ax[1].set_ylabel("rumble [dB]")
    if speed is not None:
        ax[2].plot(t, speed, lw=0.8, color="#2e7d32")
    ax[2].set_ylabel("speed [km/h]")
    ax[3].plot(t, pr, lw=0.8, color="#6a1b9a")
    ax[3].set_ylabel("pitch, steady driving [px]")
    ax[3].set_xlabel("video time [s]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video")
    parser.add_argument("--config", help="cockpit parts (default data/cockpit/<video>.json)")
    parser.add_argument("--start", default="0")
    parser.add_argument("--end")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--ffmpeg")
    parser.add_argument("--out", help="output CSV (default data/roughness/<video>.csv)")
    parser.add_argument("--reuse", action="store_true",
                        help="take the frame shifts of the last run (<out>_shifts.npz) instead of decoding the video again")
    args = parser.parse_args()

    video = Path(args.video)
    stem = stem_of(video)
    config_path = Path(args.config) if args.config else default_config(video)
    if not config_path.exists():
        raise SystemExit(f"No cockpit config {config_path}: see tools/cockpit.py --grid")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ffmpeg = find_ffmpeg(args.ffmpeg)
    fps, duration, W, H = video_info(video)
    start = parse_time(args.start)
    end = min(parse_time(args.end), duration) if args.end else duration
    box, centre, top = cockpit_window(config)
    track = read_cockpit_track(video)
    if track is None:
        print("  no <video>_cockpit.csv (tools/cockpit.py): the window stays where the box is in the config")
    else:
        print(f"  cockpit position from {track[0].name}, {track[1][0]:.0f}-{track[1][-1]:.0f} s")
        if track[1][0] > start + 5 or track[1][-1] < end - 5:
            print(f"  outside {track[1][0]:.0f}-{track[1][-1]:.0f} s the window stays at the nearest known cockpit position "
                  "and the pitch is left out (no lean): run cockpit.py over the whole video")

    out = Path(args.out) if args.out else ROOT / "data" / "roughness" / f"{stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    shifts_path = out.with_name(out.stem + "_shifts.npz")
    if args.reuse and shifts_path.exists():
        shifts = np.load(shifts_path)["shifts"].astype(float)
        shifts = shifts[(shifts[:, 0] >= start) & (shifts[:, 0] <= end)]
        print(f"Frame shifts from {shifts_path.name}")
    else:
        print(f"Cockpit jitter of {video.name}, {start:.0f}-{end:.0f} s at {fps:g} fps ...")
        shifts = measure(video, ffmpeg, start, end, fps, (W, H), box, centre, top, track, args.workers)
        # frame-to-frame shifts, for trying other measures without decoding again (--reuse)
        np.savez_compressed(shifts_path, shifts=shifts.astype(np.float32), fps=fps)
    t_out = np.arange(np.ceil(start * OUT_RATE) / OUT_RATE, end, 1 / OUT_RATE)
    jy, jx, q = jitter(shifts, fps, t_out)

    engine, speed = read_audio_track(stem)
    print("Low rumble in the sound" + (" (engine frequency from audio_speed.py left out)" if engine else "") + " ...")
    rb = rumble(source_video(video), ffmpeg, start, end, t_out, engine)
    v = np.interp(t_out, speed[0], speed[1], left=np.nan, right=np.nan) if speed else None
    pr, steady = pitch(shifts, fps, t_out, speed, (track[1], track[4]) if track else None)

    traj = read_trajectory(stem)
    if traj:
        _, tt, st, la, lo = traj
        place = [np.interp(t_out, tt, c, left=np.nan, right=np.nan) for c in (st, la, lo)]
        print(f"  stations from {traj[0].name}")
    else:
        place = [np.full(len(t_out), np.nan)] * 3

    fmt = lambda x, d: "" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "jitter_px", "jitter_x_px", "quality", "rumble_db", "pitch_px", "steady", "speed_kmh",
                    "station_m", "lat", "lon"])
        for i, t in enumerate(t_out):
            w.writerow([f"{t:.2f}", fmt(jy[i], 2), fmt(jx[i], 2), f"{q[i]:.2f}", fmt(rb[i], 1), fmt(pr[i], 1), int(steady[i]),
                        fmt(v[i] if v is not None else None, 1), fmt(place[0][i], 1), fmt(place[1][i], 7), fmt(place[2][i], 7)])
    plot(out.with_suffix(".png"), t_out, jy, q, rb, np.where(steady, pr, np.nan), v)

    ok = np.isfinite(jy) & (q >= 0.5)
    if ok.any():
        calm = np.percentile(jy[ok], 50)
        rough = ok & (jy > 3 * calm) & (rb > 3)
        print(f"  jitter median {calm:.1f} px, 95 % {np.percentile(jy[ok], 95):.1f} px; "
              f"{100 * (1 - ok.mean()):.0f} % of the time not measurable (cockpit out of view, light change)")
        if rough.any():
            # stretches where the jitter is 3x the median and the sound rumbles as well
            idx = np.flatnonzero(rough)
            groups = np.split(idx, np.flatnonzero(np.diff(idx) > OUT_RATE) + 1)
            print("  shaking and rumbling: " + ", ".join(f"{t_out[g[0]]:.1f}-{t_out[g[-1]]:.1f} s" for g in groups))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
