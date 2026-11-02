"""Engine speed, gear shifts and road speed from the engine sound of a video.

    python tools/audio_speed.py videos/samnaun2.mp4 --end 3:00 --cylinders 3
    python tools/audio_speed.py videos/samnaun2.mp4 --readings data/audio/samnaun2_readings.csv

The working copies from prepare_video.py have no audio: give the original video
(for <name>_work.mp4 a sibling <name>.* is looked for).

1. Pitch: the firing frequency f0 of the engine (rpm * cylinders / 120 for a
   four-stroke) is tracked in a spectrogram. Each candidate is scored by how
   strongly its whole harmonic series stands out against the points halfway
   between the harmonics (so half or a third of the true pitch does not win);
   smooth changes are cheap, jumps (gear shifts) cost a fixed amount.
2. Load: the share of the upper harmonics (3rd-8th) against the 1st/2nd. It
   drops when the throttle closes (shifting, coasting, braking).
3. Steady segments: stretches without pitch jumps or abrupt load changes in
   which the pitch varies little. Within one, speed = k_gear * f0 with one
   constant per gear.
4. Events in the gaps between segments: fast falls (upshift?), fast rises
   (downshift?), coasting with closed throttle (clutch pulled, braking). A
   downshift while braking shows a smaller jump than the gear step, because the
   speed falls at the same time.
5. With speed readings (--readings, e.g. read off the speedometer in the video;
   CSV time_s,speed_kmh) the gear constants are fitted, every segment gets a
   gear (directly or carried over gaps without shifts) and the speed per frame.
   Between two segments the gear must change by the number of shifts found;
   where they disagree (typically a downshift hidden in braking), the gap is
   flagged with the coasting phases in which the missing shift probably lies.

Writes to data/audio/<video>/: track.csv (per spectrogram frame, 50 ms),
segments.csv, events.csv, gears.csv (with readings) and plot.png (if
matplotlib is installed).
"""

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

from detect import parse_time
from prepare_video import find_ffmpeg

RATE = 16000
NPERSEG = 8192        # 0.5 s window: ~2 Hz resolution, enough to separate harmonics at 20 Hz spacing
HOP = 800             # 50 ms
MAX_HARMONIC_HZ = 700
JUMP_WINDOW = 5       # frames (0.25 s) over which pitch jumps are measured
VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".MP4", ".MOV", ".avi")


def source_video(video):
    """The video with the audio track: <name>_work/_full.mp4 have none, use <name>.*"""
    video = Path(video)
    for suffix in ("_work", "_full"):
        if video.stem.endswith(suffix):
            stem = video.stem[:-len(suffix)]
            for ext in VIDEO_EXT:
                cand = video.with_name(stem + ext)
                if cand.exists():
                    return cand
    return video


def read_audio(video, start, end, ffmpeg):
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(start)]
    if end is not None:
        cmd += ["-t", str(end - start)]
    cmd += ["-i", str(video), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    if len(audio) < NPERSEG:
        raise SystemExit(f"No (or too short) audio track in {video}")
    return audio


def spectrogram(audio, max_hz=1000):
    """Power in dB above the median of each frequency bin (the steady noise floor),
    clipped at 0; frequencies up to max_hz only."""
    window = np.hanning(NPERSEG).astype(np.float32)
    n = (len(audio) - NPERSEG) // HOP + 1
    freqs = np.fft.rfftfreq(NPERSEG, 1 / RATE)
    keep = freqs <= max_hz
    power = np.empty((keep.sum(), n), np.float32)
    for a in range(0, n, 2000):
        idx = np.arange(a, min(n, a + 2000))[:, None] * HOP + np.arange(NPERSEG)
        spec = np.fft.rfft(audio[idx] * window, axis=1)[:, keep]
        power[:, a:a + len(idx)] = (10 * np.log10(np.abs(spec) ** 2 + 1e-12)).T
    times = (np.arange(n) * HOP + NPERSEG / 2) / RATE
    return freqs[keep], times, np.clip(power - np.median(power, axis=1, keepdims=True), 0, None)


def at_freq(spec, freqs, fr):
    """Spectrogram values at (fractional) frequencies fr, shape fr.shape + (frames,)."""
    idx = np.interp(fr, freqs, np.arange(len(freqs)))
    lo = np.clip(np.floor(idx).astype(int), 0, len(freqs) - 2)
    w = (idx - lo)[..., None]
    return spec[lo] * (1 - w) + spec[lo + 1] * w


def track_pitch(spec, freqs, fmin, fmax):
    """(f0 per frame, confidence): harmonic comb score + Viterbi on a log-frequency grid."""
    cand = np.exp(np.linspace(np.log(fmin), np.log(fmax), int(np.log(fmax / fmin) / 0.005)))
    score = np.empty((len(cand), spec.shape[1]), np.float32)
    for i, c in enumerate(cand):
        k = np.arange(1, max(2, int(MAX_HARMONIC_HZ // c)) + 1)
        score[i] = at_freq(spec, freqs, c * k).mean(0) - at_freq(spec, freqs, c * (k + 0.5)).mean(0)
    score = np.clip(score, 0, None)
    peak = score.max(0)
    norm = score / (peak + 1e-9)
    d = np.abs(np.log(cand)[:, None] - np.log(cand)[None, :])
    trans = np.where(d < 0.045, (d / 0.015) ** 2, 12.0).astype(np.float32)
    cost, back = -norm[:, 0].copy(), np.zeros(norm.shape, np.int32)
    rows = np.arange(len(cand))
    for i in range(1, norm.shape[1]):
        total = cost[None, :] + trans
        back[:, i] = np.argmin(total, axis=1)
        cost = total[rows, back[:, i]] - 3 * norm[:, i]
    path = np.empty(norm.shape[1], int)
    path[-1] = np.argmin(cost)
    for i in range(norm.shape[1] - 1, 0, -1):
        path[i - 1] = back[path[i], i]
    return cand[path], norm[path, np.arange(len(path))] * peak


def engine_load(spec, freqs, f0):
    """Share of the upper harmonics (3rd-8th) against the 1st/2nd, smoothed over 0.45 s."""
    frames = np.arange(len(f0))
    h = np.array([spec[np.clip(np.searchsorted(freqs, f0 * k), 0, len(freqs) - 1), frames] for k in range(1, 9)])
    upper, lower = h[2:].sum(0), h[:2].sum(0)
    return np.convolve(upper / (upper + 2 * lower + 1e-9), np.ones(9) / 9, "same")


def jumps(f0):
    lf = np.log(f0)
    out = np.zeros(len(f0))
    out[JUMP_WINDOW:] = lf[JUMP_WINDOW:] - lf[:-JUMP_WINDOW]
    return out


def steady_segments(t, f0, load, min_f0, min_len=2.0, max_drift=0.08):
    """[(i, j)] index ranges without pitch jumps or abrupt load changes, pitch
    within +-max_drift (log) of the segment mean, at least min_len seconds."""
    dt = t[1] - t[0]
    smooth = np.convolve(load, np.ones(11) / 11, "same")
    dload = np.zeros(len(t))
    dload[JUMP_WINDOW:] = np.abs(smooth[JUMP_WINDOW:] - smooth[:-JUMP_WINDOW])
    ok = (np.abs(jumps(f0)) < 0.04) & (dload < 0.12) & (f0 > min_f0)
    lf, segs, i = np.log(f0), [], 0
    while i < len(t):
        if not ok[i]:
            i += 1
            continue
        j, total = i, lf[i]
        while j + 1 < len(t) and ok[j + 1] and abs(lf[j + 1] - total / (j - i + 1)) < max_drift:
            j += 1
            total += lf[j]
        if (j - i) * dt >= min_len:
            segs.append((i, j))
        i = j + 1
    return segs


def find_events(t, f0, load, a, b, step):
    """Suspicious moments between frame indices a..b: [(kind, time, value)].
    kind: up / down / down-braking (gear shift), rise / fall (throttle), coast
    (closed throttle, value = duration)."""
    dt = t[1] - t[0]
    jump, lf = jumps(f0), np.log(f0)
    low = np.convolve(load, np.ones(11) / 11, "same") < np.median(load) - 0.12
    coast_near = lambda i: low[max(0, i - int(1 / dt)):i + int(1 / dt)].any()
    events, i = [], max(a, JUMP_WINDOW)
    while i <= b:
        if abs(jump[i]) > 0.08:
            sign, k = np.sign(jump[i]), i
            while k + 1 <= b and sign * jump[k + 1] > 0.04:
                k += 1
            ratio = float(np.exp(lf[k] - lf[i - JUMP_WINDOW]))
            if abs(np.log(ratio) - np.log(step)) < 0.08:
                kind = "up"
            elif abs(np.log(ratio) + np.log(step)) < 0.08:
                kind = "down"
            elif 1.08 < ratio < 1 / step and coast_near(i):
                kind = "down-braking"
            else:
                kind = "rise" if sign > 0 else "fall"
            events.append((kind, float(t[i]), round(ratio, 3)))
            i = k + JUMP_WINDOW
        elif low[i]:
            k = i
            while k + 1 <= b and low[k + 1]:
                k += 1
            if (k - i) * dt >= 0.5:
                events.append(("coast", float(t[i]), round((k - i) * dt, 2)))
            i = k + 1
        else:
            i += 1
    return events


def read_readings(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(parse_time(r["time_s"]), float(r["speed_kmh"])) for r in csv.DictReader(f)]


def fit_gears(t, f0, segs, readings, lag):
    """Gear constants (speed per Hz) from readings inside steady segments, clustered
    in steps of > 10 %. Returns (constants ascending, {segment index: gear index})."""
    samples = []
    for x, v in readings:
        for s, (i, j) in enumerate(segs):
            if t[i] <= x - lag <= t[j]:
                near = np.abs(t - (x - lag)) < 0.2
                samples.append((v / float(np.median(f0[near])), s))
    if not samples:
        return [], {}
    clusters = []
    for k, s in sorted(samples):
        if clusters and k < 1.10 * np.mean([c[0] for c in clusters[-1]]):
            clusters[-1].append((k, s))
        else:
            clusters.append([(k, s)])
    consts = [float(np.mean([k for k, _ in c])) for c in clusters]
    votes = {}
    for g, c in enumerate(clusters):
        for _, s in c:
            votes.setdefault(s, []).append(g)
    return consts, {s: max(set(g), key=g.count) for s, g in votes.items()}


def propagate(segs, gear, gaps):
    """Carry known gears over gaps: up/down shifts change the gear by one, a gap
    without shift events and with a steady pitch keeps it."""
    changed = True
    while changed:
        changed = False
        for n, (ev, ratio) in enumerate(gaps):
            shifts = sum(1 if e[0] == "up" else -1 for e in ev if e[0] in ("up", "down", "down-braking"))
            clear = not any(e[0] == "coast" for e in ev) and (shifts != 0 or 0.9 < ratio < 1.1)
            if not clear:
                continue
            if n in gear and n + 1 not in gear:
                gear[n + 1], changed = gear[n] + shifts, True
            elif n + 1 in gear and n not in gear:
                gear[n], changed = gear[n + 1] - shifts, True
    return gear


def read_fixes(path):
    """[(video_time_s, station_m)] from the fix picker's fixes export."""
    with open(path, newline="", encoding="utf-8") as f:
        return sorted((float(r["video_time_s"]), float(r["station_m"])) for r in csv.DictReader(f)
                      if r.get("station_m") not in (None, ""))


def fix_octaves(t, f0, max_len_s=15.0):
    """Stretches tracked at twice (or half) the pitch: entered and left by big jumps in
    opposite directions (beyond 1.5x, more than a gear step), shorter than max_len_s,
    and at 1.7-2.4x (or the inverse) the pitch around them - corrected by the factor 2.
    Happens in slow hairpins, where the fundamental is weak."""
    lf = np.log(f0).copy()
    w = JUMP_WINDOW
    d = np.zeros(len(lf))
    d[w:] = lf[w:] - lf[:-w]
    big = np.flatnonzero(np.abs(d) > np.log(1.5))
    # one index per jump (the steepest of a run)
    jumps = []
    for i in big:
        if jumps and i - jumps[-1] <= w:
            if abs(d[i]) > abs(d[jumps[-1]]):
                jumps[-1] = i
        else:
            jumps.append(i)
    dt = t[1] - t[0]
    out = f0.copy()
    used = -1
    for a_i, a in enumerate(jumps):
        if a <= used:
            continue
        for b in jumps[a_i + 1:]:
            if (b - a) * dt > max_len_s:
                break
            if np.sign(d[a]) == np.sign(d[b]):
                continue
            inside = np.median(lf[a:b - w + 1]) if b - w + 1 > a else lf[a]
            around = np.median(np.r_[lf[max(0, a - w - int(2 / dt)):a - w + 1], lf[b:b + int(2 / dt)]])
            r = inside - around
            if np.log(1.7) <= abs(r) <= np.log(2.4):
                out[a - w + 1:b - w + 1] = f0[a - w + 1:b - w + 1] / (2.0 if r > 0 else 0.5)
                used = b
            break
    return out


def pitch_glitches(t, f0, ratio=1.35, window_s=3.0):
    """Frames where the tracked pitch is off its surroundings (median over window_s) by more
    than `ratio` - beyond a gear step, so a tracking error (other harmonic, slow hairpins)."""
    from scipy.ndimage import median_filter

    n = max(3, int(window_s / (t[1] - t[0])) | 1)
    ref = median_filter(np.log(f0), n, mode="nearest")
    return np.abs(np.log(f0) - ref) > np.log(ratio)


def gear_path(f0, ks, vref, valid, sigma=0.35, smooth=0.01, change=5.0):
    """Gear index per frame (Viterbi). Two terms:
    - the speed ks[g] * f0 should be close to the reference speed vref (NaN = none),
      relative deviation sigma;
    - the speed is continuous: from one frame (50 ms) to the next it changes by a
      relative `smooth` at most typically, so a pitch jump by the gear step forces the
      gear change at that very moment and a gear change without one is expensive.
    Frames not `valid` (pitch glitches) are bridged; they get the gear before them."""
    idx = np.flatnonzero(valid)
    lk, lf = np.log(ks), np.log(f0[idx])
    em = np.zeros((len(idx), len(ks)))
    m = ~np.isnan(vref[idx])
    em[m] = ((lk[None, :] + lf[m, None] - np.log(vref[idx][m])[:, None]) / sigma) ** 2
    g_count = len(ks)
    cost, back = em[0].copy(), np.zeros((len(idx), g_count), np.int8)
    dk = lk[None, :] - lk[:, None]                     # from gear a (row) to gear b (column)
    jump_cost = np.where(np.abs(np.arange(g_count)[:, None] - np.arange(g_count)[None, :]) > 1, np.inf,
                         change * (np.arange(g_count)[:, None] != np.arange(g_count)[None, :]))
    for i in range(1, len(idx)):
        trans = ((dk + (lf[i] - lf[i - 1])) / smooth) ** 2 + jump_cost
        total = cost[:, None] + trans
        back[i] = np.argmin(total, axis=0)
        cost = total[back[i], np.arange(g_count)] + em[i]
    path_v = np.zeros(len(idx), int)
    path_v[-1] = int(np.argmin(cost))
    for i in range(len(idx) - 1, 0, -1):
        path_v[i - 1] = back[i, path_v[i]]
    path = np.zeros(len(f0), int)
    path[idx] = path_v
    for i in range(1, len(f0)):
        if not valid[i]:
            path[i] = path[i - 1]
    if not valid[0]:
        path[:idx[0]] = path_v[0]
    return path


def speed_from_path(t, f0, ks, path, valid, ratio=1.15, window_s=2.0):
    """Speed per frame. Across pitch glitches, and where the speed is off its surroundings
    (median over window_s) by more than `ratio` - faster than a motorcycle accelerates or
    brakes for that short - it is interpolated linearly."""
    from scipy.ndimage import median_filter

    v = np.asarray(ks)[path] * f0
    n = max(3, int(window_s / (t[1] - t[0])) | 1)
    ok = valid.copy()
    for _ in range(2):
        ref = median_filter(v, n, mode="nearest")
        ok &= np.abs(np.log(v / ref)) <= np.log(ratio)
        v = np.where(ok, v, np.interp(t, t[ok], v[ok]))
    return v


def fit_from_fixes(t, f0, events, fixes, step, gears=6):
    """Gear constants from the distances between fixes, without speed readings.

    The gears form a ladder k0 / step^g (step from the upshift jumps); the gear per
    frame follows the mean speed of its fix interval and may change only at shift
    pitch jumps (continuity of the speed, see gear_path); k0 is chosen so that the integrated speed best matches the distances
    along the route between the fixes. Returns (constants, gear per frame,
    distance errors in m per fix interval).
    On the Samnaun 2 ride the speed agrees with the speedometer (corrected by its 6 %
    over-reading) within median 0.7 km/h, 90 % 2 km/h."""
    dt = t[1] - t[0]
    pairs = [(a, b, sa, sb) for (a, sa), (b, sb) in zip(fixes, fixes[1:]) if a >= t[0] and b <= t[-1] and b > a]
    if not pairs:
        raise SystemExit("No two fixes inside the analysed time range")
    vref = np.full(len(t), np.nan)
    for a, b, sa, sb in pairs:
        vref[(t >= a) & (t < b)] = (sb - sa) / (b - a) * 3.6
    valid = ~pitch_glitches(t, f0)
    inside = ~np.isnan(vref) & (vref > 3)
    k_mid = float(np.median(vref[inside] / f0[inside]))
    # ladder around the typical gear: two gears below it, the rest above (helmet videos on
    # mountain roads: mostly 2nd/3rd, shifts up on straights and in tunnels)
    ladder = step ** -(np.arange(gears) - 2)

    def error(k0):
        ks = k0 * ladder
        path = gear_path(f0, ks, vref, valid)
        v = speed_from_path(t, f0, ks, path, valid)
        e = [v[(t >= a) & (t < b)].sum() * dt / 3.6 - (sb - sa) for a, b, sa, sb in pairs]
        return float(np.sum(np.abs(e))), ks, path, e

    grid = k_mid * np.exp(np.arange(np.log(step) * 0.6, -np.log(step) * 0.6, 0.02))
    best = min((error(k) for k in grid), key=lambda r: r[0])
    k_best = best[1][2]
    for k in k_best * np.exp(np.arange(-0.02, 0.0201, 0.004)):
        r = error(k)
        if r[0] < best[0]:
            best = r
    _, ks, path, e = best
    used = sorted(set(path.tolist()))
    return [float(k) for k in ks], path, e, used, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--start", default="0", help="start time (s, mm:ss, h:mm:ss)")
    parser.add_argument("--end", help="end time")
    parser.add_argument("--cylinders", type=int, default=3, help="cylinders of the four-stroke engine (default 3)")
    parser.add_argument("--rpm", type=float, nargs=2, default=(800, 7000), metavar=("MIN", "MAX"),
                        help="engine speed range searched (default 800 7000)")
    parser.add_argument("--min-drive-rpm", type=float, default=1800,
                        help="steady segments only above this engine speed (default 1800)")
    parser.add_argument("--fixes", help="fixes CSV from the fix picker: fit the gears from the distances between them")
    parser.add_argument("--readings", help="CSV time_s,speed_kmh (e.g. speedometer readings) to fit the gears")
    parser.add_argument("--lag", type=float, default=0.7, help="delay of the readings behind the sound, s (speedometer: 0.7)")
    parser.add_argument("--gear-step", type=float, default=0.79,
                        help="rpm ratio of an upshift if it cannot be fitted from readings (default 0.79)")
    parser.add_argument("--first-gear", type=int, default=1, help="number of the lowest gear found (for labels)")
    parser.add_argument("--ffmpeg", help="path to ffmpeg")
    parser.add_argument("--out", help="output folder (default data/audio/<video>)")
    args = parser.parse_args()

    video = source_video(args.video)
    start = parse_time(args.start)
    end = parse_time(args.end) if args.end else None
    stem = Path(args.video).stem.removesuffix("_work").removesuffix("_full")
    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "data" / "audio" / stem
    out.mkdir(parents=True, exist_ok=True)
    hz_per_rpm = args.cylinders / 120

    print(f"Reading audio of {video.name} ...", flush=True)
    audio = read_audio(video, start, end, find_ffmpeg(args.ffmpeg))
    freqs, t, spec = spectrogram(audio)
    t = t + start
    print(f"  {len(audio) / RATE:.0f} s, {len(t)} frames; tracking pitch ...", flush=True)
    f0, conf = track_pitch(spec, freqs, args.rpm[0] * hz_per_rpm, args.rpm[1] * hz_per_rpm)
    f0 = fix_octaves(t, f0)
    load = engine_load(spec, freqs, f0)
    segs = steady_segments(t, f0, load, args.min_drive_rpm * hz_per_rpm)

    consts, gear, frame_gear = [], {}, None
    readings = read_readings(args.readings) if args.readings else []
    if args.fixes:
        events = find_events(t, f0, load, 0, len(t) - 1, args.gear_step)
        ups = [r for k, _, r in events if k == "up"]
        fstep = float(np.median(ups)) if len(ups) >= 2 else args.gear_step
        events = find_events(t, f0, load, 0, len(t) - 1, fstep)
        consts, frame_gear, errors, used, valid = fit_from_fixes(t, f0, events, read_fixes(args.fixes), fstep)
        print(f"  gear step {fstep:.3f} from {len(ups)} upshifts; distance error between fixes: "
              f"median {np.median(np.abs(errors)):.0f} m, max {np.max(np.abs(errors)):.0f} m")
        for s_, (i, j) in enumerate(segs):
            vals = frame_gear[i:j + 1]
            gear[s_] = int(np.bincount(vals).argmax())
    elif readings:
        consts, gear = fit_gears(t, f0, segs, readings, args.lag)
    step = float(np.exp(np.mean(np.diff(np.log(consts))))) ** -1 if len(consts) > 1 else args.gear_step
    gaps = []
    for (i0, j0), (i1, _) in zip(segs, segs[1:]):
        before, after = np.median(f0[max(i0, j0 - 6):j0 + 1]), np.median(f0[i1:i1 + 7])
        gaps.append((find_events(t, f0, load, j0 - 2, i1 + 2, step), float(after / before)))
    if gear and frame_gear is None:
        gear = propagate(segs, gear, gaps)

    seg_of = np.full(len(t), -1)
    for s, (i, j) in enumerate(segs):
        seg_of[i:j + 1] = s
    name = lambda g: "" if g is None else str(g + args.first_gear)
    speed = np.full(len(t), np.nan)
    if frame_gear is not None:        # every frame has a gear
        speed = speed_from_path(t, f0, consts, frame_gear, valid)
    else:
        for s, (i, j) in enumerate(segs):
            if 0 <= gear.get(s, -1) < len(consts):
                speed[i:j + 1] = consts[gear[s]] * f0[i:j + 1]

    if frame_gear is not None:
        w_ = JUMP_WINDOW * 2                      # 0.5 s
        dv = np.zeros(len(t))
        dv[w_:] = speed[w_:] - speed[:-w_]
        odd = []
        for i in range(w_, len(t)):
            if abs(dv[i]) > 8 and abs(dv[i]) >= np.abs(dv[max(0, i - w_):i + w_]).max() and (not odd or t[i] - odd[-1] > 0.5):
                odd.append(float(t[i]))
        if odd:
            print(f"  implausible speed jumps (> 8 km/h in 0.5 s) left at {', '.join(f'{x:.1f}' for x in odd)} s - "
                  "check there (tunnel echo, undetected shift)")
    with open(out / "track.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "f0_hz", "rpm", "load", "confidence", "segment", "gear", "speed_kmh"])
        for n in range(len(t)):
            s = int(seg_of[n])
            w.writerow([f"{t[n]:.2f}", f"{f0[n]:.2f}", round(f0[n] / hz_per_rpm), f"{load[n]:.3f}", f"{conf[n]:.2f}",
                        s if s >= 0 else "",
                        name(int(frame_gear[n])) if frame_gear is not None else name(gear.get(s)) if s >= 0 else "",
                        "" if np.isnan(speed[n]) else f"{speed[n]:.1f}"])
    with open(out / "segments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment", "start_s", "end_s", "f0_mean_hz", "rpm_mean", "f0_slope_hz_s", "load", "gear", "speed_mean_kmh"])
        for s, (i, j) in enumerate(segs):
            slope = np.polyfit(t[i:j + 1], f0[i:j + 1], 1)[0]
            v = speed[i:j + 1]
            w.writerow([s, f"{t[i]:.2f}", f"{t[j]:.2f}", f"{f0[i:j + 1].mean():.1f}", round(f0[i:j + 1].mean() / hz_per_rpm),
                        f"{slope:.2f}", f"{load[i:j + 1].mean():.3f}", name(gear.get(s)),
                        "" if np.isnan(v).all() else f"{np.nanmean(v):.1f}"])

    print(f"  {len(segs)} steady segments, {sum(t[j] - t[i] for i, j in segs):.0f} of {t[-1] - t[0]:.0f} s")
    if consts:
        print("  gears: " + ", ".join(f"{name(g)}: {k:.3f} km/h per Hz = {1 / (k * hz_per_rpm):.1f} rpm per km/h"
                                       for g, k in enumerate(consts)) + (f"; step {step:.3f}" if len(consts) > 1 else ""))
        with open(out / "gears.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gear", "kmh_per_hz", "rpm_per_kmh"])
            for g, k in enumerate(consts):
                w.writerow([name(g), f"{k:.4f}", f"{1 / (k * hz_per_rpm):.1f}"])
    with open(out / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gap_from_s", "gap_to_s", "gear_before", "gear_after", "shifts_found", "check", "events"])
        for n, ((ev, ratio), (_, j0), (i1, _)) in enumerate(zip(gaps, segs, segs[1:])):
            ga, gb = gear.get(n), gear.get(n + 1)
            found = sum(1 if e[0] == "up" else -1 for e in ev if e[0] in ("up", "down", "down-braking"))
            check = ""
            if ga is not None and gb is not None and found != gb - ga:
                coasts = [f"{e[1]:.1f}" for e in ev if e[0] == "coast"]
                check = f"{gb - ga - found:+d} shift(s) missing" + (f", coasting at {', '.join(coasts)} s" if coasts else "")
            if ev or check:
                text = "; ".join(f"{k} {tt:.1f}s {'x' if k != 'coast' else ''}{v}{'s' if k == 'coast' else ''}" for k, tt, v in ev)
                w.writerow([f"{t[j0]:.2f}", f"{t[i1]:.2f}", name(ga), name(gb), found, check, text])
                print(f"  {t[j0]:6.1f}-{t[i1]:6.1f} s  gear {name(ga) or '?'} -> {name(gb) or '?'}  {text}" + (f"  !! {check}" if check else ""))
    if readings and consts:
        err = [abs(speed[np.argmin(np.abs(t - (x - args.lag)))] - v) for x, v in readings]
        err = [e for e in err if not np.isnan(e)]
        if err:
            print(f"  speed vs readings: median {np.median(err):.1f}, max {max(err):.1f} km/h ({len(err)} readings)")
    try:
        plot(out / "plot.png", freqs, t, spec, f0, load, segs, gear, consts, speed, readings, args.lag, name)
    except ImportError:
        pass
    print(f"  -> {out}")


def plot(path, freqs, t, spec, f0, load, segs, gear, consts, speed, readings, lag, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 3 if consts else 2
    width = min(60, max(12, (t[-1] - t[0]) / 12))
    fig, ax = plt.subplots(rows, 1, figsize=(width, 3.2 * rows), sharex=True)
    sel = freqs <= 600
    step = max(1, len(t) // 6000)
    ax[0].pcolormesh(t[::step], freqs[sel], spec[sel][:, ::step], vmin=0, vmax=25, shading="auto", cmap="magma")
    for k in (1, 2, 3, 4):
        ax[0].plot(t, f0 * k, color="cyan", lw=0.4)
    ax[0].set_ylabel("Hz")
    ax[1].plot(t, f0, lw=0.9)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for s, (i, j) in enumerate(segs):
        g = gear.get(s)
        ax[1].axvspan(t[i], t[j], color="grey" if g is None else colours[(g + 1) % len(colours)], alpha=0.18)
        if g is not None:
            ax[1].text((t[i] + t[j]) / 2, f0[i:j + 1].max(), name(g), ha="center", va="bottom", fontsize=8)
    ax[1].set_ylabel("f0 Hz")
    twin = ax[1].twinx()
    twin.fill_between(t, 0, load, color="green", alpha=0.12)
    twin.set_ylim(0, 1.5)
    twin.set_ylabel("load", color="green")
    if consts:
        ax[2].plot(t, speed, lw=1)
        if readings:
            ax[2].plot([x - lag for x, _ in readings], [v for _, v in readings], "o", ms=3, color="red")
        ax[2].set_ylabel("km/h")
    for a in ax:
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("s")
    plt.tight_layout()
    plt.savefig(path, dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    main()
