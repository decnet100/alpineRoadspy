"""Detect roadside features in a video drive (first pass, per frame).

Runs a semantic segmentation model trained on Mapillary Vistas (Mask2Former,
Swin-L) on every n-th frame and turns the classes of interest into objects
(connected regions with bounding box): traffic signs, poles (delineator posts
mostly show up as "Pole"), street lights, guardrails.

    python tools/detect.py videos/brenner.mp4 --step 3 --trajectory data/brenner_trajectory.csv

Writes to data/detect/<video>/:
    detections.csv   one row per region and frame (+ station/position from the trajectory)
    guardrails.csv   guardrail presence left/right per frame
    overlays/        frames with the classes coloured in
    index.html       contact sheet; click a frame to open it in the fix picker

Uses the GPU (CUDA, half precision) when available. The model is downloaded
once from Hugging Face (about 0.9 GB); the video never leaves the machine.
Note: Mapillary Vistas models are licensed for non-commercial use.
"""

import argparse
import csv
import html
import os
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

# Vistas class name -> (output class, BGR colour for overlays)
CLASSES = {
    "Traffic Sign (Front)": ("traffic_sign", (255, 255, 0)),
    "Traffic Sign (Back)": ("traffic_sign_back", (255, 128, 0)),
    "Traffic Sign Frame": ("sign_frame", (0, 200, 0)),
    "Pole": ("pole", (255, 0, 255)),
    "Utility Pole": ("utility_pole", (128, 0, 255)),
    "Street Light": ("street_light", (0, 255, 255)),
    "Guard Rail": ("guardrail", (0, 140, 255)),
}
MIN_AREA = {"guardrail": 400}  # pixels; default below
DEFAULT_MIN_AREA = 30


def load_model(device):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).to(device, dtype).eval()
    return processor, model, dtype


def segment(processor, model, dtype, frames_rgb):
    import torch

    inputs = processor(images=frames_rgb, return_tensors="pt")
    inputs = {k: v.to(model.device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    sizes = [f.shape[:2] for f in frames_rgb]
    return [s.cpu().numpy() for s in processor.post_process_semantic_segmentation(out, target_sizes=sizes)]


def regions(seg, class_ids):
    """Connected regions per class of interest: (class, x1, y1, x2, y2, area, cx, cy)."""
    found = []
    for vistas_id, name in class_ids.items():
        mask = (seg == vistas_id).astype(np.uint8)
        if not mask.any():
            continue
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= MIN_AREA.get(name, DEFAULT_MIN_AREA):
                found.append((name, x, y, x + w - 1, y + h - 1, int(area), *centroids[i]))
    return found


def overlay(frame_bgr, seg, class_ids, colours, found):
    out = frame_bgr.astype(np.float32)
    for vistas_id, name in class_ids.items():
        m = seg == vistas_id
        out[m] = out[m] * 0.4 + np.array(colours[name]) * 0.6
    out = out.astype(np.uint8)
    for name, x1, y1, x2, y2, *_ in found:
        if name != "guardrail":
            cv2.rectangle(out, (int(x1) - 3, int(y1) - 3), (int(x2) + 3, int(y2) + 3), colours[name], 1)
    return out


def load_trajectory(path):
    """Trajectory CSV from fix_picker.html, indexed by video time."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    times = np.array([float(r["video_time_s"]) for r in rows])
    return times, rows


def lookup(trajectory, t):
    if trajectory is None:
        return {}
    times, rows = trajectory
    i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    if abs(times[i - 1] - t) < abs(times[i] - t):
        i -= 1
    if abs(times[i] - t) > 0.5:  # outside the fixed part of the drive
        return {}
    r = rows[i]
    return {k: r[k] for k in ("station_m", "lon", "lat", "heading_deg")}


def write_index(out_dir, video_name, video_url, frames, counts_by_frame, names):
    picker = Path(os.path.relpath(ROOT / "tools" / "fix_picker.html", out_dir)).as_posix()

    def cell(frame, t):
        counts = counts_by_frame.get(frame, {})
        tags = " ".join(f'<span class="c-{n}">{n} {c}</span>' for n, c in sorted(counts.items()))
        data = " ".join(f"data-{n}=1" for n in counts)
        link = f"{picker}?video={html.escape(video_url)}&amp;t={t:.3f}" if video_url else f"overlays/{frame:06d}.jpg"
        return (f'<a class="cell" {data} href="{link}" target="_blank">'
                f'<img loading="lazy" src="overlays/{frame:06d}.jpg"><div>{t:.1f} s · F {frame}<br>{tags}</div></a>')

    filters = " ".join(f'<label><input type="checkbox" value="{n}"> {n}</label>' for n in names)
    page = f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><title>{html.escape(video_name)} – Erkennung</title>
<style>
body {{ font: 13px system-ui, sans-serif; margin: 10px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 8px; }}
.cell {{ color: inherit; text-decoration: none; border: 1px solid #ddd; }}
.cell img {{ width: 100%; display: block; }}
.cell div {{ padding: 3px 5px; }}
.cell span {{ display: inline-block; margin-right: 4px; padding: 0 3px; background: #eee; border-radius: 3px; }}
.hide {{ display: none; }}
</style></head><body>
<h3>{html.escape(video_name)} – {len(frames)} Frames ausgewertet</h3>
<p>Nur Frames zeigen mit: {filters} · Klick öffnet die Stelle im Fixpunkt-Tool (über tools/serve.py).</p>
<div class="grid">{''.join(cell(f, t) for f, t in frames)}</div>
<script>
const boxes = [...document.querySelectorAll('input[type=checkbox]')];
boxes.forEach((b) => b.onchange = () => {{
  const on = boxes.filter((x) => x.checked).map((x) => x.value);
  document.querySelectorAll('.cell').forEach((c) => c.classList.toggle('hide', on.length > 0 && !on.some((n) => c.dataset[n])));
}});
</script></body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--step", type=int, default=10, help="evaluate every n-th frame (default 10)")
    parser.add_argument("--start", type=float, default=0, help="start time in s")
    parser.add_argument("--end", type=float, default=None, help="end time in s")
    parser.add_argument("--batch", type=int, default=4, help="frames per model call (GPU)")
    parser.add_argument("--trajectory", help="per-frame trajectory CSV exported from fix_picker.html")
    parser.add_argument("--out", type=Path, help="output folder (default data/detect/<video>)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    video = Path(args.video)
    out_dir = args.out or ROOT / "data" / "detect" / video.stem
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    trajectory = load_trajectory(args.trajectory) if args.trajectory else None
    try:
        video_url = "/" + video.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        video_url = None  # outside the repo: not reachable through serve.py

    print(f"Loading {MODEL} on {device} ...", flush=True)
    processor, model, dtype = load_model(device)
    labels = model.config.id2label
    class_ids = {i: CLASSES[n][0] for i, n in labels.items() if n in CLASSES}
    colours = {CLASSES[n][0]: CLASSES[n][1] for n in CLASSES}

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(args.start * fps)
    last = min(total - 1, int(args.end * fps)) if args.end else total - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    print(f"{video.name}: {total} frames at {fps:.2f} fps, evaluating {first}..{last} every {args.step}", flush=True)

    det_file = open(out_dir / "detections.csv", "w", newline="", encoding="utf-8")
    det = csv.writer(det_file)
    det.writerow(["frame", "video_time_s", "class", "x1", "y1", "x2", "y2", "area_px", "cx", "cy",
                  "image_side", "station_m", "lon", "lat", "heading_deg"])
    rail_file = open(out_dir / "guardrails.csv", "w", newline="", encoding="utf-8")
    rail = csv.writer(rail_file)
    # *_near_px: outer lower image parts, i.e. guardrail next to the vehicle rather than far ahead
    rail.writerow(["frame", "video_time_s", "left_px", "right_px", "left_near_px", "right_near_px", "station_m", "lon", "lat"])

    frames, counts_by_frame, batch = [], {}, []
    started = time.time()

    def flush(batch):
        segs = segment(processor, model, dtype, [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for _, b in batch])
        for (frame, bgr), seg in zip(batch, segs):
            t = frame / fps
            pos = lookup(trajectory, t)
            found = regions(seg, class_ids)
            width = bgr.shape[1]
            counts = {}
            for name, x1, y1, x2, y2, area, cx, cy in found:
                counts[name] = counts.get(name, 0) + 1
                side = "left" if cx < width / 2 else "right"  # image side only; camera roll not compensated yet
                det.writerow([frame, f"{t:.3f}", name, x1, y1, x2, y2, area, f"{cx:.1f}", f"{cy:.1f}", side,
                              pos.get("station_m", ""), pos.get("lon", ""), pos.get("lat", ""), pos.get("heading_deg", "")])
            rail_ids = [i for i, n in class_ids.items() if n == "guardrail"]
            rails = np.isin(seg, rail_ids)
            top = int(rails.shape[0] * 0.3)
            rail.writerow([frame, f"{t:.3f}", int(rails[:, : width // 2].sum()), int(rails[:, width // 2:].sum()),
                           int(rails[top:, : int(width * 0.3)].sum()), int(rails[top:, int(width * 0.7):].sum()),
                           pos.get("station_m", ""), pos.get("lon", ""), pos.get("lat", "")])
            img = overlay(bgr, seg, class_ids, colours, found)
            cv2.imwrite(str(out_dir / "overlays" / f"{frame:06d}.jpg"), cv2.resize(img, (640, int(640 * img.shape[0] / width))),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames.append((frame, t))
            counts_by_frame[frame] = counts
        done = len(frames)
        rate = done / (time.time() - started)
        print(f"  {done} frames, {rate:.2f} frames/s, at {frames[-1][1]:.1f} s", flush=True)

    frame = first
    while frame <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        if (frame - first) % args.step == 0:
            batch.append((frame, bgr))
            if len(batch) >= (args.batch if device == "cuda" else 1):
                flush(batch)
                batch = []
        frame += 1
    if batch:
        flush(batch)
    det_file.close()
    rail_file.close()

    names = sorted({n for c in counts_by_frame.values() for n in c})
    write_index(out_dir, video.name, video_url, frames, counts_by_frame, names)
    totals = {}
    for c in counts_by_frame.values():
        for n, k in c.items():
            totals[n] = totals.get(n, 0) + k
    print(f"Done: {len(frames)} frames in {time.time() - started:.0f} s; regions: {totals}")
    print(f"Open {out_dir / 'index.html'} (or via serve.py: /{out_dir.relative_to(ROOT).as_posix()}/index.html)"
          if out_dir.is_relative_to(ROOT) else f"Open {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
