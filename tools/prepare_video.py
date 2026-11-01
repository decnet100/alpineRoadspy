"""Transcode a video for the tools: H.264, constant frame rate, no audio.

    python tools/prepare_video.py videos/samnaun.webm [--height 1080]

Writes next to the input
    <name>_work.mp4   working copy (default 1080 lines) for detect.py, track.py
                      and the fix picker
    <name>_full.mp4   full resolution, only if the source is larger; track.py
                      and signs.py cut their crops from it automatically
Both come from the same decoded frames in one ffmpeg run, so their frame
numbers match exactly. H.264 because it decodes several times faster than
VP9/HEVC/AV1 and plays in every browser; a keyframe every second keeps
seeking fast. Expect larger files than a YouTube VP9 source: VP9 packs
tighter, and re-encoding keeps its artefacts. Uses NVENC if the ffmpeg build has it, otherwise libx264.
"""

import argparse
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2

COMMON_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
                Fraction(50), Fraction(60000, 1001), Fraction(60)]


def find_ffmpeg(explicit=None):
    if explicit:
        return explicit
    exe = shutil.which("ffmpeg") or os.environ.get("FFMPEG")
    if exe:
        return exe
    # winget installs a link here; a shell opened before the install does not have it on PATH yet
    links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
    if links.exists():
        return str(links)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found: install it (e.g. 'winget install ffmpeg') or 'pip install imageio-ffmpeg'")


def has_encoder(ffmpeg, name):
    out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if f" {name} " not in out:
        return False
    # listed is not enough (no NVIDIA GPU/driver): try a tiny encode
    test = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
                           "-c:v", name, "-f", "null", "-"], capture_output=True)
    return test.returncode == 0


def frame_rate(fps):
    """Snap the container's frame rate to the usual exact values (29.97 -> 30000/1001)."""
    best = min(COMMON_RATES, key=lambda r: abs(float(r) - fps))
    return best if abs(float(best) - fps) < 0.02 else Fraction(fps).limit_denominator(1001)


def encoder_args(nvenc, quality, gop):
    if nvenc:
        args = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"]
    else:
        args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(quality)]
    return args + ["-pix_fmt", "yuv420p", "-g", str(gop), "-an", "-movflags", "+faststart"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--height", type=int, default=1080, help="lines of the working copy (default 1080)")
    parser.add_argument("--cpu", action="store_true", help="use libx264 even if NVENC is available")
    # quality: NVENC -cq / x264 -crf, lower = better and larger. Defaults favour quality over
    # disk space: the full copy is used for sign crops, where fine text must survive
    parser.add_argument("--quality-work", type=int, default=20, help="quality of the working copy (default 20)")
    parser.add_argument("--quality-full", type=int, default=18, help="quality of the full-resolution copy (default 18)")
    parser.add_argument("--keyframe", type=float, default=1, help="seconds between keyframes (default 1; seeking in the fix picker; longer = smaller files)")
    parser.add_argument("--ffmpeg", help="path to ffmpeg (default: from PATH, FFMPEG variable or winget)")
    args = parser.parse_args()

    src = Path(args.video)
    cap = cv2.VideoCapture(str(src))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = frame_rate(cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    if not width:
        raise SystemExit(f"Cannot read {src}")

    ffmpeg = find_ffmpeg(args.ffmpeg)
    nvenc = not args.cpu and has_encoder(ffmpeg, "h264_nvenc")
    gop = max(1, round(float(fps) * args.keyframe))
    work = src.with_name(src.stem + "_work.mp4")
    full = src.with_name(src.stem + "_full.mp4")
    rate = ["-r", f"{fps.numerator}/{fps.denominator}", "-fps_mode", "cfr"]

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "warning", "-stats"]
    if nvenc:
        cmd += ["-hwaccel", "auto"]
    cmd += ["-i", str(src)]
    if height > args.height:
        scale = f"scale=-2:{args.height}:flags=lanczos"
        cmd += ["-filter_complex", f"[0:v]split=2[w][f];[w]{scale}[ws]",
                "-map", "[ws]", *rate, *encoder_args(nvenc, args.quality_work, gop), str(work),
                "-map", "[f]", *rate, *encoder_args(nvenc, args.quality_full, gop), str(full)]
    else:
        cmd += ["-map", "0:v:0", *rate, *encoder_args(nvenc, args.quality_full, gop), str(work)]
        full = None

    print(f"{src.name}: {width}x{height} @ {float(fps):.3f} fps -> {work.name}"
          + (f" ({args.height} lines) + {full.name} (full resolution)" if full else "")
          + f", encoder {'h264_nvenc' if nvenc else 'libx264'}", flush=True)
    subprocess.run(cmd, check=True)

    counts = []
    for path in [work] + ([full] if full else []):
        c = cv2.VideoCapture(str(path))
        counts.append(int(c.get(cv2.CAP_PROP_FRAME_COUNT)))
        print(f"  {path.name}: {int(c.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
              f"{counts[-1]} frames, {c.get(cv2.CAP_PROP_FPS):.3f} fps")
    if len(set(counts)) > 1:
        print("  WARNING: frame counts differ – crops from the full-resolution copy may be off")
    print(f"Next: python tools/detect.py {work.as_posix()} --step 3")


if __name__ == "__main__":
    main()
