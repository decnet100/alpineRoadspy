"""Prepare traffic signs for review: full-resolution crops and suggestions.

For every sign object of tools/track.py:
  1. crops: the largest and sharpest views, cut from the original frames
     (full resolution, e.g. 4K), one pass through the video;
  2. catalog match: image similarity (SigLIP) of the crops against the
     reference drawings of catalog/signs.csv -> ranked sign codes;
  3. reading (optional, --vlm): a local vision-language model reads the sign
     (code among the candidates, value, text lines exactly as printed, names
     per language, destinations of direction signs).

    python tools/signs.py videos/brenner.mp4 [--vlm Qwen/Qwen2.5-VL-7B-Instruct]

Writes data/signs/<video>/signs.json and crops/; review them in
tools/sign_review.html (via tools/serve.py). Models are downloaded once from
Hugging Face; images never leave the machine.
"""

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from videoio import full_resolution

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog" / "signs.csv"
CACHE = ROOT / "data" / "catalog_cache"
EMBED_MODEL = "google/siglip-base-patch16-224"
USER_AGENT = "alpineRoadspy (https://github.com/decnet100/alpineRoadspy)"
VIEWS = 3        # crops kept per sign
CANDIDATES = 8   # catalog suggestions per sign


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------- crops ----------

def pick_frames(dets, n=6):
    """The n largest views (distinct frames) of an object; sharpness decides later."""
    by_area = sorted(dets, key=lambda d: (d[3] - d[1]) * (d[4] - d[2]), reverse=True)
    seen, out = set(), []
    for d in by_area:
        if d[0] not in seen:
            seen.add(d[0])
            out.append(d)
        if len(out) == n:
            break
    return out


def extract_crops(video, wanted, out_dir, scale=1.0):
    """wanted: {object id: [detections]} -> {object id: [(file, sharpness, area)]}, one pass through the video.
    `scale` converts boxes to the pixels of `video` (full-resolution copy)."""
    by_frame = {}
    for oid, dets in wanted.items():
        for d in dets:
            by_frame.setdefault(d[0], []).append((oid, d))
    targets = sorted(by_frame)
    found = {oid: [] for oid in wanted}
    if not targets:
        return found
    cap = cv2.VideoCapture(str(video))
    frame = 0
    if targets[0] > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, targets[0])
        frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame > targets[0]:
            cap, frame = cv2.VideoCapture(str(video)), 0
    started = time.time()
    for n, target in enumerate(targets, 1):
        while frame < target and cap.grab():
            frame += 1
        ok, img = cap.read()
        frame += 1
        if not ok:
            break
        h, w = img.shape[:2]
        for oid, (f, *box) in by_frame[target]:
            x1, y1, x2, y2 = (v * scale for v in box)
            pad = 0.3 * max(x2 - x1, y2 - y1, 12)
            X1, Y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
            X2, Y2 = min(w, int(x2 + pad)), min(h, int(y2 + pad))
            crop = img[Y1:Y2, X1:X2]
            if crop.size == 0:
                continue
            sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            name = f"{oid}_{f}.jpg"
            cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            found[oid].append((name, sharp, (x2 - x1) * (y2 - y1)))
        if n % max(1, len(targets) // 10) == 0:
            print(f"  crops: {n}/{len(targets)} frames, {time.time() - started:.0f} s", flush=True)
    return found


def best_views(views, crop_dir):
    """Keep the VIEWS crops with the best mix of size and sharpness; delete the rest."""
    ranked = sorted(views, key=lambda v: v[1] * v[2] ** 0.5, reverse=True)
    for name, *_ in ranked[VIEWS:]:
        (crop_dir / name).unlink(missing_ok=True)
    return [v[0] for v in ranked[:VIEWS]]


# ---------- catalog ----------

def catalog_images(catalog):
    """Reference drawings as PNG (cached under data/catalog_cache), politely one at a time."""
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = {}
    for row in catalog:
        path = CACHE / (re.sub(r"[^\w.-]+", "_", row["file"]) + ".png")
        if not path.exists():
            req = urllib.request.Request(row["image_url"] + "?width=224", headers={"User-Agent": USER_AGENT})
            for attempt in range(3):
                try:
                    path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
                    break
                except Exception as exc:
                    if attempt == 2:
                        print(f"  no image for {row['code']} ({row['file']}): {exc}")
                    time.sleep(2 + 3 * attempt)
            time.sleep(0.2)
        if path.exists():
            paths[row["file"]] = path
    return paths


def load_rgb(path, background=128):
    """Image as RGB; transparent parts of the drawings on neutral grey."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Embedder:
    def __init__(self, device):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(EMBED_MODEL)   # images only, no tokenizer needed
        self.model = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()

    def __call__(self, images, batch=32):
        out = []
        for i in range(0, len(images), batch):
            inputs = self.processor(images=images[i:i + batch], return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                emb = self.model.get_image_features(**inputs)
            emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb
            out.append(self.torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))


# ---------- reading with a vision-language model ----------

PROMPT = """Das Bild zeigt ein Verkehrszeichen (oder eine Tafel) von einer Fahrt in Österreich oder der Schweiz.
Wahrscheinliche Zeichen aus dem amtlichen Katalog (Code – Bezeichnung):
{candidates}

Antworte ausschließlich mit einem JSON-Objekt:
{{"code": "<Code aus der Liste oder 'other' oder 'no_sign'>",
  "value": <Zahl auf dem Zeichen, z. B. Tempo oder Gewicht, sonst null>,
  "text_lines": [<jede Textzeile genau so wie aufgedruckt, von oben nach unten>],
  "names": {{"<Sprachcode de/it/rm/lld/fr>": "<Ortsname>"}},
  "destinations": [{{"lines": [<Zeilen eines Ziels>], "km": <Zahl oder null>, "ref": "<Straßennummer oder null>", "arrow": "<left|right|straight|null>"}}],
  "confidence": <0 bis 1>}}
Schreibe nur, was lesbar ist; unleserliche Zeilen weglassen. Mehrsprachige Namen alle angeben."""


class Reader:
    def __init__(self, model_name, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype).to(device).eval()

    def __call__(self, image_rgb, candidates):
        from PIL import Image

        text = PROMPT.format(candidates="\n".join(f"{c['code']} – {c['name']}" for c in candidates))
        messages = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(image_rgb)},
                                                 {"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=400, do_sample=False)
        answer = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        match = re.search(r"\{.*\}", answer, re.S)
        try:
            return json.loads(match.group(0)) if match else {"raw": answer}
        except json.JSONDecodeError:
            return {"raw": answer}


def upscale(img, min_side=448):
    """Small crops are enlarged so the reader sees enough detail."""
    h, w = img.shape[:2]
    f = min_side / max(1, min(h, w))
    return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC) if f > 1 else img


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--objects", type=Path, help="default data/objects/<video>/")
    parser.add_argument("--out", type=Path, help="default data/signs/<video>/")
    parser.add_argument("--vlm", help="vision-language model to read the signs, e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--include-uncertain", action="store_true", help="also signs marked uncertain by track.py")
    parser.add_argument("--device", default=None)
    parser.add_argument("--full-res", help="video to cut crops from (default: <name>_full.mp4 next to a <name>_work.mp4)")
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    video = Path(args.video)
    obj_dir = args.objects or ROOT / "data" / "objects" / video.stem
    out_dir = args.out or ROOT / "data" / "signs" / video.stem
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    objects = [o for o in read_csv(obj_dir / "objects.csv")
               if o["group"] == "sign" and (args.include_uncertain or o["status"] == "ok")]
    tracks = json.loads((obj_dir / "tracks.json").read_text(encoding="utf-8"))
    print(f"{len(objects)} signs from {obj_dir}", flush=True)

    crop_video, scale = full_resolution(video, args.full_res)
    print(f"Cutting views from {crop_video.name} ...", flush=True)
    views = extract_crops(crop_video, {o["id"]: pick_frames(tracks[o["id"]]) for o in objects}, crop_dir, scale)
    crops = {oid: best_views(v, crop_dir) for oid, v in views.items()}

    catalog = read_csv(CATALOG)
    print(f"Catalog: {len(catalog)} signs; fetching reference drawings (cached) ...", flush=True)
    images = catalog_images(catalog)
    refs = [(row, load_rgb(images[row["file"]])) for row in catalog if row["file"] in images]
    refs = [(row, img) for row, img in refs if img is not None]

    print(f"Matching against the catalog with {EMBED_MODEL} on {device} ...", flush=True)
    embed = Embedder(device)
    ref_emb = embed([img for _, img in refs])
    reader = Reader(args.vlm, device) if args.vlm else None
    if reader:
        print(f"Reading signs with {args.vlm} ...", flush=True)

    signs, started = [], time.time()
    for i, o in enumerate(objects, 1):
        imgs = [load_rgb(crop_dir / c) for c in crops[o["id"]]]
        imgs = [im for im in imgs if im is not None]
        suggestions = []
        if imgs:
            sims = embed(imgs) @ ref_emb.T          # views x catalog
            best = sims.max(axis=0)
            for k in np.argsort(-best):
                row = refs[k][0]
                if any(s["code"] == row["code"] for s in suggestions):
                    continue   # one entry per code (variants share it)
                suggestions.append({"code": row["code"], "name": row["name"], "kind": row["kind"],
                                    "file": row["file"], "score": round(float(best[k]), 3)})
                if len(suggestions) == CANDIDATES:
                    break
        reading = reader(upscale(imgs[0]), suggestions) if reader and imgs else None
        signs.append({"id": o["id"], "class": o["class"], "station_m": float(o["station_m"]), "side": o["side"],
                      "lon": float(o["lon"]), "lat": float(o["lat"]), "pass_time_s": float(o["pass_time_s"]),
                      "crops": [f"crops/{c}" for c in crops[o["id"]]], "suggestions": suggestions, "reading": reading})
        if reader and (i % 10 == 0 or i == len(objects)):
            print(f"  read {i}/{len(objects)}, {time.time() - started:.0f} s", flush=True)

    meta = {"video": video.name, "video_url": None, "embed_model": EMBED_MODEL, "reader": args.vlm,
            "created": time.strftime("%Y-%m-%d %H:%M")}
    try:
        meta["video_url"] = "/" + video.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        pass
    (out_dir / "signs.json").write_text(json.dumps({"meta": meta, "signs": signs}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"Wrote {len(signs)} signs to {out_dir / 'signs.json'}")
    print(f"Review: python tools/serve.py, then http://localhost:8000/tools/sign_review.html?signs=/"
          + (out_dir / "signs.json").relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else "")


if __name__ == "__main__":
    main()
