"""Serve the repository locally so the browser tools can be used.

Unlike ``python -m http.server`` this handler answers HTTP range requests,
which browsers need for seeking in videos. It binds to localhost only.

    python tools/serve.py [video] [--roads data/roads/gip_linknetz.gpkg]

then open the printed URL. With ``--roads`` (default: the file written by
tools/prepare_gip.py, if present) it also answers ``/api/roads?bbox=...``
with the road links in that box, used by fix_picker.html to follow roads.
"""

import argparse
import functools
import http.server
import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from gip_schema import display_name, find_link_layer, resolve

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROADS = ROOT / "data" / "roads" / "gip_linknetz.gpkg"
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
NAME_FIELDS = ["STRNAME_HOCHRANGIG", "STRNAME", "STRASSENNAME", "NAME", "LABEL"]


class RoadSource:
    """Road lines from a local GeoPackage/Shapefile, queried by WGS84 bbox.

    GIP link networks (old and new export schema, see gip_schema.py) keep their
    node ids so the browser can route on the real topology; other line datasets
    are passed on with a name only. Use the file from tools/prepare_gip.py,
    which also drops links that are not active.
    """

    def __init__(self, path):
        import pyogrio
        from pyproj import CRS, Transformer

        self.path = str(path)
        layers = [name for name, gtype in pyogrio.list_layers(self.path) if gtype and "LINE" in gtype.upper()]
        if not layers:
            raise SystemExit(f"No line layer in {path}")
        self.layer = find_link_layer(layers) or layers[0]  # GIP: LINK, not the non-routable EDGE
        info = pyogrio.read_info(self.path, layer=self.layer)
        self.gip = resolve(info["fields"])
        self.is_gip = {"FROM_NODE", "TO_NODE"} <= self.gip.keys()
        self.new_schema = "NAME_LOCAL" in self.gip
        upper = {f.upper(): f for f in info["fields"]}
        self.name_fields = [upper[n] for n in NAME_FIELDS if n in upper]
        crs = CRS.from_user_input(info["crs"])
        self.to_source = None if crs.equals(CRS.from_epsg(4326)) else \
            Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.read = pyogrio.read_dataframe
        print(f"Roads: {path} (layer {self.layer}, {info['features']} features"
              f"{', GIP topology' if self.is_gip else ''})", flush=True)

    def query(self, lon0, lat0, lon1, lat1):
        bbox = (lon0, lat0, lon1, lat1)
        if self.to_source:
            bbox = self.to_source.transform_bounds(*bbox)
        gdf = self.read(self.path, layer=self.layer, bbox=bbox)
        if gdf.crs and not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs("EPSG:4326")
        features = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            parts = [geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", []))
            if self.is_gip:
                get = lambda key: row.get(self.gip[key]) if key in self.gip else None
                props = {
                    "id": int(get("LINK_ID")) if "LINK_ID" in self.gip else int(idx),
                    "from": int(get("FROM_NODE")),
                    "to": int(get("TO_NODE")),
                    "name": display_name(get, self.new_schema),
                    "category": get("EDGECAT"),
                    # no car access (forest/farm roads, footpaths): usable, but only as a last resort
                    "cost": 3 if get("ONEWAY_CAR") == -1 else 1,
                }
            else:
                props = {"id": int(idx), "name": next((str(row[n]) for n in self.name_fields if row[n] not in (None, "")), "")}
            for k, part in enumerate(parts):
                features.append({
                    "type": "Feature",
                    "properties": dict(props, id=f"{props['id']}" if len(parts) == 1 else f"{props['id']}/{k}"),
                    "geometry": {"type": "LineString",
                                 "coordinates": [[round(x, 7), round(y, 7)] for x, y, *_ in part.coords]},
                })
        return {"type": "FeatureCollection", "features": features}


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    roads = None

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/roads":
            return self.send_roads(parse_qs(url.query))
        return super().do_GET()

    def send_roads(self, query):
        if self.roads is None:
            return self.send_json({"error": "no road file configured (serve.py --roads)"}, 404)
        if "bbox" not in query:
            return self.send_json({"available": True, "source": Path(self.roads.path).name,
                                   "gip": self.roads.is_gip})
        try:
            lon0, lat0, lon1, lat1 = map(float, query["bbox"][0].split(","))
            return self.send_json(self.roads.query(lon0, lat0, lon1, lat1))
        except Exception as exc:  # report to the browser instead of dropping the connection
            return self.send_json({"error": str(exc)}, 500)

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def send_head(self):
        self._range_remaining = None
        match = RANGE_RE.match(self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not match or os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(f.fileno()).st_size
        start, end = match.groups()
        if start:
            start, end = int(start), min(int(end) if end else size - 1, size - 1)
        elif end:  # suffix range: last N bytes
            start, end = max(size - int(end), 0), size - 1
        else:
            f.close()
            return super().send_head()
        if start > end:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        f.seek(start)
        self._range_remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        try:
            if self._range_remaining is None:
                return super().copyfile(source, outputfile)
            remaining = self._range_remaining
            while remaining > 0:
                chunk = source.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser aborted the request while seeking

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", nargs="?", help="video path relative to the repo, e.g. videos/brenner.mp4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--roads", type=Path, help=f"road lines (GeoPackage/Shapefile); default {DEFAULT_ROADS.relative_to(ROOT)} if present")
    args = parser.parse_args()

    roads = args.roads or (DEFAULT_ROADS if DEFAULT_ROADS.exists() else None)
    if roads:
        RangeRequestHandler.roads = RoadSource(roads)
    else:
        print("Roads: none (run tools/prepare_gip.py or pass --roads)", flush=True)
    handler = functools.partial(RangeRequestHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://localhost:{args.port}/tools/fix_picker.html"
    if args.video:
        url += "?video=/" + quote(Path(args.video).as_posix().lstrip("/"))
    print(f"Serving {ROOT}\nOpen {url}\n(Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
