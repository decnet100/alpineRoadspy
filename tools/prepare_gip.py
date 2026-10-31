"""Extract the routable GIP link network into a compact GeoPackage for tools/serve.py.

Download ``B_gip_network_ogd.zip`` (about 4 GB) from https://open.gip.gv.at/ogd/
(open data "GIP Graphenintegrationsplattform Österreich", CC BY, see
https://www.gip.gv.at), unzip it, then e.g. for Tyrol:

    python tools/prepare_gip.py path/to/gip_network_ogd.gpkg --bbox 10.0,46.6,13.1,47.8

This writes ``data/roads/gip_linknetz.gpkg``: layer GIP_LINKNETZ_OGD, active
links only (BAUSTATUS 5), the columns the tools use, in WGS84.
"""

import argparse
from pathlib import Path

import pyogrio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "roads" / "gip_linknetz.gpkg"
COLUMNS = ["LINK_ID", "FROM_NODE", "TO_NODE", "NAME1", "NAME2", "EDGECAT", "FRC",
           "FORMOFWAY", "ONEWAY_CAR", "URBAN", "VMAX_CAR_T", "VMAX_CAR_B", "BAUSTATUS"]


def find_link_layer(path):
    """The routable link layer: LINK in the current export, GIP_LINKNETZ_OGD in older ones."""
    layers = [name for name, _ in pyogrio.list_layers(path)]
    matches = [n for n in layers if n.lower() == "link"] + [n for n in layers if "linknetz" in n.lower()]
    if not matches:
        raise SystemExit(f"No LINK/LINKNETZ layer in {path}; layers: {', '.join(layers)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gpkg", help="unzipped GIP network GeoPackage")
    parser.add_argument("--bbox", help="lon_min,lat_min,lon_max,lat_max (WGS84) to clip to")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    layer = find_link_layer(args.gpkg)
    info = pyogrio.read_info(args.gpkg, layer=layer)
    fields = {f.upper(): f for f in info["fields"]}
    columns = [fields[c] for c in COLUMNS if c in fields]
    missing = [c for c in COLUMNS if c not in fields]
    if {"FROM_NODE", "TO_NODE"} & set(missing):
        raise SystemExit(f"Layer {layer} has no FROM_NODE/TO_NODE; fields: {', '.join(info['fields'])}")
    if missing:
        print(f"Note: columns not in source: {', '.join(missing)}")

    bbox = None
    if args.bbox:
        lon0, lat0, lon1, lat1 = map(float, args.bbox.split(","))
        bbox = Transformer.from_crs("EPSG:4326", info["crs"], always_xy=True).transform_bounds(lon0, lat0, lon1, lat1)

    where = f"{fields['BAUSTATUS']} = 5" if "BAUSTATUS" in fields else None
    print(f"Reading {layer} from {args.gpkg} ({info['features']} links in total, CRS {info['crs']}) ...")
    gdf = pyogrio.read_dataframe(args.gpkg, layer=layer, columns=columns, bbox=bbox, where=where)
    gdf = gdf.rename(columns={v: k for k, v in fields.items()}).to_crs("EPSG:4326")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pyogrio.write_dataframe(gdf, args.out, layer="GIP_LINKNETZ_OGD")
    print(f"Wrote {len(gdf)} links to {args.out}")


if __name__ == "__main__":
    main()
