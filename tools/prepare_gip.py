"""Extract the routable GIP link network into a compact GeoPackage for tools/serve.py.

Download ``B_gip_network_ogd.zip`` (about 4 GB) from https://open.gip.gv.at/ogd/
(open data "GIP Graphenintegrationsplattform Österreich", CC BY, see
https://www.gip.gv.at), unzip it, then e.g. for Tyrol:

    python tools/prepare_gip.py path/to/gip_network_ogd.gpkg --bbox 10.0,46.6,13.1,47.8

This writes ``data/roads/gip_linknetz.gpkg``: layer GIP_LINKNETZ_OGD with the
columns the tools use under their documented names (see gip_schema.py), in
WGS84, only links under construction state 5 ("active") if that code occurs.
"""

import argparse
from pathlib import Path

import pyogrio
from pyproj import Transformer

from gip_schema import find_link_layer, resolve

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "roads" / "gip_linknetz.gpkg"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gpkg", help="unzipped GIP network GeoPackage")
    parser.add_argument("--bbox", help="lon_min,lat_min,lon_max,lat_max (WGS84) to clip to")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    layers = [name for name, _ in pyogrio.list_layers(args.gpkg)]
    layer = find_link_layer(layers)
    if not layer:
        raise SystemExit(f"No LINK/LINKNETZ layer in {args.gpkg}; layers: {', '.join(layers)}")
    info = pyogrio.read_info(args.gpkg, layer=layer)
    fields = resolve(info["fields"])
    if not {"FROM_NODE", "TO_NODE"} <= fields.keys():
        raise SystemExit(f"Layer {layer}: no from/to node ids found; fields: {', '.join(info['fields'])}")
    print("Fields: " + ", ".join(f"{k}={v}" for k, v in fields.items()))

    bbox = None
    if args.bbox:
        lon0, lat0, lon1, lat1 = map(float, args.bbox.split(","))
        bbox = Transformer.from_crs("EPSG:4326", info["crs"], always_xy=True).transform_bounds(lon0, lat0, lon1, lat1)

    print(f"Reading {layer} from {args.gpkg} ({info['features']} links in total, CRS {info['crs']}) ...")
    gdf = pyogrio.read_dataframe(args.gpkg, layer=layer, columns=list(fields.values()), bbox=bbox)
    gdf = gdf.rename(columns={v: k for k, v in fields.items()}).to_crs("EPSG:4326")

    for col in ("BAUSTATUS", "ONEWAY_CAR"):
        if col in gdf:
            print(f"{col} ({fields[col]}): " + ", ".join(f"{v}: {n}" for v, n in gdf[col].value_counts(dropna=False).items()))
    # GIP construction state 5 = active (documented for BAUSTATUS); only applied if that code is used
    if "BAUSTATUS" in gdf and (gdf["BAUSTATUS"] == 5).any():
        gdf = gdf[gdf["BAUSTATUS"] == 5]
    elif "BAUSTATUS" in gdf:
        print("Note: no links with construction state 5, keeping all")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pyogrio.write_dataframe(gdf, args.out, layer="GIP_LINKNETZ_OGD")
    print(f"Wrote {len(gdf)} links to {args.out}")


if __name__ == "__main__":
    main()
