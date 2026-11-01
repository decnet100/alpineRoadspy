"""Build catalog/signs.csv: traffic signs of Austria and Switzerland with codes.

Source: the German Wikipedia picture tables, which list every sign of the
Austrian StVO (§§ 50, 52, 53, 54) and of the Swiss SSV, appendix 2, with its
number and a public-domain SVG on Wikimedia Commons:
  - "Bildtafel der Verkehrszeichen in Österreich"
  - "Bildtafel der Strassensignale in der Schweiz und in Liechtenstein seit 2023"

Codes follow OpenStreetMap's traffic_sign scheme: AT:<paragraph>.<number>
(e.g. AT:52.10a, AT:53.17a; Austria has no official sign numbers, the StVO
numbering is used) and CH:<SSV number> (e.g. CH:2.30, CH:4.27).

    python tools/build_sign_catalog.py

The CSV is meant to be checked into the repo and extended by hand.
"""

import csv
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "catalog" / "signs.csv"
PAGES = {
    "AT": "Bildtafel der Verkehrszeichen in Österreich",
    "CH": "Bildtafel der Strassensignale in der Schweiz und in Liechtenstein seit 2023",
}
# Austrian sections -> (paragraph, group)
AT_SECTIONS = [
    ("§ 50.", "50", "warning"),
    ("lit. a)", "52", "prohibitory"),
    ("lit. b)", "52", "mandatory"),
    ("lit. c)", "52", "priority"),
    ("§ 53.", "53", "information"),
    ("§ 54.", "54", "additional"),
]
# signs missing in the Wikipedia tables but common on our roads (source "manual")
EXTRA = [
    {"code": "AT:53.13a", "country": "AT", "group": "information", "kind": "direction", "name": "Vorwegweiser",
     "legal_ref": "StVO § 53 Z 13a", "file": "Hinweiszeichen 13a groß (2010).svg"},
    {"code": "AT:leittafel", "country": "AT", "group": "guidance", "kind": "pictogram",
     "name": "Leittafel (Kurve, gelb-schwarz)", "legal_ref": "RVS 05.02.13 / 05.02.12 (keine StVO-Ziffer)",
     # no Austrian drawing on Commons: German sign 625 as shape reference (in Austria yellow/black)
     "file": "Zeichen 625-10 - Richtungstafel in Kurven (Aufstellung rechts), 500x500; StVO 1992.svg"},
]
CH_GROUPS = {"1": "warning", "2": "regulatory", "3": "priority", "4": "information", "5": "additional"}

# signs whose content the review page asks for in a structured way
CITY_LIMIT = {"AT:53.17a", "AT:53.17b", "CH:4.27", "CH:4.28", "CH:4.29", "CH:4.30"}
DIRECTION = re.compile(r"wegweiser|vorwegweiser|einspurtafel|richtungstafel", re.I)
VALUE = re.compile(r"…|\.\.\.|geschwindigkeit|gewicht|höhe|breite|länge|achslast|entfernung|tonnen", re.I)


def wikitext(title):
    url = "https://de.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "parse", "page": title, "prop": "wikitext", "format": "json", "redirects": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "alpineRoadspy (https://github.com/decnet100/alpineRoadspy)"})
    return json.load(urllib.request.urlopen(req, timeout=60))["parse"]["wikitext"]["*"]


def clean(text):
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)   # [[link|label]] -> label
    text = re.sub(r"''+", "", text).replace("&shy;", "").replace("&nbsp;", " ").replace("&#8239;", " ")
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def gallery_entries(text):
    """(file, caption) for every line of every <gallery> block."""
    for block in re.findall(r"<gallery[^>]*>(.*?)</gallery>", text, re.S):
        for line in block.splitlines():
            if "|" in line:
                file, caption = line.strip().split("|", 1)
                yield re.sub(r"^(Datei|File|Bild|Image):", "", file.strip()), clean(caption)


def kind_of(code, name):
    if code in CITY_LIMIT:
        return "city_limit"
    if DIRECTION.search(name):
        return "direction"
    if VALUE.search(name):
        return "value"
    return "pictogram"


def austria(text):
    rows = []
    # split into sections by heading; stop before railway crossings / historic signs
    parts = re.split(r"^(==+[^=\n]+==+)\s*$", text, flags=re.M)
    para = group = None
    for chunk in parts:
        if chunk.startswith("=="):
            heading = chunk.strip("= ")
            if "Eisenbahnkreuzung" in heading or "Historisch" in heading:
                para = None
                continue
            for key, p, g in AT_SECTIONS:
                if key in heading:
                    para, group = p, g
            continue
        if para is None:
            continue
        for file, caption in gallery_entries(chunk):
            m = re.match(r"([0-9]+[a-z]?(?:-[a-z])?|[a-z])\s*:\s*(.+)", caption)
            if not m:
                continue
            number, name = m.groups()
            code = f"AT:{para}.{number.split('-')[0]}"
            rows.append({"code": code, "country": "AT", "group": group, "name": name,
                         "legal_ref": f"StVO § {para}" + (f" Z {number}" if para != "54" else f" lit. {number}"),
                         "kind": kind_of(code, name), "file": file})
    return rows


def switzerland(text):
    rows = []
    section = text.split("== Signalisationskatalog", 1)[-1]
    skip = False
    for heading_block in re.split(r"^(===+[^=\n]+===+)\s*$", section, flags=re.M):
        if heading_block.startswith("==="):
            # markings, Liechtenstein-only and language variants are left out
            skip = any(k in heading_block for k in ("Markierungen", "Liechtenstein", "Landessprachen"))
            continue
        if skip:
            continue
        for file, caption in gallery_entries(heading_block):
            if file.startswith("LI-"):
                continue
            m = re.match(r"([1-5]\.[0-9]+(?:\.[0-9]+)?[a-z]?)\s+(.+)", caption)
            if not m:
                continue
            number, name = m.groups()
            name = re.sub(r",\s*Schweiz$", "", name).replace("- ", "")
            code = f"CH:{number}"
            rows.append({"code": code, "country": "CH", "group": CH_GROUPS[number[0]], "name": name,
                         "legal_ref": f"SSV Anhang 2, {number}", "kind": kind_of(code, name), "file": file})
    return rows


def main():
    rows = [dict(r, source="wikipedia") for r in austria(wikitext(PAGES["AT"])) + switzerland(wikitext(PAGES["CH"]))]
    rows += [dict(r, source="manual") for r in EXTRA]
    seen, unique = set(), []
    for r in rows:
        if r["file"] not in seen:
            seen.add(r["file"])
            r["image_url"] = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(r["file"].replace(" ", "_"))
            unique.append(r)
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "country", "group", "kind", "name", "legal_ref", "file", "image_url", "source"])
        w.writeheader()
        w.writerows(unique)
    by = {}
    for r in unique:
        by[(r["country"], r["kind"])] = by.get((r["country"], r["kind"]), 0) + 1
    print(f"Wrote {len(unique)} signs to {OUT}: " + ", ".join(f"{c} {k}: {n}" for (c, k), n in sorted(by.items())))


if __name__ == "__main__":
    main()
