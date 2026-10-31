"""Field names of the GIP link layer across export versions.

Older exports (documented up to 06/2025): layer GIP_LINKNETZ_OGD with LINK_ID,
FROM_NODE, ... Since 2026: layer LINK with UUIDs plus numeric short ids
(old-to-new mapping: https://gip.gv.at/assets/downloads/2604_dokumentation_gipat_ogd.pdf).
Both are mapped to the old names, which the tools use internally.
"""

ALIASES = {
    "LINK_ID": ["LINK_ID", "SHORT_ID"],
    "FROM_NODE": ["FROM_NODE", "NODE_FROM_SHORT_ID"],
    "TO_NODE": ["TO_NODE", "NODE_TO_SHORT_ID"],
    "NAME1": ["NAME1", "NAME_TEXT_HR"],            # name of the high-ranking road (A, S, B, L, ...)
    "NAME2": ["NAME2", "SHORT_NAME_HR"],           # old: further names; new: short name, e.g. "B182"
    "NAME_LOCAL": ["NAME_TEXT_ADDITIONAL"],        # new: municipal road / farm road / common name
    "EDGECAT": ["EDGECAT", "STREETCAT", "EDGE_CATEGORY"],
    "FRC": ["FRC", "FUNCROADCLASS", "FUNCTIONAL_CLASS"],
    "FORMOFWAY": ["FORMOFWAY", "FORM_OF_WAY"],
    "ONEWAY_CAR": ["ONEWAY_CAR", "ONEWAY"],        # -1 no car access, 0/1 one-way, 2 both ways
    "URBAN": ["URBAN"],
    "VMAX_CAR_T": ["VMAX_CAR_T", "MAXSPEED_TOW_CAR"],
    "VMAX_CAR_B": ["VMAX_CAR_B", "MAXSPEED_BKW_CAR"],
    "LEVEL": ["LEVEL", "LEVEL_INTERMEDIATE"],      # logical height (bridges, tunnels)
    "BAUSTATUS": ["BAUSTATUS", "CONSTRUCTION_STATE"],
}


def resolve(fields):
    """Map the internal names to the field names present in a layer (case-insensitive)."""
    upper = {f.upper(): f for f in fields}
    out = {}
    for name, aliases in ALIASES.items():
        actual = next((upper[a] for a in [name] + aliases if a in upper), None)
        if actual:
            out[name] = actual
    return out


def find_link_layer(layers):
    """The routable link layer: LINK in current exports, GIP_LINKNETZ_OGD in older ones."""
    matches = [n for n in layers if n.lower() == "link"] + [n for n in layers if "linknetz" in n.lower()]
    return matches[0] if matches else None


def display_name(get, new_schema):
    """Road name for display, e.g. "B182 Brennerstraße"; get(name) returns a field value or None."""
    def text(name):
        value = get(name)
        return "" if value is None or value != value else str(value).strip()  # value != value: NaN

    name1, name2 = text("NAME1"), text("NAME2")
    if not new_schema:  # old schema: NAME2 holds alternative names
        return name1 or name2
    high = name1 if name2 in name1 else f"{name2} {name1}".strip()
    return high or text("NAME_LOCAL")
