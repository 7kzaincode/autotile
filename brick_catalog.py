"""
Maps internal piece types to LDraw + BrickLink part numbers and physical dims.

A piece is identified by (kind, brick_type):
  kind         = brick | plate | tile | slope | slope_inv
  brick_type   = "AxB" with A <= B (footprint in studs)

Heights (LDraw units, 24 LDU per brick height):
  brick     : 24  (full)
  plate     : 8   (1/3 of a brick)
  tile      : 8   (plate-height, no studs, smooth top)
  slope     : 24  (full brick height, sloped top, high end on one short edge)
  slope_inv : 24  (full brick height inverted slope)

LDraw / BrickLink share part numbers for these basic pieces.
"""

from __future__ import annotations

STUD_LDU  = 20
BRICK_LDU = 24
PLATE_LDU = 8       # 1/3 of a brick


# --- standard bricks (1 brick tall, studs on top) ---
BRICKS = {
    "1x1":  "3005",  "1x2":  "3004",  "1x3":  "3622",  "1x4":  "3010",
    "1x6":  "3009",  "1x8":  "3008",  "1x10": "6111",  "1x12": "6112",  "1x16": "2465",
    "2x2":  "3003",  "2x3":  "3002",  "2x4":  "3001",
    "2x6":  "2456",  "2x8":  "3007",  "2x10": "3006",
    "4x4":  "3031",  "4x6":  "2356",  "4x8":  "4202",  "4x10": "6212",
}

# --- plates (1/3 brick tall, studs on top) ---
PLATES = {
    "1x1":  "3024",  "1x2":  "3023",  "1x3":  "3623",  "1x4":  "3710",
    "1x6":  "3666",  "1x8":  "3460",  "1x10": "4477",  "1x12": "60479",
    "2x2":  "3022",  "2x3":  "3021",  "2x4":  "3020",
    "2x6":  "3795",  "2x8":  "3034",  "2x10": "3832",
    "4x4":  "3031",  "4x6":  "3032",  "4x8":  "3035",  "4x10": "3030",
    "6x6":  "3958",  "6x8":  "3036",  "6x10": "3033",
    "8x8":  "41539", "8x16": "92438",
}

# --- tiles (plate-height, smooth top, no studs) ---
TILES = {
    "1x1": "3070b", "1x2": "3069b", "1x3": "63864", "1x4": "2431",
    "1x6": "6636",  "1x8": "4162",
    "2x2": "3068b", "2x4": "87079", "2x6": "69729", "2x8": "4515",
    "4x4": "1751",  "6x6": "10202",
}

# --- 45° slopes (full brick height, top sloped to short edge) ---
SLOPES_45 = {
    "1x2": "3040",  "1x3": "4286",  "1x4": "60481",
    "2x2": "3039",  "2x3": "3038",  "2x4": "3037",  "2x6": "2875",
}

# --- 45° inverted slopes (overhang pieces) ---
SLOPES_45_INV = {
    "1x2": "3665",
    "2x2": "3660",  "2x3": "3747",  "2x4": "4861",
}

# --- specialty pet/organic pieces ---
# Round bricks (cylinders), full brick height
ROUND_BRICKS = {
    "1x1": "3062b",   # 1x1 Round Brick
    "2x2": "3941",    # 2x2 Round Brick
}
# Round plates (cylinders), 1/3 brick height
ROUND_PLATES = {
    "1x1": "4073",    # 1x1 Round Plate (eyeball)
    "2x2": "4032",    # 2x2 Round Plate
}
# Round tiles (cylinders), 1/3 brick height, smooth top
ROUND_TILES = {
    "1x1": "98138",   # 1x1 Round Tile (eye pupil)
    "2x2": "14769",   # 2x2 Round Tile
}
# Cones — 1x1 cone is 2/3 brick height tapered to a point
CONES = {
    "1x1": "4589",    # 1x1x2/3 cone (tail tip, claws)
}
# Cheese slopes — 1x1x2/3, 18° angled small slope (ear tips, paw pads)
CHEESE_SLOPES = {
    "1x1": "54200",   # 1x1x2/3 Cheese Slope
}
# Domes — hemisphere top on a 2x2 base (rounded heads / backs)
DOMES = {
    "2x2": "553",     # 2x2 Brick with Round Dome Top
}


# Public catalog: (kind, brick_type) -> {ldraw, bricklink, studs_x, studs_y, height, name}
def _parse_dims(bt: str) -> tuple[int, int]:
    a, b = bt.split("x")
    return int(a), int(b)


def _build_catalog():
    catalog = {}
    # Each row: (kind, parts_dict, height_ldu, kind_label)
    layout = [
        ("brick",        BRICKS,        BRICK_LDU,         "Brick"),
        ("plate",        PLATES,        PLATE_LDU,         "Plate"),
        ("tile",         TILES,         PLATE_LDU,         "Tile"),
        ("slope",        SLOPES_45,     BRICK_LDU,         "Slope 45°"),
        ("slope_inv",    SLOPES_45_INV, BRICK_LDU,         "Slope 45° Inv"),
        # Specialty pet/organic pieces
        ("round_brick",  ROUND_BRICKS,  BRICK_LDU,         "Round Brick"),
        ("round_plate",  ROUND_PLATES,  PLATE_LDU,         "Round Plate"),
        ("round_tile",   ROUND_TILES,   PLATE_LDU,         "Round Tile"),
        ("cone",         CONES,         2 * PLATE_LDU,     "Cone"),       # 16 LDU
        ("cheese_slope", CHEESE_SLOPES, 2 * PLATE_LDU,     "Cheese Slope"),
        ("dome",         DOMES,         BRICK_LDU,         "Dome"),
    ]
    for kind, parts, height_ldu, kind_label in layout:
        for bt, part_id in parts.items():
            sx, sy = _parse_dims(bt)
            catalog[(kind, bt)] = {
                "ldraw":     part_id,
                "bricklink": part_id,
                "studs_x":   sx,
                "studs_y":   sy,
                "height":    height_ldu,
                "name":      f"{kind_label} {sx} x {sy}",
                "kind":      kind,
                "brick_type": bt,
            }
    return catalog


CATALOG = _build_catalog()


def lookup(kind: str, brick_type: str) -> dict | None:
    return CATALOG.get((kind, brick_type))


# Backwards-compat alias — legacy code expects BRICK_CATALOG keyed only by brick_type.
BRICK_CATALOG = {
    bt: {
        "ldraw":     spec["ldraw"],
        "bricklink": spec["bricklink"],
        "studs_x":   spec["studs_x"],
        "studs_y":   spec["studs_y"],
        "name":      spec["name"],
    }
    for (k, bt), spec in CATALOG.items() if k == "brick"
}


def all_footprints(kind: str = "brick") -> list[tuple[int, int]]:
    """Return all (size_x, size_y) footprints in the catalog for the given kind,
    including 90°-rotated counterparts. Deduplicated."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for (k, _), spec in CATALOG.items():
        if k != kind:
            continue
        a, b = spec["studs_x"], spec["studs_y"]
        for fp in ((a, b), (b, a)):
            if fp not in seen:
                seen.add(fp)
                out.append(fp)
    return out
