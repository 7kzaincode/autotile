"""
Post-processing passes that improve the look of a decomposed brick model:

  - tile_top_surfaces : ADD tile pieces on top of top-exposed bricks (smooth caps)
  - slope_edges       : REPLACE stair-step top-edge bricks with 45° slopes
  - slope_overhangs   : REPLACE bottom overhang bricks with inverted slopes
  - add_baseplate     : ADD a plate baseplate beneath the model

Tiles are 1/3 brick height and sit ON the brick below (LEGO tiles snap onto
studs). Slopes/inverted-slopes are full brick height but with an angled face,
so they REPLACE a brick of the same footprint at the same layer.

Each pass returns a NEW list (input is not mutated).
"""

from __future__ import annotations

from brick_catalog import CATALOG
from voxels_to_bricks import Brick


def _occ_map(bricks: list[Brick]) -> dict[tuple[int, int, int], int]:
    """voxel (x, y, z) -> brick index. Only counts solid-volume kinds (brick,
    plate, slope, slope_inv). Tiles don't count — they're decorative caps."""
    occ: dict[tuple[int, int, int], int] = {}
    for i, b in enumerate(bricks):
        if b.kind == "tile":
            continue
        for dx in range(b.size_x):
            for dy in range(b.size_y):
                occ[(b.x + dx, b.y + dy, b.z)] = i
    return occ


def _normalized_bt(b: Brick) -> str:
    a, c = sorted((b.size_x, b.size_y))
    return f"{a}x{c}"


def _has_kind(kind: str, brick_type: str) -> bool:
    return (kind, brick_type) in CATALOG


# --- tile pass ---

def tile_top_surfaces(bricks: list[Brick]) -> list[Brick]:
    """For each top-exposed brick with a matching tile shape, ADD a tile piece
    directly above. Even if the tile shape doesn't match the brick exactly,
    decompose the brick's footprint into smaller tiles to cover everything.
    """
    occ = _occ_map(bricks)
    extras: list[Brick] = []
    for b in bricks:
        if b.kind != "brick":
            continue
        top_exposed = all(
            (b.x + dx, b.y + dy, b.z + 1) not in occ
            for dx in range(b.size_x)
            for dy in range(b.size_y)
        )
        if not top_exposed:
            continue
        bt = _normalized_bt(b)
        if _has_kind("tile", bt):
            extras.append(Brick(
                x=b.x, y=b.y, z=b.z + 1,
                size_x=b.size_x, size_y=b.size_y,
                color_id=b.color_id, kind="tile",
                rotation=b.rotation,
            ))
            continue
        # No exact tile match — decompose the brick's footprint into smaller
        # tiles. This guarantees every top-exposed stud gets covered.
        for tile_b in _cover_with_tiles(b.x, b.y, b.z + 1,
                                        b.size_x, b.size_y, b.color_id):
            extras.append(tile_b)
    return list(bricks) + extras


def _cover_with_tiles(x: int, y: int, z: int, sx: int, sy: int, color: int):
    """Greedily cover an sx*sy footprint with tiles. Uses 2x2 → 1x4 → 1x2 → 1x1."""
    consumed = [[False] * sy for _ in range(sx)]
    candidates = [(2, 2), (1, 4), (4, 1), (1, 2), (2, 1), (1, 1)]
    out: list[Brick] = []
    for ix in range(sx):
        for iy in range(sy):
            if consumed[ix][iy]:
                continue
            for (cx, cy) in candidates:
                if ix + cx > sx or iy + cy > sy:
                    continue
                if not _has_kind("tile", f"{min(cx,cy)}x{max(cx,cy)}"):
                    continue
                if any(consumed[ix + dx][iy + dy]
                       for dx in range(cx) for dy in range(cy)):
                    continue
                for dx in range(cx):
                    for dy in range(cy):
                        consumed[ix + dx][iy + dy] = True
                out.append(Brick(
                    x=x + ix, y=y + iy, z=z,
                    size_x=cx, size_y=cy,
                    color_id=color, kind="tile", rotation=0,
                ))
                break
    return out


def cheese_slope_ear_tips(bricks: list[Brick], top_zone_frac: float = 0.72) -> list[Brick]:
    """Find the very top brick of THIN tapered columns (e.g. ear tips) and
    replace it with a cheese slope (1x1x2/3 18° angled piece). This makes
    ears actually look pointy instead of square.

    The pass is intentionally limited to the top zone. Organic meshes contain
    lots of isolated underside/chest voxels, and turning those into triangular
    wedges reads as random anatomy rather than intentional LEGO design.
    """
    if not bricks:
        return bricks
    occ = _occ_map(bricks)
    max_z = max(b.z for b in bricks)
    min_tip_z = max(0, int(round(max_z * float(top_zone_frac))))
    out: list[Brick] = []
    for b in bricks:
        if b.kind != "brick" or b.size_x != 1 or b.size_y != 1:
            out.append(b); continue
        if b.z < min_tip_z:
            out.append(b); continue
        # Top must be exposed
        if (b.x, b.y, b.z + 1) in occ:
            out.append(b); continue
        # All four horizontal neighbors must be empty (this voxel is a "tip")
        neighbors_empty = all(
            (b.x + dx, b.y + dy, b.z) not in occ
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]
        )
        if not neighbors_empty:
            out.append(b); continue
        # Below must be supported (otherwise we'd just create a tip from nothing)
        if (b.x, b.y, b.z - 1) not in occ:
            out.append(b); continue
        out.append(Brick(
            x=b.x, y=b.y, z=b.z,
            size_x=1, size_y=1, color_id=b.color_id,
            kind="cheese_slope", rotation=0,
        ))
    return out


def dome_round_tops(bricks: list[Brick]) -> list[Brick]:
    """Detect 2x2 brick clusters at the top of a column that form a roughly
    circular cross-section. Replace the topmost 2x2 with a dome piece."""
    occ = _occ_map(bricks)
    # Find 2x2 bricks that have nothing above
    out: list[Brick] = []
    for b in bricks:
        if b.kind != "brick" or b.size_x != 2 or b.size_y != 2:
            out.append(b); continue
        top_exposed = all(
            (b.x + dx, b.y + dy, b.z + 1) not in occ
            for dx in (0, 1) for dy in (0, 1)
        )
        if not top_exposed:
            out.append(b); continue
        # Also check that the 2x2 is "isolated" at the top — no other 2x2
        # bricks adjacent at the same layer (otherwise we'd be flat top, not round)
        adj_brick_count = sum(
            1 for (dx, dy) in [(2, 0), (-1, 0), (0, 2), (0, -1)]
            if (b.x + dx, b.y + dy, b.z) in occ
        )
        if adj_brick_count >= 2:
            out.append(b); continue
        out.append(Brick(
            x=b.x, y=b.y, z=b.z,
            size_x=2, size_y=2, color_id=b.color_id,
            kind="dome", rotation=0,
        ))
    return out


# --- slope pass (top edges) ---

def slope_edges(bricks: list[Brick]) -> list[Brick]:
    """Find top-exposed bricks at column edges. Replace them with a 45° slope
    brick of matching size, oriented so the LOW end faces the open side.
    """
    occ = _occ_map(bricks)
    out: list[Brick] = []
    for b in bricks:
        if b.kind != "brick":
            out.append(b); continue
        bt = _normalized_bt(b)
        if not _has_kind("slope", bt):
            out.append(b); continue
        top_exposed = all(
            (b.x + dx, b.y + dy, b.z + 1) not in occ
            for dx in range(b.size_x)
            for dy in range(b.size_y)
        )
        if not top_exposed:
            out.append(b); continue
        direction = _slope_direction(b, occ)
        if direction is None:
            out.append(b); continue
        out.append(Brick(
            x=b.x, y=b.y, z=b.z,
            size_x=b.size_x, size_y=b.size_y,
            color_id=b.color_id, kind="slope",
            rotation=b.rotation, slope_dir=direction,
        ))
    return out


# --- inverted slope pass (overhangs) ---

def slope_overhangs(bricks: list[Brick]) -> list[Brick]:
    """A brick whose layer below has NOTHING under it on one side AND a neighbor
    on the same layer on the opposite side is an overhang candidate. Replace
    it with an inverted 45° slope brick."""
    occ = _occ_map(bricks)
    out: list[Brick] = []
    for b in bricks:
        if b.kind != "brick":
            out.append(b); continue
        bt = _normalized_bt(b)
        if not _has_kind("slope_inv", bt):
            out.append(b); continue
        if b.z == 0:
            out.append(b); continue
        bottom_open = all(
            (b.x + dx, b.y + dy, b.z - 1) not in occ
            for dx in range(b.size_x)
            for dy in range(b.size_y)
        )
        if not bottom_open:
            out.append(b); continue
        direction = _slope_inv_direction(b, occ)
        if direction is None:
            out.append(b); continue
        out.append(Brick(
            x=b.x, y=b.y, z=b.z,
            size_x=b.size_x, size_y=b.size_y,
            color_id=b.color_id, kind="slope_inv",
            rotation=b.rotation, slope_dir=direction,
        ))
    return out


# --- slope direction heuristics ---

def _slope_direction(b: Brick, occ: dict) -> str | None:
    perim = _exposed_perimeter(b, occ, b.z)
    below = _exposed_perimeter(b, occ, b.z - 1) if b.z > 0 else None
    candidates = []
    for d in ("+x", "-x", "+y", "-y"):
        side_clear_here    = perim[d] == _side_len(b, d)
        below_continues    = below is not None and below[d] < _side_len(b, d)
        if side_clear_here and below_continues:
            candidates.append(d)
    return candidates[0] if len(candidates) == 1 else None


def _slope_inv_direction(b: Brick, occ: dict) -> str | None:
    if b.z == 0:
        return None
    below = _exposed_perimeter(b, occ, b.z - 1)
    same  = _exposed_perimeter(b, occ, b.z)
    candidates = []
    for d in ("+x", "-x", "+y", "-y"):
        below_open    = below[d] == _side_len(b, d)
        has_neighbor  = same[d] == 0
        if below_open and has_neighbor:
            candidates.append(d)
    return candidates[0] if len(candidates) == 1 else None


def _exposed_perimeter(b: Brick, occ: dict, z: int) -> dict[str, int]:
    out = {"+x": 0, "-x": 0, "+y": 0, "-y": 0}
    for dx in range(b.size_x):
        for dy in range(b.size_y):
            x, y = b.x + dx, b.y + dy
            if dx == 0            and (x - 1, y, z) not in occ: out["-x"] += 1
            if dx == b.size_x - 1 and (x + 1, y, z) not in occ: out["+x"] += 1
            if dy == 0            and (x, y - 1, z) not in occ: out["-y"] += 1
            if dy == b.size_y - 1 and (x, y + 1, z) not in occ: out["+y"] += 1
    return out


def _side_len(b: Brick, d: str) -> int:
    return b.size_y if d in ("+x", "-x") else b.size_x


# --- baseplate pass ---

def add_baseplate(bricks: list[Brick], grid_shape: tuple[int, int, int],
                  color_id: int = 2, margin: int = 1) -> list[Brick]:
    """Prepend a single plate spanning the model's XY bbox + margin at z=-1.

    z=-1 is a sentinel "below the model" — LDraw/OBJ export must shift the
    whole model UP by one plate height so the baseplate sits at LDraw Y=0
    and the model starts at -8.
    """
    if not bricks:
        return list(bricks)
    sx, sy, _ = grid_shape
    min_x = max(0, min(b.x for b in bricks) - margin)
    min_y = max(0, min(b.y for b in bricks) - margin)
    max_x = min(sx, max(b.x + b.size_x for b in bricks) + margin)
    max_y = min(sy, max(b.y + b.size_y for b in bricks) + margin)
    w = max_x - min_x
    h = max_y - min_y
    plate = Brick(
        x=min_x, y=min_y, z=-1, size_x=w, size_y=h,
        color_id=color_id, kind="plate",
    )
    return [plate] + list(bricks)


def darken_silhouette_edges(bricks: list[Brick], palette: list[dict],
                            shift_amount: float = 0.35,
                            restrict_to_ids: list[int] | None = None) -> list[Brick]:
    """Find bricks on the OUTLINE of each layer (have an open horizontal
    neighbor) and darken their color. Mimics the cartoon-outline effect that
    designed LEGO sculptures use to define silhouettes.

    `shift_amount` = fraction toward black (0 none, 1 black).
    `restrict_to_ids` (optional) = only consider these palette IDs as darken
    targets. Keeps the output strictly within a chosen palette subset.
    """
    if not bricks or not palette:
        return bricks
    import numpy as np
    occ = _occ_map(bricks)
    # Filter the palette to the allowed set, if any
    if restrict_to_ids:
        wanted = {int(i) for i in restrict_to_ids}
        active = [p for p in palette if int(p["id"]) in wanted]
        if not active:
            active = palette
    else:
        active = palette
    pal_by_id = {p["id"]: p for p in palette}
    pal_rgb = np.array([p["rgb"] for p in active], dtype=np.float32)
    pal_ids = np.array([p["id"] for p in active], dtype=np.int32)

    out: list[Brick] = []
    for b in bricks:
        if b.kind not in ("brick", "tile", "plate", "slope"):
            out.append(b); continue
        # On the outline iff any horizontal-neighbor footprint cell is empty
        outline = False
        for dx in range(b.size_x):
            for dy in range(b.size_y):
                x, y = b.x + dx, b.y + dy
                if (dx == 0 and (x - 1, y, b.z) not in occ) or \
                   (dx == b.size_x - 1 and (x + 1, y, b.z) not in occ) or \
                   (dy == 0 and (x, y - 1, b.z) not in occ) or \
                   (dy == b.size_y - 1 and (x, y + 1, b.z) not in occ):
                    outline = True; break
            if outline: break
        if not outline:
            out.append(b); continue
        # Darken: lerp toward black, then snap to nearest palette color
        cur = pal_by_id.get(b.color_id)
        if cur is None:
            out.append(b); continue
        cur_rgb = np.array(cur["rgb"], dtype=np.float32)
        dark = cur_rgb * (1.0 - shift_amount)
        d = ((pal_rgb - dark) ** 2).sum(axis=1)
        new_id = int(pal_ids[d.argmin()])
        if new_id == b.color_id:
            out.append(b); continue
        out.append(Brick(
            x=b.x, y=b.y, z=b.z,
            size_x=b.size_x, size_y=b.size_y,
            color_id=new_id, kind=b.kind,
            rotation=b.rotation, slope_dir=b.slope_dir,
        ))
    return out


def drop_floaters(bricks: list[Brick]) -> list[Brick]:
    """Connected-component-aware floater removal.

    Builds an undirected graph of bricks where edges connect bricks that share
    a stud (vertical adjacency) OR a horizontal edge (side-by-side). A brick is
    "grounded" iff it's in a connected component that touches z=0. Drop the
    rest.

    This is correct for organic shapes like sitting animals: the body might
    not have a brick DIRECTLY below it in its own column, but it IS connected
    to the ground via the head/neck/legs path.
    """
    if not bricks:
        return bricks
    # Build occupancy with per-voxel brick index
    occ = _occ_map(bricks)
    n = len(bricks)
    parents = list(range(n))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[ra] = rb

    # Two bricks are connected if any pair of voxels in them are 6-neighbors
    # (share a face), OR overlap vertically (stud connection).
    for i, b in enumerate(bricks):
        for dx in range(b.size_x):
            for dy in range(b.size_y):
                x, y, z = b.x + dx, b.y + dy, b.z
                for (nx, ny, nz) in [(x+1, y, z), (x-1, y, z),
                                      (x, y+1, z), (x, y-1, z),
                                      (x, y, z+1), (x, y, z-1)]:
                    j = occ.get((nx, ny, nz))
                    if j is not None and j != i:
                        union(i, j)

    # Find which components contain a z=0 brick
    grounded_roots = set()
    for i, b in enumerate(bricks):
        if b.z <= 0:
            grounded_roots.add(find(i))
    if not grounded_roots:
        # Nothing on the floor — everything stays (don't nuke the model)
        return list(bricks)

    keep = [b for i, b in enumerate(bricks) if find(i) in grounded_roots]
    if len(keep) < len(bricks):
        print(f"[postprocess] dropped {len(bricks) - len(keep)} disconnected "
              f"brick(s) of {len(bricks)} (kept {len(keep)})")
    return keep


def support_floaters(bricks: list[Brick], support_color_id: int = 2) -> list[Brick]:
    """Instead of dropping floating bricks, ADD support bricks beneath them.
    Walks downward from each unsupported brick until it hits a supported
    structure or the ground, filling the column with 1x1 bricks in the
    structural color (default: Light Bluish Gray, palette id 2).
    """
    if not bricks:
        return bricks
    bricks = list(bricks)
    while True:
        occ = _occ_map(bricks)
        added: list[Brick] = []
        for b in bricks:
            if b.z <= 0 or b.kind != "brick":
                continue
            # Each footprint cell needs at least one supported voxel below it
            # somewhere in the column down to z=0.
            for dx in range(b.size_x):
                for dy in range(b.size_y):
                    x, y = b.x + dx, b.y + dy
                    if (x, y, b.z - 1) in occ:
                        continue  # this cell is already supported
                    # Walk down looking for support
                    z = b.z - 1
                    while z >= 0 and (x, y, z) not in occ:
                        added.append(Brick(
                            x=x, y=y, z=z, size_x=1, size_y=1,
                            color_id=support_color_id, kind="brick",
                        ))
                        # Insert into our running occ so subsequent cells in
                        # this footprint don't double-fill
                        occ[(x, y, z)] = len(bricks) + len(added) - 1
                        z -= 1
        if not added:
            return bricks
        bricks.extend(added)


def apply_all(bricks: list[Brick], grid_shape: tuple[int, int, int],
              do_tiles: bool = True, do_slopes: bool = True,
              do_slope_inv: bool = True, do_baseplate: bool = False,
              do_cheese_tips: bool = True, do_dome_tops: bool = False,
              drop_unsupported: bool = False,
              auto_support: bool = False,
              darken_edges: bool = False,
              palette: list[dict] | None = None,
              restrict_to_ids: list[int] | None = None,
              baseplate_color: int = 2) -> list[Brick]:
    # Order matters: auto_support BEFORE other passes (adds new pieces),
    # drop_unsupported AFTER (cleans up anything still floating).
    if auto_support:
        bricks = support_floaters(bricks, support_color_id=baseplate_color)
    if darken_edges and palette is not None:
        bricks = darken_silhouette_edges(bricks, palette, restrict_to_ids=restrict_to_ids)
    # Specialty substitutions: BEFORE slope/tile so they can replace
    # specific top-edge / tip bricks with the right shaped piece.
    if do_cheese_tips:
        bricks = cheese_slope_ear_tips(bricks)
    if do_dome_tops:
        bricks = dome_round_tops(bricks)
    if do_slopes:
        bricks = slope_edges(bricks)
    if do_slope_inv:
        bricks = slope_overhangs(bricks)
    if do_tiles:
        bricks = tile_top_surfaces(bricks)
    if drop_unsupported:
        bricks = drop_floaters(bricks)
    if do_baseplate:
        bricks = add_baseplate(bricks, grid_shape, color_id=baseplate_color)
    return bricks
