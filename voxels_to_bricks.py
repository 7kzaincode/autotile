"""
Stage 5: Decompose a voxel grid into LEGO brick placements (the core algorithm).

The main pass produces bricks. Post-processing passes can then substitute
top-exposed bricks with tiles (smooth top) and stair-step top edges with
slope pieces. Inverted slopes are added for overhangs.

Voxel grid coordinates: (X, Y, Z) where Z is the build/stack axis.
Layers are filled bottom-up (z = 0 is the lowest layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brick_catalog import all_footprints


def _sorted_footprints(prefer_x_long: bool) -> list[tuple[int, int]]:
    fps = all_footprints("brick")
    def key(fp):
        lx, ly = fp
        area = lx * ly
        if prefer_x_long:
            orientation_bonus = 1 if lx > ly else (0 if lx == ly else -1)
        else:
            orientation_bonus = 1 if ly > lx else (0 if lx == ly else -1)
        return (-area, -orientation_bonus, -max(lx, ly))
    return sorted(fps, key=key)


_FOOTPRINTS_X_LONG = _sorted_footprints(prefer_x_long=True)
_FOOTPRINTS_Y_LONG = _sorted_footprints(prefer_x_long=False)


@dataclass
class Brick:
    x: int
    y: int
    z: int
    size_x: int
    size_y: int
    color_id: int
    kind: str = "brick"          # brick | plate | tile | slope | slope_inv
    rotation: int = 0            # 0, 90, 180, 270 in the XY plane
    slope_dir: str | None = None # one of "+x", "-x", "+y", "-y" for slope/slope_inv

    def to_dict(self) -> dict:
        a, b = sorted((self.size_x, self.size_y))
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "size_x": self.size_x,
            "size_y": self.size_y,
            "brick_type": f"{a}x{b}",
            "kind": self.kind,
            "rotation": self.rotation if self.rotation else (0 if self.size_x >= self.size_y else 90),
            "color": self.color_id,
            "slope_dir": self.slope_dir,
        }


def decompose(palette_grid: np.ndarray) -> list[Brick]:
    grid = palette_grid
    sx, sy, sz = grid.shape
    consumed = np.zeros_like(grid, dtype=bool)
    bricks: list[Brick] = []

    for z in range(sz):
        below = consumed[:, :, z - 1] if z > 0 else None
        footprints = _FOOTPRINTS_X_LONG if z % 2 == 0 else _FOOTPRINTS_Y_LONG

        for x in range(sx):
            for y in range(sy):
                if consumed[x, y, z] or grid[x, y, z] == 0:
                    continue
                color = int(grid[x, y, z])
                if _try_place(grid, consumed, bricks, x, y, z, color, sx, sy,
                              footprints, below):
                    continue
                if z > 0:
                    _try_place(grid, consumed, bricks, x, y, z, color, sx, sy,
                               footprints, None)

    return bricks


def _try_place(grid, consumed, bricks, x, y, z, color, sx, sy,
               footprints, below) -> bool:
    for (lx, ly) in footprints:
        if x + lx > sx or y + ly > sy:
            continue
        region = grid[x:x + lx, y:y + ly, z]
        if not np.all(region == color):
            continue
        if consumed[x:x + lx, y:y + ly, z].any():
            continue
        if below is not None:
            if not below[x:x + lx, y:y + ly].any():
                continue
        consumed[x:x + lx, y:y + ly, z] = True
        bricks.append(Brick(x=x, y=y, z=z, size_x=lx, size_y=ly, color_id=color))
        return True
    return False
