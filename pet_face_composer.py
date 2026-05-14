"""Designed front-facing pet face module.

The side/full-body photo gives the body pose, proportions, and markings, but it
is a bad surface for recognizable front-face details. This module composes a
small protected LEGO face sprite from the front portrait and mounts it on the
model's display/front plane, anchored to the head zone inferred from the side
photo.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np


BLACK = 4
WHITE = 1
PINK = 31
MEDIUM_NOUGAT = 30
DARK_ORANGE = 23
DARK_TAN = 29


def compose_pet_face_module(
    payload: dict,
    face_photo_path: str | Path,
    feat: dict,
    *,
    body_photo_path: str | Path | None = None,
    body_gpt_data: dict | None = None,
    body_view: str | None = None,
) -> tuple[dict, dict]:
    """Attach a protected front-facing pet face sprite to ``payload``.

    The output is intentionally a low-resolution LEGO design, not a texture:
    two compact eyes with sampled iris color, a small nose, and no default mouth
    line for cats/rabbits. The sprite is anchored to the side-photo head region
    but faces the product's display/front axis, so a side-body mesh can still
    have a readable front-facing face.
    """
    bricks = payload.get("bricks") or []
    eyes = list(feat.get("eyes") or [])
    if not bricks:
        return payload, {"added": 0, "reason": "missing bricks"}
    if len(eyes) < 2:
        return payload, {"added": 0, "reason": "need two front-photo eyes"}

    try:
        from face_map import (
            _build_occ,
            _front_surface_map,
            _projection_axes,
            _side_head_target_from_photo,
        )
    except Exception as e:
        return payload, {"added": 0, "reason": f"face-map helpers unavailable: {e}"}

    grid_shape = tuple(int(v) for v in payload.get("grid_shape", (0, 0, 0)))
    if len(grid_shape) != 3 or min(grid_shape) <= 0:
        return payload, {"added": 0, "reason": "bad grid shape"}

    meta = payload.get("voxel_metadata") or {}
    front_axis = str(meta.get("front_axis") or "-y")
    axes = _projection_axes(front_axis, grid_shape)
    occ = _build_occ(bricks)
    if not occ:
        return payload, {"added": 0, "reason": "empty occupancy"}

    side_anchor = _side_head_target_from_photo(
        body_photo_path,
        body_gpt_data,
        grid_shape,
        axes,
        body_view=body_view,
    )
    head_region = _head_region(side_anchor, occ, grid_shape, axes)
    if head_region is None:
        return payload, {"added": 0, "reason": "could not locate head region"}

    surface = _front_surface_map(occ, grid_shape, axes)
    panel = _panel_from_head_region(head_region, grid_shape, axes)
    depth = _front_panel_depth(surface, panel, axes)
    if depth is None:
        return payload, {"added": 0, "reason": "no front surface near head"}

    species = _species_name(body_gpt_data)
    face_photo_path = Path(face_photo_path)
    iris_color = _sample_iris_color(face_photo_path, eyes, species=species)
    nose_color = _sample_nose_color(face_photo_path, feat, species=species)
    base_color = _most_common_head_color(bricks, head_region, axes, fallback=None)

    additions = _build_face_sprite(
        panel,
        depth,
        axes,
        grid_shape,
        iris_color=iris_color,
        nose_color=nose_color,
        base_color=base_color,
        species=species,
    )
    if not additions:
        return payload, {"added": 0, "reason": "empty face sprite"}

    out = dict(payload)
    out["bricks"] = list(bricks) + additions
    return out, {
        "added": len(additions),
        "source": "front-face-module",
        "species": species,
        "front_axis": axes["front_axis"],
        "face_plane": {
            "axis": axes["front_axis"],
            "depth": int(depth),
            "region": [int(v) for v in panel],
        },
        "head_region": [int(v) for v in head_region],
        "side_anchor": bool(side_anchor),
        "eye_centers": [
            list(_coord_from_panel(panel, 0.29, 0.64)),
            list(_coord_from_panel(panel, 0.71, 0.64)),
        ],
        "eye_style": "two-cell-iris-pupil",
        "iris_color": int(iris_color),
        "nose_color": int(nose_color),
    }


def _head_region(
    side_anchor: dict | None,
    occ: set[tuple[int, int, int]],
    grid_shape: tuple[int, int, int],
    axes: dict,
) -> tuple[int, int, int, int] | None:
    """Return a tight (u0, z0, u1, z1) head region on the display/front plane."""
    _sx, _sy, sz = grid_shape
    u_size = int(axes["u_size"])
    if side_anchor:
        u0, z0, u1, z1 = [int(v) for v in side_anchor.get("region", [0, 0, u_size, sz])]
        # Give the face module forehead/ear room but avoid letting neck/chest pull
        # the target down into the torso.
        pad_u = max(1, int(round((u1 - u0) * 0.10)))
        pad_top = max(1, int(round(sz * 0.04)))
        pad_bottom = max(1, int(round(sz * 0.08)))
        return (
            int(np.clip(u0 - pad_u, 0, max(0, u_size - 1))),
            int(np.clip(z0 - pad_bottom, 0, max(0, sz - 1))),
            int(np.clip(u1 + pad_u, 1, u_size)),
            int(np.clip(z1 + pad_top, 1, sz)),
        )

    # Fallback from geometry: use upper-front mass and choose the horizontal end
    # with more high voxels as the head.
    coords = np.array(list(occ), dtype=np.int32)
    if coords.size == 0:
        return None
    u_vals = coords[:, int(axes["u_axis"])]
    z_vals = coords[:, 2]
    upper = coords[z_vals >= np.percentile(z_vals, 58)]
    if upper.size == 0:
        upper = coords
    u_upper = upper[:, int(axes["u_axis"])]
    mid = (u_vals.min() + u_vals.max()) * 0.5
    head_right = int((u_upper >= mid).sum()) >= int((u_upper < mid).sum())
    span = int(np.clip(round(u_size * 0.28), 8, 18))
    if head_right:
        u1 = int(np.clip(u_vals.max() + 1, 1, u_size))
        u0 = max(0, u1 - span)
    else:
        u0 = int(np.clip(u_vals.min(), 0, u_size - 1))
        u1 = min(u_size, u0 + span)
    z1 = int(np.clip(np.percentile(z_vals, 98) + 1, 1, sz))
    z0 = int(np.clip(z1 - max(10, int(round(sz * 0.36))), 0, sz - 1))
    return u0, z0, u1, z1


def _panel_from_head_region(
    region: tuple[int, int, int, int],
    grid_shape: tuple[int, int, int],
    axes: dict,
) -> tuple[int, int, int, int]:
    _sx, _sy, sz = grid_shape
    u_size = int(axes["u_size"])
    u0, z0, u1, z1 = region
    center_u = int(round((u0 + u1 - 1) * 0.5))
    center_z = int(round((z0 + z1 - 1) * 0.50))
    panel_w = int(np.clip(round(u_size * 0.18), 7, 13))
    panel_h = int(np.clip(round(sz * 0.27), 9, 15))
    if u1 - u0 < panel_w:
        panel_w = min(u_size, max(7, u1 - u0 + 2))
    if z1 - z0 < panel_h:
        panel_h = min(sz, max(9, z1 - z0 + 1))
    pu0 = int(np.clip(center_u - panel_w // 2, 0, max(0, u_size - panel_w)))
    pz0 = int(np.clip(center_z - panel_h // 2, 0, max(0, sz - panel_h)))
    return pu0, pz0, min(u_size, pu0 + panel_w), min(sz, pz0 + panel_h)


def _front_panel_depth(surface: np.ndarray, panel: tuple[int, int, int, int], axes: dict) -> int | None:
    u0, z0, u1, z1 = panel
    vals = surface[u0:u1, z0:z1]
    vals = vals[vals >= 0]
    if vals.size == 0:
        return None
    if int(axes["sign"]) < 0:
        return int(np.clip(np.percentile(vals, 8), 0, int(axes["depth_size"]) - 1))
    return int(np.clip(np.percentile(vals, 92), 0, int(axes["depth_size"]) - 1))


def _build_face_sprite(
    panel: tuple[int, int, int, int],
    depth: int,
    axes: dict,
    grid_shape: tuple[int, int, int],
    *,
    iris_color: int,
    nose_color: int,
    base_color: int | None,
    species: str,
) -> list[dict]:
    u0, z0, u1, z1 = panel
    _sx, _sy, sz = grid_shape
    additions: list[dict] = []
    used: set[tuple[int, int]] = set()

    # Small base-color cheek pads make the face read as one designed module
    # without hiding the body's real side markings.
    if base_color is not None:
        for fu, fz in ((0.33, 0.50), (0.67, 0.50), (0.50, 0.42)):
            u, z = _coord_from_panel(panel, fu, fz)
            if (u, z) not in used:
                used.add((u, z))
                additions.append(_mounted_face_piece(u, depth, z, int(base_color), "face_base", axes))

    left_eye = _coord_from_panel(panel, 0.29, 0.64)
    right_eye = _coord_from_panel(panel, 0.71, 0.64)
    for side, (u, z) in (("left", left_eye), ("right", right_eye)):
        outward = -1 if side == "left" else 1
        iris_u = int(np.clip(u + outward, u0, u1 - 1))
        # Keep eyes compact: one warm iris tile + one black pupil tile. At this
        # scale, larger ovals look cartoony and overwhelm the face.
        for tu, tz, color, part in (
            (iris_u, z, iris_color, "eye_iris"),
            (u, z, BLACK, "eye_pupil"),
        ):
            if not (u0 <= tu < u1 and z0 <= tz < z1 and 0 <= tz < sz):
                continue
            if (tu, tz) in used:
                continue
            used.add((tu, tz))
            additions.append(_mounted_face_piece(tu, depth, tz, color, part, axes))

    # Nose sits centered and lower. For cats/rabbits this is the only mouth-area
    # mark; the old black mouth line read as a random stripe.
    nu, nz = _coord_from_panel(panel, 0.50, 0.39)
    if (nu, nz) not in used:
        used.add((nu, nz))
        additions.append(_mounted_face_piece(nu, depth, nz, nose_color, "nose", axes))

    # Dogs can support a tiny dark nose pad; cats/rabbits stay restrained.
    if species == "dog":
        for fu in (0.46, 0.54):
            u, z = _coord_from_panel(panel, fu, 0.37)
            if (u, z) not in used:
                used.add((u, z))
                additions.append(_mounted_face_piece(u, depth, z, nose_color, "nose", axes))
    return additions


def _coord_from_panel(panel: tuple[int, int, int, int], fu: float, fz: float) -> tuple[int, int]:
    u0, z0, u1, z1 = panel
    w = max(1, u1 - u0)
    h = max(1, z1 - z0)
    u = int(np.clip(round(u0 + fu * (w - 1)), u0, u1 - 1))
    z = int(np.clip(round(z0 + fz * (h - 1)), z0, z1 - 1))
    return u, z


def _mounted_face_piece(
    u: int,
    depth: int,
    z: int,
    color_id: int,
    face_part: str,
    axes: dict,
) -> dict:
    coord = [0, 0, int(z)]
    coord[int(axes["u_axis"])] = int(u)
    coord[int(axes["depth_axis"])] = int(depth)
    return {
        "x": int(coord[0]),
        "y": int(coord[1]),
        "z": int(coord[2]),
        "size_x": 1,
        "size_y": 1,
        "brick_type": "1x1",
        "kind": "tile",
        "rotation": 0,
        "color": int(color_id),
        "slope_dir": None,
        "mount": axes["front_axis"],
        "protected": True,
        "face_map": face_part,
        "face_module": True,
    }


def _species_name(gpt_data: dict | None) -> str:
    text = " ".join(
        str(v or "")
        for v in (
            (gpt_data or {}).get("subject_name"),
            (gpt_data or {}).get("subject"),
            (gpt_data or {}).get("animal"),
        )
    ).lower()
    for region in (gpt_data or {}).get("regions") or []:
        text += " " + str(region.get("name") or "").lower()
    if "dog" in text or "puppy" in text or "golden" in text:
        return "dog"
    if "rabbit" in text or "bunny" in text:
        return "rabbit"
    if "cat" in text or "kitten" in text:
        return "cat"
    return "pet"


def _sample_iris_color(face_photo_path: Path, eyes: list[tuple[int, int]], *, species: str) -> int:
    fallback = BLACK if species == "rabbit" else DARK_ORANGE
    if species == "dog":
        fallback = DARK_TAN
    if not eyes:
        return fallback
    try:
        from PIL import Image
        from voxels_to_palette import AVAILABILITY_PENALTY, _rgb_to_lab, load_palette

        img = Image.open(face_photo_path).convert("RGBA")
        W, H = img.size
        samples = []
        for u, v in eyes[:2]:
            r = max(8, min(W, H) // 70)
            x0, x1 = max(0, int(u) - r), min(W, int(u) + r + 1)
            y0, y1 = max(0, int(v) - r), min(H, int(v) + r + 1)
            crop = np.asarray(img.crop((x0, y0, x1, y1)))
            if crop.size == 0:
                continue
            rgb = crop[..., :3].astype(np.float32)
            alpha = crop[..., 3] > 16
            maxc = rgb.max(axis=2)
            minc = rgb.min(axis=2)
            sat = (maxc - minc) / np.maximum(maxc, 1.0)
            bright = rgb.mean(axis=2)
            # Iris is usually the most saturated non-black, non-white material
            # around the dark pupil. This excludes fur and glints better than a
            # raw median crop.
            mask = alpha & (sat > 0.22) & (bright > 35) & (bright < 230)
            if int(mask.sum()) < 4:
                mask = alpha & (bright > 40) & (bright < 210)
            if int(mask.sum()) >= 1:
                samples.append(np.median(rgb[mask], axis=0))
        if not samples:
            return fallback
        sample = np.median(np.asarray(samples, dtype=np.float32), axis=0)
        palette = load_palette()
        candidates = {18, 20, 21, 23, 24, 25, 26, 27, 29, 30}
        if species == "dog":
            candidates = {4, 25, 26, 27, 29, 30}
        entries = [p for p in palette if int(p["id"]) in candidates]
        pal_rgb = np.array([p["rgb"] for p in entries], dtype=np.float32) / 255.0
        target_lab = _rgb_to_lab((sample[None] / 255.0))[0]
        pal_lab = _rgb_to_lab(pal_rgb)
        penalty = np.array([
            AVAILABILITY_PENALTY.get(p.get("availability", "uncommon"), 0.0)
            for p in entries
        ], dtype=np.float32) ** 2
        dists = ((pal_lab - target_lab) ** 2).sum(axis=1) + penalty
        picked = int(entries[int(dists.argmin())]["id"])
        if species in {"cat", "pet"} and picked in {20, 21}:
            return DARK_ORANGE
        return picked
    except Exception:
        return fallback


def _sample_nose_color(face_photo_path: Path, feat: dict, *, species: str) -> int:
    if species == "dog":
        fallback = BLACK
        candidates = {4, 25, 26, 27, 29, 30}
    else:
        fallback = PINK
        # Pet noses should read as a deliberate rosy/dark accent. Allowing Tan
        # made cat noses disappear into the fur when CV landed slightly outside
        # the pink pixels.
        candidates = {26, 29, 30, 31}
    nose = feat.get("nose")
    if not nose:
        return fallback
    try:
        from face_map import _sample_feature_palette_id

        return _sample_feature_palette_id(
            face_photo_path,
            int(nose[0]),
            int(nose[1]),
            candidates=candidates,
            fallback=fallback,
        )
    except Exception:
        return fallback


def _most_common_head_color(
    bricks: list[dict],
    region: tuple[int, int, int, int],
    axes: dict,
    *,
    fallback: int | None,
) -> int | None:
    u0, z0, u1, z1 = region
    counts: Counter[int] = Counter()
    for b in bricks:
        if b.get("mount") or b.get("face_axis"):
            continue
        z = int(b.get("z", 0))
        if not (z0 <= z < z1):
            continue
        for du in range(int(b.get("size_x", 1)) if int(axes["u_axis"]) == 0 else int(b.get("size_y", 1))):
            u = int(b.get("x", 0)) + du if int(axes["u_axis"]) == 0 else int(b.get("y", 0)) + du
            if u0 <= u < u1:
                counts[int(b.get("color", fallback or MEDIUM_NOUGAT))] += 1
    if not counts:
        return fallback
    # Do not use feature colors as the cheek/base patch.
    for color, _n in counts.most_common():
        if color not in {BLACK, PINK}:
            return int(color)
    return fallback
