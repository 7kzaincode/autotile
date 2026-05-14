"""Clean LEGO-style color maps for pets.

Raw photo projection turns fur into noisy color speckles after LEGO palette
quantization. For organic subjects we want fewer, larger, intentional regions:
main fur, darker fur, cream muzzle/belly, etc. This module builds a coarse 2D
color map from a blurred/cropped photo and projects that map onto the voxel
grid before final quantization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from voxels_to_palette import _rgb_to_lab, AVAILABILITY_PENALTY


DEFAULT_PET_COLORS = [
    "White", "Cream",
    "Light Bluish Gray", "Dark Bluish Gray", "Black",
    "Tan", "Dark Tan", "Medium Nougat",
    "Brown", "Reddish Brown", "Dark Brown",
]

MARKING_COLORS = [
    "Black", "Dark Bluish Gray", "Dark Brown",
    "Brown", "Reddish Brown", "Dark Tan", "Medium Nougat",
]


def project_pet_color_map(
    grid,
    photo_path: str | Path,
    palette: list[dict],
    gpt_data: dict | None = None,
    *,
    front_axis: str = "-y",
    out_dir: str | Path | None = None,
    debug_stem: str | None = None,
    smooth_iterations: int = 3,
    side_depth: str | None = None,
    markings_on_visible_side_only: bool = False,
    paint_scope: str = "all",
) -> tuple[set[int], dict]:
    """Overwrite voxel RGBs with a smoothed pet color map.

    Returns ``(used_palette_ids, meta)``. The resulting colors are still RGB;
    the normal downstream quantizer converts them to palette IDs.
    """
    sx, sy, sz = grid.occupancy.shape
    depth_axis, u_axis, u_size = _photo_axes(front_axis, (sx, sy, sz))
    if u_size <= 0 or sz <= 0:
        return set(), {"applied": False, "reason": "empty grid"}

    img = Image.open(photo_path).convert("RGBA")
    rgba = np.asarray(img)
    alpha = rgba[..., 3]
    fg = alpha > 16
    if not fg.any():
        return set(), {"applied": False, "reason": "no foreground"}

    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - np.argmax(cols[::-1]))

    crop = img.crop((x0, y0, x1, y1))
    # Heavy blur kills fur strands and texture noise before downsampling.
    blur_radius = max(3.0, min(crop.size) / 90.0)
    crop = crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    small = crop.resize((u_size, sz), Image.Resampling.BILINEAR)
    small_arr = np.asarray(small)
    small_rgb = small_arr[..., :3].astype(np.float32)
    small_alpha = small_arr[..., 3] > 24

    candidates = _candidate_palette_ids(palette, gpt_data, small_rgb, small_alpha)
    if not candidates:
        return set(), {"applied": False, "reason": "no candidate palette colors"}

    base = _gpt_base_palette_id(gpt_data, palette, candidates)
    if base is None:
        base = _pick_base_palette_id(small_rgb, small_alpha, palette, candidates)
    labels = np.zeros(small_rgb.shape[:2], dtype=np.int32)
    labels[small_alpha] = int(base)
    labels, light_meta = _apply_light_coat_overlay(
        labels, small_rgb, small_alpha, palette, candidates, int(base))
    labels, region_meta = _apply_gpt_region_overlays(
        labels, gpt_data, palette, candidates, int(base),
        image_size=img.size,
        crop_bbox=(x0, y0, x1, y1),
        small_shape=labels.shape,
    )
    marking_hint = _marking_hint_mask(
        gpt_data,
        image_size=img.size,
        crop_bbox=(x0, y0, x1, y1),
        small_shape=labels.shape,
    )
    labels, marking_meta = _apply_marking_overlay(
        labels, small_rgb, small_alpha, palette, candidates,
        hint_mask=marking_hint,
    )
    labels = _majority_filter(labels, small_alpha, iterations=max(0, min(smooth_iterations, 2)))
    labels = _remove_tiny_islands(labels, small_alpha)
    labels, snap_meta = _snap_markings_to_occupied(labels, grid.occupancy, int(u_axis), int(base))
    marking_meta["snapped_cells"] = snap_meta.get("snapped", 0)

    used: set[int] = set()
    pal_rgb = {int(p["id"]): tuple(int(v) for v in p["rgb"]) for p in palette}
    if base is None:
        base = _mode_label(labels[labels > 0]) or candidates[0]
    marking_ids = set(_marking_palette_ids(palette, candidates))
    paint_scope = (paint_scope or "all").strip().lower()
    side_depth = (side_depth or "").strip().lower()
    side_surface = None
    if side_depth in {"front", "back"} or paint_scope == "surface_only":
        side_surface = _side_surface_depth_map(
            grid.occupancy, depth_axis=int(depth_axis), u_axis=int(u_axis),
            front_axis=front_axis, side=side_depth if side_depth in {"front", "back"} else "front",
        )

    # labels are stored image-top to image-bottom. Voxel z is bottom to top.
    # Voxel horizontal photo coord is X for front portraits, or Y for side/full
    # body pets when the chosen front axis is X.
    occ_idx = np.argwhere(grid.occupancy)
    for x, y, z in occ_idx:
        row = int(np.clip(sz - 1 - z, 0, sz - 1))
        coord = (int(x), int(y), int(z))
        col = int(np.clip(coord[u_axis], 0, u_size - 1))
        on_side_surface = True
        if side_surface is not None:
            surface_depth = int(side_surface[col, int(z)])
            depth = int(coord[depth_axis])
            on_side_surface = surface_depth >= 0 and abs(depth - surface_depth) <= 1
        if paint_scope == "surface_only" and not on_side_surface:
            continue
        pid = int(labels[row, col])
        if pid <= 0:
            pid = int(base)
        if (
            markings_on_visible_side_only
            and pid in marking_ids
            and pid != int(base)
            and not on_side_surface
        ):
            pid = int(base)
        grid.colors[x, y, z] = pal_rgb.get(pid, pal_rgb[int(base)])
        used.add(pid)

    debug_path = None
    if out_dir is not None:
        try:
            debug_path = _save_debug(labels, palette, out_dir, debug_stem or Path(photo_path).stem)
        except Exception as e:
            print(f"[pet-color-map] debug save skipped: {e}")

    return used, {
        "applied": True,
        "foreground_bbox": [x0, y0, x1, y1],
        "grid_map_shape": [u_size, sz],
        "front_axis": front_axis,
        "u_axis": int(u_axis),
        "depth_axis": int(depth_axis),
        "base_id": int(base),
        "base_name": _palette_name(palette, int(base)),
        "candidates": candidates,
        "used_ids": sorted(used),
        "debug_path": str(debug_path) if debug_path else None,
        "markings": marking_meta,
        "light_regions": light_meta,
        "gpt_regions": region_meta,
        "side_depth": side_depth or None,
        "paint_scope": paint_scope,
        "markings_on_visible_side_only": bool(markings_on_visible_side_only),
    }


def _photo_axes(front_axis: str, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    sx, sy, _sz = shape
    axis = (front_axis or "-y").strip().lower().lstrip("+-")
    if axis == "x":
        return 0, 1, sy
    return 1, 0, sx


def _axis_sign(front_axis: str) -> int:
    return -1 if (front_axis or "-y").strip().lower().startswith("-") else 1


def _coord_from_depth_u(
    depth: int,
    u: int,
    z: int,
    *,
    depth_axis: int,
    u_axis: int,
) -> tuple[int, int, int]:
    coord = [0, 0, int(z)]
    coord[int(depth_axis)] = int(depth)
    coord[int(u_axis)] = int(u)
    return int(coord[0]), int(coord[1]), int(coord[2])


def _side_surface_depth_map(
    occupancy: np.ndarray,
    *,
    depth_axis: int,
    u_axis: int,
    front_axis: str,
    side: str = "front",
) -> np.ndarray:
    """Return depth index of the visible side surface for each photo u/z cell."""
    sx, sy, sz = occupancy.shape
    shape = (sx, sy, sz)
    depth_size = int(shape[int(depth_axis)])
    u_size = int(shape[int(u_axis)])
    sign = _axis_sign(front_axis)
    side = (side or "front").lower()
    front_order = range(depth_size) if sign < 0 else range(depth_size - 1, -1, -1)
    back_order = range(depth_size - 1, -1, -1) if sign < 0 else range(depth_size)
    order = front_order if side == "front" else back_order
    out = np.full((u_size, sz), -1, dtype=np.int32)
    for u in range(u_size):
        for z in range(sz):
            for depth in order:
                if occupancy[_coord_from_depth_u(depth, u, z, depth_axis=depth_axis, u_axis=u_axis)]:
                    out[u, z] = int(depth)
                    break
    return out


def _snap_markings_to_occupied(
    labels: np.ndarray,
    occupancy: np.ndarray,
    u_axis: int,
    base_id: int,
    *,
    radius: int = 4,
) -> tuple[np.ndarray, dict]:
    sx, sy, sz = occupancy.shape
    u_size = sy if u_axis == 1 else sx
    occupied_uv = np.zeros((sz, u_size), dtype=bool)
    for x, y, z in np.argwhere(occupancy):
        col = int(y if u_axis == 1 else x)
        row = int(np.clip(sz - 1 - z, 0, sz - 1))
        if 0 <= col < u_size:
            occupied_uv[row, col] = True
    if not occupied_uv.any():
        return labels, {"snapped": 0}

    out = labels.copy()
    snapped = 0
    rows, cols = np.where((labels > 0) & (labels != int(base_id)))
    for row, col in zip(rows, cols):
        if 0 <= row < occupied_uv.shape[0] and 0 <= col < occupied_uv.shape[1] and occupied_uv[row, col]:
            continue
        hit = None
        for r in range(1, radius + 1):
            candidates = []
            y0, y1 = max(0, row - r), min(occupied_uv.shape[0], row + r + 1)
            x0, x1 = max(0, col - r), min(occupied_uv.shape[1], col + r + 1)
            ys, xs = np.where(occupied_uv[y0:y1, x0:x1])
            for dy, dx in zip(ys, xs):
                rr = y0 + int(dy)
                cc = x0 + int(dx)
                candidates.append((abs(rr - row) + abs(cc - col), rr, cc))
            if candidates:
                _d, rr, cc = min(candidates, key=lambda t: t[0])
                hit = (rr, cc)
                break
        if hit is None:
            continue
        out[hit] = int(labels[row, col])
        snapped += 1
    return out, {"snapped": snapped}


def _candidate_palette_ids(
    palette: list[dict],
    gpt_data: dict | None,
    small_rgb: np.ndarray,
    small_alpha: np.ndarray,
) -> list[int]:
    by_name = {p["name"].lower(): int(p["id"]) for p in palette}
    names = list(DEFAULT_PET_COLORS)

    for r in (gpt_data or {}).get("regions", []) or []:
        name = (r.get("name") or "").lower()
        color = r.get("color_name")
        # Eyes/nose/mouth are protected face-map details, not body color.
        if name in {"eye", "eyes", "eye_socket", "nose", "mouth", "pupil"}:
            continue
        if color:
            names.append(color)
    for color in (gpt_data or {}).get("recommended_lego_palette", []) or []:
        if color not in {"Black", "Pink"}:
            names.append(color)

    # White is part of the base pet palette, but this signal remains useful
    # for callers/debugging and keeps explicit GPT recommendations harmless.
    if small_alpha.any():
        fg_rgb = small_rgb[small_alpha]
        light_frac = float(((fg_rgb.min(axis=1) > 215) & (fg_rgb.max(axis=1) - fg_rgb.min(axis=1) < 28)).mean())
        if light_frac >= 0.10:
            names.append("White")

    out = []
    for name in names:
        pid = by_name.get(name.lower())
        if pid is not None and pid not in out:
            out.append(pid)
    return out


def _palette_name(palette: list[dict], pid: int) -> str:
    for p in palette:
        if int(p["id"]) == int(pid):
            return str(p["name"])
    return str(pid)


def _gpt_base_palette_id(
    gpt_data: dict | None,
    palette: list[dict],
    candidate_ids: list[int],
) -> int | None:
    if not gpt_data:
        return None
    broad = {"body", "coat", "fur", "torso", "head", "back", "chest", "belly"}
    counts: dict[str, int] = {}
    for r in gpt_data.get("regions", []) or []:
        name = (r.get("name") or "").strip().lower().replace("-", "_")
        color = r.get("color_name")
        if not color or not any(k in name for k in broad):
            continue
        counts[color] = counts.get(color, 0) + 1
    if not counts:
        return None
    picked = max(counts.items(), key=lambda kv: kv[1])[0]
    for p in palette:
        if p["name"].lower() == picked.lower() and int(p["id"]) in candidate_ids:
            return int(p["id"])
    return None


def _pick_base_palette_id(
    small_rgb: np.ndarray,
    small_alpha: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
) -> int:
    if not small_alpha.any():
        return int(candidate_ids[0])
    brightness = small_rgb.astype(np.float32).mean(axis=2)
    fg_vals = brightness[small_alpha]
    cutoff = float(np.percentile(fg_vals, 55))
    base_pixels = small_alpha & (brightness >= cutoff)
    if int(base_pixels.sum()) < 4:
        base_pixels = small_alpha
    rgb = np.median(small_rgb[base_pixels], axis=0)

    by_name = {p["name"].lower(): int(p["id"]) for p in palette}
    preferred_names = [
        "White", "Cream", "Tan", "Dark Tan", "Medium Nougat",
        "Reddish Brown", "Brown", "Light Bluish Gray",
    ]
    preferred = [
        by_name[name.lower()] for name in preferred_names
        if by_name.get(name.lower()) in candidate_ids
    ]
    pid = _nearest_palette_id(rgb, palette, preferred or candidate_ids)
    return int(pid if pid is not None else candidate_ids[0])


def _apply_light_coat_overlay(
    labels: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
    base_id: int,
) -> tuple[np.ndarray, dict]:
    """Preserve real white/cream coat areas before dark marking extraction."""
    out = labels.copy()
    if not mask.any():
        return out, {"components": 0}
    by_name = {p["name"].lower(): int(p["id"]) for p in palette}
    light_ids = [
        by_name[name.lower()]
        for name in ("White", "Cream")
        if by_name.get(name.lower()) in candidate_ids
    ]
    if not light_ids:
        return out, {"components": 0, "reason": "no light palette"}
    rgb_f = rgb.astype(np.float32)
    spread = rgb_f.max(axis=2) - rgb_f.min(axis=2)
    light = mask & (rgb_f.min(axis=2) >= 205.0) & (spread <= 46.0)
    if int(light.sum()) < max(3, int(mask.sum() * 0.025)):
        return out, {"components": 0, "reason": "too little light coat"}
    try:
        from scipy.ndimage import binary_closing, label as cc_label
        light = binary_closing(light, iterations=1) & mask
        comps, n = cc_label(light)
    except Exception:
        comps, n = light.astype(np.int32), 1
    min_size = max(3, int(mask.sum() * 0.008))
    max_size = max(min_size + 1, int(mask.sum() * 0.72))
    kept = 0
    cells = 0
    used: set[int] = set()
    for cid in range(1, n + 1):
        comp = comps == cid
        size = int(comp.sum())
        if size < min_size or size > max_size:
            continue
        pid = _nearest_palette_id(np.median(rgb[comp], axis=0), palette, light_ids)
        if pid is None or int(pid) == int(base_id):
            continue
        out[comp] = int(pid)
        kept += 1
        cells += size
        used.add(int(pid))
    return out, {"components": kept, "cells": cells, "used_ids": sorted(used)}


def _apply_gpt_region_overlays(
    labels: np.ndarray,
    gpt_data: dict | None,
    palette: list[dict],
    candidate_ids: list[int],
    base_id: int,
    *,
    image_size: tuple[int, int],
    crop_bbox: tuple[int, int, int, int],
    small_shape: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    """Use GPT's broad coat regions for LEGO-readable secondary colors.

    SAM surface-paint is disabled for side pets because it can create texture
    noise, but GPT still knows things like "white belly" or "white paws".
    This pass converts only those broad, non-face, non-marking bboxes into
    coarse color blocks before the smoothing filters run.
    """
    if not gpt_data:
        return labels, {"regions": 0}
    by_name = {p["name"].lower(): int(p["id"]) for p in palette}
    Hs, Ws = small_shape
    W, H = image_size
    cx0, cy0, cx1, cy1 = crop_bbox
    crop_w = max(1, cx1 - cx0)
    crop_h = max(1, cy1 - cy0)
    paint_keys = (
        "belly", "chest", "paws", "paw", "leg", "legs",
        "muzzle", "neck", "throat", "bib", "sock", "socks",
    )
    skip_keys = (
        "body", "coat", "fur", "torso", "back", "tail", "ear",
        "eye", "nose", "mouth", "stripe", "striped", "mark",
        "pattern", "patch", "spot",
    )
    out = labels.copy()
    painted = 0
    used: set[int] = set()
    for r in gpt_data.get("regions", []) or []:
        name = (r.get("name") or "").strip().lower().replace("-", "_")
        if not any(k in name for k in paint_keys):
            continue
        if any(k in name for k in skip_keys if k not in {"muzzle"}):
            continue
        color = r.get("color_name")
        pid = by_name.get(str(color or "").lower())
        if pid is None or pid not in candidate_ids or int(pid) == int(base_id):
            continue
        bb = r.get("bbox_normalized") or []
        if len(bb) != 4:
            continue
        x0, y0, x1, y1 = bb
        px0, py0 = int(round(x0 * W)), int(round(y0 * H))
        px1, py1 = int(round(x1 * W)), int(round(y1 * H))
        ix0, iy0 = max(px0, cx0), max(py0, cy0)
        ix1, iy1 = min(px1, cx1), min(py1, cy1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        sx0 = int(np.clip((ix0 - cx0) / crop_w * Ws, 0, Ws - 1))
        sx1 = int(np.clip(np.ceil((ix1 - cx0) / crop_w * Ws), sx0 + 1, Ws))
        sy0 = int(np.clip((iy0 - cy0) / crop_h * Hs, 0, Hs - 1))
        sy1 = int(np.clip(np.ceil((iy1 - cy0) / crop_h * Hs), sy0 + 1, Hs))
        region_mask = np.zeros_like(out, dtype=bool)
        region_mask[sy0:sy1, sx0:sx1] = True
        region_mask &= out > 0
        area = int(region_mask.sum())
        if area < max(2, int((out > 0).sum() * 0.004)):
            continue
        if area > int((out > 0).sum() * 0.45):
            continue
        out[region_mask] = int(pid)
        painted += 1
        used.add(int(pid))
    return out, {"regions": painted, "used_ids": sorted(used)}


def _marking_hint_mask(
    gpt_data: dict | None,
    *,
    image_size: tuple[int, int],
    crop_bbox: tuple[int, int, int, int],
    small_shape: tuple[int, int],
) -> np.ndarray | None:
    if not gpt_data:
        return None
    Hs, Ws = small_shape
    W, H = image_size
    cx0, cy0, cx1, cy1 = crop_bbox
    crop_w = max(1, cx1 - cx0)
    crop_h = max(1, cy1 - cy0)
    mask = np.zeros((Hs, Ws), dtype=bool)
    keys = ("stripe", "striped", "mark", "pattern", "patch", "spot", "tail_tip")
    for r in gpt_data.get("regions", []) or []:
        name = (r.get("name") or "").strip().lower().replace("-", "_")
        if not any(k in name for k in keys):
            continue
        bb = r.get("bbox_normalized") or []
        if len(bb) != 4:
            continue
        x0, y0, x1, y1 = bb
        px0, py0 = int(round(x0 * W)), int(round(y0 * H))
        px1, py1 = int(round(x1 * W)), int(round(y1 * H))
        ix0, iy0 = max(px0, cx0), max(py0, cy0)
        ix1, iy1 = min(px1, cx1), min(py1, cy1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        sx0 = int(np.clip((ix0 - cx0) / crop_w * Ws, 0, Ws - 1))
        sx1 = int(np.clip(np.ceil((ix1 - cx0) / crop_w * Ws), sx0 + 1, Ws))
        sy0 = int(np.clip((iy0 - cy0) / crop_h * Hs, 0, Hs - 1))
        sy1 = int(np.clip(np.ceil((iy1 - cy0) / crop_h * Hs), sy0 + 1, Hs))
        mask[sy0:sy1, sx0:sx1] = True
    if not mask.any():
        return None
    try:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=1)
    except Exception:
        pass
    return mask


def _quantize_small_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
) -> np.ndarray:
    labels = np.zeros(rgb.shape[:2], dtype=np.int32)
    entries = [p for p in palette if int(p["id"]) in candidate_ids]
    pal_ids = np.array([int(p["id"]) for p in entries], dtype=np.int32)
    pal_rgb = np.array([p["rgb"] for p in entries], dtype=np.float32) / 255.0
    pal_lab = _rgb_to_lab(pal_rgb)
    penalty = np.array([
        AVAILABILITY_PENALTY.get(p.get("availability", "uncommon"), 0.0)
        for p in entries
    ], dtype=np.float32) ** 2

    pixels = rgb.astype(np.float32) / 255.0
    labs = _rgb_to_lab(pixels.reshape(-1, 3)).reshape(*rgb.shape[:2], 3)
    yy, xx = np.where(mask)
    for y, x in zip(yy, xx):
        d = ((pal_lab - labs[y, x]) ** 2).sum(axis=1) + penalty
        labels[y, x] = int(pal_ids[int(d.argmin())])
    return labels


def _fill_missing(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = labels.copy()
    base = _mode_label(out[out > 0]) or 0
    out[(out <= 0) & mask] = base
    return out


def _majority_filter(labels: np.ndarray, mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    out = labels.copy()
    H, W = out.shape
    for _ in range(max(0, iterations)):
        cur = out.copy()
        for y in range(H):
            for x in range(W):
                if not mask[y, x]:
                    continue
                y0, y1 = max(0, y - 1), min(H, y + 2)
                x0, x1 = max(0, x - 1), min(W, x + 2)
                vals = cur[y0:y1, x0:x1]
                m = mask[y0:y1, x0:x1]
                picked = _mode_label(vals[m & (vals > 0)])
                if picked is not None:
                    out[y, x] = picked
    return out


def _remove_tiny_islands(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = labels.copy()
    min_size = max(2, int(mask.sum() * 0.003))
    try:
        from scipy.ndimage import label as cc_label
    except Exception:
        return out

    for pid in sorted(int(v) for v in np.unique(out) if v > 0):
        comps, n = cc_label((out == pid) & mask)
        for cid in range(1, n + 1):
            comp = comps == cid
            if int(comp.sum()) >= min_size:
                continue
            ys, xs = np.where(comp)
            for y, x in zip(ys, xs):
                y0, y1 = max(0, y - 1), min(out.shape[0], y + 2)
                x0, x1 = max(0, x - 1), min(out.shape[1], x + 2)
                neighbors = out[y0:y1, x0:x1]
                picked = _mode_label(neighbors[(neighbors > 0) & (neighbors != pid)])
                if picked is not None:
                    out[y, x] = picked
    return out


def _apply_marking_overlay(
    labels: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
    *,
    hint_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Preserve intentional dark/asymmetric markings after smoothing.

    The main color map deliberately blurs fur texture, but a pet's identity is
    often a one-sided patch, striped leg, dark tail tip, or muzzle spot. This
    pass detects connected high-contrast dark regions in the coarse photo map
    and stamps them back as larger LEGO-readable regions.
    """
    out = labels.copy()
    fg = mask & (labels > 0)
    if int(fg.sum()) < 12:
        return out, {"components": 0, "reason": "too little foreground"}

    try:
        from scipy.ndimage import binary_dilation, label as cc_label
    except Exception:
        return out, {"components": 0, "reason": "scipy unavailable"}

    brightness = rgb.astype(np.float32).mean(axis=2)
    fg_brightness = brightness[fg]
    median = float(np.median(fg_brightness))
    p30 = float(np.percentile(fg_brightness, 30))
    p18 = float(np.percentile(fg_brightness, 18))
    dark_threshold = min(median - 28.0, max(p18 + 8.0, p30 - 8.0))
    if dark_threshold < 20.0:
        return out, {"components": 0, "reason": "no dark contrast"}

    dark_mask = fg & (brightness <= dark_threshold)
    if hint_mask is not None and hint_mask.any():
        dark_mask = dark_mask & hint_mask
    if int(dark_mask.sum()) < 2:
        return out, {"components": 0, "threshold": round(dark_threshold, 2)}

    marking_ids = _marking_palette_ids(palette, candidate_ids)
    if not marking_ids:
        return out, {"components": 0, "reason": "no marking palette colors"}

    comps, n = cc_label(dark_mask)
    min_size = max(2, int(fg.sum() * 0.004))
    max_frac = 0.22 if hint_mask is None else 0.25
    max_size = max(min_size + 1, int(fg.sum() * max_frac))
    kept = 0
    pixels = 0
    used_ids: set[int] = set()

    for cid in range(1, n + 1):
        comp = comps == cid
        size = int(comp.sum())
        if size < min_size or size > max_size:
            continue
        rgb_med = np.median(rgb[comp], axis=0)
        pid = _nearest_palette_id(rgb_med, palette, marking_ids)
        if pid is None:
            continue
        grown = binary_dilation(comp, iterations=1) & fg
        out[grown] = int(pid)
        kept += 1
        pixels += int(grown.sum())
        used_ids.add(int(pid))

    if kept == 0 and hint_mask is not None and hint_mask.any():
        hinted = hint_mask & fg
        comps, n = cc_label(hinted)
        for cid in range(1, n + 1):
            comp = comps == cid
            size = int(comp.sum())
            if size < min_size or size > max_size:
                continue
            rgb_med = np.median(rgb[comp], axis=0)
            pid = _nearest_palette_id(rgb_med, palette, marking_ids)
            if pid is None:
                continue
            out[comp] = int(pid)
            kept += 1
            pixels += int(comp.sum())
            used_ids.add(int(pid))

    return out, {
        "components": kept,
        "cells": pixels,
        "used_ids": sorted(used_ids),
        "threshold": round(dark_threshold, 2),
        "hinted": bool(hint_mask is not None and hint_mask.any()),
    }


def _marking_palette_ids(palette: list[dict], candidate_ids: list[int]) -> list[int]:
    by_name = {p["name"].lower(): int(p["id"]) for p in palette}
    out = []
    for name in MARKING_COLORS:
        pid = by_name.get(name.lower())
        if pid is not None and pid in candidate_ids and pid not in out:
            out.append(pid)
    return out


def _nearest_palette_id(
    rgb: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
) -> int | None:
    entries = [p for p in palette if int(p["id"]) in candidate_ids]
    if not entries:
        return None
    pal_ids = np.array([int(p["id"]) for p in entries], dtype=np.int32)
    pal_rgb = np.array([p["rgb"] for p in entries], dtype=np.float32) / 255.0
    pal_lab = _rgb_to_lab(pal_rgb)
    lab = _rgb_to_lab(np.asarray(rgb, dtype=np.float32).reshape(1, 3) / 255.0)[0]
    penalty = np.array([
        AVAILABILITY_PENALTY.get(p.get("availability", "uncommon"), 0.0)
        for p in entries
    ], dtype=np.float32) ** 2
    d = ((pal_lab - lab) ** 2).sum(axis=1) + penalty
    return int(pal_ids[int(d.argmin())])


def _mode_label(vals) -> int | None:
    vals = np.asarray(vals, dtype=np.int32)
    vals = vals[vals > 0]
    if vals.size == 0:
        return None
    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[int(counts.argmax())])


def _save_debug(labels: np.ndarray, palette: list[dict], out_dir: str | Path, stem: str) -> Path:
    pal = {int(p["id"]): tuple(int(v) for v in p["rgb"]) for p in palette}
    img = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for pid in np.unique(labels):
        if int(pid) <= 0:
            img[labels == pid] = (255, 255, 255)
        else:
            img[labels == pid] = pal.get(int(pid), (180, 180, 180))
    out = Image.fromarray(img, "RGB").resize(
        (max(96, labels.shape[1] * 12), max(96, labels.shape[0] * 12)),
        Image.Resampling.NEAREST,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_pet_color_map.png"
    out.save(path)
    return path
