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
    "Orange", "Dark Orange", "Bright Light Orange",
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
    front_photo_path: str | Path | None = None,
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

    side_depth = (side_depth or "").strip().lower()
    crop = img.crop((x0, y0, x1, y1))
    raw_crop_rgba = np.asarray(crop)
    raw_marking_mask, raw_marking_meta = _extract_high_res_marking_mask(raw_crop_rgba)
    lego_marking_mask = _resize_marking_mask(raw_marking_mask, (sz, u_size))
    # Heavy blur kills fur strands and texture noise before downsampling.
    blur_radius = max(3.0, min(crop.size) / 90.0)
    crop = crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    small = crop.resize((u_size, sz), Image.Resampling.BILINEAR)
    small_arr = np.asarray(small)
    small_rgb = small_arr[..., :3].astype(np.float32)
    small_alpha = small_arr[..., 3] > 24
    front_light_profile = _front_photo_light_profile(front_photo_path)

    candidates = _candidate_palette_ids(palette, gpt_data, small_rgb, small_alpha)
    if not candidates:
        return set(), {"applied": False, "reason": "no candidate palette colors"}

    marking_hint = _marking_hint_mask(
        gpt_data,
        image_size=img.size,
        crop_bbox=(x0, y0, x1, y1),
        small_shape=small_alpha.shape,
    )
    feature_exclusion = _face_feature_exclusion_mask(
        gpt_data,
        image_size=img.size,
        crop_bbox=(x0, y0, x1, y1),
        small_shape=small_alpha.shape,
    )
    if feature_exclusion is not None and feature_exclusion.any():
        lego_marking_mask = lego_marking_mask & ~feature_exclusion
    side_layout_meta: dict = {}
    if side_depth in {"front", "back"}:
        lego_marking_mask, side_layout_meta = _filter_side_profile_markings(
            lego_marking_mask, small_alpha)

    base, base_meta = _photo_sampled_base_palette_id(
        small_rgb, small_alpha, palette, candidates,
        marking_mask=lego_marking_mask,
        exclusion_mask=feature_exclusion,
    )
    if base is None:
        base = _gpt_base_palette_id(gpt_data, palette, candidates)
        if base is None:
            base = _pick_base_palette_id(small_rgb, small_alpha, palette, candidates)
        base_meta = {
            "source": "fallback",
            "matched_id": int(base),
            "matched_name": _palette_name(palette, int(base)),
        }

    color_profile = _photo_color_profile(
        small_rgb, small_alpha, palette, candidates,
        base_id=int(base),
        marking_mask=lego_marking_mask,
        exclusion_mask=feature_exclusion,
    )

    labels = np.zeros(small_rgb.shape[:2], dtype=np.int32)
    labels[small_alpha] = int(base)
    visible_side_only_light_mask = np.zeros(labels.shape, dtype=bool)
    before_light = labels.copy()
    labels, light_meta = _apply_light_coat_overlay(
        labels, small_rgb, small_alpha, palette, candidates, int(base),
        preferred_id=color_profile.get("light", {}).get("matched_id"),
    )
    if side_depth in {"front", "back"} and light_meta.get("components", 0) > 0:
        visible_side_only_light_mask |= (
            (labels != before_light)
            & small_alpha
            & (labels != int(base))
        )
        light_meta["visible_side_only_cells"] = int(visible_side_only_light_mask.sum())
    labels, region_meta = _apply_gpt_region_overlays(
        labels, gpt_data, palette, candidates, int(base),
        image_size=img.size,
        crop_bbox=(x0, y0, x1, y1),
        small_shape=labels.shape,
    )
    labels, anatomy_meta, anatomy_debug = _apply_anatomy_light_overlay(
        labels, small_rgb, small_alpha, palette, candidates, int(base),
        preferred_id=color_profile.get("light", {}).get("matched_id"),
        front_light_profile=front_light_profile,
    )
    anatomy_side_only_mask = anatomy_meta.pop("_visible_side_only_mask", None)
    if side_depth in {"front", "back"} and anatomy_side_only_mask is not None:
        visible_side_only_light_mask |= anatomy_side_only_mask
    if side_depth in {"front", "back"}:
        anatomy_meta["visible_side_only_cells"] = int(visible_side_only_light_mask.sum())
    labels, warm_meta = _apply_warm_coat_overlay(
        labels, small_rgb, small_alpha, palette, candidates, int(base),
        preferred_id=color_profile.get("warm_secondary", {}).get("matched_id"),
        marking_mask=lego_marking_mask,
        exclusion_mask=feature_exclusion,
    )
    labels = _majority_filter(labels, small_alpha, iterations=max(0, min(smooth_iterations, 2)))
    labels = _remove_tiny_islands(labels, small_alpha)
    labels, marking_meta = _apply_high_res_marking_overlay(
        labels, small_rgb, small_alpha, palette, candidates,
        marking_mask=lego_marking_mask,
        preferred_id=color_profile.get("marking", {}).get("matched_id"),
    )
    if marking_meta.get("components", 0) <= 0:
        labels, fallback_marking_meta = _apply_marking_overlay(
            labels, small_rgb, small_alpha, palette, candidates,
            hint_mask=marking_hint,
        )
        fallback_marking_meta["source"] = "coarse-fallback"
        marking_meta = fallback_marking_meta
    else:
        marking_meta["source"] = "high-res-mask"
    labels, snap_meta = _snap_markings_to_occupied(labels, grid.occupancy, int(u_axis), int(base))
    marking_meta["snapped_cells"] = snap_meta.get("snapped", 0)
    marking_meta["high_res"] = raw_marking_meta
    marking_meta["face_exclusion_cells"] = (
        int(feature_exclusion.sum()) if feature_exclusion is not None else 0
    )
    marking_meta["side_profile_filter"] = side_layout_meta

    # The old path did this smoothing after markings, which turned thin
    # crescents/dots into generic blobs. Base coat gets smoothed; markings do not.

    used: set[int] = set()
    pal_rgb = {int(p["id"]): tuple(int(v) for v in p["rgb"]) for p in palette}
    if base is None:
        base = _mode_label(labels[labels > 0]) or candidates[0]
    marking_ids = set(_marking_palette_ids(palette, candidates))
    paint_scope = (paint_scope or "all").strip().lower()
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
        if (
            side_depth in {"front", "back"}
            and bool(visible_side_only_light_mask[row, col])
            and not on_side_surface
        ):
            pid = int(base)
        grid.colors[x, y, z] = pal_rgb.get(pid, pal_rgb[int(base)])
        used.add(pid)

    debug_path = None
    marking_debug_path = None
    anatomy_debug_path = None
    if out_dir is not None:
        try:
            debug_path = _save_debug(labels, palette, out_dir, debug_stem or Path(photo_path).stem)
            marking_debug_path = _save_marking_debug(
                lego_marking_mask,
                out_dir,
                debug_stem or Path(photo_path).stem,
            )
            if anatomy_debug is not None:
                anatomy_debug_path = _save_anatomy_debug(
                    anatomy_debug,
                    out_dir,
                    debug_stem or Path(photo_path).stem,
                )
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
        "base_adjustment": base_meta,
        "color_profile": color_profile,
        "candidates": candidates,
        "used_ids": sorted(used),
        "debug_path": str(debug_path) if debug_path else None,
        "marking_debug_path": str(marking_debug_path) if marking_debug_path else None,
        "anatomy_debug_path": str(anatomy_debug_path) if anatomy_debug_path else None,
        "markings": marking_meta,
        "light_regions": light_meta,
        "anatomy_light": anatomy_meta,
        "front_light_profile": front_light_profile,
        "warm_regions": warm_meta,
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


def _photo_sampled_base_palette_id(
    small_rgb: np.ndarray,
    small_alpha: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
    *,
    marking_mask: np.ndarray | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[int | None, dict]:
    """Sample the coat color from the photo, then snap that HEX to LEGO.

    GPT may call a cat "Tan", but the actual photo color can be lighter,
    warmer, or whiter. This is the primary base-coat decision path.
    """
    if not small_alpha.any():
        return None, {"source": "photo-hex", "reason": "empty alpha"}
    sample_mask = small_alpha.copy()
    if marking_mask is not None and marking_mask.shape == sample_mask.shape and marking_mask.any():
        try:
            from scipy.ndimage import binary_dilation
            sample_mask &= ~binary_dilation(marking_mask, iterations=1)
        except Exception:
            sample_mask &= ~marking_mask
    if exclusion_mask is not None and exclusion_mask.shape == sample_mask.shape:
        sample_mask &= ~exclusion_mask
    if int(sample_mask.sum()) < max(8, int(small_alpha.sum() * 0.12)):
        sample_mask = small_alpha.copy()

    rgb_all = small_rgb[sample_mask].astype(np.float32)
    brightness = rgb_all.mean(axis=1)
    spread = rgb_all.max(axis=1) - rgb_all.min(axis=1)
    light = (rgb_all.min(axis=1) >= 214.0) & (spread <= 50.0)
    warm = (
        (rgb_all[:, 0] > rgb_all[:, 1] + 12.0)
        & (rgb_all[:, 1] > rgb_all[:, 2] + 8.0)
        & (brightness >= 105.0)
        & (brightness <= 222.0)
    )
    light_frac = float(light.mean()) if len(light) else 0.0
    warm_frac = float(warm.mean()) if len(warm) else 0.0

    if light_frac >= 0.48 and warm_frac < 0.32:
        coat_pixels = rgb_all[light]
        source = "photo-hex-light-coat"
    else:
        lo, hi = np.percentile(brightness, [28, 76])
        coat_sel = (brightness >= lo) & (brightness <= hi)
        if warm_frac >= 0.18:
            coat_sel &= warm
        coat_pixels = rgb_all[coat_sel]
        source = "photo-hex-coat"
    if len(coat_pixels) < 8:
        coat_pixels = rgb_all
        source = "photo-hex-foreground"

    sample_rgb = np.median(coat_pixels, axis=0)
    pid, dist = _nearest_palette_id_exact(sample_rgb, palette, candidate_ids)
    if pid is None:
        return None, {"source": source, "reason": "no palette match"}
    return int(pid), {
        "source": source,
        "sample_rgb": [int(round(float(v))) for v in sample_rgb],
        "sample_hex": _rgb_hex(sample_rgb),
        "matched_id": int(pid),
        "matched_name": _palette_name(palette, int(pid)),
        "distance": round(float(dist), 2),
        "light_fraction": round(light_frac, 3),
        "warm_fraction": round(warm_frac, 3),
    }


def _photo_color_profile(
    small_rgb: np.ndarray,
    small_alpha: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
    *,
    base_id: int,
    marking_mask: np.ndarray | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> dict:
    """Sample the actual photo HEX colors we want the LEGO map to preserve.

    Base-coat matching alone is not enough for pets. A tan/orange cat often
    needs a light chest/paw color, a warmer coat shadow, and a separate dark
    marking color. These are sampled from pixels before we snap to LEGO.
    """
    profile: dict = {}
    if not small_alpha.any():
        return profile

    rgb = small_rgb.astype(np.float32)
    fg = small_alpha.copy()
    if exclusion_mask is not None and exclusion_mask.shape == fg.shape:
        fg &= ~exclusion_mask
    if int(fg.sum()) < 8:
        fg = small_alpha.copy()

    brightness = rgb.mean(axis=2)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    light_mask = fg & (rgb.min(axis=2) >= 214.0) & (spread <= 55.0)
    if int(light_mask.sum()) >= max(4, int(fg.sum() * 0.018)):
        light_ids = _palette_ids_by_name(palette, candidate_ids, ["White", "Cream"])
        sample = np.median(rgb[light_mask], axis=0)
        pid, dist = _nearest_palette_id_exact(sample, palette, light_ids)
        if pid is not None:
            profile["light"] = _sample_meta(sample, palette, int(pid), dist, "photo-hex-light")

    mark_mask = marking_mask if marking_mask is not None and marking_mask.shape == fg.shape else None
    if mark_mask is not None and int((mark_mask & fg).sum()) >= 2:
        marking_ids = _marking_palette_ids(palette, candidate_ids)
        sample = _dark_marking_sample(rgb[mark_mask & fg])
        pid, dist = _nearest_palette_id_exact(sample, palette, marking_ids)
        if pid is not None:
            profile["marking"] = _sample_meta(sample, palette, int(pid), dist, "photo-hex-marking")

    # A second warm coat color gives orange/sandy animals more life without
    # reintroducing fur noise. It is intentionally limited to lower-brightness
    # warm coat pixels and applied later as broad blocks.
    coat = fg.copy()
    if mark_mask is not None:
        try:
            from scipy.ndimage import binary_dilation
            coat &= ~binary_dilation(mark_mask, iterations=1)
        except Exception:
            coat &= ~mark_mask
    coat &= ~light_mask
    if int(coat.sum()) >= 12:
        coat_rgb = rgb[coat]
        coat_b = coat_rgb.mean(axis=1)
        lo, hi = np.percentile(coat_b, [14, 48])
        warm_pixels = (
            (coat_rgb[:, 0] > coat_rgb[:, 1] + 10.0)
            & (coat_rgb[:, 1] > coat_rgb[:, 2] + 6.0)
            & (coat_b >= lo)
            & (coat_b <= hi)
        )
        if int(warm_pixels.sum()) >= max(5, int(coat_rgb.shape[0] * 0.035)):
            sample = np.median(coat_rgb[warm_pixels], axis=0)
            warm_ids = _palette_ids_by_name(
                palette, candidate_ids,
                ["Medium Nougat", "Dark Tan", "Dark Orange", "Reddish Brown", "Brown"],
            )
            pid, dist = _nearest_palette_id_exact(sample, palette, warm_ids)
            # If LEGO has no close warm equivalent, painting this as a solid
            # patch invents markings from normal fur shading. Keep the sampled
            # HEX in logs, but only paint when the match is genuinely close.
            if pid is not None and int(pid) != int(base_id) and float(dist) <= 170.0:
                meta = _sample_meta(sample, palette, int(pid), dist, "photo-hex-warm-secondary")
                meta["coverage_fraction"] = round(float(warm_pixels.mean()), 3)
                profile["warm_secondary"] = meta
    return profile


def _sample_meta(
    sample_rgb: np.ndarray,
    palette: list[dict],
    pid: int,
    dist: float,
    source: str,
) -> dict:
    return {
        "source": source,
        "sample_rgb": [int(round(float(v))) for v in np.asarray(sample_rgb)],
        "sample_hex": _rgb_hex(sample_rgb),
        "matched_id": int(pid),
        "matched_name": _palette_name(palette, int(pid)),
        "distance": round(float(dist), 2),
    }


def _dark_marking_sample(rgb_pixels: np.ndarray) -> np.ndarray:
    """Sample the visual core of a dark marking, not its fuzzy antialias edge."""
    pixels = np.asarray(rgb_pixels, dtype=np.float32).reshape(-1, 3)
    if len(pixels) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if len(pixels) < 8:
        return np.median(pixels, axis=0)
    brightness = pixels.mean(axis=1)
    cutoff = float(np.percentile(brightness, 38))
    core = pixels[brightness <= cutoff]
    if len(core) < 4:
        core = pixels
    return np.median(core, axis=0)


def _palette_ids_by_name(
    palette: list[dict],
    candidate_ids: list[int],
    names: list[str],
) -> list[int]:
    by_name = {p["name"].lower(): int(p["id"]) for p in palette}
    out: list[int] = []
    for name in names:
        pid = by_name.get(name.lower())
        if pid is not None and pid in candidate_ids and pid not in out:
            out.append(pid)
    return out


def _warm_adjusted_base_palette_id(
    small_rgb: np.ndarray,
    small_alpha: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
    base_id: int,
) -> tuple[int, dict]:
    """Avoid flattening orange/nougat pets into LEGO Tan when the photo is warm."""
    base_name = _palette_name(palette, int(base_id))
    if base_name not in {"Tan", "Cream"} or not small_alpha.any():
        return int(base_id), {"changed": False, "reason": "base not warm-adjustable"}

    fg_rgb = small_rgb[small_alpha].astype(np.float32)
    brightness = fg_rgb.mean(axis=1)
    # Use the middle coat, not bright white chest/paw highlights and not dark
    # pattern markings.
    lo, hi = np.percentile(brightness, [35, 72])
    coat_pixels = fg_rgb[(brightness >= lo) & (brightness <= hi)]
    if len(coat_pixels) < 8:
        coat_pixels = fg_rgb
    rgb = np.median(coat_pixels, axis=0)
    r, g, b = [float(v) for v in rgb]
    warm_score = min(r - g, g - b)
    if r < 145.0 or warm_score < 14.0:
        return int(base_id), {
            "changed": False,
            "reason": "not warm enough",
            "rgb": [round(r, 1), round(g, 1), round(b, 1)],
            "warm_score": round(float(warm_score), 1),
        }

    by_name = {p["name"].lower(): int(p["id"]) for p in palette}
    preferred_name = "Medium Nougat" if float(rgb.mean()) < 178.0 else "Tan"
    preferred_id = by_name.get(preferred_name.lower())
    if preferred_id is None or preferred_id not in candidate_ids:
        return int(base_id), {"changed": False, "reason": "warm palette unavailable"}
    if int(preferred_id) == int(base_id):
        return int(base_id), {
            "changed": False,
            "reason": "already best warm base",
            "rgb": [round(r, 1), round(g, 1), round(b, 1)],
        }
    return int(preferred_id), {
        "changed": True,
        "from": base_name,
        "to": preferred_name,
        "rgb": [round(r, 1), round(g, 1), round(b, 1)],
        "warm_score": round(float(warm_score), 1),
    }


def _apply_light_coat_overlay(
    labels: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
    base_id: int,
    *,
    preferred_id: int | None = None,
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
        pid = int(preferred_id) if preferred_id in light_ids else None
        if pid is None:
            pid = _nearest_palette_id(np.median(rgb[comp], axis=0), palette, light_ids)
        if pid is None or int(pid) == int(base_id):
            continue
        out[comp] = int(pid)
        kept += 1
        cells += size
        used.add(int(pid))
    return out, {
        "components": kept,
        "cells": cells,
        "used_ids": sorted(used),
        "preferred_id": int(preferred_id) if preferred_id is not None else None,
    }


def _apply_anatomy_light_overlay(
    labels: np.ndarray,
    rgb: np.ndarray,
    alpha_mask: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
    base_id: int,
    *,
    preferred_id: int | None = None,
    front_light_profile: dict | None = None,
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """Propagate light coat colors through plausible pet anatomy zones.

    The raw light-color mask is often fragmented by fur, shadows, and voxel
    downsampling. This pass still requires photo evidence, but it expands that
    evidence inside specific zones such as chest, belly, paws, lower legs, and
    muzzle instead of painting arbitrary rectangles.
    """
    out = labels.copy()
    if not alpha_mask.any():
        return out, {"applied": False, "reason": "empty alpha"}, None
    light_ids = _palette_ids_by_name(palette, candidate_ids, ["White", "Cream"])
    if not light_ids:
        return out, {"applied": False, "reason": "no light palette"}, None
    pid = int(preferred_id) if preferred_id in light_ids else int(light_ids[0])
    if int(pid) == int(base_id) and len(light_ids) > 1:
        pid = int(light_ids[1])

    zones, zone_meta, debug = _pet_anatomy_zones(alpha_mask)
    if not zones:
        return out, {"applied": False, "reason": "no anatomy zones", **zone_meta}, debug

    rgb_f = rgb.astype(np.float32)
    spread = rgb_f.max(axis=2) - rgb_f.min(axis=2)
    brightness = rgb_f.mean(axis=2)
    light = (
        alpha_mask
        & (rgb_f.min(axis=2) >= 198.0)
        & (spread <= 68.0)
        & (brightness >= 205.0)
    )
    try:
        from scipy.ndimage import binary_closing, binary_dilation
        clean_light = binary_closing(light, iterations=1) & alpha_mask
        grown_light = binary_dilation(clean_light, iterations=1) & alpha_mask
    except Exception:
        clean_light = light
        grown_light = light

    config = {
        "muzzle": (0.020, 0.12, 1),
        "chest": (0.035, 0.14, 2),
        "belly": (0.035, 0.16, 2),
        "front_leg": (0.035, 0.16, 2),
        "rear_leg": (0.035, 0.16, 2),
        "front_paw": (0.020, 0.08, 1),
        "rear_paw": (0.020, 0.08, 1),
    }
    painted: dict[str, int] = {}
    evidence: dict[str, dict] = {}
    visible_side_only_mask = np.zeros(labels.shape, dtype=bool)
    total_cells = 0
    for name, (min_frac, fill_frac, grow_iters) in config.items():
        zone = zones.get(name)
        if zone is None or not zone.any():
            continue
        zone_cells = int(zone.sum())
        hits = int((clean_light & zone).sum())
        hit_frac = hits / max(zone_cells, 1)
        front_support = _front_profile_supports_zone(front_light_profile, name)
        effective_min_frac = min_frac * (0.35 if front_support else 1.0)
        effective_fill_frac = fill_frac * (0.65 if front_support else 1.0)
        evidence[name] = {
            "zone_cells": zone_cells,
            "light_cells": hits,
            "light_fraction": round(float(hit_frac), 3),
            "front_photo_support": bool(front_support),
        }
        if hits < max(1, int(round(zone_cells * effective_min_frac))):
            continue
        if hit_frac >= effective_fill_frac:
            paint = zone & alpha_mask
        else:
            paint = clean_light & zone
            try:
                from scipy.ndimage import binary_dilation
                paint = binary_dilation(paint, iterations=int(grow_iters)) & zone & alpha_mask
            except Exception:
                paint = grown_light & zone
        cells = int(paint.sum())
        if cells <= 0:
            continue
        out[paint] = int(pid)
        # Side-photo-only light evidence should not be projected onto the
        # unphotographed lateral surface. Front-photo-supported zones are safer
        # to keep because they describe centerline features such as chest,
        # muzzle, and front socks.
        if not front_support:
            visible_side_only_mask |= paint
        painted[name] = cells
        total_cells += cells
        if debug is not None:
            debug[paint] = 255

    return out, {
        "applied": bool(total_cells),
        "used_id": int(pid) if total_cells else None,
        "used_name": _palette_name(palette, int(pid)) if total_cells else None,
        "painted": painted,
        "cells": int(total_cells),
        "evidence": evidence,
        "front_photo_used": bool(front_light_profile and front_light_profile.get("applied")),
        "_visible_side_only_mask": visible_side_only_mask,
        **zone_meta,
    }, debug


def _front_profile_supports_zone(profile: dict | None, zone_name: str) -> bool:
    if not profile or not profile.get("applied"):
        return False
    if zone_name in {"muzzle"}:
        return bool(profile.get("muzzle_light"))
    if zone_name in {"chest", "front_leg"}:
        return bool(profile.get("chest_light") or profile.get("front_leg_light"))
    if zone_name in {"front_paw", "rear_paw"}:
        return bool(profile.get("paw_light"))
    return False


def _front_photo_light_profile(front_photo_path: str | Path | None) -> dict:
    if not front_photo_path:
        return {"applied": False, "reason": "no front photo"}
    try:
        img = Image.open(front_photo_path).convert("RGBA")
    except Exception as e:
        return {"applied": False, "reason": f"open failed: {e}"}
    rgba = np.asarray(img)
    alpha = rgba[..., 3] > 16
    if not alpha.any():
        return {"applied": False, "reason": "no foreground"}
    rows = np.any(alpha, axis=1)
    cols = np.any(alpha, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - np.argmax(cols[::-1]))
    crop = rgba[y0:y1, x0:x1]
    if crop.size == 0:
        return {"applied": False, "reason": "empty crop"}
    fg = crop[..., 3] > 16
    rgb = crop[..., :3].astype(np.float32)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    brightness = rgb.mean(axis=2)
    light = fg & (rgb.min(axis=2) >= 198.0) & (spread <= 72.0) & (brightness >= 205.0)
    H, W = fg.shape

    def region_fraction(xa: float, xb: float, ya: float, yb: float) -> float:
        xs0, xs1 = int(round(W * xa)), int(round(W * xb))
        ys0, ys1 = int(round(H * ya)), int(round(H * yb))
        region = fg[max(0, ys0):min(H, ys1), max(0, xs0):min(W, xs1)]
        if region.size == 0 or not region.any():
            return 0.0
        hits = light[max(0, ys0):min(H, ys1), max(0, xs0):min(W, xs1)]
        return float(hits.sum() / max(int(region.sum()), 1))

    sample = np.median(rgb[light], axis=0) if int(light.sum()) >= 4 else None
    out = {
        "applied": True,
        "foreground_bbox": [x0, y0, x1, y1],
        "light_fraction": round(float(light.sum() / max(int(fg.sum()), 1)), 3),
        "muzzle_fraction": round(region_fraction(0.32, 0.68, 0.18, 0.48), 3),
        "chest_fraction": round(region_fraction(0.28, 0.72, 0.38, 0.78), 3),
        "front_leg_fraction": round(region_fraction(0.25, 0.75, 0.58, 0.92), 3),
        "paw_fraction": round(region_fraction(0.20, 0.80, 0.78, 1.00), 3),
    }
    out["muzzle_light"] = out["muzzle_fraction"] >= 0.10
    out["chest_light"] = out["chest_fraction"] >= 0.16
    out["front_leg_light"] = out["front_leg_fraction"] >= 0.14
    out["paw_light"] = out["paw_fraction"] >= 0.12
    if sample is not None:
        out["sample_hex"] = _rgb_hex(sample)
    return out


def _pet_anatomy_zones(alpha_mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict, np.ndarray | None]:
    """Derive coarse side-profile pet anatomy zones from the subject silhouette."""
    if alpha_mask.size == 0 or not alpha_mask.any():
        return {}, {"head_side": None, "reason": "empty alpha"}, None
    H, W = alpha_mask.shape
    head_side, layout = _infer_side_profile_head_side(alpha_mask)
    if head_side not in {"left", "right"}:
        head_side = "right"
        layout = {**layout, "fallback_head_side": True}

    head_w = max(3, int(round(W * 0.26)))
    tail_w = max(2, int(round(W * 0.18)))
    neck_w = max(2, int(round(W * 0.15)))
    leg_w = max(2, int(round(W * 0.16)))

    cols = np.arange(W)
    if head_side == "right":
        head_cols = cols >= W - head_w
        chest_cols = (cols >= max(0, W - head_w - neck_w)) & (cols < W)
        torso_cols = (cols >= tail_w) & (cols < W - max(2, int(head_w * 0.55)))
        front_leg_cols = (cols >= max(0, W - head_w - leg_w)) & (cols < W - max(1, int(head_w * 0.25)))
        rear_leg_cols = (cols >= tail_w) & (cols < min(W, tail_w + leg_w))
    else:
        head_cols = cols < head_w
        chest_cols = cols < min(W, head_w + neck_w)
        torso_cols = (cols >= max(2, int(head_w * 0.55))) & (cols < W - tail_w)
        front_leg_cols = (cols >= max(1, int(head_w * 0.25))) & (cols < min(W, head_w + leg_w))
        rear_leg_cols = (cols >= max(0, W - tail_w - leg_w)) & (cols < W - tail_w)

    top_frac = _column_fraction_mask(alpha_mask, 0.00, 0.34)
    mid_frac = _column_fraction_mask(alpha_mask, 0.30, 0.68)
    lower_frac = _column_fraction_mask(alpha_mask, 0.55, 1.00)
    paw_frac = _column_fraction_mask(alpha_mask, 0.78, 1.00)
    belly_frac = _column_fraction_mask(alpha_mask, 0.55, 0.82)

    def colmask(c: np.ndarray) -> np.ndarray:
        return np.broadcast_to(c.reshape(1, W), (H, W)) & alpha_mask

    zones = {
        "head": colmask(head_cols),
        "muzzle": colmask(head_cols) & mid_frac,
        "ears": colmask(head_cols) & top_frac,
        "chest": colmask(chest_cols) & lower_frac,
        "belly": colmask(torso_cols) & belly_frac,
        "front_leg": colmask(front_leg_cols) & lower_frac,
        "rear_leg": colmask(rear_leg_cols) & lower_frac,
        "front_paw": colmask(front_leg_cols) & paw_frac,
        "rear_paw": colmask(rear_leg_cols) & paw_frac,
        "torso": colmask(torso_cols),
    }

    debug = np.zeros((H, W), dtype=np.uint8)
    debug[zones["torso"]] = 45
    debug[zones["head"]] = 85
    debug[zones["chest"]] = 125
    debug[zones["belly"]] = 165
    debug[zones["front_leg"] | zones["rear_leg"]] = 205
    debug[zones["front_paw"] | zones["rear_paw"] | zones["muzzle"]] = 235
    return zones, {
        "head_side": head_side,
        "zone_cells": {k: int(v.sum()) for k, v in zones.items()},
        **layout,
    }, debug


def _column_fraction_mask(alpha_mask: np.ndarray, start: float, end: float) -> np.ndarray:
    H, W = alpha_mask.shape
    out = np.zeros_like(alpha_mask, dtype=bool)
    start = float(np.clip(start, 0.0, 1.0))
    end = float(np.clip(end, start, 1.0))
    for x in range(W):
        ys = np.where(alpha_mask[:, x])[0]
        if ys.size == 0:
            continue
        top = int(ys.min())
        bottom = int(ys.max())
        height = max(1, bottom - top + 1)
        y0 = int(round(top + start * (height - 1)))
        y1 = int(round(top + end * (height - 1))) + 1
        out[max(0, y0):min(H, y1), x] = True
    return out & alpha_mask


def _apply_warm_coat_overlay(
    labels: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
    base_id: int,
    *,
    preferred_id: int | None = None,
    marking_mask: np.ndarray | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Add broad warm fur shadow blocks without turning fur texture noisy."""
    out = labels.copy()
    if preferred_id is None or int(preferred_id) == int(base_id):
        return out, {"components": 0, "reason": "no secondary warm color"}
    if int(preferred_id) not in set(int(v) for v in candidate_ids):
        return out, {"components": 0, "reason": "secondary color not allowed"}
    if not mask.any():
        return out, {"components": 0, "reason": "empty mask"}
    try:
        from scipy.ndimage import binary_closing, binary_opening, label as cc_label
    except Exception:
        return out, {"components": 0, "reason": "scipy unavailable"}

    rgb_f = rgb.astype(np.float32)
    brightness = rgb_f.mean(axis=2)
    fg = mask & (labels == int(base_id))
    if exclusion_mask is not None and exclusion_mask.shape == fg.shape:
        fg &= ~exclusion_mask
    if marking_mask is not None and marking_mask.shape == fg.shape:
        try:
            from scipy.ndimage import binary_dilation
            fg &= ~binary_dilation(marking_mask, iterations=1)
        except Exception:
            fg &= ~marking_mask
    if int(fg.sum()) < 24:
        return out, {"components": 0, "reason": "too little coat"}

    vals = brightness[fg]
    lo, hi = np.percentile(vals, [10, 42])
    warm = (
        (rgb_f[..., 0] > rgb_f[..., 1] + 10.0)
        & (rgb_f[..., 1] > rgb_f[..., 2] + 6.0)
    )
    region = fg & warm & (brightness >= lo) & (brightness <= hi)
    # Keep this as a designed LEGO color block, not a noisy fur texture.
    if int(region.sum()) > int(fg.sum() * 0.28):
        tighter = np.percentile(vals, 30)
        region = region & (brightness <= tighter)
    if int(region.sum()) < max(4, int(fg.sum() * 0.018)):
        return out, {
            "components": 0,
            "reason": "secondary warm region too small",
            "candidate_id": int(preferred_id),
        }
    region = binary_opening(region, iterations=1) & fg
    region = binary_closing(region, iterations=1) & fg
    comps, n = cc_label(region)
    min_size = max(3, int(fg.sum() * 0.006))
    max_size = max(min_size + 1, int(fg.sum() * 0.26))
    kept = 0
    cells = 0
    boxes: list[list[int]] = []
    for cid in range(1, n + 1):
        comp = comps == cid
        size = int(comp.sum())
        if size < min_size or size > max_size:
            continue
        out[comp] = int(preferred_id)
        kept += 1
        cells += size
        ys, xs = np.where(comp)
        if xs.size:
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    return out, {
        "components": kept,
        "cells": cells,
        "used_ids": [int(preferred_id)] if kept else [],
        "boxes": boxes[:12],
        "preferred_id": int(preferred_id),
        "preferred_name": _palette_name(palette, int(preferred_id)),
    }


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


def _face_feature_exclusion_mask(
    gpt_data: dict | None,
    *,
    image_size: tuple[int, int],
    crop_bbox: tuple[int, int, int, int],
    small_shape: tuple[int, int],
) -> np.ndarray | None:
    """Remove body-photo eyes/nose/mouth from dark marking extraction."""
    if not gpt_data:
        return None
    Hs, Ws = small_shape
    W, H = image_size
    cx0, cy0, cx1, cy1 = crop_bbox
    crop_w = max(1, cx1 - cx0)
    crop_h = max(1, cy1 - cy0)
    mask = np.zeros((Hs, Ws), dtype=bool)
    keys = {"eye", "eyes", "eye_socket", "pupil", "nose", "mouth"}
    for r in gpt_data.get("regions", []) or []:
        name = (r.get("name") or "").strip().lower().replace("-", "_")
        if name not in keys:
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


def _filter_side_profile_markings(
    marking_mask: np.ndarray,
    alpha_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """For side-body photos, keep body markings but suppress face/ear noise.

    The front portrait owns the face. The side photo owns body markings, so
    dark side-photo pixels around the visible eye/ear should not become a fake
    coat marking. Tail darkness is allowed mainly at the distal tip.
    """
    if marking_mask.size == 0 or not marking_mask.any() or not alpha_mask.any():
        return marking_mask, {"applied": False, "reason": "empty mask"}
    H, W = marking_mask.shape
    head_side, layout = _infer_side_profile_head_side(alpha_mask)
    if head_side not in {"left", "right"}:
        return marking_mask, {"applied": False, "reason": "head side unknown"}

    head_w = max(3, int(round(W * 0.30)))
    tail_w = max(3, int(round(W * 0.22)))
    tail_tip_w = max(2, int(round(W * 0.13)))
    head_mask = np.zeros_like(marking_mask, dtype=bool)
    tail_zone = np.zeros_like(marking_mask, dtype=bool)
    tail_tip = np.zeros_like(marking_mask, dtype=bool)
    if head_side == "right":
        head_mask[:, W - head_w:] = True
        tail_zone[:, :tail_w] = True
        tail_tip[:, :tail_tip_w] = True
    else:
        head_mask[:, :head_w] = True
        tail_zone[:, W - tail_w:] = True
        tail_tip[:, W - tail_tip_w:] = True

    out = marking_mask & ~head_mask
    removed_head = int(marking_mask.sum() - out.sum())
    removed_tail = 0
    kept_tail = 0
    try:
        from scipy.ndimage import label as cc_label
        comps, n = cc_label(out)
    except Exception:
        return out, {
            "applied": True,
            "head_side": head_side,
            "removed_head_cells": removed_head,
            "tail_filter": "skipped-no-scipy",
            **layout,
        }

    filtered = np.zeros_like(out, dtype=bool)
    kept_body = 0
    kept_tail = 0
    removed_noise = 0
    for cid in range(1, n + 1):
        comp = comps == cid
        size = int(comp.sum())
        if size <= 0:
            continue
        ys, xs = np.where(comp)
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        touches_top_outline = y0 <= max(1, int(round(H * 0.04))) and size <= max(3, int(alpha_mask.sum() * 0.01))
        touches_bottom_outline = y1 >= H - max(1, int(round(H * 0.03))) and size <= max(3, int(alpha_mask.sum() * 0.008))
        if touches_top_outline or touches_bottom_outline:
            removed_noise += size
            continue
        if (comp & tail_zone).any():
            if (comp & tail_tip).any():
                tail_piece = comp & tail_tip
                if int(tail_piece.sum()) >= 1:
                    filtered |= tail_piece
                    kept_tail += int(tail_piece.sum())
                removed_tail += size - int(tail_piece.sum())
                continue
            removed_tail += size
            continue
        filtered |= comp
        kept_body += size
    return filtered, {
        "applied": True,
        "head_side": head_side,
        "removed_head_cells": removed_head,
        "removed_tail_cells": removed_tail,
        "kept_tail_tip_cells": kept_tail,
        "kept_body_cells": kept_body,
        "removed_noise_cells": removed_noise,
        "head_columns": head_w,
        "tail_columns": tail_w,
        "tail_tip_columns": tail_tip_w,
        **layout,
    }


def _infer_side_profile_head_side(alpha_mask: np.ndarray) -> tuple[str | None, dict]:
    H, W = alpha_mask.shape
    seg = max(2, int(round(W * 0.25)))

    def score(side: str) -> dict:
        if side == "left":
            cols = range(0, seg)
        else:
            cols = range(W - seg, W)
        area = 0
        tops: list[int] = []
        heights: list[int] = []
        for x in cols:
            ys = np.where(alpha_mask[:, x])[0]
            if ys.size == 0:
                continue
            area += int(ys.size)
            tops.append(int(ys.min()))
            heights.append(int(ys.max() - ys.min() + 1))
        if not tops:
            return {"score": 0.0, "area": 0, "top": float(H), "height": 0.0}
        top_mean = float(np.mean(tops))
        height_mean = float(np.mean(heights))
        # Head side tends to be taller and reaches higher because of ears.
        s = float(area) + 8.0 * (H - top_mean) + 5.0 * height_mean
        return {
            "score": round(s, 2),
            "area": int(area),
            "top": round(top_mean, 2),
            "height": round(height_mean, 2),
        }

    left = score("left")
    right = score("right")
    if max(left["score"], right["score"]) <= 0:
        return None, {"left": left, "right": right}
    side = "right" if right["score"] > left["score"] else "left"
    confidence = abs(right["score"] - left["score"]) / max(right["score"], left["score"], 1.0)
    if confidence < 0.12:
        return None, {"left": left, "right": right, "confidence": round(confidence, 3)}
    return side, {
        "left": left,
        "right": right,
        "confidence": round(float(confidence), 3),
    }


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


def _extract_high_res_marking_mask(rgba: np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect real dark coat markings before the photo is blurred/downsampled."""
    if rgba.size == 0 or rgba.ndim != 3 or rgba.shape[2] < 4:
        return np.zeros(rgba.shape[:2], dtype=bool), {"components": 0, "reason": "bad image"}
    fg = rgba[..., 3] > 16
    if int(fg.sum()) < 32:
        return np.zeros(rgba.shape[:2], dtype=bool), {"components": 0, "reason": "too little foreground"}

    try:
        from scipy.ndimage import binary_closing, binary_opening, label as cc_label
    except Exception:
        binary_closing = binary_opening = None
        cc_label = None

    rgb = rgba[..., :3].astype(np.float32)
    # Light blur removes individual fur strands without destroying larger
    # tattoo-like coat marks such as crescents and dots.
    try:
        blurred = Image.fromarray(rgba[..., :3].astype(np.uint8), "RGB").filter(
            ImageFilter.GaussianBlur(radius=1.0)
        )
        rgb = np.asarray(blurred).astype(np.float32)
    except Exception:
        pass
    brightness = rgb.mean(axis=2)
    fg_vals = brightness[fg]
    median = float(np.median(fg_vals))
    threshold = float(min(median - 45.0, 145.0))
    if threshold < 20.0:
        return np.zeros(rgba.shape[:2], dtype=bool), {
            "components": 0,
            "reason": "low contrast",
            "threshold": round(threshold, 2),
        }

    dark = fg & (brightness <= threshold)
    if binary_opening is not None:
        dark = binary_opening(dark, iterations=1) & fg
        dark = binary_closing(dark, iterations=1) & fg
    if int(dark.sum()) < 8:
        return np.zeros(rgba.shape[:2], dtype=bool), {
            "components": 0,
            "threshold": round(threshold, 2),
            "reason": "too few dark pixels",
        }

    if cc_label is None:
        return dark, {
            "components": 1,
            "pixels": int(dark.sum()),
            "threshold": round(threshold, 2),
            "fallback": True,
        }

    comps, n = cc_label(dark)
    fg_area = int(fg.sum())
    min_size = max(12, int(fg_area * 0.00035))
    max_size = max(min_size + 1, int(fg_area * 0.24))
    out = np.zeros_like(dark, dtype=bool)
    kept = 0
    pixels = 0
    boxes: list[list[int]] = []
    for cid in range(1, n + 1):
        comp = comps == cid
        size = int(comp.sum())
        if size < min_size or size > max_size:
            continue
        comp_brightness = float(np.median(brightness[comp]))
        if median - comp_brightness < 35.0:
            continue
        ys, xs = np.where(comp)
        if xs.size == 0:
            continue
        out |= comp
        kept += 1
        pixels += size
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])

    return out, {
        "components": kept,
        "pixels": pixels,
        "threshold": round(threshold, 2),
        "boxes": boxes[:12],
        "source_shape": [int(rgba.shape[1]), int(rgba.shape[0])],
    }


def _resize_marking_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    H, W = int(shape[0]), int(shape[1])
    if H <= 0 or W <= 0:
        return np.zeros((max(0, H), max(0, W)), dtype=bool)
    if mask.size == 0 or not mask.any():
        return np.zeros((H, W), dtype=bool)
    img = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    coverage = np.asarray(img.resize((W, H), Image.Resampling.BOX), dtype=np.float32) / 255.0
    out = coverage >= 0.14
    # Preserve very small dots that can disappear between stud centers.
    if not out.any() and coverage.max() > 0:
        out = coverage >= max(0.04, float(coverage.max()) * 0.65)
    return out


def _apply_high_res_marking_overlay(
    labels: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
    *,
    marking_mask: np.ndarray,
    preferred_id: int | None = None,
) -> tuple[np.ndarray, dict]:
    out = labels.copy()
    fg = mask & (labels > 0)
    if int(fg.sum()) < 12 or marking_mask.size == 0 or not marking_mask.any():
        return out, {"components": 0, "reason": "no high-res marking mask"}
    marking_ids = _marking_palette_ids(palette, candidate_ids)
    if not marking_ids:
        return out, {"components": 0, "reason": "no marking palette colors"}
    preferred_id = int(preferred_id) if preferred_id in marking_ids else None
    try:
        from scipy.ndimage import label as cc_label
        clean = marking_mask & fg
        comps, n = cc_label(clean)
    except Exception:
        clean = marking_mask & fg
        comps, n = clean.astype(np.int32), 1
    min_size = max(1, int(fg.sum() * 0.0012))
    max_size = max(min_size + 1, int(fg.sum() * 0.30))
    kept = 0
    cells = 0
    used_ids: set[int] = set()
    boxes: list[list[int]] = []
    for cid in range(1, n + 1):
        comp = comps == cid
        size = int(comp.sum())
        if size < min_size or size > max_size:
            continue
        pid = _nearest_palette_id(_dark_marking_sample(rgb[comp]), palette, marking_ids)
        if preferred_id is not None and size >= max(2, int(min_size * 1.5)):
            # Larger crescent/spot components should use one consistent sampled
            # marking color; tiny antialiased cells can still choose nearest.
            pid = preferred_id
        if pid is None:
            continue
        out[comp] = int(pid)
        kept += 1
        cells += size
        used_ids.add(int(pid))
        ys, xs = np.where(comp)
        if xs.size:
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    return out, {
        "components": kept,
        "cells": cells,
        "used_ids": sorted(used_ids),
        "boxes": boxes[:12],
        "preferred_id": int(preferred_id) if preferred_id is not None else None,
    }


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


def _nearest_palette_id_exact(
    rgb: np.ndarray,
    palette: list[dict],
    candidate_ids: list[int],
) -> tuple[int | None, float]:
    """CIELAB nearest LEGO color with no GPT/availability bias."""
    entries = [p for p in palette if int(p["id"]) in candidate_ids]
    if not entries:
        return None, float("inf")
    pal_ids = np.array([int(p["id"]) for p in entries], dtype=np.int32)
    pal_rgb = np.array([p["rgb"] for p in entries], dtype=np.float32) / 255.0
    pal_lab = _rgb_to_lab(pal_rgb)
    lab = _rgb_to_lab(np.asarray(rgb, dtype=np.float32).reshape(1, 3) / 255.0)[0]
    d = ((pal_lab - lab) ** 2).sum(axis=1)
    idx = int(d.argmin())
    return int(pal_ids[idx]), float(d[idx])


def _rgb_hex(rgb: np.ndarray) -> str:
    vals = np.clip(np.asarray(rgb, dtype=np.float32), 0, 255).round().astype(int)
    return "#{:02X}{:02X}{:02X}".format(int(vals[0]), int(vals[1]), int(vals[2]))


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


def _save_marking_debug(mask: np.ndarray, out_dir: str | Path, stem: str) -> Path:
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    img[mask] = (20, 24, 28)
    img[~mask] = (245, 245, 240)
    out = Image.fromarray(img, "RGB").resize(
        (max(96, mask.shape[1] * 12), max(96, mask.shape[0] * 12)),
        Image.Resampling.NEAREST,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_pet_marking_mask.png"
    out.save(path)
    return path


def _save_anatomy_debug(debug: np.ndarray, out_dir: str | Path, stem: str) -> Path:
    debug = np.asarray(debug, dtype=np.uint8)
    img = np.zeros((*debug.shape, 3), dtype=np.uint8)
    img[debug == 0] = (245, 245, 240)
    img[(debug > 0) & (debug < 100)] = (135, 145, 170)
    img[(debug >= 100) & (debug < 155)] = (180, 150, 95)
    img[(debug >= 155) & (debug < 215)] = (120, 170, 150)
    img[debug >= 215] = (235, 235, 245)
    out = Image.fromarray(img, "RGB").resize(
        (max(96, debug.shape[1] * 12), max(96, debug.shape[0] * 12)),
        Image.Resampling.NEAREST,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_pet_anatomy_map.png"
    out.save(path)
    return path
