"""
Project a segmented photo onto a voxel grid.

This REPLACES the per-voxel texture-sampling step when semantic mode is on:
  1. Segment the photo into N regions (see segment_photo).
  2. Snap each region's dominant color to the nearest LEGO palette entry
     (CIELAB, with availability bias).
  3. For each occupied voxel, compute its UV mapping (front normal → photo
     coord), look up the region ID, set the voxel's color to the region's
     LEGO color.
  4. Voxels facing backward inherit the mirror-X region color.
  5. Side-facing voxels read the photo's edge column.

The output voxel `colors` array thus has only N distinct RGB values (one per
region) — quantization downstream becomes basically a no-op for these voxels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from segment_photo import segment_photo
from voxels_to_palette import _rgb_to_lab, AVAILABILITY_PENALTY


def _palette_id_for_name(palette: list[dict], name: str | None) -> int | None:
    if not name:
        return None
    key = name.strip().lower()
    for p in palette:
        if p["name"].lower() == key:
            return int(p["id"])
    return None


def gpt_to_restrict_ids(gpt_data: dict, palette: list[dict]) -> list[int]:
    """Turn GPT's recommended_lego_palette + region color_names into a set of
    palette IDs that the downstream quantizer is allowed to use."""
    out: set[int] = set()
    for name in gpt_data.get("recommended_lego_palette", []) or []:
        pid = _palette_id_for_name(palette, name)
        if pid is not None:
            out.add(pid)
    for r in gpt_data.get("regions", []) or []:
        pid = _palette_id_for_name(palette, r.get("color_name"))
        if pid is not None:
            out.add(pid)
    # Black for eyes is always allowed
    out.add(4)
    return sorted(out)


ACCENT_ONLY_NAMES = {
    "eye", "eyes", "eye_left", "eye_right",
    "eye_socket", "eye_socket_left", "eye_socket_right",
    "pupil", "pupils",
}

BROAD_SURFACE_NAMES = {
    "body", "head", "face", "back", "chest", "belly", "neck",
    "ears", "legs", "tail", "wing", "hull", "wall", "roof",
}

SMALL_DETAIL_NAMES = {
    "ear_interior", "muzzle", "cheek", "snout", "nose", "mouth",
    "paw", "paws", "paw_pad", "paw_pads", "tongue", "marking", "patch",
    "belly_patch", "stripe", "spot", "label", "window", "door", "trim",
    "wheel", "cap",
}


def _name_key(name: str | None) -> str:
    return (name or "").strip().lower().replace("-", "_").replace(" ", "_")


def _should_paint_region(
    name: str | None,
    area_frac: float,
    paint_mode: str,
    *,
    max_small_area: float = 0.065,
) -> tuple[bool, str | None]:
    """Decide whether a semantic region should overwrite voxel colors."""
    mode = (paint_mode or "all").strip().lower()
    key = _name_key(name)
    if key in ACCENT_ONLY_NAMES:
        return False, "accent-only"
    if mode in {"none", "palette_only", "palette-only"}:
        return False, "palette-only"
    if mode in {"all", "full"}:
        return True, None
    if mode in {"small", "details", "detail_only", "detail-only"}:
        if key in SMALL_DETAIL_NAMES:
            return True, None
        if key in BROAD_SURFACE_NAMES:
            return False, "broad-surface"
        if area_frac <= max_small_area:
            return True, None
        return False, "large-region"
    return True, None


def _build_gpt_label_map(
    gpt_data: dict,
    H: int,
    W: int,
    paint_mode: str = "all",
) -> tuple["np.ndarray", dict, list[str]]:
    """Build a per-pixel label_map from GPT region bboxes.

    Smaller-area regions paint LAST so inner regions (e.g. belly inside body)
    overwrite outer ones. Returns (label_map of shape (H, W), regions dict).
    """
    label_map = -np.ones((H, W), dtype=np.int32)
    regions_in = gpt_data.get("regions", []) or []
    sized = []
    for i, r in enumerate(regions_in):
        bb = r.get("bbox_normalized") or []
        if len(bb) != 4:
            continue
        x0, y0, x1, y1 = bb
        x0p = max(0, min(W - 1, int(round(x0 * W))))
        x1p = max(x0p + 1, min(W, int(round(x1 * W))))
        y0p = max(0, min(H - 1, int(round(y0 * H))))
        y1p = max(y0p + 1, min(H, int(round(y1 * H))))
        sized.append((i, r, (x0p, y0p, x1p, y1p), (x1p - x0p) * (y1p - y0p)))
    # Paint large boxes first, then small ones overwrite
    sized.sort(key=lambda t: -t[3])
    regions_out: dict[int, dict] = {}
    skipped: list[str] = []
    image_area = max(1, H * W)
    for rid, r, (x0p, y0p, x1p, y1p), area in sized:
        name = r.get("name")
        should_paint, reason = _should_paint_region(name, area / image_area, paint_mode)
        if not should_paint:
            skipped.append(f"{name or 'region'}:{reason}")
            continue
        label_map[y0p:y1p, x0p:x1p] = rid
        kind = "eye" if _name_key(r.get("name")) in ACCENT_ONLY_NAMES else (r.get("name") or "region")
        regions_out[rid] = {
            "kind": kind,
            "color_name": r.get("color_name"),
            "name": r.get("name"),
            "bbox": (x0p, y0p, x1p, y1p),
        }
    return label_map, regions_out, skipped


def _empty_neighbor(occ: np.ndarray, axis: int, direction: int) -> np.ndarray:
    shifted = np.roll(occ, shift=-direction, axis=axis)
    if axis == 0:
        if direction > 0:
            shifted[-1, :, :] = False
        else:
            shifted[0, :, :] = False
    elif axis == 1:
        if direction > 0:
            shifted[:, -1, :] = False
        else:
            shifted[:, 0, :] = False
    elif axis == 2:
        if direction > 0:
            shifted[:, :, -1] = False
        else:
            shifted[:, :, 0] = False
    return occ & ~shifted


def _foreground_bbox(label_map: np.ndarray) -> tuple[int, int, int, int] | None:
    fg = label_map >= 0
    if not fg.any():
        return None
    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - np.argmax(cols[::-1]))
    return x0, y0, x1, y1


def _label_at_or_nearest_in_row(label_map: np.ndarray, u_px: int, v_px: int) -> int | None:
    rid = int(label_map[v_px, u_px])
    if rid >= 0:
        return rid
    row = label_map[v_px]
    valid = np.where(row >= 0)[0]
    if len(valid) == 0:
        return None
    nearest = valid[np.argmin(np.abs(valid - u_px))]
    return int(row[nearest])


def _project_label_map_onto_voxels(
    grid,
    label_map: np.ndarray,
    region_to_rgb: dict[int, tuple[int, int, int]],
    front_axis: str = "-y",
) -> dict:
    """Project a per-pixel semantic label map onto occupied voxels.

    This is the shared hot path for GPT bbox, SAM mask, and k-means semantic
    color modes. It intentionally preserves the existing convention: front is
    voxel -Y, back-facing voxels mirror X so symmetric subjects inherit front
    colors on their backs.
    """
    bbox = _foreground_bbox(label_map)
    if bbox is None:
        return {"label_map_shape": label_map.shape, "label_map_empty": True}

    x0_bb, y0_bb, x1_bb, y1_bb = bbox
    bw, bh = max(1, x1_bb - x0_bb), max(1, y1_bb - y0_bb)
    H, W = label_map.shape
    occ = grid.occupancy
    sx, sy, sz = occ.shape
    axis_letter = (front_axis or "-y").strip().lower().lstrip("+-")
    sign = -1 if (front_axis or "-y").strip().startswith("-") else 1
    if axis_letter == "x":
        depth_axis, u_axis, u_size = 0, 1, sy
    else:
        # Current voxel convention is z-up, so y is the normal photo depth.
        depth_axis, u_axis, u_size = 1, 0, sx
    faces_front = _empty_neighbor(occ, depth_axis, sign)
    faces_back = _empty_neighbor(occ, depth_axis, -sign)

    projected = 0
    for (x, y, z) in np.argwhere(occ):
        coord = (x, y, z)
        u_coord = coord[u_axis]
        if faces_back[x, y, z] and not faces_front[x, y, z]:
            u_coord = u_size - 1 - u_coord
        u_in_bb = (u_coord + 0.5) / u_size
        v_in_bb = 1.0 - (z + 0.5) / sz
        u_px = int(np.clip(x0_bb + u_in_bb * bw, 0, W - 1))
        v_px = int(np.clip(y0_bb + v_in_bb * bh, 0, H - 1))

        rid = _label_at_or_nearest_in_row(label_map, u_px, v_px)
        if rid is None:
            continue
        target_rgb = region_to_rgb.get(rid)
        if target_rgb is None:
            continue
        grid.colors[x, y, z] = target_rgb
        projected += 1

    return {
        "label_map_shape": label_map.shape,
        "foreground_bbox": bbox,
        "projected_voxels": projected,
    }


def project_semantic_gpt(
    grid,
    photo_path: str | Path,
    palette: list[dict],
    gpt_data: dict,
    front_axis: str = "-y",
    paint_mode: str = "all",
) -> tuple[set[int], dict]:
    """GPT-driven version of project_semantic.

    Uses GPT's region bounding boxes (instead of k-means) and GPT's
    color_name (instead of CIELAB snapping) to color each voxel. Eye regions
    are forced to Black. Returns (used_lego_ids, region_meta).
    """
    from PIL import Image
    img = Image.open(photo_path)
    W, H = img.size
    label_map, regions, skipped = _build_gpt_label_map(
        gpt_data, H, W, paint_mode=paint_mode)
    if not regions:
        if skipped:
            print(f"[gpt-semantic] skipped surface paint: {skipped}")
        return set(), {
            "gpt": True,
            "paint_mode": paint_mode,
            "no_painted_regions": True,
            "skipped": skipped,
        }
    if skipped:
        print(f"[gpt-semantic] skipped surface paint: {skipped}")

    # Build region_id -> (palette_id, rgb)
    region_to_palette: dict[int, int] = {}
    region_to_rgb: dict[int, tuple[int, int, int]] = {}
    for rid, info in regions.items():
        if info["kind"] == "eye":
            pid = 4  # Black
        else:
            pid = _palette_id_for_name(palette, info.get("color_name"))
            if pid is None:
                # Fall back to neutral gray if GPT named a color we don't have
                pid = 2  # Light Bluish Gray
        region_to_palette[rid] = pid
        entry = next(p for p in palette if int(p["id"]) == pid)
        region_to_rgb[rid] = tuple(int(v) for v in entry["rgb"])

    used_ids = set(region_to_palette.values())
    projection_meta = _project_label_map_onto_voxels(
        grid, label_map, region_to_rgb, front_axis=front_axis)
    if projection_meta.get("label_map_empty"):
        return used_ids, {"label_map_shape": label_map.shape, "gpt": True}

    return used_ids, {
        "regions": len(regions),
        "lego_ids_used": sorted(used_ids),
        "region_to_palette": region_to_palette,
        "projection": projection_meta,
        "gpt": True,
        "paint_mode": paint_mode,
    }


def project_semantic(
    grid,
    photo_path: str | Path,
    palette: list[dict],
    n_regions: int = 8,
    front_axis: str = "-y",
) -> tuple[set[int], dict]:
    """Mutate `grid.colors` so every front/back/side voxel gets the region's
    LEGO-snapped color from the segmentation. Returns:
        used_lego_ids : set of LEGO palette IDs that landed on voxels
        region_meta   : dict of segmentation info (for downstream / debug)
    """
    label_map, regions, _ = segment_photo(photo_path, n_regions=n_regions)
    if not regions:
        return set(), {}

    # Snap each region's dominant RGB to a LEGO palette entry (CIELAB + bias)
    pal_rgb = np.array([p["rgb"] for p in palette], dtype=np.float32) / 255.0
    pal_lab = _rgb_to_lab(pal_rgb)
    pal_ids = np.array([p["id"] for p in palette], dtype=np.int32)
    pal_penalty = np.array([
        AVAILABILITY_PENALTY.get(p.get("availability", "uncommon"), 0.0)
        for p in palette
    ], dtype=np.float32) ** 2

    # For eye regions, force Black (palette id 4) so accents stay crisp.
    region_to_palette: dict[int, int] = {}
    region_to_rgb: dict[int, tuple[int, int, int]] = {}
    for rid, info in regions.items():
        if info["kind"] == "eye":
            region_to_palette[rid] = 4  # Black
        else:
            target_rgb = np.array(info["rgb"], dtype=np.float32) / 255.0
            target_lab = _rgb_to_lab(target_rgb[None])[0]
            dists = ((pal_lab - target_lab) ** 2).sum(axis=1) + pal_penalty
            region_to_palette[rid] = int(pal_ids[int(dists.argmin())])
        # Cache the palette RGB for direct voxel write-back
        pal_entry = next(p for p in palette if int(p["id"]) == region_to_palette[rid])
        region_to_rgb[rid] = tuple(int(v) for v in pal_entry["rgb"])

    used_ids = set(region_to_palette.values())
    projection_meta = _project_label_map_onto_voxels(
        grid, label_map, region_to_rgb, front_axis=front_axis)
    if projection_meta.get("label_map_empty"):
        return used_ids, {"label_map_shape": label_map.shape}

    return used_ids, {
        "regions": len(regions),
        "lego_ids_used": sorted(used_ids),
        "region_to_palette": region_to_palette,
        "projection": projection_meta,
    }


# ── SAM 2 variant ──────────────────────────────────────────────────────────

def _build_sam_label_map(
    masks: list,
    regions_in: list,
    H: int,
    W: int,
    paint_mode: str = "all",
) -> tuple["np.ndarray", dict, list[str]]:
    """Build a per-pixel label_map from SAM 2 masks. Smaller masks paint LAST
    so ears overlay onto head, but eye_socket / eyes are SKIPPED — those
    regions are accent-only (round_tile placed on top of body color), not
    surface paint. Painting eye_socket black would create giant black blobs
    when GPT's bbox is mis-placed.

    Returns (label_map of shape (H, W), regions dict keyed by region index).
    """
    label_map = -np.ones((H, W), dtype=np.int32)
    sized = []
    skipped = []
    image_area = max(1, H * W)
    for i, (r, mask) in enumerate(zip(regions_in, masks)):
        if mask is None:
            continue
        # Some masks may come in at a different shape if processor was upset
        if mask.shape != (H, W):
            try:
                from PIL import Image
                m = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
                    (W, H), Image.NEAREST)
                mask = (np.array(m) > 0).astype(np.uint8)
            except Exception:
                continue
        area = int(mask.sum())
        if area <= 0:
            continue
        name = r.get("name") or ""
        should_paint, reason = _should_paint_region(
            name, area / image_area, paint_mode)
        if not should_paint:
            skipped.append(f"{name or 'region'}:{reason}")
            continue
        sized.append((i, r, mask, area))

    if skipped:
        print(f"[sam-semantic] skipping surface paint: {skipped}")

    # Large masks first → small masks overwrite on top
    sized.sort(key=lambda t: -t[3])
    regions_out: dict[int, dict] = {}
    for rid, r, mask, _area in sized:
        label_map[mask > 0] = rid
        regions_out[rid] = {
            "kind": r.get("name") or "region",
            "color_name": r.get("color_name"),
            "name": r.get("name"),
        }
    return label_map, regions_out, skipped


def _resize_mask_to_image(mask, H: int, W: int):
    if mask is None:
        return None
    if mask.shape == (H, W):
        return (mask > 0).astype(np.uint8)
    try:
        from PIL import Image
        m = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
            (W, H), Image.NEAREST)
        return (np.array(m) > 0).astype(np.uint8)
    except Exception:
        return None


def project_semantic_sam(
    grid,
    photo_path: str | Path,
    palette: list[dict],
    gpt_data: dict,
    front_axis: str = "-y",
    paint_mode: str = "all",
) -> tuple[set[int], dict]:
    """SAM 2 + GPT pixel-precise version.

    For each GPT region's bbox, asks SAM 2 for the actual SHAPE inside that
    box. Builds a per-pixel label_map from those shape masks (not from
    rectangles), then UV-projects voxels onto it.

    Returns ``(used_lego_ids, meta)``. On failure (SAM unavailable or
    inference errored) returns ``(set(), {"sam_failed": True})`` so the
    caller can fall back to ``project_semantic_gpt``.
    """
    from sam_segmentation import segment_with_boxes

    regions_in = gpt_data.get("regions") or []
    if not regions_in:
        return set(), {}

    # Filter to regions with valid bboxes; preserve order for index alignment
    valid_pairs: list[tuple[int, dict, tuple]] = []
    for i, r in enumerate(regions_in):
        bb = r.get("bbox_normalized")
        if bb and len(bb) == 4:
            x0, y0, x1, y1 = bb
            valid_pairs.append((i, r, (float(x0), float(y0), float(x1), float(y1))))
    if not valid_pairs:
        return set(), {}

    boxes = [bb for _, _, bb in valid_pairs]
    valid_regions = [r for _, r, _ in valid_pairs]

    masks = segment_with_boxes(photo_path, boxes)
    if masks is None:
        # transformers missing / load failed / inference errored
        return set(), {"sam_failed": True}
    if not masks:
        return set(), {"sam_failed": True}

    from PIL import Image
    img = Image.open(photo_path)
    W, H = img.size
    masks_by_region_name = {}
    for r, mask in zip(valid_regions, masks):
        name = (r.get("name") or "").strip()
        if not name:
            continue
        normalized = _resize_mask_to_image(mask, H, W)
        if normalized is not None and normalized.sum() > 0:
            masks_by_region_name[name] = normalized
            masks_by_region_name[name.lower()] = normalized
    if masks_by_region_name:
        gpt_data["_sam_masks_by_region_name"] = masks_by_region_name

    label_map, regions, skipped = _build_sam_label_map(
        masks, valid_regions, H, W, paint_mode=paint_mode)
    if not regions:
        return set(), {
            "sam": True,
            "paint_mode": paint_mode,
            "no_painted_regions": True,
            "skipped": skipped,
        }

    # Map each region → LEGO palette ID by GPT's color_name (no CIELAB needed)
    region_to_palette: dict[int, int] = {}
    region_to_rgb: dict[int, tuple[int, int, int]] = {}
    for rid, info in regions.items():
        if info["kind"] == "eye":
            pid = 4  # Black
        else:
            pid = _palette_id_for_name(palette, info.get("color_name"))
            if pid is None:
                pid = 2  # Light Bluish Gray fallback
        region_to_palette[rid] = pid
        entry = next(p for p in palette if int(p["id"]) == pid)
        region_to_rgb[rid] = tuple(int(v) for v in entry["rgb"])

    used_ids = set(region_to_palette.values())
    projection_meta = _project_label_map_onto_voxels(
        grid, label_map, region_to_rgb, front_axis=front_axis)
    if projection_meta.get("label_map_empty"):
        return used_ids, {"sam": True, "label_map_empty": True}

    return used_ids, {
        "regions": len(regions),
        "lego_ids_used": sorted(used_ids),
        "region_to_palette": region_to_palette,
        "projection": projection_meta,
        "sam": True,
        "paint_mode": paint_mode,
        "skipped": skipped,
    }
