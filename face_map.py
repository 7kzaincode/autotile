"""Pet face overlay generated directly from the source photo.

The mesh/voxel pipeline is good at coarse body shape, but it is too unstable
for recognizable eyes. This module treats the face as a protected 2D overlay:
detect landmarks in the source photo, scale them to the voxel face, and attach
front-mounted tiles as a small LEGO mosaic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


EYE_BLACK = 4
EYE_HIGHLIGHT = 1
MOUTH_DARK = 4


def apply_eye_face_map(
    payload: dict,
    photo_path: str | Path,
    feat: dict,
    *,
    mirror_features: bool = True,
    body_photo_path: str | Path | None = None,
    body_gpt_data: dict | None = None,
    body_view: str | None = None,
) -> tuple[dict, dict]:
    return apply_pet_face_map(
        payload, photo_path, feat, mirror_features=mirror_features,
        include_nose_mouth=False,
        body_photo_path=body_photo_path,
        body_gpt_data=body_gpt_data,
        body_view=body_view,
    )


def apply_pet_face_map(
    payload: dict,
    photo_path: str | Path,
    feat: dict,
    *,
    mirror_features: bool = True,
    include_nose_mouth: bool = True,
    body_photo_path: str | Path | None = None,
    body_gpt_data: dict | None = None,
    body_view: str | None = None,
) -> tuple[dict, dict]:
    """Add multi-stud eye patches to a brick payload.

    Returns ``(payload, meta)``. A single canonical face plane is inferred from
    the head/landmark region, then all face-map pieces use that same plane so
    eyes, nose, and mouth read as one coherent LEGO face.
    """
    if not payload.get("bricks") or not feat.get("eyes"):
        return payload, {"added": 0, "reason": "missing bricks or eyes"}

    grid_shape = tuple(payload["grid_shape"])
    sx, sy, sz = grid_shape
    bbox = feat["bbox"]
    front_axis = str((payload.get("voxel_metadata") or {}).get("front_axis") or "-y")
    axes = _projection_axes(front_axis, grid_shape)
    head_bbox = tuple(feat.get("head_bbox") or _landmark_head_bbox(feat, bbox))
    side_anchor = _side_head_target_from_photo(
        body_photo_path,
        body_gpt_data,
        grid_shape,
        axes,
        body_view=body_view,
    )
    occ = _build_occ(payload["bricks"])
    if side_anchor:
        front_face = _front_face_target_from_side_anchor(side_anchor, grid_shape, occ)
        if front_face:
            axes = front_face["axes"]
            target = front_face["target"]
        else:
            target = side_anchor
    else:
        target = _target_face_region(feat, bbox, head_bbox, grid_shape, axes)
    surface = _front_surface_map(occ, grid_shape, axes)
    eyes = list(feat.get("eyes") or [])
    single_visible_eye = bool(target.get("single_visible_eye")) and not target.get("front_face_anchor")
    if single_visible_eye and eyes:
        eyes = [_single_side_eye_point(eyes)]
    elif mirror_features and len(eyes) == 1:
        ex, ey = eyes[0]
        mid_x = (head_bbox[0] + head_bbox[2]) / 2
        eyes = [eyes[0], (int(2 * mid_x - ex), ey)]

    eye_source_points = eyes[:1 if single_visible_eye else 2]
    eye_xz = [
        _side_profile_feature_uz("eye", u, v, head_bbox, target)
        if target.get("side_profile_anchor") and not target.get("front_face_anchor")
        else _photo_uv_to_region_uz(u, v, head_bbox, target)
        for u, v in eye_source_points
    ]
    centers = []
    for vx, vz in eye_xz:
        hit = _nearest_surface_cell(surface, axes["u_size"], sz, vx, vz, radius=4)
        if hit is not None:
            centers.append(hit)
        else:
            centers.append((vx, vz))
    centers = sorted(set(centers), key=lambda p: p[0])
    if len(centers) < 2 and mirror_features and centers and not single_visible_eye:
        vx, vz = centers[0]
        mirror = _nearest_surface_cell(surface, axes["u_size"], sz, axes["u_size"] - 1 - vx, vz, radius=5)
        if mirror is not None:
            centers = sorted(set([centers[0], mirror]), key=lambda p: p[0])
    if not centers:
        return payload, {"added": 0, "reason": "no front surface near eyes"}
    valid, reason = _validate_eye_centers(centers, target, axes, grid_shape)
    if not valid:
        return payload, {
            "added": 0,
            "reason": reason,
            "eye_centers": [[int(x), int(z)] for x, z in centers[:2]],
            "head_bbox": [int(v) for v in head_bbox],
            "target": target,
        }

    plane = _infer_face_plane(surface, grid_shape, centers, feat, head_bbox, target, axes)
    plane_depth = int(plane["depth"])

    if len(centers) >= 2:
        eye_sep = max(1, abs(centers[-1][0] - centers[0][0]))
    else:
        eye_sep = max(3, sx // 3)
    # Scale from the detected inter-eye distance. The source bunny has large,
    # glossy oval eyes; 1x1 dots destroy the expression.
    if target.get("front_face_anchor"):
        min_eye_w = 1
        max_eye_w = 2
        base_eye_w = int(np.clip(round(eye_sep * 0.18), min_eye_w, max_eye_w))
        base_eye_h = int(np.clip(base_eye_w + 1, 2, 3))
    else:
        min_eye_w = 2 if target.get("side_like") else 3
        max_eye_w = 4 if target.get("side_like") else 5
        base_eye_w = int(np.clip(round(eye_sep * 0.24), min_eye_w, max_eye_w))
        base_eye_h = int(np.clip(round(base_eye_w * 1.35), 3, 6))

    max_overlay_tiles = 42
    clamped = False

    def _build_eye_tiles(w: int, h: int) -> list[dict]:
        if target.get("front_face_anchor"):
            return _front_pet_eye_tiles(
                photo_path,
                centers[:2],
                eye_source_points[:2],
                w,
                h,
                surface,
                plane_depth,
                axes,
                sz,
            )
        tiles = []
        used_cells: set[tuple[int, int]] = set()
        for cx, cz in centers[:2]:
            cells = _ellipse_cells(cx, cz, w, h)
            highlight = _highlight_cell(cx, cz, w, h)
            for tx, tz in cells:
                if not (0 <= tx < axes["u_size"] and 0 <= tz < sz):
                    continue
                if not _near_occupied(surface, axes["u_size"], sz, tx, tz, radius=2):
                    continue
                key = (tx, tz)
                if key in used_cells:
                    continue
                used_cells.add(key)
                color = EYE_HIGHLIGHT if (tx, tz) == highlight else EYE_BLACK
                tiles.append(_mounted_tile(tx, plane_depth, tz, color, "eye", axes))
        return tiles

    additions = []
    eye_w = base_eye_w
    eye_h = base_eye_h
    for candidate_w in range(base_eye_w, max(0, min_eye_w - 1), -1):
        candidate_h = int(np.clip(round(candidate_w * 1.25), 2, max(2, base_eye_h)))
        eye_tiles = _build_eye_tiles(candidate_w, candidate_h)
        if eye_tiles and len(eye_tiles) <= max_overlay_tiles:
            eye_w, eye_h = candidate_w, candidate_h
            additions = eye_tiles
            clamped = candidate_w != base_eye_w or candidate_h != base_eye_h
            break

    if not additions:
        additions = _build_eye_tiles(max(1, min_eye_w), 2)
        eye_w, eye_h = max(1, min_eye_w), 2
        clamped = True

    if include_nose_mouth:
        details = []
        details.extend(_nose_tiles(photo_path, feat, head_bbox, target, grid_shape, surface, plane_depth, eye_sep, axes))
        if not target.get("front_face_anchor"):
            details.extend(_mouth_tiles(feat, head_bbox, target, grid_shape, surface, plane_depth, eye_sep, axes))
        if len(additions) + len(details) <= max_overlay_tiles:
            additions.extend(details)
        else:
            clamped = True
            for piece in details:
                if len(additions) >= max_overlay_tiles:
                    break
                additions.append(piece)

    if len(additions) > max_overlay_tiles:
        return payload, {
            "added": 0,
            "reason": f"face overlay too large ({len(additions)} tiles)",
            "head_bbox": [int(v) for v in head_bbox],
            "target": target,
        }

    if additions:
        payload["bricks"] = list(payload["bricks"]) + additions
    return payload, {
        "added": len(additions),
        "face_plane": plane,
        "eye_centers": [
            list(_coord_from_axes(x, plane_depth, z, axes))
            for x, z in centers[:2]
        ],
        "eye_size": [eye_w, eye_h],
        "clamped": bool(clamped),
        "head_bbox": [int(v) for v in head_bbox],
        "target": target,
        "side_anchor": bool(side_anchor),
    }


def _build_occ(bricks: list[dict]) -> set[tuple[int, int, int]]:
    occ: set[tuple[int, int, int]] = set()
    for b in bricks:
        if b.get("mount") or b.get("face_axis"):
            continue
        for dx in range(int(b.get("size_x", 1))):
            for dy in range(int(b.get("size_y", 1))):
                occ.add((int(b["x"]) + dx, int(b["y"]) + dy, int(b["z"])))
    return occ


def _projection_axes(front_axis: str, grid_shape: tuple) -> dict:
    sx, sy, _sz = grid_shape
    raw = (front_axis or "-y").strip().lower()
    sign = -1 if raw.startswith("-") else 1
    axis = raw.lstrip("+-")
    if axis == "x":
        return {
            "front_axis": "-x" if sign < 0 else "+x",
            "depth_axis": 0,
            "u_axis": 1,
            "u_size": int(sy),
            "depth_size": int(sx),
            "sign": sign,
        }
    return {
        "front_axis": "-y" if sign < 0 else "+y",
        "depth_axis": 1,
        "u_axis": 0,
        "u_size": int(sx),
        "depth_size": int(sy),
        "sign": sign,
    }


def _coord_from_axes(u: int, depth: int, z: int, axes: dict) -> tuple[int, int, int]:
    coord = [0, 0, int(z)]
    coord[int(axes["u_axis"])] = int(u)
    coord[int(axes["depth_axis"])] = int(depth)
    return int(coord[0]), int(coord[1]), int(coord[2])


def _photo_uv_to_voxel(u: int, v: int, bbox: tuple, grid_shape: tuple) -> tuple[int, int]:
    return _photo_uv_to_uz(u, v, bbox, grid_shape, _projection_axes("-y", grid_shape))


def _photo_uv_to_uz(u: int, v: int, bbox: tuple, grid_shape: tuple, axes: dict) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    _sx, _sy, sz = grid_shape
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    vx = int(np.clip(((u - x0) / bw) * int(axes["u_size"]), 0, int(axes["u_size"]) - 1))
    vz = int(np.clip((1.0 - ((v - y0) / bh)) * sz, 0, sz - 1))
    return vx, vz


def _photo_uv_to_region_uz(u: int, v: int, bbox: tuple, target: dict) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    u_n = float(np.clip((u - x0) / bw, 0.0, 1.0))
    v_n = float(np.clip((v - y0) / bh, 0.0, 1.0))
    u0, z0, u1, z1 = target["region"]
    tw = max(1, u1 - u0)
    th = max(1, z1 - z0)
    vu = int(np.clip(round(u0 + u_n * (tw - 1)), u0, u1 - 1))
    vz = int(np.clip(round(z1 - 1 - v_n * (th - 1)), z0, z1 - 1))
    return vu, vz


def _landmark_head_bbox(feat: dict, bbox: tuple) -> tuple[int, int, int, int]:
    pts = []
    pts.extend(feat.get("eyes") or [])
    if feat.get("nose"):
        pts.append(feat["nose"])
    if feat.get("mouth"):
        pts.extend(feat["mouth"])
    if not pts:
        return tuple(bbox)
    xs = [int(p[0]) for p in pts]
    ys = [int(p[1]) for p in pts]
    x0, y0, x1, y1 = bbox
    eye_dx = 0
    eyes = feat.get("eyes") or []
    if len(eyes) >= 2:
        eye_dx = abs(int(eyes[1][0]) - int(eyes[0][0]))
    span = max(24, eye_dx, max(xs) - min(xs), max(ys) - min(ys))
    pad_x = int(round(span * 0.85))
    pad_top = int(round(span * 1.05))
    pad_bottom = int(round(span * 1.20))
    cx = int(round(float(np.mean(xs))))
    top = min(ys)
    bottom = max(ys)
    return (
        max(x0, cx - pad_x),
        max(y0, top - pad_top),
        min(x1, cx + pad_x),
        min(y1, bottom + pad_bottom),
    )


def _target_face_region(
    feat: dict,
    full_bbox: tuple,
    head_bbox: tuple,
    grid_shape: tuple,
    axes: dict,
) -> dict:
    _sx, _sy, sz = grid_shape
    side_like = axes["depth_axis"] == 0
    pts = []
    pts.extend(feat.get("eyes") or [])
    if feat.get("nose"):
        pts.append(feat["nose"])
    if feat.get("mouth"):
        pts.extend(feat["mouth"])
    if pts:
        mean_u = int(round(float(np.mean([p[0] for p in pts]))))
        mean_v = int(round(float(np.mean([p[1] for p in pts]))))
    else:
        mean_u = int(round((head_bbox[0] + head_bbox[2]) / 2))
        mean_v = int(round((head_bbox[1] + head_bbox[3]) / 2))
    center_u, center_z = _photo_uv_to_uz(mean_u, mean_v, full_bbox, grid_shape, axes)
    if side_like:
        # Full-body side pets only get a small head-local target. Mapping the
        # head photo onto the whole body is what created back-mounted eye bars.
        u_span = int(np.clip(round(axes["u_size"] * 0.24), 8, 18))
        z_span = int(np.clip(round(sz * 0.38), 9, 22))
    else:
        u_span = int(np.clip(round(axes["u_size"] * 0.58), 8, axes["u_size"]))
        z_span = int(np.clip(round(sz * 0.45), 8, sz))
    u0 = int(np.clip(center_u - u_span // 2, 0, max(0, axes["u_size"] - u_span)))
    z0 = int(np.clip(center_z - z_span // 2, 0, max(0, sz - z_span)))
    return {
        "region": [u0, z0, min(axes["u_size"], u0 + u_span), min(sz, z0 + z_span)],
        "side_like": bool(side_like),
    }


def _side_head_target_from_photo(
    body_photo_path: str | Path | None,
    body_gpt_data: dict | None,
    grid_shape: tuple,
    axes: dict,
    *,
    body_view: str | None = None,
) -> dict | None:
    """Infer where the side-view head sits on the voxel model.

    Front portraits are excellent for eye/nose proportions, but they do not tell
    us where the head lives on a side-profile body mesh. This anchor uses the
    body photo's alpha silhouette: the head is the end with the most upper-body
    mass/ears, while the other end is usually tail/hips. GPT head-ish boxes are
    accepted only when they agree with that silhouette signal.
    """
    if body_photo_path is None:
        return None
    if (body_view or "").lower() not in {"left", "right", "side"}:
        return None

    try:
        from PIL import Image
        img = Image.open(body_photo_path).convert("RGBA")
    except Exception:
        return None

    arr = np.asarray(img)
    alpha = arr[..., 3]
    fg = alpha > 16
    if not fg.any():
        rgb = arr[..., :3]
        fg = rgb.min(axis=2) < 245
    if not fg.any():
        return None

    ys, xs = np.where(fg)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = fg[y0:y1, x0:x1]
    h, w = crop.shape
    if w < 8 or h < 8:
        return None

    head_side = _infer_side_head_end(crop, body_gpt_data, (x0, y0, x1, y1), img.size)
    upper = crop[: max(1, int(round(h * 0.58))), :]
    col_counts = upper.sum(axis=0)
    significant = np.where(col_counts > max(2, int(round(h * 0.012))))[0]
    if significant.size == 0:
        col_counts = crop.sum(axis=0)
        significant = np.where(col_counts > max(2, int(round(h * 0.018))))[0]
    if significant.size == 0:
        return None

    # Keep the anchor tight. A full-body target lets one eye drift onto the back.
    max_span = int(np.clip(round(int(axes["u_size"]) * 0.34), 10, 22))
    if head_side == "right":
        px1 = int(significant.max()) + 1
        px0 = max(int(significant.min()), px1 - max(8, int(round(w * 0.36))))
    else:
        px0 = int(significant.min())
        px1 = min(int(significant.max()) + 1, px0 + max(8, int(round(w * 0.36))))

    band = crop[:, max(0, px0): min(w, px1)]
    bys, _bxs = np.where(band)
    if bys.size == 0:
        return None
    py0 = int(bys.min())
    py1 = int(bys.max()) + 1
    # Do not let chest/legs expand the face target downward.
    py1 = min(py1, int(round(h * 0.72)))

    u0 = int(np.floor((px0 / max(1, w)) * int(axes["u_size"])))
    u1 = int(np.ceil((px1 / max(1, w)) * int(axes["u_size"])))
    if u1 - u0 > max_span:
        if head_side == "right":
            u0 = max(0, u1 - max_span)
        else:
            u1 = min(int(axes["u_size"]), u0 + max_span)
    u0 = int(np.clip(u0, 0, max(0, int(axes["u_size"]) - 1)))
    u1 = int(np.clip(max(u0 + 1, u1), u0 + 1, int(axes["u_size"])))

    _sx, _sy, sz = grid_shape
    z0 = int(np.floor((1.0 - (py1 / max(1, h))) * sz))
    z1 = int(np.ceil((1.0 - (py0 / max(1, h))) * sz))
    z0 = int(np.clip(z0, 0, max(0, sz - 1)))
    z1 = int(np.clip(max(z0 + 1, z1), z0 + 1, sz))
    max_z_span = int(np.clip(round(sz * 0.56), 10, 24))
    if z1 - z0 > max_z_span:
        z0 = max(0, z1 - max_z_span)

    return {
        "region": [u0, z0, u1, z1],
        "side_like": True,
        "side_profile_anchor": True,
        "single_visible_eye": True,
        "head_side": head_side,
        "body_view": (body_view or "").lower(),
        "source": "body-photo-head-anchor",
        "silhouette_bbox": [x0, y0, x1, y1],
    }


def _front_face_target_from_side_anchor(
    side_anchor: dict,
    grid_shape: tuple,
    occ: set[tuple[int, int, int]],
) -> dict | None:
    """Create a front-facing face plane from a side-photo head anchor.

    The side/body photo tells us which end of the model is the head. The actual
    pet face should still be composed from the front portrait, so the face plane
    is mounted on the head end (+X or -X), with its horizontal axis across the
    model's depth/width (Y).
    """
    if not side_anchor or not occ:
        return None
    sx, sy, sz = grid_shape
    head_right = side_anchor.get("head_side", "right") == "right"
    face_axis = "+x" if head_right else "-x"
    axes = _projection_axes(face_axis, grid_shape)
    surface = _front_surface_map(occ, grid_shape, axes)
    if not (surface >= 0).any():
        return None

    x0, z0, x1, z1 = [int(v) for v in side_anchor.get("region", [0, 0, sx, sz])]
    z0 = int(np.clip(z0 - max(2, int(round(sz * 0.10))), 0, max(0, sz - 1)))
    z1 = int(np.clip(z1 + max(1, int(round(sz * 0.04))), z0 + 1, sz))

    depth_mask = surface >= 0
    if head_right:
        depth_mask &= surface >= max(0, x0 - 2)
    else:
        depth_mask &= surface <= min(sx - 1, x1 + 1)
    z_mask = np.zeros_like(depth_mask, dtype=bool)
    z_mask[:, z0:z1] = True
    mask = depth_mask & z_mask

    if mask.any():
        us, zs = np.where(mask)
        u0 = int(max(0, us.min() - 2))
        u1 = int(min(sy, us.max() + 3))
        z0 = int(max(0, zs.min() - 1))
        z1 = int(min(sz, zs.max() + 2))
    else:
        # Fallback: a centered face patch across the head width.
        span = int(np.clip(round(sy * 0.62), 8, sy))
        u0 = int(max(0, (sy - span) // 2))
        u1 = int(min(sy, u0 + span))

    min_span = min(sy, 8)
    if u1 - u0 < min_span:
        mid = int(round((u0 + u1) * 0.5))
        u0 = int(max(0, mid - min_span // 2))
        u1 = int(min(sy, u0 + min_span))

    return {
        "axes": axes,
        "target": {
            "region": [u0, z0, u1, z1],
            "side_like": False,
            "front_face_anchor": True,
            "head_side": side_anchor.get("head_side", "right"),
            "body_view": side_anchor.get("body_view"),
            "source": "front-face-body-head-anchor",
            "side_region": side_anchor.get("region"),
        },
    }


def _infer_side_head_end(
    crop: np.ndarray,
    body_gpt_data: dict | None,
    silhouette_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> str:
    h, w = crop.shape
    upper = crop[: max(1, int(round(h * 0.58))), :]
    q = max(1, w // 4)
    left_score = float(upper[:, :q].sum())
    right_score = float(upper[:, -q:].sum())

    gpt_xs = _headish_gpt_centers(body_gpt_data, silhouette_bbox, image_size)
    if gpt_xs:
        gpt_center = float(np.median(gpt_xs))
        if gpt_center <= 0.34 and left_score >= right_score * 0.55:
            return "left"
        if gpt_center >= 0.66 and right_score >= left_score * 0.55:
            return "right"

    return "right" if right_score >= left_score else "left"


def _headish_gpt_centers(
    body_gpt_data: dict | None,
    silhouette_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> list[float]:
    if not body_gpt_data:
        return []
    names = {
        "head", "face", "ear", "ears", "ear_interior", "eye", "eyes",
        "eye_socket", "nose", "mouth", "muzzle", "snout", "cheek",
    }
    W, H = image_size
    x0, _y0, x1, _y1 = silhouette_bbox
    sw = max(1, x1 - x0)
    centers: list[float] = []
    for region in body_gpt_data.get("regions") or []:
        name = str(region.get("name") or "").strip().lower()
        if name not in names:
            continue
        bbox = region.get("bbox_normalized")
        if not bbox or len(bbox) != 4:
            continue
        try:
            gx = ((float(bbox[0]) + float(bbox[2])) * 0.5 * W - x0) / sw
        except Exception:
            continue
        if -0.2 <= gx <= 1.2:
            centers.append(float(np.clip(gx, 0.0, 1.0)))

    feats = (body_gpt_data or {}).get("features") or {}
    for key in ("eye_left", "eye_right", "nose", "mouth"):
        p = feats.get(key) or {}
        vals = p.get("position_normalized") if isinstance(p, dict) else None
        if not vals or len(vals) != 2:
            continue
        try:
            gx = (float(vals[0]) * W - x0) / sw
        except Exception:
            continue
        if -0.2 <= gx <= 1.2:
            centers.append(float(np.clip(gx, 0.0, 1.0)))
    return centers


def _single_side_eye_point(eyes: list[tuple[int, int]]) -> tuple[int, int]:
    if len(eyes) == 1:
        return eyes[0]
    xs = [int(p[0]) for p in eyes[:2]]
    ys = [int(p[1]) for p in eyes[:2]]
    return int(round(float(np.mean(xs)))), int(round(float(np.mean(ys))))


def _side_profile_feature_uz(
    part: str,
    u: int,
    v: int,
    bbox: tuple,
    target: dict,
) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    bh = max(1, y1 - y0)
    v_n = float(np.clip((v - y0) / bh, 0.0, 1.0))
    u0, z0, u1, z1 = target["region"]
    tw = max(1, u1 - u0)
    th = max(1, z1 - z0)
    head_right = target.get("head_side", "right") == "right"
    if part == "nose":
        u_frac = 0.88 if head_right else 0.12
    elif part == "mouth":
        u_frac = 0.82 if head_right else 0.18
    else:
        u_frac = 0.64 if head_right else 0.36
    vu = int(np.clip(round(u0 + u_frac * (tw - 1)), u0, u1 - 1))
    vz = int(np.clip(round(z1 - 1 - v_n * (th - 1)), z0, z1 - 1))
    return vu, vz


def _validate_eye_centers(
    centers: list[tuple[int, int]],
    target: dict,
    axes: dict,
    grid_shape: tuple,
) -> tuple[bool, str | None]:
    if len(centers) < 2:
        return True, None
    u0, z0, u1, z1 = target["region"]
    c = sorted(centers[:2], key=lambda p: p[0])
    eye_sep = abs(c[1][0] - c[0][0])
    eye_dz = abs(c[1][1] - c[0][1])
    region_w = max(1, u1 - u0)
    region_h = max(1, z1 - z0)
    if target.get("side_like") and eye_sep > max(6, int(round(region_w * 0.70))):
        return False, f"eye separation {eye_sep} exceeds head region"
    if eye_dz > max(4, int(round(region_h * 0.34))):
        return False, f"eye vertical mismatch {eye_dz} exceeds head region"
    for u, z in c:
        if not (u0 - 2 <= u <= u1 + 1 and z0 - 2 <= z <= z1 + 1):
            return False, "eye center outside head region"
    return True, None


def _front_surface(occ: set[tuple[int, int, int]], axes: dict, u: int, vz: int) -> int | None:
    size = int(axes["depth_size"])
    depths = range(size) if int(axes["sign"]) < 0 else range(size - 1, -1, -1)
    for depth in depths:
        if _coord_from_axes(u, depth, vz, axes) in occ:
            return int(depth)
    return None


def _front_surface_map(
    occ: set[tuple[int, int, int]],
    grid_shape: tuple,
    axes: dict,
) -> np.ndarray:
    _sx, _sy, sz = grid_shape
    u_size = int(axes["u_size"])
    out = np.full((u_size, sz), -1, dtype=np.int32)
    for x in range(u_size):
        for z in range(sz):
            y = _front_surface(occ, axes, x, z)
            if y is not None:
                out[x, z] = int(y)
    return out


def _nearest_surface_cell(
    surface: np.ndarray,
    sx: int,
    sz: int,
    vx: int,
    vz: int,
    *,
    radius: int = 3,
) -> tuple[int, int] | None:
    vx = int(np.clip(vx, 0, sx - 1))
    vz = int(np.clip(vz, 0, sz - 1))
    for r in range(radius + 1):
        candidates = []
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if max(abs(dx), abs(dz)) != r:
                    continue
                nx = int(np.clip(vx + dx, 0, sx - 1))
                nz = int(np.clip(vz + dz, 0, sz - 1))
                if surface[nx, nz] >= 0:
                    candidates.append((abs(dx) + abs(dz), nx, nz))
        if candidates:
            _dist, nx, nz = min(candidates, key=lambda t: t[0])
            return nx, nz
    return None


def _near_occupied(surface: np.ndarray, sx: int, sz: int, vx: int, vz: int, radius: int = 2) -> bool:
    return _nearest_surface_cell(surface, sx, sz, vx, vz, radius=radius) is not None


def _infer_face_plane(
    surface: np.ndarray,
    grid_shape: tuple,
    centers: list[tuple[int, int]],
    feat: dict,
    bbox: tuple,
    target: dict,
    axes: dict,
) -> dict:
    sx, sy, sz = grid_shape
    xs = [c[0] for c in centers]
    zs = [c[1] for c in centers]
    if feat.get("nose"):
        if target.get("side_profile_anchor"):
            nx, nz = _side_profile_feature_uz("nose", *feat["nose"], bbox, target)
        else:
            nx, nz = _photo_uv_to_region_uz(*feat["nose"], bbox, target)
        xs.append(nx)
        zs.append(nz)
    if feat.get("mouth"):
        for p in feat["mouth"]:
            if target.get("side_profile_anchor"):
                mx, mz = _side_profile_feature_uz("mouth", *p, bbox, target)
            else:
                mx, mz = _photo_uv_to_region_uz(*p, bbox, target)
            xs.append(mx)
            zs.append(mz)
    cx = int(round(float(np.mean(xs))))
    cz = int(round(float(np.mean(zs))))
    eye_sep = max(2, max(xs) - min(xs)) if len(xs) >= 2 else max(2, int(axes["u_size"]) // 4)
    rx = max(3, int(round(eye_sep * 0.85)))
    rz = max(4, int(round(eye_sep * 0.95)))
    tu0, tz0, tu1, tz1 = target["region"]
    x0, x1 = max(tu0, cx - rx), min(tu1, cx + rx + 1)
    z0, z1 = max(tz0, cz - rz), min(tz1, cz + rz + 1)
    vals = surface[x0:x1, z0:z1]
    vals = vals[vals >= 0]
    if vals.size == 0:
        vals = surface[surface >= 0]
    if vals.size == 0:
        y = 0
    else:
        # Use a front-biased percentile. Minimum can hit one stray snout/whisker
        # voxel; median makes eyes sink into cheeks.
        percentile = 18 if int(axes["sign"]) < 0 else 82
        y = int(np.clip(np.percentile(vals, percentile), 0, int(axes["depth_size"]) - 1))
    return {
        "depth": y,
        "y": y if int(axes["depth_axis"]) == 1 else None,
        "axis": axes["front_axis"],
        "center": [cx, cz],
        "region": [x0, z0, x1, z1],
    }


def _front_surface_near(
    occ: set[tuple[int, int, int]],
    sx: int,
    sy: int,
    sz: int,
    vx: int,
    vz: int,
    *,
    radius: int = 2,
) -> tuple[int, int, int] | None:
    axes = _projection_axes("-y", (sx, sy, sz))
    vx = int(np.clip(vx, 0, sx - 1))
    vz = int(np.clip(vz, 0, sz - 1))
    for r in range(radius + 1):
        candidates = []
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if max(abs(dx), abs(dz)) != r:
                    continue
                nx = int(np.clip(vx + dx, 0, sx - 1))
                nz = int(np.clip(vz + dz, 0, sz - 1))
                vy = _front_surface(occ, axes, nx, nz)
                if vy is not None:
                    candidates.append((abs(dx) + abs(dz), nx, vy, nz))
        if candidates:
            _dist, nx, vy, nz = min(candidates, key=lambda t: t[0])
            return nx, vy, nz
    return None


def _ellipse_cells(cx: int, cz: int, w: int, h: int) -> list[tuple[int, int]]:
    rx = max(1.0, w / 2.0)
    rz = max(1.0, h / 2.0)
    cells = []
    for dx in range(-int(np.ceil(rx)), int(np.ceil(rx)) + 1):
        for dz in range(-int(np.ceil(rz)), int(np.ceil(rz)) + 1):
            # Slightly vertical oval, evaluated at cell centers.
            if ((dx + 0.0) / rx) ** 2 + ((dz + 0.0) / rz) ** 2 <= 1.0:
                cells.append((cx + dx, cz + dz))
    return cells


def _highlight_cell(cx: int, cz: int, w: int, h: int) -> tuple[int, int]:
    return cx + max(1, w // 3), cz + max(1, h // 4)


def _front_pet_eye_tiles(
    photo_path: str | Path,
    centers: list[tuple[int, int]],
    source_points: list[tuple[int, int]],
    w: int,
    h: int,
    surface: np.ndarray,
    plane_depth: int,
    axes: dict,
    sz: int,
) -> list[dict]:
    tiles: list[dict] = []
    used_cells: set[tuple[int, int]] = set()
    for idx, (cx, cz) in enumerate(centers[:2]):
        src = source_points[min(idx, len(source_points) - 1)] if source_points else None
        iris_color = (
            _sample_eye_iris_palette_id(photo_path, *src)
            if src is not None else 26
        )
        cells = _ellipse_cells(cx, cz, w, h)
        pupil = (cx, cz)
        highlight = _highlight_cell(cx, cz, w, h) if w >= 3 else None
        for tx, tz in cells:
            if not (0 <= tx < axes["u_size"] and 0 <= tz < sz):
                continue
            if not _near_occupied(surface, axes["u_size"], sz, tx, tz, radius=2):
                continue
            key = (tx, tz)
            if key in used_cells:
                continue
            used_cells.add(key)
            if (tx, tz) == pupil:
                color = EYE_BLACK
            elif highlight is not None and (tx, tz) == highlight:
                color = EYE_HIGHLIGHT
            else:
                color = iris_color
            tiles.append(_mounted_tile(tx, plane_depth, tz, color, "eye", axes))
    return tiles


def _sample_eye_iris_palette_id(photo_path: str | Path, u: int, v: int) -> int:
    try:
        from PIL import Image
        from voxels_to_palette import load_palette, _rgb_to_lab, AVAILABILITY_PENALTY
        img = Image.open(photo_path).convert("RGBA")
        W, H = img.size
        r = max(6, min(W, H) // 90)
        x0, x1 = max(0, int(u) - r), min(W, int(u) + r + 1)
        y0, y1 = max(0, int(v) - r), min(H, int(v) + r + 1)
        crop = np.asarray(img.crop((x0, y0, x1, y1)))
        if crop.size == 0:
            return 26
        rgb = crop[..., :3].astype(np.float32)
        alpha = crop[..., 3] > 16
        bright = rgb.mean(axis=2)
        spread = rgb.max(axis=2) - rgb.min(axis=2)
        # Ignore black pupil, white glints, and nearly gray fur around the eye.
        iris = alpha & (bright > 45) & (bright < 220) & (spread > 18)
        if int(iris.sum()) < 4:
            iris = alpha & (bright > 35) & (bright < 235)
        if int(iris.sum()) < 1:
            return 26
        sample = np.median(rgb[iris], axis=0)
        palette = load_palette()
        candidates = {25, 26, 27, 28, 29, 30, 40}
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
        # White/cream glints can still sneak in; amber eyes read better as a
        # warm LEGO color with a black pupil.
        if picked in {1, 40}:
            return 30
        return picked
    except Exception:
        return 26


def _nose_tiles(
    photo_path: str | Path,
    feat: dict,
    bbox: tuple,
    target: dict,
    grid_shape: tuple,
    surface: np.ndarray,
    plane_depth: int,
    eye_sep: int,
    axes: dict,
) -> list[dict]:
    if not feat.get("nose"):
        return []
    _sx, _sy, sz = grid_shape
    if target.get("side_profile_anchor"):
        vx, vz = _side_profile_feature_uz("nose", *feat["nose"], bbox, target)
    else:
        vx, vz = _photo_uv_to_region_uz(*feat["nose"], bbox, target)
    hit = _nearest_surface_cell(surface, axes["u_size"], sz, vx, vz, radius=4)
    if hit is not None:
        vx, vz = hit
    color = _sample_feature_palette_id(
        photo_path,
        *feat["nose"],
        candidates={4, 25, 26, 27, 28, 29, 31, 40},
        fallback=29,
    )
    w = int(np.clip(round(eye_sep * 0.14), 1, 3))
    cells = [(vx, vz)]
    if w >= 2:
        cells.extend([(vx - 1, vz), (vx + 1, vz)])
    return [
        _mounted_tile(x, plane_depth, z, color, "nose", axes)
        for x, z in cells
        if 0 <= x < axes["u_size"] and 0 <= z < sz and _near_occupied(surface, axes["u_size"], sz, x, z, radius=2)
    ]


def _mouth_tiles(
    feat: dict,
    bbox: tuple,
    target: dict,
    grid_shape: tuple,
    surface: np.ndarray,
    plane_depth: int,
    eye_sep: int,
    axes: dict,
) -> list[dict]:
    if not feat.get("mouth"):
        return []
    _sx, _sy, sz = grid_shape
    (u1, v1), (u2, v2) = feat["mouth"]
    if target.get("side_profile_anchor"):
        vx1, vz1 = _side_profile_feature_uz("mouth", u1, v1, bbox, target)
        vx2, vz2 = _side_profile_feature_uz("mouth", u2, v2, bbox, target)
    else:
        vx1, vz1 = _photo_uv_to_region_uz(u1, v1, bbox, target)
        vx2, vz2 = _photo_uv_to_region_uz(u2, v2, bbox, target)
    vx_mid = int(round((vx1 + vx2) / 2))
    vz_mid = int(round((vz1 + vz2) / 2))
    hit = _nearest_surface_cell(surface, axes["u_size"], sz, vx_mid, vz_mid, radius=4)
    if hit is not None:
        vx_mid, vz_mid = hit
    half = int(np.clip(round(eye_sep * 0.10), 1, 2))
    cells = [(vx_mid + dx, vz_mid) for dx in range(-half, half + 1)]
    return [
        _mounted_tile(x, plane_depth, z, MOUTH_DARK, "mouth", axes)
        for x, z in cells
        if 0 <= x < axes["u_size"] and 0 <= z < sz and _near_occupied(surface, axes["u_size"], sz, x, z, radius=2)
    ]


def _sample_feature_palette_id(
    photo_path: str | Path,
    u: int,
    v: int,
    *,
    candidates: set[int] | None = None,
    fallback: int,
) -> int:
    try:
        from PIL import Image
        from voxels_to_palette import load_palette, _rgb_to_lab, AVAILABILITY_PENALTY
        img = Image.open(photo_path).convert("RGBA")
        W, H = img.size
        x0, x1 = max(0, u - 5), min(W, u + 6)
        y0, y1 = max(0, v - 5), min(H, v + 6)
        crop = np.asarray(img.crop((x0, y0, x1, y1)))
        mask = crop[..., 3] > 16
        if not mask.any():
            return fallback
        rgb = np.median(crop[mask][..., :3].astype(np.float32), axis=0)
        palette = load_palette()
        entries = [p for p in palette if candidates is None or int(p["id"]) in candidates]
        if not entries:
            return fallback
        pal_rgb = np.array([p["rgb"] for p in entries], dtype=np.float32) / 255.0
        target_lab = _rgb_to_lab((rgb[None] / 255.0))[0]
        pal_lab = _rgb_to_lab(pal_rgb)
        penalty = np.array([
            AVAILABILITY_PENALTY.get(p.get("availability", "uncommon"), 0.0)
            for p in entries
        ], dtype=np.float32) ** 2
        dists = ((pal_lab - target_lab) ** 2).sum(axis=1) + penalty
        return int(entries[int(dists.argmin())]["id"])
    except Exception:
        return fallback


def _mounted_tile(
    u: int,
    depth: int,
    z: int,
    color_id: int,
    face_part: str = "face",
    axes: dict | None = None,
) -> dict:
    axes = axes or _projection_axes("-y", (max(1, u + 1), max(1, depth + 1), max(1, z + 1)))
    x, y, z = _coord_from_axes(u, depth, z, axes)
    return {
        "x": int(x), "y": int(y), "z": int(z),
        "size_x": 1, "size_y": 1, "brick_type": "1x1",
        "kind": "tile", "rotation": 0,
        "color": int(color_id), "slope_dir": None,
        "mount": axes["front_axis"],
        "protected": True,
        "face_map": face_part,
    }
