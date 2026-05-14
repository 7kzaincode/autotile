"""
Project an input photo onto the voxel grid as a color texture.

The mesh has the right shape; this stage paints colors onto it. Three knobs
matter for output quality:

1. `back_mode` — how deep into the model to project the front-view photo.
   - "front_only" (default): only the FRONT-most occupied voxel of each column.
     Back stays default color. No mirror artifact.
   - "front_half":   front half of each column gets colors; back half stays default.
   - "through":     every occupied voxel gets the front-view color. Causes a
     mirror effect on the back of cylindrical/symmetric objects.

2. `blur_radius` — Gaussian blur on the photo before projection. Smooths
   gradients and highlights so each region maps to a single LEGO palette
   color instead of speckling across multiple. 0 = off; 2-3 is a good default.

3. `cluster_colors` — k-means quantize the photo to k dominant colors BEFORE
   projection. Eliminates anti-aliased boundary speckle. 0 = off; 6-10 is a
   sweet spot for product photos.

4. `auto_crop` — find the photo's foreground bbox and remap so its corners
   align with the voxel grid's corners. Default True.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from mesh_to_voxels import VoxelGrid


def project_photo(
    grid: VoxelGrid,
    photo_path: str | Path,
    front_axis: str = "-y",
    back_mode: str = "uv",
    skip_background: bool = True,
    auto_crop: bool = True,
    blur_radius: float = 2.0,
    cluster_colors: int = 8,
) -> VoxelGrid:
    """Paint photo pixels onto the voxel grid.

    back_mode options:
      - "none"        — skip; trust the mesh's own texture
      - "front_only"  — only paint the frontmost voxel in each column
      - "front_half"  — paint the front half of each column
      - "through"     — paint every occupied voxel in the column (mirror artifact)
      - "uv"          — RECOMMENDED. Photo-as-UV-texture: each voxel's outward
                        normal determines where it samples from the photo.
                        Front-facing → direct, back → mirrored, side → edge.
                        Gives proper color variation (eyes, ears, belly) without
                        the mirror artifact of "through".
    """
    if back_mode == "none":
        return grid
    if back_mode == "uv":
        return _project_photo_uv(
            grid, photo_path, front_axis=front_axis,
            blur_radius=blur_radius, cluster_colors=cluster_colors,
            auto_crop=auto_crop,
        )
    img = Image.open(photo_path)
    has_alpha = (
        img.mode in ("RGBA", "LA")
        or (img.mode == "P" and "transparency" in img.info)
    )
    if has_alpha:
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")

    # Pre-process: blur THEN cluster. Blur first so anti-aliased edges become
    # smooth gradients, then clustering snaps everything to ~k colors.
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if has_alpha:
        photo = np.asarray(img)
        alpha = photo[..., 3].copy()
        photo = photo[..., :3].copy()
    else:
        photo = np.asarray(img).copy()
        alpha = None

    if cluster_colors > 0:
        photo = _kmeans_quantize(photo, k=cluster_colors, mask=alpha)

    if auto_crop:
        photo, alpha = _crop_to_foreground(photo, alpha)

    H, W = photo.shape[:2]
    occ = grid.occupancy
    sx, sy, sz = occ.shape

    axis, sign = _parse_axis(front_axis)
    if axis == "y":
        depth_axis, u_axis, v_axis = 1, 0, 2
        u_size, v_size = sx, sz
    elif axis == "x":
        depth_axis, u_axis, v_axis = 0, 1, 2
        u_size, v_size = sy, sz
    elif axis == "z":
        depth_axis, u_axis, v_axis = 2, 0, 1
        u_size, v_size = sx, sy
    else:
        raise ValueError(f"unknown front_axis: {front_axis}")

    for ui in range(u_size):
        u_px = int(np.clip((ui + 0.5) / u_size * W, 0, W - 1))
        for vi in range(v_size):
            v_px = int(np.clip((v_size - 1 - vi + 0.5) / v_size * H, 0, H - 1))
            rgb = photo[v_px, u_px]

            if skip_background:
                if alpha is not None and alpha[v_px, u_px] < 32:
                    continue
                if _is_background(rgb):
                    continue

            col_slicer = [slice(None)] * 3
            col_slicer[u_axis] = ui
            col_slicer[v_axis] = vi
            col_occ = occ[tuple(col_slicer)]
            occupied_idx = np.where(col_occ)[0]
            if len(occupied_idx) == 0:
                continue

            depth_indices = _pick_depths(occupied_idx, sign, back_mode)
            for d in depth_indices:
                idx = [0, 0, 0]
                idx[u_axis] = ui
                idx[v_axis] = vi
                idx[depth_axis] = int(d)
                grid.colors[idx[0], idx[1], idx[2]] = rgb

    return grid


def _project_photo_uv(
    grid: "VoxelGrid",
    photo_path,
    front_axis: str = "-y",
    blur_radius: float = 2.0,
    cluster_colors: int = 8,
    auto_crop: bool = True,
) -> "VoxelGrid":
    """Photo-as-UV-texture projection.

    For every occupied voxel:
      1. Determine its outward normal direction (which face is exposed to empty space)
      2. If front-facing: sample photo at (voxel_x, voxel_z) → photo (u, v)
      3. If back-facing: sample photo at mirrored (sx-1-voxel_x, voxel_z)
      4. If side-facing (left/right): sample at the edge column of the photo
      5. Interior voxels: keep the existing color (from mesh's texture, if any)

    This delivers the photo's actual spatial color variation onto the model —
    eyes show up where the photo has eyes, the belly is light where the photo
    is light, ears are dark where the photo is dark — and without the mirror
    artifact of "through" mode.
    """
    img = Image.open(photo_path)
    has_alpha = img.mode in ("RGBA", "LA")
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if cluster_colors > 0:
        # Mild posterization so palette quantization lands cleanly
        if has_alpha:
            rgba = np.asarray(img.convert("RGBA"))
            photo = rgba[..., :3].copy()
            alpha = rgba[..., 3]
        else:
            photo = np.asarray(img.convert("RGB")).copy()
            alpha = None
        photo = _kmeans_quantize(photo, k=cluster_colors, mask=alpha)
    else:
        if has_alpha:
            rgba = np.asarray(img.convert("RGBA"))
            photo = rgba[..., :3].copy()
            alpha = rgba[..., 3]
        else:
            photo = np.asarray(img.convert("RGB")).copy()
            alpha = None

    if auto_crop:
        photo, alpha = _crop_to_foreground(photo, alpha)
    H, W = photo.shape[:2]

    occ = grid.occupancy
    colors = grid.colors
    sx, sy, sz = occ.shape

    # Outward normals: 6-neighbor empty-direction counts per voxel.
    # We pre-compute a "facing" map: integer per voxel where each bit signals
    # whether a neighbor in that direction is empty.
    FRONT, BACK, LEFT, RIGHT, BOTTOM, TOP = 1, 2, 4, 8, 16, 32

    # Build with simple slicing — fast in NumPy.
    def neighbor_empty(shift_axis: int, shift_dir: int) -> np.ndarray:
        pad = np.ones_like(occ, dtype=bool)
        shifted = np.roll(occ, shift=-shift_dir, axis=shift_axis)
        # Outside-grid neighbors count as "empty" (so silhouette edge voxels face out)
        if shift_axis == 0:
            if shift_dir > 0:
                shifted[-1, :, :] = False
            else:
                shifted[0, :, :] = False
        elif shift_axis == 1:
            if shift_dir > 0:
                shifted[:, -1, :] = False
            else:
                shifted[:, 0, :] = False
        elif shift_axis == 2:
            if shift_dir > 0:
                shifted[:, :, -1] = False
            else:
                shifted[:, :, 0] = False
        return occ & ~shifted

    axis_letter = (front_axis or "-y").strip().lower().lstrip("+-")
    sign = -1 if (front_axis or "-y").strip().startswith("-") else 1
    if axis_letter == "x":
        depth_axis, u_axis, u_size = 0, 1, sy
        u_minus_axis, u_plus_axis = (1, -1), (1, +1)
    else:
        # The brick viewer treats z as height, so y is the normal camera-depth
        # axis for projected photos. If an upstream caller passes z, fall back
        # to y because photo vertical mapping already uses z.
        depth_axis, u_axis, u_size = 1, 0, sx
        u_minus_axis, u_plus_axis = (0, -1), (0, +1)

    faces_front = neighbor_empty(depth_axis, sign)
    faces_back = neighbor_empty(depth_axis, -sign)
    faces_u_minus = neighbor_empty(*u_minus_axis)
    faces_u_plus = neighbor_empty(*u_plus_axis)
    # Top/bottom (Z) — usually photo doesn't help here; we leave default
    # but they get counted as "side-ish"

    # Walk every occupied voxel; classify and sample.
    occ_idx = np.argwhere(occ)
    for (x, y, z) in occ_idx:
        coord = (x, y, z)
        u_coord = coord[u_axis]
        is_front = faces_front[x, y, z]
        is_back = faces_back[x, y, z]
        is_side = faces_u_minus[x, y, z] or faces_u_plus[x, y, z]

        # Map this voxel to a photo (u, v).
        # u (horizontal) from x: 0..sx → 0..W
        # v (vertical from top) from z: high-z = top of photo
        if is_back:
            # Mirror x across the symmetry plane so back inherits front colors
            mu = u_size - 1 - u_coord
            u_px = int(np.clip((mu + 0.5) / u_size * W, 0, W - 1))
        elif is_side and not is_front:
            # Side-facing: use the photo's edge column on the appropriate side
            if faces_u_minus[x, y, z]:
                u_px = 0
            else:
                u_px = W - 1
        else:
            u_px = int(np.clip((u_coord + 0.5) / u_size * W, 0, W - 1))
        v_px = int(np.clip((sz - 1 - z + 0.5) / sz * H, 0, H - 1))

        # Background-pixel filter: keep the existing color if photo here is bg
        rgb = photo[v_px, u_px]
        if alpha is not None and alpha[v_px, u_px] < 32:
            continue
        if _is_background(rgb) and not is_front:
            continue
        colors[x, y, z] = rgb

    return grid


def _pick_depths(occupied_idx: np.ndarray, sign: int, back_mode: str):
    """Choose which voxels along the depth axis to paint."""
    if back_mode == "through":
        return occupied_idx
    if back_mode == "front_only":
        return [occupied_idx[0] if sign > 0 else occupied_idx[-1]]
    if back_mode == "front_half":
        n = len(occupied_idx)
        half = max(1, n // 2)
        return occupied_idx[:half] if sign > 0 else occupied_idx[-half:]
    raise ValueError(f"unknown back_mode: {back_mode}")


def _kmeans_quantize(photo: np.ndarray, k: int, mask=None) -> np.ndarray:
    """Cluster pixels into k dominant colors. Replaces each pixel with its
    cluster centroid. Uses sklearn if available, else a fast NumPy fallback."""
    H, W, _ = photo.shape
    if mask is not None:
        sel = mask > 32
    else:
        # exclude near-pure background from the clustering sample
        mn = photo.min(axis=2); mx = photo.max(axis=2)
        sel = (mn <= 250) & (mx >= 8)
    pixels = photo[sel].astype(np.float32)
    if len(pixels) < k:
        return photo
    # Downsample to keep clustering fast on big photos
    if len(pixels) > 20000:
        idx = np.random.default_rng(0).choice(len(pixels), 20000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels
    centers = _kmeans_fit(sample, k, iters=12)
    flat = photo.reshape(-1, 3).astype(np.float32)
    d = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = d.argmin(axis=1)
    out = centers[nearest].astype(np.uint8).reshape(H, W, 3)
    return out


def _kmeans_fit(samples: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
    """Tiny NumPy k-means. Deterministic seed for reproducibility."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(samples), k, replace=False)
    centers = samples[idx].copy()
    for _ in range(iters):
        d = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = d.argmin(axis=1)
        new_centers = centers.copy()
        for ci in range(k):
            members = samples[assign == ci]
            if len(members) > 0:
                new_centers[ci] = members.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers


def _is_background(rgb) -> bool:
    """Pure white/black pixels are background after AI bg-removal."""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    if min(r, g, b) > 248:
        return True
    if max(r, g, b) < 8:
        return True
    return False


def _crop_to_foreground(photo: np.ndarray, alpha):
    if alpha is not None:
        mask = alpha > 32
    else:
        mn = photo.min(axis=2)
        mx = photo.max(axis=2)
        mask = (mn <= 248) & (mx >= 8)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return photo, alpha
    y0, y1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    x0, x1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    h, w = photo.shape[:2]
    if (y1 - y0) > 0.95 * h and (x1 - x0) > 0.95 * w:
        return photo, alpha
    cropped = photo[y0:y1, x0:x1]
    cropped_a = alpha[y0:y1, x0:x1] if alpha is not None else None
    return cropped, cropped_a


def _parse_axis(s: str) -> tuple[str, int]:
    s = s.strip().lower()
    sign = -1 if s.startswith("-") else 1
    letter = s.lstrip("+-")
    if letter not in {"x", "y", "z"}:
        raise ValueError(f"axis must be x/y/z (with optional sign): {s!r}")
    return letter, sign
