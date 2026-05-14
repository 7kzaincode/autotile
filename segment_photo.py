"""
Semantic-ish photo segmentation (local, free).

Why this exists:
  Sampling per-voxel from the photo gives noisy color gradients — every voxel
  ends up slightly different and the LEGO output looks like a Minecraft blob.
  Real LEGO models have BIG UNIFORM REGIONS — body is one color, head is
  another, eyes are crisp accents.

  We achieve this by SEGMENTING the photo FIRST (not the voxels), then
  projecting region IDs onto voxels. Each region maps to ONE LEGO color.

Algorithm:
  1. Spatial+color k-means on the photo's foreground pixels.
     - Spatial weight pulls neighboring pixels into the same cluster
       (anatomy-aware: head pixels cluster together regardless of slight color variation)
     - Color weight separates regions with different hues.
  2. Detect small dark spots in the upper foreground (eyes) and assign them
     their own region — protects features that k-means would otherwise absorb.
  3. Return: a per-pixel region label map + dominant color per region.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def segment_photo(
    photo_path: str | Path,
    n_regions: int = 8,
    spatial_weight: float = 0.6,
    detect_eyes: bool = True,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Segment the photo into n_regions + optional eye regions.

    Returns:
        label_map  : H×W int array, region ID per pixel (-1 = background)
        regions    : dict[region_id] = {rgb, size, kind, centroid_xy}
                     kind ∈ {"body", "eye"}
        photo_rgb  : the (auto-cropped) photo as an ndarray, useful downstream
    """
    img = Image.open(photo_path)
    has_alpha = img.mode in ("RGBA", "LA")
    if has_alpha:
        rgba = np.asarray(img.convert("RGBA"))
        rgb = rgba[..., :3].copy()
        alpha = rgba[..., 3]
        fg_mask = alpha > 32
    else:
        rgb = np.asarray(img.convert("RGB")).copy()
        mn = rgb.min(axis=2)
        mx = rgb.max(axis=2)
        fg_mask = (mn <= 248) & (mx >= 8)

    H, W = rgb.shape[:2]

    # Auto-crop to foreground bbox so spatial coordinates are meaningful
    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)
    if not rows.any() or not cols.any():
        return -np.ones((H, W), dtype=np.int32), {}, rgb
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    rgb_crop = rgb[y0:y1, x0:x1]
    fg_crop = fg_mask[y0:y1, x0:x1]
    Hc, Wc = rgb_crop.shape[:2]

    # Build 5D features (y, x, r, g, b) on foreground pixels only
    yy, xx = np.mgrid[0:Hc, 0:Wc]
    pixel_idx = np.argwhere(fg_crop)
    if len(pixel_idx) < n_regions * 4:
        # Not enough foreground; bail with a single region
        out = -np.ones((H, W), dtype=np.int32)
        return out, {}, rgb

    ys = pixel_idx[:, 0].astype(np.float32) / max(1, Hc)
    xs = pixel_idx[:, 1].astype(np.float32) / max(1, Wc)
    pix_rgb = rgb_crop[fg_crop].astype(np.float32) / 255.0

    features = np.stack([
        ys * spatial_weight,
        xs * spatial_weight,
        pix_rgb[:, 0],
        pix_rgb[:, 1],
        pix_rgb[:, 2],
    ], axis=-1)

    # K-means
    rng = np.random.default_rng(0)
    n_init = min(n_regions, len(features))
    init = rng.choice(len(features), n_init, replace=False)
    centers = features[init].copy()
    for _ in range(20):
        d = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = d.argmin(axis=1)
        new_centers = centers.copy()
        for ci in range(n_init):
            m = features[assign == ci]
            if len(m) > 0:
                new_centers[ci] = m.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    d = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = d.argmin(axis=1)

    # Build label map in cropped frame, then place back in full-frame
    label_crop = -np.ones((Hc, Wc), dtype=np.int32)
    label_crop[pixel_idx[:, 0], pixel_idx[:, 1]] = assign

    # CLEANUP: drop tiny disconnected components per label. Any component
    # smaller than 0.5% of foreground gets reassigned to its surround majority.
    # This kills isolated "spot" clusters that produce speckle when projected.
    label_crop = _drop_tiny_clusters(label_crop, min_frac=0.005)

    label_map = -np.ones((H, W), dtype=np.int32)
    label_map[y0:y1, x0:x1] = label_crop

    regions: dict[int, dict] = {}
    for r in range(n_init):
        mask = label_map == r
        if not mask.any():
            continue
        pixels = rgb[mask].astype(np.float32)
        # Median is more robust than mean for region dominant color
        med = np.median(pixels, axis=0).astype(np.uint8)
        ys_r, xs_r = np.where(mask)
        regions[r] = {
            "rgb":      tuple(int(v) for v in med),
            "size":     int(mask.sum()),
            "kind":     "body",
            "centroid": (float(ys_r.mean()), float(xs_r.mean())),
        }

    if detect_eyes:
        eye_regions = _detect_eye_regions(rgb, fg_mask, y0, y1)
        next_id = (max(regions) if regions else -1) + 1
        for ex, ey, r_pixels in eye_regions:
            mask = np.zeros_like(label_map, dtype=bool)
            for (py, px) in r_pixels:
                if 0 <= py < H and 0 <= px < W:
                    mask[py, px] = True
            label_map[mask] = next_id
            regions[next_id] = {
                "rgb":      (0, 0, 0),
                "size":     int(mask.sum()),
                "kind":     "eye",
                "centroid": (float(ey), float(ex)),
            }
            next_id += 1

    return label_map, regions, rgb


def _label_majority_filter(labels: np.ndarray, radius: int = 2) -> np.ndarray:
    """Replace each pixel's label with the majority label in its
    (2*radius+1)² window. Background (-1) is preserved."""
    H, W = labels.shape
    out = labels.copy()
    pad = np.pad(labels, radius, mode="edge")
    for y in range(H):
        for x in range(W):
            if labels[y, x] < 0:
                continue
            window = pad[y:y + 2 * radius + 1, x:x + 2 * radius + 1].ravel()
            window = window[window >= 0]
            if window.size == 0:
                continue
            vals, counts = np.unique(window, return_counts=True)
            out[y, x] = int(vals[counts.argmax()])
    return out


def _drop_tiny_clusters(labels: np.ndarray, min_frac: float = 0.005) -> np.ndarray:
    """For each label, find its connected components. Drop components
    smaller than min_frac of the foreground; their pixels inherit the
    majority label from their surround."""
    try:
        from scipy.ndimage import label as cc_label
    except ImportError:
        return labels
    fg = labels >= 0
    total_fg = int(fg.sum())
    if total_fg == 0:
        return labels
    threshold = max(8, int(total_fg * min_frac))
    out = labels.copy()
    for lid in np.unique(labels):
        if lid < 0:
            continue
        mask = labels == lid
        components, n = cc_label(mask)
        if n <= 1:
            continue
        for cid in range(1, n + 1):
            comp_mask = components == cid
            if comp_mask.sum() < threshold:
                out[comp_mask] = -2   # mark "needs reassign"
    # Reassign marked pixels via majority of their non-marked neighbors
    if (out == -2).any():
        marked = np.argwhere(out == -2)
        # One-pass nearest-neighbor majority (good enough)
        H, W = out.shape
        for (y, x) in marked:
            best, best_count = None, 0
            for r in (1, 2, 3, 5):
                y0, y1 = max(0, y - r), min(H, y + r + 1)
                x0, x1 = max(0, x - r), min(W, x + r + 1)
                neighborhood = out[y0:y1, x0:x1].ravel()
                neighborhood = neighborhood[neighborhood >= 0]
                if neighborhood.size == 0:
                    continue
                vals, counts = np.unique(neighborhood, return_counts=True)
                idx = counts.argmax()
                if counts[idx] > best_count:
                    best, best_count = int(vals[idx]), int(counts[idx])
                if best_count >= 4:
                    break
            out[y, x] = best if best is not None else -1
    return out


def _detect_eye_regions(rgb: np.ndarray, fg_mask: np.ndarray,
                        y0: int, y1: int) -> list[tuple[int, int, list]]:
    """Find pairs of dark blobs in the upper portion of the foreground that
    look like eyes. Returns (cx, cy, pixel_list) per detected eye."""
    bright = rgb.astype(np.float32).mean(axis=2)
    upper_cut = y0 + int((y1 - y0) * 0.55)
    fg_upper = fg_mask.copy()
    fg_upper[upper_cut:, :] = False
    if not fg_upper.any():
        return []
    thr = np.percentile(bright[fg_upper], 18)
    dark = (bright < thr) & fg_upper
    try:
        from scipy.ndimage import label, center_of_mass
    except ImportError:
        return []
    labeled, n = label(dark)
    if n == 0:
        return []
    H, W = rgb.shape[:2]

    candidates: list[tuple[int, int, int, list]] = []
    for cid in range(1, n + 1):
        mask = labeled == cid
        s = int(mask.sum())
        if s < 4 or s > 400:
            continue
        cy, cx = center_of_mass(mask)
        candidates.append((int(cx), int(cy), s, list(zip(*np.where(mask)))))

    if len(candidates) < 2:
        return []
    bbox_h = max(1, y1 - y0)
    cols_fg = np.any(fg_mask, axis=0)
    if not cols_fg.any():
        return []
    bbox_w = max(1, int(len(cols_fg) - np.argmax(cols_fg[::-1])
                        - np.argmax(cols_fg)))

    pairs = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            cx_i, cy_i, _, _ = candidates[i]
            cx_j, cy_j, _, _ = candidates[j]
            dy = abs(cy_i - cy_j)
            dx = abs(cx_i - cx_j)
            if dy < 0.10 * bbox_h and 0.05 * bbox_w < dx < 0.6 * bbox_w:
                score = -dy - abs(dx - 0.2 * bbox_w)
                pairs.append((score, i, j))
    if not pairs:
        return []
    pairs.sort(reverse=True)
    _, i, j = pairs[0]
    out = []
    for k in (i, j):
        cx, cy, _, px_list = candidates[k]
        out.append((cx, cy, px_list))
    return out
