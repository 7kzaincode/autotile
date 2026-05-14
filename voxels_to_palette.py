"""
Stage 3-4: Snap per-voxel colors to the LEGO palette.

Uses CIELAB color distance instead of raw RGB so the quantization is
perceptually correct — RGB distance treats greens-yellows as much "closer" than
they look to a human eye. CIELAB distance lines up with how the brain compares
colors.

Loads `lego_palette.json`, converts both the photo voxel colors and palette
entries into LAB space (via sRGB → linear RGB → XYZ → LAB), then assigns each
voxel the nearest palette id. Returns a 3D int array of palette ids (0 where
unoccupied).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PALETTE_PATH = Path(__file__).parent / "lego_palette.json"


def load_palette(path: str | Path = PALETTE_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


AVAILABILITY_PENALTY = {"common": 0.0, "uncommon": 8.0, "rare": 25.0}


def quantize(colors: np.ndarray, occupancy: np.ndarray,
             palette: list[dict] | None = None,
             denoise: bool = True,
             max_colors: int = 0,
             pre_cluster: int = 0,
             availability_bias: bool = True,
             restrict_to_ids: list[int] | None = None) -> np.ndarray:
    """Map each occupied voxel's RGB to its nearest LEGO palette id (in LAB).

    `denoise=True` applies a 3D MODE filter on the palette ids AFTER quantization
    to kill salt-and-pepper speckle without blurring edges.

    `max_colors > 0` consolidates the result to that many distinct LEGO colors
    by keeping only the most-used ones and reassigning all other voxels to
    their perceptually-nearest kept color. Real LEGO sets use 5-10 colors —
    this option mimics that aesthetic.
    """
    palette = palette or load_palette()
    if restrict_to_ids:
        wanted = set(int(i) for i in restrict_to_ids)
        active = [p for p in palette if int(p["id"]) in wanted]
        if active:
            palette_for_match = active
        else:
            palette_for_match = palette
    else:
        palette_for_match = palette
    palette_rgb = np.array([p["rgb"] for p in palette_for_match], dtype=np.float32) / 255.0
    palette_lab = _rgb_to_lab(palette_rgb)
    palette_ids = np.array([p["id"] for p in palette_for_match], dtype=np.int32)

    ids = np.zeros(occupancy.shape, dtype=np.int32)
    occ_idx = np.argwhere(occupancy)
    if occ_idx.size == 0:
        return ids

    raw_samples = colors[occupancy]
    if pre_cluster > 0 and len(raw_samples) > pre_cluster:
        raw_samples = _kmeans_simplify(raw_samples, k=pre_cluster)
    samples_rgb = raw_samples.astype(np.float32) / 255.0
    samples_lab = _rgb_to_lab(samples_rgb)
    dists = ((samples_lab[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    if availability_bias:
        penalties = np.array([
            AVAILABILITY_PENALTY.get(p.get("availability", "uncommon"), 0.0)
            for p in palette_for_match
        ], dtype=np.float32) ** 2
        dists = dists + penalties[None, :]
    nearest = palette_ids[dists.argmin(axis=1)]
    for (ix, iy, iz), pid in zip(occ_idx, nearest):
        ids[ix, iy, iz] = pid

    if denoise:
        # Run 2 iterations for semantic-projected outputs (more speckle-prone
        # at voxel-grid scale than per-pixel projection)
        ids = _mode_filter(ids, occupancy, iterations=2)
    if max_colors > 0:
        ids = _consolidate(ids, occupancy, palette, palette_lab, palette_ids, max_colors)
    return ids


def region_color(colors: np.ndarray, occupancy: np.ndarray,
                 n_regions: int = 8, color_weight: float = 1.5) -> np.ndarray:
    """Spatial+color clustering: voxels in the same (xyz + rgb) cluster get
    the same color. Produces the "designed LEGO" look — one region per body
    part instead of per-voxel color sampling.

    `n_regions` is roughly how many distinct colored areas you want.
    `color_weight` trades off spatial vs color: higher = colors matter more,
    lower = spatial coherence matters more (bigger regions).
    """
    occ_idx = np.argwhere(occupancy)
    if occ_idx.size == 0:
        return colors
    rgb = colors[occupancy].astype(np.float32)
    # Normalize position to [0, 1] over grid dims; multiply color by weight
    shape = np.array(occupancy.shape, dtype=np.float32)
    pos = occ_idx.astype(np.float32) / shape
    features = np.concatenate([pos, (rgb / 255.0) * color_weight], axis=1)

    rng = np.random.default_rng(0)
    init = rng.choice(len(features), min(n_regions, len(features)), replace=False)
    centers = features[init].copy()
    for _ in range(15):
        d = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = d.argmin(axis=1)
        new_centers = centers.copy()
        for ci in range(len(centers)):
            members = features[assign == ci]
            if len(members) > 0:
                new_centers[ci] = members.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    d = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = d.argmin(axis=1)

    out = colors.copy()
    for cluster in range(len(centers)):
        mask = assign == cluster
        if not mask.any():
            continue
        # Median (not mean) color of cluster members — robust to outliers
        median_rgb = np.median(rgb[mask], axis=0).astype(np.uint8)
        idxs = occ_idx[mask]
        for (x, y, z) in idxs:
            out[x, y, z] = median_rgb
    return out


def mirror_occupancy(occupancy: np.ndarray, colors: np.ndarray,
                     axis: str = "x") -> tuple[np.ndarray, np.ndarray]:
    """Take the union of occupancy across the mirror plane. Fills in missing
    limbs (e.g., TRELLIS gave us a bunny with only a left foot — this clones
    the foot to the right side using the same color).

    Returns (new_occupancy, new_colors).
    """
    occ = occupancy.copy()
    col = colors.copy()
    sx, sy, sz = occ.shape
    if axis != "x":
        return occ, col  # only x-axis mirror supported for now
    for x in range(sx // 2):
        mx = sx - 1 - x
        for y in range(sy):
            for z in range(sz):
                a, b = occ[x, y, z], occ[mx, y, z]
                if a and not b:
                    occ[mx, y, z] = True
                    col[mx, y, z] = col[x, y, z]
                elif b and not a:
                    occ[x, y, z] = True
                    col[x, y, z] = col[mx, y, z]
    return occ, col


def photo_palette(photo_path, palette: list[dict], n_colors: int = 6,
                  min_size_frac: float = 0.01) -> list[int]:
    """Extract dominant colors from the photo and snap each to the nearest LEGO
    palette entry (CIELAB distance). Returns a list of palette IDs.

    Background pixels (near-white / near-black / transparent) are excluded.
    Clusters smaller than `min_size_frac` of foreground pixels are dropped
    (e.g. tiny anti-aliased boundary clusters).

    The returned IDs make a SUBSET that can be passed to quantize() so the
    LEGO output is restricted to colors actually present in the photo.
    """
    from PIL import Image
    img = Image.open(photo_path)
    has_alpha = img.mode in ("RGBA", "LA")
    if has_alpha:
        rgba = np.asarray(img.convert("RGBA"))
        alpha = rgba[..., 3]
        rgb = rgba[..., :3]
        mask = alpha > 32
    else:
        rgb = np.asarray(img.convert("RGB"))
        mn = rgb.min(axis=2); mx = rgb.max(axis=2)
        mask = (mn <= 248) & (mx >= 8)
    if not mask.any():
        return [p["id"] for p in palette[:n_colors]]
    pixels = rgb[mask].astype(np.float32)

    # k-means in RGB (cheap and faithful for dominant color sampling)
    rng = np.random.default_rng(0)
    if len(pixels) > 30000:
        idx = rng.choice(len(pixels), 30000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels
    k = min(n_colors, max(2, len(sample) // 100))
    init = rng.choice(len(sample), k, replace=False)
    centers = sample[init].copy()
    for _ in range(20):
        d = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = d.argmin(axis=1)
        new_centers = centers.copy()
        for ci in range(k):
            m = sample[assign == ci]
            if len(m) > 0:
                new_centers[ci] = m.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    # Drop tiny clusters
    d = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = d.argmin(axis=1)
    counts = np.bincount(assign, minlength=k)
    keep_mask = counts >= max(1, int(min_size_frac * len(sample)))
    centers = centers[keep_mask]
    if len(centers) == 0:
        return [p["id"] for p in palette[:n_colors]]

    # Snap each center to its nearest LEGO palette entry in CIELAB
    palette_rgb = np.array([p["rgb"] for p in palette], dtype=np.float32) / 255.0
    palette_lab = _rgb_to_lab(palette_rgb)
    palette_ids = np.array([p["id"] for p in palette], dtype=np.int32)
    centers_lab = _rgb_to_lab(centers / 255.0)
    chosen = []
    seen = set()
    for c in centers_lab:
        d = ((palette_lab - c) ** 2).sum(axis=1)
        # Prefer common colors when distances are similar
        avail = np.array([
            AVAILABILITY_PENALTY.get(p.get("availability", "uncommon"), 0.0)
            for p in palette
        ], dtype=np.float32) ** 2
        order = np.argsort(d + avail * 0.3)  # mild bias toward common
        for idx in order:
            pid = int(palette_ids[idx])
            if pid not in seen:
                seen.add(pid)
                chosen.append(pid)
                break

    print(f"[photo_palette] derived {len(chosen)} colors: "
          f"{[next(p['name'] for p in palette if p['id'] == cid) for cid in chosen]}")
    return chosen


def mirror_palette_ids(ids: np.ndarray, palette: list[dict], axis: str = "x") -> np.ndarray:
    """Mirror the QUANTIZED palette ids across the model so each left/right
    pair shares a single, more-saturated id. Runs AFTER quantization so it
    survives downstream region_color clustering and produces strictly
    symmetric output. Returns a new array.
    """
    out = ids.copy()
    sx, sy, sz = out.shape
    if axis != "x":
        return out
    sat = {p["id"]: max(p["rgb"]) - min(p["rgb"]) for p in palette}
    for x in range(sx // 2):
        mx = sx - 1 - x
        for y in range(sy):
            for z in range(sz):
                a, b = int(out[x, y, z]), int(out[mx, y, z])
                if a == 0 or b == 0 or a == b:
                    continue
                if sat.get(a, 0) >= sat.get(b, 0):
                    out[mx, y, z] = a
                else:
                    out[x, y, z] = b
    return out


def mirror_colors(colors: np.ndarray, occupancy: np.ndarray, axis: str = "x") -> np.ndarray:
    """For each voxel column along the mirror axis, take the most-saturated /
    distinct color across the symmetric pair and use it for both. Useful when
    a textured 3D model has a "default white" untextured side (TRELLIS hallucination)
    and a properly-textured side — this gives the well-textured side's colors
    to both halves.

    Returns a new colors array (occupancy is unchanged).
    """
    out = colors.copy()
    sx, sy, sz, _ = colors.shape
    if axis == "x":
        for x in range(sx // 2):
            mx = sx - 1 - x
            for y in range(sy):
                for z in range(sz):
                    if not (occupancy[x, y, z] and occupancy[mx, y, z]):
                        continue
                    c1, c2 = colors[x, y, z], colors[mx, y, z]
                    chosen = _pick_more_distinct(c1, c2)
                    out[x, y, z] = chosen
                    out[mx, y, z] = chosen
    return out


def _pick_more_distinct(c1, c2):
    """Pick the color farther from neutral gray (more saturated)."""
    def score(c):
        return max(c) - min(c)
    return c1 if score(c1) >= score(c2) else c2


def _kmeans_simplify(samples: np.ndarray, k: int, iters: int = 15) -> np.ndarray:
    """Cluster voxel RGB into k dominant colors and replace each sample with
    its cluster centroid. Same idea as posterizing — gets rid of mottled
    gradients that the LEGO palette would otherwise split into many SKUs."""
    rng = np.random.default_rng(0)
    n = len(samples)
    idx = rng.choice(n, k, replace=False)
    centers = samples[idx].astype(np.float32)
    samples_f = samples.astype(np.float32)
    for _ in range(iters):
        d = ((samples_f[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = d.argmin(axis=1)
        new_centers = centers.copy()
        for ci in range(k):
            members = samples_f[assign == ci]
            if len(members) > 0:
                new_centers[ci] = members.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    d = ((samples_f[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = d.argmin(axis=1)
    return np.clip(centers[assign], 0, 255).astype(np.uint8)


def _consolidate(ids: np.ndarray, occupancy: np.ndarray,
                 palette: list[dict],
                 palette_lab: np.ndarray, palette_ids: np.ndarray,
                 max_colors: int) -> np.ndarray:
    """Keep the top-N most-used palette ids; reassign every other voxel to
    its perceptually-nearest KEPT color. Mimics how a LEGO designer picks a
    small palette for a model and uses it everywhere."""
    from collections import Counter
    counts = Counter(ids[occupancy].tolist())
    counts.pop(0, None)
    if len(counts) <= max_colors:
        return ids
    kept_ids = [cid for cid, _ in counts.most_common(max_colors)]
    kept_set = set(kept_ids)

    # Build LAB centers for the kept ids
    id_to_idx = {int(pid): i for i, pid in enumerate(palette_ids.tolist())}
    kept_lab = np.array([palette_lab[id_to_idx[k]] for k in kept_ids])
    kept_arr = np.array(kept_ids, dtype=np.int32)

    out = ids.copy()
    to_fix = np.argwhere(occupancy & ~np.isin(ids, list(kept_set)))
    if to_fix.size == 0:
        return out
    for (ix, iy, iz) in to_fix:
        # find this voxel's original LAB color (= the LAB of its current palette id)
        cur = int(ids[ix, iy, iz])
        cur_lab = palette_lab[id_to_idx[cur]]
        d = ((kept_lab - cur_lab) ** 2).sum(axis=1)
        out[ix, iy, iz] = kept_arr[int(d.argmin())]
    return out


def _mode_filter(ids: np.ndarray, occupancy: np.ndarray,
                 iterations: int = 1, aggressiveness: float = 0.6) -> np.ndarray:
    """3x3x3 mode filter: any voxel that's the minority among its neighbors
    gets reassigned to the majority. Preserves edges (no blending) while
    killing isolated speckle.

    Run multiple iterations for stronger cleanup of speckled outputs.
    """
    from collections import Counter
    sx, sy, sz = ids.shape
    cur = ids.copy()
    for _ in range(max(1, iterations)):
        out = cur.copy()
        changed = False
        for (ix, iy, iz) in np.argwhere(occupancy):
            x0, x1 = max(0, ix - 1), min(sx, ix + 2)
            y0, y1 = max(0, iy - 1), min(sy, iy + 2)
            z0, z1 = max(0, iz - 1), min(sz, iz + 2)
            nbr = cur[x0:x1, y0:y1, z0:z1].ravel()
            nbr = nbr[nbr > 0]
            if nbr.size < 6:
                continue
            c = Counter(nbr.tolist())
            most_common, count = c.most_common(1)[0]
            this_count = c.get(int(cur[ix, iy, iz]), 0)
            if count >= int(aggressiveness * nbr.size) and this_count <= count // 2:
                if int(cur[ix, iy, iz]) != most_common:
                    out[ix, iy, iz] = most_common
                    changed = True
        if not changed:
            break
        cur = out
    return cur


# --- color space conversions (vectorized, NumPy) ---

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    threshold = 0.04045
    return np.where(c <= threshold, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB (0..1) -> XYZ (D65). Input shape (N, 3)."""
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    return rgb @ M.T


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    # D65 reference white
    ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xyz = xyz / ref
    eps = (6 / 29) ** 3
    kappa = (29 / 6) ** 2 / 3
    f = np.where(xyz > eps, np.cbrt(xyz), kappa * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..1) -> CIELAB. Input shape (N, 3)."""
    linear = _srgb_to_linear(rgb)
    xyz = _rgb_to_xyz(linear)
    return _xyz_to_lab(xyz)
