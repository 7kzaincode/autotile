"""
LEGO-ify an input photo into flat color regions.

Originally this tried to call SDXL img2img on free HF Spaces. Those Spaces
are broken / unreliable, so we do it locally with PIL + numpy. Result:

  - Edge-preserving smoothing (median filter — like a poor man's bilateral)
  - K-means cluster colors into N distinct regions
  - Optional vibrance boost (saturation) so colors land on more distinct
    LEGO palette entries downstream

Takes ~0.5-2s for a typical photo. No HF / Replicate calls.

Presets:
  lego    : aggressive (4 colors, heavy smoothing) — most LEGO-like
  toy     : 6 colors, moderate smoothing — looks like a rendered toy
  cartoon : 8 colors, light smoothing — preserves more detail
  render  : 12 colors, edge-preserve smoothing — cinematic
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


PRESETS = {
    "lego":    {"n_colors": 4,  "smooth_radius": 4, "saturation": 1.4, "posterize_bits": 4},
    "toy":     {"n_colors": 6,  "smooth_radius": 3, "saturation": 1.2, "posterize_bits": 5},
    "cartoon": {"n_colors": 8,  "smooth_radius": 2, "saturation": 1.3, "posterize_bits": 5},
    "render":  {"n_colors": 12, "smooth_radius": 2, "saturation": 1.1, "posterize_bits": 6},
}


def stylize_photo(
    photo_path: str | Path,
    out_dir: str | Path = "test_photos",
    preset: str = "lego",
    strength: float = 0.5,
    **_ignored,
) -> Path:
    """Local LEGO-style stylization. `strength` scales smoothing + color reduction."""
    photo_path = Path(photo_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PRESETS.get(preset, PRESETS["lego"]).copy()

    # strength 0 = no change, 1 = full preset. Scale the aggressive settings.
    s = max(0.0, min(1.0, strength)) * 2.0  # presets are tuned for strength=0.5
    cfg["smooth_radius"] = max(1, int(cfg["smooth_radius"] * s + 0.5))
    cfg["n_colors"]      = max(2, int(round(cfg["n_colors"] / max(0.5, s))))

    print(f"[stylize] local preset={preset!r}  cfg={cfg}")

    img = Image.open(photo_path)
    has_alpha = img.mode in ("RGBA", "LA")
    if has_alpha:
        rgba = np.asarray(img.convert("RGBA"))
        alpha = rgba[..., 3]
        img = Image.fromarray(rgba[..., :3], mode="RGB")
    else:
        img = img.convert("RGB")
        alpha = None

    # Saturate FIRST so the smoothing preserves richer color
    if cfg["saturation"] != 1.0:
        img = ImageEnhance.Color(img).enhance(cfg["saturation"])
    # Edge-preserving smoothing via median filter
    if cfg["smooth_radius"] > 0:
        img = img.filter(ImageFilter.MedianFilter(size=cfg["smooth_radius"] * 2 + 1))
    # Posterize to gently band colors
    if cfg.get("posterize_bits"):
        from PIL import ImageOps
        img = ImageOps.posterize(img, int(cfg["posterize_bits"]))

    # K-means quantize to n_colors dominant colors
    arr = np.asarray(img).copy()
    arr = _kmeans_quantize(arr, k=cfg["n_colors"], mask=alpha)

    out = Image.fromarray(arr, mode="RGB")
    if alpha is not None:
        out = out.convert("RGBA")
        out.putalpha(Image.fromarray(alpha, mode="L"))

    out_path = out_dir / f"{photo_path.stem}_stylized.png"
    out.save(out_path)
    print(f"[stylize] wrote {out_path}")
    return out_path


def _kmeans_quantize(img: np.ndarray, k: int, mask=None) -> np.ndarray:
    H, W, _ = img.shape
    if mask is not None:
        sel = mask > 32
    else:
        mn = img.min(axis=2); mx = img.max(axis=2)
        sel = (mn <= 248) & (mx >= 8)
    pixels = img[sel].astype(np.float32)
    if len(pixels) < k:
        return img
    if len(pixels) > 30000:
        idx = np.random.default_rng(0).choice(len(pixels), 30000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels
    centers = _kmeans_fit(sample, k, iters=12)
    flat = img.reshape(-1, 3).astype(np.float32)
    d = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = d.argmin(axis=1)
    return centers[assign].astype(np.uint8).reshape(H, W, 3)


def _kmeans_fit(samples: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(samples), k, replace=False)
    centers = samples[idx].astype(np.float32)
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
