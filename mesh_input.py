"""Prepare a mesh-generation-only image from the cleaned photo.

Image-to-3D models are very literal about high-frequency detail: fur strands,
thin whiskers, watermark edges, and ragged alpha cutouts can become real mesh
geometry. The color pipeline should still use the original bg-removed photo,
but the mesh model usually benefits from a softer silhouette-first input.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def prepare_mesh_input(
    photo_path: str | Path,
    out_dir: str | Path,
    subject: str | None = None,
    *,
    enabled: bool = True,
) -> Path:
    """Return an image path suitable for image-to-3D mesh generation.

    The output keeps alpha, smooths the subject texture, reduces color noise,
    and erodes/re-expands the alpha mask enough to remove hairlike edge wisps.
    """
    photo_path = Path(photo_path)
    if not enabled:
        return photo_path

    img = Image.open(photo_path).convert("RGBA")
    rgb = img.convert("RGB")
    alpha = img.getchannel("A")

    # Remove one-pixel whiskers/fur/watermark scratches from the silhouette,
    # then gently grow the mask back so the main body does not shrink.
    alpha = alpha.filter(ImageFilter.MinFilter(3))
    alpha = alpha.filter(ImageFilter.MaxFilter(3))
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.4))

    # Smooth high-frequency texture before the 3D model sees it. This keeps
    # broad shape cues while making fur/watermark lettering much less geometry-
    # worthy. The original photo is still used later for color projection.
    rgb = rgb.filter(ImageFilter.MedianFilter(size=5))
    rgb = rgb.filter(ImageFilter.GaussianBlur(radius=1.2))
    rgb = ImageEnhance.Contrast(rgb).enhance(0.82)
    rgb = ImageEnhance.Sharpness(rgb).enhance(0.35)
    rgb = ImageOps.posterize(rgb, bits=5)

    out = Image.merge("RGBA", (*rgb.split(), alpha))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{photo_path.stem}_meshinput.png"
    out.save(out_path)
    return out_path
