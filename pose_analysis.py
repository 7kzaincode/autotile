"""Lightweight photo pose analysis for pipeline safety decisions.

The 3D and LEGO stages make very different assumptions for front-facing pets
than for side-profile pets. This module deliberately stays heuristic and fast:
it reads the foreground silhouette and returns enough metadata for the server
to disable risky front-face behaviors when the uploaded animal is wide/profile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


PET_HINTS = {
    "pet", "animal", "dog", "puppy", "cat", "kitten", "rabbit", "bunny",
    "hamster", "guinea", "horse", "bird",
}


def is_pet_subject(subject: str | None) -> bool:
    text = (subject or "").strip().lower()
    if not text:
        return False
    return any(hint in text for hint in PET_HINTS)


def analyze_photo_pose(photo_path: str | Path, subject: str | None = None) -> dict:
    """Return coarse pose metadata from the foreground silhouette.

    The main decision currently needed is whether the subject is a side-profile
    pet. Side-profile animals should not be mirrored left/right; face accents
    are still allowed because the face-map stage now anchors details to a
    detected surface plane instead of blindly assuming a centered front face.
    """
    photo_path = Path(photo_path)
    img = Image.open(photo_path)
    W, H = img.size
    mask = _foreground_mask(img)
    if not mask.any():
        return {
            "pose": "unknown",
            "confidence": 0.0,
            "bbox": [0, 0, W, H],
            "bbox_aspect": round(W / max(1, H), 3),
            "is_side_profile": False,
            "safe_mirror": True,
            "safe_face_accents": True,
            "reason": "no foreground mask",
        }

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - np.argmax(cols[::-1]))
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    aspect = bw / bh
    width_frac = bw / max(1, W)
    height_frac = bh / max(1, H)

    pet = is_pet_subject(subject)
    # Front-view pets like the bunny are tall/narrow. Full-body pet photos
    # become wider even when the head faces the camera; that is still enough
    # reason to disable symmetry because one-sided patches and legs matter.
    side_score = 0.0
    if pet:
        side_score += max(0.0, min(1.0, (aspect - 0.58) / 0.34)) * 0.70
        side_score += max(0.0, min(1.0, (width_frac - 0.48) / 0.32)) * 0.30
    is_side = pet and side_score >= 0.30
    pose = "side_profile" if is_side else ("front_or_three_quarter" if pet else "unknown")
    confidence = side_score if is_side else (1.0 - min(1.0, side_score))

    return {
        "pose": pose,
        "confidence": round(float(confidence), 3),
        "bbox": [x0, y0, x1, y1],
        "bbox_aspect": round(float(aspect), 3),
        "width_frac": round(float(width_frac), 3),
        "height_frac": round(float(height_frac), 3),
        "is_side_profile": bool(is_side),
        "safe_mirror": not is_side,
        "safe_face_accents": True,
        "reason": (
            "wide pet silhouette"
            if is_side else
            "not a wide side-profile pet"
        ),
    }


def _foreground_mask(img: Image.Image) -> np.ndarray:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha = np.asarray(img.convert("RGBA"))[..., 3]
        return alpha > 16

    arr = np.asarray(img.convert("RGB"))
    # Fallback for non-alpha images: treat near-white and near-black as likely
    # background, which covers the common clean studio and black-stock cases.
    mn = arr.min(axis=2)
    mx = arr.max(axis=2)
    return (mn < 245) & (mx > 12)
