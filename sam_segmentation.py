"""
SAM 2 — local pixel-perfect segmentation.

Given a photo and a set of normalized bboxes (typically from GPT vision),
returns one binary mask per box: the actual SHAPE of the object inside the
box, not just the rectangle. Replaces the bbox-painting used by
`project_semantic_gpt` so that ears get ear-shaped masks, head gets a
head-shaped mask, etc.

Runs locally via HuggingFace transformers. Free, no quota. Mac MPS / CUDA /
CPU auto-selected. Lazy model load — first call is ~5-10s (load+download
on cold start), subsequent calls ~0.5-2s on Mac MPS.

Falls back gracefully (returns None) on any failure — caller should treat
that as a signal to use the rectangle-bbox method instead.

Cache: by SHA-256 of (photo_bytes + box-list-json), as .npz, so the same
photo + region set never re-runs inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


CACHE_DIR = Path(__file__).parent / "sam_cache"
DEFAULT_MODEL = "facebook/sam2-hiera-small"

_MODEL = None
_PROCESSOR = None
_DEVICE = None
_LOADED_ID = None


def _pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(model_id: str = DEFAULT_MODEL):
    """Lazy singleton. Returns (model, processor, device) or (None, None, None)."""
    global _MODEL, _PROCESSOR, _DEVICE, _LOADED_ID
    if _MODEL is not None and _LOADED_ID == model_id:
        return _MODEL, _PROCESSOR, _DEVICE
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
        from transformers.utils import logging as hf_logging
    except ImportError as e:
        print(f"[sam] transformers/torch not installed yet ({e}); falling back")
        return None, None, None

    device = _pick_device()
    print(f"[sam] loading {model_id} on {device} (first call may download weights)…")
    old_verbosity = None
    try:
        # HF's public SAM2 image checkpoint currently carries `sam2_video`
        # metadata, which makes Transformers emit a scary compatibility warning
        # even though the image segmentation path below is the supported one for
        # box-prompt masks. Keep real load/inference failures visible, but hide
        # that known noisy loader warning from the app log.
        old_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        proc = Sam2Processor.from_pretrained(model_id)
        model = Sam2Model.from_pretrained(model_id).to(device).eval()
    except Exception as e:
        print(f"[sam] failed to load {model_id}: {e}")
        return None, None, None
    finally:
        if old_verbosity is not None:
            hf_logging.set_verbosity(old_verbosity)
    _PROCESSOR = proc
    _MODEL = model
    _DEVICE = device
    _LOADED_ID = model_id
    print(f"[sam] ready ({model_id} on {device})")
    return model, proc, device


def _photo_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 14), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _boxes_hash(boxes) -> str:
    return hashlib.sha256(
        json.dumps(boxes, sort_keys=True, default=float).encode()
    ).hexdigest()[:12]


def segment_with_boxes(
    photo_path: str | Path,
    boxes_normalized: list,
    force_refresh: bool = False,
    model_id: str = DEFAULT_MODEL,
) -> Optional[list[np.ndarray]]:
    """Run SAM 2 with one bbox prompt per region.

    Args:
        photo_path: path to the photo (any PIL-readable format).
        boxes_normalized: list of (x0, y0, x1, y1) in [0,1].
        force_refresh: bypass disk cache.
        model_id: HF model id (default: facebook/sam2-hiera-small).

    Returns:
        list of binary masks (uint8 ndarrays, shape HxW, values 0/1), one per
        input box. None on any failure (caller should fall back).
    """
    photo_path = Path(photo_path)
    if not photo_path.exists():
        return None
    if not boxes_normalized:
        return []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ph = _photo_hash(photo_path)
    bh = _boxes_hash([list(b) for b in boxes_normalized])
    cache_file = CACHE_DIR / f"{ph}_{bh}.npz"
    if cache_file.exists() and not force_refresh:
        try:
            data = np.load(cache_file)
            out = [data[f"m{i}"] for i in range(len(boxes_normalized))]
            print(f"[sam] cache hit ({cache_file.name})")
            return out
        except Exception:
            pass

    model, proc, device = _load_model(model_id)
    if model is None:
        return None

    try:
        import torch
        from PIL import Image
        img = Image.open(photo_path).convert("RGB")
        W, H = img.size

        # Normalized → pixel boxes, clipped to image
        px_boxes: list[list[float]] = []
        for (x0, y0, x1, y1) in boxes_normalized:
            px_boxes.append([
                max(0.0, float(x0) * W),
                max(0.0, float(y0) * H),
                min(float(W), float(x1) * W),
                min(float(H), float(y1) * H),
            ])

        # SAM 2 processor: input_boxes is [batch, num_boxes, 4]
        inputs = proc(
            images=img,
            input_boxes=[px_boxes],
            return_tensors="pt",
        )
        # Move tensors to device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        # pred_masks shape: [batch, num_boxes, num_masks, h, w]
        original_sizes = inputs.get("original_sizes")
        if original_sizes is None:
            original_sizes = torch.tensor([[H, W]])
        masks = proc.post_process_masks(
            outputs.pred_masks.cpu(),
            original_sizes.cpu(),
        )[0]
        # masks: [num_boxes, num_masks (=1 since multimask_output=False), H, W]
        # Could be torch tensor or np array depending on version; normalize.
        if hasattr(masks, "numpy"):
            masks_np = masks.numpy()
        else:
            masks_np = np.asarray(masks)

        out: list[np.ndarray] = []
        for i in range(masks_np.shape[0]):
            m = masks_np[i]
            # Reduce mask-dim if present
            while m.ndim > 2:
                m = m[0]
            out.append((m > 0).astype(np.uint8))

        try:
            np.savez_compressed(cache_file, **{f"m{i}": m for i, m in enumerate(out)})
            cov = [round(float(m.mean()), 3) for m in out]
            print(f"[sam] segmented {len(out)} regions, coverage={cov} → {cache_file.name}")
        except Exception as e:
            print(f"[sam] cache write skipped: {e}")
        return out
    except Exception as e:
        import traceback
        print(f"[sam] inference failed: {e}")
        traceback.print_exc()
        return None


def is_available() -> bool:
    """Cheap check: are transformers + torch importable? Doesn't load weights."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return hasattr(__import__("transformers"), "Sam2Model")
    except ImportError:
        return False
