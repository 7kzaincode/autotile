"""
Background removal via `briaai/BRIA-RMBG-2.0`.

TRELLIS and other photo-to-3D models generate much cleaner output when the
input photo has a transparent / pure-white background. Real photos often have
busy backgrounds that the AI mistakes for part of the object.

This module runs the photo through BRIA-RMBG, saves the result with alpha,
and returns the new path. Takes ~3-8s per call on CPU.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


RMBG_SPACES = [
    # (space_id, api_endpoint, result_index, kwargs_key)
    # Tried in order; first one that succeeds wins.
    ("briaai/BRIA-RMBG-2.0",      "/image",  1,    "image"),
    ("briaai/BRIA-RMBG-1.4",      "/predict", None, None),     # positional
    ("not-lain/background-removal", "/png",   None, None),
    ("not-lain/background-removal", "/image", 1,    "image"),
]


def _coerce(obj, space_url: str | None = None) -> Path | None:
    if obj is None:
        return None
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        return p if p.exists() else None
    if isinstance(obj, dict):
        url = obj.get("url") or obj.get("href")
        if isinstance(url, str) and url.startswith("http"):
            import tempfile
            ext = Path(url.split("?")[0]).suffix or ".png"
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            for ch in r.iter_content(1 << 14):
                tmp.write(ch)
            tmp.close()
            return Path(tmp.name)
        for k in ("value", "path", "name", "orig_name"):
            v = obj.get(k)
            if isinstance(v, str):
                cand = Path(v)
                if cand.exists():
                    return cand
                if space_url:
                    try:
                        return _coerce(f"{space_url}/file={v}")
                    except Exception:
                        pass
    if isinstance(obj, (tuple, list)):
        for x in obj:
            r = _coerce(x, space_url)
            if r is not None:
                return r
    return None


def remove_background(photo_path: str | Path, out_dir: str | Path = "test_photos") -> Path:
    """Strip the background from `photo_path`. Tries multiple HF Spaces in
    order; falls back if one is in CONFIG_ERROR / RUNTIME_ERROR / unreachable."""
    load_dotenv()
    from gradio_client import Client, handle_file

    photo_path = Path(photo_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    last_err = None
    for space_id, endpoint, idx, kw in RMBG_SPACES:
        try:
            try:
                client = Client(space_id, hf_token=hf_token) if hf_token else Client(space_id)
            except TypeError:
                client = Client(space_id, token=hf_token) if hf_token else Client(space_id)

            space_url = f"https://{space_id.replace('/', '-').lower()}.hf.space"
            print(f"[rembg] trying {space_id}{endpoint}")
            t0 = time.time()
            if kw:
                result = client.predict(**{kw: handle_file(str(photo_path))}, api_name=endpoint)
            else:
                result = client.predict(handle_file(str(photo_path)), api_name=endpoint)
            print(f"[rembg] took {time.time() - t0:.1f}s")

            if idx is not None and isinstance(result, (tuple, list)) and len(result) > idx:
                src = _coerce(result[idx], space_url=space_url) or _coerce(result, space_url=space_url)
            else:
                src = _coerce(result, space_url=space_url)
            if src is None:
                raise ValueError(f"Could not extract output from {space_id}: {result!r}")

            out_path = out_dir / f"{photo_path.stem}_nobg.png"
            out_path.write_bytes(src.read_bytes())
            print(f"[rembg] saved via {space_id} to {out_path}")
            return out_path
        except Exception as e:
            print(f"[rembg] {space_id} failed: {str(e)[:120]}")
            last_err = e
            continue

    raise RuntimeError(
        f"All BRIA-RMBG fallback Spaces failed. Last error: {last_err}"
    )
