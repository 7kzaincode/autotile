"""
GPT-4o-mini vision: structured semantic analysis of a photo.

Replaces our heuristic dark-spot detection and k-means segmentation with
real semantic understanding. ONE call per UNIQUE photo (hash-cached), and
low-detail mode keeps cost at ~$0.0015 per image.

Returns a JSON dict describing the subject:
  - subject_type, subject_name, confidence
  - anatomical regions with bbox + dominant color
  - facial features (eyes / nose / mouth) with positions + colors
  - recommended LEGO color palette

If OPENAI_API_KEY is missing or the call fails, returns None and the
pipeline falls back to the existing heuristics.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv


CACHE_DIR = Path(__file__).parent / "gpt_cache"


SYSTEM_PROMPT = """You analyze photos for a LEGO model generator. Look at the
subject (a single object/animal on a clean background) and return STRICT JSON
matching this schema, no markdown, no commentary:

{
  "subject_type": "pet" | "person" | "bottle_or_can" | "building" | "vehicle" | "toy" | "other",
  "subject_name": "<short noun, e.g. 'rabbit', 'pepsi bottle', 'house', 'red car'>",
  "confidence": <0.0 to 1.0>,
  "is_symmetric_lr": <bool>,
  "regions": [
    {
      "name": "<anatomical/structural part name — see list below>",
      "bbox_normalized": [x0, y0, x1, y1],
      "dominant_color_hex": "#RRGGBB",
      "color_name": "<EXACT LEGO color name from the palette list>"
    }
  ],
  "features": {
    "eye_left":   { "position_normalized": [x, y] } | null,
    "eye_right":  { "position_normalized": [x, y] } | null,
    "nose":       { "position_normalized": [x, y], "color_name": "..." } | null,
    "mouth":      { "position_normalized": [x, y] } | null
  },
  "recommended_lego_palette": [<5-10 LEGO color names>]
}

CRITICAL — color enumeration rule:
You MUST find color VARIATION even on subjects that look monochrome at a glance.
Never return only one color across all regions. Specifically:

- For PETS: look for distinct colors on belly, paws, ear interior (often pink),
  muzzle/cheek, around-eye area, nose, tongue, and any patches/markings. A
  "tan rabbit" is NEVER just tan — it has lighter belly (Cream or White),
  darker paws/ears (Medium Nougat or Dark Tan), pink inner ears, pink nose,
  black eye area, white/cream cheek. A golden retriever/dog is also NEVER just
  Tan — use Cream for chest/muzzle highlights, Tan or Medium Nougat for main
  fur, Dark Tan / Reddish Brown / Brown for darker ears/back/shadows, Pink for
  tongue or rosy nose, and Black for eyes/nose/mouth. Enumerate ALL of these
  as separate regions. If the animal is side-profile and only one eye is
  visible, return only the visible eye and set the hidden eye to null.
  Ignore watermarks, logos, and background lettering; they are not subject
  regions and must not influence the palette.
- For VEHICLES/SHIPS: hull, sail, mast, flag, rope, window, deck, accent —
  each typically a different LEGO color.
- For BUILDINGS: wall, roof, window, door, trim — each distinct.

Aim for AT LEAST 5 regions, often 8-12, even on uniform subjects. Smaller
regions (eye sockets, ear interior, paw pads, belly) MUST be enumerated
separately even when they overlap with larger ones — the system handles
overlap correctly (smaller paints on top).

LEGO color palette (use these EXACT names in `color_name`):
White, Cream, Light Bluish Gray, Dark Bluish Gray, Black,
Red, Dark Red, Bright Red, Coral, Pink, Bright Pink, Magenta,
Blue, Dark Blue, Medium Blue, Light Blue, Azure,
Green, Dark Green, Bright Green, Lime, Olive Green, Sand Green,
Yellow, Bright Light Yellow, Orange, Dark Orange, Bright Light Orange,
Brown, Reddish Brown, Dark Brown, Tan, Dark Tan, Medium Nougat,
Purple, Dark Purple, Light Aqua, Dark Turquoise, Sand Blue, Sand Red.

Region name vocabulary (pick the most specific that fits, or use a descriptive
noun like "belly_patch", "paw", "ear_interior", "muzzle", "cheek", "snout"):
head, face, body, belly, chest, back, ears, ear_interior, eyes, eye_socket,
nose, mouth, muzzle, cheek, paws, legs, tail, neck, wing,
hull, deck, sail, mast, flag, rope, window, door, wheel, cap, label,
wall, roof, chimney, trim, other.

Coordinates are normalized [0,1] where (0,0) is top-left of the FULL photo.
Be conservative — return null for features you can't clearly see.
Pick LEGO colors CLOSE to the actual photo color. Prefer common LEGO colors
(White, Black, Red, Blue, Yellow, Tan, Reddish Brown, Light Bluish Gray)
over rare ones when the choice is close.
"""

# Bump this whenever SYSTEM_PROMPT changes so old cache entries are ignored
# and we re-call GPT with the new prompt.
SYSTEM_PROMPT_VERSION = "v3"


def _photo_hash(photo_path: Path) -> str:
    h = hashlib.sha256()
    with open(photo_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 14), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def analyze_photo(photo_path: str | Path, force_refresh: bool = False,
                  detail: str = "high") -> dict | None:
    """Return structured semantic data about the photo (or None on failure).

    `force_refresh=False` (default) reads from disk cache — same hash never
    re-calls the API. Cache lives at gpt_cache/<sha256>_<prompt_version>.json
    so prompt changes auto-invalidate stale entries.

    detail="high" gives GPT the full image (~$0.005/call); "low" sees a
    512px thumbnail (~$0.0007/call). High is required for good color
    enumeration — low routinely under-counts color variation.
    """
    load_dotenv()
    photo_path = Path(photo_path)
    if not photo_path.exists():
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = _photo_hash(photo_path)
    cache_file = CACHE_DIR / f"{h}_{SYSTEM_PROMPT_VERSION}_{detail}.json"
    if cache_file.exists() and not force_refresh:
        try:
            data = json.loads(cache_file.read_text())
            print(f"[gpt_vision] cache hit ({cache_file.name})")
            return data
        except Exception:
            pass  # fall through and re-fetch

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[gpt_vision] OPENAI_API_KEY not set, skipping")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("[gpt_vision] openai library not installed")
        return None

    with open(photo_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    suffix = photo_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    data_url = f"data:image/{mime};base64,{img_b64}"

    print(f"[gpt_vision] calling GPT-4o-mini (~$0.0015) for {photo_path.name}")
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this photo for LEGO generation."},
                    {"type": "image_url",
                     "image_url": {"url": data_url, "detail": detail}},
                ]},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=1500,
            temperature=0.0,
        )
    except Exception as e:
        print(f"[gpt_vision] API call failed: {e}")
        return None

    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[gpt_vision] could not parse JSON: {e}")
        print(f"raw: {raw[:200]}")
        return None

    # Save to cache
    try:
        cache_file.write_text(json.dumps(data, indent=2))
        usage = resp.usage
        if usage:
            print(f"[gpt_vision] cached -> {cache_file.name}  "
                  f"(tokens: in={usage.prompt_tokens}, out={usage.completion_tokens})")
    except Exception:
        pass

    return data


def features_from_gpt(gpt_data: dict, photo_size: tuple[int, int]) -> dict:
    """Convert GPT's normalized coords into pixel coords. Returns:
        {"eyes": [(u,v), (u,v)], "nose": (u,v) or None,
         "mouth": ((u0,v),(u1,v)) or None}
    """
    H, W = photo_size
    out = {"eyes": [], "nose": None, "mouth": None}
    feats = (gpt_data or {}).get("features", {})

    def to_px(p):
        if not p:
            return None
        try:
            x, y = p["position_normalized"]
            return (int(x * W), int(y * H))
        except (KeyError, TypeError, ValueError):
            return None

    el, er = to_px(feats.get("eye_left")), to_px(feats.get("eye_right"))
    if el and er:
        out["eyes"] = [el, er]
    elif el or er:
        out["eyes"] = [el or er]
    out["nose"] = to_px(feats.get("nose"))
    m = to_px(feats.get("mouth"))
    if m is not None:
        # Mouth as a horizontal segment ~10% of width
        mw = max(2, int(W * 0.06))
        out["mouth"] = ((m[0] - mw, m[1]), (m[0] + mw, m[1]))
    return out
