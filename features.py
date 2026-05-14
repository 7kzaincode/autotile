"""
Detect anatomical features in the input photo and place specialty accent
pieces at the corresponding voxel positions.

Eyes:  1x1 round_plate (white) + 1x1 round_tile (black pupil) stacked
       toward the camera. Renders in Three.js as cylinder pair.
Nose:  1x1 round_tile in a pink/dark color, central below eyes.
Mouth: 1x2 dark tile below nose.

Symmetric enforcement: if only one of a pair is detected (eye or paw),
mirror it to the other side using the model's X symmetry axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


# Palette IDs we use for the canonical features.
# These match lego_palette.json:
#   4  = Black
#   1  = White
#   31 = Pink
#   25 = Brown
EYE_PUPIL_COLOR = 4    # Black
EYE_BALL_COLOR  = 1    # White
NOSE_COLOR      = 31   # Pink (bunnies/pets — nose is pink/rosy)
MOUTH_COLOR     = 4    # Black (a thin dark line)


def detect_features(photo_path: str | Path) -> dict:
    """Return a dict with detected eye / nose / mouth (u, v) pixel coords.

    Keys may be missing if a feature can't be confidently found:
      {
        "eyes":   [(u1, v1), (u2, v2)],
        "nose":   (u, v) | None,
        "mouth":  ((u_left, v), (u_right, v)) | None,
        "bbox":   (x0, y0, x1, y1),   # foreground bbox in the photo
        "shape":  (H, W),
      }
    """
    raw = Image.open(photo_path)
    has_alpha = raw.mode in ("RGBA", "LA") or (raw.mode == "P" and "transparency" in raw.info)
    if has_alpha:
        rgba = raw.convert("RGBA")
        rgba_arr = np.asarray(rgba)
        alpha = rgba_arr[..., 3]
        # Composite transparent background to white before brightness tests.
        arr = rgba_arr[..., :3].astype(np.float32)
        arr[alpha <= 16] = 255
        fg = alpha > 16
    else:
        img = raw.convert("RGB")
        arr = np.asarray(img).astype(np.float32)
        bright0 = arr.mean(axis=2)
        fg = bright0 < 250
    H, W = arr.shape[:2]

    bright = arr.mean(axis=2)
    if not fg.any():
        return {"eyes": [], "nose": None, "mouth": None,
                "bbox": (0, 0, W, H), "head_bbox": (0, 0, W, H),
                "shape": (H, W)}
    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)

    # ── EYES ───────────────────────────────────────────────────────────
    # Dark spot pair in upper half, similar Y, horizontal separation.
    eyes = _detect_eye_pair(bright, fg, y0, y1, x0, x1, bbox_w, bbox_h)

    # ── NOSE ───────────────────────────────────────────────────────────
    # If eyes found, look for a dark spot between/below them.
    nose = None
    if len(eyes) == 2:
        eye_y = (eyes[0][1] + eyes[1][1]) / 2
        eye_x_mid = (eyes[0][0] + eyes[1][0]) / 2
        eye_dx = abs(eyes[0][0] - eyes[1][0])
        # Nose lives ~ 1.2-2x the eye-spacing below the eyes,
        # within ~half the eye-spacing horizontally of center
        nose_search_y0 = int(eye_y + 0.25 * eye_dx)
        nose_search_y1 = int(eye_y + 0.95 * eye_dx)
        nose_search_x0 = int(eye_x_mid - 0.8 * eye_dx)
        nose_search_x1 = int(eye_x_mid + 0.8 * eye_dx)
        bbox_area = bbox_w * bbox_h
        nose = _find_central_dark_spot(
            bright, fg,
            max(0, nose_search_y0), min(H, nose_search_y1),
            max(0, nose_search_x0), min(W, nose_search_x1),
            min_size=max(2, int(bbox_area * 0.00001)),
            max_size=max(300, int(bbox_area * 0.02)),
            prefer_y=nose_search_y0 + 0.25 * max(1, nose_search_y1 - nose_search_y0),
        )
        if nose is None:
            nose = (
                int(eye_x_mid),
                int(min(y1 - 1, max(y0, eye_y + 0.72 * eye_dx))),
            )

    # ── MOUTH ──────────────────────────────────────────────────────────
    # Just below the nose, look for a horizontal dark stripe.
    mouth = None
    if nose is not None:
        nose_x, nose_y = nose
        mouth_search_y0 = int(nose_y + 0.10 * (eye_dx if eyes else 30))
        mouth_search_y1 = int(nose_y + 0.55 * (eye_dx if eyes else 60))
        mouth_search_x0 = int(nose_x - 0.35 * (eye_dx if eyes else 60))
        mouth_search_x1 = int(nose_x + 0.35 * (eye_dx if eyes else 60))
        mouth = _find_horizontal_dark_stripe(
            bright, fg,
            max(0, mouth_search_y0), min(H, mouth_search_y1),
            max(0, mouth_search_x0), min(W, mouth_search_x1),
        )
        if mouth is None and eyes:
            mouth_y = int(min(y1 - 1, nose_y + 0.25 * eye_dx))
            mouth_w = max(2, int(0.18 * eye_dx))
            mouth = ((nose_x - mouth_w, mouth_y), (nose_x + mouth_w, mouth_y))

    head_bbox = _head_bbox_from_landmarks(
        eyes=eyes, nose=nose, mouth=mouth, bbox=(x0, y0, x1, y1),
    )
    return {
        "eyes":  eyes,
        "nose":  nose,
        "mouth": mouth,
        "bbox":  (x0, y0, x1, y1),
        "head_bbox": head_bbox,
        "shape": (H, W),
    }


def detect_dark_spots(photo_path: str | Path, min_size: int = 4, max_size: int = 200,
                      max_features: int = 2) -> list[tuple[int, int]]:
    """Backwards-compat wrapper — returns eye-pair coords only."""
    feat = detect_features(photo_path)
    return feat["eyes"][:max_features]


# ── internal helpers ──────────────────────────────────────────────────

def _head_bbox_from_landmarks(
    *,
    eyes: list[tuple[int, int]],
    nose,
    mouth,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    pts: list[tuple[int, int]] = []
    pts.extend(eyes or [])
    if nose is not None:
        pts.append(nose)
    if mouth is not None:
        pts.extend(mouth)
    if not pts:
        return bbox
    x0, y0, x1, y1 = bbox
    xs = [int(p[0]) for p in pts]
    ys = [int(p[1]) for p in pts]
    eye_dx = abs(eyes[1][0] - eyes[0][0]) if len(eyes or []) >= 2 else 0
    span = max(24, eye_dx, max(xs) - min(xs), max(ys) - min(ys))
    cx = int(round(float(np.mean(xs))))
    top = min(ys)
    bottom = max(ys)
    return (
        max(x0, cx - int(round(span * 0.85))),
        max(y0, top - int(round(span * 1.05))),
        min(x1, cx + int(round(span * 0.85))),
        min(y1, bottom + int(round(span * 1.20))),
    )

def _detect_eye_pair(bright, fg, y0, y1, x0, x1, bbox_w, bbox_h) -> list[tuple[int, int]]:
    eye_cut = y0 + int((y1 - y0) * 0.55)
    fg_upper = fg.copy()
    fg_upper[eye_cut:, :] = False
    if not fg_upper.any():
        return []
    try:
        from scipy.ndimage import label, center_of_mass
    except ImportError:
        return []
    min_size = max(4, int(bbox_w * bbox_h * 0.00002))
    max_size = max(500, int(bbox_w * bbox_h * 0.018))
    preferred_dx = 0.34 * bbox_w
    preferred_y = y0 + 0.32 * bbox_h
    best_pair = None

    for pct in (2, 5, 10, 15):
        thr = np.percentile(bright[fg_upper], pct)
        dark = (bright < thr) & fg_upper
        labeled, n = label(dark)
        if n == 0:
            continue
        candidates: list[tuple[int, int, int]] = []
        for cid in range(1, n + 1):
            mask = labeled == cid
            s = int(mask.sum())
            if s < min_size or s > max_size:
                continue
            cy, cx = center_of_mass(mask)
            cx_i, cy_i = int(cx), int(cy)
            if not (x0 <= cx_i <= x1 and y0 <= cy_i <= eye_cut):
                continue
            candidates.append((cx_i, cy_i, s))
        if len(candidates) < 2:
            continue

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                cx_i, cy_i, si = candidates[i]
                cx_j, cy_j, sj = candidates[j]
                dy = abs(cy_i - cy_j)
                dx = abs(cx_i - cx_j)
                mean_y = 0.5 * (cy_i + cy_j)
                if dy > 0.14 * bbox_h or not (0.08 * bbox_w < dx < 0.60 * bbox_w):
                    continue
                score = (
                    -1.4 * dy
                    -0.65 * abs(dx - preferred_dx)
                    -0.30 * abs(mean_y - preferred_y)
                    +0.015 * min(si + sj, max_size)
                )
                if best_pair is None or score > best_pair[0]:
                    best_pair = (score, i, j, candidates)

    if best_pair is None:
        return []
    _, i, j, candidates = best_pair
    pts = [(candidates[i][0], candidates[i][1]),
           (candidates[j][0], candidates[j][1])]
    return sorted(pts, key=lambda p: p[0])


def _find_central_dark_spot(bright, fg, y0, y1, x0, x1,
                             min_size: int = 2, max_size: int = 200,
                             prefer_y: float | None = None):
    """Find a dark spot in a search box. Returns (cx, cy) or None."""
    if y1 <= y0 or x1 <= x0:
        return None
    sub_bright = bright[y0:y1, x0:x1]
    sub_fg = fg[y0:y1, x0:x1]
    if not sub_fg.any():
        return None
    thr = np.percentile(sub_bright[sub_fg], 25)
    dark = (sub_bright < thr) & sub_fg
    try:
        from scipy.ndimage import label, center_of_mass
    except ImportError:
        return None
    labeled, n = label(dark)
    if n == 0:
        return None
    best = None
    best_score = -1
    cx_mid = (x1 - x0) / 2
    py = prefer_y - y0 if prefer_y is not None else (y1 - y0) * 0.35
    for cid in range(1, n + 1):
        mask = labeled == cid
        s = int(mask.sum())
        if s < min_size or s > max_size:
            continue
        cy, cx = center_of_mass(mask)
        mean_bright = float(sub_bright[mask].mean())
        # Score: prefer central, upper, genuinely dark spots. Large low mouth /
        # sign-edge shadows should not beat a compact nose mark.
        score = (
            -abs(cx - cx_mid) * 1.3
            -abs(cy - py) * 1.0
            -mean_bright * 0.7
            +min(s, 250) * 0.03
        )
        if score > best_score:
            best_score = score
            best = (int(cx + x0), int(cy + y0))
    return best


def _find_horizontal_dark_stripe(bright, fg, y0, y1, x0, x1):
    """Find the centroid of dark pixels in a search box (mouth shape)."""
    if y1 <= y0 or x1 <= x0:
        return None
    sub_bright = bright[y0:y1, x0:x1]
    sub_fg = fg[y0:y1, x0:x1]
    if not sub_fg.any():
        return None
    thr = np.percentile(sub_bright[sub_fg], 25)
    dark = (sub_bright < thr) & sub_fg
    if dark.sum() < 4:
        return None
    ys, xs = np.where(dark)
    cy = int(ys.mean() + y0)
    x_min = int(xs.min() + x0)
    x_max = int(xs.max() + x0)
    if x_max - x_min < 3:
        return None
    return ((x_min, cy), (x_max, cy))


# ── feature → voxel placement ────────────────────────────────────────

def _photo_uv_to_voxel(u: int, v: int, bbox: tuple, grid_shape: tuple):
    """Convert photo (u, v) → voxel (x, z) within the foreground bbox."""
    x0, y0, x1, y1 = bbox
    sx, sy, sz = grid_shape
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    u_n = (u - x0) / bw
    v_n = (v - y0) / bh
    vx = int(np.clip(u_n * sx, 0, sx - 1))
    vz = int(np.clip((1 - v_n) * sz, 0, sz - 1))
    return vx, vz


def _photo_silhouette_bbox(photo_path: str | Path) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) bbox of the subject in the photo.

    Uses the alpha channel if present (bg-removed PNG), otherwise falls back to
    a near-white-background heuristic. This bbox is what GPT's normalized
    coords MUST be referenced against — not the full photo dimensions.
    """
    from PIL import Image
    img = Image.open(photo_path)
    W, H = img.size
    if img.mode == "RGBA":
        alpha = np.array(img.split()[-1])
        fg = alpha > 16
    else:
        arr = np.array(img.convert("RGB"))
        # Foreground = not near-white (background)
        fg = arr.min(axis=-1) < 240
    if not fg.any():
        return (0, 0, W, H)
    rows = np.any(fg, axis=1); cols = np.any(fg, axis=0)
    y0 = int(np.argmax(rows)); y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols)); x1 = int(len(cols) - np.argmax(cols[::-1]))
    return (x0, y0, x1, y1)


EYE_REGION_NAMES = {
    "eye", "eyes", "eye_socket", "eye_left", "eye_right",
    "eye_socket_left", "eye_socket_right",
}


def _eye_centroids_from_sam(gpt_data: dict | None) -> list[tuple[int, int]] | None:
    """Return eye points from SAM masks in full-photo pixel coordinates."""
    if not gpt_data:
        return None
    masks_by_name = gpt_data.get("_sam_masks_by_region_name") or {}
    if not masks_by_name:
        return None

    pts: list[tuple[int, int]] = []
    for region in gpt_data.get("regions") or []:
        name = (region.get("name") or "").strip()
        name_l = name.lower()
        if name_l not in EYE_REGION_NAMES:
            continue
        mask = masks_by_name.get(name)
        if mask is None:
            mask = masks_by_name.get(name_l)
        if mask is None:
            continue
        mask_arr = np.asarray(mask)
        ys, xs = np.where(mask_arr > 0)
        if len(xs) == 0:
            continue

        if name_l in {"eyes", "eye_socket"}:
            x_mid = float(xs.mean())
            left = xs < x_mid
            right = xs >= x_mid
            if left.sum() >= 8 and right.sum() >= 8:
                left_pt = (int(xs[left].mean()), int(ys[left].mean()))
                right_pt = (int(xs[right].mean()), int(ys[right].mean()))
                if abs(right_pt[0] - left_pt[0]) >= max(2, mask_arr.shape[1] * 0.03):
                    pts.extend([left_pt, right_pt])
                    continue

        pts.append((int(xs.mean()), int(ys.mean())))

    if not pts:
        return None
    pts.sort(key=lambda p: p[0])
    return pts[:2]


def add_anatomical_features(payload: dict, photo_path: str | Path,
                              mirror_features: bool = True,
                              use_gpt: bool = False,
                              gpt_data: dict | None = None,
                              body_photo_path: str | Path | None = None,
                              body_gpt_data: dict | None = None,
                              body_view: str | None = None) -> dict:
    """Detect eyes / nose / mouth and place specialty pieces at voxel positions.

    `use_gpt=True` calls GPT-4o-mini vision (cached) for semantic landmark
    detection instead of the heuristic dark-spot pipeline. Far more reliable
    on photos where eyes aren't darker than fur, or features are unclear.

    If `gpt_data` is supplied (e.g. by the caller who already ran analysis),
    it is used directly instead of re-loading from the on-disk cache.
    """
    if not payload.get("bricks"):
        return payload

    # Pixel CV is the primary source for face landmarks. GPT is useful for
    # naming regions and palettes, but it has repeatedly been too approximate
    # for exact eyes/nose/mouth coordinates.
    feat = detect_features(photo_path)
    print(f"[features] CV: eyes={len(feat.get('eyes') or [])} "
          f"nose={'y' if feat.get('nose') else 'n'} "
          f"mouth={'y' if feat.get('mouth') else 'n'}  "
          f"silhouette_bbox={feat.get('bbox')}")

    if use_gpt or gpt_data is not None:
        try:
            from gpt_vision import analyze_photo, features_from_gpt
            from PIL import Image
            if gpt_data is None:
                gpt_data = analyze_photo(photo_path)
            if gpt_data:
                img = Image.open(photo_path)
                W, H = img.size
                gpt_feats = features_from_gpt(gpt_data, (H, W))
                # GPT coords are normalized to the FULL photo, but voxels span
                # only the silhouette. Use the alpha/foreground bbox so eye at
                # photo-(0.5, 0.25) maps correctly to bunny-(0.5, 0.25) of its
                # own silhouette, not 0.25 of the photo.
                sil_bbox = _photo_silhouette_bbox(photo_path)
                gpt_feat = {
                    "eyes":  gpt_feats["eyes"],
                    "nose":  gpt_feats["nose"],
                    "mouth": gpt_feats["mouth"],
                    "bbox":  sil_bbox,
                    "shape": (H, W),
                }
                if len(feat.get("eyes") or []) == 0 and gpt_feat["eyes"]:
                    feat["eyes"] = gpt_feat["eyes"]
                    feat["bbox"] = sil_bbox
                if feat.get("nose") is None and gpt_feat["nose"]:
                    feat["nose"] = gpt_feat["nose"]
                    feat["bbox"] = sil_bbox
                if feat.get("mouth") is None and gpt_feat["mouth"]:
                    feat["mouth"] = gpt_feat["mouth"]
                    feat["bbox"] = sil_bbox
                print(f"[features] GPT: eyes={len(gpt_feats['eyes'])} "
                      f"nose={'y' if gpt_feats['nose'] else 'n'} "
                      f"mouth={'y' if gpt_feats['mouth'] else 'n'}  "
                      f"silhouette_bbox={sil_bbox} (used only for missing CV points)")
        except Exception as e:
            print(f"[features] GPT path failed, falling back: {e}")
    bbox = feat["bbox"]
    grid_shape = tuple(payload["grid_shape"])
    sx, sy, sz = grid_shape

    # Build occupancy index: (x, y, z) → True
    occ = {}
    for b in payload["bricks"]:
        for dx in range(b["size_x"]):
            for dy in range(b["size_y"]):
                occ.setdefault((b["x"] + dx, b["y"] + dy, b["z"]), True)

    accents: list[dict] = []
    face_map_applied = False

    try:
        before = len(payload.get("bricks") or [])
        face_meta = {"added": 0, "reason": "not attempted"}
        side_body_pet = body_photo_path is not None and (body_view or "").lower() in {"left", "right", "side"}
        if side_body_pet:
            face_meta = {
                "added": 0,
                "reason": "side-body pet mesh has no guaranteed front-face surface; synthetic face overlay disabled",
            }
            face_map_applied = True
            print(f"[face-map] skipped: {face_meta['reason']}")
        else:
            from face_map import apply_pet_face_map
            payload, face_meta = apply_pet_face_map(
                payload, photo_path, feat, mirror_features=mirror_features,
                body_photo_path=body_photo_path,
                body_gpt_data=body_gpt_data,
                body_view=body_view,
            )
        added = len(payload.get("bricks") or []) - before
        if added:
            face_map_applied = True
            print(f"[face-map] added {added} protected face piece(s) "
                  f"plane={face_meta.get('face_plane')} "
                  f"eye_size={face_meta.get('eye_size')} centers={face_meta.get('eye_centers')} "
                  f"anchor={face_meta.get('source') or face_meta.get('target', {}).get('source', 'front-photo')}")
        else:
            if not side_body_pet:
                print(f"[face-map] no eye overlay: {face_meta.get('reason')}")
    except Exception as e:
        print(f"[face-map] failed, falling back to legacy dots: {e}")
        eyes = feat["eyes"]
        if mirror_features and len(eyes) == 1:
            ex, ey = eyes[0]
            mid_x = (bbox[0] + bbox[2]) / 2
            mirror_x = int(2 * mid_x - ex)
            eyes = [eyes[0], (mirror_x, ey)]
        for (u, v) in eyes[:2]:
            vx, vz = _photo_uv_to_voxel(u, v, bbox, grid_shape)
            front_vy = None
            for vy in range(sy):
                if (vx, vy, vz) in occ:
                    front_vy = vy
                    break
            if front_vy is not None:
                accents.append(_accent(vx, front_vy, vz, EYE_PUPIL_COLOR, kind="round_tile"))

    # Nose: 1x1 round_tile pink, central
    if not face_map_applied and feat["nose"]:
        u, v = feat["nose"]
        vx, vz = _photo_uv_to_voxel(u, v, bbox, grid_shape)
        front_vy = None
        for vy in range(sy):
            if (vx, vy, vz) in occ:
                front_vy = vy
                break
        if front_vy is not None:
            nose_color = _sample_feature_palette_id(
                photo_path, u, v,
                candidates={4, 25, 26, 27, 28, 29, 31, 40},
                fallback=NOSE_COLOR,
            )
            accents.append(_accent(vx, front_vy, vz, nose_color, kind="round_tile"))

    # Mouth: a 1x2 dark tile under the nose
    if not face_map_applied and feat["mouth"]:
        (u1, v1), (u2, v2) = feat["mouth"]
        # Place a single 1x2 tile spanning the mouth width
        vx_l, vz_l = _photo_uv_to_voxel(u1, v1, bbox, grid_shape)
        vx_r, vz_r = _photo_uv_to_voxel(u2, v2, bbox, grid_shape)
        vx_mid = (vx_l + vx_r) // 2
        width = max(1, abs(vx_r - vx_l))
        # Cap to 2 — anything wider would interfere with face proportions
        width = min(width, 2)
        front_vy = None
        for vy in range(sy):
            if (vx_mid, vy, vz_l) in occ:
                front_vy = vy
                break
        if front_vy is not None:
            accents.append({
                "x": vx_mid - width // 2, "y": front_vy, "z": vz_l,
                "size_x": width, "size_y": 1, "brick_type": f"1x{width}",
                "kind": "tile", "rotation": 0,
                "color": MOUTH_COLOR, "slope_dir": None,
                "mount": "-y",
            })

    if accents:
        payload["bricks"] = list(payload["bricks"]) + accents
        kinds_added = ", ".join(sorted({a["kind"] for a in accents}))
        print(f"[features] added {len(accents)} anatomical piece(s) ({kinds_added})")
    else:
        print("[features] no anatomical features placed")
    return payload


# Back-compat alias
add_eye_accents = add_anatomical_features


def _accent(vx: int, vy: int, vz: int, color_id: int,
            kind: str = "round_tile") -> dict:
    return {
        "x": vx, "y": vy, "z": vz,
        "size_x": 1, "size_y": 1, "brick_type": "1x1",
        "kind": kind, "rotation": 0,
        "color": color_id, "slope_dir": None,
        "mount": "-y",
    }


def _sample_feature_palette_id(
    photo_path: str | Path,
    u: int,
    v: int,
    *,
    candidates: set[int] | None = None,
    fallback: int,
) -> int:
    """Pick a LEGO color for a tiny facial accent from local photo pixels."""
    try:
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
