"""
Stage 0: Photo -> 3D mesh.

Two backends:
- Replicate (paid, fast, reliable) — uses version-pinned community models.
- HuggingFace Spaces (free, rate-limited, can be flaky) — uses gradio_client.

Backend is chosen by model preset name:
  triposr / hunyuan3d / hunyuan3d-turbo / trellis   -> Replicate
  triposr-hf / hunyuan3d-hf                         -> HuggingFace Spaces
  hf:<owner>/<space>                                -> custom HF Space
  <owner>/<name>[:version]                          -> custom Replicate model

The downloaded mesh path is returned; callers feed it into the brick pipeline.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


# Replicate (paid) presets — pinned to known-working version hashes.
REPLICATE_PRESETS = {
    "triposr":         "camenduru/tripo-sr:e0d3fe8abce3ba86497ea3530d9eae59af7b2231b6c82bedfc32b0732d35ec3a",
    "hunyuan3d":       "tencent/hunyuan3d-2:b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb",
    "hunyuan3d-turbo": "ndreca/hunyuan3d-2:0602bae6db1ce420f2690339bf2feb47e18c0c722a1f02e9db9abd774abaff5d",
    "trellis":         "firtoz/trellis:e8f6c45206993f297372f5436b90350817bd9b4a0d52d2a76df50c1c8afa2b3c",
}

# HuggingFace Space presets — (space_id, backend_kind).
HF_PRESETS = {
    "triposr-hf":           ("Pheerakarn/TripoSR", "triposr"),
    "hunyuan3d-hf":         ("frogleo/Image-to-3D", "hunyuan3d_shape"),
    "hunyuan3d-textured":   ("tencent/Hunyuan3D-2", "hunyuan3d_textured"),
    "hunyuan3d-2.1-hf":     ("Jbowyer/Hunyuan3D-2.1", "hunyuan3d_2_1"),
    "trellis-hf":           ("microsoft/TRELLIS", "trellis"),
    "trellis-2-hf":         ("microsoft/TRELLIS.2", "trellis2"),
    "sf3d-hf":              ("stabilityai/stable-fast-3d", "sf3d"),
}

# Auto-fallback chain — when the user picks "auto", we try these in order
# until one works. Each is a known textured (or partially-textured) backend.
# Updated as Spaces come up/down. The chain mixes textured and shape-only
# so we always return SOMETHING usable.
AUTO_FALLBACK_CHAIN = [
    # User chose reliability-first (2026-05-12). Hunyuan3D 2.1 has the
    # best success rate on organic subjects; TRELLIS.2 produces cleaner
    # geometry when healthy but has been hitting RuntimeError frequently.
    "hunyuan3d-2.1-hf",        # Jbowyer's Hunyuan3D 2.1 — most reliable
    "trellis-2-hf",            # Microsoft TRELLIS.2 — cleaner when it works
    "hunyuan3d-textured",      # Tencent's Hunyuan3D-2 — taller geometry
    "trellis-hf",              # Microsoft TRELLIS v1
    "sf3d-hf",                 # Stability SF3D — faster but lower detail
    "hunyuan3d-hf",            # frogleo shape-only (last resort)
]


def _hf_space_state(space_id: str, timeout: float = 5.0) -> str:
    """Returns the Space's runtime stage ('RUNNING' / 'CONFIG_ERROR' / etc.)
    or 'UNKNOWN' if the HF API can't be reached."""
    import os
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    h = {"Authorization": f"Bearer {tok}"} if tok else {}
    try:
        r = requests.get(f"https://huggingface.co/api/spaces/{space_id}",
                         headers=h, timeout=timeout)
        if r.ok:
            return r.json().get("runtime", {}).get("stage", "UNKNOWN")
    except Exception:
        pass
    return "UNKNOWN"


# Approximate GPU duration each Space allocates (seconds). Used to predict
# whether a Space request will hit the "ZeroGPU quota exceeded" wall.
SPACE_GPU_DURATION = {
    "trellis-2-hf":     120,
    "trellis-hf":       120,
    "hunyuan3d-2.1-hf":  90,
    "hunyuan3d-textured": 90,
    "sf3d-hf":           45,
    "hunyuan3d-hf":      45,
    "triposr-hf":        15,
}

MESH_EXTS = {".obj", ".glb", ".gltf", ".ply", ".stl", ".fbx"}
MESH_KEYS = ("model_file", "mesh", "obj", "glb", "model", "output")


# ---------- HuggingFace Spaces backend ----------

def _download_to_tmp(url: str) -> Path:
    import tempfile
    ext = Path(urlparse(url).path).suffix or ".glb"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    for chunk in r.iter_content(chunk_size=1 << 14):
        tmp.write(chunk)
    tmp.close()
    return Path(tmp.name)


def _coerce_gradio_file(obj, space_url: str | None = None) -> Path | None:
    """Gradio file outputs come in 3 shapes:
      - a local filepath that gradio_client already downloaded
      - a dict with 'path' (server-side) and/or 'url' (downloadable)
      - a server-side path string like '/static/...' or '/tmp/gradio/...'
    Resolve all of them to a local file we can read.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, Path)):
        p = Path(obj)
        if p.exists():
            return p
        # Server-side path — fetch via the Space's static endpoint
        if space_url:
            url = f"{space_url.rstrip('/')}/file={obj}"
            try:
                return _download_to_tmp(url)
            except Exception:
                pass
        return None

    if isinstance(obj, dict):
        # Prefer a directly-downloadable URL
        url = obj.get("url") or obj.get("href")
        if isinstance(url, str) and url.startswith("http"):
            try:
                return _download_to_tmp(url)
            except Exception:
                pass
        # Otherwise try the path (gradio uses different key names across versions/components)
        for key in ("value", "path", "name", "orig_name"):
            v = obj.get(key)
            if isinstance(v, str) and v:
                local = _coerce_gradio_file(v, space_url=space_url)
                if local:
                    return local
            if isinstance(v, dict):
                local = _coerce_gradio_file(v, space_url=space_url)
                if local:
                    return local
    return None


class HFSpaceBrokenError(RuntimeError):
    """Raised when an upstream HF Space throws a non-recoverable exception."""


def _run_hf_space(photo_path: Path, out_dir: Path, preset: str,
                  extra_inputs: dict | None = None) -> Path:
    from gradio_client import Client, handle_file

    space_id, kind = HF_PRESETS[preset]
    print(f"[HF] space={space_id}  kind={kind}")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        print("[HF] using authenticated HF token (higher quota)")
        try:
            client = Client(space_id, hf_token=hf_token)  # newer gradio_client
        except TypeError:
            client = Client(space_id, token=hf_token)     # older gradio_client
    else:
        client = Client(space_id)
    # The Space's static endpoint, used to fetch '/static/...' or '/tmp/gradio/...' paths
    space_url = f"https://{space_id.replace('/', '-').lower()}.hf.space"
    multiviews = (extra_inputs or {}).get("multiviews", {}) or {}
    mesh_intent = str((extra_inputs or {}).get("mesh_intent") or "").strip()
    if mesh_intent:
        print(f"[HF] mesh intent: {mesh_intent[:180]}")

    def _mv_file(label: str):
        path = multiviews.get(label)
        if path is None:
            return None
        return handle_file(str(path))

    def _mv_count() -> int:
        return sum(1 for v in multiviews.values() if v is not None)

    if kind == "triposr":
        # Pheerakarn/TripoSR is a 2-stage API: preprocess then generate.
        processed = client.predict(
            handle_file(str(photo_path)),
            True,    # remove_background
            0.85,    # foreground_ratio
            api_name="/preprocess",
        )
        processed_path = _coerce_gradio_file(processed, space_url=space_url)
        obj_path, glb_path = client.predict(
            handle_file(str(processed_path)),
            256,     # marching_cubes_resolution
            api_name="/generate",
        )
        local_src = (_coerce_gradio_file(glb_path, space_url=space_url)
                     or _coerce_gradio_file(obj_path, space_url=space_url))
    elif kind == "hunyuan3d_shape":
        # frogleo/Image-to-3D returns (html_str, download_filepath, glb_path_str, obj_path_str).
        result = client.predict(
            image=handle_file(str(photo_path)),
            steps=5,
            guidance_scale=5.5,
            seed=1234,
            octree_resolution=256,
            num_chunks=8000,
            target_face_num=10000,
            randomize_seed=True,
            api_name="/gen_shape",
        )
        local_src = (
            _coerce_gradio_file(result[1], space_url=space_url)
            or _coerce_gradio_file(result[2], space_url=space_url)
            or _coerce_gradio_file(result[3], space_url=space_url)
        )
        if local_src is None:
            raise ValueError(f"Could not extract mesh file from result: {result!r}")
    elif kind == "sf3d":
        # stabilityai/stable-fast-3d — fast textured 3D, one-call API.
        # As of writing, this Space throws an uncaught exception upstream.
        # We surface a clear actionable error so the user knows to switch model.
        print("[HF] running SF3D /run_button (~20-30s)")
        try:
            result = client.predict(
                input_image=handle_file(str(photo_path)),
                foreground_ratio=0.85,
                remesh_option="None",
                vertex_count=-1,
                texture_size=1024,
                api_name="/run_button",
            )
        except Exception as e:
            msg = str(e)
            if "verbose error reporting" in msg or "has raised an exception" in msg:
                raise HFSpaceBrokenError(
                    "The SF3D Space (stabilityai/stable-fast-3d) is currently "
                    "broken upstream — its /run_button endpoint throws an internal "
                    "error on every call. Switch the model to 'TRELLIS Textured' "
                    "or 'Hunyuan3D Shape' instead."
                ) from e
            raise
        local_src = (
            _coerce_gradio_file(result[1], space_url=space_url)
            if isinstance(result, (tuple, list)) and len(result) > 1 else None
        )
        if local_src is None:
            raise ValueError(f"Could not extract SF3D mesh from result: {result!r}")
    elif kind == "trellis2":
        # microsoft/TRELLIS.2 — newer 2-call workflow, slightly different param shapes.
        try:
            client.predict(api_name="/start_session")
        except Exception:
            pass
        print("[HF] running TRELLIS.2 /image_to_3d (~120s)")
        # All guidance/sampling params left at Space defaults so we don't
        # trigger internal RuntimeErrors from out-of-range values.
        client.predict(
            image=handle_file(str(photo_path)),
            seed=0,
            resolution="1024",
            ss_guidance_strength=7.5,
            ss_guidance_rescale=0.7,
            ss_sampling_steps=12,
            ss_rescale_t=5.0,
            shape_slat_guidance_strength=7.5,
            shape_slat_guidance_rescale=0.5,
            shape_slat_sampling_steps=12,
            shape_slat_rescale_t=3.0,
            tex_slat_guidance_strength=1.0,
            tex_slat_guidance_rescale=0.0,
            tex_slat_sampling_steps=12,
            tex_slat_rescale_t=3.0,
            api_name="/image_to_3d",
        )
        print("[HF] extracting GLB (TRELLIS.2)")
        result = client.predict(
            decimation_target=300000,  # Space default
            texture_size=2048,
            api_name="/extract_glb",
        )
        local_src = (
            _coerce_gradio_file(result, space_url=space_url)
            if not isinstance(result, (tuple, list))
            else (
                _coerce_gradio_file(result[0], space_url=space_url)
                or _coerce_gradio_file(result[1], space_url=space_url)
            )
        )
        if local_src is None:
            raise ValueError(f"Could not extract TRELLIS.2 mesh: {result!r}")
    elif kind == "trellis":
        # microsoft/TRELLIS: two-call workflow.
        # 1) /image_to_3d generates the 3D asset (returns video preview + state)
        # 2) /extract_glb downloads the textured GLB
        try:
            client.predict(api_name="/start_session")
        except Exception:
            pass

        # Multi-view: pass front + any provided side/back as a "multiimages" list.
        # TRELLIS uses these to condition texture/geometry from multiple angles.
        multi = []
        for label in ("front", "back", "left", "right"):
            p = multiviews.get(label)
            if p is not None:
                caption = label
                if mesh_intent and label == "front":
                    caption = "front view: centered forward-facing head and face reference"
                elif mesh_intent and label in {"left", "right"}:
                    caption = f"{label} side view: body proportions, legs, tail, and markings"
                multi.append({"image": handle_file(str(p)), "caption": caption})
        if multi:
            print(f"[HF] running TRELLIS /image_to_3d with {len(multi)+1} views")
        else:
            print("[HF] running TRELLIS /image_to_3d (~60-90s, single view)")
        client.predict(
            image=handle_file(str(photo_path)),
            multiimages=multi,
            seed=0,
            ss_guidance_strength=7.5,
            ss_sampling_steps=12,
            slat_guidance_strength=3.0,
            slat_sampling_steps=12,
            multiimage_algo="stochastic",
            api_name="/image_to_3d",
        )
        print("[HF] extracting GLB")
        result = client.predict(
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/extract_glb",
        )
        local_src = (
            _coerce_gradio_file(result, space_url=space_url)
            if not isinstance(result, (tuple, list))
            else (
                _coerce_gradio_file(result[0], space_url=space_url)
                or _coerce_gradio_file(result[1], space_url=space_url)
            )
        )
        if local_src is None:
            raise ValueError(f"Could not extract textured mesh from TRELLIS result: {result!r}")
    elif kind == "hunyuan3d_2_1":
        # Jbowyer/Hunyuan3D-2.1 — same shape as tencent's but no `caption` param.
        # Try /generation_all (textured), fall back to /shape_generation if it errors.
        mv_n = _mv_count()
        if mv_n:
            print(f"[HF] Hunyuan3D-2.1 /generation_all (shape+texture, {mv_n} labeled view(s))")
        else:
            print("[HF] Hunyuan3D-2.1 /generation_all (shape+texture)")
        try:
            result = client.predict(
                image=handle_file(str(photo_path)),
                mv_image_front=_mv_file("front"),
                mv_image_back=_mv_file("back"),
                mv_image_left=_mv_file("left"),
                mv_image_right=_mv_file("right"),
                steps=30, guidance_scale=5.0, seed=1234,
                octree_resolution=256,
                check_box_rembg=True,
                num_chunks=8000,
                randomize_seed=True,
                api_name="/generation_all",
            )
        except Exception as e:
            print(f"[HF] /generation_all failed ({type(e).__name__}); falling back to /shape_generation")
            result = client.predict(
                image=handle_file(str(photo_path)),
                mv_image_front=_mv_file("front"),
                mv_image_back=_mv_file("back"),
                mv_image_left=_mv_file("left"),
                mv_image_right=_mv_file("right"),
                steps=30, guidance_scale=5.0, seed=1234,
                octree_resolution=256,
                check_box_rembg=True,
                num_chunks=8000,
                randomize_seed=True,
                api_name="/shape_generation",
            )
        # Result tuple: try indices 0 and 1 for the file
        if isinstance(result, (tuple, list)):
            local_src = None
            for entry in result:
                local_src = _coerce_gradio_file(entry, space_url=space_url)
                if local_src is not None:
                    break
        else:
            local_src = _coerce_gradio_file(result, space_url=space_url)
        if local_src is None:
            raise ValueError(f"Could not extract Hunyuan3D-2.1 mesh: {result!r}")
    elif kind == "hunyuan3d_textured":
        # tencent/Hunyuan3D-2 /generation_all = shape + texture in one call.
        # Returns (file_white, file_textured, output_html, mesh_stats, seed).
        # The /generation_all endpoint has historically been buggy upstream
        # ("NameError" on their side), so we fall back to /shape_generation.
        mv_n = _mv_count()
        if mv_n:
            print(f"[HF] running /generation_all (shape + texture; {mv_n} labeled view(s); takes ~60s)")
        else:
            print("[HF] running /generation_all (shape + texture; takes ~60s)")
        try:
            result = client.predict(
                caption=mesh_intent or None,
                image=handle_file(str(photo_path)),
                mv_image_front=_mv_file("front"),
                mv_image_back=_mv_file("back"),
                mv_image_left=_mv_file("left"),
                mv_image_right=_mv_file("right"),
                steps=30, guidance_scale=5.0, seed=1234,
                octree_resolution=256, check_box_rembg=True,
                num_chunks=8000, randomize_seed=True,
                api_name="/generation_all",
            )
        except Exception as e:
            print(f"[HF] /generation_all failed ({type(e).__name__}); "
                  f"falling back to /shape_generation (gray mesh)")
            result = client.predict(
                caption=mesh_intent or None,
                image=handle_file(str(photo_path)),
                mv_image_front=_mv_file("front"),
                mv_image_back=_mv_file("back"),
                mv_image_left=_mv_file("left"),
                mv_image_right=_mv_file("right"),
                steps=30, guidance_scale=5.0, seed=1234,
                octree_resolution=256, check_box_rembg=True,
                num_chunks=8000, randomize_seed=True,
                api_name="/shape_generation",
            )
        # Prefer textured (idx 1) if present, fall back to shape (idx 0)
        if isinstance(result, (tuple, list)):
            local_src = None
            for entry in result:
                local_src = _coerce_gradio_file(entry, space_url=space_url)
                if local_src is not None:
                    break
        else:
            local_src = _coerce_gradio_file(result, space_url=space_url)
        if local_src is None:
            raise ValueError(f"Could not extract textured mesh from result: {result!r}")
    else:
        raise ValueError(f"Unknown HF preset kind: {kind}")

    if local_src is None or not local_src.exists():
        raise ValueError(f"Could not resolve mesh output to a local file (got {local_src!r})")
    suffix = local_src.suffix or ".glb"
    out_path = out_dir / f"{photo_path.stem}{suffix}"
    out_path.write_bytes(local_src.read_bytes())
    print(f"[HF] saved mesh to {out_path}")
    return out_path


# ---------- Replicate backend ----------

def _build_replicate_inputs(model_id: str, fh) -> dict:
    if "tripo-sr" in model_id:
        return {"image_path": fh, "do_remove_background": True, "foreground_ratio": 0.85}
    if "hunyuan3d-2" in model_id:
        return {"image": fh, "remove_background": True}
    if "trellis" in model_id:
        return {"images": [fh], "generate_model": True, "generate_color": True}
    return {"image": fh}


def _output_to_url(output) -> str:
    """Walk Replicate output (str / dict / list / FileOutput) for the mesh URL."""
    fallback = None

    def walk(node):
        nonlocal fallback
        if node is None:
            return None
        if isinstance(node, str):
            if any(node.lower().split("?")[0].endswith(ext) for ext in MESH_EXTS):
                return node
            if fallback is None and node.startswith("http"):
                fallback = node
            return None
        if isinstance(node, dict):
            for key in MESH_KEYS:
                if key in node:
                    found = walk(node[key])
                    if found:
                        return found
            for v in node.values():
                found = walk(v)
                if found:
                    return found
            return None
        if isinstance(node, (list, tuple)):
            for v in node:
                found = walk(v)
                if found:
                    return found
            return None
        for attr in ("url", "get_url"):
            v = getattr(node, attr, None)
            if callable(v):
                try:
                    return walk(v())
                except Exception:
                    pass
            elif isinstance(v, str):
                return walk(v)
        return None

    found = walk(output)
    if found:
        return found
    if fallback:
        return fallback
    raise ValueError(f"Could not extract a mesh URL from Replicate output: {output!r}")


def _run_replicate(photo_path: Path, out_dir: Path, model_id: str, extra: dict | None) -> Path:
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    import replicate
    print(f"[Replicate] model={model_id}")
    with open(photo_path, "rb") as fh:
        inputs = _build_replicate_inputs(model_id, fh)
        if extra:
            inputs.update(extra)
        t0 = time.time()
        output = replicate.run(model_id, input=inputs)
    print(f"[Replicate] generation took {time.time() - t0:.1f}s")
    mesh_url = _output_to_url(output)
    suffix = Path(urlparse(mesh_url).path).suffix or ".glb"
    out_path = out_dir / f"{photo_path.stem}{suffix}"
    r = requests.get(mesh_url, stream=True, timeout=180)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 14):
            f.write(chunk)
    print(f"[Replicate] saved mesh to {out_path}")
    return out_path


# ---------- public entry point ----------

def photo_to_mesh(
    photo_path: str | Path,
    out_dir: str | Path = "test_meshes",
    model: str = "triposr-hf",
    extra_inputs: dict | None = None,
) -> Path:
    load_dotenv()
    photo_path = Path(photo_path)
    if not photo_path.exists():
        raise FileNotFoundError(photo_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-fallback: try the user's pick first, then each chain entry in turn.
    # Skips Spaces that the HF API reports as CONFIG_ERROR / BUILD_ERROR / RUNTIME_ERROR.
    if model == "auto" or model in HF_PRESETS:
        chain = [model] if model != "auto" else []
        for fallback_model in AUTO_FALLBACK_CHAIN:
            if fallback_model != model and fallback_model not in chain:
                chain.append(fallback_model)

        errors = []
        quota_remaining = None  # learned from quota errors as we go
        for attempt in chain:
            if attempt not in HF_PRESETS:
                continue
            space_id, _ = HF_PRESETS[attempt]
            state = _hf_space_state(space_id)
            if state in ("CONFIG_ERROR", "BUILD_ERROR", "RUNTIME_ERROR"):
                print(f"[HF] skipping {attempt} ({space_id}): state={state}")
                errors.append((attempt, f"state={state}"))
                continue
            # Skip Spaces we KNOW won't fit in the remaining ZeroGPU quota
            needed = SPACE_GPU_DURATION.get(attempt, 0)
            if quota_remaining is not None and needed > quota_remaining:
                msg = f"needs ~{needed}s, only {quota_remaining}s quota left"
                print(f"[HF] skipping {attempt}: {msg}")
                errors.append((attempt, msg))
                continue
            # Retry transient CUDA OOM errors before falling through —
            # the Space often frees memory in a few seconds.
            import re, time as _time
            attempts_for_model = 0
            max_oom_retries = 2
            while True:
                try:
                    print(f"[HF] attempt with model={attempt}")
                    return _run_hf_space(photo_path, out_dir, attempt, extra_inputs=extra_inputs)
                except Exception as e:
                    msg = str(e) or type(e).__name__
                    # CUDA OOM = retry with backoff (not a permanent failure)
                    if "CUDA out of memory" in msg and attempts_for_model < max_oom_retries:
                        wait = 10 * (attempts_for_model + 1)
                        print(f"[HF] {attempt} CUDA OOM, retrying in {wait}s "
                              f"(attempt {attempts_for_model + 2}/{max_oom_retries + 1})")
                        _time.sleep(wait)
                        attempts_for_model += 1
                        continue
                    errors.append((attempt, msg[:120]))
                    # Parse quota messages so we can skip further expensive attempts
                    m = re.search(r"(\d+)s? requested vs\.?\s*(\d+)s? left", msg)
                    if m:
                        quota_remaining = int(m.group(2))
                        print(f"[HF] {attempt} hit quota wall; {quota_remaining}s left for the day")
                    else:
                        print(f"[HF] {attempt} failed: {type(e).__name__}: {msg[:100]}")
                    break  # leave the retry loop and try next model
            continue  # next model in chain
        raise RuntimeError(
            "All HF Space attempts failed:\n  " +
            "\n  ".join(f"{m} -> {err}" for m, err in errors)
        )

    if model.startswith("hf:"):
        space_id = model[3:]
        HF_PRESETS["__custom__"] = (space_id, "triposr")
        try:
            return _run_hf_space(photo_path, out_dir, "__custom__", extra_inputs=extra_inputs)
        finally:
            del HF_PRESETS["__custom__"]

    model_id = REPLICATE_PRESETS.get(model, model)
    return _run_replicate(photo_path, out_dir, model_id, extra_inputs)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path)
    ap.add_argument("--model", default="triposr-hf")
    ap.add_argument("--out-dir", type=Path, default=Path("test_meshes"))
    args = ap.parse_args()
    photo_to_mesh(args.photo, out_dir=args.out_dir, model=args.model)
