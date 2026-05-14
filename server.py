"""
FastAPI server for the lego-ai web app.

Endpoints:
    GET  /                       -> redirects to /viewer/
    GET  /viewer/*               -> static viewer files
    GET  /output/*               -> static brick JSON outputs
    POST /api/generate           -> photo upload, returns bricks JSON
    POST /api/generate-from-mesh -> mesh upload, returns bricks JSON

Run:
    uvicorn server:app --reload --port 8765
"""

from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from budget import resolve as resolve_budget
from build_stats import build_stats
from features import add_anatomical_features
from ldraw_export import to_ldraw
from mesh_to_voxels import load_mesh, make_hollow_shell, voxelize
from mesh_input import prepare_mesh_input
from obj_export import to_obj
from parts_list import parts_list, parts_list_bricklink_xml, parts_list_csv, summary
from pet_color_map import project_pet_color_map
from pet_reference_prompts import (
    build_identity_extraction_prompt,
    build_qa_prompt,
    build_reference_prompt_bundle,
)
from pose_analysis import analyze_photo_pose, is_pet_subject
from postprocess import apply_all as postprocess_all
from texture_projection import project_photo
from voxels_to_bricks import decompose
from semantic_projection import (
    project_semantic, project_semantic_gpt, project_semantic_sam,
    gpt_to_restrict_ids,
)
from voxels_to_palette import (
    load_palette, mirror_colors, mirror_occupancy, mirror_palette_ids,
    photo_palette, quantize, region_color,
)
import server_logs

# Tee stdout so all pipeline print() lines land in the in-memory log ring,
# which the viewer subscribes to via /api/logs/stream.
server_logs.install()


load_dotenv()
ROOT = Path(__file__).parent
PHOTOS_DIR = ROOT / "test_photos"
MESHES_DIR = ROOT / "test_meshes"
OUTPUT_DIR = ROOT / "output"
for d in (PHOTOS_DIR, MESHES_DIR, OUTPUT_DIR):
    d.mkdir(exist_ok=True)

app = FastAPI(title="lego-ai")
app.include_router(server_logs.router)
app.mount("/viewer", StaticFiles(directory=str(ROOT / "viewer"), html=True), name="viewer")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
# Expose pipeline artifacts so the dev bar can preview each step:
# uploaded photos, bg-removed photos, AI-generated 3D meshes.
app.mount("/photos",  StaticFiles(directory=str(PHOTOS_DIR)),  name="photos")
app.mount("/meshes",  StaticFiles(directory=str(MESHES_DIR)),  name="meshes")


@app.on_event("startup")
async def _capture_main_loop():
    """Save the main asyncio loop so worker threads can publish logs back."""
    import asyncio as _asyncio
    server_logs.set_loop(_asyncio.get_running_loop())


@app.get("/")
def root():
    return RedirectResponse(url="/viewer/")


@app.get("/api/presets")
def api_presets():
    """Return subject presets so the viewer can use the backend as source of truth."""
    from subject_preset import PRESETS, list_presets
    return {"order": list_presets(), "presets": PRESETS}


def _identity_profile_from_payload(payload: dict | None) -> dict:
    payload = payload or {}
    profile = payload.get("animal_identity_profile", payload)
    if not isinstance(profile, dict):
        raise HTTPException(400, "animal_identity_profile must be an object")
    return profile


@app.post("/api/pet-reference/identity-prompt")
def api_pet_reference_identity_prompt(payload: dict | None = Body(None)):
    """Return the Stage 1 prompt for extracting a pet identity JSON profile."""
    payload = payload or {}
    source_files = payload.get("source_files") or []
    if isinstance(source_files, str):
        source_files = [source_files]
    if not isinstance(source_files, list):
        raise HTTPException(400, "source_files must be a list of filenames")
    return {"prompt": build_identity_extraction_prompt(source_files)}


@app.post("/api/pet-reference/prompts")
def api_pet_reference_prompts(payload: dict = Body(...)):
    """Return the full standardized multi-view reference prompt bundle."""
    if not isinstance(payload, dict):
        raise HTTPException(400, "request body must be a JSON object")
    profile = _identity_profile_from_payload(payload)
    source_files = payload.get("source_files") or []
    generated_files = payload.get("generated_files")
    if isinstance(source_files, str):
        source_files = [source_files]
    if generated_files is not None and isinstance(generated_files, str):
        generated_files = [generated_files]
    return build_reference_prompt_bundle(
        profile,
        source_files=source_files,
        generated_files=generated_files,
    )


@app.post("/api/pet-reference/qa-prompt")
def api_pet_reference_qa_prompt(payload: dict = Body(...)):
    """Return the QA prompt for generated front/right/left/back references."""
    if not isinstance(payload, dict):
        raise HTTPException(400, "request body must be a JSON object")
    profile = _identity_profile_from_payload(payload)
    generated_files = payload.get("generated_files")
    if generated_files is not None and isinstance(generated_files, str):
        generated_files = [generated_files]
    return {"prompt": build_qa_prompt(profile, generated_files=generated_files)}


def _safe_filename(name: str, label: str = "filename") -> str:
    """Reject anything with path separators or parent refs. Returns the name."""
    if not name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(400, f"Invalid {label}: {name!r}")
    return name


def _safe_upload_name(filename: str | None, default_stem: str = "upload",
                      default_suffix: str = ".jpg") -> tuple[str, str]:
    """Return sanitized (stem, suffix) for user-uploaded filenames."""
    raw = Path(filename or default_stem).name
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(raw).stem)
    stem = re.sub(r"\s+", " ", stem).strip(" ._-")
    stem = stem[:80] or default_stem
    suffix = Path(raw).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix or ""):
        suffix = default_suffix
    return stem, suffix


def _infer_reference_view_from_filename(filename: str | None) -> str | None:
    """Infer front/back/left/right from a user reference-photo filename."""
    stem = Path(filename or "").stem.lower()
    tokens = set(re.findall(r"[a-z0-9]+", stem))
    if not tokens:
        return None

    side_tokens = {"side", "profile", "view", "ref", "reference"}
    front = bool(tokens & {"front", "frontal", "face"})
    back = bool(tokens & {"back", "rear", "behind", "posterior"})
    right = bool(tokens & {
        "right", "rhs", "rightside", "rightprofile", "rightview",
        "rside", "rprofile", "rview",
    }) or ("r" in tokens and bool(tokens & side_tokens))
    left = bool(tokens & {
        "left", "lhs", "leftside", "leftprofile", "leftview",
        "lside", "lprofile", "lview",
    }) or ("l" in tokens and bool(tokens & side_tokens))

    if front and not back:
        return "front"
    if back and not front:
        return "back"
    if right and not left:
        return "right"
    if left and not right:
        return "left"
    return None


def _save_payload(payload: dict, out_name: str) -> Path:
    out_path = OUTPUT_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return out_path


def _record_run_artifact(label: str, url: str, kind: str = "file"):
    """Append a pipeline artifact to the Run tab."""
    arts = list(server_logs.LAST_RUN.get("artifacts") or [])
    arts.append({"label": label, "url": url, "kind": kind, "ts": time.time()})
    server_logs.update_run(artifacts=arts)


def _effective_voxel_supersample(value: int, resolution: int) -> int:
    """0 means Auto: smooth <=48-res grids, keep 64-res runs fast."""
    try:
        v = int(value)
    except Exception:
        v = 0
    if v <= 0:
        return 2 if int(resolution) <= 48 else 1
    return max(1, min(v, 3))


def _pose_adjusted_flags(
    subject: str,
    pose_meta: dict | None,
    mirror: bool,
    mirror_geometry: bool,
    voxel_symmetry: bool,
    accent_features: bool,
) -> tuple[bool, bool, bool, bool, list[str]]:
    """Disable risky symmetry assumptions when a pet photo is side-profile."""
    if not (pose_meta or {}).get("is_side_profile") or not is_pet_subject(subject):
        return mirror, mirror_geometry, voxel_symmetry, accent_features, []
    disabled = []
    if mirror:
        disabled.append("mirror-colors")
    if mirror_geometry:
        disabled.append("mirror-geometry")
    if voxel_symmetry:
        disabled.append("voxel-symmetry")
    return False, False, False, accent_features, disabled


def _skip_pet_color_mirror(subject: str, mirror: bool) -> bool:
    return bool(mirror and is_pet_subject(subject))


def _semantic_paint_mode(subject: str, pose_meta: dict | None) -> str:
    """Pick how aggressively semantic masks should overwrite photo texture."""
    if is_pet_subject(subject):
        if (pose_meta or {}).get("is_side_profile"):
            return "none"
        return "small"
    return "all"


def _postprocess_piece_flags(
    subject: str,
    do_tiles: bool,
    do_slopes: bool,
    do_slope_inv: bool,
) -> tuple[bool, bool, bool, bool, list[str]]:
    """Return effective postprocess flags.

    Pet models are currently kept to rectangular bricks/plates plus top tiles.
    Wedge-like substitutions look noisy on organic geometry and have been the
    source of random triangular pieces around faces, chests, and bellies.
    """
    disabled: list[str] = []
    do_cheese_tips = do_slopes
    if is_pet_subject(subject):
        if do_slopes:
            disabled.append("slopes")
        if do_slope_inv:
            disabled.append("inverted slopes")
        if do_cheese_tips:
            disabled.append("cheese slopes")
        do_slopes = False
        do_slope_inv = False
        do_cheese_tips = False
    return do_tiles, do_slopes, do_slope_inv, do_cheese_tips, disabled


def _run_pipeline(
    mesh_path: Path,
    resolution: int,
    up_axis: str,
    subject_type: str = "",
    photo_path: Path | None = None,
    color_photo_path: Path | None = None,  # use this for body colors / markings
    face_photo_path: Path | None = None,   # use this for face landmarks / accents
    other_side_photo_path: Path | None = None,
    body_view: str | None = None,
    other_side_view: str | None = None,
    smooth_iterations: int = 0,
    back_mode: str = "front_only",
    blur_radius: float = 2.0,
    cluster_colors: int = 8,
    max_colors: int = 6,
    pre_cluster: int = 12,
    mirror: bool = False,
    mirror_geometry: bool = False,
    region_colors: int = 0,
    hollow: bool = False,
    drop_floaters: bool = False,
    auto_support: bool = False,
    darken_edges: bool = False,
    accent_features: bool = False,
    use_photo_palette: bool = False,
    photo_palette_size: int = 6,
    semantic_color: bool = False,
    semantic_regions: int = 8,
    semantic_paint_mode: str = "all",
    use_gpt_vision: bool = False,
    use_sam: bool = True,
    pose_meta: dict | None = None,
    voxel_supersample: int = 0,
    voxel_aa_threshold: float = 0.5,
    voxel_hole_fill: bool = True,
    voxel_symmetry: bool = True,
    do_tiles: bool = True,
    do_slopes: bool = True,
    do_slope_inv: bool = True,
    do_baseplate: bool = False,
) -> dict:
    import numpy as np
    with server_logs.stage("load-mesh", "Load AI-generated 3D mesh from disk",
                            file=mesh_path.name, smooth=smooth_iterations):
        mesh = load_mesh(mesh_path, smooth_iterations=smooth_iterations)
        server_logs.step(f"mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    with server_logs.stage("voxelize",
                            f"Convert 3D mesh into a {resolution}^3 voxel grid (each voxel = 1 LEGO stud)",
                            resolution=resolution, up=up_axis):
        effective_supersample = _effective_voxel_supersample(voxel_supersample, resolution)
        grid = voxelize(
            mesh,
            resolution=resolution,
            up_axis=up_axis,
            subject_type=subject_type,
            pose_hint=(pose_meta or {}).get("pose"),
            voxel_supersample=effective_supersample,
            aa_threshold=voxel_aa_threshold,
            fill_holes=voxel_hole_fill,
            symmetry_axis="x" if (mirror_geometry and voxel_symmetry) else None,
        )
        n_occ = int(grid.occupancy.sum())
        meta = grid.metadata or {}
        chosen_up = meta.get("up_axis", up_axis)
        aa = meta.get("anti_alias") or {}
        hole = meta.get("hole_fill") or {}
        sym = meta.get("symmetry") or {}
        fit = meta.get("auto_fit") or {}
        server_logs.step(
            f"grid {tuple(grid.shape)}, {n_occ} occupied voxels (~{n_occ} bricks before merge)"
        )
        if chosen_up != up_axis:
            server_logs.step(f"auto up-axis chose {chosen_up!r} from requested {up_axis!r}")
        if aa.get("enabled"):
            server_logs.step(f"anti-alias {aa.get('supersample')}x from fine grid {tuple(aa.get('fine_shape', []))}")
        if hole.get("added", 0):
            server_logs.step(f"hole-fill added {hole.get('added')} voxel(s)")
        if sym.get("axis"):
            server_logs.step(f"voxel symmetry axis={sym.get('axis')} added {sym.get('added', 0)} voxel(s)")
        if fit:
            server_logs.step(f"auto-fit cropped {tuple(fit.get('old_shape', []))} → {tuple(fit.get('new_shape', []))}")
        server_logs.update_run(
            chosen_up=chosen_up,
            front_axis=meta.get("front_axis", "-y"),
            voxel_shape=list(grid.shape),
            voxel_supersample=effective_supersample,
        )

    front_axis = str((grid.metadata or {}).get("front_axis") or "-y")

    # The COLOR photo is the bg-removed (but NOT stylized) image. Eyes/features
    # survive there. The default falls back to photo_path so older callers still
    # work.
    color_photo = color_photo_path or photo_path
    face_photo = face_photo_path or color_photo

    gpt_data = None
    if use_gpt_vision and color_photo is not None and color_photo.exists():
        with server_logs.stage("gpt-vision",
                                "Call GPT-4o-mini to identify subject + per-region colors + feature positions",
                                photo=color_photo.name):
            try:
                from gpt_vision import analyze_photo
                gpt_data = analyze_photo(color_photo)
                if gpt_data:
                    subj = gpt_data.get("subject_name") or "?"
                    conf = gpt_data.get("confidence")
                    regions = gpt_data.get("regions") or []
                    palette_rec = gpt_data.get("recommended_lego_palette") or []
                    server_logs.step(f"subject={subj!r} confidence={conf}")
                    server_logs.step(f"{len(regions)} regions: " +
                                     ", ".join(f"{r.get('name','?')}={r.get('color_name','?')}"
                                               for r in regions[:8]))
                    server_logs.step(f"recommended palette: {palette_rec}")
                else:
                    server_logs.step("returned None (key missing or API error)")
            except Exception as e:
                print(f"[gpt] analysis failed: {e}")
                gpt_data = None

    if photo_path is not None and photo_path.exists():
        with server_logs.stage("project-photo",
                                "Sample each voxel's surface color from the photo (UV-projected texture)",
                                back_mode=back_mode):
            project_photo(
                grid, photo_path,
                front_axis=front_axis,
                back_mode=back_mode,
                blur_radius=blur_radius,
                cluster_colors=cluster_colors,
            )

    if mirror_geometry:
        if voxel_symmetry:
            with server_logs.stage("mirror-geometry",
                                    "Symmetry was already enforced during voxelization so projected colors see both sides"):
                sym = (grid.metadata or {}).get("symmetry") or {}
                server_logs.step(
                    f"voxel-stage symmetry axis={sym.get('axis', 'x')} added {sym.get('added', 0)} voxel(s)"
                )
        else:
            with server_logs.stage("mirror-geometry",
                                    "Fill missing limbs by mirroring left↔right (symmetric subjects)"):
                new_occ, new_col = mirror_occupancy(grid.occupancy, grid.colors, axis="x")
                grid.occupancy[:] = new_occ
                grid.colors[:] = new_col
                server_logs.step(f"now {int(grid.occupancy.sum())} voxels")

    if hollow:
        with server_logs.stage("hollow",
                                "Empty interior voxels; keep only shell + structural ribs (~70% piece reduction)"):
            before = int(grid.occupancy.sum())
            grid = make_hollow_shell(grid, wall_thickness=1)
            after = int(grid.occupancy.sum())
            server_logs.step(f"{before} → {after} voxels ({100*after//max(1,before)}% kept)")

    if _skip_pet_color_mirror(subject_type, mirror):
        with server_logs.stage("mirror-colors",
                                "Pet markings are often asymmetric, so color mirroring is skipped"):
            server_logs.step("kept one-sided fur patches, stripes, spots, and tail markings")
    elif mirror:
        with server_logs.stage("mirror-colors",
                                "Force left↔right color symmetry (mirrors RGB values across X)"):
            grid.colors = mirror_colors(grid.colors, grid.occupancy, axis="x")

    if region_colors > 0:
        with server_logs.stage("region-color",
                                f"Cluster voxels into {region_colors} spatial color regions",
                                n=region_colors):
            grid.colors = region_color(grid.colors, grid.occupancy,
                                        n_regions=region_colors, color_weight=1.5)

    palette = load_palette()
    # Semantic mode: segment the photo into N regions, snap each region to a
    # LEGO color, project region IDs onto voxels. Each voxel gets its region's
    # exact LEGO color — quantization downstream becomes a no-op for these.
    # Eyes are detected and forced to Black.
    restrict_ids = None
    used_gpt_color = False
    if gpt_data and color_photo is not None and color_photo.exists():
        sam_ok = False
        if use_sam:
            with server_logs.stage("sam-semantic",
                                    "SAM 2 refines GPT bboxes into anatomical masks for detail paint / palette guidance",
                                    paint_mode=semantic_paint_mode):
                try:
                    used_ids, sam_meta = project_semantic_sam(
                        grid, color_photo, palette, gpt_data,
                        front_axis=front_axis,
                        paint_mode=semantic_paint_mode,
                    )
                    if not sam_meta.get("sam_failed"):
                        restrict_ids = sorted(set(used_ids) | set(gpt_to_restrict_ids(gpt_data, palette)))
                        used_gpt_color = True
                        sam_ok = True
                        if used_ids:
                            used_names = [p["name"] for p in palette if int(p["id"]) in used_ids]
                            server_logs.step(f"{len(used_ids)} painted LEGO colors: {used_names}")
                        else:
                            server_logs.step("palette guidance only; photo texture preserved")
                        server_logs.step(f"restrict_ids (SAM ∪ GPT-recommended): {restrict_ids}")
                    else:
                        server_logs.step("no usable masks, will fall through to GPT-bbox painting")
                except Exception as e:
                    server_logs.step(f"errored: {e} — falling back to bbox")

        if not sam_ok:
            with server_logs.stage("gpt-semantic",
                                    "Paint voxels using GPT's rectangular region bboxes (fallback when SAM unavailable)"):
                try:
                    used_ids, _ = project_semantic_gpt(
                        grid, color_photo, palette, gpt_data,
                        front_axis=front_axis,
                        paint_mode=semantic_paint_mode,
                    )
                    restrict_ids = sorted(set(used_ids) | set(gpt_to_restrict_ids(gpt_data, palette)))
                    used_gpt_color = True
                    if used_ids:
                        used_names = [p["name"] for p in palette if int(p["id"]) in used_ids]
                        server_logs.step(f"{len(used_ids)} painted colors: {used_names}")
                    else:
                        server_logs.step("palette guidance only; photo texture preserved")
                    server_logs.step(f"restrict_ids: {restrict_ids}")
                except Exception as e:
                    server_logs.step(f"errored: {e} — will try k-means")

    if (
        not used_gpt_color
        and semantic_color
        and semantic_paint_mode not in {"none", "palette_only", "palette-only"}
        and color_photo is not None
        and color_photo.exists()
    ):
        with server_logs.stage("kmeans-semantic",
                                f"K-means cluster the photo into {semantic_regions} regions, snap each to LEGO palette",
                                regions=semantic_regions):
            try:
                used_ids, _ = project_semantic(
                    grid, color_photo, palette,
                    n_regions=semantic_regions, front_axis=front_axis,
                )
                restrict_ids = list(used_ids)
                used_names = [p["name"] for p in palette if int(p["id"]) in used_ids]
                server_logs.step(f"{len(used_ids)} colors: {used_names}")
            except Exception as e:
                server_logs.step(f"errored: {e}")

    if use_photo_palette and not restrict_ids and color_photo is not None and color_photo.exists():
        with server_logs.stage("photo-palette",
                                f"Extract {photo_palette_size} dominant photo colors as the LEGO palette"):
            try:
                restrict_ids = photo_palette(color_photo, palette, n_colors=photo_palette_size)
                server_logs.step(f"palette ids: {restrict_ids}")
            except Exception as e:
                server_logs.step(f"errored: {e}")

    if semantic_color and is_pet_subject(subject_type) and color_photo is not None and color_photo.exists():
        with server_logs.stage("pet-color-map",
                                "Replace furry photo texture with a smoothed LEGO-style pet color map"):
            try:
                side_depth = "front" if (body_view or "").lower() in {"left", "right"} else None
                color_ids, color_meta = project_pet_color_map(
                    grid,
                    color_photo,
                    palette,
                    gpt_data,
                    front_axis=front_axis,
                    out_dir=PHOTOS_DIR,
                    debug_stem=color_photo.stem,
                    side_depth=side_depth,
                    markings_on_visible_side_only=bool(side_depth),
                    front_photo_path=face_photo if face_photo != color_photo else None,
                )
                if (
                    other_side_photo_path is not None
                    and other_side_photo_path.exists()
                    and (other_side_view or "").lower() in {"left", "right"}
                ):
                    other_ids, other_meta = project_pet_color_map(
                        grid,
                        other_side_photo_path,
                        palette,
                        None,
                        front_axis=front_axis,
                        out_dir=PHOTOS_DIR,
                        debug_stem=other_side_photo_path.stem,
                        side_depth="back",
                        markings_on_visible_side_only=False,
                        paint_scope="surface_only",
                    )
                    color_ids |= set(other_ids)
                    color_meta["other_side"] = other_meta
                if color_ids:
                    restrict_ids = sorted(set(restrict_ids or []) | set(color_ids))
                    used_gpt_color = True
                names = [p["name"] for p in palette if int(p["id"]) in color_ids]
                server_logs.step(f"{len(color_ids)} clean color-map colors: {names}")
                server_logs.step(f"map shape: {color_meta.get('grid_map_shape')}")
                base_sample = color_meta.get("base_adjustment") or {}
                if base_sample.get("sample_hex"):
                    server_logs.step(
                        f"base coat: {color_meta.get('base_name')} "
                        f"from photo {base_sample.get('sample_hex')} → LEGO {base_sample.get('matched_name')}"
                    )
                else:
                    server_logs.step(f"base coat: {color_meta.get('base_name')}")
                color_profile = color_meta.get("color_profile") or {}
                light_sample = color_profile.get("light") or {}
                if light_sample.get("sample_hex"):
                    server_logs.step(
                        "light coat: "
                        f"{light_sample.get('sample_hex')} → LEGO {light_sample.get('matched_name')}"
                    )
                warm_sample = color_profile.get("warm_secondary") or {}
                warm_regions = color_meta.get("warm_regions") or {}
                if warm_sample.get("sample_hex"):
                    server_logs.step(
                        "warm coat accent: "
                        f"{warm_sample.get('sample_hex')} → LEGO {warm_sample.get('matched_name')} "
                        f"({warm_regions.get('components', 0)} region(s))"
                    )
                marking_sample = color_profile.get("marking") or {}
                if marking_sample.get("sample_hex"):
                    server_logs.step(
                        "marking color: "
                        f"{marking_sample.get('sample_hex')} → LEGO {marking_sample.get('matched_name')}"
                    )
                debug_path = color_meta.get("debug_path")
                if debug_path:
                    p = Path(debug_path)
                    _record_run_artifact("pet color map", f"/photos/{p.name}", "image")
                marking_debug_path = color_meta.get("marking_debug_path")
                if marking_debug_path:
                    p = Path(marking_debug_path)
                    _record_run_artifact("pet marking mask", f"/photos/{p.name}", "image")
                anatomy_debug_path = color_meta.get("anatomy_debug_path")
                if anatomy_debug_path:
                    p = Path(anatomy_debug_path)
                    _record_run_artifact("pet anatomy map", f"/photos/{p.name}", "image")
                anatomy_light = color_meta.get("anatomy_light") or {}
                if anatomy_light.get("applied"):
                    painted = anatomy_light.get("painted") or {}
                    server_logs.step(
                        "anatomy light: "
                        + ", ".join(f"{k}={v}" for k, v in sorted(painted.items()))
                    )
                front_light = color_meta.get("front_light_profile") or {}
                if front_light.get("applied"):
                    server_logs.step(
                        "front light profile: "
                        f"muzzle={front_light.get('muzzle_light')} "
                        f"chest={front_light.get('chest_light')} "
                        f"paws={front_light.get('paw_light')}"
                    )
                markings = color_meta.get("markings") or {}
                if markings:
                    side_filter = markings.get("side_profile_filter") or {}
                    server_logs.step(
                        f"markings: {markings.get('components', 0)} component(s) via {markings.get('source')}"
                    )
                    if side_filter.get("applied"):
                        server_logs.step(
                            "side-profile marking filter: "
                            f"head={side_filter.get('head_side')} "
                            f"removed_head={side_filter.get('removed_head_cells', 0)} "
                            f"removed_tail={side_filter.get('removed_tail_cells', 0)} "
                            f"kept_body={side_filter.get('kept_body_cells', 0)} "
                            f"kept_tail_tip={side_filter.get('kept_tail_tip_cells', 0)}"
                        )
            except Exception as e:
                server_logs.step(f"errored: {e}; keeping previous colors")

    with server_logs.stage("quantize",
                            "Snap each voxel's RGB to the nearest LEGO palette color (CIELAB)",
                            max_colors=max_colors, pre_cluster=pre_cluster,
                            restrict=len(restrict_ids) if restrict_ids else "—"):
        palette_grid = quantize(grid.colors, grid.occupancy, palette,
                                pre_cluster=pre_cluster, max_colors=max_colors,
                                restrict_to_ids=restrict_ids)
        unique_ids = sorted(set(int(v) for v in np.unique(palette_grid) if v > 0))
        unique_names = [p["name"] for p in palette if int(p["id"]) in unique_ids]
        server_logs.step(f"{len(unique_ids)} final colors: {unique_names}")

    if _skip_pet_color_mirror(subject_type, mirror):
        with server_logs.stage("mirror-palette",
                                "Pet markings are often asymmetric, so palette mirroring is skipped"):
            server_logs.step("palette IDs left unchanged for custom markings")
    elif mirror:
        with server_logs.stage("mirror-palette",
                                "Enforce L↔R symmetry at the palette-ID level (prevents asymmetric color drift)"):
            palette_grid = mirror_palette_ids(palette_grid, palette, axis="x")

    with server_logs.stage("decompose",
                            "Greedy decomposition: merge adjacent same-color voxels into LEGO bricks"):
        bricks = decompose(palette_grid)
        server_logs.step(f"{len(bricks)} bricks generated from voxel grid")

    pp_tiles, pp_slopes, pp_slope_inv, pp_cheese_tips, disabled_piece_kinds = (
        _postprocess_piece_flags(subject_type, do_tiles, do_slopes, do_slope_inv)
    )
    with server_logs.stage("postprocess",
                            "Apply top tiles, optional slopes, and floater cleanup",
                            tiles=pp_tiles, slopes=pp_slopes, inv_slopes=pp_slope_inv,
                            cheese_slopes=pp_cheese_tips,
                            drop_floaters=drop_floaters):
        if disabled_piece_kinds:
            server_logs.step(
                "pet tile-only mode disabled: " + ", ".join(disabled_piece_kinds)
            )
        before = len(bricks)
        bricks = postprocess_all(
            bricks, grid.shape,
            do_tiles=pp_tiles, do_slopes=pp_slopes,
            do_slope_inv=pp_slope_inv, do_baseplate=do_baseplate,
            do_cheese_tips=pp_cheese_tips,
            drop_unsupported=drop_floaters,
            auto_support=auto_support,
            darken_edges=darken_edges, palette=palette,
            restrict_to_ids=restrict_ids,
        )
        server_logs.step(f"{before} → {len(bricks)} bricks after post-processing")

    payload_partial = {
        "grid_shape": list(grid.shape),
        "voxel_metadata": grid.metadata,
        "bricks": [b.to_dict() for b in bricks],
    }
    # Anatomical feature placement (eyes, nose, mouth) on the FACE photo.
    # For pets this is intentionally separate from the side/body photo: a
    # front portrait gives much better eye/nose/mouth placement.
    if accent_features and face_photo is not None and face_photo.exists():
        with server_logs.stage("accent-features",
                                "Place eye / nose / mouth pieces (round_tile / round_plate) at GPT-supplied positions"):
            try:
                before = len(payload_partial["bricks"])
                feature_gpt = gpt_data if face_photo == color_photo else None
                payload_partial = add_anatomical_features(
                    payload_partial, face_photo,
                    use_gpt=use_gpt_vision, gpt_data=feature_gpt,
                    body_photo_path=color_photo if is_pet_subject(subject_type) else None,
                    body_gpt_data=gpt_data if is_pet_subject(subject_type) else None,
                    body_view=body_view if is_pet_subject(subject_type) else None,
                )
                added = len(payload_partial["bricks"]) - before
                server_logs.step(f"added {added} accent pieces")
            except Exception as e:
                server_logs.step(f"errored: {e}")
    return {
        "source": str(mesh_path.name),
        "resolution": resolution,
        "grid_shape": list(payload_partial.get("grid_shape", list(grid.shape))),
        "pitch": float(grid.pitch),
        "voxel_metadata": payload_partial.get("voxel_metadata", grid.metadata),
        "palette": palette,
        "bricks": payload_partial["bricks"],
    }


@app.post("/api/generate")
async def generate(
    photo: UploadFile | None = File(None),
    photo_back: UploadFile | None = File(None),
    photo_left: UploadFile | None = File(None),
    photo_right: UploadFile | None = File(None),
    photo_refs: list[UploadFile] | None = File(None),
    # NEW: subject preset (pet/vehicle/building/other) - if set, overrides
    # the dozen knobs below with vetted defaults. See subject_preset.py.
    subject: str = Form(""),
    preset_client_applied: bool = Form(False),
    resolution: int = Form(64),
    model: str = Form("auto"),
    up: str = Form("auto"),
    tiles: bool = Form(True),
    slopes: bool = Form(True),
    slope_inv: bool = Form(True),
    baseplate: bool = Form(False),
    back_mode: str = Form("uv"),
    blur_radius: float = Form(2.0),
    cluster_colors: int = Form(8),
    max_colors: int = Form(8),
    pre_cluster: int = Form(12),
    mirror: bool = Form(True),
    mirror_geometry: bool = Form(True),
    region_colors: int = Form(0),
    hollow: bool = Form(True),
    drop_floaters: bool = Form(True),
    auto_support: bool = Form(False),
    darken_edges: bool = Form(False),
    accent_features: bool = Form(True),
    smooth_iterations: int = Form(2),
    semantic_color: bool = Form(True),
    semantic_regions: int = Form(5),
    use_gpt_vision: bool = Form(True),
    use_sam: bool = Form(True),
    voxel_supersample: int = Form(0),
    voxel_aa_threshold: float = Form(0.5),
    voxel_hole_fill: bool = Form(True),
    voxel_symmetry: bool = Form(True),
    remove_bg: bool = Form(True),
    mesh_cleanup: bool = Form(True),
    # Deprecated / advanced — kept for backwards compatibility, default off.
    # `stylize` (LEGO-ify) is net-negative on color fidelity, keep off.
    # `use_photo_palette` is superseded by semantic+GPT, kept off.
    budget: str = Form(""),
    use_photo_palette: bool = Form(False),
    photo_palette_size: int = Form(6),
    stylize: bool = Form(False),
    stylize_preset: str = Form("lego"),
    stylize_strength: float = Form(0.5),
):
    """Photo(s) -> AI mesh -> LEGO bricks."""
    from photo_to_mesh import photo_to_mesh
    from subject_preset import resolve as resolve_subject

    timestamp = int(time.time())
    pipeline_t0 = time.time()
    submitted_names = [
        getattr(photo, "filename", None),
        getattr(photo_back, "filename", None),
        getattr(photo_left, "filename", None),
        getattr(photo_right, "filename", None),
        *[getattr(uf, "filename", None) for uf in (photo_refs or [])],
    ]
    run_photo_name = next((name for name in submitted_names if name), "reference batch")
    # Reset artifact list at start of each run so the dev bar shows only the
    # current run's outputs (not stale links from a previous run).
    server_logs.update_run(status="starting", photo=run_photo_name,
                           subject=subject, ts=timestamp, artifacts=[])
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  PIPELINE START — {run_photo_name!r} · subject={subject!r}")
    print(f"══════════════════════════════════════════════════════════════════")

    def _record_artifact(label: str, url: str, kind: str = "file"):
        """Append a step's output to LAST_RUN['artifacts'] so the dev bar
        Run tab can show a preview link."""
        _record_run_artifact(label, url, kind)

    async def _save(uf: UploadFile | None, label: str) -> Path | None:
        if uf is None or not getattr(uf, "filename", None):
            return None
        stem, suffix = _safe_upload_name(uf.filename, default_stem=label, default_suffix=".jpg")
        p = PHOTOS_DIR / f"{timestamp}_{label}_{stem}{suffix}"
        with open(p, "wb") as f:
            f.write(await uf.read())
        return p

    with server_logs.stage("upload",
                            "Save uploaded photo(s) to disk as the pipeline's starting point"):
        try:
            role_paths: dict[str, Path | None] = {
                "front": await _save(photo, "front"),
                "back": await _save(photo_back, "back"),
                "left": await _save(photo_left, "left"),
                "right": await _save(photo_right, "right"),
            }
            batch_unknown: list[str] = []
            batch_duplicates: list[str] = []
            batch_assigned: list[str] = []
            for uf in photo_refs or []:
                if uf is None or not getattr(uf, "filename", None):
                    continue
                role = _infer_reference_view_from_filename(uf.filename)
                if role is None:
                    batch_unknown.append(uf.filename)
                    continue
                if role_paths.get(role) is not None:
                    batch_duplicates.append(uf.filename)
                    continue
                role_paths[role] = await _save(uf, role)
                batch_assigned.append(f"{role}={uf.filename}")

            photo_path = role_paths["front"]
            photo_back_path = role_paths["back"]
            photo_left_path = role_paths["left"]
            photo_right_path = role_paths["right"]
            if photo_path is None:
                server_logs.update_run(status="error")
                raise HTTPException(
                    400,
                    "Upload needs a front/face photo. Use the Front input or include "
                    "a batch file named like 'cat front.png'.",
                )
            server_logs.step(f"front photo saved to {photo_path.name}")
            _record_artifact("input photo", f"/photos/{photo_path.name}", "image")
            if batch_assigned:
                server_logs.step(f"auto-sorted reference batch: {batch_assigned}")
            if batch_unknown:
                server_logs.step(f"ignored unsorted batch file(s): {batch_unknown}")
            if batch_duplicates:
                server_logs.step(f"ignored duplicate batch file(s): {batch_duplicates}")
            extras = [p for p in (photo_back_path, photo_left_path, photo_right_path) if p]
            if extras:
                server_logs.step(f"extra views: {[p.name for p in extras]}")
                for p in extras:
                    _record_artifact(f"extra view ({p.stem.split('_')[1]})",
                                     f"/photos/{p.name}", "image")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Failed to save photo(s): {e}")

    # Subject presets are still supported for API callers. The web client
    # applies presets into the form first, then sends preset_client_applied=true
    # so manual Advanced overrides survive.
    if subject and not preset_client_applied:
        prof = resolve_subject(subject)
        if prof:
            print(f"[subject] preset {subject!r}: {prof}")
            for k, v in prof.items():
                if k == "tiles":      tiles = v
                elif k == "slopes":   slopes = v
                elif k == "slope_inv": slope_inv = v
                elif k == "mesh_smoothing":
                    smooth_iterations = {"none": 0, "light": 2, "heavy": 5}.get(v, 2)
                elif k in ("resolution", "up", "remove_bg", "mirror", "mirror_geometry",
                           "hollow", "drop_floaters", "accent_features",
                           "use_gpt_vision", "semantic_color", "semantic_regions",
                           "max_colors", "pre_cluster", "back_mode",
                           "voxel_supersample", "voxel_hole_fill", "voxel_symmetry"):
                    locals()[k]  # for static checkers; real assignment below
            # Apply (locals() trick doesn't work for params; use explicit reassign)
            resolution      = prof.get("resolution", resolution)
            up              = prof.get("up", up)
            remove_bg       = prof.get("remove_bg", remove_bg)
            mirror          = prof.get("mirror", mirror)
            mirror_geometry = prof.get("mirror_geometry", mirror_geometry)
            hollow          = prof.get("hollow", hollow)
            drop_floaters   = prof.get("drop_floaters", drop_floaters)
            accent_features = prof.get("accent_features", accent_features)
            use_gpt_vision  = prof.get("use_gpt_vision", use_gpt_vision)
            use_sam         = prof.get("use_sam", use_sam)
            semantic_color  = prof.get("semantic_color", semantic_color)
            semantic_regions = prof.get("semantic_regions", semantic_regions)
            max_colors      = prof.get("max_colors", max_colors)
            pre_cluster     = prof.get("pre_cluster", pre_cluster)
            back_mode       = prof.get("back_mode", back_mode)
            tiles           = prof.get("tiles", tiles)
            slopes          = prof.get("slopes", slopes)
            slope_inv       = prof.get("slope_inv", slope_inv)
            voxel_supersample = prof.get("voxel_supersample", voxel_supersample)
            voxel_hole_fill = prof.get("voxel_hole_fill", voxel_hole_fill)
            voxel_symmetry = prof.get("voxel_symmetry", voxel_symmetry)

    # Budget preset (orthogonal to subject; cost-focused)
    if budget:
        bprof = resolve_budget(budget)
        resolution = bprof["resolution"]
        hollow = bprof["hollow"]
        region_colors = bprof["region_colors"]
        max_colors = bprof["max_colors"]
        print(f"[budget] applied profile {budget!r}: {bprof}")

    pet_multi_photo = is_pet_subject(subject)
    if pet_multi_photo and not (photo_left_path or photo_right_path):
        server_logs.update_run(status="error")
        raise HTTPException(
            400,
            "Pet generation needs a front face photo plus at least one side full-body photo. "
            "Upload the side view as Left or Right.",
        )

    face_photo_path = photo_path
    body_photo_path = (photo_left_path or photo_right_path or photo_path)
    primary_side_label = (
        "left" if body_photo_path == photo_left_path
        else "right" if body_photo_path == photo_right_path
        else "front"
    )
    other_side_photo_path = (
        photo_right_path if primary_side_label == "left"
        else photo_left_path if primary_side_label == "right"
        else None
    )
    other_side_label = (
        "right" if primary_side_label == "left" and other_side_photo_path is not None
        else "left" if primary_side_label == "right" and other_side_photo_path is not None
        else None
    )
    if pet_multi_photo:
        server_logs.step(
            f"pet photo roles: face=front, body/mesh={primary_side_label}"
            + (f", other_side={other_side_label}" if other_side_label else "")
        )

    # Optional background removal — runs FIRST so SDXL stylization works
    # on the cleanly-cut subject instead of the busy photo background.
    # In the pet pipeline, remove backgrounds for both the face portrait and
    # side/body photos so downstream face placement and body markings use clean
    # foreground bboxes.
    if remove_bg:
        with server_logs.stage("rembg",
                                "Strip photo backgrounds via BRIA-RMBG so the 3D model sees only the subject"):
            try:
                from rembg import remove_background
                removed_by_key: dict[str, Path] = {}

                async def _remove_one(label: str, path: Path | None) -> Path | None:
                    if path is None:
                        return None
                    key = str(path.resolve())
                    if key in removed_by_key:
                        return removed_by_key[key]
                    # Run blocking gradio_client call in threadpool — keeps the
                    # asyncio event loop free to deliver SSE log events live.
                    out = await run_in_threadpool(
                        remove_background, path, out_dir=PHOTOS_DIR)
                    removed_by_key[key] = out
                    server_logs.step(f"{label} bg-removed photo saved to {out.name}")
                    _record_artifact(f"{label} bg-removed photo",
                                     f"/photos/{out.name}", "image")
                    return out

                await _remove_one("front face" if pet_multi_photo else "front", photo_path)
                await _remove_one("back", photo_back_path)
                await _remove_one("left side", photo_left_path)
                await _remove_one("right side", photo_right_path)

                def _mapped(path: Path | None) -> Path | None:
                    if path is None:
                        return None
                    return removed_by_key.get(str(path.resolve()), path)

                photo_path = _mapped(photo_path)
                photo_back_path = _mapped(photo_back_path)
                photo_left_path = _mapped(photo_left_path)
                photo_right_path = _mapped(photo_right_path)
                face_photo_path = _mapped(face_photo_path)
                body_photo_path = _mapped(body_photo_path)
                other_side_photo_path = _mapped(other_side_photo_path)
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(502, f"Background removal failed: {e}")

    # IMPORTANT: keep the bg-removed-but-NOT-stylized path for color analysis
    # (segmentation, eye detection, palette extraction). The stylized image
    # destroys fine features like eyes and small color variations by reducing
    # the photo to N flat colors. Stylization is only useful as 3D-MODEL input.
    # For pets, this is the side/body photo, while `face_photo_path` remains
    # the front portrait used only for eyes / nose / mouth.
    photo_path_for_color = body_photo_path

    pose_meta = {}
    semantic_paint_mode = _semantic_paint_mode(subject, pose_meta)
    if photo_path_for_color is not None:
        with server_logs.stage("pose-analysis",
                                "Classify pet photo pose so side-profile subjects do not get front-face assumptions"):
            try:
                pose_meta = analyze_photo_pose(photo_path_for_color, subject)
                semantic_paint_mode = _semantic_paint_mode(subject, pose_meta)
                server_logs.step(
                    f"pose={pose_meta.get('pose')} confidence={pose_meta.get('confidence')} "
                    f"aspect={pose_meta.get('bbox_aspect')} reason={pose_meta.get('reason')}"
                )
                mirror, mirror_geometry, voxel_symmetry, accent_features, disabled = (
                    _pose_adjusted_flags(
                        subject, pose_meta, mirror, mirror_geometry,
                        voxel_symmetry, accent_features,
                    )
                )
                if disabled:
                    server_logs.step(
                        "disabled for side-profile pet: " + ", ".join(disabled)
                    )
                server_logs.step(f"semantic paint mode: {semantic_paint_mode}")
                server_logs.update_run(
                    pose=pose_meta.get("pose"),
                    pose_confidence=pose_meta.get("confidence"),
                    semantic_paint_mode=semantic_paint_mode,
                )
            except Exception as e:
                server_logs.step(f"pose analysis failed: {e}; continuing with submitted settings")

    mesh_source_path = body_photo_path
    mesh_photo_path = mesh_source_path
    if mesh_cleanup and mesh_photo_path is not None:
        with server_logs.stage("mesh-input",
                                "Prepare a softened mesh-only image to reduce fur, whisker, and watermark geometry"):
            try:
                mesh_photo_path = await run_in_threadpool(
                    prepare_mesh_input, mesh_photo_path, PHOTOS_DIR, subject,
                )
                if mesh_photo_path != mesh_source_path:
                    server_logs.step(f"mesh input saved to {mesh_photo_path.name}")
                    _record_artifact("mesh input photo",
                                     f"/photos/{mesh_photo_path.name}", "image")
            except Exception as e:
                server_logs.step(f"mesh input cleanup failed: {e}; using original")
                mesh_photo_path = mesh_source_path

    # Optional stylization pass — produces the 3D model input
    if stylize and mesh_photo_path is not None:
        with server_logs.stage("stylize",
                                "Pre-stylize photo via SDXL — produces cleaner 3D input but often hurts color fidelity",
                                preset=stylize_preset, strength=stylize_strength):
            try:
                from stylize import stylize_photo
                mesh_photo_path = await run_in_threadpool(
                    stylize_photo, mesh_photo_path,
                    out_dir=PHOTOS_DIR,
                    preset=stylize_preset, strength=stylize_strength,
                )
                server_logs.step(f"saved stylized photo to {mesh_photo_path.name}")
                _record_artifact("stylized photo",
                                 f"/photos/{mesh_photo_path.name}", "image")
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(502, f"Stylization failed: {e}")

    # Multi-view inputs preserve their camera labels for models that support
    # explicit front/left/right/back slots. For pets the primary mesh/color
    # image is the side view, while the front portrait still conditions head
    # geometry and drives face features later.
    multiviews: dict[str, Path] = {}
    for label, path in {
        "front": face_photo_path,
        "back": photo_back_path,
        "left": photo_left_path,
        "right": photo_right_path,
    }.items():
        if path is None:
            continue
        if not pet_multi_photo and path == mesh_photo_path:
            continue
        multiviews[label] = path

    with server_logs.stage("photo-to-mesh",
                            "Call an AI model (TRELLIS / Hunyuan / SF3D) to generate a 3D mesh from the photo — usually the longest step",
                            model=model):
        try:
            # Threadpool: HF Space calls are blocking sync HTTP requests that
            # would freeze the event loop and stall SSE delivery if awaited here.
            extra = {"multiviews": multiviews} if multiviews else {}
            if pet_multi_photo:
                extra["mesh_intent"] = (
                    "Create a realistic full-body pet mesh. Use the side view for "
                    "body length, legs, tail, and coat markings, but use the front "
                    "view to keep the head/face centered and forward-facing with "
                    "symmetric ears and eye sockets. Ignore loose whisker/fur strands "
                    "as geometry."
                )
            if not extra:
                extra = None
            mesh_path = await run_in_threadpool(
                lambda: photo_to_mesh(
                    mesh_photo_path, out_dir=MESHES_DIR, model=model,
                    extra_inputs=extra,
                )
            )
            server_logs.step(f"mesh saved to {mesh_path.name}")
            _record_artifact("3D mesh", f"/meshes/{mesh_path.name}", "mesh")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(502, f"Mesh generation failed: {e}")

    try:
        # Run the whole brick pipeline in threadpool — voxelize, SAM, quantize,
        # decompose are all blocking numpy/torch work that would freeze the
        # event loop and stall the SSE log stream.
        payload = await run_in_threadpool(lambda: _run_pipeline(
            mesh_path, resolution=resolution, up_axis=up,
            subject_type=subject,
            photo_path=photo_path_for_color,
            color_photo_path=photo_path_for_color,
            face_photo_path=face_photo_path,
            other_side_photo_path=other_side_photo_path,
            body_view=primary_side_label,
            other_side_view=other_side_label,
            back_mode=back_mode, blur_radius=blur_radius, cluster_colors=cluster_colors,
            max_colors=max_colors, pre_cluster=pre_cluster, mirror=mirror,
            mirror_geometry=mirror_geometry, region_colors=region_colors,
            hollow=hollow, drop_floaters=drop_floaters,
            auto_support=auto_support, darken_edges=darken_edges,
            accent_features=accent_features,
            smooth_iterations=smooth_iterations,
            use_photo_palette=use_photo_palette,
            photo_palette_size=photo_palette_size,
            semantic_color=semantic_color,
            semantic_regions=semantic_regions,
            semantic_paint_mode=semantic_paint_mode,
            use_gpt_vision=use_gpt_vision,
            use_sam=use_sam,
            pose_meta=pose_meta,
            voxel_supersample=voxel_supersample,
            voxel_aa_threshold=voxel_aa_threshold,
            voxel_hole_fill=voxel_hole_fill,
            voxel_symmetry=voxel_symmetry,
            do_tiles=tiles, do_slopes=slopes,
            do_slope_inv=slope_inv, do_baseplate=baseplate,
        ))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Brick decomposition failed: {e}")

    out_path = _save_payload(payload, f"{photo_path_for_color.stem}.json")
    payload["_saved_to"] = f"/output/{out_path.name}"
    payload["_mesh_name"] = mesh_path.name
    payload["_photo_name"] = photo_path_for_color.name
    payload["_face_photo_name"] = face_photo_path.name if face_photo_path else None
    payload["_side_photo_name"] = body_photo_path.name if body_photo_path else None
    payload["_other_side_photo_name"] = other_side_photo_path.name if other_side_photo_path else None
    payload["_body_view"] = primary_side_label
    payload["_other_side_view"] = other_side_label
    payload["_mesh_input_name"] = mesh_photo_path.name if mesh_photo_path else None
    payload["_pose"] = pose_meta
    payload["_semantic_paint_mode"] = semantic_paint_mode
    _record_artifact("final brick JSON", payload["_saved_to"], "json")
    total_s = time.time() - pipeline_t0
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  PIPELINE DONE — {len(payload.get('bricks', []))} bricks · total {total_s:.1f}s")
    print(f"══════════════════════════════════════════════════════════════════\n")
    server_logs.update_run(
        status="done", bricks=len(payload.get("bricks", [])),
        photo=photo_path_for_color.name, face_photo=payload.get("_face_photo_name"),
        side_photo=payload.get("_side_photo_name"), mesh=mesh_path.name,
        subject=subject, resolution=resolution,
        chosen_up=payload.get("voxel_metadata", {}).get("up_axis", up),
        front_axis=payload.get("voxel_metadata", {}).get("front_axis", "-y"),
        used_gpt=use_gpt_vision, elapsed_s=round(total_s, 1),
        pose=pose_meta.get("pose"), semantic_paint_mode=semantic_paint_mode,
    )
    return JSONResponse(payload)


@app.post("/api/redecompose")
async def redecompose(
    mesh_name: str = Form(...),
    subject: str = Form(""),
    preset_client_applied: bool = Form(False),
    resolution: int = Form(64),
    up: str = Form("auto"),
    photo_name: str = Form(""),
    face_photo_name: str = Form(""),
    other_side_photo_name: str = Form(""),
    body_view: str = Form(""),
    other_side_view: str = Form(""),
    tiles: bool = Form(True),
    slopes: bool = Form(True),
    slope_inv: bool = Form(True),
    baseplate: bool = Form(False),
    back_mode: str = Form("uv"),
    blur_radius: float = Form(2.0),
    cluster_colors: int = Form(8),
    max_colors: int = Form(8),
    pre_cluster: int = Form(12),
    mirror: bool = Form(True),
    mirror_geometry: bool = Form(True),
    region_colors: int = Form(0),
    hollow: bool = Form(True),
    drop_floaters: bool = Form(True),
    auto_support: bool = Form(False),
    darken_edges: bool = Form(False),
    accent_features: bool = Form(True),
    smooth_iterations: int = Form(2),
    use_photo_palette: bool = Form(False),
    photo_palette_size: int = Form(6),
    semantic_color: bool = Form(True),
    semantic_regions: int = Form(5),
    use_gpt_vision: bool = Form(True),
    use_sam: bool = Form(True),
    voxel_supersample: int = Form(0),
    voxel_aa_threshold: float = Form(0.5),
    voxel_hole_fill: bool = Form(True),
    voxel_symmetry: bool = Form(True),
    budget: str = Form(""),
):
    """Re-run the brick pipeline on a previously generated mesh — no AI call."""
    from subject_preset import resolve as resolve_subject
    mesh_name = _safe_filename(mesh_name, "mesh_name")
    mesh_path = MESHES_DIR / mesh_name
    if not mesh_path.exists():
        raise HTTPException(404, f"Mesh not found: {mesh_name}")
    photo_path: Path | None = None
    if photo_name:
        photo_name = _safe_filename(photo_name, "photo_name")
        candidate = PHOTOS_DIR / photo_name
        if candidate.exists():
            photo_path = candidate
    face_photo_path: Path | None = None
    if face_photo_name:
        face_photo_name = _safe_filename(face_photo_name, "face_photo_name")
        candidate = PHOTOS_DIR / face_photo_name
        if candidate.exists():
            face_photo_path = candidate
    other_side_photo_path: Path | None = None
    if other_side_photo_name:
        other_side_photo_name = _safe_filename(other_side_photo_name, "other_side_photo_name")
        candidate = PHOTOS_DIR / other_side_photo_name
        if candidate.exists():
            other_side_photo_path = candidate
    if subject and not preset_client_applied:
        prof = resolve_subject(subject)
        if prof:
            print(f"[subject] redecompose preset {subject!r}: {prof}")
            resolution      = prof.get("resolution", resolution)
            up              = prof.get("up", up)
            mirror          = prof.get("mirror", mirror)
            mirror_geometry = prof.get("mirror_geometry", mirror_geometry)
            hollow          = prof.get("hollow", hollow)
            drop_floaters   = prof.get("drop_floaters", drop_floaters)
            accent_features = prof.get("accent_features", accent_features)
            use_gpt_vision  = prof.get("use_gpt_vision", use_gpt_vision)
            use_sam         = prof.get("use_sam", use_sam)
            semantic_color  = prof.get("semantic_color", semantic_color)
            semantic_regions = prof.get("semantic_regions", semantic_regions)
            max_colors      = prof.get("max_colors", max_colors)
            pre_cluster     = prof.get("pre_cluster", pre_cluster)
            back_mode       = prof.get("back_mode", back_mode)
            voxel_supersample = prof.get("voxel_supersample", voxel_supersample)
            voxel_hole_fill = prof.get("voxel_hole_fill", voxel_hole_fill)
            voxel_symmetry = prof.get("voxel_symmetry", voxel_symmetry)
            if "mesh_smoothing" in prof:
                smooth_iterations = {"none": 0, "light": 2, "heavy": 5}.get(prof["mesh_smoothing"], smooth_iterations)
    if budget:
        bprof = resolve_budget(budget)
        resolution = bprof["resolution"]
        hollow = bprof["hollow"]
        region_colors = bprof["region_colors"]
        max_colors = bprof["max_colors"]
    pose_meta = {}
    semantic_paint_mode = _semantic_paint_mode(subject, pose_meta)
    if photo_path is not None:
        try:
            pose_meta = analyze_photo_pose(photo_path, subject)
            semantic_paint_mode = _semantic_paint_mode(subject, pose_meta)
            mirror, mirror_geometry, voxel_symmetry, accent_features, disabled = (
                _pose_adjusted_flags(
                    subject, pose_meta, mirror, mirror_geometry,
                    voxel_symmetry, accent_features,
                )
            )
            if disabled:
                print(f"[pose] redecompose side-profile pet: disabled {disabled}")
        except Exception as e:
            print(f"[pose] redecompose pose analysis failed: {e}")
    try:
        payload = _run_pipeline(
            mesh_path, resolution=resolution, up_axis=up,
            subject_type=subject,
            photo_path=photo_path,
            color_photo_path=photo_path,
            face_photo_path=face_photo_path or photo_path,
            other_side_photo_path=other_side_photo_path,
            body_view=body_view,
            other_side_view=other_side_view,
            back_mode=back_mode, blur_radius=blur_radius, cluster_colors=cluster_colors,
            max_colors=max_colors, pre_cluster=pre_cluster, mirror=mirror,
            mirror_geometry=mirror_geometry, region_colors=region_colors,
            hollow=hollow, drop_floaters=drop_floaters,
            auto_support=auto_support, darken_edges=darken_edges,
            accent_features=accent_features,
            smooth_iterations=smooth_iterations,
            use_photo_palette=use_photo_palette,
            photo_palette_size=photo_palette_size,
            semantic_color=semantic_color,
            semantic_regions=semantic_regions,
            semantic_paint_mode=semantic_paint_mode,
            use_gpt_vision=use_gpt_vision,
            use_sam=use_sam,
            pose_meta=pose_meta,
            voxel_supersample=voxel_supersample,
            voxel_aa_threshold=voxel_aa_threshold,
            voxel_hole_fill=voxel_hole_fill,
            voxel_symmetry=voxel_symmetry,
            do_tiles=tiles, do_slopes=slopes,
            do_slope_inv=slope_inv, do_baseplate=baseplate,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Brick decomposition failed: {e}")
    out_path = _save_payload(payload, f"{mesh_path.stem}.json")
    payload["_saved_to"] = f"/output/{out_path.name}"
    payload["_mesh_name"] = mesh_path.name
    if photo_path:
        payload["_photo_name"] = photo_path.name
    if face_photo_path:
        payload["_face_photo_name"] = face_photo_path.name
    if other_side_photo_path:
        payload["_other_side_photo_name"] = other_side_photo_path.name
    if body_view:
        payload["_body_view"] = body_view
    if other_side_view:
        payload["_other_side_view"] = other_side_view
    payload["_pose"] = pose_meta
    payload["_semantic_paint_mode"] = semantic_paint_mode
    return JSONResponse(payload)


@app.post("/api/generate-from-mesh")
async def generate_from_mesh(
    mesh: UploadFile = File(...),
    resolution: int = Form(64),
    up: str = Form("auto"),
    subject: str = Form(""),
    voxel_supersample: int = Form(0),
    voxel_aa_threshold: float = Form(0.5),
    voxel_hole_fill: bool = Form(True),
    voxel_symmetry: bool = Form(False),
):
    """Mesh upload -> LEGO bricks. Skips photo->3D when you already have a mesh."""
    stem, suffix = _safe_upload_name(mesh.filename, default_stem="upload", default_suffix=".obj")
    mesh_path = MESHES_DIR / f"{int(time.time())}_{stem}{suffix}"
    with open(mesh_path, "wb") as f:
        f.write(await mesh.read())
    try:
        payload = _run_pipeline(
            mesh_path,
            resolution=resolution,
            up_axis=up,
            subject_type=subject,
            voxel_supersample=voxel_supersample,
            voxel_aa_threshold=voxel_aa_threshold,
            voxel_hole_fill=voxel_hole_fill,
            voxel_symmetry=voxel_symmetry,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Brick decomposition failed: {e}")
    out_path = _save_payload(payload, f"{mesh_path.stem}.json")
    payload["_saved_to"] = f"/output/{out_path.name}"
    return JSONResponse(payload)


@app.post("/api/parts-list")
async def api_parts_list(
    payload: dict = Body(...),
    format: str = "json",
):
    """Convert a bricks payload (possibly edited in the viewer) into a parts list.

    `format`: json | csv | bricklink-xml
    """
    fmt = format.lower()
    if fmt == "csv":
        text = parts_list_csv(payload)
        return Response(text, media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="parts.csv"'})
    if fmt in ("bricklink", "bricklink-xml", "xml"):
        text = parts_list_bricklink_xml(payload)
        return Response(text, media_type="application/xml",
                        headers={"Content-Disposition": 'attachment; filename="bricklink_wanted.xml"'})
    return JSONResponse({"rows": parts_list(payload), "summary": summary(payload)})


@app.post("/api/ldraw")
async def api_ldraw(payload: dict = Body(...)):
    """Return an LDraw .ldr file representation of the bricks (post-edits)."""
    text = to_ldraw(payload)
    return Response(text, media_type="application/x-ldraw",
                    headers={"Content-Disposition": 'attachment; filename="model.ldr"'})


@app.post("/api/stats")
async def api_stats(payload: dict = Body(...)):
    """Return brick count, cost estimate, build time, and stability score."""
    return JSONResponse(build_stats(payload))


@app.post("/api/obj")
async def api_obj(payload: dict = Body(...)):
    """Return a Wavefront OBJ (and its MTL) representation of the bricks."""
    obj_text, mtl_text = to_obj(payload)
    # ZIP both into one file so the browser gets both
    import io as _io
    import zipfile
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.obj", obj_text)
        z.writestr("model.mtl", mtl_text)
    buf.seek(0)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="model.zip"'})
