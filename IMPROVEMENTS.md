# lego-ai Improvements

Architecture proposals from the 2026-05-12 review. Status reflects the
follow-up implementation pass.

## 1. Extract Shared Semantic UV Projection — Done

What changed:
- Added `_project_label_map_onto_voxels()` in `semantic_projection.py`.
- `project_semantic_gpt()`, `project_semantic()`, and `project_semantic_sam()`
  now share the same label-map projection logic.
- Added `test_semantic_label_projection_shared_helper()`.

Why this mattered:
- `semantic_projection.py:120-174`, `semantic_projection.py:218-287`, and
  `semantic_projection.py:429-487` repeat the same occupied-voxel to photo-pixel
  projection logic with small variations.
- This makes bug fixes risky: bbox alignment, background fallback, or
  `front_axis` behavior has to be changed in multiple places.

New shape:
```python
def _project_label_map_onto_voxels(
    grid,
    label_map: np.ndarray,
    region_to_rgb: dict[int, tuple[int, int, int]],
    *,
    mirror_back: bool = True,
) -> dict:
    ...
```

Migration cost:
- About 80-120 lines touched in `semantic_projection.py`.
- Low runtime risk if covered by one synthetic label-map test and one cached
  bunny smoke test.

## 2. Serve Presets From The Backend — Done

What changed:
- Added `GET /api/presets`.
- `viewer/main.js` hydrates its presets from the endpoint and keeps bundled
  defaults as a fallback.
- Added `test_presets_endpoint_uses_subject_preset_source()`.

Why this mattered:
- `subject_preset.py:12` and `viewer/main.js:70` duplicate the same preset data.
- The review fixed the worst override bug by adding `preset_client_applied`, but
  the duplication can still drift.

New shape:
```python
@app.get("/api/presets")
def api_presets():
    return {"presets": PRESETS}
```

The viewer would fetch this once at startup, normalize server keys to DOM IDs,
and keep its local default only as a fallback if the request fails.

Migration cost:
- About 60-100 lines in `server.py` and `viewer/main.js`.
- Medium UI risk: startup preset application is currently synchronous, so the
  fetch needs a small loading/fallback path.

## 3. Dataclass HF Model Presets — Deferred

Why current code is costly:
- `photo_to_mesh.py:33` uses `(space_id, kind)` tuples and a string dispatch
  chain in `_run_hf_space`.
- Unknown `kind` values are only caught at runtime.

New shape:
```python
@dataclass(frozen=True)
class HFModelPreset:
    key: str
    space_id: str
    handler: Literal["triposr", "trellis", "trellis2", ...]
    quota_seconds: int
```

Migration cost:
- About 80 lines in `photo_to_mesh.py`.
- Low user-facing risk but easy to do wrong because every HF endpoint has
  different parameters.

Recommendation:
- Defer. The current string dispatch is not pretty, but it is explicit and the
  active bug in anonymous HF client creation is fixed.

## 4. Vectorize Semantic And Mirror Loops — Deferred

Why current code is costly:
- `semantic_projection.py` loops over every occupied voxel in Python.
- `voxels_to_palette.py:144`, `voxels_to_palette.py:257`, and
  `voxels_to_palette.py:282` do nested Python loops over grid coordinates.

New shape:
- Precompute `occ_idx = np.argwhere(occupancy)`.
- Compute mirrored x, u/v pixel arrays, label ids, and RGB assignment with
  numpy gather/scatter.
- Use array slicing for mirror operations.

Migration cost:
- About 120-180 lines across `semantic_projection.py` and
  `voxels_to_palette.py`.
- Medium risk because coordinate orientation bugs are easy here.

Recommendation:
- Defer unless 64-res redecompose becomes too slow. At 48-res, local cached
  pipeline verification stayed interactive.

## 5. Structured Logger Wrapper — Deferred

Why current code is costly:
- The dev bar now captures stdout and stderr, but most modules still use raw
  `print()`.
- `server_logs.stage()` gives useful timings, while module-level logs are
  free-form.

New shape:
```python
server_logs.info("message", module="sam", model=model_id)
server_logs.warn("message", module="hf", attempt=attempt)
server_logs.error("message", exc=e)
```

Migration cost:
- About 100 lines if done incrementally.
- Low risk if `print()` remains as a fallback.

Recommendation:
- Defer. The current dev bar is useful after stderr capture; structured logs
  would polish operations but are not blocking model quality.

## 6. Upload Filename Normalization — Done

What changed:
- Added `_safe_upload_name()` in `server.py`.
- Photo uploads and mesh uploads use sanitized stems and safe suffixes.
- Added unit tests for unicode/markup names and invalid suffixes.

Why this mattered:
- `server.py:398` uses `Path(upload.filename).stem`, which avoids path traversal
  but preserves odd characters in saved names.
- The viewer now escapes cache/status filenames, but cleaner names would reduce
  log/UI surprises.

New shape:
```python
def _safe_upload_stem(name: str) -> str:
    stem = Path(name or "upload").stem
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" .") or "upload"
```

Migration cost:
- Under 20 lines in `server.py`.
- Low risk, but existing cached files keep their old names.

Remaining note:
- Existing cached files keep their old names. New uploads get normalized.

## 7. 3D Mesh To Voxel Quality Pass — Done

What changed:
- Added `up=auto` in `mesh_to_voxels.py`, using PCA/extents to choose the
  source up-axis. Pets/buildings favor the long upright axis; vehicles favor
  the short height axis.
- Added optional anti-aliased voxelization via 2x supersampling and coverage
  downsampling. The viewer exposes this as Auto / Off / 2x Smooth.
- Added tight auto-fit cropping, conservative 1-voxel pinhole filling, and
  voxel-stage X symmetry before photo/color projection.
- Raised subject presets and default form/API resolution from 48 to 64.
- Added `voxel_metadata` to generated payloads and the Run tab now shows chosen
  up-axis, voxel shape, and anti-aliasing level.
- Anatomical accents now carry `mount="-y"` and the viewer renders them as
  front-mounted pieces instead of visually embedding them inside face bricks.

Why this mattered:
- The latest Hunyuan bunny mesh had Z as the true vertical axis, but the old
  default `up=y` produced a squat/deep model. Auto mode selects `z` for that
  cached mesh and produces a tall grid.
- Mirroring geometry after photo projection meant filled-in limbs could inherit
  stale colors. Voxel-stage symmetry lets the normal projection passes paint
  both sides.
- Black eye/nose/mouth accents previously occupied the same voxel as the body
  surface, making them appear buried in the model.

Migration cost:
- About 250 lines across `mesh_to_voxels.py`, `server.py`, `subject_preset.py`,
  `features.py`, `viewer/index.html`, `viewer/main.js`, and tests.
- Runtime cost is configurable: Auto supersampling uses 2x only at 48-res or
  lower; 64-res defaults to no supersampling unless manually enabled.

Remaining note:
- The implementation keeps LEGO studs buildable. It does not introduce true
  non-cubic physical voxels, because that would distort the model or break
  brick compatibility; instead it preserves aspect through per-axis grid
  dimensions, tight fitting, and exported voxel metadata.
