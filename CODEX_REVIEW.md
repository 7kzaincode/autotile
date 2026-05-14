# lego-ai Review

Date: 2026-05-12

## 1. Executive Summary

- Implemented the priority eye-placement fix: SAM eye-region masks are now
  surfaced from semantic projection and `features.py` prefers their centroids
  over GPT point coordinates.
- Implemented the first wave of improvement proposals: shared semantic
  projection helper, `/api/presets`, frontend preset hydration, and upload
  filename normalization.
- Implemented the 3D-model → voxelization quality pass: Auto up-axis,
  optional supersampling, tight auto-fit, pinhole fill, voxel-stage symmetry,
  64-res defaults, voxel metadata, and front-mounted feature accents.
- Fixed several correctness bugs in the server/viewer/export paths: Advanced
  overrides now survive subject presets, anonymous HF Spaces work, OBJ
  baseplates sit below the model, and baseplates are not counted as floaters.
- Tightened tests so they run against timestamped cached artifacts instead of
  assuming `bunny.obj` / `bunny.json` always exist.
- Added the missing `pytest` requirement, installed it in the venv, and verified
  the documented test command: `32 passed`.
- Could not stage changes because `/Users/zainkhan/lego-ai` is not a Git repo
  in this checkout (`git status` returns "not a git repository").

## 2. Bugs Fixed

### Mesh Up-Axis And Voxel Quality Pass

Files:
- `mesh_to_voxels.py`
- `server.py`
- `subject_preset.py`
- `viewer/index.html`
- `viewer/main.js`

What was wrong:
- The problematic Hunyuan bunny mesh was Z-up, but the old default forced
  `up=y`, producing a squat/deep voxel grid. Semantic projection also tied
  "front" to the requested source up-axis, which would paint vertically when
  switching to `up=z`.

Fix:
- Added `up=auto` with PCA/extents source-axis detection.
- Kept voxel-space front canonical as `-y`.
- Added optional 2x anti-aliased voxelization, tight auto-fit, conservative
  one-voxel hole-fill, and voxel-stage X symmetry before color projection.
- Default subject presets now use resolution 64 and `up=auto`.

Verification:
- Cached bunny mesh smoke chose `up_axis z` and produced a tall grid
  `[11, 18, 33]` at 32-res, instead of the old squat/deep orientation.

### Feature Accents Were Visually Embedded

Files:
- `features.py`
- `viewer/main.js`

What was wrong:
- Eye/nose/mouth accents were appended at the same voxel coordinate as the
  face surface. In the viewer they rendered as normal horizontal pieces, so
  black eye pieces could look buried inside body bricks.

Fix:
- Anatomical accents now carry `mount="-y"`.
- The viewer renders mounted pieces as thin front-face pieces, with no studs,
  offset outside the face surface.

### SAM Eye Centroids Replace GPT Eye Points

Files:
- `semantic_projection.py:337`
- `semantic_projection.py:397`
- `features.py:258`
- `features.py:333`

What was wrong:
- `project_semantic_sam()` computed good SAM masks, but discarded them after
  painting colors. `add_anatomical_features()` still used GPT point landmarks,
  which were known to land eyes too high.

Before:
```python
label_map, regions = _build_sam_label_map(masks, valid_regions, H, W)
gpt_feats = features_from_gpt(gpt_data, (H, W))
```

After:
```python
gpt_data["_sam_masks_by_region_name"] = masks_by_region_name
sam_eye_pts = _eye_centroids_from_sam(gpt_data)
gpt_feats = {**gpt_feats, "eyes": sam_eye_pts}
```

Verification:
- Ran cached bunny through `_run_pipeline(...)`.
- Confirmed log lines:
  `[features] using SAM eye_socket centroid: (375, 557) instead of GPT point`
  and `(436, 543)`.
- Result placed two black `round_tile` eye accents at voxel z=24.

### Subject Preset Advanced Overrides Were Ignored

Files:
- `server.py:348`
- `server.py:423`
- `server.py:576`
- `server.py:618`
- `viewer/main.js:834`
- `viewer/main.js:893`

What was wrong:
- The viewer applied the selected preset into the form, then sent
  `subject=pet`. The server applied the preset again, overwriting any Advanced
  fields the user changed.

Fix:
- Added `preset_client_applied` form flag.
- Viewer sends `preset_client_applied=true`.
- Server applies `subject` defaults only when that flag is false, preserving API
  behavior for callers that send only `subject`.

### Anonymous HF Space Calls Crashed

File:
- `photo_to_mesh.py:164`

What was wrong:
- `_run_hf_space()` only created `client` inside `if hf_token`. If no token was
  set, public Spaces crashed with `UnboundLocalError`.

Fix:
```python
else:
    client = Client(space_id)
```

Also fixed a Replicate file-handle leak in `photo_to_mesh.py:493` by opening
the upload image in a `with` block.

### Tracebacks Did Not Reach Dev Bar

Files:
- `server_logs.py:117`
- `server_logs.py:255`

What was wrong:
- `traceback.print_exc()` writes to stderr, but only stdout was tee'd into the
  browser log ring. Also, invalid cache kinds returned a tuple payload with HTTP
  200 instead of a real 400.

Fix:
- `server_logs.install()` now tees both stdout and stderr.
- Unknown cache kind now raises `HTTPException(400, ...)`.

### OBJ Baseplate Was Vertically Wrong

Files:
- `obj_export.py:64`
- `obj_export.py:71`
- `tests/test_pipeline.py:128`

What was wrong:
- `base_y` for z=-1 baseplates was computed, then ignored. Baseplate vertices
  started at y=0 instead of below the model.

Fix:
```python
y_voxel = base_y + height * dz
```

Added a regression assertion that OBJ output with a baseplate has negative y
vertices.

### Baseplates Counted As Unsupported

Files:
- `build_stats.py:113`
- `viewer/main.js:502`

What was wrong:
- Baseplates use z=-1, but support checks only treated z=0 as grounded. This
  could inflate unsupported counts in stats and the HUD.

Fix:
```python
if b["z"] <= 0: continue
if (b.z <= 0) return;
```

### Viewer Parts List Merged Different Piece Kinds

File:
- `viewer/main.js:648`

What was wrong:
- The local viewer parts list grouped only by `(brick_type, color)`, merging
  `brick 1x2`, `tile 1x2`, `plate 1x2`, etc.

Fix:
- Group by `(kind, brick_type, color)`.
- Include kind in the displayed row.
- Mirror backend kind price multipliers for better local cost estimates.

### Dev Bar Markup Escaping

Files:
- `viewer/main.js:1148`
- `viewer/main.js:1177`

What was wrong:
- Cache latest filenames and run status values were inserted via `innerHTML`
  without escaping. Upload names are local, but they can still contain markup.

Fix:
- Escaped dynamic cache/status values with `escapeHtml()`.

### Tests Assumed Stale Artifact Names

Files:
- `tests/test_pipeline.py:31`
- `tests/test_pipeline.py:134`
- `tests/test_pipeline.py:270`
- `tests/test_server.py:19`

What was wrong:
- Offline tests assumed `test_meshes/bunny.obj`; HTTP tests assumed
  `/output/bunny.json`. This checkout has timestamped bunny artifacts.

Fix:
- Tests now discover `*bunny*.obj` and `*bunny*.json`.
- The decomposition orientation test now uses a synthetic grid instead of
  depending on one bunny mesh's shape.

### Shared Semantic Projection Helper

Files:
- `semantic_projection.py:82`
- `semantic_projection.py:134`
- `semantic_projection.py:219`
- `semantic_projection.py:273`
- `semantic_projection.py:418`

What changed:
- Extracted the duplicated label-map-to-voxel loop into
  `_project_label_map_onto_voxels()`.
- GPT bbox, k-means semantic, and SAM semantic paths now call the same helper.
- Added `tests/test_pipeline.py::test_semantic_label_projection_shared_helper`.

### Backend-Served Presets

Files:
- `server.py:80`
- `viewer/main.js:128`
- `tests/test_api_unit.py:10`

What changed:
- Added `GET /api/presets` using `subject_preset.PRESETS`.
- Viewer hydrates preset defaults from the backend, while preserving local
  fallback values if the request fails.

### Upload Filename Normalization

Files:
- `server.py:94`
- `server.py:422`
- `server.py:703`
- `tests/test_api_unit.py:19`

What changed:
- Added `_safe_upload_name()` to normalize stems and suffixes for uploaded
  photos and uploaded meshes.
- New uploads avoid markup/unicode surprises in saved filenames and UI cache
  lists.

## 3. Bugs Found But Not Fixed

- Medium: `texture_projection._project_photo_uv()` accepts `front_axis` but
  still assumes voxel front is `-y` internally (`texture_projection.py:144` and
  `texture_projection.py:224`). The current server path effectively uses that
  convention, so I left it alone; fix with the shared UV helper proposed in
  `IMPROVEMENTS.md`.
- Medium: `gradio_client.Client.predict()` calls have no explicit timeout in
  `photo_to_mesh.py` and `rembg.py`. If a Space hangs rather than returning a
  runtime state or error, the request can still hang. Recommended action:
  isolate calls in a worker with a timeout/retry boundary.
- Low: several inline `style=""` attributes remain in `viewer/index.html`
  (`viewer/index.html:434`, `viewer/index.html:606`, `viewer/index.html:622`,
  etc.). They are not breaking behavior, but a polish pass should move them to
  CSS classes.
- Low: upload filenames are path-traversal safe via `Path(...).stem`, but not
  normalized (`server.py:401`). Recommended action: add a `_safe_upload_stem()`
  sanitizer before the app is public.

## 4. Consistency Issues Fixed

- Client/server subject preset behavior now matches the UI promise: preset
  buttons set defaults, and Advanced overrides remain effective.
- Test fixtures now match the actual cache naming style produced by the app.
- Viewer local parts list now matches backend `parts_list.py` grouping by kind.
- DOM ID audit passed: every `$('id')` in `viewer/main.js` exists in
  `viewer/index.html`.

## 5. Architecture Proposals

See `IMPROVEMENTS.md` for full details. Summary:
- Extract shared semantic UV projection helper. Implemented.
- Serve subject presets from `/api/presets` to remove duplication. Implemented.
- Replace HF preset tuples with a dataclass.
- Vectorize semantic projection and mirror loops if 64-res performance matters.
- Add structured logger helpers around the stdout/stderr tee.
- Normalize upload filenames. Implemented.

## 6. Dead Code Candidates

- `features.detect_features()` and `detect_dark_spots()` are fallback paths now
  that GPT vision is on by default. Keep for no-key/offline runs, but they are
  cold path.
- `region_colors` is hidden in `viewer/index.html:640` and still threaded
  through `server.py`; likely removable after confirming no saved workflows use
  it.
- `use_photo_palette` and `stylize` remain under Rarely Needed. They are useful
  escape hatches, but both overlap with the newer GPT/SAM semantic color path.

## 7. Performance Observations

- Cached bunny local pipeline at 48-res completed the non-AI brick stages in a
  few seconds: voxelize 0.07s, SAM cache hit 0.10s, quantize 0.13s, decompose
  0.19s in the observed run. Hosted photo-to-mesh remains the real bottleneck.
- `semantic_projection.py` still loops over occupied voxels in Python. At
  48-res this is fine; at 64-res it is the first local vectorization target.
- Palette loading is repeated through `load_palette()`. It is tiny JSON, so not
  worth changing unless profiling shows repeated runs spending time there.

## 8. Open Questions

- Should baseplate be part of default `pet` preset, or stay manual?
- Should Advanced overrides always win for API callers too, or is the current
  split right: API `subject` applies presets unless `preset_client_applied=true`?
- Do you want inline styles cleaned up now, or is behavior/quality still the
  only priority for the pitch demo?
- Should we add a small generated preview/screenshot test for eye placement, or
  is the current pipeline log plus accent-coordinate verification enough?

## 9. What I Tested

Commands:
```bash
.venv/bin/python -c "import server; print('ok')"
.venv/bin/python -m py_compile server.py photo_to_mesh.py mesh_to_voxels.py texture_projection.py voxels_to_palette.py voxels_to_bricks.py postprocess.py semantic_projection.py gpt_vision.py sam_segmentation.py features.py rembg.py stylize.py segment_photo.py subject_preset.py server_logs.py budget.py build_stats.py parts_list.py ldraw_export.py obj_export.py brick_catalog.py
node --check viewer/main.js
.venv/bin/python tests/test_pipeline.py
.venv/bin/python tests/test_server.py
.venv/bin/python -m pytest -q
```

Results:
- `import server`: ok.
- `node --check viewer/main.js`: ok.
- Standalone offline tests: 15/15 passed.
- Standalone HTTP tests against the running local server: 8/8 passed.
- Pytest suite: 27 passed in 2.99s.
- Cached bunny pipeline verification: SAM eye centroid path fired and produced
  two black round-tile eye accents.
