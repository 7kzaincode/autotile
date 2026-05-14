# lego-ai — Full Project Context for Deep Review

> This document is a self-contained brief for a code-review agent (Codex 4.6)
> that has never seen this codebase. It captures the why, the what, the
> decisions already made, the open issues, and the user's preferences so the
> agent can do an end-to-end audit + improvement pass without re-asking
> obvious questions.

---

## 1. Who & Why

- **Founder**: Zain Khan (zainkhan@waterloo.ca), 2nd-year Computer Engineering at
  University of Waterloo.
- **Target**: YC Fall 2026 application.
- **Product thesis**: "Shutterfly for LEGO" — upload a photo of your pet / car /
  ship / house, get back a buildable LEGO model: brick JSON, .ldr file,
  BrickLink XML wanted-list, parts cost estimate, build instructions.
- **Working directory**: `/Users/zainkhan/lego-ai`
- **Local dev URL**: `http://localhost:8765/viewer/`
- **API docs**: `http://localhost:8765/docs`

The codebase is a working prototype. The user is iterating on quality and
UX before pitching investors. They prefer **organic / pet subjects** as the
flagship use-case ("Shutterfly for LEGO" naturally targets sentimental
items: pets, kids, family).

---

## 2. The Pipeline (Big Picture)

```
photo (.jpg/.webp/.png)
  │
  ▼
[upload]            Save to test_photos/<ts>_front_<name>
  │
  ▼
[rembg]             BRIA-RMBG (HF Space) — strip background
  │                 Keeps a separate `photo_path_for_color` (bg-removed
  │                 but NOT stylized) so color analysis sees real features.
  ▼
[stylize] (opt)     SDXL pre-stylize (default OFF — hurts color fidelity)
  │
  ▼
[photo-to-mesh]     AI 3D model from HF Spaces or Replicate.
  │                 Auto-fallback chain (in photo_to_mesh.py):
  │                   hunyuan3d-2.1-hf → trellis-2-hf →
  │                   hunyuan3d-textured → trellis-hf → sf3d-hf →
  │                   hunyuan3d-hf
  │                 Each tries up to 2× with retry on CUDA OOM, with
  │                 quota-aware skipping when HF Pro budget too small.
  │                 Output: .obj mesh in test_meshes/.
  ▼
[load-mesh]         trimesh load + optional Taubin volume-preserving
  │                 smoothing (kills TRELLIS surface bumps)
  ▼
[voxelize]          trimesh voxelizer at configurable resolution
  │                 (32 chunky / 48 detail / 64 premium). Default 64.
  │                 Supports up=auto (PCA/extents source-axis selection),
  │                 optional 2x anti-aliased voxelization, tight auto-fit,
  │                 conservative 1-voxel hole fill, and voxel-stage X
  │                 symmetry before color projection.
  ▼
[gpt-vision]        ONE call to GPT-4o-mini-vision at detail="high"
  │                 (~$0.005/call, cached by photo SHA + prompt version).
  │                 Returns: subject_name, confidence, regions[] with
  │                 bbox_normalized + color_name + (head/body/ears/etc),
  │                 features{eye_left, eye_right, nose, mouth}, and
  │                 recommended_lego_palette[].
  │                 Prompt now DEMANDS color variation enumeration
  │                 (was the source of the "all tan" bug).
  ▼
[project-photo]     Per-voxel texture sampling. Computes outward face
  │                 normal, samples photo via UV mapping. back_mode="uv"
  │                 mirrors front colors to the back for symmetric subjects.
  ▼
[mirror-geometry]   (opt) Fill missing limbs by mirroring X-axis occupancy
  │
  ▼
[hollow]            (opt) Replace solid interior with single-voxel shell
  │                 plus structural rib bracing every N voxels.
  │                 ~70% piece reduction — biggest cost lever.
  ▼
[mirror-colors]     (opt) Enforce L↔R color symmetry at RGB level
  │
  ▼
[sam-semantic]      SAM 2 (facebook/sam2-hiera-small) refines GPT's
  │                 rectangular bboxes into pixel-perfect anatomical
  │                 masks. Runs LOCALLY on Mac MPS (~1-2s per photo
  │                 after first-call model load of ~10s). Cached in
  │                 sam_cache/<photo_hash>_<boxes_hash>.npz.
  │                 FALLS BACK gracefully to gpt-semantic if it errors.
  │
  ▼ (or)
[gpt-semantic]      Fallback if SAM unavailable: paint label_map from
  │                 GPT's rectangular bboxes (smaller paints last).
  │
  ▼ (or)
[kmeans-semantic]   Final fallback: photo k-means clustering (no AI)
  │
  ▼
[quantize]          Snap each voxel RGB to nearest LEGO palette color
  │                 in CIELAB space, with availability_bias (rare colors
  │                 penalized). restrict_to_ids constrains output to
  │                 the SAM/GPT-recommended palette ⋃ region colors.
  ▼
[mirror-palette]    (opt) Symmetry at palette-ID level (prevents drift
  │                 after per-voxel sampling)
  ▼
[decompose]         Greedy brick decomposition. Alternates 1×N vs N×1
  │                 orientation per layer to break stripe artifacts.
  │                 Produces brick records {x, y, z, size_x, size_y,
  │                 brick_type, kind, rotation, color, slope_dir}.
  ▼
[postprocess]       Replace top-of-column voxels with tiles (no studs),
  │                 stair-steps with slopes, isolated 2x2 tops with
  │                 domes, tapered column tips with cheese slopes.
  │                 Optional: drop floaters, auto-support floaters,
  │                 darken edges (cartoon outline), baseplate.
  ▼
[accent-features]   (opt) Place anatomical pieces — eyes (round_tile,
  │                 1x1 black, mirrored if only one detected), nose
  │                 (round_tile, pink), mouth (1x2 dark tile). Uses
  │                 GPT-supplied positions from `features.*`, mapped
  │                 through the actual photo silhouette bbox (NOT
  │                 full photo dimensions — that was a previously fixed
  │                 bug).
  ▼
[output]            JSON payload {bricks[], palette, grid_shape, pitch}
                    saved to output/<photo>.json.
                    Viewer renders via Three.js with InstancedMesh per
                    brick kind (cylinder for round_*, cone, dome, etc.).
                    Exports: .ldr (LDraw), .obj (3D printable), .csv
                    (parts list), BrickLink Wanted-List XML.
```

---

## 3. Files & Modules

### Core pipeline
| File | Lines | Role |
|------|------:|------|
| [server.py](server.py) | ~570 | FastAPI app: `/api/generate`, `/api/redecompose`, `/api/generate-from-mesh`, `/api/parts-list`, `/api/ldraw`, `/api/obj`, `/api/stats`. The `_run_pipeline()` function is the heart — orchestrates every stage. |
| [photo_to_mesh.py](photo_to_mesh.py) | ~600 | HF Spaces + Replicate dispatch. `HF_PRESETS` maps model names to (space_id, kind). `AUTO_FALLBACK_CHAIN` defines try order. Each `kind` has its own handler (`triposr`, `hunyuan3d_2_1`, `trellis`, `trellis2`, `sf3d`). |
| [mesh_to_voxels.py](mesh_to_voxels.py) | — | `load_mesh()`, `voxelize()`, `choose_auto_up_axis()`, `make_hollow_shell()`. Taubin smoothing, auto up-axis, supersampled voxelization, auto-fit, hole-fill, voxel-stage symmetry. |
| [texture_projection.py](texture_projection.py) | — | `project_photo()`: per-voxel UV sampling with `back_mode` strategies. |
| [voxels_to_palette.py](voxels_to_palette.py) | ~480 | `quantize()` (CIELAB + availability bias), `region_color()` (k-means in (x,y,z,r,g,b) 6D), `mirror_colors()`, `mirror_palette_ids()`, `photo_palette()`, `mirror_occupancy()`. |
| [voxels_to_bricks.py](voxels_to_bricks.py) | — | `decompose()`: greedy brick decomposition with alternating layer orientation. |
| [postprocess.py](postprocess.py) | ~470 | `apply_all()` orchestrates tile/slope/dome/cheese/baseplate replacement + floater handling. |
| [semantic_projection.py](semantic_projection.py) | ~300 | `project_semantic()` (k-means), `project_semantic_gpt()` (GPT bboxes), `project_semantic_sam()` (SAM-refined masks), `gpt_to_restrict_ids()`. |

### AI integrations
| File | Role |
|------|------|
| [gpt_vision.py](gpt_vision.py) | GPT-4o-mini-vision wrapper. SHA-256 cache by photo + prompt version. Prompt is intentionally **aggressive about color enumeration** — pet subjects should never be reported as monochrome. |
| [sam_segmentation.py](sam_segmentation.py) | Local SAM 2 via `transformers` (Sam2Model / Sam2Processor). Lazy singleton load, auto-picks MPS / CUDA / CPU. Cache by `(photo_hash, boxes_hash)`. |
| [features.py](features.py) | `add_anatomical_features()`: detect or read eye/nose/mouth positions and place LEGO pieces at voxel surface. Has both heuristic (dark-spot) and GPT paths. Silhouette-bbox correction is critical — without it eye coords are measured against full photo but applied to voxel grid that only covers subject. |
| [rembg.py](rembg.py) | BRIA-RMBG wrapper. Fallback chain: 2.0 → 1.4 → not-lain. |
| [stylize.py](stylize.py) | SDXL stylization. **Default OFF** — net negative for color fidelity. Kept around as advanced opt. |
| [segment_photo.py](segment_photo.py) | k-means photo segmentation for the legacy semantic path. |

### Auxiliary
| File | Role |
|------|------|
| [subject_preset.py](subject_preset.py) | `resolve(name)` returns a complete config dict for `pet` / `vehicle` / `building` / `other`. Mirrored client-side in `viewer/main.js` `SUBJECT_PRESETS`. |
| [server_logs.py](server_logs.py) | In-memory log ring buffer, SSE streaming endpoint `/api/logs/stream`, `/api/status`, `/api/cache` with clear endpoints. Provides `stage()` context manager + `step()` for verbose pipeline logging. |
| [budget.py](budget.py) | Predefined resolution/color/hollow profiles (tiny / compact / standard / detailed / premium). |
| [build_stats.py](build_stats.py) | Brick count, cost estimate, build time, stability score for a payload. |
| [parts_list.py](parts_list.py) | CSV / BrickLink-XML / JSON parts list exports. |
| [ldraw_export.py](ldraw_export.py) | `.ldr` file export. |
| [obj_export.py](obj_export.py) | `.obj` + `.mtl` export for 3D printing or Blender. |
| [brick_catalog.py](brick_catalog.py) | 74 catalog entries across 11 kinds (round_brick, round_plate, round_tile, cone, cheese_slope, dome, etc.) with LDraw + BrickLink part numbers per (kind, brick_type). |
| [lego_palette.json](lego_palette.json) | 40+ palette entries: id, name, rgb, ldraw, bricklink, availability (common/uncommon/rare). |

### Viewer (frontend)
| File | Role |
|------|------|
| [viewer/index.html](viewer/index.html) | Single-page UI. Light Notion-style theme. Form panel right side, HUD top-left, dev bar bottom (collapsed by default). |
| [viewer/main.js](viewer/main.js) | ~1000 lines vanilla JS + Three.js. Scene + OrbitControls + raycast selection + InstancedMesh-per-kind rendering. SUBJECT_PRESETS applies form defaults. FormData submission to `/api/generate`. Dev-bar SSE log streaming, cache panel, run-status panel. |

### Caches (gitignored typically)
| Dir | Contains |
|-----|----------|
| `gpt_cache/` | `<photo_sha>_<prompt_version>_<detail>.json` — GPT results |
| `sam_cache/` | `<photo_sha>_<boxes_sha>.npz` — SAM masks |
| `test_photos/` | All uploaded + bg-removed + stylized versions |
| `test_meshes/` | AI-generated .obj meshes |
| `output/` | Generated brick JSON files |

---

## 4. Tech Stack

- **Backend**: FastAPI + uvicorn, Python 3.12 in `.venv`
- **3D**: trimesh, numpy, scikit-learn (k-means)
- **AI**: openai (GPT-4o-mini-vision), transformers + torch + torchvision
  (SAM 2 local), gradio_client (HF Spaces), replicate (paid Replicate models)
- **Frontend**: Three.js 0.160 via importmap (no build step), vanilla JS,
  CSS variables (no Tailwind — see §11)
- **Auth**: `.env` holds `HF_TOKEN` (HF Pro), `OPENAI_API_KEY`,
  `REPLICATE_API_TOKEN`
- **Logging**: tee'd stdout → in-memory ring buffer → SSE to browser dev bar

---

## 5. Key Architectural Decisions (and why)

### Three-tier color fallback: SAM → GPT-bbox → k-means
SAM 2 gives pixel-perfect anatomical masks but only if `transformers` + torch
load successfully. GPT bboxes are rectangular but still semantic. K-means is
the dumb-but-always-works backup. Pipeline tries each in order, falls
through on failure. **Important**: failures must not break the pipeline —
each path returns sentinel data (`{"sam_failed": True}` etc.) instead of
raising.

### Two separate photo paths
`photo_path` is the **3D-model-input** photo (may be stylized).
`color_photo_path` is the **bg-removed-but-not-stylized** photo — used for
GPT vision, SAM, k-means segmentation, feature detection. Stylization
destroys eye detail and fine color variation, so we never let the color
pipeline see the stylized version. Falls back to `photo_path` if
`color_photo_path` is None.

### GPT prompt aggressively demands color variation
The bug we fixed last: GPT-4o-mini at low detail returned "Tan" for all 4
regions of a tan bunny. The new prompt (v2) explicitly tells GPT:
> "A 'tan rabbit' is NEVER just tan — it has lighter belly (Cream / White),
> darker paws (Medium Nougat), pink ear interior, pink nose, black eye area,
> white cheek."

Plus we switched to `detail="high"` (~$0.005/call vs $0.0007). Cache key
includes prompt version (`v2`) and detail level so prompt changes auto-
invalidate stale results.

### Subject preset as primary UX control
Four buttons replaced ~15 toggles: `pet / vehicle / building / other`. Each
preset sets resolution, mirror, hollow, semantic_regions, max_colors,
back_mode, etc. The Advanced disclosure exposes everything underneath for
overrides. This is server-side (`subject_preset.py`) AND client-side
(`viewer/main.js SUBJECT_PRESETS`) — they must stay in sync.

### Local SAM, hosted 3D mesh
SAM is small (~200M params, ~700MB weights), runs fine on Mac MPS, and is
deterministic so cache hits are safe. Free forever once weights are pulled.
3D mesh generation needs a real GPU — hosted on HF Spaces (HF Pro covers
25 GPU min/day) or Replicate (pay-per-run backup).

### Auto-fallback chain reordered with Hunyuan first
User asked to favor reliability on 2026-05-12. Order is now:
`hunyuan3d-2.1-hf → trellis-2-hf → hunyuan3d-textured → trellis-hf →
sf3d-hf → hunyuan3d-hf`. Each step checks Space stage (RUNNING vs
CONFIG_ERROR) and skips dead Spaces before trying. CUDA-OOM errors trigger
retry with 10s backoff.

### Stylize OFF by default
Pre-3D SDXL stylization (the "LEGO-ify input" option) is a net negative.
Theory: cleaner photo → cleaner 3D. Reality: it destroys fine color
variation and small features that downstream steps need. Kept in the UI
under "Rarely Needed" with explicit warning text.

---

## 6. Recent Bug Fixes (last few sessions)

| Bug | Root cause | Fix |
|-----|------------|-----|
| All-tan bunny output | GPT prompt under-specified, returned 1 color for all 4 regions | Rewrote prompt to demand color variation; switched to `detail="high"`; bumped cache key with prompt version |
| Eye placement wrong | `bbox=(0,0,W,H)` ignored the subject silhouette inside the photo | Added `_photo_silhouette_bbox()` using alpha channel; coords now mapped relative to subject bbox |
| Devbar wouldn't click + Generate did nothing + no grid rendered | `applySubjectPreset()` called `setStatus()` before `statusEl` const was initialized (TDZ ReferenceError killed all subsequent JS) | Moved setStatus call out of initial preset application; only fires on change |
| Form panel had right-edge overflow | Children with `width: 100%` + `margin-right: 24px` overflow the parent by margin size | Switched from child margins to parent `padding: 0 24px 20px` |
| Form had a scrollbar slider | Content slightly exceeded panel max-height | Hide scrollbar via `scrollbar-width: none` + `::-webkit-scrollbar { display: none }`; content still scrolls via wheel |
| GPT data wasn't being used for COLOR (only features) | `features.py` consumed GPT but `_run_pipeline` skipped it for color | Added `project_semantic_gpt()` and `project_semantic_sam()` that consume `gpt_data` directly for voxel painting + restrict_ids |

---

## 7. Recent Additions (the last big features)

### Subject preset
`pet/vehicle/building/other` dropdown sets all sane defaults. Mirrored
server-side (`subject_preset.py`) and client-side (`SUBJECT_PRESETS` in
`main.js`).

### Dev bar with live logs
`server_logs.py` tees stdout into a 500-entry ring buffer, exposes
`/api/logs/stream` SSE. Browser dev bar at the bottom of the page streams
live, filterable by level (info / warn / error), with auto-scroll and a
clear button. Three tabs: Logs / Cache / Run. The Cache tab shows file
counts + clear buttons for each cache (gpt, meshes, outputs, photos).

### Verbose stage-by-stage logging
Just added (this session): `server_logs.stage()` is a context manager that
prints `▶ [name] description` on entry, `✓ [name] done in 1.23s` on exit,
with timing and counter updates via `step()` calls. Wrapped every step in
`_run_pipeline()` and the `/api/generate` handler. Each log line explains
WHAT the step does and WHY.

### SAM 2 integration
Local pixel-perfect segmentation. Refines GPT's bboxes into actual
anatomical shapes. Free, runs on Mac MPS. Lazy load (first call ~10s incl.
weight download, then ~1-2s/call). Falls back gracefully to GPT bbox path.

### Notion light theme
Switched from dark to a Notion-identical light theme:
- `rgb(255,255,255)` bg, `rgb(247,247,245)` canvas
- `rgb(55,53,47)` text, Notion's exact warm dark gray
- Subtle `rgba(55,53,47,0.09)` borders
- Black primary button (user explicitly asked for black, not blue)
- Title Case headers (no ALL CAPS)
- Custom-styled file inputs via `::file-selector-button`
- Hidden scrollbars, thin scrollbar-on-hover
- Notion's signature shadow:
  `rgba(15,15,15,0.05) 0 0 0 1px, rgba(15,15,15,0.1) 0 3px 6px,
   rgba(15,15,15,0.2) 0 9px 24px`

---

## 8. Known Open Issues / Hot Spots

### Quality
- **Mesh quality from HF Spaces is the ceiling** — TRELLIS / Hunyuan from a
  single photo produce bumpy / asymmetric / weirdly-proportioned 3D models.
  Voxelization + LEGO decomposition can't recover detail the mesh doesn't
  have.
- **Eyes still small on 48-res** — 1x1 round_tile at 1/16 of body width
  reads as a pixel, not an eye. The 2x2 footprint check often fails on
  curved heads so fallback to 1x1 is common. Should explore: white sclera
  base (round_plate) + black pupil (round_tile) sandwich.
- **Eye PLACEMENT is wrong, not just size — confirmed bug as of 2026-05-12.**
  Even with GPT v2 prompt + `detail="high"`, the `features.eye_left` /
  `eye_right` point coordinates GPT returns are inaccurate (typically
  10–15% too high — eyes land on the forehead / between the ears rather
  than on the face). **GPT-4o-mini-vision is not a precision-localization
  model.** This is a known model limitation, not a prompt issue.

  **RECOMMENDED FIX (Option 1 — preferred, free, no new API):**
  Replace the `features.eye_left/eye_right` point coords with the
  **centroid of the SAM `eye_socket` mask**. GPT already returns
  `eye_socket` as a `regions[]` entry with a rough bbox; SAM 2 then
  refines that bbox into a pixel-perfect mask in `sam_cache/*.npz`.
  The mask centroid is a far more accurate eye position than GPT's
  loose point estimate.

  Implementation sketch:
  ```python
  # In features.py add_anatomical_features() — GPT branch:
  # If gpt_data has a region named "eye_socket" or "eyes", and SAM ran,
  # prefer the centroid of that SAM mask over features.eye_left/eye_right.

  def _eye_centroids_from_sam(gpt_data, sam_masks_by_region_name):
      """Returns [(x, y), (x, y)] in pixel coords, or None."""
      regions = gpt_data.get("regions") or []
      eye_regions = [r for r in regions
                     if (r.get("name") or "").lower() in
                        ("eye_socket", "eyes", "eye_left", "eye_right",
                         "eye_socket_left", "eye_socket_right")]
      if not eye_regions:
          return None
      pts = []
      for r in eye_regions:
          mask = sam_masks_by_region_name.get(r["name"])
          if mask is None or mask.sum() == 0:
              continue
          ys, xs = np.where(mask > 0)
          pts.append((int(xs.mean()), int(ys.mean())))
      if not pts:
          return None
      # If one eye region: mirror across X for the pair (existing behavior)
      # If two regions: sort by x so eye_left=leftmost, eye_right=rightmost
      if len(pts) == 1:
          return pts
      pts.sort(key=lambda p: p[0])
      return pts[:2]
  ```

  Wire-up: `server.py _run_pipeline()` already builds the SAM masks list
  inside `project_semantic_sam`. Surface them (e.g. cache in `gpt_data`
  itself as `"_sam_masks_by_region_name"`) so `add_anatomical_features`
  can read them. Then in `features.py`, when `use_gpt=True` and SAM
  masks are present, prefer the SAM-derived eye points over GPT's
  features.eye_* coords. Fall back to GPT points if no eye_socket
  region was named or SAM masks are empty.

  Acceptance test: run on the cached bunny — eyes should land on the
  face (around y≈0.40 of the silhouette bbox, not y≈0.25).

- **Monochrome subjects depend on GPT prompt success** — even with v2
  prompt + `detail="high"`, GPT can still under-segment uniform subjects.
  Worth testing what kinds of photos trip it up.
- **Side/back views of the model are guessed** — `back_mode="uv"` mirrors
  front colors to the back via the symmetry assumption. For asymmetric
  subjects this can look strange.

### Performance
- **3D mesh generation is 30s-5min depending on HF Space queue.** Out of
  our control. Replicate gives ~30s consistent for $0.30/run.
- **SAM first call is ~10s** for weight download + model load. Subsequent
  calls ~1-2s. Cache hits are instant.
- **`add_anatomical_features` has python-loop over voxel occupancy** —
  fine for 48^3, slow for 64^3.
- **`project_semantic_*` has python-loop over occupied voxels** — could
  vectorize with numpy gather.

### Reliability
- **HF Spaces go down regularly** — every BRIA-RMBG, TRELLIS, Hunyuan Space
  has hit `CONFIG_ERROR` at some point. Fallback chains handle this but
  add latency.
- **GPT cache is keyed only by photo bytes** — same photo with different
  subject preset = same cache hit. Reasonable but might surprise.
- **No explicit timeout on HF Space calls** — gradio_client can hang
  indefinitely if a Space is unresponsive (vs CONFIG_ERROR which is
  detectable).
- **No retry on OpenAI rate limit** — would just fail the pipeline.

### Code health
- **Some duplicated UV-projection logic** between
  `project_semantic_gpt` and `project_semantic_sam`. Could extract a
  shared `_project_label_map_onto_voxels` helper.
- **Lots of inline `print()` calls** instead of structured logging.
  `server_logs.step()` exists; the rest of the modules don't use it.
- **`features.py` has both heuristic + GPT branches with overlapping
  logic** — the heuristic path (`detect_features`) might be entirely
  dead code if `use_gpt=True` is always set.
- **No tests.** Zero.
- **Several "rarely needed" options in the UI that probably nobody uses**
  (`use_photo_palette`, `stylize`, `region_colors` — kept as hidden
  inputs for backwards compat).

### UX
- **The viewer can edit/recolor bricks but UX for it is hidden** — drag,
  shift+click multi-select, delete, ctrl+z are the only docs and they're
  in the HUD hint text. No tooltip help on the palette.
- **No undo across re-builds** — re-running pipeline replaces the bricks
  entirely.

---

## 9. The User's Preferences (durable)

- **Communication style**: extremely concise replies, no fluff. Get to
  the point. The user uses CAPS when frustrated — it means they want
  the bug FIXED, not over-explained.
- **UI aesthetic**: light Notion theme. Black primary buttons (NOT blue).
  Title Case for everything. Consistent capitalization. Generous
  breathing room (proper padding). No scrollbars on the main form.
- **Cost-aware**: has a $5 OpenAI budget, has HF Pro ($9/mo), prefers
  free / already-paid solutions when possible. SAM local > SAM Replicate
  for this reason. HF Pro Spaces > Replicate 3D models.
- **Quality > speed**: willing to wait 5min for a good mesh if it means
  better output.
- **Pet/organic subjects first.** Vehicles/buildings second. Wants the
  flagship use case nailed before broadening.
- **Stop suggesting "have you tried Tailwind/React" as fixes** — they're
  framework changes, not bug fixes. Address actual issues in the existing
  vanilla stack.
- **Wants verbose logs during generation** so they can debug what's
  happening at each stage in real time. Just shipped this in the dev bar.

---

## 10. Things Already Considered + Rejected

| Idea | Why not (yet) |
|------|---------------|
| Replicate TRELLIS as primary | HF Pro is paid and works; Replicate adds $0.30/run with no quality upside while HF Pro quota lasts |
| SAM 3D Objects (Meta) | Requires GPU + PyTorch3D + no good hosted API yet; revisit if a Replicate port exists |
| React + Tailwind refactor | 2-3 days of work, won't fix the actual CSS bugs which are model issues not framework issues |
| Self-host 3D mesh GPU box | Premature optimization; doesn't make sense until paying-customer volume |
| LPub3D headless instruction generation | Nice-to-have for the final product; not blocking the demo |
| BrickLink stock API integration | Same — defer until first customer |
| EVF-SAM-2 (text-prompted) | Community Space, less stable than transformers route. The point-prompt path through GPT bboxes works |

---

## 11. The Browser/UI Refactor Conversation

The user asked about React + Tailwind. I pushed back with:

> The ugliness was a CSS box-model bug I introduced. Tailwind is just
> utility classes — same model, same bugs possible. Don't refactor the
> framework to fix CSS bugs.

The decision was to keep vanilla JS + CSS variables, fix the bugs in the
existing stack. **If a future refactor is appropriate, it should be a
planned migration (component reuse, hiring) — not a "make it look better"
patch.**

---

## 12. What's Working Right Now (smoke-tested)

- ✅ Server starts clean on port 8765
- ✅ Form panel renders with Notion theme, no scrollbar, no overflow
- ✅ Dev bar at the bottom expands on click, streams logs live
- ✅ Subject preset dropdown applies form defaults on change
- ✅ GPT vision call returns structured JSON (tested with bunny photo)
- ✅ SAM 2 loads on MPS, segments boxes into masks in ~1.5s (tested)
- ✅ Hunyuan3D 2.1 generates meshes (5min queue + ~30s actual)
- ✅ Brick decomposition + post-processing produces ~2500-3000 bricks at
  resolution 48 with hollow=true
- ✅ Three.js viewer renders bricks with per-kind InstancedMesh
- ✅ Exports: .ldr, .obj, .csv, BrickLink XML all working
- ✅ Verbose stage-by-stage logging with timing now in dev bar

---

## 13. Environment + Setup

```bash
# Project venv (already exists)
.venv/bin/python

# Run server
.venv/bin/uvicorn server:app --port 8765 --host 127.0.0.1

# Logs (server stdout is tee'd to BOTH terminal and /tmp/lego-server.log,
# and into the in-memory ring for the dev bar)
tail -f /tmp/lego-server.log

# Kill server
lsof -ti tcp:8765 | xargs -r kill -9

# Run a Python sanity check
.venv/bin/python -c "import server; print(sorted(r.path for r in server.app.routes))"
```

`.env` contents (real tokens):
```
HF_TOKEN=hf_***                # HF Pro, 25 GPU min/day
OPENAI_API_KEY=sk-proj-***     # $5 budget
REPLICATE_API_TOKEN=r8_***     # paid backup
```

---

## 14. What I Want You (Codex 4.6) to Do

See the **review command** at the bottom of this document.
