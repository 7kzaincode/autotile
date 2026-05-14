# Codex 4.6 Review Command — lego-ai

Paste the prompt below to Codex 4.6 after pointing it at this repo. The
prompt assumes Codex has read [CODEX_CONTEXT.md](CODEX_CONTEXT.md) first.

---

## The Command (paste this verbatim)

```
You are doing a deep, end-to-end review of the lego-ai codebase at
/Users/zainkhan/lego-ai. This is a multi-hour, autonomous review session —
take your time, do it thoroughly, and prefer correctness over speed.

STEP 1 — Read the context first.
Before touching code, read CODEX_CONTEXT.md in full. That document captures:
- the project's purpose (photo → LEGO, YC Fall 2026)
- the full pipeline (rembg → 3D mesh → voxelize → GPT → SAM → decompose
  → postprocess → accent features → export)
- every key architectural decision and WHY it was made
- the user's preferences (Notion light theme, black buttons, cost-aware,
  pet/organic focus, verbose logs, concise replies)
- recent bug fixes and what they teach about the codebase
- open issues already identified
- decisions already considered and rejected (DO NOT re-propose them
  unless you have a genuinely new angle)

If anything in the context document is unclear or you suspect is wrong,
flag it before touching code.

STEP 2 — Build a mental map.
Read every Python module in the project root (server.py, photo_to_mesh.py,
mesh_to_voxels.py, texture_projection.py, voxels_to_palette.py,
voxels_to_bricks.py, postprocess.py, semantic_projection.py, gpt_vision.py,
sam_segmentation.py, features.py, rembg.py, stylize.py, segment_photo.py,
subject_preset.py, server_logs.py, budget.py, build_stats.py, parts_list.py,
ldraw_export.py, obj_export.py, brick_catalog.py) and the viewer files
(viewer/index.html, viewer/main.js). Build an internal model of:

- the call graph (what calls what)
- the data shapes flowing between modules (voxel grid, brick records,
  payload dict)
- which functions are dead code vs hot path
- where state is implicit / shared / module-global

STEP 3 — Aggressive bug hunt.
For each module, look for:

1. CRASHES & EXCEPTIONS
   - Unhandled exceptions that could break the pipeline mid-run
   - Off-by-one errors in voxel indexing
   - Numpy axis confusion (x vs y vs z; H vs W)
   - PIL image mode confusion (RGB vs RGBA vs L)
   - `KeyError` / `AttributeError` from optional dict fields
   - Implicit None propagation (functions returning None silently)
   - Race conditions in the SSE log streaming
   - Threading / async issues in server_logs.py (the stdout tee is
     mutating sys.stdout globally — review carefully)

2. LOGIC BUGS
   - Silent fallthrough that masks real failures (e.g. SAM errors out
     and pipeline continues with bad data)
   - Cache key issues that cause stale data after legitimate changes
     (gpt_cache, sam_cache, mesh cache — verify all are versioned/keyed
     correctly)
   - Coordinate-system mismatches (photo coords vs voxel coords vs
     normalized vs pixel)
   - Symmetry / mirror operations that destroy intentional asymmetry
   - Float / int conversion losing precision
   - "Empty mask" or "empty regions" cases producing weird results
     instead of erroring cleanly

3. CONSISTENCY ISSUES
   - Subject-preset client-side (viewer/main.js SUBJECT_PRESETS) vs
     server-side (subject_preset.py PRESETS) — they MUST match;
     verify exhaustively
   - Form param defaults in /api/generate vs /api/redecompose — these
     have diverged before
   - Label/option capitalization in viewer/index.html — user has
     explicitly asked for Title Case consistency
   - HF model IDs and `kind` mappings in HF_PRESETS vs the per-kind
     handler dispatch — verify each kind has a handler
   - File path handling — Pathlib vs str inconsistency

4. SECURITY
   - The /api/generate-from-mesh endpoint accepts arbitrary uploads.
     Is the path traversal protection adequate? (_safe_filename)
   - Are uploaded photo filenames sanitized?
   - Any os.system / shell=True usage? (there shouldn't be)
   - Secrets accidentally logged?

5. PERFORMANCE
   - Python loops over occupied voxels in semantic_projection.py and
     features.py — these are 10k-50k iterations. Worth vectorizing?
   - The dev-bar log SSE — does it leak subscribers on disconnect?
   - File I/O in tight loops?
   - Re-reading the palette JSON on every pipeline run instead of caching?

6. DEAD / REDUNDANT CODE
   - The "Rarely Needed" UI section in index.html (stylize, photo-palette,
     budget) has overlapping behavior with semantic/GPT. Could it be
     simplified or removed?
   - `region_colors` flag is now hidden but still threaded through. Worth
     removing entirely?
   - `features.detect_features` (heuristic) vs `gpt_vision` GPT path —
     is the heuristic ever exercised?
   - Any imports that don't get used?

STEP 4 — Consistency + UX audit.
Read viewer/index.html top to bottom:
- Every label, every option, every hint text — Title Case where expected?
- Every form input has both a server-side handler AND main.js wiring?
- Every hidden disclosure (Advanced, Import/Export JSON) actually
  works when opened?
- No inline `style=""` attributes left that should be in the CSS?
- Buttons sized / padded consistently?
- The dev bar tabs (Logs / Cache / Run) all functional?

Read viewer/main.js bottom to top:
- Every $('id') refers to an element that actually exists in the HTML?
- Event listeners all reference current selectors?
- FormData fields all match the corresponding server Form() params?
- No leftover references to removed elements (apply-test-preset-btn was
  one — verify no others)?

STEP 5 — Architecture proposals.
If you spot any of these patterns, propose a concrete refactor (with code):

- The same UV projection code copy-pasted across project_semantic*
  variants. Extract a shared helper.
- Pipeline stages are inline in _run_pipeline as if/with blocks. Could
  this be a pipeline registry where each stage is a self-contained
  class? (Only propose if it would simplify, not complicate.)
- The HF_PRESETS / handler dispatch in photo_to_mesh.py uses string-based
  `kind`. Could be enum + dataclass.
- Settings duplication between server-side and client-side subject_preset.
  Could the server expose `/api/presets` and the client fetch it,
  eliminating the duplication?

Architecture proposals must include:
- WHY the current code is bad (specific lines)
- WHAT the new shape looks like (sketch the new API)
- COST of the migration (lines changed, risk)
- WHETHER to do it now or defer

STEP 5b — PRIORITY FIX: Eye placement from SAM mask centroids.

The single most user-visible quality bug right now is eye placement:
GPT-4o-mini-vision returns eye point coords that are ~10-15% too high,
so eyes land on the forehead instead of the face. CODEX_CONTEXT.md §8
("Eyes still small on 48-res" section) has a full implementation sketch
labeled "RECOMMENDED FIX (Option 1)" — read it now.

Summary of what to do:
1. In project_semantic_sam (semantic_projection.py), after computing SAM
   masks per GPT region, expose them by region name (e.g. store on
   gpt_data["_sam_masks_by_region_name"] or return as second tuple element).
2. In add_anatomical_features (features.py), when GPT data + SAM masks
   are both available, find the mask whose region is named one of
   ("eye_socket", "eyes", "eye_left", "eye_right", "eye_socket_left",
   "eye_socket_right"). Compute its centroid in pixel space. Use that
   instead of GPT's features.eye_left / eye_right point coords.
3. Preserve existing fallback to GPT point coords when no eye_socket
   region exists or its SAM mask is empty.
4. Sort the centroid points by x so eye_left=leftmost, eye_right=rightmost.
5. Add a print line: "[features] using SAM eye_socket centroid: (x, y)
   instead of GPT point" so the user can see in the dev bar which path
   ran.

Acceptance: on the cached bunny photo
(test_photos/*bunny*nobg.png), the eyes should land near the middle of
the face, not on the forehead. Verify by re-generating and inspecting
the rendered model in the viewer.

This is a PRIORITY fix — do it before architecture proposals. It's
~30 lines across two files and unblocks the user's biggest visual
complaint.

STEP 6 — Implement what's clearly correct.
For each bug you find with high confidence (>90%): fix it. Run the
sanity check after each fix:

    .venv/bin/python -c "import server; print('ok')"

If imports break, fix immediately. Never leave the codebase in a broken
state.

For each improvement with medium confidence: write the fix in a way that
preserves the old code path as a fallback, with a print statement
explaining what changed. Let the user choose to commit later.

For each architectural proposal: DO NOT implement without explicit
approval. Write it up in a new file IMPROVEMENTS.md with the proposal
+ pros/cons + how the user can opt in.

STEP 7 — Verbose logging audit.
The user wants very verbose, debuggable logs during generation. We just
shipped server_logs.stage() and step(). Audit:
- Every major pipeline step has a stage() wrapper? (server.py)
- Each photo_to_mesh handler (triposr, hunyuan3d_*, trellis*, sf3d, etc.)
  prints what it's about to do BEFORE the long HF call?
- Errors are caught + logged with traceback, not swallowed?
- Stage timings are accurate (not double-counted)?
- The dev-bar SSE doesn't drop log lines under burst load?

STEP 8 — Write the report.
Create a new file CODEX_REVIEW.md at the project root containing:

1. Executive summary (what you found, in 5 bullets)
2. Bugs fixed (with file:line and what was wrong)
3. Bugs found but not fixed (with severity + recommended action)
4. Consistency issues fixed
5. Architecture proposals (with pros/cons each)
6. Dead code candidates (with justification)
7. Performance observations (with benchmark estimates)
8. Open questions for the user
9. What you tested + how

Be concrete. File paths, line numbers, before/after snippets. The user
will read this start-to-finish and act on it.

STEP 9 — DO NOT push to git or commit without permission.
Stage changes locally only. The user reviews and commits manually.

STEP 10 — Take your time.
This is a long autonomous run. There is no rush. If you find yourself
about to skip a module because "it's probably fine" — DON'T. Read it.
The cost of an extra 30 seconds reading is much lower than the cost of
shipping a bug to YC.

User preferences for your work:
- Concise communication (the user has limited patience for fluff)
- Direct, specific findings — not vague "could be improved"
- Show file:line for everything
- Don't suggest framework refactors (React/Tailwind) as bug fixes — see
  context doc §10/§11
- Black buttons (not blue) for primary actions in UI
- Title Case in UI labels
- Notion-light aesthetic — stay consistent if you touch any CSS

GO.
```

---

## How to Use This

1. Open Codex 4.6 in this directory: `/Users/zainkhan/lego-ai`
2. Paste **CODEX_CONTEXT.md** as the first message (or attach it)
3. Paste **the command above** as the second message
4. Walk away — let it run

When it finishes, it will leave:
- A `CODEX_REVIEW.md` at the project root with findings + fixes
- A `IMPROVEMENTS.md` with architectural proposals awaiting your approval
- Local changes staged (not committed) for the high-confidence fixes

Review at your leisure and commit selectively.
