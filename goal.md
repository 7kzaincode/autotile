# Overnight Goal

Make the LEGO pet generator end-to-end product-usable for the provided cat front/side photos, while improving the general pipeline for future pets.

Workspace:

- `/Users/zainkhan/lego-ai`

Primary test photos:

- Front face photo: `/Users/zainkhan/lego-ai/animal testing photos/cat/catfront.png`
- Side body photo: `/Users/zainkhan/lego-ai/animal testing photos/cat/cat_side.png`

## Core Objective

When Zain returns, the app should produce a visibly credible LEGO pet model with:

- accurate body shape/proportions
- clean LEGO-like coloring
- faithful side markings
- correct white/cream chest, belly, paws, and muzzle placement
- no random wedge/slope artifacts
- no fake mirrored markings
- a believable face solution, or a clearly implemented safe fallback if true face placement cannot be made reliable on side-body meshes

Work autonomously and iterate overnight. Run tests and inspect outputs after each major change. Use cached meshes/rebuilds where possible to avoid wasting Hugging Face calls. Only regenerate AI meshes when the code change genuinely needs it.



## Implementation Priorities

1. Stabilize geometry and shaping.
   Ensure the side photo drives body length, legs, tail, and silhouette. Avoid weird floaters and thin artifacts. Keep pets restricted to bricks, plates, and tiles for now; no wedges, slopes, inverted slopes, or cheese pieces. Improve voxel cleanup if needed.

2. Build anatomy-aware coloring.
   Add or improve anatomical zones: head, ears, muzzle, chest, belly, front legs, rear legs, paws, tail, torso side. Use front photo for face/chest/muzzle/front leg color context. Use side photo for body/tail/side markings. Keep unique side markings only on the side they appear on. Infer white paws, white lower legs, white chest, white underbelly, and muzzle more accurately. Improve ear color/symmetry only where appropriate.

3. Improve palette and texture strategy.
   Sample actual photo RGB/HEX colors from foreground subject regions. Match sampled colors to closest real LEGO palette colors. Avoid treating shadows as coat markings. Keep coat colors clean and LEGO-like, not noisy/furry. Preserve high-confidence markings like crescent, dots, tail tip, leg patches. Output useful debug artifacts: subject color samples, LEGO palette matches, color map, marking mask, anatomy map.

4. Solve or safely redesign face.
   Do not reintroduce the bad flat/double-face module. Investigate a better face strategy that works with side-body meshes and front photos. The final model should ideally have a front-facing face with two eyes, iris/pupil coloring, nose, and muzzle. If the mesh does not contain a reliable front-facing face surface, implement a safe fallback that does not create ugly fake geometry: either skip face with a clear warning, or add small clean face detail only when placement passes strict quality checks. Add quality checks: eyes must be on head, not body; two eyes visible; sane size; nose centered; no random face pieces.

5. Add validation and regression tests.
   Add tests for anatomy zones, side-specific markings, palette sampling, white chest/paws/belly handling, and face safety checks. Run all relevant tests.

6. Inspect and iterate.
   Generate or rebuild the cat example. Inspect final JSON and debug artifacts. If possible, use the app/browser preview or screenshots to visually evaluate. If the output is bad, diagnose why, patch, and rerun. Continue until the result is meaningfully better or a hard limitation is documented.

## End State

Leave the repo in a working state with tests passing. Restart the local server when done.

Summarize:

- what changed
- what improved visually
- what artifacts Zain should open
- what still needs work
- whether the current approach is good enough for product testing or needs a larger architecture change

Do not stop after making only one small fix. Keep iterating until the overnight session ends or the pipeline is genuinely improved end to end.

## Journal Requirement

Continuously append to `overnight_journal.txt` while working.

Write in plain English. Include:

- what changed
- what was tested
- what looked better or worse
- what you were thinking
- what frustrated you or blocked you
- what you plan to try next

Do not leave the journal empty. Add an entry after every meaningful implementation or test loop. Good luck. Do not let me down.