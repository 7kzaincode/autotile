"""
Stage 1-2: Load a 3D mesh and voxelize it to a uniform grid.

Output: a 3D numpy boolean array (occupancy) plus per-voxel surface colors
sampled from the mesh, and the world-space bounds so downstream stages can
reason about brick sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class VoxelGrid:
    occupancy: np.ndarray          # shape (X, Y, Z) bool
    colors: np.ndarray             # shape (X, Y, Z, 3) uint8, 0 where unoccupied
    pitch: float                   # world units per voxel
    origin: np.ndarray             # world-space coord of voxel (0,0,0) corner
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.occupancy.shape)


def load_mesh(path: str | Path, strip_islands: bool = True,
              island_min_frac: float = 0.05, repair: bool = True,
              smooth_iterations: int = 0) -> trimesh.Trimesh:
    """Load a mesh and run cleanup passes.

    `strip_islands` drops small disconnected components.
    `repair` fixes face winding / inverted normals / small holes — AI meshes
    are often subtly broken in ways voxelization mostly hides but which can
    produce gaps in the LEGO output.
    """
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a single mesh: {type(mesh)}")
    if strip_islands and len(mesh.faces) > 0:
        try:
            mesh, meta = _strip_small_islands(mesh, min_frac=island_min_frac)
            if meta.get("dropped", 0):
                print(f"[mesh] dropped {meta['dropped']} small island(s) "
                      f"({meta['components']} components -> {meta['kept']})")
        except Exception as e:
            print(f"[mesh] island filter skipped: {e}")
    if repair:
        try:
            before_faces = len(mesh.faces)
            trimesh.repair.fix_inversion(mesh)
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fill_holes(mesh)
            after_faces = len(mesh.faces)
            if after_faces != before_faces:
                print(f"[mesh] repaired: faces {before_faces} -> {after_faces}")
        except Exception as e:
            print(f"[mesh] repair skipped: {e}")
    if smooth_iterations > 0:
        # Heavy smoothing shrinks the mesh and can disconnect thin features
        # (collapsing a "neck" between body and legs). Use Humphrey-class
        # smoothing instead of plain Laplacian — it preserves volume.
        # Cap iterations to a safe range.
        iters = max(1, min(int(smooth_iterations), 8))
        try:
            print(f"[mesh] volume-preserving smoothing x{iters}")
            # Taubin (lambda/mu) prevents the shrinking that plain Laplacian causes.
            trimesh.smoothing.filter_taubin(mesh, iterations=iters,
                                            lamb=0.5, nu=-0.53)
        except Exception:
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=min(2, iters), lamb=0.3)
            except Exception as e2:
                print(f"[mesh] smoothing skipped: {e2}")
    return mesh


def _strip_small_islands(mesh: trimesh.Trimesh, min_frac: float = 0.05) -> tuple[trimesh.Trimesh, dict]:
    """Drop tiny disconnected face components without materializing meshes.

    ``mesh.split()`` creates one Trimesh object per connected component. AI
    meshes can contain hundreds of thousands of one-face islands, which can
    kill the server. This label-based pass keeps the same behavior but uses a
    sparse graph over face adjacency instead.
    """
    n_faces = int(len(mesh.faces))
    if n_faces <= 0:
        return mesh, {"components": 0, "kept": 0, "dropped": 0}
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adjacency.size == 0:
        return mesh, {"components": 1, "kept": 1, "dropped": 0}

    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except Exception:
        # scipy is part of the project env, but keep a conservative fallback.
        if n_faces > 100_000:
            print("[mesh] scipy unavailable; skipping island split on large mesh")
            return mesh, {"components": 1, "kept": 1, "dropped": 0}
        components = mesh.split(only_watertight=False)
        if len(components) <= 1:
            return mesh, {"components": len(components), "kept": len(components), "dropped": 0}
        largest = max(len(c.faces) for c in components)
        min_faces = max(1, int(largest * min_frac))
        kept = [c for c in components if len(c.faces) >= min_faces]
        out = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
        return out, {
            "components": len(components),
            "kept": len(kept),
            "dropped": len(components) - len(kept),
        }

    rows = np.concatenate([adjacency[:, 0], adjacency[:, 1]])
    cols = np.concatenate([adjacency[:, 1], adjacency[:, 0]])
    data = np.ones(len(rows), dtype=np.uint8)
    graph = coo_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
    n_components, labels = connected_components(
        graph, directed=False, return_labels=True)
    if n_components <= 1:
        return mesh, {"components": int(n_components), "kept": int(n_components), "dropped": 0}

    counts = np.bincount(labels, minlength=n_components)
    largest = int(counts.max())
    min_faces = max(1, int(largest * float(min_frac)))
    keep_labels = np.flatnonzero(counts >= min_faces)
    keep = np.isin(labels, keep_labels)
    if keep.all():
        return mesh, {"components": int(n_components), "kept": int(len(keep_labels)), "dropped": 0}

    mesh = mesh.copy()
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh, {
        "components": int(n_components),
        "kept": int(len(keep_labels)),
        "dropped": int(n_components - len(keep_labels)),
    }


def voxelize(mesh: trimesh.Trimesh, resolution: int = 32, up_axis: str = "y",
             remove_floaters: bool = True, floater_min_frac: float = 0.01,
             *,
             subject_type: str = "",
             pose_hint: str | None = None,
             voxel_supersample: int = 1,
             aa_threshold: float = 0.5,
             auto_fit: bool = True,
             fill_holes: bool = True,
             symmetry_axis: str | None = None) -> VoxelGrid:
    """Voxelize the mesh into roughly `resolution` voxels along its longest axis.

    `up_axis` is the axis of the SOURCE mesh that points up, or "auto".
    The pipeline treats voxel Z as the LEGO stacking axis, so we rotate the
    mesh so its up-axis becomes Z before voxelizing. Auto mode uses a small
    PCA/extents heuristic: pets/buildings favor the long upright axis, while
    vehicles favor the short height axis.

    `voxel_supersample > 1` voxelizes at a finer pitch, then downsamples by
    occupancy coverage. A value of 2 approximates 8 sub-samples per final
    voxel and smooths stair-stepped silhouettes.

    `auto_fit=True` crops empty voxel margins after cleanup so the brick grid
    tightly wraps the subject instead of preserving voxelizer padding.

    `fill_holes=True` fills conservative one-voxel pinholes/pits in the grid.

    `symmetry_axis="x"` mirrors occupancy during voxelization, before photo
    projection, so reconstructed limbs can still receive normal projected
    colors downstream.

    `remove_floaters=True` keeps only voxel connected-components that are at
    least `floater_min_frac` (default 1%) of the largest component's volume.
    This kills the small disconnected blobs TRELLIS / Hunyuan3D sometimes leave.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive (got {resolution!r})")

    requested_up_axis = (up_axis or "y").lower()
    original_extents = np.asarray(mesh.extents, dtype=float)
    chosen_up_axis = (
        choose_auto_up_axis(mesh, subject_type, pose_hint=pose_hint)
        if requested_up_axis == "auto"
        else requested_up_axis
    )
    mesh = _mesh_with_up_axis(mesh, chosen_up_axis)

    oriented_extents = np.asarray(mesh.extents, dtype=float)
    front_axis = _choose_photo_front_axis(oriented_extents, subject_type, pose_hint)
    max_extent = float(oriented_extents.max()) if oriented_extents.size else 0.0
    if max_extent <= 0:
        raise ValueError("mesh has zero extents and cannot be voxelized")

    voxel_supersample = int(max(1, min(int(voxel_supersample or 1), 3)))
    aa_threshold = float(np.clip(aa_threshold, 0.01, 1.0))
    pitch = max_extent / resolution

    matrix, origin, aa_meta = _voxelize_matrix(
        mesh,
        pitch=pitch,
        supersample=voxel_supersample,
        aa_threshold=aa_threshold,
    )

    if remove_floaters:
        matrix = _strip_voxel_floaters(matrix, floater_min_frac)

    hole_meta = {"added": 0}
    if fill_holes:
        matrix, hole_meta = _fill_voxel_pinholes(matrix)

    symmetry_meta = {"axis": None, "added": 0}
    if symmetry_axis:
        matrix, symmetry_meta = enforce_voxel_symmetry(matrix, symmetry_axis)

    fit_meta = None
    if auto_fit:
        matrix, origin, fit_meta = _crop_empty_margins(matrix, origin, pitch)

    colors = _sample_surface_colors(mesh, matrix, pitch, origin)
    metadata = {
        "requested_up_axis": requested_up_axis,
        "up_axis": chosen_up_axis,
        "front_axis": front_axis,
        "subject_type": subject_type,
        "pose_hint": pose_hint or None,
        "source_extents": [float(v) for v in original_extents],
        "oriented_extents": [float(v) for v in oriented_extents],
        "resolution": int(resolution),
        "voxel_scale": [float(pitch), float(pitch), float(pitch)],
        "voxel_supersample": voxel_supersample,
        "aa_threshold": aa_threshold if voxel_supersample > 1 else None,
        "anti_alias": aa_meta,
        "auto_fit": fit_meta,
        "hole_fill": hole_meta,
        "symmetry": symmetry_meta,
    }
    return VoxelGrid(occupancy=matrix, colors=colors, pitch=pitch,
                     origin=origin, metadata=metadata)


def _choose_photo_front_axis(
    oriented_extents: np.ndarray,
    subject_type: str,
    pose_hint: str | None,
) -> str:
    """Pick which voxel axis is camera depth for photo projection.

    For front portraits the existing convention is photo X -> voxel X and
    camera depth -> -Y. For full-body/side pets, the longest horizontal axis is
    body length, so photo X should map to that long axis. We therefore use the
    shorter non-vertical axis as depth.
    """
    subject = (subject_type or "").strip().lower()
    side_like = "side" in (pose_hint or "").lower() or "full_body" in (pose_hint or "").lower()
    pet_like = "pet" in subject or subject in {"cat", "dog", "rabbit", "bunny", "animal"}
    if not (pet_like and side_like):
        return "-y"
    ext = np.asarray(oriented_extents, dtype=float)
    if ext.shape != (3,) or not np.isfinite(ext).all():
        return "-y"
    return "-x" if float(ext[0]) <= float(ext[1]) else "-y"


def choose_auto_up_axis(
    mesh: trimesh.Trimesh,
    subject_type: str = "",
    *,
    pose_hint: str | None = None,
) -> str:
    """Choose the source axis most likely to be vertical for the subject.

    Photo-to-3D Spaces do not agree on coordinate systems. Front/portrait pets
    usually stand along their longest axis. Full-body side-profile pets are the
    exception: their longest axis is body length, so their height is usually the
    middle extent. Vehicles use the shortest axis as height.
    """
    subject = (subject_type or "").strip().lower()
    vehicle_like = subject in {"vehicle", "car", "ship", "boat", "plane", "airplane"}
    side_like = "side" in (pose_hint or "").lower() or "full_body" in (pose_hint or "").lower()
    axes = np.array(["x", "y", "z"])
    extents = np.asarray(mesh.extents, dtype=float)
    if extents.shape != (3,) or not np.isfinite(extents).all() or extents.max() <= 0:
        return "y"

    ext_norm = extents / max(float(extents.max()), 1e-9)
    try:
        verts = np.asarray(mesh.vertices, dtype=float)
        centered = verts - verts.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)
        pc = vecs[:, int(np.argmin(vals) if vehicle_like else np.argmax(vals))]
        pca_scores = np.abs(pc)
        pca_scores = pca_scores / max(float(pca_scores.max()), 1e-9)
    except Exception:
        pca_scores = ext_norm if not vehicle_like else (1.0 - ext_norm)

    if vehicle_like:
        # Height is normally the smallest extent. Blend PCA with raw extents so
        # an axis-aligned car/ship is handled even if the mesh has noisy detail.
        inverse_extent = 1.0 - (extents / max(float(extents.max()), 1e-9))
        scores = 0.75 * inverse_extent + 0.25 * pca_scores
    elif side_like and "pet" in subject:
        sorted_ext = np.sort(ext_norm)
        middle = float(sorted_ext[1])
        middle_extent = 1.0 - np.abs(ext_norm - middle) / max(middle, 1e-9)
        # Trust extents more than PCA here: fur/ears/tails can distort principal
        # components, but the middle bbox axis is a stable full-body height cue.
        scores = 0.85 * middle_extent + 0.15 * pca_scores
    else:
        scores = 0.65 * ext_norm + 0.35 * pca_scores

    if float(scores.max() - scores.min()) < 0.08:
        chosen = "y"
    else:
        chosen = str(axes[int(np.argmax(scores))])
    score_text = ", ".join(f"{a}={s:.2f}" for a, s in zip(axes, scores))
    print(f"[voxelize] auto up-axis selected {chosen!r} "
          f"(subject={subject or 'unknown'}, pose={pose_hint or 'unknown'}, scores: {score_text})")
    return chosen


def _mesh_with_up_axis(mesh: trimesh.Trimesh, up_axis: str) -> trimesh.Trimesh:
    out = mesh.copy()
    up_axis = up_axis.lower()
    if up_axis == "y":
        # Rotate +90° about X so +Y maps to +Z.
        out.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif up_axis == "x":
        # Rotate -90° about Y so +X maps to +Z.
        out.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
    elif up_axis != "z":
        raise ValueError(f"up_axis must be x, y, z, or auto (got {up_axis!r})")
    return out


def _voxelize_matrix(
    mesh: trimesh.Trimesh,
    *,
    pitch: float,
    supersample: int,
    aa_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if supersample <= 1:
        vox = mesh.voxelized(pitch=pitch).fill()
        origin = np.asarray(vox.transform[:3, 3], dtype=float) - pitch / 2.0
        return vox.matrix.astype(bool), origin, {"enabled": False}

    fine_pitch = pitch / supersample
    vox = mesh.voxelized(pitch=fine_pitch).fill()
    fine_matrix = vox.matrix.astype(bool)
    origin = np.asarray(vox.transform[:3, 3], dtype=float) - fine_pitch / 2.0
    matrix = _downsample_occupancy(fine_matrix, supersample, aa_threshold)
    print(f"[voxelize] anti-aliased voxelization {supersample}x: "
          f"{tuple(fine_matrix.shape)} -> {tuple(matrix.shape)} "
          f"(threshold={aa_threshold:.2f})")
    return matrix, origin, {
        "enabled": True,
        "supersample": int(supersample),
        "threshold": float(aa_threshold),
        "fine_shape": [int(v) for v in fine_matrix.shape],
    }


def _downsample_occupancy(matrix: np.ndarray, factor: int, threshold: float) -> np.ndarray:
    if factor <= 1:
        return matrix.astype(bool)
    pad = []
    for size in matrix.shape:
        rem = size % factor
        pad.append((0, 0 if rem == 0 else factor - rem))
    padded = np.pad(matrix.astype(bool), pad, mode="constant", constant_values=False)
    sx, sy, sz = padded.shape
    blocks = padded.reshape(
        sx // factor, factor,
        sy // factor, factor,
        sz // factor, factor,
    )
    coverage = blocks.mean(axis=(1, 3, 5))
    return coverage >= threshold


def _crop_empty_margins(
    matrix: np.ndarray,
    origin: np.ndarray,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    if not matrix.any():
        return matrix, origin, None
    coords = np.argwhere(matrix)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    if np.all(mins == 0) and np.all(maxs == np.array(matrix.shape)):
        return matrix, origin, None
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))
    cropped = matrix[slices]
    new_origin = np.asarray(origin, dtype=float) + mins.astype(float) * pitch
    print(f"[voxelize] auto-fit cropped empty margins: "
          f"{tuple(matrix.shape)} -> {tuple(cropped.shape)}")
    return cropped, new_origin, {
        "old_shape": [int(v) for v in matrix.shape],
        "new_shape": [int(v) for v in cropped.shape],
        "offset": [int(v) for v in mins],
    }


def _fill_voxel_pinholes(occupancy: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fill conservative one-voxel holes without eroding the silhouette."""
    occ = occupancy.astype(bool)
    if occ.size == 0 or not occ.any():
        return occ, {"added": 0}
    p = np.pad(occ, 1, mode="constant", constant_values=False)
    center = p[1:-1, 1:-1, 1:-1]
    neighbors = (
        p[:-2, 1:-1, 1:-1].astype(np.uint8) + p[2:, 1:-1, 1:-1].astype(np.uint8) +
        p[1:-1, :-2, 1:-1].astype(np.uint8) + p[1:-1, 2:, 1:-1].astype(np.uint8) +
        p[1:-1, 1:-1, :-2].astype(np.uint8) + p[1:-1, 1:-1, 2:].astype(np.uint8)
    )
    opposite_pairs = (
        (p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1]).astype(np.uint8) +
        (p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1]).astype(np.uint8) +
        (p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:]).astype(np.uint8)
    )
    fill = (~center) & ((neighbors >= 5) | ((neighbors >= 4) & (opposite_pairs >= 2)))
    out = occ | fill
    added = int(out.sum() - occ.sum())
    if added:
        print(f"[voxelize] hole-fill added {added} voxel(s)")
    return out, {"added": added}


def enforce_voxel_symmetry(occupancy: np.ndarray, axis: str = "x") -> tuple[np.ndarray, dict]:
    axis_l = (axis or "x").lower()
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis_l)
    if axis_index is None:
        raise ValueError(f"symmetry_axis must be x, y, or z (got {axis!r})")
    mirrored = np.flip(occupancy.astype(bool), axis=axis_index)
    out = occupancy.astype(bool) | mirrored
    added = int(out.sum() - occupancy.sum())
    if added:
        print(f"[voxelize] symmetry axis={axis_l} added {added} mirrored voxel(s)")
    return out, {"axis": axis_l, "added": added}


def make_hollow_shell(grid: VoxelGrid, wall_thickness: int = 1,
                      brace_every: int = 6) -> VoxelGrid:
    """Carve out the interior, leaving a `wall_thickness`-voxel shell PLUS
    structural cross-braces every `brace_every` voxels in each axis. The
    braces stop the shell from collapsing when actually assembled.

    `brace_every = 0` disables bracing.
    """
    occ = grid.occupancy
    if not occ.any():
        return grid
    from scipy.ndimage import binary_erosion
    structure = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ], dtype=bool)
    eroded = occ.copy()
    for _ in range(max(0, wall_thickness)):
        eroded = binary_erosion(eroded, structure=structure)
    shell = occ & ~eroded

    if brace_every > 0:
        # Add 1-voxel-thick "rib" slabs through the interior at regular intervals.
        # For each rib position along each axis, keep the FULL original cross-section.
        ribs = np.zeros_like(occ)
        sx, sy, sz = occ.shape
        for x in range(0, sx, brace_every):
            ribs[x, :, :] |= occ[x, :, :]
        for y in range(0, sy, brace_every):
            ribs[:, y, :] |= occ[:, y, :]
        for z in range(0, sz, brace_every):
            ribs[:, :, z] |= occ[:, :, z]
        shell = shell | (ribs & occ)

    print(f"[hollow] kept {int(shell.sum())} of {int(occ.sum())} voxels "
          f"({100*shell.sum()/max(1, occ.sum()):.0f}%, brace_every={brace_every})")
    new_colors = grid.colors.copy()
    new_colors[~shell] = 0
    metadata = dict(grid.metadata)
    metadata["hollow"] = {
        "before": int(occ.sum()),
        "after": int(shell.sum()),
        "wall_thickness": int(wall_thickness),
        "brace_every": int(brace_every),
    }
    return VoxelGrid(occupancy=shell, colors=new_colors,
                     pitch=grid.pitch, origin=grid.origin,
                     metadata=metadata)


def _strip_voxel_floaters(occupancy: np.ndarray, min_frac: float) -> np.ndarray:
    """Find 6-connected components in the occupancy grid; keep only those at
    least `min_frac` * (largest component) in volume."""
    try:
        from scipy.ndimage import label
    except ImportError:
        print("[voxelize] scipy not available, skipping floater removal")
        return occupancy
    labeled, n_components = label(occupancy)
    if n_components <= 1:
        return occupancy
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background
    largest = sizes.max()
    threshold = max(1, int(largest * min_frac))
    keep = sizes >= threshold
    keep[0] = False
    kept_labels = np.where(keep)[0]
    out = np.isin(labeled, kept_labels)
    dropped = n_components - len(kept_labels)
    if dropped > 0:
        kept_vox = int(out.sum())
        total_vox = int(occupancy.sum())
        print(f"[voxelize] removed {dropped} floater(s) "
              f"({total_vox - kept_vox} voxels of {total_vox})")
    return out


def _sample_surface_colors(
    mesh: trimesh.Trimesh,
    occupancy: np.ndarray,
    pitch: float,
    origin: np.ndarray,
) -> np.ndarray:
    """For each occupied voxel, sample the color of the nearest mesh surface point.

    Falls back to a neutral gray if the mesh has no usable color data.
    """
    colors = np.zeros((*occupancy.shape, 3), dtype=np.uint8)
    occ_idx = np.argwhere(occupancy)
    if occ_idx.size == 0:
        return colors

    centers = origin + (occ_idx + 0.5) * pitch

    has_visual_color = (
        getattr(mesh.visual, "kind", None) in {"vertex", "face", "texture"}
    )
    if not has_visual_color:
        colors[occupancy] = (180, 180, 180)
        return colors

    try:
        closest, _, face_ids = trimesh.proximity.closest_point(mesh, centers)
        sampled = _colors_from_faces(mesh, face_ids, closest)
    except Exception:
        sampled = np.full((len(centers), 3), 180, dtype=np.uint8)

    for (ix, iy, iz), rgb in zip(occ_idx, sampled):
        colors[ix, iy, iz] = rgb
    return colors


def _colors_from_faces(mesh: trimesh.Trimesh, face_ids: np.ndarray, points: np.ndarray) -> np.ndarray:
    visual = mesh.visual
    kind = getattr(visual, "kind", None)

    if kind == "face" and visual.face_colors is not None:
        return visual.face_colors[face_ids][:, :3].astype(np.uint8)

    if kind == "vertex" and visual.vertex_colors is not None:
        faces = mesh.faces[face_ids]
        vcols = visual.vertex_colors[:, :3].astype(np.float32)
        return vcols[faces].mean(axis=1).astype(np.uint8)

    if kind == "texture":
        sampled = _sample_texture(mesh, face_ids)
        if sampled is not None:
            return sampled

    return np.full((len(points), 3), 180, dtype=np.uint8)


def _sample_texture(mesh: trimesh.Trimesh, face_ids: np.ndarray) -> np.ndarray | None:
    """Sample baseColor texture at the UV of each hit face. Tries the various
    places trimesh / gltf stores textures (PBR `baseColorTexture`, legacy
    `image`, simple material color)."""
    try:
        uv = mesh.visual.uv
        if uv is None:
            return None
        faces = mesh.faces[face_ids]
        face_uvs = uv[faces].mean(axis=1)  # one UV per hit face (centroid)
    except Exception:
        return None

    image = None
    mat = getattr(mesh.visual, "material", None)
    if mat is not None:
        # PBRMaterial (glTF) keeps baseColorTexture; SimpleMaterial keeps image
        for attr in ("baseColorTexture", "image"):
            tex = getattr(mat, attr, None)
            if tex is not None:
                image = np.asarray(tex)
                break

    if image is None or image.size == 0:
        # Fall back to baseColorFactor (a flat color)
        if mat is not None:
            factor = getattr(mat, "baseColorFactor", None)
            if factor is not None and len(factor) >= 3:
                return np.tile(np.array(factor[:3], dtype=np.uint8), (len(face_ids), 1))
        return None

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    h, w = image.shape[:2]
    u = np.clip((face_uvs[:, 0] * w).astype(int), 0, w - 1)
    v = np.clip(((1.0 - face_uvs[:, 1]) * h).astype(int), 0, h - 1)
    return image[v, u, :3].astype(np.uint8)
