"""
End-to-end pipeline: photo or mesh -> JSON of LEGO brick placements.

Usage:
    # From a mesh
    python pipeline.py --mesh test_meshes/bunny.obj --resolution 32

    # From a photo (calls Replicate to generate the 3D mesh first)
    python pipeline.py --photo my_dog.jpg --resolution 48 --model triposr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mesh_to_voxels import load_mesh, voxelize
from postprocess import apply_all as postprocess_all
from voxels_to_palette import quantize, load_palette
from voxels_to_bricks import decompose


def run(
    mesh_path: Path,
    resolution: int,
    out_path: Path,
    up_axis: str = "y",
) -> None:
    print(f"[1/4] Loading mesh: {mesh_path}")
    mesh = load_mesh(mesh_path)
    print(f"      vertices={len(mesh.vertices)}  faces={len(mesh.faces)}")

    print(f"[2/4] Voxelizing at resolution={resolution}  up_axis={up_axis}")
    grid = voxelize(mesh, resolution=resolution, up_axis=up_axis)
    occupied = int(grid.occupancy.sum())
    print(f"      shape={grid.shape}  occupied={occupied}  pitch={grid.pitch:.4f}")

    print("[3/4] Quantizing colors to LEGO palette")
    palette = load_palette()
    palette_grid = quantize(grid.colors, grid.occupancy, palette)

    print("[4/4] Decomposing into bricks")
    bricks = decompose(palette_grid)
    print(f"      bricks={len(bricks)}")
    bricks = postprocess_all(bricks, grid.shape,
                             do_tiles=True, do_slopes=True,
                             do_slope_inv=True, do_baseplate=False)
    print(f"      after postprocess: {len(bricks)}")

    payload = {
        "source": str(mesh_path),
        "resolution": resolution,
        "grid_shape": list(grid.shape),
        "pitch": grid.pitch,
        "palette": palette,
        "bricks": [b.to_dict() for b in bricks],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mesh", type=Path, help="Path to a 3D mesh file (.obj/.stl/.glb/...)")
    src.add_argument("--photo", type=Path, help="Path to an input photo (generates the mesh first via Replicate)")
    ap.add_argument("--resolution", type=int, default=32,
                    help="Voxel count along longest axis (default 32)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path (defaults to output/<name>.json)")
    ap.add_argument("--up", choices=["x", "y", "z"], default="y",
                    help="Up axis of the source mesh (default y)")
    ap.add_argument("--model", default="triposr",
                    help="Replicate model preset (triposr|hunyuan3d|trellis) or owner/name[:version]")
    args = ap.parse_args()

    if args.photo:
        # Photo path -> generate mesh via Replicate first
        from photo_to_mesh import photo_to_mesh
        print(f"[0/4] Generating 3D mesh from photo: {args.photo}")
        mesh_path = photo_to_mesh(args.photo, out_dir=Path("test_meshes"), model=args.model)
        name = args.photo.stem
        # Meshes from photo-to-3D models are typically Y-up; let the flag still override
    else:
        mesh_path = args.mesh
        name = mesh_path.stem

    out_path = args.out or Path("output") / f"{name}.json"
    run(mesh_path, args.resolution, out_path, up_axis=args.up)


if __name__ == "__main__":
    main()
