"""
Regression tests covering the offline pipeline (no HTTP / no Replicate / no HF).

Run with:  python -m pytest tests/  (or just `python tests/test_pipeline.py`)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brick_catalog import BRICK_CATALOG, CATALOG
from build_stats import build_stats
from face_map import apply_eye_face_map, apply_pet_face_map
from features import add_anatomical_features, detect_features
from ldraw_export import to_ldraw
from mesh_to_voxels import (
    _fill_voxel_pinholes,
    _strip_small_islands,
    choose_auto_up_axis,
    enforce_voxel_symmetry,
    load_mesh,
    voxelize,
)
from obj_export import to_obj
from parts_list import parts_list, parts_list_bricklink_xml, parts_list_csv, summary
from pet_color_map import project_pet_color_map
from pose_analysis import analyze_photo_pose
from postprocess import apply_all as postprocess_all, cheese_slope_ear_tips
from semantic_projection import _build_gpt_label_map, _project_label_map_onto_voxels
from subject_preset import resolve as resolve_subject_preset
from texture_projection import project_photo
from voxels_to_bricks import decompose
from voxels_to_palette import load_palette, quantize


def _first_existing(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted((ROOT / "test_meshes").glob(pattern))
        if matches:
            return matches[0]
    return None


BUNNY = _first_existing(["bunny.obj", "*bunny*.obj"])


def test_auto_up_axis_picks_tall_axis_for_pet():
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 4.0))
    assert choose_auto_up_axis(mesh, subject_type="pet") == "z"
    grid = voxelize(
        mesh,
        resolution=16,
        up_axis="auto",
        subject_type="pet",
        fill_holes=False,
    )
    assert grid.metadata["up_axis"] == "z"
    assert grid.shape[2] >= grid.shape[0]
    assert grid.shape[2] >= grid.shape[1]


def test_auto_up_axis_uses_middle_extent_for_side_pet():
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 5.0))
    assert choose_auto_up_axis(mesh, subject_type="pet", pose_hint="side_profile") == "y"
    grid = voxelize(
        mesh,
        resolution=16,
        up_axis="auto",
        subject_type="pet",
        pose_hint="side_profile",
        fill_holes=False,
    )
    assert grid.metadata["up_axis"] == "y"
    assert grid.metadata["pose_hint"] == "side_profile"


def test_auto_up_axis_uses_short_height_for_vehicle():
    mesh = trimesh.creation.box(extents=(8.0, 2.0, 1.0))
    assert choose_auto_up_axis(mesh, subject_type="vehicle") == "z"
    grid = voxelize(
        mesh,
        resolution=16,
        up_axis="auto",
        subject_type="vehicle",
        fill_holes=False,
    )
    assert grid.metadata["up_axis"] == "z"
    assert grid.shape[2] < grid.shape[0]


def test_voxel_supersampling_records_antialias_metadata():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    grid = voxelize(
        mesh,
        resolution=12,
        up_axis="z",
        voxel_supersample=2,
        fill_holes=False,
    )
    assert grid.metadata["anti_alias"]["enabled"] is True
    assert grid.metadata["voxel_supersample"] == 2
    assert grid.occupancy.any()


def test_voxel_hole_fill_closes_one_voxel_pinhole():
    occ = np.ones((3, 3, 3), dtype=bool)
    occ[1, 1, 1] = False
    out, meta = _fill_voxel_pinholes(occ)
    assert out[1, 1, 1]
    assert meta["added"] == 1


def test_voxel_symmetry_mirrors_occupancy_during_voxelize():
    occ = np.zeros((4, 3, 2), dtype=bool)
    occ[0, 1, 1] = True
    out, meta = enforce_voxel_symmetry(occ, axis="x")
    assert out[0, 1, 1]
    assert out[3, 1, 1]
    assert meta == {"axis": "x", "added": 1}


def test_strip_small_islands_without_split_materialization():
    big = trimesh.creation.box(extents=(1, 1, 1))
    tiny = trimesh.Trimesh(
        vertices=np.array([[4, 0, 0], [4.1, 0, 0], [4, 0.1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        process=False,
    )
    mesh = trimesh.util.concatenate([big, tiny])
    out, meta = _strip_small_islands(mesh, min_frac=0.5)
    assert meta["components"] == 2
    assert meta["kept"] == 1
    assert meta["dropped"] == 1
    assert len(out.faces) < len(mesh.faces)


def _build_payload(mesh_path, resolution=32, photo_path=None, postprocess=True):
    if mesh_path is None or not Path(mesh_path).exists():
        print("[skip] no cached bunny mesh")
        return None
    mesh = load_mesh(mesh_path)
    grid = voxelize(mesh, resolution=resolution, up_axis="y")
    if photo_path is not None:
        project_photo(grid, photo_path)
    palette = load_palette()
    pgrid = quantize(grid.colors, grid.occupancy, palette)
    bricks = decompose(pgrid)
    if postprocess:
        bricks = postprocess_all(bricks, grid.shape)
    return {
        "source": mesh_path.name,
        "resolution": resolution,
        "grid_shape": list(grid.shape),
        "pitch": float(grid.pitch),
        "palette": palette,
        "bricks": [b.to_dict() for b in bricks],
    }


def test_palette_has_ldraw_and_bricklink_ids():
    pal = load_palette()
    assert len(pal) >= 20
    for entry in pal:
        assert "ldraw" in entry, f"palette[{entry['id']}] missing ldraw id"
        assert "bricklink" in entry, f"palette[{entry['id']}] missing bricklink id"
        assert isinstance(entry["ldraw"], int)


def test_brick_catalog_consistency():
    for bt, cat in BRICK_CATALOG.items():
        assert "ldraw" in cat and "bricklink" in cat
        a, b = sorted((cat["studs_x"], cat["studs_y"]))
        assert f"{a}x{b}" == bt, f"catalog key {bt} mismatches dims {cat}"


def test_catalog_has_multiple_kinds():
    kinds = {k for (k, _) in CATALOG.keys()}
    # Core kinds must all be present (specialty pet/organic kinds may also exist)
    required = {"brick", "plate", "tile", "slope", "slope_inv"}
    assert required <= kinds, f"missing required kinds: {required - kinds}"
    # Specialty kinds for pets — added in Phase A of the organic pipeline rebuild
    specialty = {"round_brick", "round_plate", "round_tile", "cone",
                 "cheese_slope", "dome"}
    assert specialty <= kinds, f"missing specialty kinds: {specialty - kinds}"
    # Every brick type should also have at least a tile entry for the most common sizes
    for bt in ["1x1", "1x2", "2x2", "2x4"]:
        assert ("tile", bt) in CATALOG, f"missing tile {bt}"
    # Specialty 1x1 round tile / plate / cone available
    for k in ("round_tile", "round_plate", "cone", "cheese_slope"):
        assert (k, "1x1") in CATALOG, f"missing 1x1 {k}"


def test_postprocess_adds_tiles_and_slopes():
    payload = _build_payload(BUNNY, resolution=32, postprocess=True)
    if payload is None:
        return
    kinds = {b["kind"] for b in payload["bricks"]}
    assert "brick" in kinds
    assert "tile" in kinds  # bunny should get some tile caps


def test_postprocess_baseplate_prepends_plate():
    raw_payload = _build_payload(BUNNY, resolution=32, postprocess=False)
    if raw_payload is None:
        return
    # Manually run post with baseplate=True
    from postprocess import apply_all
    from voxels_to_bricks import Brick
    bricks = [
        Brick(
            x=b["x"], y=b["y"], z=b["z"],
            size_x=b["size_x"], size_y=b["size_y"],
            color_id=b["color"], kind=b.get("kind", "brick"),
        )
        for b in raw_payload["bricks"]
    ]
    with_base = apply_all(bricks, tuple(raw_payload["grid_shape"]),
                          do_baseplate=True)
    # First piece should be the baseplate plate
    assert with_base[0].kind == "plate"
    assert with_base[0].z == -1
    assert with_base[0].size_x >= 5 and with_base[0].size_y >= 5
    obj, _ = to_obj({"palette": load_palette(), "bricks": [b.to_dict() for b in with_base]})
    y_values = [float(line.split()[2]) for line in obj.splitlines()
                if line.startswith("v ")]
    assert min(y_values) < 0.0


def test_bunny_pipeline():
    payload = _build_payload(BUNNY, resolution=32, postprocess=False)
    if payload is None:
        return
    assert len(payload["grid_shape"]) == 3, payload["grid_shape"]
    assert all(d > 0 for d in payload["grid_shape"]), payload["grid_shape"]
    assert max(payload["grid_shape"]) >= 24, payload["grid_shape"]
    assert 100 < len(payload["bricks"]) < 2500, len(payload["bricks"])

    # Every brick has a valid brick_type and a known color id
    pal_ids = {p["id"] for p in payload["palette"]}
    for b in payload["bricks"]:
        assert b["brick_type"] in BRICK_CATALOG
        assert b["color"] in pal_ids


def test_parts_list_matches_bricks():
    payload = _build_payload(BUNNY, resolution=32)
    if payload is None:
        return
    rows = parts_list(payload)
    assert sum(r["quantity"] for r in rows) == len(payload["bricks"])
    s = summary(payload)
    assert s["total_bricks"] == len(payload["bricks"])


def test_parts_list_csv_format():
    payload = _build_payload(BUNNY, resolution=32)
    if payload is None:
        return
    csv = parts_list_csv(payload)
    lines = csv.strip().splitlines()
    header = lines[0].split(",")
    assert "quantity" in header and "bricklink_part" in header
    n_unique = len({(b.get("kind", "brick"), b["brick_type"], b["color"])
                    for b in payload["bricks"]})
    assert len(lines) - 1 == n_unique


def test_bricklink_xml_well_formed():
    payload = _build_payload(BUNNY, resolution=32)
    if payload is None:
        return
    xml = parts_list_bricklink_xml(payload)
    assert xml.startswith("<INVENTORY>") and xml.endswith("</INVENTORY>")
    assert "<ITEMTYPE>P</ITEMTYPE>" in xml
    n_items = xml.count("<ITEM>")
    n_unique = len({(b.get("kind", "brick"), b["brick_type"], b["color"])
                    for b in payload["bricks"]})
    assert n_items == n_unique


def test_ldraw_export_line_count():
    payload = _build_payload(BUNNY, resolution=32)
    if payload is None:
        return
    text = to_ldraw(payload)
    n_placements = sum(1 for line in text.splitlines() if line.startswith("1 "))
    assert n_placements == len(payload["bricks"])
    # First column after "1 " is the color, third value is X, fourth is Y, etc.
    for line in text.splitlines():
        if not line.startswith("1 "): continue
        parts = line.split()
        assert len(parts) == 15, f"bad LDraw line: {line!r}"
        assert parts[-1].endswith(".dat")


def test_texture_projection_increases_color_variety():
    """Find a pepsi photo + matching mesh and verify texture brings more colors in."""
    photos_dir = ROOT / "test_photos"
    meshes_dir = ROOT / "test_meshes"
    candidates = [
        (meshes_dir / "1778539253_pepsi.obj", photos_dir / "1778539253_pepsi.png"),
    ]
    pair = next(((m, p) for m, p in candidates if m.exists() and p.exists()), None)
    if pair is None:
        print("[skip] no pepsi mesh+photo pair cached")
        return
    mesh_path, photo_path = pair
    plain  = _build_payload(mesh_path, resolution=48)
    tex    = _build_payload(mesh_path, resolution=48, photo_path=photo_path)
    n_plain = len({b["color"] for b in plain["bricks"]})
    n_tex   = len({b["color"] for b in tex["bricks"]})
    assert n_tex > n_plain, f"texture didn't add colors ({n_plain} -> {n_tex})"


def test_obj_export_well_formed():
    payload = _build_payload(BUNNY, resolution=32)
    if payload is None:
        return
    obj_text, mtl_text = to_obj(payload)
    n_v = sum(1 for line in obj_text.splitlines() if line.startswith("v "))
    n_f = sum(1 for line in obj_text.splitlines() if line.startswith("f "))
    assert n_v == 8 * len(payload["bricks"]), f"{n_v} verts != 8 * {len(payload['bricks'])} bricks"
    assert n_f == 12 * len(payload["bricks"]), f"{n_f} faces != 12 * {len(payload['bricks'])} bricks"
    # MTL has at least one newmtl entry per distinct color
    n_mat = sum(1 for line in mtl_text.splitlines() if line.startswith("newmtl "))
    n_colors = len({b["color"] for b in payload["bricks"]})
    assert n_mat == n_colors


def test_build_stats():
    payload = _build_payload(BUNNY, resolution=32)
    if payload is None:
        return
    s = build_stats(payload)
    assert s["total_bricks"] == len(payload["bricks"])
    assert s["estimated_cost_usd"] > 0
    assert 0.0 <= s["support_score"] <= 1.0
    assert s["floating_bricks"] >= 0
    assert s["estimated_build_time_minutes"] > 0


def test_empty_payload_doesnt_crash_exports():
    """Empty payloads from the editor (all bricks deleted) shouldn't crash."""
    payload = {
        "source": "empty",
        "resolution": 0,
        "grid_shape": [0, 0, 0],
        "pitch": 0.0,
        "palette": load_palette(),
        "bricks": [],
    }
    assert parts_list(payload) == []
    csv = parts_list_csv(payload)
    assert csv.startswith("quantity,")
    xml = parts_list_bricklink_xml(payload)
    assert "<INVENTORY>" in xml and "</INVENTORY>" in xml
    ldraw = to_ldraw(payload)
    assert ldraw.startswith("0 ")
    obj, mtl = to_obj(payload)
    assert "mtllib" in obj
    s = build_stats(payload)
    assert s["total_bricks"] == 0


def test_brick_decomp_alternates_orientation_per_layer():
    """Even layers prefer x-long, odd layers prefer y-long. Verify by counts."""
    pgrid = np.ones((8, 8, 2), dtype=np.int32)
    bricks = [b.to_dict() for b in decompose(pgrid)]
    by_layer_orient = {}
    for b in bricks:
        z = b["z"]
        if b["size_x"] == b["size_y"]:
            continue
        long_x = b["size_x"] > b["size_y"]
        by_layer_orient.setdefault(z, Counter())[long_x] += 1
    # Across the model, at least one layer should be majority x-long and at
    # least one majority y-long.
    has_x = has_y = False
    for z, c in by_layer_orient.items():
        if c[True] > c[False]: has_x = True
        if c[False] > c[True]: has_y = True
    assert has_x and has_y, "decomp didn't produce mixed orientations across layers"


def test_semantic_label_projection_shared_helper():
    class Grid:
        pass

    grid = Grid()
    grid.occupancy = np.zeros((4, 3, 4), dtype=bool)
    grid.occupancy[:, 0, :] = True
    grid.occupancy[:, 2, :] = True
    grid.colors = np.zeros((4, 3, 4, 3), dtype=np.uint8)
    label_map = np.array([
        [-1, 1, 1, -1],
        [-1, 1, 1, -1],
        [-1, 2, 2, -1],
        [-1, 2, 2, -1],
    ], dtype=np.int32)

    meta = _project_label_map_onto_voxels(
        grid, label_map,
        {1: (10, 20, 30), 2: (200, 210, 220)},
    )

    assert meta["projected_voxels"] == int(grid.occupancy.sum())
    assert grid.colors[0, 0, 3].tolist() == [10, 20, 30]
    assert grid.colors[0, 2, 3].tolist() == [10, 20, 30]
    assert grid.colors[0, 0, 0].tolist() == [200, 210, 220]


def test_pose_analysis_detects_wide_pet_as_side_profile(tmp_path):
    p = tmp_path / "wide_pet.png"
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for y in range(35, 70):
        for x in range(5, 95):
            img.putpixel((x, y), (200, 150, 90, 255))
    img.save(p)

    meta = analyze_photo_pose(p, subject="pet")
    assert meta["is_side_profile"] is True
    assert meta["safe_mirror"] is False
    assert meta["safe_face_accents"] is True


def test_pet_preset_keeps_asymmetric_markings():
    preset = resolve_subject_preset("pet")
    assert preset["mirror"] is False
    assert preset["mirror_geometry"] is False
    assert preset["voxel_symmetry"] is False
    assert preset["slopes"] is False
    assert preset["slope_inv"] is False


def test_semantic_paint_mode_none_is_palette_only():
    gpt_data = {
        "regions": [
            {"name": "body", "bbox_normalized": [0.0, 0.0, 1.0, 1.0], "color_name": "Tan"},
            {"name": "tongue", "bbox_normalized": [0.45, 0.35, 0.55, 0.45], "color_name": "Pink"},
        ]
    }
    label_map, regions, skipped = _build_gpt_label_map(gpt_data, 100, 100, paint_mode="none")
    assert regions == {}
    assert np.all(label_map == -1)
    assert any("palette-only" in s for s in skipped)


def test_semantic_small_mode_keeps_detail_not_body():
    gpt_data = {
        "regions": [
            {"name": "body", "bbox_normalized": [0.0, 0.0, 1.0, 1.0], "color_name": "Tan"},
            {"name": "tongue", "bbox_normalized": [0.45, 0.35, 0.55, 0.45], "color_name": "Pink"},
        ]
    }
    label_map, regions, skipped = _build_gpt_label_map(gpt_data, 100, 100, paint_mode="small")
    assert len(regions) == 1
    assert next(iter(regions.values()))["name"] == "tongue"
    assert (label_map >= 0).any()
    assert any("body:broad-surface" in s for s in skipped)


def test_feature_detection_uses_alpha_silhouette(tmp_path):
    p = tmp_path / "face.png"
    img = Image.new("RGBA", (120, 140), (0, 0, 0, 0))
    for y in range(20, 125):
        for x in range(30, 90):
            img.putpixel((x, y), (210, 170, 120, 255))
    for cx, cy in [(48, 55), (72, 55), (60, 74)]:
        for y in range(cy - 3, cy + 4):
            for x in range(cx - 3, cx + 4):
                img.putpixel((x, y), (20, 15, 10, 255))
    img.save(p)

    feat = detect_features(p)
    assert feat["bbox"] == (30, 20, 90, 125)
    assert len(feat["eyes"]) == 2
    assert feat["nose"] is not None


def test_face_map_builds_multi_cell_eyes():
    bricks = [
        {
            "x": x, "y": 0, "z": z,
            "size_x": 1, "size_y": 1, "brick_type": "1x1",
            "kind": "brick", "rotation": 0, "color": 28,
            "slope_dir": None,
        }
        for x in range(20) for z in range(20)
    ]
    payload = {"grid_shape": [20, 4, 20], "bricks": bricks}
    feat = {
        "eyes": [(35, 40), (65, 40)],
        "bbox": (0, 0, 100, 100),
        "nose": None,
        "mouth": None,
    }

    out, meta = apply_eye_face_map(payload, "unused.png", feat)
    added = out["bricks"][len(bricks):]
    assert meta["added"] >= 12
    assert all(b.get("protected") for b in added)
    assert {b["color"] for b in added} >= {1, 4}


def test_pet_face_map_uses_one_plane_for_all_parts(tmp_path):
    p = tmp_path / "face.png"
    Image.new("RGBA", (120, 120), (210, 170, 120, 255)).save(p)
    bricks = [
        {
            "x": x, "y": (0 if x < 10 else 2), "z": z,
            "size_x": 1, "size_y": 1, "brick_type": "1x1",
            "kind": "brick", "rotation": 0, "color": 28,
            "slope_dir": None,
        }
        for x in range(20) for z in range(20)
    ]
    payload = {"grid_shape": [20, 5, 20], "bricks": bricks}
    feat = {
        "eyes": [(35, 40), (65, 40)],
        "nose": (50, 56),
        "mouth": ((43, 66), (57, 66)),
        "bbox": (0, 0, 100, 100),
    }

    out, meta = apply_pet_face_map(payload, p, feat)
    added = out["bricks"][len(bricks):]
    ys = {b["y"] for b in added}
    parts = {b.get("face_map") for b in added}
    assert len(ys) == 1
    assert {"eye", "nose", "mouth"} <= parts
    assert meta["face_plane"]["y"] in ys


def test_pet_face_map_uses_photo_length_axis_for_side_pet(tmp_path):
    p = tmp_path / "side_face.png"
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for y in range(10, 90):
        for x in range(10, 90):
            img.putpixel((x, y), (220, 220, 220, 255))
    img.save(p)

    bricks = [
        {
            "x": 0, "y": y, "z": z, "size_x": 1, "size_y": 1,
            "brick_type": "1x1", "kind": "brick", "rotation": 0,
            "color": 1, "slope_dir": None,
        }
        for y in range(8) for z in range(6)
    ]
    payload = {
        "grid_shape": [4, 8, 6],
        "voxel_metadata": {"front_axis": "-x"},
        "bricks": bricks,
    }
    feat = {
        "eyes": [(30, 35), (70, 35)],
        "nose": (50, 55),
        "mouth": ((42, 66), (58, 66)),
        "bbox": (0, 0, 100, 100),
    }

    out, meta = apply_pet_face_map(payload, p, feat)
    added = out["bricks"][len(bricks):]
    assert added
    assert {b["mount"] for b in added} == {"-x"}
    assert len({b["x"] for b in added}) == 1
    assert len({b["y"] for b in added if b.get("face_map") == "eye"}) > 1
    assert meta["face_plane"]["axis"] == "-x"


def test_pet_face_map_clamps_full_body_side_face_to_head_region(tmp_path):
    p = tmp_path / "cat_face.png"
    img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    for y in range(80, 956):
        for x in range(379, 986):
            img.putpixel((x, y), (230, 230, 230, 255))
    img.save(p)

    bricks = [
        {
            "x": 1, "y": y, "z": z, "size_x": 1, "size_y": 1,
            "brick_type": "1x1", "kind": "brick", "rotation": 0,
            "color": 1, "slope_dir": None,
        }
        for y in range(65) for z in range(40)
    ]
    payload = {
        "grid_shape": [19, 65, 40],
        "voxel_metadata": {"front_axis": "-x", "pose_hint": "side_profile"},
        "bricks": bricks,
    }
    feat = {
        "eyes": [(520, 250), (610, 250)],
        "nose": (565, 320),
        "mouth": ((548, 360), (582, 360)),
        "bbox": (379, 79, 986, 956),
        "head_bbox": (470, 130, 670, 430),
    }

    out, meta = apply_pet_face_map(payload, p, feat)
    added = out["bricks"][len(bricks):]
    assert 0 < len(added) <= 42
    assert {b["mount"] for b in added} == {"-x"}
    assert max(b["y"] for b in added) - min(b["y"] for b in added) <= 18
    assert meta["target"]["side_like"] is True


def test_pet_face_map_retries_smaller_overlay_instead_of_dropping_eyes(tmp_path):
    p = tmp_path / "large_eye_face.png"
    Image.new("RGBA", (200, 200), (220, 185, 135, 255)).save(p)
    bricks = [
        {
            "x": x, "y": 0, "z": z,
            "size_x": 1, "size_y": 1, "brick_type": "1x1",
            "kind": "brick", "rotation": 0, "color": 28,
            "slope_dir": None,
        }
        for x in range(48) for z in range(42)
    ]
    payload = {"grid_shape": [48, 4, 42], "voxel_metadata": {"front_axis": "-y"}, "bricks": bricks}
    feat = {
        "eyes": [(20, 55), (180, 55)],
        "nose": (100, 105),
        "mouth": ((80, 135), (120, 135)),
        "bbox": (0, 0, 200, 200),
        "head_bbox": (0, 10, 200, 160),
    }

    out, meta = apply_pet_face_map(payload, p, feat)
    added = out["bricks"][len(bricks):]
    assert 0 < len(added) <= 42
    assert any(b.get("face_map") == "eye" for b in added)
    assert meta["clamped"] is True


def test_pet_face_map_anchors_side_profile_face_to_body_head(tmp_path):
    face = tmp_path / "front_face.png"
    face_img = Image.new("RGBA", (200, 200), (220, 185, 135, 255))
    for cx, cy in [(70, 70), (130, 70)]:
        for y in range(cy - 7, cy + 8):
            for x in range(cx - 7, cx + 8):
                if (x - cx) ** 2 + (y - cy) ** 2 <= 7 ** 2:
                    face_img.putpixel((x, y), (150, 88, 38, 255))
        for y in range(cy - 2, cy + 3):
            for x in range(cx - 2, cx + 3):
                face_img.putpixel((x, y), (12, 10, 8, 255))
    face_img.save(face)
    side = tmp_path / "right_side.png"
    img = Image.new("RGBA", (160, 110), (0, 0, 0, 0))
    # Body and tail.
    for y in range(45, 98):
        for x in range(20, 118):
            img.putpixel((x, y), (220, 185, 135, 255))
    for y in range(62, 82):
        for x in range(2, 32):
            img.putpixel((x, y), (220, 185, 135, 255))
    # Head/ears on the right end.
    for y in range(18, 70):
        for x in range(108, 153):
            img.putpixel((x, y), (220, 185, 135, 255))
    for y in range(4, 28):
        for x in range(118, 134):
            img.putpixel((x, y), (220, 185, 135, 255))
    img.save(side)

    bricks = [
        {
            "x": x, "y": y, "z": z,
            "size_x": 1, "size_y": 1, "brick_type": "1x1",
            "kind": "brick", "rotation": 0, "color": 28,
            "slope_dir": None,
        }
        for x in range(44, 65) for y in range(3, 20) for z in range(18, 44)
    ]
    payload = {
        "grid_shape": [65, 23, 45],
        "voxel_metadata": {"front_axis": "-y", "pose_hint": "side_profile"},
        "bricks": bricks,
    }
    feat = {
        "eyes": [(70, 70), (130, 70)],
        "nose": (100, 104),
        "mouth": ((88, 124), (112, 124)),
        "bbox": (0, 0, 200, 200),
        "head_bbox": (45, 35, 155, 145),
    }

    out, meta = apply_pet_face_map(
        payload,
        face,
        feat,
        body_photo_path=side,
        body_gpt_data={
            "regions": [
                {"name": "tail", "bbox_normalized": [0.0, 0.55, 0.2, 0.75]},
                {"name": "eye_socket", "bbox_normalized": [0.78, 0.30, 0.88, 0.42]},
                {"name": "muzzle", "bbox_normalized": [0.84, 0.38, 0.95, 0.52]},
            ]
        },
        body_view="right",
    )
    added = out["bricks"][len(bricks):]
    eye_tiles = [b for b in added if b.get("face_map") == "eye"]
    assert eye_tiles
    assert meta["side_anchor"] is True
    assert meta["target"]["front_face_anchor"] is True
    assert meta["target"]["head_side"] == "right"
    assert len(meta["eye_centers"]) == 2
    assert {b["mount"] for b in eye_tiles} == {"+x"}
    assert min(b["x"] for b in eye_tiles) >= 58
    assert max(b["y"] for b in eye_tiles) - min(b["y"] for b in eye_tiles) >= 3
    assert 4 in {b["color"] for b in eye_tiles}
    assert {b["color"] for b in eye_tiles} & {25, 26, 28, 29, 30}
    assert not any(b.get("face_map") == "mouth" for b in added)


def test_side_body_pet_features_skip_unsafe_face_overlay(tmp_path):
    p = tmp_path / "front_face.png"
    img = Image.new("RGBA", (120, 140), (0, 0, 0, 0))
    for y in range(20, 125):
        for x in range(30, 90):
            img.putpixel((x, y), (210, 170, 120, 255))
    for cx, cy in [(48, 55), (72, 55), (60, 74)]:
        for y in range(cy - 3, cy + 4):
            for x in range(cx - 3, cx + 4):
                img.putpixel((x, y), (20, 15, 10, 255))
    img.save(p)

    bricks = [
        {
            "x": x, "y": 0, "z": z,
            "size_x": 1, "size_y": 1, "brick_type": "1x1",
            "kind": "brick", "rotation": 0, "color": 28,
            "slope_dir": None,
        }
        for x in range(20) for z in range(20)
    ]
    payload = {"grid_shape": [20, 4, 20], "voxel_metadata": {"front_axis": "-x"}, "bricks": bricks}

    out = add_anatomical_features(
        payload,
        p,
        use_gpt=False,
        body_photo_path=p,
        body_view="left",
    )
    assert len(out["bricks"]) == len(bricks)
    assert not any(b.get("face_map") for b in out["bricks"])


def test_cheese_slope_tips_only_apply_near_model_top():
    from voxels_to_bricks import Brick

    bricks = [
        Brick(1, 1, 0, 1, 1, 28),
        Brick(1, 1, 1, 1, 1, 28),  # lower isolated tip: should stay brick
        Brick(4, 1, 0, 1, 1, 28),
        Brick(4, 1, 9, 1, 1, 28),
        Brick(4, 1, 10, 1, 1, 28),  # true top-zone tip
    ]

    out = cheese_slope_ear_tips(bricks)
    by_pos = {(b.x, b.y, b.z): b for b in out}
    assert by_pos[(1, 1, 1)].kind == "brick"
    assert by_pos[(4, 1, 10)].kind == "cheese_slope"


def test_pet_tile_only_postprocess_does_not_add_wedges():
    from voxels_to_bricks import Brick

    bricks = [
        Brick(1, 1, 0, 1, 1, 28),
        Brick(1, 1, 1, 1, 1, 28),
        Brick(4, 1, 0, 1, 1, 28),
        Brick(4, 1, 9, 1, 1, 28),
        Brick(4, 1, 10, 1, 1, 28),
    ]

    out = postprocess_all(
        bricks, (8, 4, 12),
        do_tiles=True,
        do_slopes=False,
        do_slope_inv=False,
        do_cheese_tips=False,
    )
    kinds = {b.kind for b in out}
    assert "tile" in kinds
    assert "cheese_slope" not in kinds
    assert "slope" not in kinds
    assert "slope_inv" not in kinds


def test_pet_color_map_smooths_to_small_palette(tmp_path):
    from mesh_to_voxels import VoxelGrid

    p = tmp_path / "pet.png"
    img = Image.new("RGBA", (80, 100), (0, 0, 0, 0))
    # Tan body with cream belly and tiny dark noise that should be smoothed.
    for y in range(10, 95):
        for x in range(15, 65):
            img.putpixel((x, y), (205, 170, 120, 255))
    for y in range(45, 90):
        for x in range(32, 48):
            img.putpixel((x, y), (230, 210, 165, 255))
    for x, y in [(25, 30), (55, 38), (20, 80)]:
        img.putpixel((x, y), (40, 30, 20, 255))
    img.save(p)

    occ = np.ones((12, 3, 16), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))
    palette = load_palette()

    used, meta = project_pet_color_map(grid, p, palette, out_dir=tmp_path)
    assert meta["applied"] is True
    assert len(used) <= 4
    assert meta["debug_path"]
    assert Path(meta["debug_path"]).exists()
    assert grid.colors[grid.occupancy].sum() > 0


def test_pet_color_map_preserves_one_sided_dark_marking(tmp_path):
    from mesh_to_voxels import VoxelGrid

    p = tmp_path / "white_cat_patch.png"
    img = Image.new("RGBA", (80, 100), (0, 0, 0, 0))
    for y in range(5, 95):
        for x in range(10, 70):
            img.putpixel((x, y), (238, 238, 230, 255))
    # One visible leg patch on the subject's right side. It should not be
    # mirrored away or smoothed into the white base coat.
    for y in range(45, 92):
        for x in range(50, 62):
            shade = 65 if (y // 5) % 2 else 105
            img.putpixel((x, y), (shade, shade, shade - 8, 255))
    img.save(p)

    occ = np.ones((12, 2, 16), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))
    palette = load_palette()

    used, meta = project_pet_color_map(grid, p, palette, out_dir=tmp_path)
    assert meta["applied"] is True
    assert meta["markings"]["components"] >= 1
    assert used & {3, 4, 25, 26, 27, 29, 30}

    brightness = grid.colors.astype(np.float32).mean(axis=3)
    right_vals = brightness[7:11][grid.occupancy[7:11]]
    left_vals = brightness[1:5][grid.occupancy[1:5]]
    assert float(right_vals.mean()) < float(left_vals.mean()) - 25.0


def test_pet_color_map_preserves_high_res_crescent_and_dots(tmp_path):
    from mesh_to_voxels import VoxelGrid
    from PIL import ImageDraw

    p = tmp_path / "crescent_cat.png"
    img = Image.new("RGBA", (320, 210), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((20, 35, 300, 190), fill=(214, 174, 125, 255))
    d.polygon([(265, 70), (318, 95), (265, 125)], fill=(214, 174, 125, 255))
    d.line((15, 150, 5, 198), fill=(180, 120, 70, 255), width=20)
    # A crescent plus three separate dots: this used to collapse into one blob
    # because markings were detected after blur/downsample.
    d.ellipse((135, 55, 265, 172), fill=(58, 57, 53, 255))
    d.ellipse((170, 68, 275, 166), fill=(214, 174, 125, 255))
    for cx, cy in [(65, 72), (56, 116), (92, 95)]:
        d.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), fill=(50, 50, 46, 255))
    img.save(p)

    occ = np.ones((4, 65, 46), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))

    used, meta = project_pet_color_map(
        grid, p, load_palette(), out_dir=tmp_path, front_axis="-x",
    )
    assert meta["markings"]["source"] == "high-res-mask"
    assert meta["markings"]["components"] >= 4
    assert meta["markings"]["high_res"]["components"] >= 4
    assert meta["marking_debug_path"]
    assert Path(meta["marking_debug_path"]).exists()
    assert used & {3, 4, 25, 26, 27, 29, 30}


def test_pet_color_map_samples_photo_hex_for_base_coat(tmp_path):
    from mesh_to_voxels import VoxelGrid
    from PIL import ImageDraw

    p = tmp_path / "warm_cat.png"
    img = Image.new("RGBA", (120, 80), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((10, 12, 110, 68), fill=(198, 158, 118, 255))
    img.save(p)

    occ = np.ones((4, 24, 14), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))
    gpt_data = {
        "regions": [
            {"name": "body", "bbox_normalized": [0.1, 0.1, 0.9, 0.9], "color_name": "Tan"}
        ],
        "recommended_lego_palette": ["Tan", "Cream", "Medium Nougat", "Dark Tan"],
    }

    used, meta = project_pet_color_map(
        grid, p, load_palette(), gpt_data, out_dir=tmp_path, front_axis="-x",
    )
    assert meta["base_adjustment"]["source"].startswith("photo-hex")
    assert meta["base_adjustment"]["sample_hex"].startswith("#")
    assert meta["base_name"] == meta["base_adjustment"]["matched_name"]
    assert meta["base_id"] in used


def test_pet_color_map_samples_light_and_marking_hexes(tmp_path):
    from mesh_to_voxels import VoxelGrid
    from PIL import ImageDraw

    p = tmp_path / "profiled_cat.png"
    img = Image.new("RGBA", (180, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((10, 22, 170, 108), fill=(202, 166, 128, 255))
    d.rectangle((128, 50, 162, 104), fill=(238, 232, 222, 255))
    d.ellipse((70, 38, 138, 94), fill=(64, 57, 50, 255))
    d.ellipse((94, 44, 146, 90), fill=(202, 166, 128, 255))
    img.save(p)

    occ = np.ones((4, 32, 18), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))

    used, meta = project_pet_color_map(
        grid, p, load_palette(), out_dir=tmp_path, front_axis="-x",
    )
    profile = meta["color_profile"]
    assert profile["light"]["sample_hex"].startswith("#")
    assert profile["marking"]["sample_hex"].startswith("#")
    assert profile["light"]["matched_name"] in {"White", "Cream"}
    assert profile["marking"]["matched_name"] in {
        "Black", "Dark Bluish Gray", "Dark Brown", "Brown",
        "Reddish Brown", "Dark Tan", "Medium Nougat",
    }
    assert profile["light"]["matched_id"] in used
    assert profile["marking"]["matched_id"] in used


def test_pet_color_map_filters_head_noise_and_keeps_tail_tip(tmp_path):
    from mesh_to_voxels import VoxelGrid
    from PIL import ImageDraw

    p = tmp_path / "side_cat_head_tail.png"
    img = Image.new("RGBA", (320, 210), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    coat = (214, 174, 125, 255)
    dark = (48, 47, 44, 255)
    d.ellipse((30, 55, 260, 175), fill=coat)
    d.ellipse((240, 28, 318, 118), fill=coat)
    d.polygon([(300, 25), (318, 0), (315, 58)], fill=coat)
    d.line((40, 148, 5, 195), fill=coat, width=23)
    d.ellipse((0, 178, 25, 208), fill=dark)       # real dark tail tip
    d.rectangle((38, 140, 52, 156), fill=dark)    # tail shadow/stripe to drop
    d.rectangle((282, 45, 300, 88), fill=dark)    # head/ear/eye noise to drop
    d.ellipse((128, 65, 248, 170), fill=dark)     # body crescent
    d.ellipse((160, 74, 255, 165), fill=coat)
    for cx, cy in [(72, 82), (62, 118), (95, 103)]:
        d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=dark)
    img.save(p)

    occ = np.ones((4, 65, 42), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))

    _used, meta = project_pet_color_map(
        grid, p, load_palette(), out_dir=tmp_path, front_axis="-x", side_depth="front",
    )
    boxes = meta["markings"]["boxes"]
    assert meta["markings"]["side_profile_filter"]["head_side"] == "right"
    assert not any(b[0] >= 50 for b in boxes), boxes
    assert not any(9 <= b[0] <= 15 and b[2] <= 18 for b in boxes), boxes
    assert any(b[0] <= 4 and b[1] >= 30 for b in boxes), boxes
    assert any(24 <= b[0] <= 35 and 12 <= b[1] <= 18 for b in boxes), boxes


def test_pet_color_map_projects_side_pet_along_body_length(tmp_path):
    from mesh_to_voxels import VoxelGrid

    p = tmp_path / "side_pet_patch.png"
    img = Image.new("RGBA", (100, 70), (0, 0, 0, 0))
    for y in range(5, 65):
        for x in range(5, 95):
            img.putpixel((x, y), (240, 240, 235, 255))
    for y in range(25, 60):
        for x in range(65, 80):
            img.putpixel((x, y), (70, 70, 65, 255))
    img.save(p)

    occ = np.ones((4, 12, 8), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))

    used, meta = project_pet_color_map(
        grid, p, load_palette(), out_dir=tmp_path, front_axis="-x",
    )
    assert meta["u_axis"] == 1
    assert meta["grid_map_shape"] == [12, 8]
    assert used & {3, 4, 25, 26, 27, 29, 30}

    brightness = grid.colors.astype(np.float32).mean(axis=3)
    marked_body_length = brightness[:, 8:11, :][grid.occupancy[:, 8:11, :]]
    unmarked_body_length = brightness[:, 1:4, :][grid.occupancy[:, 1:4, :]]
    assert float(marked_body_length.mean()) < float(unmarked_body_length.mean()) - 20.0


def test_pet_color_map_keeps_one_side_marking_on_visible_surface(tmp_path):
    from mesh_to_voxels import VoxelGrid

    p = tmp_path / "right_side_patch.png"
    img = Image.new("RGBA", (100, 70), (0, 0, 0, 0))
    for y in range(5, 65):
        for x in range(5, 95):
            img.putpixel((x, y), (218, 178, 126, 255))
    for y in range(24, 52):
        for x in range(50, 72):
            img.putpixel((x, y), (58, 58, 55, 255))
    img.save(p)

    occ = np.ones((6, 12, 8), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))

    used, meta = project_pet_color_map(
        grid, p, load_palette(), out_dir=tmp_path, front_axis="-x",
        side_depth="front", markings_on_visible_side_only=True,
    )
    assert meta["side_depth"] == "front"
    assert meta["markings_on_visible_side_only"] is True
    assert used & {3, 4, 25, 26, 27, 29, 30}

    brightness = grid.colors.astype(np.float32).mean(axis=3)
    front_marked = brightness[0:2, 6:9, 2:5][grid.occupancy[0:2, 6:9, 2:5]]
    back_same_uv = brightness[-2:, 6:9, 2:5][grid.occupancy[-2:, 6:9, 2:5]]
    assert float(front_marked.mean()) < float(back_same_uv.mean()) - 30.0


def test_pet_color_map_keeps_side_only_light_patch_on_visible_surface(tmp_path):
    from mesh_to_voxels import VoxelGrid

    p = tmp_path / "right_side_white_patch.png"
    img = Image.new("RGBA", (100, 70), (0, 0, 0, 0))
    for y in range(5, 65):
        for x in range(5, 95):
            img.putpixel((x, y), (218, 178, 126, 255))
    for y in range(34, 56):
        for x in range(50, 72):
            img.putpixel((x, y), (241, 238, 228, 255))
    img.save(p)

    occ = np.ones((6, 12, 8), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))

    used, meta = project_pet_color_map(
        grid, p, load_palette(), out_dir=tmp_path, front_axis="-x",
        side_depth="front", markings_on_visible_side_only=True,
    )
    assert meta["side_depth"] == "front"
    assert meta["anatomy_light"]["visible_side_only_cells"] > 0
    assert 1 in used

    brightness = grid.colors.astype(np.float32).mean(axis=3)
    front_light = brightness[0:2, 6:9, 1:5][grid.occupancy[0:2, 6:9, 1:5]]
    back_same_uv = brightness[-2:, 6:9, 1:5][grid.occupancy[-2:, 6:9, 1:5]]
    assert float(front_light.mean()) > float(back_same_uv.mean()) + 25.0


def test_pet_color_map_preserves_broad_white_chest_region(tmp_path):
    from mesh_to_voxels import VoxelGrid

    p = tmp_path / "tan_cat_white_chest.png"
    img = Image.new("RGBA", (120, 80), (0, 0, 0, 0))
    for y in range(5, 75):
        for x in range(5, 115):
            img.putpixel((x, y), (214, 174, 125, 255))
    for y in range(22, 72):
        for x in range(82, 112):
            img.putpixel((x, y), (238, 236, 226, 255))
    img.save(p)

    occ = np.ones((4, 12, 8), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))
    gpt_data = {
        "regions": [
            {"name": "body", "bbox_normalized": [0.04, 0.06, 0.96, 0.94], "color_name": "Tan"},
            {"name": "chest", "bbox_normalized": [82 / 120, 22 / 80, 112 / 120, 72 / 80], "color_name": "White"},
        ],
        "recommended_lego_palette": ["Tan", "White", "Dark Tan"],
    }

    used, meta = project_pet_color_map(
        grid, p, load_palette(), gpt_data, out_dir=tmp_path, front_axis="-x",
    )
    assert meta["base_name"] in {"Tan", "Medium Nougat"}
    assert meta["gpt_regions"]["regions"] >= 1 or meta["light_regions"]["components"] >= 1
    assert 1 in used
    brightness = grid.colors.astype(np.float32).mean(axis=3)
    white_chest = brightness[:, 9:12, 1:6][grid.occupancy[:, 9:12, 1:6]]
    tan_body = brightness[:, 1:5, 1:6][grid.occupancy[:, 1:5, 1:6]]
    assert float(white_chest.mean()) > float(tan_body.mean()) + 35.0


def test_pet_color_map_propagates_light_anatomy_regions(tmp_path):
    from mesh_to_voxels import VoxelGrid
    from PIL import ImageDraw

    p = tmp_path / "side_pet_white_socks_belly_muzzle.png"
    img = Image.new("RGBA", (200, 120), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tan = (202, 166, 128, 255)
    light = (240, 236, 226, 255)
    dark = (58, 56, 52, 255)
    d.rectangle((30, 36, 150, 82), fill=tan)          # torso
    d.rectangle((145, 20, 194, 70), fill=tan)         # head
    d.polygon([(162, 20), (173, 2), (182, 22)], fill=tan)
    d.rectangle((135, 72, 154, 115), fill=tan)        # front leg
    d.rectangle((56, 76, 75, 115), fill=tan)          # rear leg
    d.line((30, 76, 4, 102), fill=tan, width=18)      # tail
    d.rectangle((144, 54, 164, 106), fill=light)      # chest/front leg
    d.rectangle((86, 72, 138, 92), fill=light)        # underbelly
    d.rectangle((135, 98, 158, 118), fill=light)      # front paw
    d.rectangle((50, 98, 80, 118), fill=light)        # rear paw
    d.rectangle((174, 40, 196, 58), fill=light)       # muzzle
    d.ellipse((86, 48, 126, 80), fill=dark)           # side marking
    d.ellipse((104, 52, 132, 77), fill=tan)
    img.save(p)

    occ = np.ones((4, 40, 24), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))
    gpt_data = {
        "regions": [
            {"name": "body", "bbox_normalized": [0.15, 0.30, 0.75, 0.70], "color_name": "Tan"},
            {"name": "chest", "bbox_normalized": [0.72, 0.45, 0.82, 0.88], "color_name": "White"},
            {"name": "belly", "bbox_normalized": [0.43, 0.60, 0.69, 0.77], "color_name": "White"},
            {"name": "paws", "bbox_normalized": [0.25, 0.80, 0.79, 0.98], "color_name": "White"},
            {"name": "muzzle", "bbox_normalized": [0.87, 0.33, 0.98, 0.50], "color_name": "White"},
        ],
        "recommended_lego_palette": ["Tan", "White", "Dark Bluish Gray"],
    }

    used, meta = project_pet_color_map(
        grid,
        p,
        load_palette(),
        gpt_data,
        out_dir=tmp_path,
        front_axis="-x",
        side_depth="front",
        markings_on_visible_side_only=True,
        front_photo_path=p,
    )

    assert meta["anatomy_light"]["applied"] is True
    assert meta["front_light_profile"]["applied"] is True
    assert meta["anatomy_light"]["front_photo_used"] is True
    assert meta["anatomy_debug_path"]
    assert Path(meta["anatomy_debug_path"]).exists()
    assert 1 in used
    brightness = grid.colors.astype(np.float32).mean(axis=3)
    tan_body = brightness[:, 6:16, 12:17][grid.occupancy[:, 6:16, 12:17]]
    chest = brightness[:, 29:35, 5:14][grid.occupancy[:, 29:35, 5:14]]
    belly = brightness[0:2, 17:28, 5:11][grid.occupancy[0:2, 17:28, 5:11]]
    rear_paw = brightness[:, 10:16, 0:5][grid.occupancy[:, 10:16, 0:5]]
    front_paw = brightness[:, 27:34, 0:5][grid.occupancy[:, 27:34, 0:5]]
    assert float(chest.mean()) > float(tan_body.mean()) + 25.0
    assert float(np.median(belly)) > float(tan_body.mean()) + 20.0
    assert float(rear_paw.mean()) > float(tan_body.mean()) + 20.0
    assert float(front_paw.mean()) > float(tan_body.mean()) + 20.0


def test_pet_color_map_keeps_white_base_and_rejects_broad_shadows(tmp_path):
    from mesh_to_voxels import VoxelGrid

    p = tmp_path / "white_cat_shadow.png"
    img = Image.new("RGBA", (120, 80), (0, 0, 0, 0))
    for y in range(5, 75):
        for x in range(5, 115):
            img.putpixel((x, y), (240, 240, 236, 255))
    # Broad photo shadow: should collapse back to white, not become gray fur.
    for y in range(30, 68):
        for x in range(42, 95):
            img.putpixel((x, y), (172, 176, 174, 255))
    # True one-leg stripe marking inside GPT-marked region.
    for y in range(35, 70):
        for x in range(72, 82):
            img.putpixel((x, y), (45, 48, 45, 255))
    img.save(p)

    occ = np.ones((4, 12, 8), dtype=bool)
    colors = np.zeros((*occ.shape, 3), dtype=np.uint8)
    grid = VoxelGrid(occ, colors, pitch=1.0, origin=np.zeros(3))
    gpt_data = {
        "regions": [
            {
                "name": "body",
                "bbox_normalized": [5 / 120, 5 / 80, 115 / 120, 75 / 80],
                "color_name": "White",
            },
            {
                "name": "striped_leg_pattern",
                "bbox_normalized": [72 / 120, 35 / 80, 82 / 120, 70 / 80],
                "color_name": "Black",
            }
        ],
        "recommended_lego_palette": ["White", "Black", "Light Bluish Gray"],
    }

    used, meta = project_pet_color_map(
        grid, p, load_palette(), gpt_data, out_dir=tmp_path, front_axis="-x",
    )
    assert meta["base_name"] == "White"
    assert meta["markings"]["components"] >= 1
    assert 1 in used
    brightness = grid.colors.astype(np.float32).mean(axis=3)
    shadow_region = brightness[:, 2:5, :][grid.occupancy[:, 2:5, :]]
    assert float(shadow_region.mean()) > 210.0


# --- standalone runner ---

def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return fails == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
