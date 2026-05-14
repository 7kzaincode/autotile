"""
Subject-type presets. Replaces the half-dozen overlapping color/region knobs
with one dropdown the user actually understands.

Each preset is a complete pipeline config. `resolve(name)` returns a dict of
overrides; missing keys mean "use the request value or the function default."
"""

from __future__ import annotations


PRESETS: dict[str, dict] = {
    # Organic pets — keep unique asymmetric markings; use accents for faces.
    "pet": {
        "resolution": 64,
        "up": "auto",
        "remove_bg": True,
        "mirror": False,
        "mirror_geometry": False,
        "voxel_symmetry": False,
        "voxel_hole_fill": True,
        "voxel_supersample": 0,
        "hollow": True,
        "drop_floaters": True,
        "mesh_smoothing": "light",
        "tiles": True,
        "slopes": False,
        "slope_inv": False,
        "accent_features": True,
        "use_gpt_vision": True,
        "use_sam": True,
        "semantic_color": True,
        "semantic_regions": 8,
        "max_colors": 10,
        "pre_cluster": 16,
        "back_mode": "uv",
    },
    # Ships, cars, planes — symmetric, MORE colors, no anatomical features
    "vehicle": {
        "resolution": 64,
        "up": "auto",
        "remove_bg": True,
        "mirror": True,
        "mirror_geometry": True,
        "voxel_symmetry": True,
        "voxel_hole_fill": True,
        "voxel_supersample": 0,
        "hollow": True,
        "drop_floaters": True,
        "mesh_smoothing": "light",
        "tiles": True,
        "slopes": True,
        "slope_inv": True,
        "accent_features": False,
        "use_gpt_vision": True,
        "use_sam": True,
        "semantic_color": True,
        "semantic_regions": 10,
        "max_colors": 14,
        "pre_cluster": 20,
        "back_mode": "uv",
    },
    # Houses, towers, architecture — block shapes dominate, less mirror benefit
    "building": {
        "resolution": 64,
        "up": "auto",
        "remove_bg": True,
        "mirror": False,
        "mirror_geometry": False,
        "voxel_symmetry": False,
        "voxel_hole_fill": True,
        "voxel_supersample": 0,
        "hollow": True,
        "drop_floaters": True,
        "mesh_smoothing": "light",
        "tiles": True,
        "slopes": True,
        "slope_inv": False,
        "accent_features": False,
        "use_gpt_vision": True,
        "use_sam": True,
        "semantic_color": True,
        "semantic_regions": 8,
        "max_colors": 10,
        "pre_cluster": 16,
        "back_mode": "uv",
    },
    # Everything else / asymmetric / unsure
    "other": {
        "resolution": 64,
        "up": "auto",
        "remove_bg": True,
        "mirror": False,
        "mirror_geometry": False,
        "voxel_symmetry": False,
        "voxel_hole_fill": True,
        "voxel_supersample": 0,
        "hollow": True,
        "drop_floaters": True,
        "mesh_smoothing": "light",
        "tiles": True,
        "slopes": True,
        "slope_inv": True,
        "accent_features": False,
        "use_gpt_vision": True,
        "use_sam": True,
        "semantic_color": True,
        "semantic_regions": 8,
        "max_colors": 10,
        "pre_cluster": 16,
        "back_mode": "uv",
    },
}


def resolve(name: str) -> dict:
    return PRESETS.get((name or "").strip().lower(), {})


def list_presets() -> list[str]:
    return list(PRESETS.keys())
