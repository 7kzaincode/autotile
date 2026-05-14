from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pet_reference_prompts import (  # noqa: E402
    VIEW_KEYS,
    build_identity_extraction_prompt,
    build_qa_prompt,
    build_reference_prompt_bundle,
    build_view_prompt,
    normalize_identity_profile,
)


def _sample_profile():
    return {
        "species": "cat",
        "breed_or_type": "domestic short hair",
        "body_shape": "slender standing cat with long tail",
        "fur_or_skin_texture": "short smooth fur",
        "base_coat": "warm tan coat with white chest",
        "eye_description": "amber eyes",
        "ear_description": "upright triangular ears",
        "face_description": "tan forehead and cheeks with white muzzle",
        "tail_description": "long relaxed downward tail",
        "front_view": {
            "confidence": "high",
            "visible_features": "front face, chest, front legs",
            "visible_markings": "white chest blaze and white muzzle; tan forehead remains visible",
            "unknown_or_unclear": "back markings not visible",
        },
        "right_side_view": {
            "confidence": "high",
            "visible_features": "right flank, tail, legs",
            "visible_markings": "irregular charcoal crescent, three dark spots, darker tail tip",
            "unknown_or_unclear": "left side not visible",
        },
        "left_side_view": {
            "confidence": "low",
            "visible_features": "",
            "visible_markings": "",
            "unknown_or_unclear": "left side was not provided",
        },
        "rear_view": {
            "confidence": "low",
            "visible_features": "",
            "visible_markings": "",
            "unknown_or_unclear": "rear was not provided",
        },
        "global_distinguishing_features": [
            "warm tan coat",
            "arbitrary dark right-side flank markings",
        ],
        "strict_do_not_invent": [
            "do not mirror the right-side crescent to the left side",
        ],
    }


def test_identity_extraction_prompt_forbids_mirroring_and_invention():
    prompt = build_identity_extraction_prompt(["front.png", "right.png"])
    assert "JSON only" in prompt
    assert "Do not invent markings" in prompt
    assert "Do not assume the left and right sides are symmetrical" in prompt
    assert "front.png" in prompt
    assert "right.png" in prompt


def test_normalize_identity_profile_defaults_missing_views_to_low_confidence():
    out = normalize_identity_profile({"species": "dog"})
    assert out["species"] == "dog"
    for key in VIEW_KEYS:
        assert out[key]["confidence"] == "low"


def test_view_prompts_are_separate_images_not_collage():
    profile = _sample_profile()
    front = build_view_prompt(profile, "front_view")
    right = build_view_prompt(profile, "right_side_view")
    assert front["output_file"] == "front_reference.png"
    assert right["output_file"] == "right_reference.png"
    assert "no collage" in front["prompt"]
    assert "no grid" in right["prompt"]
    assert "full body" in right["prompt"]


def test_missing_side_prompt_keeps_unknown_area_simple_and_not_mirrored():
    profile = _sample_profile()
    left = build_view_prompt(profile, "left_side_view")["prompt"]
    assert "Confidence: low" in left
    assert "left side was not provided" in left
    assert "Do not copy, mirror, or transfer right-side markings onto the left side" in left
    assert "keep that area simple and naturally consistent with the base coat" in left


def test_prompt_bundle_preserves_arbitrary_marking_language():
    profile = _sample_profile()
    bundle = build_reference_prompt_bundle(profile, source_files=["front.png", "right.png"])
    right_prompt = bundle["view_prompts"]["right_side_view"]["prompt"]
    assert "irregular charcoal crescent" in right_prompt
    assert "three dark spots" in right_prompt
    assert "darker tail tip" in right_prompt
    assert bundle["recommended_generation"]["mode"] == "four_separate_images"


def test_qa_prompt_penalizes_mirrored_markings_and_bad_side_pose():
    qa = build_qa_prompt(_sample_profile())
    assert "Penalize invented markings" in qa
    assert "Penalize mirrored markings" in qa
    assert "turns its head toward the camera" in qa
    assert "front_reference.png" in qa
    assert "right_reference.png" in qa
