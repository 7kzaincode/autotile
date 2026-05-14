"""Prompt builders for pet reference-view generation.

This module does not call an image model. It creates the identity extraction,
per-view generation, and QA prompts used by the optional standardized
reference-image layer. Keeping prompt construction here makes it testable and
lets the app inspect the exact instructions before spending image credits.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


VIEW_KEYS = ("front_view", "right_side_view", "left_side_view", "rear_view")

VIEW_OUTPUTS = {
    "front_view": "front_reference.png",
    "right_side_view": "right_reference.png",
    "left_side_view": "left_reference.png",
    "rear_view": "back_reference.png",
}

VIEW_TITLES = {
    "front_view": "Front View",
    "right_side_view": "Right-Side Profile",
    "left_side_view": "Left-Side Profile",
    "rear_view": "Rear / Back View",
}

VIEW_SPECIFIC_TASKS = {
    "front_view": (
        "Generate the animal from a direct front view. The camera should face "
        "the animal straight-on. The animal's face, chest, front legs, and "
        "front paws should be visible and centered."
    ),
    "right_side_view": (
        "Generate the animal from its anatomical right side only. The camera "
        "should be perpendicular to the animal's right side, like a clinical "
        "side-profile reference photo."
    ),
    "left_side_view": (
        "Generate the animal from its anatomical left side only. The camera "
        "should be perpendicular to the animal's left side, like a clinical "
        "side-profile reference photo."
    ),
    "rear_view": (
        "Generate the animal from directly behind. The camera should face the "
        "animal's back and rear symmetrically."
    ),
}


DEFAULT_PROFILE: dict[str, Any] = {
    "species": "",
    "breed_or_type": "",
    "body_shape": "",
    "fur_or_skin_texture": "",
    "base_coat": "",
    "eye_description": "",
    "ear_description": "",
    "face_description": "",
    "tail_description": "",
    "default_pose": "neutral standing four-point stance",
    "front_view": {
        "confidence": "low",
        "visible_features": "",
        "visible_markings": "",
        "unknown_or_unclear": "",
    },
    "right_side_view": {
        "confidence": "low",
        "visible_features": "",
        "visible_markings": "",
        "unknown_or_unclear": "",
    },
    "left_side_view": {
        "confidence": "low",
        "visible_features": "",
        "visible_markings": "",
        "unknown_or_unclear": "",
    },
    "rear_view": {
        "confidence": "low",
        "visible_features": "",
        "visible_markings": "",
        "unknown_or_unclear": "",
    },
    "global_distinguishing_features": [],
    "strict_do_not_invent": [],
    "notes_for_generation": "",
}


def _clean_text(value: Any, fallback: str = "not specified") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [text]


def normalize_identity_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Fill missing identity-profile fields without inventing pet details."""
    profile = profile or {}
    out = deepcopy(DEFAULT_PROFILE)

    for key in (
        "species",
        "breed_or_type",
        "body_shape",
        "fur_or_skin_texture",
        "base_coat",
        "eye_description",
        "ear_description",
        "face_description",
        "tail_description",
        "default_pose",
        "notes_for_generation",
    ):
        if key in profile and profile[key] is not None:
            out[key] = str(profile[key]).strip()

    for key in VIEW_KEYS:
        incoming = profile.get(key) or {}
        if not isinstance(incoming, dict):
            incoming = {}
        view = out[key]
        for field in ("confidence", "visible_features", "visible_markings", "unknown_or_unclear"):
            if field in incoming and incoming[field] is not None:
                view[field] = str(incoming[field]).strip()
        confidence = view.get("confidence", "").lower().strip()
        if confidence not in {"high", "medium", "low"}:
            view["confidence"] = "low"

    out["global_distinguishing_features"] = _as_list(
        profile.get("global_distinguishing_features")
    )
    out["strict_do_not_invent"] = _as_list(profile.get("strict_do_not_invent"))
    return out


def build_identity_extraction_prompt(source_files: list[str] | None = None) -> str:
    """Prompt for a vision model to extract a structured animal profile."""
    files = [str(f) for f in (source_files or []) if str(f).strip()]
    file_block = ""
    if files:
        file_block = "\nReference photo filenames:\n" + "\n".join(f"- {f}" for f in files) + "\n"

    return f"""You are analyzing reference photos of a single animal.
{file_block}
Create a structured animal_identity_profile that can be used to generate clean medical-style reference images of the same animal from multiple angles.

Your job is to describe only what is visible or strongly supported by the photos. Do not invent markings. Do not assume the left and right sides are symmetrical. Do not copy a marking from one side to the other unless both sides are clearly shown or explicitly described.

Return the result as JSON only.

Use this schema:

{{
  "species": "",
  "breed_or_type": "",
  "body_shape": "",
  "fur_or_skin_texture": "",
  "base_coat": "",
  "eye_description": "",
  "ear_description": "",
  "face_description": "",
  "tail_description": "",
  "default_pose": "neutral standing four-point stance",
  "front_view": {{
    "confidence": "high | medium | low",
    "visible_features": "",
    "visible_markings": "",
    "unknown_or_unclear": ""
  }},
  "right_side_view": {{
    "confidence": "high | medium | low",
    "visible_features": "",
    "visible_markings": "",
    "unknown_or_unclear": ""
  }},
  "left_side_view": {{
    "confidence": "high | medium | low",
    "visible_features": "",
    "visible_markings": "",
    "unknown_or_unclear": ""
  }},
  "rear_view": {{
    "confidence": "high | medium | low",
    "visible_features": "",
    "visible_markings": "",
    "unknown_or_unclear": ""
  }},
  "global_distinguishing_features": [],
  "strict_do_not_invent": [],
  "notes_for_generation": ""
}}

Rules:
- If a side is not visible in the photos, mark its confidence as low.
- If a marking is visible only on the right side, describe it only in right_side_view.
- If a marking is visible only on the left side, describe it only in left_side_view.
- If a rear view is not provided, do not invent rear markings. Use simple, plausible continuation of the base coat only.
- Tail position should be based on the real photos. If the real photos show the tail relaxed/downward, all generated views should use a relaxed downward tail.
- Avoid decorative, artistic, fantasy, cartoon, or stylized wording.
- The goal is anatomical/reference accuracy, not a beautiful illustration.
"""


def build_shared_identity_block(profile: dict[str, Any] | None) -> str:
    p = normalize_identity_profile(profile)
    features = p["global_distinguishing_features"]
    strict = p["strict_do_not_invent"]
    feature_lines = "\n".join(f"- {x}" for x in features) if features else "- none specified"
    strict_lines = "\n".join(f"- {x}" for x in strict) if strict else "- no unsupported markings or traits"

    return f"""Shared animal identity:

Species: {_clean_text(p.get("species"))}
Breed/type: {_clean_text(p.get("breed_or_type"))}
Body shape: {_clean_text(p.get("body_shape"))}
Fur/skin texture: {_clean_text(p.get("fur_or_skin_texture"))}
Base coat: {_clean_text(p.get("base_coat"))}
Eyes: {_clean_text(p.get("eye_description"))}
Ears: {_clean_text(p.get("ear_description"))}
Face: {_clean_text(p.get("face_description"))}
Tail: {_clean_text(p.get("tail_description"))}

Global distinguishing features:
{feature_lines}

Strict do-not-invent list:
{strict_lines}

Generation rules:
- Generate the same individual animal described above.
- Use a plain light-grey or off-white studio background.
- Use flat, even, clinical lighting.
- Use sharp focus and high detail.
- Show the full body clearly.
- Use a neutral standing four-point stance.
- Keep all visible legs natural and uncrossed.
- Keep the tail relaxed and downward unless the identity profile explicitly says otherwise.
- Do not add props, collars, text, labels, borders, grass, furniture, outdoor scenery, or other animals.
- Do not stylize the animal.
- Do not make the image look like a cartoon, painting, plush toy, toy model, or fantasy creature.
- Do not invent new distinctive markings.
- Do not mirror markings from one side to the other.
- If a region is unknown, keep it simple and naturally consistent with the base coat.
"""


def _view_requirements(view_key: str) -> list[str]:
    if view_key == "front_view":
        return [
            "The animal must face the camera directly.",
            "The body should be aligned forward, not angled sideways.",
            "Both eyes should be visible if anatomically appropriate.",
            "The chest and front legs should be clearly visible.",
            "Preserve all front-visible markings described in the identity profile.",
            "Do not add side markings that are not visible from the front.",
            "Tail should remain relaxed/downward if visible.",
            "The image should look like a clean clinical reference photo for segmentation and comparison.",
        ]
    if view_key == "rear_view":
        return [
            "Show the animal from a direct rear/back view.",
            "The back, hindquarters, hind legs, rear paws, tail position, and rear-visible coat should be clear.",
            "The animal should not turn its head back toward the camera.",
            "This should be a true rear view, not a three-quarter rear angle.",
            "Preserve only rear markings that are visible or explicitly described in the identity profile.",
            "Do not invent rear markings.",
            "Do not assume right-side or left-side markings wrap around to the back unless explicitly supported.",
            "If the rear is unknown, keep the rear coat simple and naturally consistent with the base coat.",
            "Tail should remain relaxed/downward and centered/natural unless the identity profile explicitly says otherwise.",
            "The full body must be visible, centered, and uncropped.",
        ]

    side = "right" if view_key == "right_side_view" else "left"
    other = "left" if side == "right" else "right"
    return [
        f"Show the animal's {side} side.",
        "The animal's head, nose, eyes, chest, torso, legs, and feet must point in the same forward direction as the body.",
        "The animal must not turn its head toward the camera.",
        "This should be a true side profile, not a three-quarter view.",
        f"Preserve only the markings specifically described for the {side} side.",
        f"Do not copy, mirror, or transfer {other}-side markings onto the {side} side.",
        f"Do not invent new {side}-side markings.",
        f"If part of the {side} side is unknown, keep that area simple and naturally consistent with the base coat.",
        "Tail should remain relaxed/downward and anatomically natural.",
        "The full body must be visible, centered, and uncropped.",
    ]


def _negative_constraints(view_key: str) -> str:
    base = [
        "No text",
        "no labels",
        "no borders",
        "no collage",
        "no grid",
        "no props",
        "no extra animals",
        "no distorted limbs",
        "no missing legs",
        "no extra legs",
        "no artistic filters",
    ]
    if view_key in {"right_side_view", "left_side_view"}:
        base.extend(["no three-quarter angle", "no head turned toward viewer"])
    if view_key == "rear_view":
        base.extend(["no three-quarter angle", "no head turned back toward camera"])
    base.extend(["no raised tail", "no curled tail"])
    return ", ".join(base) + "."


def build_view_prompt(profile: dict[str, Any] | None, view_key: str) -> dict[str, str]:
    """Build one view-specific image prompt."""
    if view_key not in VIEW_KEYS:
        raise ValueError(f"unknown view key: {view_key}")
    p = normalize_identity_profile(profile)
    view = p[view_key]
    requirements = "\n".join(f"- {line}" for line in _view_requirements(view_key))

    prompt = f"""Output file: {VIEW_OUTPUTS[view_key]}

Create a full-resolution medical-style {VIEW_TITLES[view_key].lower()} reference image of the animal.

{build_shared_identity_block(p)}

View-specific task:
{VIEW_SPECIFIC_TASKS[view_key]}

{VIEW_TITLES[view_key]} evidence:
Confidence: {_clean_text(view.get("confidence"), "low")}
Visible features: {_clean_text(view.get("visible_features"))}
Visible markings: {_clean_text(view.get("visible_markings"))}
Unknown or unclear areas: {_clean_text(view.get("unknown_or_unclear"))}

Requirements:
{requirements}

Negative constraints:
{_negative_constraints(view_key)}
"""
    return {
        "view": view_key,
        "output_file": VIEW_OUTPUTS[view_key],
        "prompt": prompt,
    }


def build_all_view_prompts(profile: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    return {key: build_view_prompt(profile, key) for key in VIEW_KEYS}


def build_qa_prompt(
    profile: dict[str, Any] | None,
    generated_files: list[str] | None = None,
) -> str:
    files = generated_files or [VIEW_OUTPUTS[key] for key in VIEW_KEYS]
    file_lines = "\n".join(f"- {f}" for f in files)
    p = normalize_identity_profile(profile)

    return f"""Evaluate these generated animal reference images against the original animal_identity_profile.

Animal identity profile:
{p}

Check each file separately:
{file_lines}

Return JSON only.

For each image, score:
{{
  "file": "",
  "view_correctness": 0-10,
  "identity_consistency": 0-10,
  "marking_accuracy": 0-10,
  "pose_quality": 0-10,
  "tail_position": 0-10,
  "segmentation_cleanliness": 0-10,
  "major_errors": [],
  "minor_errors": [],
  "should_regenerate": true,
  "regeneration_instruction": ""
}}

Rules:
- Penalize any side-profile image where the animal turns its head toward the camera.
- Penalize any image where the tail is raised or curled when the identity profile requires a relaxed downward tail.
- Penalize invented markings.
- Penalize mirrored markings that are not supported by the identity profile.
- Penalize extra legs, missing legs, distorted anatomy, props, text, borders, or background clutter.
- If only one view fails, recommend regenerating only that view.
"""


def build_reference_prompt_bundle(
    profile: dict[str, Any] | None,
    source_files: list[str] | None = None,
    generated_files: list[str] | None = None,
) -> dict[str, Any]:
    """Return every prompt needed for the standardized reference layer."""
    normalized = normalize_identity_profile(profile)
    return {
        "animal_identity_profile": normalized,
        "identity_extraction_prompt": build_identity_extraction_prompt(source_files),
        "shared_identity_block": build_shared_identity_block(normalized),
        "view_prompts": build_all_view_prompts(normalized),
        "qa_prompt": build_qa_prompt(normalized, generated_files),
        "recommended_generation": {
            "mode": "four_separate_images",
            "reason": (
                "Separate images keep each view higher resolution and avoid "
                "cross-quadrant marking bleed or hidden collage distortions."
            ),
            "suggested_size": "1024x1024",
        },
    }
