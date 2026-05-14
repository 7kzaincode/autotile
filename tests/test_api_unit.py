from __future__ import annotations

from fastapi.testclient import TestClient

import photo_to_mesh
import server
import server_logs
from subject_preset import PRESETS


def _patch_server_dirs(monkeypatch, tmp_path):
    photos = tmp_path / "photos"
    meshes = tmp_path / "meshes"
    outputs = tmp_path / "output"
    for p in (photos, meshes, outputs):
        p.mkdir()
    monkeypatch.setattr(server, "PHOTOS_DIR", photos)
    monkeypatch.setattr(server, "MESHES_DIR", meshes)
    monkeypatch.setattr(server, "OUTPUT_DIR", outputs)
    return photos, meshes, outputs


def _patch_cache_dirs(monkeypatch, tmp_path):
    dirs = {
        "GPT_CACHE_DIR": tmp_path / "gpt_cache",
        "MESH_CACHE_DIR": tmp_path / "meshes",
        "OUTPUT_DIR": tmp_path / "output",
        "PHOTOS_DIR": tmp_path / "photos",
    }
    for name, path in dirs.items():
        path.mkdir()
        monkeypatch.setattr(server_logs, name, path)
    return dirs


def test_presets_endpoint_uses_subject_preset_source():
    client = TestClient(server.app)
    r = client.get("/api/presets")
    assert r.status_code == 200
    data = r.json()
    assert data["order"] == list(PRESETS.keys())
    assert data["presets"]["pet"]["accent_features"] is True
    assert data["presets"]["pet"]["resolution"] == 64
    assert data["presets"]["pet"]["up"] == "auto"
    assert data["presets"]["pet"]["voxel_hole_fill"] is True
    assert data["presets"]["pet"]["slopes"] is False
    assert data["presets"]["pet"]["slope_inv"] is False
    assert data["presets"]["vehicle"]["semantic_regions"] == 10


def test_safe_upload_name_normalizes_markup_and_unicode():
    stem, suffix = server._safe_upload_name("../bunny 🐰 <script>.WEBP")
    assert stem == "bunny _ _script"
    assert suffix == ".webp"


def test_safe_upload_name_replaces_bad_suffix():
    stem, suffix = server._safe_upload_name("ship.ldr;rm", default_suffix=".obj")
    assert stem == "ship"
    assert suffix == ".obj"


def test_reference_view_filename_inference():
    assert server._infer_reference_view_from_filename("cat front.png") == "front"
    assert server._infer_reference_view_from_filename("cat face reference.png") == "front"
    assert server._infer_reference_view_from_filename("cat back.png") == "back"
    assert server._infer_reference_view_from_filename("cat rear view.png") == "back"
    assert server._infer_reference_view_from_filename("cat r side.png") == "right"
    assert server._infer_reference_view_from_filename("cat right-side.png") == "right"
    assert server._infer_reference_view_from_filename("cat l side.png") == "left"
    assert server._infer_reference_view_from_filename("cat left_profile.png") == "left"
    assert server._infer_reference_view_from_filename("bright cat.png") is None
    assert server._infer_reference_view_from_filename("cat side.png") is None


def test_cache_clear_all_endpoint_removes_all_cache_buckets(monkeypatch, tmp_path):
    dirs = _patch_cache_dirs(monkeypatch, tmp_path)
    for name, path in dirs.items():
        (path / f"{name}.txt").write_text("cached")
    client = TestClient(server.app)

    r = client.delete("/api/cache")

    assert r.status_code == 200
    data = r.json()
    assert data["cleared"] == "all"
    assert data["total_removed"] == 4
    for path in dirs.values():
        assert list(path.iterdir()) == []


def test_pet_generate_requires_side_photo_before_ai_calls(monkeypatch, tmp_path):
    _patch_server_dirs(monkeypatch, tmp_path)
    client = TestClient(server.app)
    r = client.post(
        "/api/generate",
        data={
            "subject": "pet",
            "preset_client_applied": "true",
            "remove_bg": "false",
        },
        files={"photo": ("front.png", b"not-a-real-image", "image/png")},
    )
    assert r.status_code == 400
    assert "front face photo plus at least one side" in r.text


def test_pet_generate_routes_side_photo_to_mesh_and_front_to_features(monkeypatch, tmp_path):
    _photos, meshes, _outputs = _patch_server_dirs(monkeypatch, tmp_path)
    client = TestClient(server.app)
    captured = {}
    fake_mesh = meshes / "unit_test_pet_mesh.obj"
    fake_mesh.write_text("# fake mesh\n")

    def fake_photo_to_mesh(photo_path, out_dir, model, extra_inputs=None):
        captured["mesh_photo"] = photo_path
        captured["extra_inputs"] = extra_inputs or {}
        return fake_mesh

    def fake_run_pipeline(mesh_path, **kwargs):
        captured["pipeline_kwargs"] = kwargs
        return {
            "source": mesh_path.name,
            "resolution": kwargs["resolution"],
            "grid_shape": [1, 1, 1],
            "pitch": 1.0,
            "voxel_metadata": {"up_axis": kwargs["up_axis"], "front_axis": "-y"},
            "palette": [],
            "bricks": [],
        }

    monkeypatch.setattr(photo_to_mesh, "photo_to_mesh", fake_photo_to_mesh)
    monkeypatch.setattr(server, "_run_pipeline", fake_run_pipeline)

    r = client.post(
        "/api/generate",
        data={
            "subject": "pet",
            "preset_client_applied": "true",
            "remove_bg": "false",
            "mesh_cleanup": "false",
            "use_gpt_vision": "false",
            "use_sam": "false",
        },
        files={
            "photo": ("face.png", b"face-bytes", "image/png"),
            "photo_left": ("left.png", b"left-bytes", "image/png"),
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "_left_left" in captured["mesh_photo"].name
    assert "_left_left" in captured["pipeline_kwargs"]["photo_path"].name
    assert "_left_left" in captured["pipeline_kwargs"]["color_photo_path"].name
    assert "_front_face" in captured["pipeline_kwargs"]["face_photo_path"].name
    assert captured["pipeline_kwargs"]["body_view"] == "left"
    assert "_left_left" in data["_photo_name"]
    assert "_front_face" in data["_face_photo_name"]
    assert data["_body_view"] == "left"
    mv = captured["extra_inputs"]["multiviews"]
    assert {"front", "left"} <= set(mv)
    assert "_front_face" in mv["front"].name
    assert "_left_left" in mv["left"].name


def test_pet_generate_accepts_autosorted_reference_batch(monkeypatch, tmp_path):
    _photos, meshes, _outputs = _patch_server_dirs(monkeypatch, tmp_path)
    client = TestClient(server.app)
    captured = {}
    fake_mesh = meshes / "unit_test_pet_mesh.obj"
    fake_mesh.write_text("# fake mesh\n")

    def fake_photo_to_mesh(photo_path, out_dir, model, extra_inputs=None):
        captured["mesh_photo"] = photo_path
        captured["extra_inputs"] = extra_inputs or {}
        return fake_mesh

    def fake_run_pipeline(mesh_path, **kwargs):
        captured["pipeline_kwargs"] = kwargs
        return {
            "source": mesh_path.name,
            "resolution": kwargs["resolution"],
            "grid_shape": [1, 1, 1],
            "pitch": 1.0,
            "voxel_metadata": {"up_axis": kwargs["up_axis"], "front_axis": "-y"},
            "palette": [],
            "bricks": [],
        }

    monkeypatch.setattr(photo_to_mesh, "photo_to_mesh", fake_photo_to_mesh)
    monkeypatch.setattr(server, "_run_pipeline", fake_run_pipeline)

    r = client.post(
        "/api/generate",
        data={
            "subject": "pet",
            "preset_client_applied": "true",
            "remove_bg": "false",
            "mesh_cleanup": "false",
            "use_gpt_vision": "false",
            "use_sam": "false",
        },
        files=[
            ("photo_refs", ("cat r side.png", b"right-bytes", "image/png")),
            ("photo_refs", ("cat front.png", b"front-bytes", "image/png")),
            ("photo_refs", ("cat back.png", b"back-bytes", "image/png")),
            ("photo_refs", ("cat l side.png", b"left-bytes", "image/png")),
        ],
    )

    assert r.status_code == 200, r.text
    data = r.json()
    assert "_left_cat l side" in captured["mesh_photo"].name
    assert "_front_cat front" in captured["pipeline_kwargs"]["face_photo_path"].name
    assert "_right_cat r side" in captured["pipeline_kwargs"]["other_side_photo_path"].name
    assert captured["pipeline_kwargs"]["body_view"] == "left"
    assert captured["pipeline_kwargs"]["other_side_view"] == "right"
    mv = captured["extra_inputs"]["multiviews"]
    assert {"front", "back", "left", "right"} <= set(mv)
    assert "_back_cat back" in mv["back"].name
    assert data["_body_view"] == "left"
    assert data["_other_side_view"] == "right"


def test_pet_reference_prompt_endpoints_build_multiview_bundle():
    client = TestClient(server.app)
    profile = {
        "species": "cat",
        "base_coat": "warm tan coat",
        "right_side_view": {
            "confidence": "high",
            "visible_markings": "one irregular dark side patch",
            "unknown_or_unclear": "",
        },
        "left_side_view": {
            "confidence": "low",
            "unknown_or_unclear": "left side not shown",
        },
    }

    r = client.post(
        "/api/pet-reference/prompts",
        json={
            "animal_identity_profile": profile,
            "source_files": ["front.png", "right.png"],
        },
    )

    assert r.status_code == 200, r.text
    data = r.json()
    assert data["recommended_generation"]["mode"] == "four_separate_images"
    assert "front.png" in data["identity_extraction_prompt"]
    assert "right_reference.png" == data["view_prompts"]["right_side_view"]["output_file"]
    assert "one irregular dark side patch" in data["view_prompts"]["right_side_view"]["prompt"]
    assert "Do not copy, mirror, or transfer right-side markings onto the left side" in data["view_prompts"]["left_side_view"]["prompt"]


def test_pet_reference_qa_prompt_endpoint():
    client = TestClient(server.app)
    r = client.post(
        "/api/pet-reference/qa-prompt",
        json={
            "animal_identity_profile": {"species": "dog"},
            "generated_files": ["front.png", "right.png"],
        },
    )
    assert r.status_code == 200
    prompt = r.json()["prompt"]
    assert "Penalize invented markings" in prompt
    assert "front.png" in prompt
    assert "right.png" in prompt
