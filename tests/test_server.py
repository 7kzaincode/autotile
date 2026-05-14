"""
HTTP integration tests against a running server on localhost:8765.

Skips gracefully if the server isn't reachable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://localhost:8765"


def _first_bunny_json() -> Path | None:
    matches = sorted((ROOT / "output").glob("*bunny*.json"))
    return matches[0] if matches else None


def _server_up() -> bool:
    try:
        r = requests.get(f"{BASE}/viewer/", timeout=2)
        return r.ok
    except Exception:
        return False


def test_static_routes_serve():
    assert _server_up(), "server not running"
    paths = ["/viewer/", "/viewer/main.js"]
    bunny_json = _first_bunny_json()
    if bunny_json is not None:
        paths.append(f"/output/{bunny_json.name}")
    for path in paths:
        r = requests.get(BASE + path, timeout=5)
        assert r.ok, f"{path} returned {r.status_code}"


def test_root_redirect():
    assert _server_up()
    r = requests.get(BASE + "/", timeout=5, allow_redirects=True)
    assert r.ok
    assert "/viewer" in r.url


def _bunny_payload():
    p = _first_bunny_json()
    if p is not None:
        return json.load(open(p))
    palette = json.load(open(ROOT / "lego_palette.json"))
    return {
        "source": "test-fixture",
        "resolution": 8,
        "grid_shape": [4, 4, 3],
        "pitch": 1.0,
        "voxel_metadata": {"up_axis": "z", "front_axis": "-y"},
        "palette": palette,
        "bricks": [
            {
                "x": 0, "y": 0, "z": 0,
                "size_x": 2, "size_y": 2,
                "brick_type": "2x2", "kind": "brick",
                "rotation": 0, "color": 28,
                "slope_dir": None,
            },
            {
                "x": 1, "y": 1, "z": 1,
                "size_x": 1, "size_y": 1,
                "brick_type": "1x1", "kind": "tile",
                "rotation": 0, "color": 4,
                "slope_dir": None,
            },
        ],
    }


def test_parts_list_endpoints():
    assert _server_up()
    payload = _bunny_payload()
    # json
    r = requests.post(f"{BASE}/api/parts-list", json=payload, timeout=15)
    assert r.ok
    data = r.json()
    assert "rows" in data and "summary" in data
    assert data["summary"]["total_bricks"] == len(payload["bricks"])
    # csv
    r = requests.post(f"{BASE}/api/parts-list?format=csv", json=payload, timeout=15)
    assert r.ok
    assert r.headers["Content-Type"].startswith("text/csv")
    assert r.text.splitlines()[0].startswith("quantity,")
    # bricklink xml
    r = requests.post(f"{BASE}/api/parts-list?format=bricklink-xml", json=payload, timeout=15)
    assert r.ok
    assert "<INVENTORY>" in r.text


def test_ldraw_endpoint():
    assert _server_up()
    payload = _bunny_payload()
    r = requests.post(f"{BASE}/api/ldraw", json=payload, timeout=15)
    assert r.ok
    text = r.text
    n_lines = sum(1 for l in text.splitlines() if l.startswith("1 "))
    assert n_lines == len(payload["bricks"])


def test_redecompose_with_cached_mesh():
    assert _server_up()
    meshes = list((ROOT / "test_meshes").glob("*pepsi*.obj"))
    if not meshes:
        print("[skip] no cached pepsi mesh")
        return
    mesh_name = meshes[0].name
    r = requests.post(f"{BASE}/api/redecompose", data={
        "mesh_name": mesh_name,
        "resolution": "32",
        "up": "y",
    }, timeout=120)
    assert r.ok, r.text
    data = r.json()
    assert len(data["bricks"]) > 100


def test_stats_endpoint():
    assert _server_up()
    payload = _bunny_payload()
    r = requests.post(f"{BASE}/api/stats", json=payload, timeout=15)
    assert r.ok
    s = r.json()
    assert s["total_bricks"] == len(payload["bricks"])
    assert "estimated_cost_usd" in s and s["estimated_cost_usd"] > 0
    assert 0.0 <= s["support_score"] <= 1.0


def test_obj_endpoint():
    assert _server_up()
    payload = _bunny_payload()
    r = requests.post(f"{BASE}/api/obj", json=payload, timeout=30)
    assert r.ok
    assert r.headers["Content-Type"] == "application/zip"
    import io as _io
    import zipfile
    z = zipfile.ZipFile(_io.BytesIO(r.content))
    names = set(z.namelist())
    assert {"model.obj", "model.mtl"} <= names


def test_redecompose_rejects_path_traversal():
    assert _server_up()
    r = requests.post(f"{BASE}/api/redecompose", data={
        "mesh_name": "../etc/passwd",
        "resolution": "32",
        "up": "y",
    }, timeout=10)
    assert r.status_code == 400, f"path traversal not blocked: {r.status_code}"


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
    if not _server_up():
        print("Server not running on localhost:8765 — skipping HTTP tests")
        sys.exit(0)
    ok = _run_all()
    sys.exit(0 if ok else 1)
