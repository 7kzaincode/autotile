# AutoTile

Turn a photo or 3D mesh into a buildable brick model, complete with a browser-based 3D preview, parts list, cost estimate, and export files for LDraw, BrickLink, Blender, or 3D printing.

AutoTile is an experimental “photo to bricks” pipeline. It combines AI-generated meshes, semantic image analysis, voxel processing, LEGO-aware colour quantization, and greedy brick decomposition to preserve the subject while keeping the result practical to build.

## What it produces

- Interactive Three.js model preview
- Brick placement JSON
- LDraw `.ldr` model
- Wavefront `.obj` and `.mtl` files
- CSV parts list
- BrickLink Wanted List XML
- Estimated part cost, build time, and stability score

## Pipeline

```text
Photo or mesh
  -> background removal / optional photo-to-3D
  -> mesh cleanup and voxelization
  -> GPT vision + SAM 2 semantic analysis
  -> LEGO palette quantization
  -> brick decomposition and surface detailing
  -> browser preview and manufacturing exports
```

The colour pipeline uses a three-level fallback: SAM 2 masks when available, GPT-derived regions if segmentation fails, and k-means clustering as the always-available path. AI integrations fail gracefully so an optional model outage does not have to stop the complete conversion pipeline.

## Highlights

- Photo and common 3D-mesh inputs
- Pet, vehicle, building, and general subject presets
- Configurable voxel resolution and build budget
- Automatic up-axis detection and model fitting
- Optional hollow shells and internal bracing to reduce piece count
- CIELAB colour matching against a LEGO-oriented palette
- Anatomical feature placement for pet models
- Symmetry controls for geometry and colour
- 70+ catalogued brick/plate/tile shapes with LDraw and BrickLink IDs
- API endpoints for generation, re-decomposition, statistics, and exports
- Automated tests for core geometry, colour, export, and API behavior

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn
- **Geometry:** trimesh, NumPy, scikit-learn, rtree
- **Vision:** PyTorch, Transformers, SAM 2, GPT vision
- **Model generation:** Hugging Face Spaces and Replicate adapters
- **Viewer:** Three.js and vanilla JavaScript
- **Testing:** pytest

## Run locally

### Prerequisites

- Python 3.11 or newer
- A virtual environment is strongly recommended
- API credentials for whichever optional AI providers you enable

```bash
git clone https://github.com/7kzaincode/autotile.git
cd autotile
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file as needed:

```dotenv
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_hugging_face_token
REPLICATE_API_TOKEN=your_replicate_token
```

Start the API and viewer:

```bash
uvicorn server:app --host 0.0.0.0 --port 8765
```

Open:

- Viewer: [http://localhost:8765/viewer/](http://localhost:8765/viewer/)
- API reference: [http://localhost:8765/docs](http://localhost:8765/docs)

## CLI usage

Convert an existing mesh without invoking photo-to-3D services:

```bash
python pipeline.py --mesh path/to/model.obj --resolution 32
```

Generate a mesh from a photo before conversion:

```bash
python pipeline.py --photo path/to/photo.jpg --resolution 48 --model triposr
```

Generated JSON is written to `output/<name>.json` unless `--out` is provided.

## API surface

| Endpoint | Purpose |
| --- | --- |
| `GET /api/presets` | Return subject presets |
| `POST /api/generate` | Run the full photo pipeline |
| `POST /api/generate-from-mesh` | Convert an uploaded mesh |
| `POST /api/redecompose` | Rebuild bricks from a cached mesh |
| `POST /api/parts-list` | Export JSON, CSV, or BrickLink XML |
| `POST /api/ldraw` | Export an LDraw model |
| `POST /api/obj` | Export OBJ/MTL geometry |
| `POST /api/stats` | Calculate build statistics |

Interactive request schemas are available through FastAPI at `/docs`.

## Testing

```bash
pytest
```

The suite covers pipeline geometry, colour projection, brick decomposition, parts/export formats, pet-reference handling, upload safety, and API routes. Tests that require external AI services should be treated separately from deterministic unit tests.

## Project structure

```text
server.py                 FastAPI orchestration and public endpoints
pipeline.py               Standalone mesh/photo conversion CLI
mesh_to_voxels.py         Mesh cleanup, orientation, and voxelization
voxels_to_palette.py      Colour quantization and symmetry
voxels_to_bricks.py       Greedy brick decomposition
semantic_projection.py   SAM/GPT/k-means region projection
photo_to_mesh.py          Hugging Face and Replicate adapters
parts_list.py             CSV and BrickLink exports
ldraw_export.py           LDraw export
obj_export.py             OBJ/MTL export
viewer/                   Three.js browser interface
tests/                    Unit and API tests
```

## Status

AutoTile is a working research prototype, not a guarantee that every generated model is structurally sound or purchasable exactly as rendered. Review stability, part availability, scale, and model-provider output before treating a result as final build instructions.
