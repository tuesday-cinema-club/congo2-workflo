# ComfyUI Movie Runner

Render storyboard keyframes and short clips from a YAML script via ComfyUI (API format workflows).

## Requirements
- Python 3.10+ with `websocket-client`, `pyyaml`, `python-dotenv`
- A running ComfyUI server reachable at `COMFY_SERVER_ADDRESS` (default `127.0.0.1:8188`)
- API-format workflows exported from ComfyUI:
  - `workflow_image_api.json` (uses CLIP nodes 6/7, SaveImage 9 by default)
  - `workflow_video_api.json` (uses CLIP nodes 6/7, latent 55, SaveVideo 58 by default)
 - ComfyUI folder on disk (ignored by git) containing your models/checkpoints/inputs/outputs

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install websocket-client pyyaml python-dotenv
```
Optionally set overrides in `.env`:
```
COMFY_SERVER_ADDRESS=127.0.0.1:8188
IMAGE_WORKFLOW_PATH=workflow_image_api.json
VIDEO_WORKFLOW_PATH=workflow_video_api.json
COMFY_INPUT_DIR=ComfyUI/input  # where keyframes get copied for video input
```

### Where to place files in ComfyUI
- `ComfyUI/` (ignored by git) should be your local ComfyUI install.
- Place your API-format workflow exports in this repo root as `workflow_image_api.json` and `workflow_video_api.json` (or set paths in `.env`).
- Models/checkpoints/VAEs stay in the usual ComfyUI folders:
  - `ComfyUI/models/checkpoints/` (e.g., `z-image-turbo-fp8-aio.safetensors`, `wan2.2_ti2v_5B_fp16.safetensors`)
  - `ComfyUI/models/vae/` (e.g., `wan2.2_vae.safetensors`)
  - `ComfyUI/models/clip/` (e.g., `umt5_xxl_fp8_e4m3fn_scaled.safetensors`)
- Generated keyframes are copied to `ComfyUI/input/` (configurable via `COMFY_INPUT_DIR`) so the video workflow can read them.

## Running
Render all acts/scenes/beats from a YAML script and save outputs to `movie_output/`:
```
python movie_runner.py ^
  --script test_script.yaml ^
  --image-workflow workflow_image_api.json ^
  --video-workflow workflow_video_api.json ^
  --output-dir movie_output
```
Outputs per beat:
- PNG keyframe: `movie_output/beat_<beat_id>_keyframe.png`
- MP4 clip: `movie_output/beat_<beat_id>_clip.mp4`
- Manifest: `movie_output/render_manifest.json` (one entry per beat with status/paths)

## Script format
See `test_script.yaml` for the structure (acts → scenes → beats with shot/action/location/lighting/props/players/sfx/vfx/dialogue). The prompt builder in `movie_runner.py` constructs a Stable Diffusion–friendly string from these fields.

## Workflow node IDs
Default IDs assumed by the runner (adjust constants in `movie_runner.py` if your export differs):
- Image: positive 6, negative 7, SaveImage 9
- Video: positive 6, negative 7, latent 55 (expects an `image` input if available), SaveVideo 58

## Notes and troubleshooting
- The runner copies the generated keyframe into `COMFY_INPUT_DIR` and passes the filename to the video latent node if it exposes an `image` input. If your video workflow lacks that input, it will animate from noise and print a warning—add a LoadImage/encode path or an `image` input to node 55.
- If the script hangs before image creation, ensure your ComfyUI server is up, the node IDs match your workflow JSON, and the required Python deps are installed.
- For different workflows, update the node ID constants (top of `movie_runner.py`) or mirror the defaults in `.env`.
