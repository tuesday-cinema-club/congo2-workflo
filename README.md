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
PLAYERS_PATH=players.yaml      # shared character definitions for prompts
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
See `test_script.yaml` for the structure (acts → scenes → beats with shot/action/location/lighting/props/players/sfx/vfx/dialogue). The prompt builder in `movie_runner.py` constructs a Stable Diffusion–friendly string from these fields. If you provide a `players.yaml` mapping of names to descriptions, those shared definitions are used whenever a player appears (unless a scene/beat already supplies its own description).

## Workflow node IDs
Default IDs assumed by the runner (adjust constants in `movie_runner.py` if your export differs):
- Image (Flux2): Positive 6, Negative 7, SaveImage 9, Noise 25. Optional character ref path: LoadImage 45 → VAEEncode 44 → ReferenceLatent 43 (auto-skipped if no ref image).
- Video (Wan2.2 I2V): Positive 6, Negative 7, Latent 55 (start image from LoadImage 83), SaveVideo 58, LoadAudio 90 (silent fallback if none provided).

## Notes and troubleshooting
- The runner copies the generated keyframe into `COMFY_INPUT_DIR` and passes the filename to the video latent node if it exposes an `image` input. If your video workflow lacks that input, it will animate from noise and print a warning―add a LoadImage/encode path or an `image` input to node 55.
- If the script hangs before image creation, ensure your ComfyUI server is up, the node IDs match your workflow JSON, and the required Python deps are installed.
- For different workflows, update the node ID constants (top of `movie_runner.py`) or mirror the defaults in `.env`.

## Current setup overview

### Image workflow (`workflow_image_api.json`, Flux2)
- Models: `diffusion_models/flux2_dev_fp8mixed.safetensors`, `vae/flux2-vae.safetensors`, `text_encoders/mistral_3_small_flux2_bf16.safetensors`
- Nodes: Positive 6 / Negative 7 / Save 9 / Noise 25 / (optional) Ref Load 45 → Ref Encode 44 → Ref Cond 43
- Behavior: per-beat seed jitter; character reference is skipped automatically if no image is available.

### Video workflow (`workflow_video_api.json`, Wan 2.2 I2V)
- Models: `diffusion_models/wan2.2_ti2v_5B_fp16.safetensors`, `vae/wan2.2_vae.safetensors`, `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `text_encoders/clip_l.safetensors`, `clip_vision/llava_llama3_vision.safetensors`
- Nodes: Positive 6 / Negative 7 / Latent 55 (start image from LoadImage 83) / SaveVideo 58 / LoadAudio 90 (silent fallback if no audio provided)
- Behavior: uses the generated keyframe as the start image; audio is muxed (per-beat `sfx`, global `--audio-file`, or generated TTS if enabled).

### Repo layout (simplified)
```
root/
  movie_runner.py
  workflow_image_api.json
  workflow_video_api.json
  congo2_anime.yaml
  players.yaml                # descriptions + optional ref images
  ComfyUI/                    # your ComfyUI checkout (ignored by git)
    models/
      diffusion_models/flux2_dev_fp8mixed.safetensors
      diffusion_models/wan2.2_ti2v_5B_fp16.safetensors
      vae/flux2-vae.safetensors
      vae/wan2.2_vae.safetensors
      text_encoders/mistral_3_small_flux2_bf16.safetensors
      text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
      text_encoders/clip_l.safetensors
      clip_vision/llava_llama3_vision.safetensors
    input/   # keyframes + audio copied here automatically
    output/  # ComfyUI outputs
```
