import argparse
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import urllib.request
import urllib.parse
from websocket import create_connection
from dotenv import load_dotenv
import yaml  # pip install pyyaml

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Address of your running ComfyUI server
SERVER_ADDRESS = os.getenv("COMFY_SERVER_ADDRESS", "127.0.0.1:8188")

# Node IDs for the IMAGE workflow (from workflow_image_api.json, API format)
IMAGE_POS_PROMPT_NODE = "6"   # CLIPTextEncode (positive)
IMAGE_NEG_PROMPT_NODE = "7"   # CLIPTextEncode (negative)
IMAGE_SAVE_NODE       = "9"   # SaveImage

# Node IDs for the VIDEO workflow (from workflow_video_api.json, API format)
VIDEO_POS_PROMPT_NODE = "6"    # CLIPTextEncode (positive)
VIDEO_NEG_PROMPT_NODE = "7"    # CLIPTextEncode (negative)
VIDEO_LATENT_NODE     = "55"   # Wan22ImageToVideoLatent (optional image input)
VIDEO_OUTPUT_NODE     = "58"   # SaveVideo
COMFY_INPUT_DIR       = os.getenv("COMFY_INPUT_DIR", os.path.join("ComfyUI", "input"))

DEFAULT_NEGATIVE = (
    "text, caption, logo, watermark, UI, interface, jpeg artifacts, blurry, low quality, "
    "oversaturated, extra limbs, deformed hands, distorted faces, duplicate heads, low detail, "
    "noisy background"
)


# -------------------------------------------------------------------
# DATA MODELS FOR THE SCRIPT
# -------------------------------------------------------------------

@dataclass
class Player:
    name: str
    description: str


@dataclass
class Beat:
    id: str
    shot: str
    action: str
    sfx: Optional[str] = None
    vfx: Optional[str] = None
    dialogue: Optional[str] = None


@dataclass
class Scene:
    name: str
    location: str
    style: Optional[str]
    lighting: Optional[str]
    props: List[str]
    players: List[Player]
    beats: List[Beat]


@dataclass
class Act:
    name: str
    scenes: List[Scene]


@dataclass
class Script:
    title: str
    theme: Optional[str]
    acts: List[Act]


# -------------------------------------------------------------------
# YAML PARSING
# -------------------------------------------------------------------

def load_script_from_yaml(path: str) -> Script:
    """Load the movie script YAML into our dataclasses."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    acts: List[Act] = []
    for act_data in data.get("acts", []):
        scenes: List[Scene] = []
        for scene_data in act_data.get("scenes", []):
            players = [
                Player(
                    name=p.get("name", "Unknown"),
                    description=p.get("description", "")
                )
                for p in scene_data.get("players", [])
            ]

            beats = [
                Beat(
                    id=b.get("id", ""),
                    shot=b.get("shot", ""),
                    action=b.get("action", ""),
                    sfx=b.get("sfx"),
                    vfx=b.get("vfx"),
                    dialogue=b.get("dialogue"),
                )
                for b in scene_data.get("beats", [])
            ]

            scene = Scene(
                name=scene_data.get("name", "Untitled Scene"),
                location=scene_data.get("location", ""),
                style=scene_data.get("style"),
                lighting=scene_data.get("lighting"),
                props=scene_data.get("props", []) or [],
                players=players,
                beats=beats,
            )
            scenes.append(scene)

        acts.append(Act(name=act_data.get("name", "Untitled Act"), scenes=scenes))

    return Script(
        title=data.get("title", "Untitled Film"),
        theme=data.get("theme"),
        acts=acts,
    )


# -------------------------------------------------------------------
# PROMPT BUILDING
# -------------------------------------------------------------------

def _render_prompt_from_dict(beat: Dict[str, Any]) -> str:
    """Turn one beat dict into a strong, SD-friendly prompt string."""
    global_style = (
        "black-and-green VHS jungle cyberpunk, 1990s sci-fi thriller, "
        "35mm film grain, anamorphic lens, cinematic composition, "
        "high contrast, subtle color fringing, no text"
    )

    shot_bits: List[str] = []
    shot = beat.get("shot") or {}
    if shot.get("type"):
        shot_bits.append(f"{shot['type']} shot")
    if shot.get("framing"):
        shot_bits.append(shot["framing"])
    if shot.get("camera"):
        shot_bits.append(f"camera is {shot['camera']}")
    shot_desc = ", ".join(shot_bits)

    location = beat.get("location") or ""
    action = beat.get("action") or ""
    players = ", ".join(beat.get("players", []))
    props = ", ".join(beat.get("props", []))

    situation_parts: List[str] = []
    if players:
        situation_parts.append(players)
    if action:
        situation_parts.append(action)
    if location:
        situation_parts.append(f"location: {location}")
    if props:
        situation_parts.append(f"important props: {props}")
    situation_desc = ", ".join(situation_parts)

    mood_bits: List[str] = []
    if beat.get("theme"):
        mood_bits.append(beat["theme"])
    if beat.get("style"):
        mood_bits.append(beat["style"])
    if beat.get("lighting"):
        mood_bits.append(f"lighting: {beat['lighting']}")
    if beat.get("vfx"):
        mood_bits.append(f"visual effects: {beat['vfx']}")
    mood_desc = ", ".join(mood_bits)

    prompt = ", ".join(
        x
        for x in [
            situation_desc,
            shot_desc,
            mood_desc,
            global_style,
        ]
        if x
    )

    return prompt


def build_image_prompt(script: Script, act: Act, scene: Scene, beat: Beat) -> str:
    """Map our dataclasses to the beat-dict format and render the prompt."""
    if isinstance(beat.shot, dict):
        shot = dict(beat.shot)
        if "camera_movement" in shot and "camera" not in shot:
            shot["camera"] = shot.get("camera_movement")
    elif beat.shot:
        shot = {"type": beat.shot}
    else:
        shot = {}

    players = []
    for p in scene.players:
        if p.description:
            players.append(f"{p.name} ({p.description})")
        else:
            players.append(p.name)

    vfx = beat.vfx
    if isinstance(vfx, list):
        vfx = ", ".join(str(x) for x in vfx)

    sfx = beat.sfx
    if isinstance(sfx, list):
        sfx = ", ".join(str(x) for x in sfx)

    beat_dict = {
        "id": beat.id,
        "theme": script.theme,
        "style": scene.style,
        "location": scene.location,
        "lighting": scene.lighting,
        "props": scene.props or [],
        "players": players,
        "shot": shot,
        "action": beat.action,
        "sfx": sfx,
        "vfx": vfx,
        "dialogue": beat.dialogue,
    }

    return _render_prompt_from_dict(beat_dict)


# -------------------------------------------------------------------
# COMFYUI API HELPERS
# -------------------------------------------------------------------

def _http_post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def queue_prompt(prompt_workflow: Dict[str, Any], client_id: str) -> str:
    """Send a workflow (API-format JSON) to ComfyUI and return the prompt_id."""
    payload = {"prompt": prompt_workflow, "client_id": client_id}
    result = _http_post_json(f"http://{SERVER_ADDRESS}/prompt", payload)
    return result["prompt_id"]


def wait_for_prompt(prompt_id: str, client_id: str) -> None:
    """Connect to the websocket and block until the prompt is finished executing."""
    ws = create_connection(f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}")
    try:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message.get("type") == "executing":
                    data = message.get("data", {})
                    if data.get("prompt_id") == prompt_id and data.get("node") is None:
                        break  # node == None means the whole graph finished
    finally:
        ws.close()


def fetch_history(prompt_id: str) -> Dict[str, Any]:
    url = f"http://{SERVER_ADDRESS}/history/{prompt_id}"
    with urllib.request.urlopen(url) as resp:
        all_hist = json.loads(resp.read())
    return all_hist[prompt_id]


def fetch_file(filename: str, subfolder: str, folder_type: str) -> bytes:
    """Download a file (image or video) from ComfyUI's /view endpoint."""
    params = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": folder_type}
    )
    url = f"http://{SERVER_ADDRESS}/view?{params}"
    with urllib.request.urlopen(url) as resp:
        return resp.read()


# -------------------------------------------------------------------
# IMAGE GENERATION FROM ONE BEAT
# -------------------------------------------------------------------

def run_image_workflow(
    workflow_path: str,
    prompt_text: str,
    output_dir: str,
    beat_id: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
) -> str:
    """Run the image workflow once for a beat and save the PNG to output_dir."""
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    if not isinstance(workflow, dict) or "nodes" in workflow:
        raise RuntimeError(
            f"{workflow_path} does not look like an API-format workflow. "
            f"Make sure you used 'Save (API format)' in ComfyUI."
        )

    if IMAGE_POS_PROMPT_NODE not in workflow:
        raise KeyError(
            f"Positive prompt node id {IMAGE_POS_PROMPT_NODE} not found in workflow."
        )

    workflow[IMAGE_POS_PROMPT_NODE]["inputs"]["text"] = prompt_text
    if IMAGE_NEG_PROMPT_NODE in workflow:
        workflow[IMAGE_NEG_PROMPT_NODE]["inputs"]["text"] = negative_prompt

    client_id = str(uuid.uuid4())

    print(f"\n=== Running IMAGE for beat {beat_id} ===")
    print(f"Prompt:\n{prompt_text}\n")

    prompt_id = queue_prompt(workflow, client_id)
    start_time = time.time()
    wait_for_prompt(prompt_id, client_id)
    elapsed = time.time() - start_time

    print(f"Image generation finished in {elapsed:.1f} seconds. Fetching output...")

    history = fetch_history(prompt_id)
    outputs = history["outputs"]
    if IMAGE_SAVE_NODE not in outputs or "images" not in outputs[IMAGE_SAVE_NODE]:
        raise RuntimeError(
            f"No images found in outputs of node {IMAGE_SAVE_NODE}. "
            f"Check that this is your SaveImage node."
        )

    img_info = outputs[IMAGE_SAVE_NODE]["images"][0]
    img_bytes = fetch_file(
        filename=img_info["filename"],
        subfolder=img_info["subfolder"],
        folder_type=img_info["type"],
    )

    os.makedirs(output_dir, exist_ok=True)
    local_name = f"beat_{beat_id}_keyframe.png"
    local_path = os.path.join(output_dir, local_name)

    with open(local_path, "wb") as f:
        f.write(img_bytes)

    print(f"Saved keyframe to {local_path}")
    return local_path


# -------------------------------------------------------------------
# VIDEO GENERATION FROM ONE BEAT (animate the keyframe)
# -------------------------------------------------------------------

def run_video_workflow(
    workflow_path: str,
    prompt_text: str,
    first_frame_path: str,
    output_dir: str,
    beat_id: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    copy_frame_to_input: bool = True,
) -> str:
    """Run the video workflow once for a beat and save the MP4 to output_dir."""
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    if not isinstance(workflow, dict) or "nodes" in workflow:
        raise RuntimeError(
            f"{workflow_path} does not look like an API-format workflow. "
            f"Make sure you used 'Save (API format)' in ComfyUI."
        )

    if VIDEO_POS_PROMPT_NODE not in workflow:
        raise KeyError(
            f"Video positive prompt node id {VIDEO_POS_PROMPT_NODE} not found in workflow."
        )

    workflow[VIDEO_POS_PROMPT_NODE]["inputs"]["text"] = prompt_text
    if VIDEO_NEG_PROMPT_NODE in workflow:
        workflow[VIDEO_NEG_PROMPT_NODE]["inputs"]["text"] = negative_prompt

    # Ensure the first frame is available where Comfy expects it
    image_for_workflow = first_frame_path
    if copy_frame_to_input and COMFY_INPUT_DIR:
        os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
        dest = os.path.join(COMFY_INPUT_DIR, os.path.basename(first_frame_path))
        if os.path.abspath(first_frame_path) != os.path.abspath(dest):
            shutil.copyfile(first_frame_path, dest)
        image_for_workflow = os.path.basename(dest)

    latent_node = workflow.get(VIDEO_LATENT_NODE)
    if latent_node and isinstance(latent_node.get("inputs"), dict):
        if "image" in latent_node["inputs"]:
            latent_node["inputs"]["image"] = image_for_workflow
        else:
            print("Warning: video latent node has no 'image' input; animating from noise.")

    client_id = str(uuid.uuid4())

    print(f"\n=== Running VIDEO for beat {beat_id} ===")
    print(f"Prompt:\n{prompt_text}\n")

    prompt_id = queue_prompt(workflow, client_id)
    start_time = time.time()
    wait_for_prompt(prompt_id, client_id)
    elapsed = time.time() - start_time

    print(f"Video generation finished in {elapsed:.1f} seconds. Fetching output...")

    history = fetch_history(prompt_id)
    outputs = history["outputs"]
    if VIDEO_OUTPUT_NODE not in outputs:
        raise RuntimeError(
            f"No outputs found for video save node {VIDEO_OUTPUT_NODE}. "
            f"Check that this is your SaveVideo node."
        )

    video_out = outputs[VIDEO_OUTPUT_NODE]
    file_list = (
        video_out.get("videos")
        or video_out.get("video")
        or video_out.get("Filenames")
        or video_out.get("filenames")
    )
    if not file_list:
        raise RuntimeError("No video files recorded in SaveVideo outputs.")

    vid_info = file_list[0]
    vid_bytes = fetch_file(
        filename=vid_info["filename"],
        subfolder=vid_info.get("subfolder", ""),
        folder_type=vid_info.get("type", "output"),
    )

    os.makedirs(output_dir, exist_ok=True)
    local_name = f"beat_{beat_id}_clip.mp4"
    local_path = os.path.join(output_dir, local_name)
    with open(local_path, "wb") as f:
        f.write(vid_bytes)

    print(f"Saved clip to {local_path}")
    return local_path


# -------------------------------------------------------------------
# MAIN: TEST ONE BEAT END-TO-END (image + video)
# -------------------------------------------------------------------

def main():
    load_dotenv()  # optional .env support

    parser = argparse.ArgumentParser(description="ComfyUI movie runner (image + video test)")
    parser.add_argument(
        "--script",
        required=True,
        help="Path to YAML script file (acts/scenes/beats).",
    )
    parser.add_argument(
        "--image-workflow",
        default="workflow_image_api.json",
        help="API-format JSON for the image workflow.",
    )
    parser.add_argument(
        "--video-workflow",
        default="workflow_video_api.json",
        help="API-format JSON for the video workflow.",
    )
    parser.add_argument(
        "--output-dir",
        default="movie_output",
        help="Directory where keyframes and videos will be stored.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Where to write a JSON manifest of rendered beats (default: <output-dir>/render_manifest.json).",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.script):
        print(f"Script file not found: {args.script}")
        sys.exit(1)

    if not os.path.isfile(args.image_workflow):
        print(f"Image workflow JSON not found: {args.image_workflow}")
        sys.exit(1)

    if not os.path.isfile(args.video_workflow):
        print(f"Video workflow JSON not found: {args.video_workflow}")
        sys.exit(1)

    try:
        script = load_script_from_yaml(args.script)
    except Exception as e:
        print(f"Failed to load YAML script: {e}")
        sys.exit(1)

    if not script.acts or not script.acts[0].scenes or not script.acts[0].scenes[0].beats:
        print("Script has no acts/scenes/beats to process.")
        sys.exit(1)

    first_act = script.acts[0]
    first_scene = first_act.scenes[0]
    first_beat = first_scene.beats[0]

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.output_dir, "render_manifest.json")
    manifest: List[Dict[str, Any]] = []

    for act in script.acts:
        print(f"\n=== Act: {act.name} ===")
        for scene in act.scenes:
            print(f"\n--- Scene: {scene.name} ---")
            for beat in scene.beats:
                print(f"\nBeat: {beat.id}")
                prompt_text = build_image_prompt(script, act, scene, beat)

                try:
                    frame_path = run_image_workflow(
                        workflow_path=args.image_workflow,
                        prompt_text=prompt_text,
                        output_dir=args.output_dir,
                        beat_id=beat.id or "beat",
                    )
                except Exception as e:
                    print(f"❌ Error during image generation for {beat.id}: {e}")
                    manifest.append(
                        {
                            "beat": beat.id,
                            "act": act.name,
                            "scene": scene.name,
                            "status": "image_failed",
                            "error": str(e),
                        }
                    )
                    continue

                try:
                    clip_path = run_video_workflow(
                        workflow_path=args.video_workflow,
                        prompt_text=prompt_text,
                        first_frame_path=frame_path,
                        output_dir=args.output_dir,
                        beat_id=beat.id or "beat",
                    )
                    manifest.append(
                        {
                            "beat": beat.id,
                            "act": act.name,
                            "scene": scene.name,
                            "status": "ok",
                            "frame": frame_path,
                            "clip": clip_path,
                        }
                    )
                except Exception as e:
                    print(f"❌ Error during video generation for {beat.id}: {e}")
                    manifest.append(
                        {
                            "beat": beat.id,
                            "act": act.name,
                            "scene": scene.name,
                            "status": "video_failed",
                            "frame": frame_path,
                            "error": str(e),
                        }
                    )

    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest written to {manifest_path}")
    except Exception as e:
        print(f"Warning: could not write manifest: {e}")

    print("\nAll beats processed.")


if __name__ == "__main__":
    main()
