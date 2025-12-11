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
    "noisy background, monochrome, green tint, color cast, mushy motion, warped anatomy"
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
    shot: Any
    action: str
    duration_seconds: Optional[int] = None
    sfx: Optional[str] = None
    vfx: Optional[str] = None
    dialogue: Optional[str] = None
    style: Optional[str] = None
    lighting: Optional[str] = None
    location: Optional[str] = None
    theme: Optional[str] = None
    props: Optional[List[str]] = None
    players: Optional[List[str]] = None


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
    # Optional shared player definitions (lowercased name -> description)
    player_definitions: Dict[str, str] = None


# -------------------------------------------------------------------
# YAML PARSING
# -------------------------------------------------------------------

def load_player_definitions(path: str) -> Dict[str, str]:
    """Load a shared player definition file (players.yaml).
    Expected shape:
    players:
      Karen Ross: "ruthless TraviCom executive..."
      Amy: "robotic ape with camera eye..."
    If the file is missing or malformed, return an empty mapping.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}

    players = data.get("players") if isinstance(data, dict) else None
    if players is None:
        players = data if isinstance(data, dict) else {}

    out: Dict[str, str] = {}
    for name, desc in players.items():
        key = str(name).strip().lower()
        out[key] = str(desc).strip()
    return out


def load_script_from_yaml(path: str) -> Script:
    """Load the movie script YAML into our dataclasses.
    Supports the simple test_script format and the richer congo-style format.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    def _list_of_strings(val: Any) -> List[str]:
        """Normalize any list-or-string field into a list of strings."""
        if not val:
            return []
        if isinstance(val, list):
            out: List[str] = []
            for item in val:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("id") or item.get("description")
                    out.append(str(name or item))
                else:
                    out.append(str(item))
            return out
        return [str(val)]

    title = data.get("title", "Untitled Film")
    theme = data.get("theme") or data.get("global_theme")

    acts: List[Act] = []
    for act_data in data.get("acts", []):
        act_name = act_data.get("name") or act_data.get("title") or act_data.get("id", "Untitled Act")
        scenes: List[Scene] = []
        for scene_data in act_data.get("scenes", []):
            scene_name = scene_data.get("name", "Untitled Scene")
            scene_location = scene_data.get("location", "")
            scene_style = scene_data.get("style")
            scene_lighting = scene_data.get("lighting")
            scene_props = scene_data.get("props", []) or []
            scene_players: List[Player] = [
                Player(name=p.get("name", "Unknown"), description=p.get("description", ""))
                for p in scene_data.get("players", [])
            ]

            beats: List[Beat] = []
            for beat_data in scene_data.get("beats", []):
                comp = beat_data.get("composition", {}) or {}
                # seed scene props/players/lighting from composition if scene lacks them
                if not scene_props and comp.get("props"):
                    scene_props = list(comp.get("props") or [])
                if not scene_players and comp.get("players"):
                    scene_players = [Player(name=str(p), description="") for p in comp.get("players")]
                if not scene_lighting and comp.get("lighting"):
                    scene_lighting = comp.get("lighting")
                if not scene_style and comp.get("style"):
                    scene_style = comp.get("style")
                if scene_location == "" and comp.get("location"):
                    scene_location = comp.get("location")

                shot = beat_data.get("shot", "")
                action = beat_data.get("action", "")
                beat_theme = beat_data.get("theme") or comp.get("theme")
                beat_style = beat_data.get("style") or comp.get("style")
                beat_lighting = beat_data.get("lighting") or comp.get("lighting")
                beat_location = beat_data.get("location") or comp.get("location")
                beat_props = _list_of_strings(beat_data.get("props") or comp.get("props"))
                beat_players = _list_of_strings(beat_data.get("players") or comp.get("players"))
                duration_val = beat_data.get("duration_seconds")
                try:
                    duration_val = int(duration_val) if duration_val is not None else None
                except Exception:
                    duration_val = None

                sfx_val = beat_data.get("sfx", comp.get("sfx"))
                vfx_val = beat_data.get("vfx", comp.get("vfx"))

                dialogue_val = beat_data.get("dialogue") or comp.get("dialogue")
                if isinstance(dialogue_val, list):
                    # list of {speaker, line}
                    parts = []
                    for d in dialogue_val:
                        if isinstance(d, dict):
                            speaker = d.get("speaker")
                            line = d.get("line")
                            if speaker and line:
                                parts.append(f"{speaker}: {line}")
                            elif line:
                                parts.append(str(line))
                        else:
                            parts.append(str(d))
                    dialogue_val = "; ".join(parts)

                beat = Beat(
                    id=beat_data.get("id", ""),
                    shot=shot,
                    action=action,
                    duration_seconds=duration_val,
                    sfx=sfx_val,
                    vfx=vfx_val,
                    dialogue=dialogue_val,
                    style=beat_style,
                    lighting=beat_lighting,
                    location=beat_location,
                    theme=beat_theme,
                    props=beat_props or None,
                    players=beat_players or None,
                )
                beats.append(beat)

            scene = Scene(
                name=scene_name,
                location=scene_location,
                style=scene_style,
                lighting=scene_lighting,
                props=scene_props or [],
                players=scene_players or [],
                beats=beats,
            )
            scenes.append(scene)

        acts.append(Act(name=act_name, scenes=scenes))

    return Script(
        title=title,
        theme=theme,
        acts=acts,
    )


# -------------------------------------------------------------------
# PROMPT BUILDING
# -------------------------------------------------------------------

def _render_prompt_from_dict(beat: Dict[str, Any]) -> str:
    """Turn one beat dict into a strong, SD-friendly prompt string."""
    global_style = (
        "cinematic 35mm film still, anamorphic lens, sharp subject focus, "
        "rich contrast, rain mist in air, filmic grain, detailed textures, no text"
    )
    color_guard = (
        "balanced color grade, natural skin and metal tones, red emergency strobes "
        "against cool moonlight, no green wash, no monochrome"
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
            color_guard,
            global_style,
        ]
        if x
    )

    return prompt


def build_image_prompt(
    script: Script,
    act: Act,
    scene: Scene,
    beat: Beat,
    player_defs: Optional[Dict[str, str]] = None,
) -> str:
    """Map our dataclasses to the beat-dict format and render the prompt."""
    player_defs = player_defs or script.player_definitions or {}

    if isinstance(beat.shot, dict):
        shot = dict(beat.shot)
        if "camera_movement" in shot and "camera" not in shot:
            shot["camera"] = shot.get("camera_movement")
    elif beat.shot:
        shot = {"type": beat.shot}
    else:
        shot = {}

    def describe_player(name: str, inline_desc: Optional[str]) -> str:
        """Return a consistent player string using shared definitions when available."""
        base_key = name.split("(")[0].strip().lower()
        shared = player_defs.get(name.lower()) or player_defs.get(base_key)
        desc = inline_desc or shared
        return f"{name} ({desc})" if desc else name

    players: List[str] = []
    for p in scene.players:
        players.append(describe_player(p.name, p.description))
    if beat.players:
        players.extend(describe_player(str(x), None) for x in beat.players)
    # de-duplicate while preserving order
    seen_players = set()
    players = [p for p in players if not (p in seen_players or seen_players.add(p))]

    props: List[str] = []
    if scene.props:
        props.extend(scene.props)
    if beat.props:
        props.extend(beat.props)
    seen_props = set()
    props = [p for p in props if not (p in seen_props or seen_props.add(p))]

    lighting = beat.lighting or scene.lighting
    style = beat.style or scene.style
    location = beat.location or scene.location
    theme = beat.theme or script.theme

    vfx = beat.vfx
    if isinstance(vfx, list):
        vfx = ", ".join(str(x) for x in vfx)

    sfx = beat.sfx
    if isinstance(sfx, list):
        sfx = ", ".join(str(x) for x in sfx)

    beat_dict = {
        "id": beat.id,
        "theme": theme,
        "style": style,
        "location": location,
        "lighting": lighting,
        "props": props,
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
                mtype = message.get("type")

                if mtype == "execution_error":
                    data = message.get("data", {})
                    err = data.get("error") or data
                    raise RuntimeError(f"Comfy execution error: {err}")

                if mtype == "executing":
                    data = message.get("data", {})
                    if data.get("prompt_id") == prompt_id and data.get("node") is None:
                        # node == None means the whole graph finished
                        break
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
    duration_seconds: Optional[int] = None,
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
    image_for_workflow = os.path.abspath(first_frame_path)
    if copy_frame_to_input and COMFY_INPUT_DIR:
        os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
        dest = os.path.join(COMFY_INPUT_DIR, os.path.basename(first_frame_path))
        if os.path.abspath(first_frame_path) != os.path.abspath(dest) or not os.path.isfile(dest):
            shutil.copyfile(first_frame_path, dest)
        image_for_workflow = os.path.basename(dest)  # Wan expects filenames from the input folder

    latent_node = workflow.get(VIDEO_LATENT_NODE)
    if latent_node and isinstance(latent_node.get("inputs"), dict):
        if "image" in latent_node["inputs"]:
            latent_node["inputs"]["image"] = image_for_workflow
        else:
            print("Warning: video latent node has no 'image' input; animating from noise.")
        # Adjust length based on beat duration if provided
        if duration_seconds is not None:
            # Try to get fps from the CreateVideo node (57) if present, else default
            fps = 24
            create_video_node = workflow.get("57")
            if create_video_node and isinstance(create_video_node.get("inputs"), dict):
                fps = int(create_video_node["inputs"].get("fps", fps))
            try:
                frames = max(1, int(duration_seconds) * fps)
                if "length" in latent_node["inputs"]:
                    latent_node["inputs"]["length"] = frames
            except Exception:
                pass

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
        or video_out.get("images")  # some SaveVideo variants use 'images'
        or video_out.get("output")
        or video_out.get("outputs")
        or video_out.get("files")
        or video_out.get("Filenames")
        or video_out.get("filenames")
    )
    if not file_list:
        # Persist the raw outputs for troubleshooting
        debug_path = os.path.join(output_dir, f"beat_{beat_id}_video_outputs.json")
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(video_out, f, indent=2)
            print(f"Warning: no video files recorded; dumped node output to {debug_path}")
        except Exception:
            pass
        raise RuntimeError(
            f"No video files recorded in SaveVideo outputs. Keys: {list(video_out.keys())}"
        )

    vid_info = file_list[0]
    vid_bytes = fetch_file(
        filename=vid_info.get("filename") or vid_info.get("name"),
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

    players_path = os.getenv("PLAYERS_PATH", "players.yaml")
    player_definitions = load_player_definitions(players_path)
    if player_definitions:
        print(f"Loaded {len(player_definitions)} player definitions from {players_path}")
        script.player_definitions = player_definitions

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
                prompt_text = build_image_prompt(
                    script, act, scene, beat, player_defs=player_definitions
                )

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
                        duration_seconds=beat.duration_seconds,
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
