import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
import uuid
import random
import base64
import math
import array
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import urllib.request
import urllib.parse
from websocket import create_connection
import yaml  # pip install pyyaml
from dotenv import load_dotenv

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# Load .env early so all config constants see overrides
load_dotenv()

# Address of your running ComfyUI server
SERVER_ADDRESS = os.getenv("COMFY_SERVER_ADDRESS", "127.0.0.1:8188")

# Node IDs for the IMAGE workflow (from workflow_image_api.json, API format)
IMAGE_POS_PROMPT_NODE = os.getenv("IMAGE_POSITIVE_PROMPT_NODE", "6")
IMAGE_NEG_PROMPT_NODE = os.getenv("IMAGE_NEGATIVE_PROMPT_NODE", "7")
IMAGE_SAVE_NODE = os.getenv("IMAGE_SAVE_NODE", "9")
IMAGE_NOISE_NODE = os.getenv("IMAGE_NOISE_NODE", "25")  # RandomNoise / seedable node
IMAGE_REF_LOAD_NODE = os.getenv("IMAGE_REF_LOAD_NODE", "45")  # LoadImage for character ref
IMAGE_REF_ENCODE_NODE = os.getenv("IMAGE_REF_ENCODE_NODE", "44")  # VAEEncode for ref
IMAGE_REF_COND_NODE = os.getenv("IMAGE_REF_COND_NODE", "43")  # ReferenceLatent

# Node IDs for the VIDEO workflow (from workflow_video_api.json, API format)
# Updated for Hunyuan I2V workflow defaults:
VIDEO_POS_PROMPT_NODE = (
    os.getenv("VIDEO_POSITIVE_PROMPT_NODE")
    or os.getenv("VIDEO_POS_NODE")
    or "6"
)  # Wan CLIPTextEncode (positive)
VIDEO_NEG_PROMPT_NODE = (
    os.getenv("VIDEO_NEGATIVE_PROMPT_NODE")
    or os.getenv("VIDEO_NEG_NODE")
    or "7"
)  # Wan CLIPTextEncode (negative)
VIDEO_LATENT_NODE = os.getenv("VIDEO_LATENT_NODE", "55")  # Wan22ImageToVideoLatent (takes start_image)
VIDEO_LOADIMAGE_NODE = os.getenv("VIDEO_LOADIMAGE_NODE", "83")  # LoadImage feeding the latent
VIDEO_OUTPUT_NODE = os.getenv("VIDEO_OUTPUT_NODE", "58")  # SaveVideo/SaveWEBM
VIDEO_AUDIO_NODE = os.getenv("VIDEO_AUDIO_NODE", "90")  # LoadAudio feeding CreateVideo (optional)
# How strongly the vision encoder is interleaved into the text encoder (Hunyuan).
# Higher values mean more weight on text vs image. Lower values avoid overlong token sequences.
VIDEO_IMAGE_INTERLEAVE = int(os.getenv("VIDEO_IMAGE_INTERLEAVE", "1"))
# Optional sampler/noise node IDs used for seeding variability
VIDEO_SAMPLER_NODE = os.getenv("VIDEO_SAMPLER_NODE", "3")  # KSampler
VIDEO_NOISE_NODE = os.getenv("VIDEO_NOISE_NODE", "")  # not used in Wan flow
COMFY_INPUT_DIR = os.getenv("COMFY_INPUT_DIR", os.path.join("ComfyUI", "input"))

DEFAULT_NEGATIVE = (
    "text, caption, logo, watermark, UI, interface, jpeg artifacts, blurry, low quality, "
    "oversaturated, extra limbs, deformed hands, distorted faces, duplicate heads, low detail, "
    "noisy background, monochrome, green tint, color cast, mushy motion, warped anatomy"
)


def extract_model_names(workflow_path: str) -> str:
    """Extract model/ckpt names from a Comfy API workflow for logging."""
    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = []
        for node in data.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            for key, val in inputs.items():
                if isinstance(val, str) and any(
                    tok in key.lower() for tok in ("unet", "clip", "vae", "model", "ckpt")
                ):
                    names.append(f"{key}={val}")
        deduped = list(dict.fromkeys(names))
        return "; ".join(deduped) if deduped else "unknown"
    except Exception:
        return "unknown"


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
    audio_prompt: Optional[str] = None
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

def load_player_definitions(path: str) -> (Dict[str, str], Dict[str, str]):
    """Load shared player definitions and optional reference images (players.yaml).
    Supported shapes:
    players:
      Karen Ross: "ruthless TraviCom executive..."
      Amy:
        description: "robotic ape with camera eye..."
        image: "refs/amy.png"
    Returns (descriptions, images) both keyed by lowercase name.
    """
    if not os.path.isfile(path):
        return {}, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}, {}

    players = data.get("players") if isinstance(data, dict) else None
    if players is None:
        players = data if isinstance(data, dict) else {}

    descs: Dict[str, str] = {}
    images: Dict[str, str] = {}
    for name, val in players.items():
        key = str(name).strip().lower()
        if isinstance(val, dict):
            desc = val.get("description") or val.get("desc") or ""
            img = val.get("image")
            descs[key] = str(desc).strip()
            if img:
                images[key] = str(img).strip()
        else:
            descs[key] = str(val).strip()
    return descs, images


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
                beat_audio_prompt = beat_data.get("audio_prompt") or comp.get("audio_prompt")
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
                    audio_prompt=beat_audio_prompt,
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
        "cinematic anime film frame, painterly detail, sharp subject focus, "
        "depth layering, volumetric light shafts, clean line art, subtle film grain, "
        "no text, hard-light German expressionist black-and-white overlayer: stark shadows, "
        "high contrast silhouettes, dramatic lighting shapes"
    )
    color_guard = (
        "balanced color grade, natural skin and metal tones, controlled highlights, "
        "no over-saturation, no green wash, no monochrome"
    )
    motion_guard = "no motion blur, no ghosting, coherent limbs and faces, grounded perspective"

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
            motion_guard,
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
    # Prefer beat-specific players to avoid duplicating scene players/descriptions
    if beat.players:
        players.extend(describe_player(str(x), None) for x in beat.players)
    else:
        for p in scene.players:
            players.append(describe_player(p.name, p.description))
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

    # Deterministic per-beat style/shot modifiers to avoid repetitive frames
    def _variation_tags(beat_id: str) -> str:
        rng = random.Random(f"beat-var-{beat_id}")
        camera_angles = [
            "low-angle dramatic",
            "high-angle lookout",
            "Dutch tilt",
            "over-the-shoulder",
            "bird's-eye view",
            "close-up push in",
            "medium two-shot",
            "wide establishing",
            "profile tracking",
            "rear follow cam",
        ]
        lenses = [
            "24mm wide lens",
            "28mm wide lens",
            "35mm lens",
            "50mm portrait lens",
            "70mm telephoto lens",
        ]
        movements = [
            "slow dolly left-to-right",
            "slow dolly right-to-left",
            "crane down",
            "crane up reveal",
            "steadicam walk-through",
            "handheld jitter",
            "locked-off tripod",
        ]
        palettes = [
            "golden hour warmth",
            "blue hour cool",
            "neon rim light",
            "hard noir contrast",
            "soft overcast",
        ]
        time_of_day = [
            "night with practicals",
            "dawn mist",
            "midday harsh sun",
            "late afternoon glow",
            "rainy evening streets",
        ]
        picks = [
            rng.choice(camera_angles),
            rng.choice(lenses),
            rng.choice(movements),
            rng.choice(palettes),
            rng.choice(time_of_day),
        ]
        return ", ".join(picks)

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

    base_prompt = _render_prompt_from_dict(beat_dict)
    variation = _variation_tags(beat.id or act.name)
    return f"{base_prompt}, distinct composition: {variation}"


def build_audio_prompt(
    script: Script,
    act: Act,
    scene: Scene,
    beat: Beat,
) -> str:
    """Generate an audio-focused prompt for ambience/foley."""
    parts: List[str] = []
    if beat.audio_prompt:
        parts.append(beat.audio_prompt)
    else:
        if scene.location or beat.location:
            parts.append(f"environment: {beat.location or scene.location}")
        if beat.action:
            parts.append(f"foreground action sounds: {beat.action}")
        if beat.props:
            parts.append(f"props: {', '.join(beat.props)}")
        if beat.vfx:
            parts.append(f"ambient effects: {beat.vfx}")
        if beat.sfx:
            parts.append(f"intended sfx: {beat.sfx}")
        if beat.dialogue:
            parts.append("include subtle presence for dialogue space, no voice")
        if scene.lighting or beat.lighting:
            parts.append(f"mood: {beat.lighting or scene.lighting}")
    parts.append("no music, natural foley, cinematic spatial mix, balanced levels, avoid clipping")
    return "; ".join(p for p in parts if p)


# -------------------------------------------------------------------
# COMFYUI API HELPERS
# -------------------------------------------------------------------

def ensure_comfy_available(address: str, timeout: float = 3.0) -> None:
    """Fail fast with a clear message if ComfyUI is not reachable."""
    url = f"http://{address}/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            resp.read(64)
    except Exception as exc:
        raise ConnectionError(
            f"Could not reach ComfyUI at {address}. "
            f"Start the server (e.g., `python ComfyUI/main.py --listen --port {address.split(':')[-1]}`) "
            f"or set COMFY_SERVER_ADDRESS to the correct host:port. "
            f"Original error: {exc}"
        ) from exc


def synthesize_audio(text: str, out_path: str, voice: Optional[str] = None, rate: int = 180) -> bool:
    """
    Simple TTS using pyttsx3 (offline, Windows SAPI-friendly).
    Returns True on success, False otherwise.
    """
    try:
        import pyttsx3  # type: ignore
    except Exception:
        return False


def ensure_placeholder_image(path: str) -> str:
    """Create a tiny valid PNG placeholder if it doesn't exist."""
    if os.path.isfile(path):
        return path
    png_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAucB9W4ybdkAAAAASUVORK5CYII="
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_bytes)
    return path


def ensure_silence_wav(path: str, seconds: float = 1.0, sample_rate: int = 44100) -> str:
    """Create a small silent WAV if missing to satisfy audio inputs."""
    if os.path.isfile(path):
        return path
    import wave
    import struct

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n_frames = int(sample_rate * seconds)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        silence_frame = struct.pack("<h", 0)
        wf.writeframes(silence_frame * n_frames)
    return path


def synthesize_ambience(prompt: str, out_path: str, duration: float = 6.0, sample_rate: int = 44100) -> bool:
    """
    Lightweight procedural ambience generator (wind/hum/ticks) using stdlib only.
    Not studio-grade, but better than silence for background fill.
    """
    try:
        rng = random.Random(f"amb-{prompt}")
        total = int(duration * sample_rate)
        arr = array.array("h")

        # base parameters
        hum_freq = rng.uniform(50.0, 180.0)
        hum_amp = rng.uniform(0.05, 0.12)
        wind_amp = rng.uniform(0.12, 0.2)
        tick_interval = rng.randint(sample_rate // 2, sample_rate * 2)
        tick_amp = rng.uniform(0.1, 0.2)

        hum_phase = 0.0
        wind = 0.0

        for i in range(total):
            # simple low-passed noise for wind
            n = (rng.random() - 0.5) * 2.0
            wind = wind * 0.985 + n * 0.015

            # hum sine
            hum_phase += 2.0 * math.pi * hum_freq / sample_rate
            hum = math.sin(hum_phase)

            # occasional ticks/pops
            tick = 0.0
            if tick_interval and (i % tick_interval == 0):
                tick = tick_amp * (rng.random() - 0.5) * 2.0
                tick_interval = rng.randint(sample_rate // 2, sample_rate * 2)

            sample = wind * wind_amp + hum * hum_amp + tick
            sample = max(-1.0, min(1.0, sample))
            arr.append(int(sample * 32767))

        with wave.open(out_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(arr.tobytes())
        return os.path.isfile(out_path)
    except Exception:
        return False


def run_hunyuan_foley(
    clip_path: str,
    prompt_text: str,
    repo_path: str,
    model_path: str,
    config_path: str,
    output_dir: str,
    num_steps: int = 10,
    ffmpeg_path: Optional[str] = "ffmpeg",
    enable_offload: bool = False,
) -> Optional[str]:
    """
    Call HunyuanVideo-Foley's infer.py to generate a WAV, then mux it into the clip with ffmpeg.
    Returns new clip path on success, or None on failure.
    """
    clip_path = os.path.abspath(clip_path)
    output_dir = os.path.abspath(output_dir)
    repo_path = os.path.abspath(repo_path)
    model_path = os.path.abspath(model_path)
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(clip_path):
        print(f"Warning: Foley clip not found on disk: {clip_path}")
        return None
    try:
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            sys.executable,
            os.path.join(repo_path, "infer.py"),
            "--model_path",
            model_path,
            "--config_path",
            config_path,
            "--single_video",
            clip_path,
            "--single_prompt",
            prompt_text,
            "--num_inference_steps",
            str(num_steps),
            "--output_dir",
            output_dir,
        ]
        if enable_offload:
            cmd.append("--enable_offload")
        subprocess.run(cmd, check=True, cwd=repo_path)
    except Exception as exc:
        print(f"Warning: Foley generation failed for {clip_path}: {exc}")
        return None

    # pick most recent wav in output_dir
    wav_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print("Warning: Foley generation produced no wav files.")
        return None
    wav_path = max(wav_files, key=os.path.getmtime)

    # mux with ffmpeg
    try:
        new_path = os.path.splitext(clip_path)[0] + "_with_foley.mp4"
        cmd = [
            ffmpeg_path or "ffmpeg",
            "-y",
            "-i",
            clip_path,
            "-i",
            wav_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            new_path,
        ]
        subprocess.run(cmd, check=True)
        return new_path
    except Exception as exc:
        print(f"Warning: ffmpeg mux failed: {exc}")
        return None
    try:
        engine = pyttsx3.init()
        if voice:
            engine.setProperty("voice", voice)
        engine.setProperty("rate", rate)
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return os.path.isfile(out_path)
    except Exception:
        return False


def _http_post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        raise ConnectionError(f"Failed to POST to {url}. Is ComfyUI running at {SERVER_ADDRESS}? {exc}") from exc


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
    character_image: Optional[str] = None,
) -> str:
    """Run the image workflow once for a beat and save the PNG to output_dir."""
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    if not isinstance(workflow, dict) or "nodes" in workflow:
        raise RuntimeError(
            f"{workflow_path} does not look like an API-format workflow. "
            f"Make sure you used 'Save (API format)' in ComfyUI."
        )

    pos_node_id = IMAGE_POS_PROMPT_NODE if IMAGE_POS_PROMPT_NODE in workflow else None
    if pos_node_id is None:
        raise KeyError(
            f"Positive prompt node id {IMAGE_POS_PROMPT_NODE} not found in workflow."
        )

    workflow[pos_node_id]["inputs"]["text"] = prompt_text
    neg_node_id = IMAGE_NEG_PROMPT_NODE if IMAGE_NEG_PROMPT_NODE in workflow else None
    if neg_node_id:
        workflow[neg_node_id]["inputs"]["text"] = negative_prompt

    # Jitter seeds per beat so compositions vary
    seed_val = abs(hash(f"img-{beat_id}-{time.time()}")) % 2**31
    # Sampler seed (if present)
    sampler_node = workflow.get("3")
    if sampler_node and isinstance(sampler_node.get("inputs"), dict) and "seed" in sampler_node["inputs"]:
        sampler_node["inputs"]["seed"] = seed_val
    # RandomNoise / noise node
    noise_node = workflow.get(IMAGE_NOISE_NODE)
    if noise_node and isinstance(noise_node.get("inputs"), dict):
        key = "noise_seed" if "noise_seed" in noise_node["inputs"] else "seed"
        if key in noise_node["inputs"]:
            noise_node["inputs"][key] = seed_val

    # Wire a character reference image; otherwise fall back to plain conditioning
    def disable_ref_nodes():
        guider = workflow.get("22")
        if guider and isinstance(guider.get("inputs"), dict):
            guider["inputs"]["conditioning"] = ["26", 0]
        # Remove ref nodes to avoid validation errors
        for nid in (IMAGE_REF_COND_NODE, IMAGE_REF_ENCODE_NODE, IMAGE_REF_LOAD_NODE):
            workflow.pop(nid, None)

    if character_image and os.path.isfile(character_image):
        try:
            if COMFY_INPUT_DIR:
                os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
                dest = os.path.join(COMFY_INPUT_DIR, os.path.basename(character_image))
                if os.path.abspath(character_image) != os.path.abspath(dest) or not os.path.isfile(dest):
                    shutil.copyfile(character_image, dest)
                image_for_workflow = os.path.basename(dest)
            else:
                image_for_workflow = character_image
            load_ref = workflow.get(IMAGE_REF_LOAD_NODE)
            if load_ref and isinstance(load_ref.get("inputs"), dict):
                load_ref["inputs"]["image"] = image_for_workflow
        except Exception:
            disable_ref_nodes()
    else:
        disable_ref_nodes()

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
    audio_path: Optional[str] = None,
) -> str:
    """Run the video workflow once for a beat and save the MP4 to output_dir."""
    # Hunyuan text encoder is sensitive to long prompts; truncate by clauses then by length
    def _shorten_prompt(text: str, max_chars: int = 100, max_parts: int = 3) -> str:
        flat = " ".join(text.split())
        parts = [p.strip() for p in flat.split(",") if p.strip()]
        out_parts = []
        total = 0
        for p in parts[:max_parts]:
            if total + len(p) + 2 > max_chars:
                break
            out_parts.append(p)
            total += len(p) + 2
        short = ", ".join(out_parts) if out_parts else flat[:max_chars]
        return short[:max_chars]

    safe_prompt = _shorten_prompt(prompt_text)

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    if not isinstance(workflow, dict) or "nodes" in workflow:
        raise RuntimeError(
            f"{workflow_path} does not look like an API-format workflow. "
            f"Make sure you used 'Save (API format)' in ComfyUI."
        )

    pos_node_id = VIDEO_POS_PROMPT_NODE if VIDEO_POS_PROMPT_NODE in workflow else ("80" if "80" in workflow else None)
    if pos_node_id is None:
        raise KeyError(
            f"Video positive prompt node id {VIDEO_POS_PROMPT_NODE} not found in workflow."
        )

    # Give each clip a fresh seed so videos don't all look identical
    seed_val = abs(hash(f"{beat_id}-{time.time()}")) % 2**31
    # KSampler seed (old workflows)
    sampler_node = workflow.get("3")
    if sampler_node and isinstance(sampler_node.get("inputs"), dict) and "seed" in sampler_node["inputs"]:
        sampler_node["inputs"]["seed"] = seed_val
    # SamplerCustomAdvanced / RandomNoise (Hunyuan workflow)
    adv_sampler = workflow.get(VIDEO_SAMPLER_NODE)
    if adv_sampler and isinstance(adv_sampler.get("inputs"), dict) and "seed" in adv_sampler["inputs"]:
        adv_sampler["inputs"]["seed"] = seed_val
    noise_node = workflow.get(VIDEO_NOISE_NODE)
    if noise_node and isinstance(noise_node.get("inputs"), dict):
        key = "noise_seed" if "noise_seed" in noise_node["inputs"] else "seed"
        if key in noise_node["inputs"]:
            noise_node["inputs"][key] = seed_val

    # Inject prompt text into the positive node; Hunyuan text encoder uses 'text' (and accepts 'prompt' as alias)
    pos_inputs = workflow[pos_node_id].setdefault("inputs", {})
    pos_inputs["text"] = safe_prompt
    # Some builds expect 'prompt' instead of 'text'
    pos_inputs["prompt"] = safe_prompt
    if "image_interleave" in pos_inputs:
        try:
            pos_inputs["image_interleave"] = max(1, int(VIDEO_IMAGE_INTERLEAVE))
        except Exception:
            pass
    if "max_new_tokens" in pos_inputs:
        try:
            # keep within a sane range to avoid over-long sequences
            pos_inputs["max_new_tokens"] = min(96, int(pos_inputs["max_new_tokens"]))
        except Exception:
            pass

    neg_node_id = VIDEO_NEG_PROMPT_NODE if (VIDEO_NEG_PROMPT_NODE and VIDEO_NEG_PROMPT_NODE in workflow) else None
    if neg_node_id:
        workflow[neg_node_id]["inputs"]["text"] = negative_prompt

    # Ensure the first frame is available where Comfy expects it
    image_for_workflow = os.path.abspath(first_frame_path)
    if copy_frame_to_input and COMFY_INPUT_DIR:
        os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
        dest = os.path.join(COMFY_INPUT_DIR, os.path.basename(first_frame_path))
        if os.path.abspath(first_frame_path) != os.path.abspath(dest) or not os.path.isfile(dest):
            shutil.copyfile(first_frame_path, dest)
        image_for_workflow = os.path.basename(dest)  # pass basename to LoadImage

    # If a LoadImage node exists (common in Hunyuan I2V), update it to point to the keyframe
    load_image_node_id = (
        VIDEO_LOADIMAGE_NODE
        if VIDEO_LOADIMAGE_NODE in workflow
        else ("83" if "83" in workflow else None)
    )
    load_image_node = workflow.get(load_image_node_id) if load_image_node_id else None
    if load_image_node and isinstance(load_image_node.get("inputs"), dict):
        load_image_node["inputs"]["image"] = image_for_workflow
        # Some LoadImage variants also use 'choose file type'
        if "choose file type" in load_image_node["inputs"]:
            load_image_node["inputs"]["choose file type"] = "image"

    latent_node_id = VIDEO_LATENT_NODE if VIDEO_LATENT_NODE in workflow else ("78" if "78" in workflow else None)
    latent_node = workflow.get(latent_node_id) if latent_node_id else None
    if latent_node and isinstance(latent_node.get("inputs"), dict):
        if "image" in latent_node["inputs"]:
            latent_node["inputs"]["image"] = image_for_workflow
        if "start_image" in latent_node["inputs"]:
            if load_image_node_id and load_image_node_id in workflow:
                latent_node["inputs"]["start_image"] = [load_image_node_id, 0]
            else:
                latent_node["inputs"]["start_image"] = image_for_workflow
        # Adjust length based on beat duration if provided
        if duration_seconds is not None:
            # Try to get fps from the CreateVideo node (57) if present, else default
            fps = 24
            create_video_node = workflow.get("57")
            if create_video_node and isinstance(create_video_node.get("inputs"), dict):
                fps = int(create_video_node["inputs"].get("fps", fps))
            try:
                frames = max(1, int(duration_seconds) * fps)
                # Keep length modest for 8GB cards
                frames = min(frames, 8 * fps)
                if "length" in latent_node["inputs"]:
                    latent_node["inputs"]["length"] = frames
            except Exception:
                pass

    # If a LoadAudio node exists, wire in provided audio or fallback silence
    audio_node_id = VIDEO_AUDIO_NODE if VIDEO_AUDIO_NODE in workflow else None
    if audio_node_id:
        try:
            if not audio_path:
                silent_dir = COMFY_INPUT_DIR or "."
                audio_path = ensure_silence_wav(os.path.join(silent_dir, "audio_silence.wav"))
            os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
            audio_dest = os.path.join(COMFY_INPUT_DIR, os.path.basename(audio_path))
            if os.path.abspath(audio_path) != os.path.abspath(audio_dest) or not os.path.isfile(audio_dest):
                shutil.copyfile(audio_path, audio_dest)
            audio_node = workflow.get(audio_node_id)
            if audio_node and isinstance(audio_node.get("inputs"), dict):
                audio_node["inputs"]["audio"] = os.path.basename(audio_dest)
        except Exception:
            pass

    client_id = str(uuid.uuid4())

    # Encourage motion/camera movement in the video prompt
    rng = random.Random(f"vid-move-{beat_id}")
    motion_tags = [
        "dynamic camera pan and tilt",
        "parallax background drift",
        "cloth and hair react to wind",
        "subject walks through frame",
        "slow orbiting camera move",
        "depth-filled tracking shot",
        "environmental effects in motion",
    ]
    move_prompt = f"{prompt_text}, animated sequence, {rng.choice(motion_tags)}"

    print(f"\n=== Running VIDEO for beat {beat_id} ===")
    print(f"Prompt:\n{move_prompt}\n")

    # Inject the motion-enhanced prompt into the graph
    pos_inputs = workflow[pos_node_id].setdefault("inputs", {})
    pos_inputs["text"] = move_prompt
    pos_inputs["prompt"] = move_prompt
    if neg_node_id:
        workflow[neg_node_id]["inputs"]["text"] = negative_prompt

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
    remote_filename = vid_info.get("filename") or vid_info.get("name")
    if not remote_filename:
        raise RuntimeError("Video output missing filename.")
    vid_bytes = fetch_file(
        filename=remote_filename,
        subfolder=vid_info.get("subfolder", ""),
        folder_type=vid_info.get("type", "output"),
    )

    os.makedirs(output_dir, exist_ok=True)
    _, ext = os.path.splitext(remote_filename)
    local_ext = ext or ".mp4"
    local_name = f"beat_{beat_id}_clip{local_ext}"
    local_path = os.path.join(output_dir, local_name)
    with open(local_path, "wb") as f:
        f.write(vid_bytes)

    print(f"Saved clip to {local_path}")
    return local_path


# -------------------------------------------------------------------
# MAIN: TEST ONE BEAT END-TO-END (image + video)
# -------------------------------------------------------------------

def main():
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip beats that already have outputs in the output directory; continue where a previous run left off.",
    )
    parser.add_argument(
        "--auto-duration",
        action="store_true",
        help="Override uniform durations by auto-jittering per beat to keep clips varied and GPU-friendly.",
    )
    parser.add_argument(
        "--audio-file",
        default=None,
        help="Optional path to an audio file to mux into every clip (or leave None to rely on per-beat sfx paths).",
    )
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Auto-generate simple narration audio per beat (pyttsx3 required). Uses beat dialogue or prompt text.",
    )
    parser.add_argument(
        "--foley-enable",
        action="store_true",
        help="Run HunyuanVideo-Foley on each rendered clip and mux the result (requires repo/model/config).",
    )
    parser.add_argument(
        "--foley-repo",
        default=None,
        help="Path to HunyuanVideo-Foley repo (where infer.py lives).",
    )
    parser.add_argument(
        "--foley-model-path",
        default=None,
        help="Path to HunyuanVideo-Foley pretrained model directory.",
    )
    parser.add_argument(
        "--foley-config",
        default=None,
        help="Path to the HunyuanVideo-Foley config YAML (e.g., configs/hunyuanvideo-foley-xxl.yaml).",
    )
    parser.add_argument(
        "--foley-steps",
        type=int,
        default=10,
        help="Denoising steps for HunyuanVideo-Foley sampling (default: 10).",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="ffmpeg binary to use for muxing foley audio (default: ffmpeg in PATH).",
    )
    parser.add_argument(
        "--foley-offload",
        action="store_true",
        help="Enable model offloading in HunyuanVideo-Foley to save VRAM/RAM.",
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

    image_models = extract_model_names(args.image_workflow)
    video_models = extract_model_names(args.video_workflow)

    try:
        ensure_comfy_available(SERVER_ADDRESS)
    except Exception as e:
        print(e)
        sys.exit(1)

    try:
        script = load_script_from_yaml(args.script)
    except Exception as e:
        print(f"Failed to load YAML script: {e}")
        sys.exit(1)

    players_path = os.getenv("PLAYERS_PATH", "players.yaml")
    player_definitions, player_images = load_player_definitions(players_path)
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
    log_path = os.path.join(args.output_dir, "run_log.txt")
    manifest: List[Dict[str, Any]] = []
    manifest_index: Dict[str, Dict[str, Any]] = {}

    # Load existing manifest if resuming
    if args.resume and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f) or []
            if isinstance(existing, list):
                for entry in existing:
                    beat_id = entry.get("beat")
                    if beat_id:
                        manifest_index[beat_id] = entry
                manifest = list(manifest_index.values())
                print(f"Resume enabled: loaded {len(manifest)} entries from manifest.")
        except Exception as e:
            print(f"Warning: could not load existing manifest for resume: {e}")

    # Helper to auto-jitter durations when requested or when all beats are identical
    def derive_duration(beat: Beat) -> Optional[int]:
        base = beat.duration_seconds
        if args.auto_duration or base is None:
            # Hash beat id to a duration between 3 and 8 seconds (GPU-safe)
            rng = random.Random(f"dur-{beat.id}")
            return int(rng.uniform(3.0, 8.0))
        return base

    def choose_character_image(beat: Beat, scene: Scene) -> Optional[str]:
        # Prefer beat-level player images, then scene player images
        def lookup(name: str) -> Optional[str]:
            key = name.split("(")[0].strip().lower()
            return player_images.get(key)

        if beat.players:
            for p in beat.players:
                img = lookup(str(p))
                if img and os.path.isfile(img):
                    return img
        for p in scene.players:
            img = lookup(p.name)
            if img and os.path.isfile(img):
                return img
        return None

    def select_audio(beat: Beat, prompt_text: str, audio_prompt: str) -> Optional[str]:
        # Prefer per-beat sfx if it points to a file; else use CLI audio_file
        sfx_path = None
        if beat.sfx and isinstance(beat.sfx, str) and os.path.isfile(beat.sfx):
            sfx_path = beat.sfx
        chosen = sfx_path or args.audio_file

        # If generation requested and no file chosen, synthesize quick narration or ambience
        if args.generate_audio and not chosen:
            text_for_audio = beat.dialogue or beat.action or audio_prompt
            audio_out = os.path.join(args.output_dir, f"beat_{beat.id}_audio.wav")
            ok = False
            if text_for_audio:
                ok = synthesize_audio(text_for_audio[:280], audio_out)
            if not ok:
                # fallback to procedural ambience using the prompt text
                ok = synthesize_ambience(audio_prompt or prompt_text, audio_out, duration=derive_duration(beat) or 6.0)
            if ok:
                chosen = audio_out
        return chosen

    for act in script.acts:
        print(f"\n=== Act: {act.name} ===")
        for scene in act.scenes:
            print(f"\n--- Scene: {scene.name} ---")
            for beat in scene.beats:
                print(f"\nBeat: {beat.id}")
                prompt_text = build_image_prompt(
                    script, act, scene, beat, player_defs=player_definitions
                )
                audio_prompt = beat.audio_prompt or build_audio_prompt(script, act, scene, beat)

                beat_start = time.perf_counter()
                image_time = 0.0
                video_time = 0.0
                foley_time = 0.0

                beat_id = beat.id or "beat"
                frame_path = os.path.join(args.output_dir, f"beat_{beat_id}_keyframe.png")
                clip_base = os.path.join(args.output_dir, f"beat_{beat_id}_clip")
                existing_clip = None
                for ext in [".mp4", ".webm", ".webp"]:
                    candidate = f"{clip_base}{ext}"
                    if os.path.isfile(candidate):
                        existing_clip = candidate
                        break
                clip_path = existing_clip or f"{clip_base}.mp4"

                # Skip if clip already exists and resume enabled
                if args.resume and existing_clip:
                    print(f"Resume: clip already exists for {beat_id}, skipping.")
                    final_clip = clip_path
                    beat_total = time.perf_counter() - beat_start
                    try:
                        log_line = f"{datetime.now().isoformat()} beat={beat_id} image_time={image_time:.2f}s video_time={video_time:.2f}s foley_time={foley_time:.2f}s total={beat_total:.2f}s image_models={image_models} video_models={video_models} foley_model={args.foley_model_path or 'none'} final_clip={final_clip}"
                        with open(log_path, "a", encoding="utf-8") as lf:
                            lf.write(log_line + "\n")
                        print(log_line)
                    except Exception:
                        pass

                    manifest_index[beat_id] = {
                        "beat": beat.id,
                        "act": act.name,
                        "scene": scene.name,
                        "status": "ok",
                        "frame": frame_path if os.path.isfile(frame_path) else None,
                        "clip": existing_clip,
                    }
                    continue

                # Use existing frame if present
                frame_ready = os.path.isfile(frame_path)
                if not frame_ready:
                    try:
                        t_img = time.perf_counter()
                        frame_path = run_image_workflow(
                            workflow_path=args.image_workflow,
                            prompt_text=prompt_text,
                            output_dir=args.output_dir,
                            beat_id=beat_id,
                            character_image=choose_character_image(beat, scene),
                        )
                        image_time = time.perf_counter() - t_img
                        frame_ready = True
                    except Exception as e:
                        print(f"❌ Error during image generation for {beat.id}: {e}")
                        manifest_index[beat_id] = {
                            "beat": beat.id,
                            "act": act.name,
                            "scene": scene.name,
                            "status": "image_failed",
                            "error": str(e),
                        }
                        continue
                else:
                    print(f"Resume: using existing frame for {beat_id}: {frame_path}")

                # Run video if needed
                try:
                    t_vid = time.perf_counter()
                    clip_path = run_video_workflow(
                        workflow_path=args.video_workflow,
                        prompt_text=prompt_text,
                        first_frame_path=frame_path,
                        output_dir=args.output_dir,
                        beat_id=beat_id,
                        duration_seconds=derive_duration(beat),
                        audio_path=select_audio(beat, prompt_text, audio_prompt),
                    )
                    video_time = time.perf_counter() - t_vid
                    # Optional: run HunyuanVideo-Foley and remux
                    if (
                        args.foley_enable
                        and args.foley_repo
                        and args.foley_model_path
                        and args.foley_config
                    ):
                        foley_out_dir = os.path.join(args.output_dir, "foley_outputs")
                        t_foley = time.perf_counter()
                        new_clip = run_hunyuan_foley(
                            clip_path=clip_path,
                            prompt_text=audio_prompt or prompt_text,
                            repo_path=args.foley_repo,
                            model_path=args.foley_model_path,
                            config_path=args.foley_config,
                            output_dir=foley_out_dir,
                            num_steps=args.foley_steps,
                            ffmpeg_path=args.ffmpeg_path,
                            enable_offload=args.foley_offload,
                        )
                        if new_clip and os.path.isfile(new_clip):
                            foley_time = time.perf_counter() - t_foley
                            clip_path = new_clip
                    final_clip = clip_path
                    beat_total = time.perf_counter() - beat_start
                    try:
                        log_line = f"{datetime.now().isoformat()} beat={beat_id} image_time={image_time:.2f}s video_time={video_time:.2f}s foley_time={foley_time:.2f}s total={beat_total:.2f}s image_models={image_models} video_models={video_models} foley_model={args.foley_model_path or 'none'} final_clip={final_clip}"
                        with open(log_path, "a", encoding="utf-8") as lf:
                            lf.write(log_line + "\\n")
                        print(log_line)
                    except Exception:
                        pass

                    manifest_index[beat_id] = {
                        "beat": beat.id,
                        "act": act.name,
                        "scene": scene.name,
                        "status": "ok",
                        "frame": frame_path if frame_ready else None,
                        "clip": clip_path,
                    }
                except Exception as e:
                    print(f"❌ Error during video generation for {beat.id}: {e}")
                    manifest_index[beat_id] = {
                        "beat": beat.id,
                        "act": act.name,
                        "scene": scene.name,
                        "status": "video_failed",
                        "frame": frame_path if frame_ready else None,
                        "error": str(e),
                    }

    # Flush manifest from index to list to avoid duplicates
    manifest = list(manifest_index.values())

    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest written to {manifest_path}")
    except Exception as e:
        print(f"Warning: could not write manifest: {e}")

    print("\nAll beats processed.")


if __name__ == "__main__":
    main()
