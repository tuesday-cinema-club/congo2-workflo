import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import urllib.request
import urllib.parse

from websocket import create_connection
from dotenv import load_dotenv
import yaml


# -----------------------------
# Data model for script parsing
# -----------------------------


@dataclass
class Composition:
    """Defines the visual/audio composition of a beat."""
    theme: str
    style: str
    props: List[str]
    location: str
    lighting: str
    players: List[str]
    sfx: List[str]
    vfx: List[str]
    dialogue: str


@dataclass
class Action:
    """A single action = one shot to render."""
    id: str
    shot_type: str
    shot_description: str
    duration_seconds: int


@dataclass
class Beat:
    """A beat in a scene; holds composition + actions."""
    id: str
    summary: str
    composition: Composition
    actions: List[Action]


@dataclass
class Scene:
    """A scene: multiple beats."""
    id: str
    name: str
    default_location: Optional[str]
    default_lighting: Optional[str]
    beats: List[Beat]


@dataclass
class Act:
    """An act: multiple scenes."""
    id: str
    name: str
    scenes: List[Scene]


@dataclass
class Film:
    """Top-level film structure."""
    title: str
    theme: str
    global_style: str
    acts: List[Act]


# -----------------------------
# Script loading & validation
# -----------------------------


def load_film_script(path: str) -> Film:
    """
    Load and validate the film YAML script.
    Raises ValueError on missing required fields.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Script file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "film" not in data:
        raise ValueError("Script must have top-level 'film' key.")

    fdata = data["film"]

    # Basic validation
    for key in ["title", "theme", "global_style", "acts"]:
        if key not in fdata:
            raise ValueError(f"Missing required film field: {key}")

    acts: List[Act] = []
    for act_data in fdata["acts"]:
        for key in ["id", "name", "scenes"]:
            if key not in act_data:
                raise ValueError(f"Act missing required field '{key}': {act_data}")
        scenes: List[Scene] = []
        for scene_data in act_data["scenes"]:
            for key in ["id", "name", "beats"]:
                if key not in scene_data:
                    raise ValueError(f"Scene missing required field '{key}': {scene_data}")

            beats: List[Beat] = []
            for beat_data in scene_data["beats"]:
                for key in ["id", "summary", "composition", "actions"]:
                    if key not in beat_data:
                        raise ValueError(f"Beat missing required field '{key}': {beat_data}")

                comp_data = beat_data["composition"]
                # Ensure all composition fields exist
                for key in [
                    "theme",
                    "style",
                    "props",
                    "location",
                    "lighting",
                    "players",
                    "sfx",
                    "vfx",
                    "dialogue",
                ]:
                    if key not in comp_data:
                        raise ValueError(
                            f"Composition missing field '{key}' in beat {beat_data['id']}"
                        )

                composition = Composition(
                    theme=comp_data["theme"],
                    style=comp_data["style"],
                    props=list(comp_data["props"] or []),
                    location=comp_data["location"],
                    lighting=comp_data["lighting"],
                    players=list(comp_data["players"] or []),
                    sfx=list(comp_data["sfx"] or []),
                    vfx=list(comp_data["vfx"] or []),
                    dialogue=str(comp_data["dialogue"]),
                )

                actions: List[Action] = []
                for action_data in beat_data["actions"]:
                    for key in ["id", "shot_type", "shot_description", "duration_seconds"]:
                        if key not in action_data:
                            raise ValueError(
                                f"Action missing field '{key}' in beat {beat_data['id']}"
                            )
                    actions.append(
                        Action(
                            id=action_data["id"],
                            shot_type=action_data["shot_type"],
                            shot_description=action_data["shot_description"],
                            duration_seconds=int(action_data["duration_seconds"]),
                        )
                    )

                beats.append(
                    Beat(
                        id=beat_data["id"],
                        summary=beat_data["summary"],
                        composition=composition,
                        actions=actions,
                    )
                )

            scenes.append(
                Scene(
                    id=scene_data["id"],
                    name=scene_data["name"],
                    default_location=scene_data.get("default_location"),
                    default_lighting=scene_data.get("default_lighting"),
                    beats=beats,
                )
            )

        acts.append(Act(id=act_data["id"], name=act_data["name"], scenes=scenes))

    return Film(
        title=fdata["title"],
        theme=fdata["theme"],
        global_style=fdata["global_style"],
        acts=acts,
    )


# -----------------------------
# Prompt construction
# -----------------------------


def build_prompt(
    film: Film,
    act: Act,
    scene: Scene,
    beat: Beat,
    action: Action,
) -> str:
    """
    Turn composition + shot info into a single text prompt for the image/video model.
    You can tune this formatting to your preferred "prompt voice".
    """
    comp = beat.composition

    props_text = ", ".join(comp.props)
    players_text = "; ".join(comp.players)
    sfx_text = ", ".join(comp.sfx)
    vfx_text = ", ".join(comp.vfx)

    prompt = f"""
{film.global_style}.
Film title: "{film.title}".
Act: {act.name}, Scene: {scene.name}, Beat: {beat.id} - {beat.summary}.

Theme of this moment: {comp.theme}.
Visual style: {comp.style}.

Location: {comp.location or scene.default_location or ""}.
Lighting: {comp.lighting or scene.default_lighting or ""}.

Props in frame: {props_text}.
Characters in frame: {players_text}.

Shot type: {action.shot_type}.
Shot description: {action.shot_description}.
Intended duration: about {action.duration_seconds} seconds of smooth motion.

Atmospheric VFX: {vfx_text}.
Implied SFX (do not show text, just mood): {sfx_text}.

Implied dialogue (no subtitles on screen): {comp.dialogue}.
Do not show any text or subtitles.
    """.strip()

    return prompt


# -----------------------------
# ComfyUI client utilities
# -----------------------------


class ComfyClient:
    """
    Small helper class to talk to a local ComfyUI server using the HTTP + WebSocket API.
    """

    def __init__(self, server_address: str):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def _queue_prompt(self, prompt_workflow: Dict[str, Any]) -> str:
        """Send workflow JSON to ComfyUI's /prompt endpoint and return prompt_id."""
        payload = {"prompt": prompt_workflow, "client_id": self.client_id}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://{self.server_address}/prompt", data=data
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        return result["prompt_id"]

    def _wait_for_completion(self, prompt_id: str) -> None:
        """Listen on WebSocket until this prompt finishes executing."""
        ws = create_connection(
            f"ws://{self.server_address}/ws?clientId={self.client_id}"
        )
        try:
            while True:
                raw = ws.recv()
                if isinstance(raw, str):
                    msg = json.loads(raw)
                    if msg.get("type") == "executing":
                        data = msg.get("data", {})
                        if data.get("node") is None and data.get("prompt_id") == prompt_id:
                            # None node + matching prompt_id = done
                            break
        finally:
            ws.close()

    def _get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Fetch the execution history for a given prompt_id."""
        url = f"http://{self.server_address}/history/{prompt_id}"
        with urllib.request.urlopen(url) as resp:
            all_hist = json.loads(resp.read())
        return all_hist[prompt_id]

    def _download_file(
        self, filename: str, subfolder: str, folder_type: str
    ) -> bytes:
        """Download an image or video file from ComfyUI."""
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        url = f"http://{self.server_address}/view?{params}"
        with urllib.request.urlopen(url) as resp:
            return resp.read()

    def run_image_workflow(
        self,
        workflow_path: str,
        positive_prompt_node: str,
        save_node: str,
        new_prompt: str,
    ) -> str:
        """
        Run the image workflow:
          - load JSON
          - replace the positive prompt text
          - queue and wait
          - download the resulting image
          - returns local filename of saved image
        """
        with open(workflow_path, "r", encoding="utf-8") as f:
            wf = json.load(f)

        if positive_prompt_node not in wf:
            raise KeyError(
                f"Positive prompt node '{positive_prompt_node}' not found in workflow JSON."
            )

        # Comfy API format usually has `inputs['text']` for CLIPTextEncode nodes
        wf[positive_prompt_node]["inputs"]["text"] = new_prompt

        prompt_id = self._queue_prompt(wf)
        print(f"[image] queued prompt_id={prompt_id}")
        self._wait_for_completion(prompt_id)
        print(f"[image] completed prompt_id={prompt_id}")

        history = self._get_history(prompt_id)

        if save_node not in history["outputs"]:
            raise KeyError(
                f"Save node '{save_node}' not found in history outputs. "
                f"Available keys: {list(history['outputs'].keys())}"
            )

        images = history["outputs"][save_node].get("images", [])
        if not images:
            raise RuntimeError("No images found in save-node outputs.")

        info = images[0]
        data = self._download_file(info["filename"], info["subfolder"], info["type"])

        # Save image next to script
        local_name = f"shot_{prompt_id}.png"
        with open(local_name, "wb") as f:
            f.write(data)

        print(f"[image] saved first frame: {local_name}")
        return local_name

    def run_video_workflow(
        self,
        workflow_path: str,
        positive_prompt_node: str,
        loadimage_node: str,
        output_node: str,
        new_prompt: str,
        image_filename: str,
    ) -> str:
        """
        Run the video workflow:
          - load JSON
          - set the positive prompt text
          - set the LoadImage node filename
          - queue and wait
          - DOES NOT download full video (usually bigger), but
            returns the filename that Comfy saved in its output folder.
        """
        with open(workflow_path, "r", encoding="utf-8") as f:
            wf = json.load(f)

        if positive_prompt_node not in wf:
            raise KeyError(
                f"Video positive prompt node '{positive_prompt_node}' not found."
            )
        if loadimage_node not in wf:
            raise KeyError(f"Video LoadImage node '{loadimage_node}' not found.")
        if output_node not in wf:
            raise KeyError(f"Video output node '{output_node}' not found.")

        # Update text prompt
        wf[positive_prompt_node]["inputs"]["text"] = new_prompt

        # Update image filename for the LoadImage node.
        # NOTE: Comfy expects just the filename that is placed in ComfyUI/input.
        # You can either copy the image there manually OR set your workflow to use
        # an absolute path. Adjust this depending on how you configure the node.
        wf[loadimage_node]["inputs"]["image"] = image_filename

        prompt_id = self._queue_prompt(wf)
        print(f"[video] queued prompt_id={prompt_id}")
        self._wait_for_completion(prompt_id)
        print(f"[video] completed prompt_id={prompt_id}")

        history = self._get_history(prompt_id)
        outputs = history["outputs"].get(output_node)

        if outputs is None:
            raise KeyError(
                f"Output node '{output_node}' not found in history outputs. "
                f"Available keys: {list(history['outputs'].keys())}"
            )

        filenames = outputs.get("Filenames") or outputs.get("filenames")
        if not filenames:
            raise RuntimeError("No filenames recorded in video output node.")

        # VHS_VideoCombine typically stores a list of filename dicts
        info = filenames[0]
        video_filename = info["filename"]
        print(f"[video] Comfy saved video as: {video_filename} (in its output folder)")
        return video_filename


# -----------------------------
# Main orchestration
# -----------------------------


def main():
    # -------------------------
    # CLI + env configuration
    # -------------------------
    parser = argparse.ArgumentParser(
        description="Script-driven ComfyUI director: pass a film script, get shots."
    )
    parser.add_argument(
        "--script",
        required=True,
        help="Path to YAML film script (acts/scenes/beats/actions).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse script and build prompts, but do NOT call ComfyUI.",
    )
    args = parser.parse_args()

    load_dotenv()  # loads .env in current directory

    server_address = os.getenv("COMFY_SERVER_ADDRESS", "127.0.0.1:8188")
    image_wf_path = os.getenv("IMAGE_WORKFLOW_PATH", "workflow_image_api.json")
    video_wf_path = os.getenv("VIDEO_WORKFLOW_PATH", "workflow_video_api.json")

    image_pos_node = os.getenv("IMAGE_POSITIVE_PROMPT_NODE", "6")
    image_save_node = os.getenv("IMAGE_SAVE_NODE", "9")

    video_pos_node = os.getenv("VIDEO_POSITIVE_PROMPT_NODE", "6")
    video_loadimage_node = os.getenv("VIDEO_LOADIMAGE_NODE", "56")
    video_output_node = os.getenv("VIDEO_OUTPUT_NODE", "64")

    manifest_path = os.getenv("OUTPUT_MANIFEST", "render_manifest.json")

    print(f"Using ComfyUI server at: {server_address}")
    print(f"Image workflow: {image_wf_path}")
    print(f"Video workflow: {video_wf_path}")

    # Load script
    try:
        film = load_film_script(args.script)
    except Exception as e:
        print(f"[ERROR] Failed to load script: {e}")
        sys.exit(1)

    print(f"Loaded film: {film.title}")
    print(f"Acts: {len(film.acts)}")

    client = ComfyClient(server_address)

    # This manifest will record what we rendered per shot
    manifest: List[Dict[str, Any]] = []

    # Iterate through the structure
    for act in film.acts:
        print(f"\n=== Act: {act.id} - {act.name} ===")
        for scene in act.scenes:
            print(f"\n--- Scene: {scene.id} - {scene.name} ---")
            for beat in scene.beats:
                print(f"\nBeat: {beat.id} - {beat.summary}")
                for action in beat.actions:
                    shot_label = f"{act.id}_{scene.id}_{beat.id}_{action.id}"
                    print(f"\n[SHOT] {shot_label}")

                    # Build prompt from composition + shot
                    prompt_text = build_prompt(film, act, scene, beat, action)

                    # If dry-run: just show prompt, do nothing else
                    if args.dry_run:
                        print("DRY RUN PROMPT:\n")
                        print(prompt_text)
                        print("\n---")
                        continue

                    # 1. First frame
                    try:
                        first_frame_filename = client.run_image_workflow(
                            workflow_path=image_wf_path,
                            positive_prompt_node=image_pos_node,
                            save_node=image_save_node,
                            new_prompt=prompt_text,
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] Failed to generate first frame for shot {shot_label}: {e}"
                        )
                        # Record failure and continue to next shot
                        manifest.append(
                            {
                                "shot": shot_label,
                                "status": "image_failed",
                                "error": str(e),
                            }
                        )
                        continue

                    # IMPORTANT:
                    # If your VIDEO workflow's LoadImage node expects the image to be in
                    # ComfyUI/input, you may want to copy or move the file there here.
                    # For now, we assume the workflow is set up to read from an absolute path
                    # or that you manually configure this as needed.

                    # 2. Video from first frame
                    try:
                        video_filename = client.run_video_workflow(
                            workflow_path=video_wf_path,
                            positive_prompt_node=video_pos_node,
                            loadimage_node=video_loadimage_node,
                            output_node=video_output_node,
                            new_prompt=prompt_text,
                            image_filename=first_frame_filename,
                        )
                        manifest.append(
                            {
                                "shot": shot_label,
                                "status": "ok",
                                "first_frame": first_frame_filename,
                                "video": video_filename,
                                "duration_seconds": action.duration_seconds,
                            }
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] Failed to generate video for shot {shot_label}: {e}"
                        )
                        manifest.append(
                            {
                                "shot": shot_label,
                                "status": "video_failed",
                                "first_frame": first_frame_filename,
                                "error": str(e),
                            }
                        )
                        # continue to next shot

    # Save manifest so you can see which shots succeeded/failed
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nRender manifest saved to: {manifest_path}")
    except Exception as e:
        print(f"[WARN] Could not save manifest: {e}")


if __name__ == "__main__":
    main()
