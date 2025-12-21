import re
import yaml
import os

# --- Configuration ---
INPUT_FILE = r'c:\Users\Mage\Desktop\cozy\CONGO 2.0_ The Feature Film Outline.txt'
OUTPUT_FILE = r'c:\Users\Mage\Desktop\cozy\congo2_complete_veo.yaml'

GLOBAL_STYLE = "black-and-white German Expressionist noir with selective single-color highlights; hard contrast, razor shadows, wet asphalt sheen, canted frames; film grain and gate weave; inked accent pops; 1990s hand-painted cel-anime grit, rough pencil line art, bold color blocking; one-take war-film camera flow"
ACCENTS = ["neon red accent", "cold cyan accent", "acid yellow accent", "violet neon accent"]
ASPECT = "2.39:1 anamorphic frame, grounded noir framing, shallow depth, lens grime"

# --- Parsing Logic ---

def parse_outline(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    acts = []
    acts_map = {} # Map ID to Act object
    current_act = None
    current_scene = None
    
    # State tracking
    in_beat_table = False
    header_count = 0 

    # Regex patterns
    act_pattern = re.compile(r'ACT ([IVX]+): (.+) \(Runtime:')
    scene_pattern = re.compile(r'🎬 (Act [IVX]+, Scene \d+|\w+): (.+) \(')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect Act (Initial Definition in Outline)
        act_match = act_pattern.search(line)
        if act_match and "Breakdown" not in line: 
            in_beat_table = False
            act_roman = act_match.group(1)
            act_id = f"Act_{act_roman}"
            act_title = act_match.group(2)
            
            if act_id not in acts_map:
                new_act = {
                    "id": act_id,
                    "title": act_title,
                    "scenes": []
                }
                acts.append(new_act)
                acts_map[act_id] = new_act
            
            current_act = acts_map[act_id]
            current_scene = None
            i += 1
            continue

        # Detect Scene
        scene_match = scene_pattern.search(line)
        if scene_match:
            in_beat_table = False
            header_count = 0
            
            # Extract Act from Scene Header to switch context
            # Group 1 is like "Act I, Scene 1"
            full_scene_ref = scene_match.group(1)
            
            act_ref_match = re.search(r'Act ([IVX]+)', full_scene_ref)
            if act_ref_match:
                act_roman = act_ref_match.group(1)
                act_id_ref = f"Act_{act_roman}"
                if act_id_ref in acts_map:
                    current_act = acts_map[act_id_ref]
                else:
                    # Fallback if Act wasn't defined earlier, though it should be
                    # Or maybe create it?
                    pass

            scene_name_raw = scene_match.group(2)
            
            sc_num_match = re.search(r'Scene (\d+)', line)
            # Use found scene number, or calculate based on current act scenes count
            sc_num = sc_num_match.group(1) if sc_num_match else str(len(current_act["scenes"]) + 1)

            # Clean ID construction: A{ActRoman}_S{SceneNum}
            # Clean Act Roman from ID (Act_I -> I)
            roman = current_act['id'].replace("Act_", "")
            scene_id = f"A{roman}_S{sc_num}"
            
            # Clean up scene name 
            scene_name = scene_name_raw.strip()

            current_scene = {
                "id": scene_id,
                "name": scene_name,
                "beats": []
            }
            if current_act:
                current_act["scenes"].append(current_scene)
            
            i += 1
            continue

        # Detect Start of Beat Table
        # We look for "Beat #" which seems to be the first line of the header block
        if "Beat #" in line:
            in_beat_table = True
            header_count = 0
            # Skip this line (Beat #) and the next 4 (Time, Duration, Shot, Script)
            # We just set flag and continue, the loop will handle skipping if we track it
            # But simpler to just skip the header block now
            i += 5 
            continue

        if in_beat_table:
            # Check for end of section/empty lines that signify end
            if "________" in line or (not line and i+1 < len(lines) and not lines[i+1].strip()):
                 # If we hit a delimiter line or double empty line, maybe stop?
                 # Actually single empty lines might exist between beats.
                 # Let's verify if the current line looks like a beat number
                 pass
            
            # Try to read a 5-line block
            # Line 1: Beat Num (digits)
            if re.match(r'^\d+$', line):
                try:
                    beat_num = line
                    # Look ahead safely
                    if i + 4 >= len(lines):
                        break
                        
                    time_range = lines[i+1].strip()
                    duration = lines[i+2].strip()
                    shot_desc = lines[i+3].strip()
                    dialogue = lines[i+4].strip()
                    
                    # Heuristic to ensure it's a beat: duration is usually '5'
                    # and time_range has ':'
                    if duration == '5' and ':' in time_range:
                        beat_data = {
                            "num": beat_num,
                            "time": time_range,
                            "duration": int(duration),
                            "description": shot_desc,
                            "dialogue": dialogue
                        }
                        if current_scene:
                            current_scene["beats"].append(beat_data)
                        
                        i += 5
                        continue
                except Exception as e:
                    print(f"Error parsing beat at line {i}: {e}")
            
            # If we are in table but didn't match a beat start, just advance
            # This handles empty lines between beats
            
        i += 1

    return acts

def generate_yaml(acts):
    yaml_structure = {
        "title": "CONGO 2.0 Dark Gritty Anime - BW Noir Text-to-Video Prompts (Complete)",
        "source_file": "CONGO 2.0_ The Feature Film Outline.txt",
        "style": GLOBAL_STYLE,
        "aspect": ASPECT,
        "accent_colors": ACCENTS,
        "acts": []
    }

    accent_idx = 0

    for act in acts:
        yaml_act = {
            "id": act["id"],
            "title": act["title"],
            "scenes": []
        }
        
        for scene in act["scenes"]:
            yaml_scene = {
                "id": scene["id"],
                "name": scene["name"],
                "beats": []
            }
            
            for beat in scene["beats"]:
                # Construct ID
                # Format: A1_S1_B01
                # We need to pad beat num
                b_num = int(beat["num"])
                b_id = f"{scene['id']}_B{b_num:02d}"
                
                # Pick accent
                accent = ACCENTS[accent_idx % len(ACCENTS)]
                accent_idx += 1
                
                # Construct Prompt
                # Use the description and dialogue
                # Add " Audio: " if dialogue exists and isn't just "SOUND: ..." or empty
                
                content = beat["description"]
                audio_cue = beat["dialogue"]
                
                # Clean up content (remove "INT." "EXT." prefixes sometimes? maybe keep them for context)
                # The user wants "add mroe details... with audio queues"
                
                full_prompt_text = (
                    f"{GLOBAL_STYLE}; "
                    f"{accent}; "
                    f"{ASPECT}; "
                    f"Beat {b_id}: {scene['name']} ? "
                    f"{content} "
                    f"AUDIO/DIALOGUE: {audio_cue}"
                )
                
                yaml_beat = {
                    "id": b_id,
                    "duration_seconds": beat["duration"],
                    "prompt": full_prompt_text
                }
                yaml_scene["beats"].append(yaml_beat)
            
            yaml_act["scenes"].append(yaml_scene)
        
        yaml_structure["acts"].append(yaml_act)

    return yaml_structure

# --- Main Execution ---

if __name__ == "__main__":
    print(f"Parsing {INPUT_FILE}...")
    acts = parse_outline(INPUT_FILE)
    print(f"Found {len(acts)} acts.")
    
    total_scenes = sum(len(a["scenes"]) for a in acts)
    print(f"Found {total_scenes} scenes.")
    
    total_beats = sum(len(s["beats"]) for a in acts for s in a["scenes"])
    print(f"Found {total_beats} beats.")
    
    yaml_data = generate_yaml(acts)
    
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Custom dumper to handle long strings nicelly? 
        # For now default dump is fine, but maybe we want block style for prompts
        yaml.dump(yaml_data, f, sort_keys=False, width=1000, default_flow_style=False)
    
    print("Done.")
