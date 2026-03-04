import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dest", default="./data/scenario_1_cam2_cam3")
    args = parser.parse_args()
    root = Path(args.base_dest)

    for p in root.glob("*"):
        episode_json_file = p/"meta/episodes.jsonl"
        temp_path = episode_json_file.with_suffix(".tmp")

        with episode_json_file.open("r") as f:
            line_count = sum(1 for _ in f)
        half_count = line_count//2

        with episode_json_file.open("r") as fin, temp_path.open("w") as fout:
            for line in fin:
                data = json.loads(line)
                episode_index = data['episode_index']
                if episode_index > half_count-1:
                    text = data['action_config'][0]['english_action_text'] 
                    data['action_config'][0]['english_action_text'] = "Annotated image version: "+ text
                fout.write(json.dumps(data) + "\n")
        temp_path.replace(episode_json_file)

    print("Text enhancement based on image annotation is completed!")

if __name__ == "__main__":
    main()