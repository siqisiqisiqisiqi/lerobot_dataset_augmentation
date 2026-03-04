#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def save_json(p: Path, obj):
    with open(p, "w") as f:
        json.dump(obj, f, indent=4)

def read_jsonl(p: Path):
    data = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def append_jsonl(p: Path, items):
    with open(p, "a") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest-dir", required=True, help='e.g. "./data/scenario_1"')
    args = parser.parse_args()

    scenario_dir = Path(args.dest_dir)

    # ---------------- info.json ----------------
    info_path = scenario_dir / "meta" / "info.json"
    info = load_json(info_path)

    # double the totals
    info["total_episodes"] = info["total_episodes"] * 2
    info["total_frames"] = info["total_frames"] * 2
    info["total_videos"] = info["total_videos"] * 2

    # update split string (train: 0:TOTAL)
    if "splits" not in info:
        info["splits"] = {}
    info["splits"]["train"] = f"0:{info['total_episodes']}"

    save_json(info_path, info)
    print("info file modification completed")

    # ---------------- episodes_stats.jsonl ----------------
    stats_path = scenario_dir / "meta" / "episodes_stats.jsonl"
    data = read_jsonl(stats_path)

    start_index = len(data)
    new_items = []

    for i, item in enumerate(data):
        new_item = dict(item)  # shallow copy
        new_index = start_index + i

        new_item["episode_index"] = new_index

        # keep the same structure as your original code
        if "stats" not in new_item:
            new_item["stats"] = {}
        if "episode_index" not in new_item["stats"]:
            new_item["stats"]["episode_index"] = {}

        new_item["stats"]["episode_index"]["min"] = [new_index]
        new_item["stats"]["episode_index"]["max"] = [new_index]
        new_item["stats"]["episode_index"]["mean"] = [float(new_index)]
        new_item["stats"]["episode_index"]["std"] = [float(new_index)]

        new_items.append(new_item)

    append_jsonl(stats_path, new_items)
    print("stats info file modification completed")

if __name__ == "__main__":
    main()