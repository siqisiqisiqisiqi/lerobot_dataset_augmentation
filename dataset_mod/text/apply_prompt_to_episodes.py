#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, items):
    with path.open("w") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dest-dir", required=True)
    parser.add_argument("--dest-dir", default="./data/scenario_1_cam2_cam3/2025.12.01")
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    meta_root = dest_dir / "meta"

    in_file = meta_root / "episodes.jsonl"
    lib_file = meta_root / "prompt_augment.jsonl"
    task_file = meta_root / "tasks.jsonl"

    # load task maps: task string -> task_index
    task_map = {}
    for obj in read_jsonl(task_file):
        task_map[obj["task"]] = int(obj["task_index"])

    # load augmented prompts: task_index -> [variants...]
    task_aug_map = {}
    for obj in read_jsonl(lib_file):
        task_aug_map[int(obj["task_index"])] = obj["task_des"]

    def build_action_config(episode: dict):
        task_des = episode.get("tasks")
        if task_des is None:
            return None

        task_index = task_map.get(task_des)
        if task_index is None:
            return None

        task_aug_des = task_aug_map.get(task_index)
        if not task_aug_des:
            return None

        element = random.choice(task_aug_des)
        return [{"english_action_text": element}]

    # read, modify, write back (in place)
    new_items = []
    for episode in read_jsonl(in_file):
        action_config = build_action_config(episode)
        if action_config is not None:
            episode["action_config"] = action_config
        new_items.append(episode)

    write_jsonl(in_file, new_items)
    print("Done! File modified in-place.")


if __name__ == "__main__":
    main()