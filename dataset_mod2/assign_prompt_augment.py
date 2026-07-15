
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DEFAULT_SCENARIOS = ("scenario_10",)
DEFAULT_PROMPT_FILE_NAME = "prompt_augment_visual.jsonl"
DEFAULT_SEED = 0


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_meta_dirs(root: Path, scenarios: list[str], dates: set[str] | None) -> list[Path]:
    meta_dirs = []
    for scenario in scenarios:
        for date_dir in sorted((root / scenario).glob("20*.*.*")):
            if dates and date_dir.name not in dates:
                continue
            meta_dir = date_dir / "meta"
            if (meta_dir / "episodes.jsonl").exists():
                meta_dirs.append(meta_dir)
    return meta_dirs


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def assign_one_meta_dir(
    meta_dir: Path,
    prompt_file_name: str,
    rng: random.Random,
    keep_existing: bool,
) -> bool:
    tasks = read_jsonl(meta_dir / "tasks.jsonl")
    prompts = read_jsonl(meta_dir / prompt_file_name)
    episodes_path = meta_dir / "episodes.jsonl"
    episodes = read_jsonl(episodes_path)

    task_to_index = {normalize(row["task"]): int(row["task_index"]) for row in tasks}
    index_to_prompts = {
        int(row["task_index"]): [normalize(text) for text in row["task_des"]]
        for row in prompts
    }

    changed = False
    for episode in episodes:
        if keep_existing and "action_config" in episode:
            continue

        task = episode.get("tasks")
        if task is None:
            continue

        task_index = task_to_index[normalize(task)]
        prompt = rng.choice(index_to_prompts[task_index])
        episode["action_config"] = [{"english_action_text": prompt}]
        changed = True

    if changed:
        write_jsonl(episodes_path, episodes)

    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--scenarios", nargs="*", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--prompt-file-name", default=DEFAULT_PROMPT_FILE_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dates = set(args.dates) if args.dates else None
    changed_count = 0

    for meta_dir in iter_meta_dirs(args.root, args.scenarios, dates):
        changed = assign_one_meta_dir(
            meta_dir,
            args.prompt_file_name,
            rng,
            args.keep_existing,
        )
        changed_count += int(changed)
        print(f"Processed {meta_dir}")

    print(f"Updated {changed_count} episodes.jsonl files.")


if __name__ == "__main__":
    main()
