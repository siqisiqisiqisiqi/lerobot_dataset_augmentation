from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SCENARIOS = ("scenario_10",)
TRACK_TO_CATEGORY_SCENARIOS = {}

STANDARD_CATEGORIES = [
    {"id": 0, "name": "hand"},
    {"id": 1, "name": "soda bottle"},
    {"id": 2, "name": "pad"},
    {"id": 3, "name": "box"},
    {"id": 4, "name": "glass bottle"},
    {"id": 5, "name": "glass cup"},
    {"id": 6, "name": "blue cup"},
    {"id": 7, "name": "red cup"},
    {"id": 8, "name": "cup holder"},
]
VALID_CATEGORY_IDS = {category["id"] for category in STANDARD_CATEGORIES}


def normalize_scenario(scenario: str) -> str:
    scenario = scenario.strip()
    if scenario.startswith("scenario_"):
        return scenario
    return f"scenario_{scenario}"


def iter_coco_files(
    root: Path,
    scenario_names: tuple[str, ...],
    dates: set[str] | None,
) -> list[tuple[str, Path]]:
    coco_files: set[tuple[str, Path]] = set()

    for scenario_name in scenario_names:
        scenario_dir = root / scenario_name
        if not scenario_dir.exists():
            continue

        for annotate_dir in scenario_dir.glob("*/videos_annotate"):
            if dates is not None and annotate_dir.parent.name not in dates:
                continue

            for path in annotate_dir.rglob("coco.json"):
                coco_files.add((scenario_name, path))
            for path in annotate_dir.rglob("*.coco.json"):
                coco_files.add((scenario_name, path))

    return sorted(coco_files)


def update_annotation(annotation: dict, scenario_name: str) -> bool:
    changed = False

    if scenario_name in TRACK_TO_CATEGORY_SCENARIOS and "track_id" in annotation:
        track_id = annotation["track_id"]
        if not isinstance(track_id, int):
            raise ValueError(f"Expected integer track_id, got {track_id!r}")

        category_id = track_id
        if category_id not in VALID_CATEGORY_IDS:
            raise ValueError(
                f"Converted category_id {category_id} is not in {sorted(VALID_CATEGORY_IDS)}"
            )

        if annotation.get("category_id") != category_id:
            annotation["category_id"] = category_id
            changed = True

    if "track_id" in annotation:
        annotation.pop("track_id")
        changed = True

    return changed


def update_coco_file(scenario_name: str, path: Path) -> bool:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    changed = False
    for annotation in data.get("annotations", []):
        changed = update_annotation(annotation, scenario_name) or changed

    if data.get("categories") != STANDARD_CATEGORIES:
        data["categories"] = STANDARD_CATEGORIES
        changed = True

    if not changed:
        return False

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")
    tmp_path.replace(path)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize COCO categories and remove track_id fields."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repo/data root containing scenario folders.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(DEFAULT_SCENARIOS),
        help="Scenario names or numbers to process, such as scenario_10 or 10.",
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Optional date folder names to process, such as 2026.07.09.",
    )
    args = parser.parse_args()

    scenario_names = tuple(normalize_scenario(scenario) for scenario in args.scenarios)
    dates = set(args.dates) if args.dates else None
    coco_files = iter_coco_files(args.root, scenario_names, dates)
    changed_count = 0

    for scenario_name, path in coco_files:
        if update_coco_file(scenario_name, path):
            changed_count += 1

    print(f"Updated {changed_count} of {len(coco_files)} COCO annotation files.")


if __name__ == "__main__":
    main()
