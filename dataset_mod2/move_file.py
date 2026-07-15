from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def normalize_scenario(scenario: str) -> str:
    scenario = scenario.strip()
    return scenario if scenario.startswith("scenario_") else f"scenario_{scenario}"


def normalize_annotate_cam(camera: str) -> str:
    camera = camera.strip().removeprefix("observation.images.")
    if camera.startswith("cam_"):
        return camera
    if camera.startswith("cam") and camera[3:].isdigit():
        return f"cam_{camera[3:]}"
    return camera


def copy_annotations(root: Path, scenario: str, annotate_cam: str) -> int:
    source_root = root / f"{scenario}_{annotate_cam}_annotate"
    if not source_root.exists():
        print(f"Skip missing annotation folder: {source_root}")
        return 0

    camera_name = annotate_cam.replace("_", "")
    copied_count = 0

    for date_dir in sorted(source_root.glob("202*.*.*")):
        if not date_dir.is_dir():
            continue

        destination_root = (
            root
            / scenario
            / date_dir.name
            / "videos_annotate"
            / "chunk-000"
            / f"observation.images.{camera_name}"
        )
        destination_root.mkdir(parents=True, exist_ok=True)

        for source_path in sorted(date_dir.glob("*.coco.json")):
            shutil.copy2(source_path, destination_root / source_path.name)
            copied_count += 1

    return copied_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy SAM COCO annotations into scenario_x/date/videos_annotate."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--scenario", default="scenario_10")
    parser.add_argument("--annotate-cams", nargs="+", default=["cam_2"])
    args = parser.parse_args()

    scenario = normalize_scenario(args.scenario)
    annotate_cams = [normalize_annotate_cam(camera) for camera in args.annotate_cams]

    total = 0
    for annotate_cam in annotate_cams:
        count = copy_annotations(args.root, scenario, annotate_cam)
        total += count
        print(f"Copied {count} files for {scenario}/{annotate_cam}.")

    print(f"Done. Copied {total} annotation files.")


if __name__ == "__main__":
    main()
